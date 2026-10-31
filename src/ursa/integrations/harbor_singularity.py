"""A Dockerfile and singularity-compose environment for Harbor tasks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import platform
import re
import secrets
import shlex
import shutil
import signal
import ssl
import stat
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from copy import deepcopy
from functools import cache, partial
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, ClassVar, NoReturn, override

import archspec.cpu
import yaml
from filelock import AsyncFileLock
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.environments.definition import should_use_prebuilt_docker_image
from harbor.environments.docker.compose_env import legacy_log_mount_env_vars
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import NetworkMode, NetworkPolicy
from harbor.models.trial.paths import EnvironmentPaths
from harbor.utils.env import resolve_env_vars
from pathspec import GitIgnoreSpec

from ursa.integrations.harbor_dockerfile import (
    DOCKERFILE_TRANSLATOR_VERSION,
    Definition,
    DockerfileUnsupported,
    DockerIgnore,
    dockerfile_workdir,
    image_config,
    translate_dockerfile,
)

_COMPOSE_SERVICE_FIELDS = frozenset({
    "build",
    "command",
    "depends_on",
    "deploy",
    "env_file",
    "environment",
    "expose",
    "image",
    "ports",
    "volumes",
})
_COMPOSE_BUILD_FIELDS = frozenset({"args", "context", "dockerfile", "target"})
_HOST_CA_CERTIFICATES_FILENAME = "host-ca-certificates.crt"
_COMPOSE_VARIABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Compose files are loaded several times per trial; warn once per variable.
_warned_unset_compose_variables: set[str] = set()
_logger = logging.getLogger(__name__)
# Without fakeroot, Singularity keeps the host's HOME path but does not mount
# it under --containall, so commands get a private writable home here instead.
# Fakeroot without a disk overlay uses it too, since /root is read-only then.
_CONTAINER_HOME = "/harbor-home"
# Runtime diagnostics that mean ``--fakeroot`` itself is unusable, as opposed
# to an unrelated probe failure such as a pull error or a broken image.
_FAKEROOT_FAILURE_MARKERS = (
    "fakeroot",
    "subuid",
    "subgid",
    "user namespace",
    "uid map",
    "gid map",
    "newuidmap",
    "newgidmap",
)
# ``apptainer`` builds a translated definition file with the runtime itself;
# the others build an OCI image that the runtime then converts.
_SIF_BUILDERS = ("auto", "apptainer", "buildah", "podman", "docker")
_OCI_BUILDERS = ("buildah", "podman", "docker")
# Rootless OCI builder errors that mean no ``RUN`` step can work on this
# host, as on hosts without /etc/subuid entries, rather than a task error.
_OCI_RUN_FAILURE_MARKERS = (
    "supplemental groups",
    "/dev/net/tun",
    "newuidmap",
    "newgidmap",
    "subuid",
    "subgid",
)
# Apptainer bind-mounts the host's fakeroot(1) into %post; it cannot load in
# images with an older glibc than the host's, or with musl.
_FAKEROOT_COMMAND_FAILURE_MARKERS = (
    "error while starting the `faked' daemon",
    "required by /.singularity.d/libs/libfakeroot",
)
# Apptainer binds the host's /etc/localtime by default (a ``bind path`` in
# apptainer.conf). Over a disk overlay, tzdata's postinst then cannot replace
# it ("Device or resource busy"), so apt-get install of python3 and other
# packages that need tzdata fails. Docker leaves the image's file in place.
_HOST_LOCALTIME = "/etc/localtime"
# Apptainer also binds the instance's --workdir over /tmp and /var/tmp, a
# separate filesystem from the overlay, so os.rename between /tmp and the rest
# of the container fails with EXDEV. Over a disk overlay with fakeroot the
# adapter keeps the image's directories instead, which the overlay makes
# writable, as Docker does, and leaves the task's workdir unbound in the
# overlay with them; see ``_harbor_writable_binds``. Without fakeroot the
# invoking user cannot write the image's root-owned directories, even /tmp,
# through the overlay, so only the workdir's and the binds are writable.
_IMAGE_TMP = "tmp"
# How long a timed-out or cancelled exec waits for its output pipes to close
# after its process tree is killed. Background jobs it started keep them open.
_EXEC_OUTPUT_DRAIN_SEC = 10
# Overlay size for tasks without storage_mb, matching what Terminal-Bench
# tasks declare.
_DEFAULT_OVERLAY_MB = 10240
# Without /etc/subuid, Apptainer's fakeroot wraps every instance exec in the
# fakeroot(1) command, whose faked daemon exits with that exec. Processes the
# exec left running, such as a tmux server or a ``nohup`` job, keep the
# preloaded libfakeroot and then block forever on their next stat or chown.
# An exec under that wrapper sees FAKEROOTKEY, which the first instance exec
# checks for; subuid-mapped fakeroot sets none. Apptainer skips its wrapper
# when FAKEROOTKEY is already set, so in that mode instance execs set it and
# run this script, which shares one faked per instance, started on first use
# and replaced if it has died, and preloads libfakeroot for that daemon.
# Without the fakeroot command's files it only clears FAKEROOTKEY.
_FAKEROOT_COMMAND_MARKER = "ursa-fakeroot-command"
# Records the shared faked's ``key:pid`` in the instance's own /dev/shm,
# which --containall makes a fresh tmpfs per instance. /tmp, in the --workdir
# or the overlay, outlives an instance restart while the daemon does not, so
# the probe that runs first in every new instance also removes the record.
_SHARED_FAKED_STATE = "/dev/shm/.ursa-faked"
_FAKEROOT_COMMAND_PROBE = (
    f"rm -f {_SHARED_FAKED_STATE}; "
    f'if [ -n "${{FAKEROOTKEY-}}" ]; then echo {_FAKEROOT_COMMAND_MARKER}; fi'
)
_SHARED_FAKED_SCRIPT = (
    "unset FAKEROOTKEY; "
    f"libs=/.singularity.d/libs; state={_SHARED_FAKED_STATE}; "
    'if [ -x "$libs/faked" ] && [ -r "$libs/libfakeroot.so" ]; then '
    'key_pid=$(cat "$state" 2>/dev/null); '
    "case $key_pid in "
    '*:*) [ "$(cat "/proc/${key_pid#*:}/comm" 2>/dev/null)" = faked ] '
    "|| key_pid= ;; "
    "*) key_pid= ;; esac; "
    'if [ -z "$key_pid" ]; then '
    'key_pid=$("$libs/faked" </dev/null 2>/dev/null) && '
    'echo "$key_pid" > "$state.$$" && mv -f "$state.$$" "$state"; fi; '
    "case $key_pid in *:*) "
    "FAKEROOTKEY=${key_pid%%:*}; "
    "LD_PRELOAD=$libs/libfakeroot.so${LD_PRELOAD:+:$LD_PRELOAD}; "
    "FAKED_MODE=unknown-is-root; FAKEROOTDONTTRYCHOWN=1; "
    "export FAKEROOTKEY LD_PRELOAD FAKED_MODE FAKEROOTDONTTRYCHOWN;; "
    "esac; fi; "
    'exec "$@"'
)
# Commands that sanitize their environment drop the preloaded libfakeroot and
# its FAKEROOTKEY, so they and their children run as the real, root-mapped
# uid: chown to any other id fails with "Invalid argument", and apt's drop to
# _apt fails with "setgroups 65534 failed". The shared faked cannot reach them
# (``/etc/ld.so.preload`` would load libfakeroot, but not its key), so under
# the fakeroot command the instance start runs ``_FAKEROOT_COMMAND_SETUP``,
# which keeps apt as root and shadows the image's sudo with this shim: inside
# the instance a root caller needs no sudo to become root, and its commands
# keep the environment, or switch users through su as Harbor's user switching
# does. Non-root callers and sudo's other modes get the image's sudo.
_SUDO_SHIM_MARKER = "ursa-sudo-shim"
_SUDO_SHIM_PATH = "/usr/local/sbin/sudo"
_SUDO_SHIM_STASH = "/usr/local/lib/ursa/sudo"
_APT_FAKEROOT_CONFIG = "/etc/apt/apt.conf.d/00ursa-fakeroot"
_SUDO_SHIM = r"""#!/bin/sh
# ursa-sudo-shim: written by URSA's Harbor Apptainer environment. Under
# Apptainer's fakeroot command every process already runs as (fake) root,
# and sudo would strip the fakeroot preload from the command's environment.
# For a root caller this runs the command with its environment kept, or as
# another user through su, like Harbor's user switching. Other callers, and
# sudo's other modes, get the image's sudo.
real=
IFS=:
for dir in $PATH:/usr/bin:/bin:/usr/sbin:/sbin; do
    if [ -x "$dir/sudo" ] && ! grep -q ursa-sudo-shim "$dir/sudo" 2>/dev/null
    then
        real=$dir/sudo
        break
    fi
done
unset IFS
if [ -z "$real" ]; then
    echo "sudo: command not found" >&2
    exit 127
fi
[ "$(id -u)" = 0 ] || exec "$real" "$@"
parse() {
    while [ $# -gt 0 ]; do
        case $1 in
            --) count=$((count + 1)); return ;;
            --user=*) user=${1#*=} ;;
            --chdir=*) dir=${1#*=} ;;
            --group=* | --close-from=* | --prompt=* | --role=* | --type=* | \
                --command-timeout=* | --preserve-env=*) ;;
            --user | --chdir | --group | --close-from | --prompt | --role | \
                --type | --command-timeout)
                [ $# -ge 2 ] || { delegate=1; return; }
                case $1 in
                    --user) user=$2 ;;
                    --chdir) dir=$2 ;;
                esac
                count=$((count + 1))
                shift
                ;;
            --login) login=1 ;;
            --shell) shell=1 ;;
            --set-home) home=1 ;;
            --background) background=1 ;;
            --preserve-env | --non-interactive | --stdin | --askpass | \
                --bell | --preserve-groups | --reset-timestamp) ;;
            -[!-]*)
                flags=${1#-}
                while [ -n "$flags" ]; do
                    flag=${flags%"${flags#?}"}
                    flags=${flags#?}
                    case $flag in
                        u | D | g | C | p | r | t | T)
                            if [ -n "$flags" ]; then
                                value=$flags
                                flags=
                            else
                                [ $# -ge 2 ] || { delegate=1; return; }
                                value=$2
                                count=$((count + 1))
                                shift
                            fi
                            case $flag in
                                u) user=$value ;;
                                D) dir=$value ;;
                            esac
                            ;;
                        i) login=1 ;;
                        s) shell=1 ;;
                        H) home=1 ;;
                        b) background=1 ;;
                        A | B | E | k | n | N | P | S) ;;
                        *) delegate=1; return ;;
                    esac
                done
                ;;
            -*) delegate=1; return ;;
            *) return ;;
        esac
        count=$((count + 1))
        shift
    done
}
quote() {
    for arg; do
        arg=$(printf '%sx' "$arg" | sed "s/'/'\\\\''/g")
        printf "'%s' " "${arg%x}"
    done
}
user= dir= login= shell= home= background= delegate= count=0
parse "$@"
[ -z "$delegate" ] || exec "$real" "$@"
shift "$count"
[ $# -gt 0 ] || [ -n "$login$shell" ] || exec "$real"
case $user in
    \#*) user=$(getent passwd "${user#?}" | cut -d: -f1) ;;
esac
entry=$(getent passwd "${user:-root}")
if [ -z "$entry" ]; then
    echo "sudo: unknown user ${user:-root}" >&2
    exit 1
fi
target=${entry%%:*}
target_home=$(printf '%s' "$entry" | cut -d: -f6)
target_shell=$(printf '%s' "$entry" | cut -d: -f7)
target_shell=${target_shell:-/bin/sh}
export SUDO_USER="$(id -un)" SUDO_UID="$(id -u)" SUDO_GID="$(id -g)"
export SUDO_COMMAND="$*"
if [ -n "$dir" ]; then
    cd "$dir" || exit 1
fi
if [ -n "$login" ]; then
    command="cd $(quote "$target_home") 2>/dev/null; exec $(quote "$target_shell") -l"
    [ $# -eq 0 ] || command="$command -c $(quote "$(quote "$@")")"
elif [ -n "$shell" ]; then
    command="exec $(quote "${SHELL:-$target_shell}")"
    [ $# -eq 0 ] || command="$command -c $(quote "$(quote "$@")")"
else
    command="exec env -- $(quote "$@")"
fi
if [ "$target" = "$(id -un)" ]; then
    [ -z "$home$login" ] || export HOME="$target_home"
    set -- /bin/sh -c "$command"
else
    set -- su "$target" -s /bin/sh -c "$command"
fi
if [ -n "$background" ]; then
    "$@" &
    exit 0
fi
exec "$@"
"""
# Installs the shim only over an image's /usr/bin/sudo, and later when an apt
# install adds sudo, so ``command -v sudo`` still fails where the image has
# none, as under Docker. Without a writable overlay every write just fails.
_FAKEROOT_COMMAND_SETUP = (
    "if [ -d /etc/apt ]; then "
    "mkdir -p /etc/apt/apt.conf.d 2>/dev/null; "
    "printf '%s\\n' "
    "'// Written by URSA: under the fakeroot command apt cannot drop to _apt.' "
    "'APT::Sandbox::User \"root\";' "
    "'// Written by URSA: give up on an unanswered mirror and try another.' "
    "'Acquire::http::Timeout \"20\";' "
    "'Acquire::https::Timeout \"20\";' "
    "'Acquire::Retries \"5\";' "
    "'Acquire::ForceIPv4 \"true\";' "
    "'// Written by URSA: retry at once; a delayed retry can leave apt waiting forever.' "
    "'Acquire::Retries::Delay \"false\";' "
    f"'DPkg::Post-Invoke {{ \"[ -e {_SUDO_SHIM_PATH} ] || "
    f"! [ -x /usr/bin/sudo ] || cp {_SUDO_SHIM_STASH} {_SUDO_SHIM_PATH} || "
    "true\"; };' "
    f"> {_APT_FAKEROOT_CONFIG} 2>/dev/null; fi; "
    f"if mkdir -p {PurePosixPath(_SUDO_SHIM_STASH).parent} "
    f"{PurePosixPath(_SUDO_SHIM_PATH).parent} 2>/dev/null && "
    f"printf '%s\\n' \"$1\" > {_SUDO_SHIM_STASH} 2>/dev/null && "
    f"chmod 755 {_SUDO_SHIM_STASH} && [ -x /usr/bin/sudo ] && "
    f"{{ ! [ -e {_SUDO_SHIM_PATH} ] || "
    f"grep -q {_SUDO_SHIM_MARKER} {_SUDO_SHIM_PATH}; }}; then "
    f"cp {_SUDO_SHIM_STASH} {_SUDO_SHIM_PATH}; fi; true"
)


@cache
def _runtime_is_apptainer(runtime: str) -> bool:
    """Detect Apptainer, including through a ``singularity`` alias."""
    if Path(runtime).name == "apptainer":
        return True
    try:
        result = subprocess.run(
            [runtime, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.stdout.strip().lower().startswith("apptainer")


_COMPOSE_BRIDGE_REQUIREMENT = (
    "Multi-container tasks need a setuid Apptainer installation with "
    "/etc/subuid and /etc/subgid entries for the invoking user, or root (or "
    "Docker): an unprivileged Apptainer gives non-root users only "
    "--network none."
)


@cache
def _runtime_lacks_setuid_starter(runtime: str) -> bool:
    """Whether ``runtime`` is certainly an unprivileged (non-setuid) install.

    A setuid installation ships a root-owned setuid ``starter-suid`` next to
    ``starter`` in ``$LIBEXECDIR/{apptainer,singularity}/bin``; SingularityCE
    3.6 and Apptainer 1.x both report ``LIBEXECDIR`` from ``buildcfg``.
    Without that starter, Apptainer runs in a user namespace and gives non-root
    users only ``--network none``. The answer is ``True`` only when a
    ``starter`` is found without a setuid ``starter-suid`` beside it, so an
    unexpected layout or a failing ``buildcfg`` never rejects a setuid
    installation.
    """
    libexec_dirs = []
    try:
        result = subprocess.run(
            [runtime, "buildcfg"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        result = None
    if result is not None and not result.returncode:
        for line in result.stdout.splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "LIBEXECDIR" and value.strip():
                libexec_dirs.append(Path(value.strip()))
    # Relocated unprivileged installs keep libexec beside bin.
    try:
        libexec_dirs.append(Path(runtime).resolve().parent.parent / "libexec")
    except OSError:
        pass
    found_starter = False
    for libexec in libexec_dirs:
        for name in ("apptainer", "singularity"):
            bin_dir = libexec / name / "bin"
            try:
                starter_suid = (bin_dir / "starter-suid").stat()
            except OSError:
                pass
            else:
                if starter_suid.st_uid == 0 and (
                    starter_suid.st_mode & stat.S_ISUID
                ):
                    return False
            found_starter = found_starter or (bin_dir / "starter").is_file()
    return found_starter


def _compose_uses_host_network(config: Mapping[str, Any]) -> bool:
    """Whether a Compose project is a lone ``main`` that skips the bridge."""
    services = config["services"]
    return (
        list(services) == ["main"]
        and not services["main"].get("ports")
        and services["main"].get("deploy", {}).get("replicas", 1) == 1
    )


def _openssl_context() -> ssl.SSLContext:
    """Return an empty context of the stdlib, OpenSSL-backed SSLContext."""
    # ``truststore.inject_into_ssl()``, which the URSA CLI calls, replaces
    # ssl.SSLContext with a subclass whose get_ca_certs raises
    # NotImplementedError.
    context_class = next(
        cls for cls in ssl.SSLContext.__mro__ if cls.__module__ == "ssl"
    )
    return context_class(ssl.PROTOCOL_TLS_CLIENT)


def _host_ca_certificates_pem() -> str:
    """Serialize the host OpenSSL trust anchors as one PEM bundle."""
    context = _openssl_context()
    paths = ssl.get_default_verify_paths()
    if paths.cafile is not None:
        context.load_verify_locations(cafile=paths.cafile)
    if paths.capath is not None:
        loaded: set[Path] = set()
        for entry in sorted(Path(paths.capath).iterdir()):
            if re.fullmatch(r"[0-9a-fA-F]{8}\.\d+", entry.name) is None:
                continue
            try:
                resolved = entry.resolve(strict=True)
                if resolved in loaded or not resolved.is_file():
                    continue
                context.load_verify_locations(cafile=str(resolved))
            except (OSError, ssl.SSLError):
                continue
            loaded.add(resolved)
    return "".join(
        ssl.DER_cert_to_PEM_cert(certificate)
        for certificate in context.get_ca_certs(binary_form=True)
    )


def _merge_compose(base: dict[str, Any], overlay: dict[str, Any]) -> None:
    for key, value in overlay.items():
        current = base.get(key)
        if (
            key == "volumes"
            and isinstance(current, list)
            and isinstance(value, list)
        ):
            merged = list(current)
            positions = {
                _volume_target(volume): index
                for index, volume in enumerate(merged)
            }
            for volume in value:
                target = _volume_target(volume)
                if target in positions:
                    merged[positions[target]] = volume
                else:
                    positions[target] = len(merged)
                    merged.append(volume)
            base[key] = merged
        elif isinstance(current, dict) and isinstance(value, dict):
            _merge_compose(current, value)
        else:
            base[key] = value


def _volume_target(volume: str) -> str:
    """Return the container path of a normalized ``src:dst[:ro]`` volume."""
    return volume.split(":")[1]


def _compose_variable(
    name: str, environ: Mapping[str, str], template: str
) -> str:
    if name in environ:
        return environ[name]
    # Docker Compose substitutes an empty string and warns once.
    if name not in _warned_unset_compose_variables:
        _warned_unset_compose_variables.add(name)
        _logger.warning(
            "Compose variable %r in %r is not set; defaulting to an empty "
            "string",
            name,
            template,
        )
    return ""


def _compose_closing_brace(template: str, start: int) -> int:
    depth = 1
    index = start
    while index < len(template):
        if template.startswith("$$", index):
            index += 2
        elif template.startswith("${", index):
            depth += 1
            index += 2
        elif template[index] == "}":
            depth -= 1
            if depth == 0:
                return index
            index += 1
        else:
            index += 1
    raise ValueError(f"Unterminated Compose interpolation in {template!r}")


def _interpolate_compose_braced(
    expression: str, environ: Mapping[str, str], template: str
) -> str:
    match = _COMPOSE_VARIABLE.match(expression)
    if match is None:
        raise ValueError(f"Invalid Compose interpolation in {template!r}")
    name = match.group()
    modifier = expression[match.end() :]
    if not modifier:
        return _compose_variable(name, environ, template)
    operator = next(
        (
            candidate
            for candidate in (":-", ":?", ":+", "-", "?", "+")
            if modifier.startswith(candidate)
        ),
        None,
    )
    if operator is None:
        raise ValueError(f"Invalid Compose interpolation in {template!r}")
    # As in Docker Compose, the colon forms also treat an empty value as
    # unset, and the argument is only expanded when it is used.
    value = environ.get(name)
    present = value is not None and (value != "" or ":" not in operator)
    argument = modifier[len(operator) :]
    if operator.endswith("-"):
        return value if present else _interpolate_compose(argument, environ)
    if operator.endswith("+"):
        return _interpolate_compose(argument, environ) if present else ""
    if present:
        return value
    message = _interpolate_compose(argument, environ)
    raise ValueError(
        f"Compose variable {name!r} is required"
        + (f": {message}" if message else "")
    )


def _interpolate_compose(template: str, environ: Mapping[str, str]) -> str:
    """Expand Docker Compose ``$VAR`` and ``${VAR...}`` interpolation.

    Supports ``$$`` escapes and the ``-``, ``?``, and ``+`` modifiers, each
    with and without a colon. As in Docker Compose, a plain reference to an
    unset variable becomes an empty string with a warning.
    """
    parts = []
    index = 0
    while (start := template.find("$", index)) != -1:
        parts.append(template[index:start])
        if template.startswith("$$", start):
            parts.append("$")
            index = start + 2
        elif template.startswith("${", start):
            end = _compose_closing_brace(template, start + 2)
            parts.append(
                _interpolate_compose_braced(
                    template[start + 2 : end], environ, template
                )
            )
            index = end + 1
        elif match := _COMPOSE_VARIABLE.match(template, start + 1):
            parts.append(_compose_variable(match.group(), environ, template))
            index = match.end()
        else:
            raise ValueError(f"Invalid Compose interpolation in {template!r}")
    parts.append(template[index:])
    return "".join(parts)


def _interpolate_compose_tree(value: Any, environ: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _interpolate_compose(value, environ)
    if isinstance(value, list):
        return [_interpolate_compose_tree(item, environ) for item in value]
    if isinstance(value, dict):
        return {
            key: _interpolate_compose_tree(item, environ)
            for key, item in value.items()
        }
    return value


def _compose_path(value: str, base_dir: Path) -> str:
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base_dir / path).resolve())


def _normalize_compose_service(
    name: str, service: dict[str, Any], base_dir: Path
) -> None:
    unsupported = set(service) - _COMPOSE_SERVICE_FIELDS
    if unsupported:
        raise ValueError(
            "singularity-compose does not support these fields on service "
            f"{name!r}: {', '.join(sorted(unsupported))}"
        )

    build = service.get("build")
    if isinstance(build, str):
        service["build"] = _compose_path(build, base_dir)
    elif isinstance(build, dict):
        unsupported = set(build) - _COMPOSE_BUILD_FIELDS
        if unsupported:
            raise ValueError(
                "singularity-compose does not support these build fields on "
                f"service {name!r}: {', '.join(sorted(unsupported))}"
            )
        context = build.get("context", ".")
        if not isinstance(context, str):
            raise ValueError(
                f"Docker Compose build context on service {name!r} must be a string"
            )
        build["context"] = _compose_path(context, base_dir)
        if not isinstance(build.get("dockerfile", "Dockerfile"), str):
            raise ValueError(
                f"Docker Compose dockerfile on service {name!r} must be a string"
            )
        if not isinstance(build.get("target", ""), str):
            raise ValueError(
                f"Docker Compose build target on service {name!r} must be a string"
            )
        if not isinstance(build.get("args", {}), (dict, list)):
            raise ValueError(
                f"Docker Compose build args on service {name!r} must be a mapping or list"
            )
        build_args = build.get("args", {})
        if (
            isinstance(build_args, dict)
            and not all(isinstance(key, str) for key in build_args)
        ) or (
            isinstance(build_args, list)
            and not all(isinstance(entry, str) for entry in build_args)
        ):
            raise ValueError(
                f"Docker Compose build args on service {name!r} must use string names"
            )
    elif build is not None:
        raise ValueError(
            f"Docker Compose build on service {name!r} must be a path or mapping"
        )

    command = service.get("command", "")
    if not isinstance(command, (str, list)) or (
        isinstance(command, list)
        and not all(isinstance(argument, str) for argument in command)
    ):
        raise ValueError(
            f"Docker Compose command on service {name!r} must contain only strings"
        )
    environment = service.get("environment", {})
    if (
        not isinstance(environment, (dict, list))
        or (
            isinstance(environment, dict)
            and not all(isinstance(key, str) for key in environment)
        )
        or (
            isinstance(environment, list)
            and not all(isinstance(entry, str) for entry in environment)
        )
    ):
        raise ValueError(
            f"Docker Compose environment on service {name!r} must contain only strings"
        )

    deploy = service.get("deploy")
    if deploy is not None:
        if not isinstance(deploy, dict) or set(deploy) - {
            "replicas",
            "resources",
        }:
            raise ValueError(
                "singularity-compose supports only deploy.replicas and "
                f"deploy.resources on service {name!r}"
            )
        # Like the task's cpus and memory_mb, Compose resource limits are not
        # enforced; Harbor's Terminal-Bench mapper always emits them.
        deploy.pop("resources", None)
        replicas = deploy.get("replicas", 1)
        if (
            not isinstance(replicas, int)
            or isinstance(replicas, bool)
            or replicas < 1
        ):
            raise ValueError(
                f"Docker Compose replicas on service {name!r} must be a positive integer"
            )

    env_files = service.get("env_file")
    if env_files is not None:
        entries = env_files if isinstance(env_files, list) else [env_files]
        normalized = []
        for entry in entries:
            required = True
            if isinstance(entry, dict):
                if (
                    set(entry) - {"path", "required"}
                    or not isinstance(entry.get("path"), str)
                    or not isinstance(entry.get("required", True), bool)
                ):
                    raise ValueError(
                        "singularity-compose cannot represent env_file options "
                        f"on service {name!r}: {entry!r}"
                    )
                required = entry.get("required", True)
                entry = entry["path"]
            if not isinstance(entry, str):
                raise ValueError(
                    f"Docker Compose env_file on service {name!r} must contain paths"
                )
            path = _compose_path(entry, base_dir)
            if required or Path(path).is_file():
                normalized.append(path)
        service["env_file"] = normalized

    depends_on = service.get("depends_on")
    if isinstance(depends_on, dict):
        for dependency, options in depends_on.items():
            if not isinstance(dependency, str):
                raise ValueError(
                    f"Docker Compose depends_on on service {name!r} must use string names"
                )
            options = options or {}
            if not isinstance(options, dict) or (
                set(options) - {"condition", "required", "restart"}
                or options.get("condition", "service_started")
                != "service_started"
                or options.get("required", True) is not True
                or options.get("restart", False) is not False
            ):
                raise ValueError(
                    "singularity-compose supports only service_started "
                    f"depends_on conditions on service {name!r}"
                )
        service["depends_on"] = list(depends_on)
    elif depends_on is not None and (
        not isinstance(depends_on, list)
        or not all(isinstance(dependency, str) for dependency in depends_on)
    ):
        raise ValueError(
            f"Docker Compose depends_on on service {name!r} must be a list of names"
        )

    volumes = service.get("volumes")
    if volumes is not None and not isinstance(volumes, list):
        raise ValueError(
            f"Docker Compose volumes on service {name!r} must be a list"
        )
    if isinstance(volumes, list):
        for index, volume in enumerate(volumes):
            if not isinstance(volume, dict):
                continue
            if (
                set(volume) - {"read_only", "source", "target", "type"}
                or volume.get("type") != "bind"
                or not isinstance(volume.get("read_only", False), bool)
                or not isinstance(volume.get("source"), str)
                or not isinstance(volume.get("target"), str)
            ):
                raise ValueError(
                    "singularity-compose supports only bind mounts "
                    f"on service {name!r}: {volume!r}"
                )
            volumes[index] = f"{volume['source']}:{volume['target']}" + (
                ":ro" if volume.get("read_only") else ""
            )
        for index, volume in enumerate(volumes):
            parts = volume.split(":") if isinstance(volume, str) else []
            if (
                len(parts) not in (2, 3)
                or not parts[0].startswith(("/", ".", "~"))
                or not parts[1].startswith("/")
            ):
                raise ValueError(
                    "singularity-compose supports only SOURCE:TARGET[:MODE] "
                    f"bind mounts on service {name!r}: {volume!r}"
                )
            mode = parts[2] if len(parts) == 3 else "rw"
            if mode not in {"ro", "rw"}:
                raise ValueError(
                    "singularity-compose supports only the ro and rw bind "
                    f"mount modes on service {name!r}: {volume!r}"
                )
            # Singularity splits its bind list on commas.
            if "," in volume:
                raise ValueError(
                    "Singularity bind mounts cannot contain commas on "
                    f"service {name!r}: {volume!r}"
                )
            source = _compose_path(parts[0], base_dir)
            volumes[index] = f"{source}:{parts[1]}" + (
                ":ro" if mode == "ro" else ""
            )

    ports = service.get("ports")
    if ports is not None and not isinstance(ports, list):
        raise ValueError(
            f"Docker Compose ports on service {name!r} must be a list"
        )
    if isinstance(ports, list):
        for index, port in enumerate(ports):
            if not isinstance(port, dict):
                port = str(port)
                if re.fullmatch(r"[0-9]+:[0-9]+", port) is None:
                    raise ValueError(
                        "singularity-compose supports only TCP published ports "
                        f"on service {name!r}: {port!r}"
                    )
                ports[index] = port
                continue
            if (
                set(port) - {"protocol", "published", "target"}
                or port.get("protocol", "tcp") != "tcp"
                or port.get("published") is None
                or port.get("target") is None
            ):
                raise ValueError(
                    "singularity-compose supports only TCP published ports "
                    f"on service {name!r}: {port!r}"
                )
            ports[index] = f"{port['published']}:{port['target']}"


def _load_docker_compose(
    paths: list[Path], environ: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Load, interpolate, validate, and merge Docker Compose files.

    Interpolation uses ``environ``, which defaults to the host environment,
    and happens once here so later stages see literal values.
    """
    environ = os.environ if environ is None else environ
    config: dict[str, Any] = {"services": {}}
    for path in paths:
        try:
            overlay = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Invalid Docker Compose file {path}: {exc}"
            ) from exc
        if not isinstance(overlay, dict):
            raise ValueError(f"Docker Compose file must be a mapping: {path}")
        try:
            overlay = _interpolate_compose_tree(overlay, environ)
        except ValueError as exc:
            raise ValueError(f"Docker Compose file {path}: {exc}") from exc
        unsupported = set(overlay) - {"name", "services", "version"}
        if unsupported:
            raise ValueError(
                "singularity-compose does not support these top-level Docker "
                f"Compose fields: {', '.join(sorted(unsupported))}"
            )
        services = overlay.get("services", {})
        if not isinstance(services, dict):
            raise ValueError("Docker Compose 'services' must be a mapping")
        for name, service in services.items():
            if not isinstance(name, str) or not isinstance(service, dict):
                raise ValueError(
                    "Docker Compose services must be named mappings"
                )
            _normalize_compose_service(name, service, path.parent)
        _merge_compose(config, overlay)
    config["services"].setdefault("main", {})
    services = config["services"]
    for name, service in services.items():
        unknown = set(service.get("depends_on", [])) - set(services)
        if unknown:
            raise ValueError(
                f"Docker Compose service {name!r} depends on unknown services: "
                + ", ".join(sorted(unknown))
            )

    visiting: list[str] = []
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            cycle = visiting[visiting.index(name) :] + [name]
            raise ValueError(
                "Docker Compose dependency cycle: " + " -> ".join(cycle)
            )
        if name in visited:
            return
        visiting.append(name)
        for dependency in services[name].get("depends_on", []):
            visit(dependency)
        visiting.pop()
        visited.add(name)

    for name in services:
        visit(name)
    return config


def _compose_key(identity: str, service: str) -> str:
    service_hash = hashlib.sha256(service.encode()).hexdigest()[:6]
    safe_service = re.sub(r"[^a-z0-9-]", "-", service.lower())[:24].rstrip("-")
    return f"u{identity}-{safe_service}-{service_hash}"


def _resolve_compose_value(
    key: str, value: Any, environ: Mapping[str, str] | None = None
) -> str:
    if value is None:
        try:
            return (os.environ if environ is None else environ)[key]
        except KeyError as exc:
            raise ValueError(
                f"Environment variable {key!r} is not set on the host"
            ) from exc
    # ``_load_docker_compose`` has already interpolated Compose values.
    return str(value).lower() if isinstance(value, bool) else str(value)


def _read_compose_env_file(
    path: Path, environ: Mapping[str, str]
) -> dict[str, str]:
    environment = {}
    for number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, separator, value = line.partition("=")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(
                f"Invalid environment entry at {path}:{number}: {raw_line!r}"
            )
        if not separator:
            environment[key] = _resolve_compose_value(key, None, environ)
        elif len(value) >= 2 and value[0] == value[-1] == "'":
            environment[key] = value[1:-1]
        else:
            if len(value) >= 2 and value[0] == value[-1] == '"':
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            # Docker Compose resolves earlier entries of the file first.
            environment[key] = _interpolate_compose(
                value, {**environ, **environment}
            )
    return environment


def _compose_environment(
    service_name: str,
    service: dict[str, Any],
    main_environment: Mapping[str, str],
    interpolation_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environ = (
        os.environ
        if interpolation_environment is None
        else interpolation_environment
    )
    environment = {}
    for path in service.get("env_file", []):
        environment.update(_read_compose_env_file(Path(path), environ))
    configured = service.get("environment", {})
    if isinstance(configured, dict):
        environment.update({
            key: _resolve_compose_value(key, value, environ)
            for key, value in configured.items()
        })
    else:
        for entry in configured:
            key, separator, value = entry.partition("=")
            environment[key] = _resolve_compose_value(
                key, value if separator else None, environ
            )
    if service_name == "main":
        environment.update(main_environment)
    return environment


def _write_compose_environment(
    project_dir: Path,
    identity: str,
    service_name: str,
    environment: dict[str, str],
) -> Path | None:
    if not environment:
        return None
    invalid = [
        key
        for key in environment
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None
    ]
    if invalid:
        raise ValueError(
            "Invalid Compose environment variable names: "
            + ", ".join(sorted(invalid))
        )
    path = project_dir / f"{_compose_key(identity, service_name)}.env.sh"
    path.write_text(
        "".join(
            f"export {key}={shlex.quote(value)}\n"
            for key, value in sorted(environment.items())
        )
    )
    path.chmod(0o600)
    return path


def _compose_volumes(
    service_name: str,
    service: dict[str, Any],
    *,
    staging_dir: Path,
    environment_file: Path | None,
    main_mounts: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    """Return a service's volumes and the binds to pass as start options.

    singularity-compose 0.1.19 splits each volume into exactly ``src:dst``,
    so read-only binds go to ``start.options``, which follow the volumes on
    the command line. Binds nested inside a read-only one go there too, so
    the read-only parent cannot hide them.
    """
    volumes = list(service.get("volumes", []))
    for volume in volumes:
        source = volume.split(":")[0]
        if not Path(source).exists():
            raise FileNotFoundError(
                f"Bind mount source on service {service_name!r} does not "
                f"exist: {source}"
            )
    targets = {_volume_target(volume) for volume in volumes}
    if "/staging" in targets:
        raise ValueError("/staging is reserved by the Harbor environment")
    volumes.append(f"{staging_dir}:/staging")
    targets.add("/staging")
    if service_name == "main":
        for mount in main_mounts:
            if mount.get("type") != "bind":
                raise ValueError(
                    "Singularity supports only Harbor bind mounts, got "
                    f"{mount.get('type')!r}"
                )
            target = str(mount["target"])
            if target == "/staging":
                raise ValueError(
                    f"Duplicate or reserved main service mount target: {target}"
                )
            if target in targets:
                volumes = [
                    volume
                    for volume in volumes
                    if _volume_target(volume) != target
                ]
            targets.add(target)
            volumes.append(
                f"{mount['source']}:{target}"
                + (":ro" if mount.get("read_only") else "")
            )
    if environment_file is not None:
        target = "/.singularity.d/env/91-ursa-compose.sh"
        if target in targets:
            raise ValueError(
                f"Compose environment mount conflicts on service {service_name!r}"
            )
        volumes.append(f"{environment_file}:{target}")
    # Mount parents before the paths nested inside them; a later parent bind
    # would hide an earlier nested one.
    volumes.sort(
        key=lambda volume: len(PurePosixPath(_volume_target(volume)).parts)
    )
    read_only_targets = [
        PurePosixPath(_volume_target(volume))
        for volume in volumes
        if volume.endswith(":ro")
    ]

    def under_read_only(volume: str) -> bool:
        target = PurePosixPath(_volume_target(volume))
        return any(
            target == parent or parent in target.parents
            for parent in read_only_targets
        )

    return (
        [volume for volume in volumes if not under_read_only(volume)],
        [volume for volume in volumes if under_read_only(volume)],
    )


def _add_compose_home(
    instance: dict[str, Any], home_root: Path | None, key: str
) -> None:
    if home_root is None:
        return
    home = home_root / key
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    instance["start"]["options"].append(f"home={home}:{_CONTAINER_HOME}")


async def docker_compose_to_singularity_compose(
    docker_compose_paths: Path | Sequence[Path],
    singularity_compose_path: Path,
    *,
    identity: str,
    image_resolver: Callable[[str, dict[str, Any]], Awaitable[str]],
    staging_dir: Path,
    main_environment: Mapping[str, str] | None = None,
    main_mounts: Sequence[Mapping[str, Any]] = (),
    network_mode: NetworkMode = NetworkMode.PUBLIC,
    fakeroot: bool = True,
    main_overlay: Path | None = None,
    service_workdirs: Mapping[str, Sequence[Path]] | None = None,
    interpolation_environment: Mapping[str, str] | None = None,
    home_root: Path | None = None,
    main_no_mounts: Sequence[str] = (),
) -> dict[str, list[str]]:
    """Convert one or more Docker Compose files into a singularity-compose file."""
    paths = (
        [docker_compose_paths]
        if isinstance(docker_compose_paths, Path)
        else list(docker_compose_paths)
    )
    config = _load_docker_compose(paths, interpolation_environment)
    service_keys = {
        name: _compose_key(identity, name) for name in config["services"]
    }
    replicas_by_service = {
        name: service.get("deploy", {}).get("replicas", 1)
        for name, service in config["services"].items()
    }
    # A lone main service needs no bridge: host networking matches the
    # non-Compose path and avoids fakeroot CNI, which many hosts lack.
    host_network = _compose_uses_host_network(config)
    instance_keys = {
        name: (
            [service_keys[name]]
            if replicas == 1 or service_workdirs is None
            else [
                f"{service_keys[name]}-r{replica}"
                for replica in range(1, replicas + 1)
            ]
        )
        for name, replicas in replicas_by_service.items()
    }
    if service_workdirs is not None:
        for name, replicas in replicas_by_service.items():
            try:
                workdirs = service_workdirs[name]
            except KeyError as exc:
                raise ValueError(
                    f"Missing Singularity workdir for service {name!r}"
                ) from exc
            if len(workdirs) != replicas:
                raise ValueError(
                    f"Expected {replicas} Singularity workdir(s) for "
                    f"service {name!r}, got {len(workdirs)}"
                )
    instances = {}
    instance_names = {}
    project_dir = singularity_compose_path.parent
    project_dir.mkdir(parents=True, exist_ok=True)
    for name, service in config["services"].items():
        key = service_keys[name]
        environment_file = _write_compose_environment(
            project_dir,
            identity,
            name,
            _compose_environment(
                name,
                service,
                main_environment or {},
                interpolation_environment,
            ),
        )
        start_options = ["containall"]
        if home_root is None:
            start_options.append("no-home")
        if fakeroot:
            start_options.insert(0, "fakeroot")
        if name == "main" and main_overlay is not None:
            start_options.append(f"overlay={main_overlay}")
            start_options.extend(f"no-mount={path}" for path in main_no_mounts)
        elif main_overlay is not None:
            # ``singularity-compose up --read_only`` is required to prevent
            # its implicit tmpfs overlay from conflicting with main's disk
            # overlay. Preserve the normal tmpfs behavior for sidecars.
            start_options.append("writable-tmpfs")
        instance: dict[str, Any] = {
            "image": await image_resolver(name, service),
            "network": {
                "allocate_ip": network_mode != NetworkMode.NO_NETWORK,
                "enable": True,
            },
            "start": {"options": start_options},
        }
        if network_mode == NetworkMode.NO_NETWORK:
            instance["network"]["type"] = "none"
        elif host_network:
            instance["network"]["enable"] = False
        volumes, start_binds = _compose_volumes(
            name,
            service,
            staging_dir=staging_dir,
            environment_file=environment_file,
            main_mounts=main_mounts,
        )
        if volumes:
            instance["volumes"] = volumes
        # singularity-compose passes these through as ``--bind=src:dst[:ro]``
        # after the volume binds.
        start_options.extend(f"bind={bind}" for bind in start_binds)
        if depends_on := service.get("depends_on", []):
            instance["depends_on"] = [
                instance_key
                for dependency in depends_on
                for instance_key in instance_keys[dependency]
            ]
        if ports := service.get("ports", []):
            instance["ports"] = ports
        if command := service.get("command"):
            instance["start"]["args"] = (
                shlex.join(command) if isinstance(command, list) else command
            )
        replicas = replicas_by_service[name]
        if service_workdirs is None and replicas != 1:
            instance["deploy"] = {"replicas": replicas}
        if service_workdirs is None:
            _add_compose_home(instance, home_root, key)
            instances[key] = instance
            instance_names[name] = [
                f"{key}{replica}" for replica in range(1, replicas + 1)
            ]
        else:
            for instance_key, workdir in zip(
                instance_keys[name], service_workdirs[name], strict=True
            ):
                replica = deepcopy(instance)
                replica["start"]["options"].append(f"workdir={workdir}")
                _add_compose_home(replica, home_root, instance_key)
                instances[instance_key] = replica
            instance_names[name] = [
                f"{instance_key}1" for instance_key in instance_keys[name]
            ]
    singularity_compose_path.write_text(
        yaml.safe_dump(
            {"version": "2.0", "instances": instances}, sort_keys=False
        )
    )
    return instance_names


class DockerfileSingularityEnvironment(BaseEnvironment):
    """Run Dockerfile and supported Compose tasks with Singularity or Apptainer."""

    _compose_filename = "docker-compose.yaml"
    # Fakeroot availability is a property of the host and runtime, so the
    # first failed probe in a process is reused by later environments (for
    # example a separate verifier) instead of probing again.
    _fakeroot_unavailable_reasons: ClassVar[dict[str, str]] = {}
    # ``singularity_fakeroot=false`` is explained once per process.
    _fakeroot_disabled_warned: ClassVar[bool] = False
    # Builders that failed for a host reason, such as missing fakeroot or
    # subuid support, keyed by executable; later builds in the process skip
    # them instead of failing the same way for every task.
    _unusable_builders: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        *args,
        singularity_image_cache_dir: Path | str | None = None,
        singularity_force_pull: bool = False,
        singularity_fakeroot: bool | str = True,
        singularity_overlay: bool = True,
        singularity_no_mount: str | None = None,
        singularity_startup_timeout_sec: float = 300,
        singularity_builder: str | None = None,
        **kwargs,
    ) -> None:
        if (
            not math.isfinite(singularity_startup_timeout_sec)
            or singularity_startup_timeout_sec <= 0
        ):
            raise ValueError(
                "singularity_startup_timeout_sec must be positive and finite"
            )
        if singularity_no_mount is not None:
            raise ValueError(
                "singularity_no_mount is not configurable; this environment "
                "always isolates mounts with --containall and a private home"
            )

        self._image_cache_dir = (
            Path(singularity_image_cache_dir)
            if singularity_image_cache_dir
            else self._default_image_cache_dir()
        )
        self._force_pull = singularity_force_pull
        self._fakeroot_setting = self._parse_fakeroot(singularity_fakeroot)
        self._builder_setting = self._parse_builder(singularity_builder)
        # ``start()`` verifies fakeroot with a runtime probe before the first
        # fakeroot command and fails the trial if the host cannot provide it.
        self._fakeroot = self._fakeroot_setting
        self._overlay_enabled = singularity_overlay
        self._runtime_path: str | None = None
        super().__init__(*args, **kwargs)
        self._startup_timeout_sec = singularity_startup_timeout_sec
        self._lifecycle_lock = asyncio.Lock()
        self._scratch_dir: Path | None = None
        self._staging_dir: Path | None = None
        self._runtime_workdir_temp: tempfile.TemporaryDirectory[str] | None = (
            None
        )
        self._runtime_workdir_root: Path | None = None
        self._sif_path: Path | None = None
        self._overlay_path: Path | None = None
        self._workdir = self._resolve_workdir()
        identity = hashlib.sha256(self.session_id.encode()).hexdigest()[:16]
        self._instance_name = f"ursa{identity}{secrets.token_hex(4)}"
        self._compose_identity = f"{identity[:8]}{secrets.token_hex(4)}"
        self._instance_started = False
        self._instance_start_attempted = False
        # Set by the first instance exec when Apptainer wraps execs in the
        # fakeroot command; see ``_SHARED_FAKED_SCRIPT``.
        self._shared_faked = False
        self._warned_user_switch_without_fakeroot = False
        self._fakeroot_disabled_explained = False
        self._compose_project_dir: Path | None = None
        self._compose_file: Path | None = None
        self._compose_instances: dict[str, list[str]] = {}
        self._compose_uses_sudo = False
        self._compose_host_network = False
        self._sif_use_token = f"{os.getpid()}-{secrets.token_hex(16)}"
        self._sifs_in_use: set[Path] = set()

    @staticmethod
    def _parse_fakeroot(value: bool | int | str) -> bool:
        if isinstance(value, bool):
            return value
        # Harbor's CLI JSON-decodes ``--ek`` values, so ``=1`` arrives as int.
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no"}:
                return False
        raise ValueError("singularity_fakeroot must be true or false")

    @staticmethod
    def _parse_builder(value: str | None) -> str:
        if value is None:
            value = os.environ.get("URSA_HARBOR_SIF_BUILDER") or "auto"
        normalized = str(value).strip().lower()
        if normalized not in _SIF_BUILDERS:
            raise ValueError(
                "singularity_builder must be one of " + ", ".join(_SIF_BUILDERS)
            )
        return normalized

    @staticmethod
    @override
    def type() -> EnvironmentType:
        return EnvironmentType.SINGULARITY

    @classmethod
    @override
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities()

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        # Singularity 3.6 can create an isolated network namespace, but cannot
        # securely enforce allowlists or switch a running instance's network.
        # With a persistent overlay, policy changes are implemented by
        # restarting the instance with the requested namespace configuration.
        return EnvironmentCapabilities(
            mounted=True,
            disable_internet=True,
            dynamic_network_policy=self._overlay_enabled,
            docker_compose=True,
        )

    @property
    def _dockerfile_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    @property
    def _staging(self) -> Path:
        if self._staging_dir is None:
            raise RuntimeError(
                "Staging directory not initialized — call start() first"
            )
        return self._staging_dir

    @property
    @override
    def _uses_compose(self) -> bool:
        environment_dir = getattr(self, "environment_dir", None)
        task_compose = bool(
            environment_dir
            and (environment_dir / self._compose_filename).is_file()
        )
        return task_compose or bool(
            getattr(self, "extra_docker_compose_paths", [])
        )

    @staticmethod
    def _default_image_cache_dir() -> Path:
        configured = os.environ.get("URSA_HARBOR_SIF_CACHE")
        if configured:
            return Path(configured).expanduser()
        cache_home = os.environ.get("XDG_CACHE_HOME")
        root = (
            Path(cache_home).expanduser()
            if cache_home
            else Path.home() / ".cache"
        )
        return root / "ursa" / "harbor" / "sif"

    @staticmethod
    def _compose_start_lock_path() -> Path:
        cache_home = os.environ.get("XDG_CACHE_HOME")
        root = (
            Path(cache_home).expanduser()
            if cache_home
            else Path.home() / ".cache"
        )
        return root / "ursa" / "harbor" / "singularity-compose.lock"

    @staticmethod
    def _runtime() -> str:
        runtime = shutil.which("apptainer") or shutil.which("singularity")
        if runtime is None:
            raise RuntimeError("Apptainer or Singularity is required")
        return runtime

    def _instance_runtime(self) -> str:
        runtime = getattr(self, "_runtime_path", None)
        if runtime is None:
            runtime = self._runtime()
            self._runtime_path = runtime
        return runtime

    @override
    def _startup_env(self) -> dict[str, str]:
        return {
            **self._host_ca_certificates_env(),
            **super()._startup_env(),
        }

    @classmethod
    @override
    def preflight(cls) -> None:
        """Verify that a Singularity-compatible runtime is installed."""
        try:
            cls._runtime()
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

    @override
    def _validate_definition(self) -> None:
        unsupported_mounts = [
            mount for mount in self._host_mounts if mount.get("type") != "bind"
        ]
        if unsupported_mounts:
            raise ValueError("Singularity supports only Harbor bind mounts")
        config = None
        if self._uses_compose:
            config = self._load_compose_config()
            sidecars = set(config["services"]) - {"main"}
            if sidecars and any(
                policy.network_mode == NetworkMode.NO_NETWORK
                for policy in (
                    self._network_policy,
                    *self._phase_network_policies,
                )
            ):
                raise ValueError(
                    "singularity-compose cannot provide sidecar networking "
                    "during a Harbor 'no-network' phase"
                )
        main = config["services"]["main"] if config is not None else {}
        if (
            not ({"build", "image"} & set(main))
            and not self.task_env_config.docker_image
            and not self._dockerfile_path.is_file()
        ):
            raise FileNotFoundError(
                f"Singularity environment requires {self._dockerfile_path}"
            )

    def _compose_paths(self) -> list[Path]:
        paths = []
        task_compose = self.environment_dir / self._compose_filename
        if task_compose.is_file():
            paths.append(task_compose)
        paths.extend(self.extra_docker_compose_paths)
        return paths

    def _compose_interpolation_environment(self) -> dict[str, str]:
        """Return the variables Harbor's Docker backend gives Compose.

        Harbor's infrastructure variables take precedence over the host
        environment, task ``[environment].env``, and persistent env.
        """
        environment = dict(os.environ)
        if self.task_env_config.env:
            environment.update(resolve_env_vars(self.task_env_config.env))
        environment.update(self._persistent_env)
        environment.update({
            "CONTEXT_DIR": str(self.environment_dir.resolve()),
            "MAIN_IMAGE_NAME": f"hb__{self.environment_id}".lower(),
            **legacy_log_mount_env_vars(self._host_mounts, host_value="source"),
        })
        if docker_image := self.task_env_config.docker_image:
            environment["PREBUILT_IMAGE_NAME"] = docker_image
        if (cpus := self.task_env_config.cpus) is not None:
            environment["CPUS"] = str(cpus)
        if memory_mb := self.task_env_config.memory_mb:
            environment["MEMORY"] = f"{memory_mb}M"
        return environment

    def _load_compose_config(self) -> dict[str, Any]:
        return _load_docker_compose(
            self._compose_paths(), self._compose_interpolation_environment()
        )

    @override
    def _resolve_workdir(self) -> str:
        """Resolve task overrides and Dockerfile WORKDIR instructions."""
        if self.task_env_config.workdir is not None:
            return self.task_env_config.workdir
        dockerfile_path = self._dockerfile_path
        main = (
            self._load_compose_config()["services"]["main"]
            if self._uses_compose
            else {}
        )
        compose_build = main.get("build")
        build_args: tuple[str, ...] = ()
        target = None
        if compose_build is not None:
            dockerfile_path, _, build_args, target = self._compose_build_inputs(
                compose_build
            )
        elif (
            self._uses_compose
            and "image" in main
            and self._main_image(force_build=False) == main["image"]
        ):
            return "/"
        elif not dockerfile_path.is_file():
            return "/"
        # Every builder starts in the base image's WORKDIR, which matters
        # for a relative WORKDIR and is looked up only then.
        warnings: list[str] = []
        workdir = dockerfile_workdir(
            dockerfile_path.read_text(),
            build_args=build_args,
            target=target,
            image_config=image_config,
            warnings=warnings,
        )
        for warning in warnings:
            self.logger.warning("%s: %s", dockerfile_path, warning)
        return workdir

    def _dockerfile_cache_path_sync(
        self,
        dockerfile_path: Path,
        context_dir: Path,
        build_args: tuple[str, ...],
        target: str | None,
        builder: str | None = None,
        unread_configs: tuple[str, ...] = (),
    ) -> Path:
        dockerfile_path = dockerfile_path.resolve()
        context_dir = context_dir.resolve()
        digest = hashlib.sha256()
        key: dict[str, Any] = {
            "architecture": platform.machine(),
            "build_args": build_args,
            "microarchitecture": archspec.cpu.host().name,
            "target": target,
        }
        # Definition builds are keyed apart from OCI builds, whose existing
        # keys stay valid, and by the translator so its fixes rebuild.
        # A build without a base image's config is keyed apart, so it never
        # stands in for one with it.
        if builder == "apptainer":
            key["builder"] = builder
            key["translator"] = DOCKERFILE_TRANSLATOR_VERSION
            if unread_configs:
                key["unread_configs"] = unread_configs
        digest.update(
            json.dumps(key, separators=(",", ":"), sort_keys=True).encode()
        )
        ignore_path = context_dir / ".dockerignore"
        # The translation copies exactly what Docker's matching keeps; OCI
        # keys keep the gitignore matching they were made with.
        ignored: Callable[[str], bool] | None = None
        if ignore_path.is_file():
            lines = ignore_path.read_text().splitlines()
            ignored = (
                DockerIgnore(lines).ignored
                if builder == "apptainer"
                else GitIgnoreSpec.from_lines(lines).match_file
            )
        dockerfile_in_context = None
        with suppress(ValueError):
            dockerfile_in_context = dockerfile_path.relative_to(
                context_dir
            ).as_posix()
        for path in sorted(context_dir.rglob("*")):
            relative = path.relative_to(context_dir).as_posix()
            if (
                ignored is not None
                and relative not in {dockerfile_in_context, ".dockerignore"}
                and ignored(relative)
            ):
                continue
            digest.update(relative.encode())
            digest.update(str(path.lstat().st_mode).encode())
            if path.is_symlink():
                digest.update(os.readlink(path).encode())
            elif path.is_file():
                digest.update(path.read_bytes())
        if dockerfile_in_context is None:
            digest.update(str(dockerfile_path).encode())
            digest.update(dockerfile_path.read_bytes())
        return (
            self._image_cache_dir / f"dockerfile-{digest.hexdigest()[:24]}.sif"
        )

    async def _dockerfile_cache_path(
        self,
        dockerfile_path: Path | None = None,
        context_dir: Path | None = None,
        build_args: tuple[str, ...] = (),
        target: str | None = None,
        builder: str | None = None,
        unread_configs: tuple[str, ...] = (),
    ) -> Path:
        return await asyncio.to_thread(
            self._dockerfile_cache_path_sync,
            dockerfile_path or self._dockerfile_path,
            context_dir or self.environment_dir,
            build_args,
            target,
            builder,
            unread_configs,
        )

    @staticmethod
    def _image_uri(image: str) -> str:
        """Return the runtime pull URI for a Compose or task image reference."""
        if "://" in image:
            return image
        name = image.rsplit("/", 1)[-1]
        if ":" not in name and "@" not in name:
            image = f"{image}:latest"
        return f"docker://{image}"

    def _pulled_image_cache_path(self, uri: str) -> Path:
        # A digest-pinned reference keys the cache by its digest. Tags are
        # keyed as written; ``singularity_force_pull`` refreshes them.
        digest = hashlib.sha256(
            json.dumps(
                {"architecture": platform.machine(), "image": uri},
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        return self._image_cache_dir / f"image-{digest[:24]}.sif"

    def _prebuilt_image(self, force_build: bool) -> str | None:
        """Return the task image to pull instead of building, as in Harbor."""
        image = self.task_env_config.docker_image
        if not should_use_prebuilt_docker_image(
            self.environment_dir, docker_image=image, force_build=force_build
        ):
            return None
        return image

    async def _pull_image_sif(self, image: str, force_pull: bool) -> Path:
        """Pull an OCI image into the SIF cache, or use a local SIF file."""
        if "://" not in image and image.endswith(".sif"):
            path = Path(image).expanduser()
            if not await self._is_valid_sif(path):
                raise FileNotFoundError(f"Singularity image not found: {path}")
            return path
        uri = self._image_uri(image)
        output = self._pulled_image_cache_path(uri)
        self._image_cache_dir.mkdir(parents=True, exist_ok=True)
        if not force_pull and await self._is_valid_sif(output):
            return output
        async with AsyncFileLock(output.with_suffix(".lock")):
            if not force_pull and await self._is_valid_sif(output):
                return output
            temporary = output.with_suffix(".tmp.sif")
            temporary.unlink(missing_ok=True)
            try:
                await self._run(
                    self._instance_runtime(), "pull", str(temporary), uri
                )
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
        return output

    async def _is_valid_sif(self, path: Path) -> bool:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        try:
            await self._run(self._instance_runtime(), "sif", "list", str(path))
        except RuntimeError:
            return False
        return True

    @staticmethod
    def _sif_in_use_path(path: Path) -> Path:
        return Path(f"{path}.in-use")

    async def _mark_sif_in_use(self, path: Path) -> None:
        """Refresh a cached SIF and record this environment as a user."""
        async with AsyncFileLock(path.with_suffix(".lock")):
            os.utime(path)
            if path in self._sifs_in_use:
                return
            marker = self._sif_in_use_path(path)
            users = (
                set(marker.read_text().splitlines())
                if marker.is_file()
                else set()
            )
            users.add(self._sif_use_token)
            marker.write_text("".join(f"{user}\n" for user in sorted(users)))
            self._sifs_in_use.add(path)

    async def _release_sifs_in_use(self) -> None:
        """Remove this environment from each cached SIF's users."""
        for path in tuple(self._sifs_in_use):
            try:
                async with AsyncFileLock(path.with_suffix(".lock")):
                    marker = self._sif_in_use_path(path)
                    users = (
                        set(marker.read_text().splitlines())
                        if marker.is_file()
                        else set()
                    )
                    users.discard(self._sif_use_token)
                    if users:
                        marker.write_text(
                            "".join(f"{user}\n" for user in sorted(users))
                        )
                    else:
                        marker.unlink(missing_ok=True)
            except Exception as exc:
                self.logger.warning(
                    "Failed to release cached Singularity image %s: %s",
                    path,
                    exc,
                )
            else:
                self._sifs_in_use.remove(path)

    @staticmethod
    async def _terminate_host_process(
        process: asyncio.subprocess.Process,
    ) -> None:
        if process.returncode is not None:
            return
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()

    @staticmethod
    async def _await_cleanup(cleanup: Awaitable[Any]) -> Any:
        """Finish ownership cleanup even if the caller is cancelled again."""
        task = asyncio.ensure_future(cleanup)
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        return task.result()

    def _record_command_failure(
        self,
        path: Path | None,
        command: tuple[str, ...],
        *,
        return_code: int | None,
        output: BinaryIO | None = None,
        start_error: OSError | None = None,
    ) -> str:
        if path is None:
            return ""
        absolute = path.absolute()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as log:
                log.write(b"=== failed build command ===\n")
                log.write(f"command: {shlex.join(command)}\n".encode())
                code = (
                    "not started" if return_code is None else str(return_code)
                )
                log.write(f"exit code: {code}\n".encode())
                log.write(b"--- output ---\n")
                if start_error is not None:
                    log.write(f"{start_error}\n".encode(errors="replace"))
                elif output is not None:
                    output.flush()
                    output.seek(0)
                    shutil.copyfileobj(output, log)
                log.write(b"\n")
            path.chmod(0o600)
        except OSError as exc:
            self.logger.error(
                "Could not write failed build output to %s: %s", absolute, exc
            )
            return f"\nBuild log could not be written to {absolute}: {exc}"
        return f"\nBuild log: {absolute}"

    def _surface_build_log(self, path: Path) -> None:
        try:
            with path.open("rb") as log:
                binary_stream = getattr(sys.stderr, "buffer", None)
                if binary_stream is not None:
                    shutil.copyfileobj(log, binary_stream)
                    binary_stream.flush()
                    return
                while chunk := log.read(64 * 1024):
                    sys.stderr.write(chunk.decode(errors="replace"))
                sys.stderr.flush()
        except (OSError, UnicodeError, ValueError) as exc:
            self.logger.error("Could not display build log %s: %s", path, exc)

    async def _run(
        self,
        *command: str,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        failure_log_path: Path | None = None,
    ) -> str:
        """Run a host command; return its stdout and stderr unless logged."""
        captured = (
            tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix="ursa-harbor-build-",
                suffix=".log",
            )
            if failure_log_path is not None
            else None
        )
        try:
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=(
                        captured
                        if captured is not None
                        else asyncio.subprocess.PIPE
                    ),
                    stderr=(
                        asyncio.subprocess.STDOUT
                        if captured is not None
                        else asyncio.subprocess.PIPE
                    ),
                    start_new_session=True,
                    cwd=cwd,
                    env=env,
                )
            except OSError as exc:
                log_note = self._record_command_failure(
                    failure_log_path,
                    command,
                    return_code=None,
                    start_error=exc,
                )
                raise RuntimeError(
                    f"Could not start command ({shlex.join(command)}): {exc}"
                    f"{log_note}"
                ) from exc
            communication = asyncio.create_task(
                process.wait()
                if captured is not None
                else process.communicate()
            )
            try:
                result = await asyncio.shield(communication)
            except asyncio.CancelledError:
                await self._await_cleanup(self._terminate_host_process(process))
                await self._await_cleanup(communication)
                raise
            if process.returncode:
                if captured is not None:
                    log_note = self._record_command_failure(
                        failure_log_path,
                        command,
                        return_code=process.returncode,
                        output=captured,
                    )
                    detail = ""
                else:
                    stdout, stderr = result
                    details = []
                    if stdout:
                        details.append(
                            "stdout:\n" + stdout.decode(errors="replace")
                        )
                    if stderr:
                        details.append(
                            "stderr:\n" + stderr.decode(errors="replace")
                        )
                    detail = "\n" + ("\n".join(details) or "<no output>")
                    log_note = ""
                raise RuntimeError(
                    f"Command failed ({shlex.join(command)}) with exit code "
                    f"{process.returncode}:{detail}{log_note}"
                )
            if captured is not None:
                return ""
            stdout, stderr = result
            return stdout.decode(errors="replace") + stderr.decode(
                errors="replace"
            )
        finally:
            if captured is not None:
                captured.close()

    async def _build_with(
        self,
        builder: str,
        output: Path,
        temporary: Path,
        force_build: bool,
        dockerfile_path: Path,
        context_dir: Path,
        build_args: tuple[str, ...],
        target: str | None,
        build_log_path: Path,
    ) -> None:
        builder_name = Path(builder).name
        tag = f"ursa-harbor-{output.stem}-{builder_name}-{secrets.token_hex(8)}"
        remove_command = (
            (builder, "rmi", "--force", tag)
            if builder_name == "buildah"
            else (builder, "image", "rm", "--force", tag)
        )
        builder_options = [
            option
            for build_arg in build_args
            for option in ("--build-arg", build_arg)
        ]
        if target is not None:
            builder_options.extend(["--target", target])
        temporary.unlink(missing_ok=True)
        try:
            await self._run(
                builder,
                "build",
                *(["--pull"] if force_build else []),
                *builder_options,
                "--tag",
                tag,
                "--file",
                str(dockerfile_path),
                str(context_dir),
                failure_log_path=build_log_path,
            )
            with tempfile.TemporaryDirectory(
                prefix="ursa-harbor-oci-"
            ) as temp_dir:
                archive = Path(temp_dir) / "image.tar"
                if builder_name == "buildah":
                    await self._run(
                        builder,
                        "push",
                        tag,
                        f"docker-archive:{archive}",
                        failure_log_path=build_log_path,
                    )
                else:
                    await self._run(
                        builder,
                        "save",
                        "-o",
                        str(archive),
                        tag,
                        failure_log_path=build_log_path,
                    )
                await self._run(
                    self._instance_runtime(),
                    "build",
                    str(temporary),
                    f"docker-archive://{archive}",
                    failure_log_path=build_log_path,
                )
                temporary.replace(output)
        finally:
            try:
                await self._run(*remove_command)
            except RuntimeError as exc:
                self.logger.warning(
                    "Failed to remove build image %s: %s", tag, exc
                )
            finally:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _last_build_output(path: Path) -> str:
        """Return the output of the last failed command in a build log."""
        try:
            with path.open("rb") as log:
                size = log.seek(0, os.SEEK_END)
                log.seek(max(0, size - 256 * 1024))
                text = log.read().decode(errors="replace")
        except OSError:
            return ""
        block = text.rpartition("=== failed build command ===")[2]
        return block.partition("--- output ---\n")[2]

    def _dockerfile_builders(self) -> list[str]:
        """Return the builders to try, in order.

        ``apptainer`` stands for a definition-file build by the runtime
        itself. With ``singularity_builder=auto`` it comes first when the
        runtime is Apptainer: it needs no subuid range, without which
        rootless Buildah and Podman cannot run any ``RUN`` step, and it costs
        no failed build to find that out. SingularityCE keeps the OCI
        builders. A builder that already failed for a host reason in this
        process is skipped unless it was requested explicitly.
        """
        setting = self._builder_setting
        builders = []
        skipped = []
        if setting in {"auto", "apptainer"}:
            try:
                runtime = self._instance_runtime()
            except RuntimeError as exc:
                reason: str | None = str(exc)
            else:
                reason = (
                    self._unusable_builders.get(runtime)
                    if _runtime_is_apptainer(runtime)
                    else f"{runtime} is not Apptainer"
                )
            if reason is None:
                builders.append("apptainer")
            elif setting == "apptainer":
                raise RuntimeError(
                    f"singularity_builder=apptainer cannot build: {reason}"
                )
            elif "is not Apptainer" not in reason:
                skipped.append(f"apptainer: {reason}")
        for name in _OCI_BUILDERS:
            if setting not in {"auto", name}:
                continue
            builder = shutil.which(name)
            if builder is None:
                continue
            reason = self._unusable_builders.get(builder)
            if reason is not None and setting == "auto":
                skipped.append(f"{name}: {reason}")
                continue
            builders.append(builder)
        if not builders:
            needed = (
                "Apptainer, buildah, podman, or docker"
                if setting == "auto"
                else setting
            )
            detail = f" ({'; '.join(skipped)})" if skipped else ""
            raise RuntimeError(
                f"Building a Dockerfile for Singularity requires {needed}"
                f"{detail}"
            )
        return builders

    def _note_unusable_builder(
        self, builder: str, build_log_path: Path
    ) -> None:
        """Skip a builder for the rest of the process after a host failure."""
        output = self._last_build_output(build_log_path)
        if builder == "apptainer":
            key = self._instance_runtime()
            markers = _FAKEROOT_FAILURE_MARKERS
            lines = [line for line in output.splitlines() if "FATAL" in line]
        else:
            key = builder
            markers = _OCI_RUN_FAILURE_MARKERS
            lines = [
                line for line in output.splitlines() if "error" in line.lower()
            ]
        reason = next(
            (
                line.strip()
                for line in lines
                if any(marker in line.lower() for marker in markers)
            ),
            None,
        )
        if reason is not None and key not in self._unusable_builders:
            self._unusable_builders[key] = reason
            self.logger.warning(
                "Not using %s for later builds in this process: %s",
                Path(builder).name,
                reason,
            )

    async def _build_with_apptainer(
        self,
        definition: Definition,
        output: Path,
        temporary: Path,
        context_dir: Path,
        build_log_path: Path,
    ) -> None:
        """Build a translated Dockerfile with ``apptainer build --fakeroot``."""
        runtime = self._instance_runtime()
        for note in definition.notes:
            self.logger.debug("Dockerfile translation: %s", note)
        for warning in definition.warnings:
            self.logger.warning("Dockerfile translation: %s", warning)
        # Binds from the caller's environment would also apply to %post.
        environment = {
            key: value
            for key, value in os.environ.items()
            if re.fullmatch(
                r"(APPTAINER|SINGULARITY)_(BIND|BINDPATH|MOUNT)", key
            )
            is None
        }
        options = ["--fakeroot"] if os.geteuid() != 0 else []
        temporary.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(prefix="ursa-harbor-def-") as staging:
            try:
                definition_path = await asyncio.to_thread(
                    definition.write, Path(staging)
                )
            except (OSError, tarfile.TarError, ValueError) as exc:
                raise RuntimeError(
                    f"Could not stage the Apptainer build: {exc}"
                ) from exc

            async def build(*extra: str) -> None:
                await self._run(
                    runtime,
                    "build",
                    *options,
                    *extra,
                    str(temporary),
                    str(definition_path),
                    cwd=context_dir,
                    env=environment,
                    failure_log_path=build_log_path,
                )

            try:
                try:
                    await build()
                except RuntimeError:
                    failure = self._last_build_output(build_log_path)
                    if not options or not any(
                        marker in failure
                        for marker in _FAKEROOT_COMMAND_FAILURE_MARKERS
                    ):
                        raise
                    # A root-mapped user namespace alone still runs package
                    # managers that do not drop privileges, such as apk.
                    self.logger.warning(
                        "The host's fakeroot command cannot run in %s; "
                        "retrying with --ignore-fakeroot-command",
                        definition.stages[-1].image,
                    )
                    temporary.unlink(missing_ok=True)
                    await build("--ignore-fakeroot-command")
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)

    async def _build_dockerfile_sif(
        self,
        force_build: bool,
        *,
        dockerfile_path: Path | None = None,
        context_dir: Path | None = None,
        build_args: tuple[str, ...] = (),
        target: str | None = None,
    ) -> Path:
        dockerfile_path = dockerfile_path or self._dockerfile_path
        context_dir = context_dir or self.environment_dir
        inputs = (dockerfile_path, context_dir, build_args, target)
        setting = self._builder_setting
        # Definition and OCI builds are cached apart. Either one is reused
        # before any builder is looked up, so a SIF built elsewhere, say on
        # a login node, also serves a node without that builder.
        apptainer_path = oci_path = None
        if setting in {"auto", "apptainer"}:
            apptainer_path = await self._dockerfile_cache_path(
                *inputs, builder="apptainer"
            )
        if setting != "apptainer":
            oci_path = await self._dockerfile_cache_path(*inputs)
        self._image_cache_dir.mkdir(parents=True, exist_ok=True)
        if not force_build:
            for output in (apptainer_path, oci_path):
                if output is not None and await self._is_valid_sif(output):
                    return output
        builders = self._dockerfile_builders()
        oci_builders = [b for b in builders if b != "apptainer"]
        definition = None
        if "apptainer" in builders:
            # Translate first, so an unsupported Dockerfile fails or falls
            # back before any command runs.
            try:
                definition = await asyncio.to_thread(
                    translate_dockerfile,
                    dockerfile_path,
                    context_dir,
                    build_args=build_args,
                    target=target,
                    image_config=image_config,
                )
            except DockerfileUnsupported as exc:
                if not oci_builders:
                    raise RuntimeError(
                        f"Apptainer cannot build this Dockerfile: {exc}"
                    ) from exc
                self.logger.info(
                    "Building with an OCI builder instead of Apptainer: %s",
                    exc,
                )
            else:
                if definition.unread_configs:
                    apptainer_path = await self._dockerfile_cache_path(
                        *inputs,
                        builder="apptainer",
                        unread_configs=tuple(definition.unread_configs),
                    )
        groups: list[tuple[Path, list[str]]] = []
        if definition is not None and apptainer_path is not None:
            groups.append((apptainer_path, ["apptainer"]))
        if oci_builders and oci_path is not None:
            groups.append((oci_path, oci_builders))
        failures = []
        failed_logs: list[Path] = []
        for output, group in groups:
            async with AsyncFileLock(output.with_suffix(".lock")):
                if not force_build and await self._is_valid_sif(output):
                    return output
                temporary = output.with_suffix(".tmp.sif")
                build_log_path = output.with_suffix(".build.log")
                build_log_path.unlink(missing_ok=True)
                # Every attempt appends to the same log; name it once at the
                # end.
                attempt_log_note = f"\nBuild log: {build_log_path.absolute()}"
                for builder in group:
                    try:
                        if builder == "apptainer":
                            assert definition is not None
                            await self._build_with_apptainer(
                                definition,
                                output,
                                temporary,
                                context_dir,
                                build_log_path,
                            )
                        else:
                            await self._build_with(
                                builder,
                                output,
                                temporary,
                                force_build,
                                dockerfile_path,
                                context_dir,
                                build_args,
                                target,
                                build_log_path,
                            )
                    except RuntimeError as exc:
                        failures.append(str(exc).replace(attempt_log_note, ""))
                        self._note_unusable_builder(builder, build_log_path)
                        continue
                    build_log_path.unlink(missing_ok=True)
                    return output
                if build_log_path.is_file():
                    failed_logs.append(build_log_path)
                if group == ["apptainer"] and len(groups) > 1:
                    # Keep this log even if an OCI builder succeeds, since
                    # it may show a translation problem.
                    self.logger.warning(
                        "Apptainer could not build %s; trying OCI builders. "
                        "Build log: %s",
                        dockerfile_path,
                        build_log_path.absolute(),
                    )
        for path in failed_logs:
            self._surface_build_log(path)
        log_note = "".join(
            f" Build log: {path.absolute()}" for path in failed_logs
        )
        raise RuntimeError(
            "No container builder succeeded: " + "; ".join(failures) + log_note
        )

    def _compose_key(self, service: str) -> str:
        identity = getattr(
            self,
            "_compose_identity",
            hashlib.sha256(self.session_id.encode()).hexdigest()[:16],
        )
        return _compose_key(identity, service)

    @staticmethod
    def _compose_build_paths(
        build: str | dict[str, Any],
    ) -> tuple[Path, Path]:
        if isinstance(build, str):
            context = Path(build)
            return context / "Dockerfile", context
        context = Path(build.get("context", "."))
        dockerfile = context / build.get("dockerfile", "Dockerfile")
        return dockerfile, context

    def _compose_build_inputs(
        self, build: str | dict[str, Any]
    ) -> tuple[Path, Path, tuple[str, ...], str | None]:
        dockerfile, context = self._compose_build_paths(build)
        if isinstance(build, str):
            return dockerfile, context, (), None
        raw_args = build.get("args", {})
        resolved_args: dict[str, str] = {}
        environ = self._compose_interpolation_environment()
        if isinstance(raw_args, dict):
            resolved_args = {
                str(key): _resolve_compose_value(str(key), value, environ)
                for key, value in raw_args.items()
            }
        else:
            for entry in raw_args:
                key, separator, value = entry.partition("=")
                resolved_args[key] = _resolve_compose_value(
                    key, value if separator else None, environ
                )
        args = tuple(
            f"{key}={value}" for key, value in sorted(resolved_args.items())
        )
        return dockerfile, context, args, build.get("target")

    def _main_compose_build_inputs(
        self,
    ) -> tuple[Path, Path, tuple[str, ...], str | None] | None:
        if not self._uses_compose:
            return None
        build = self._load_compose_config()["services"]["main"].get("build")
        if build is None:
            return None
        return self._compose_build_inputs(build)

    @staticmethod
    def _compose_service_image(
        service_name: str, service: dict[str, Any]
    ) -> str | None:
        if "build" in service or "image" not in service:
            return None
        image = service["image"]
        if not isinstance(image, str) or not image:
            raise ValueError(
                f"Docker Compose image on service {service_name!r} "
                "must be a non-empty string"
            )
        return image

    def _main_image(self, force_build: bool) -> str | None:
        """Return the image the main SIF is pulled from, if it is not built.

        As with Harbor's Docker backend, the task's prebuilt ``docker_image``
        wins over any Dockerfile unless the trial forces a build, and an
        explicit Compose ``main.image`` replaces that prebuilt image. When
        Harbor would build, ``main.image`` only names the built image, so it
        is pulled only if there is no Dockerfile to build.
        """
        prebuilt = self._prebuilt_image(force_build)
        if self._uses_compose:
            main = self._load_compose_config()["services"]["main"]
            image = self._compose_service_image("main", main)
            if image is not None and (
                prebuilt is not None or not self._dockerfile_path.is_file()
            ):
                return image
        return prebuilt

    async def _build_main_sif(
        self, force_build: bool, *, refresh: bool = False
    ) -> Path:
        # ``refresh`` and ``singularity_force_pull`` refresh the cached SIF
        # but never turn a prebuilt-image task into a Dockerfile build.
        refresh = force_build or refresh or self._force_pull
        image = self._main_image(force_build)
        if image is not None:
            return await self._pull_image_sif(image, refresh)
        build = self._main_compose_build_inputs()
        if build is None:
            return await self._build_dockerfile_sif(refresh)
        dockerfile, context, build_args, target = build
        return await self._build_dockerfile_sif(
            refresh,
            dockerfile_path=dockerfile,
            context_dir=context,
            build_args=build_args,
            target=target,
        )

    async def build_sifs(
        self, force_build: bool = False, *, refresh: bool = False
    ) -> list[Path]:
        """Build or pull every SIF without starting an instance.

        This includes the main environment and each Compose sidecar. Images
        named by ``docker_image`` or a Compose ``image`` are pulled into the
        SIF cache, so a login node can prepare compute nodes without registry
        access. ``force_build`` has Harbor's meaning and can switch a
        prebuilt-image task to a build; ``refresh`` only rebuilds or re-pulls
        the SIFs a trial without ``--force-build`` would use.
        """
        paths = [await self._build_main_sif(force_build, refresh=refresh)]
        if not self._uses_compose:
            return paths

        refresh = force_build or refresh or self._force_pull
        compose_config = self._load_compose_config()
        seen_builds = set()
        if self._main_image(force_build) is None:
            seen_builds.add(
                self._main_compose_build_inputs()
                or (self._dockerfile_path, self.environment_dir, (), None)
            )
        for name, service in compose_config["services"].items():
            if name == "main":
                continue
            image = self._compose_service_image(name, service)
            if image is not None:
                path = await self._pull_image_sif(image, refresh)
            else:
                build = self._compose_build_inputs(service["build"])
                if build in seen_builds:
                    continue
                seen_builds.add(build)
                dockerfile, context, build_args, target = build
                path = await self._build_dockerfile_sif(
                    refresh,
                    dockerfile_path=dockerfile,
                    context_dir=context,
                    build_args=build_args,
                    target=target,
                )
            if path not in paths:
                paths.append(path)
        return paths

    async def _compose_image(
        self,
        service_name: str,
        service: dict[str, Any],
        force_build: bool,
    ) -> str:
        image = self._compose_service_image(service_name, service)
        if service_name == "main":
            if self._sif_path is None:
                raise RuntimeError("Main Singularity image is not prepared")
            path = self._sif_path
        elif image is not None:
            # Pull into the shared cache; singularity-compose would otherwise
            # pull a private copy into every per-trial project directory.
            path = await self._pull_image_sif(image, force_build)
        else:
            dockerfile, context, build_args, target = (
                self._compose_build_inputs(service["build"])
            )
            path = await self._build_dockerfile_sif(
                force_build,
                dockerfile_path=dockerfile,
                context_dir=context,
                build_args=build_args,
                target=target,
            )
        await self._mark_sif_in_use(path)
        return str(path)

    async def _prepare_compose_project(self, force_build: bool) -> None:
        self._compose_project_dir = Path(
            tempfile.mkdtemp(prefix="ursa-harbor-compose-")
        )
        self._compose_project_dir.chmod(0o700)
        runtime_bin = self._compose_project_dir / "bin"
        runtime_bin.mkdir(mode=0o700)
        (runtime_bin / "singularity").symlink_to(self._instance_runtime())
        self._compose_file = (
            self._compose_project_dir / "singularity-compose.yml"
        )
        compose_config = self._load_compose_config()
        self._compose_uses_sudo = not self._fakeroot and any(
            service.get("ports")
            for service in compose_config["services"].values()
        )
        service_workdirs = {}
        for name, service in compose_config["services"].items():
            replicas = service.get("deploy", {}).get("replicas", 1)
            service_workdirs[name] = tuple(
                self._runtime_workdir(
                    f"compose-{self._compose_key(name)}-{replica}"
                )
                for replica in range(1, replicas + 1)
            )
        main_mounts = list(self._host_mounts)
        for source, target in self._harbor_writable_binds():
            source.mkdir(parents=True, exist_ok=True)
            source.chmod(0o777)
            main_mounts.append({
                "type": "bind",
                "source": str(source),
                "target": str(target),
            })
        self._compose_instances = await docker_compose_to_singularity_compose(
            self._compose_paths(),
            self._compose_file,
            identity=self._compose_identity,
            image_resolver=partial(
                self._compose_image, force_build=force_build
            ),
            staging_dir=self._staging,
            main_environment=self._startup_env(),
            main_mounts=main_mounts,
            network_mode=self._network_policy.network_mode,
            fakeroot=self._fakeroot,
            main_overlay=self._overlay_path,
            service_workdirs=service_workdirs,
            interpolation_environment=self._compose_interpolation_environment(),
            home_root=(
                self._staging / "harbor-home"
                if self._uses_private_home
                else None
            ),
            main_no_mounts=self._overlay_no_mounts,
        )
        self._instance_name = self._compose_instances["main"][0]

    def _set_compose_network_policy(
        self, network_policy: NetworkPolicy
    ) -> None:
        if self._compose_file is None:
            raise RuntimeError("Singularity Compose project is not prepared")
        compose = yaml.safe_load(self._compose_file.read_text())
        if not isinstance(compose, dict) or not isinstance(
            compose.get("instances"), dict
        ):
            raise RuntimeError("Invalid generated singularity-compose file")
        isolated = network_policy.network_mode == NetworkMode.NO_NETWORK
        instances = compose["instances"].values()
        # Match the conversion: a lone main instance uses host networking.
        host_network = len(instances) == 1 and not any(
            instance.get("ports") for instance in instances
        )
        self._compose_host_network = host_network
        for instance in instances:
            network = instance.setdefault("network", {})
            network["allocate_ip"] = not isolated
            network["enable"] = isolated or not host_network
            if isolated:
                network["type"] = "none"
            else:
                network.pop("type", None)
        self._compose_file.write_text(yaml.safe_dump(compose, sort_keys=False))

    async def _run_compose(self, *arguments: str) -> str:
        if self._compose_file is None or self._compose_project_dir is None:
            raise RuntimeError("Singularity Compose project is not prepared")
        executable = shutil.which("singularity-compose")
        if executable is None:
            raise RuntimeError(
                "Docker Compose tasks require singularity-compose; "
                "install `ursa-ai[harbor]`"
            )
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("SINGULARITYENV_", "APPTAINERENV_"))
        }
        runtime_bin = self._compose_project_dir / "bin"
        environment["PATH"] = os.pathsep.join(
            part for part in (str(runtime_bin), environment.get("PATH")) if part
        )
        return await self._run(
            executable,
            "--file",
            str(self._compose_file),
            "--project-name",
            self._compose_key("project"),
            *arguments,
            cwd=self._compose_project_dir,
            env=environment,
        )

    async def _verify_compose_instances(self, compose_output: str) -> None:
        """Fail when ``singularity-compose up`` left an instance unstarted.

        singularity-compose 0.1.19 reports a failed ``instance start`` as an
        ERROR line and still exits 0, so the first exec would otherwise fail
        with "no instance found".
        """
        expected = [
            instance
            for instances in self._compose_instances.values()
            for instance in instances
        ]
        if not expected:
            return
        output = compose_output.strip()
        try:
            running = await self._list_instances()
        except RuntimeError as exc:
            raise RuntimeError(
                f"{exc}\nsingularity-compose output:\n"
                + (output or "<no output>")
            ) from exc
        missing = [instance for instance in expected if instance not in running]
        if not missing:
            return
        fatal = [
            line.strip() for line in output.splitlines() if "FATAL" in line
        ]
        message = (
            "singularity-compose up exited successfully but did not start "
            f"instance(s): {', '.join(missing)}"
        )
        if fatal:
            message += "\n" + "\n".join(fatal)
        if "--network=none" in output or "network requires root" in output:
            message += f"\n{_COMPOSE_BRIDGE_REQUIREMENT}"
        message += "\nsingularity-compose output:\n" + (output or "<no output>")
        raise RuntimeError(message)

    def _compose_needs_bridge(self) -> bool:
        """Whether this Compose project starts instances on a bridge."""
        if _compose_uses_host_network(self._load_compose_config()):
            return False
        return any(
            policy.network_mode != NetworkMode.NO_NETWORK
            for policy in (self._network_policy, *self._phase_network_policies)
        )

    def _raise_if_compose_bridge_unavailable(self) -> None:
        """Fail before building when the runtime offers only --network none.

        Detection is conservative: root, ``singularity_fakeroot=false``, and
        any installation with a setuid starter are never rejected here; an
        undetected failure is still caught after ``up``.
        """
        if (
            not self._uses_compose
            or not self._fakeroot_setting
            or os.geteuid() == 0
        ):
            # Without fakeroot, singularity-compose never asks the runtime for
            # a bridge: it uses sudo for published ports and otherwise adds
            # ``--network none``.
            return
        if not self._compose_needs_bridge():
            return
        runtime = self._instance_runtime()
        if _runtime_lacks_setuid_starter(runtime):
            raise RuntimeError(
                "This task's Compose project needs bridge networking for its "
                f"sidecars or published ports, but {runtime} is an "
                "unprivileged installation without a setuid starter. "
                f"{_COMPOSE_BRIDGE_REQUIREMENT}"
            )

    def _add_compose_service_aliases(self) -> None:
        if self._compose_project_dir is None:
            raise RuntimeError("Singularity Compose project is not prepared")
        hosts_path = self._compose_project_dir / "etc.hosts"
        if not hosts_path.is_file():
            raise RuntimeError("singularity-compose did not create etc.hosts")
        lines = hosts_path.read_text().splitlines()
        aliases = []
        for service, instance_names in self._compose_instances.items():
            instance_name = instance_names[0]
            address = next(
                (
                    line.split()[0]
                    for line in lines
                    if len(line.split()) >= 2
                    and line.split()[1] == instance_name
                ),
                None,
            )
            if address is None:
                raise RuntimeError(
                    "singularity-compose did not assign an address to "
                    f"service {service!r}"
                )
            aliases.append(f"{address}\t{service}\n")
        with hosts_path.open("a") as hosts:
            hosts.writelines(aliases)

    def _cleanup_compose_project(self) -> None:
        project_dir = getattr(self, "_compose_project_dir", None)
        if project_dir is not None:
            shutil.rmtree(project_dir, ignore_errors=True)
        self._compose_project_dir = None
        self._compose_file = None
        self._compose_instances = {}
        self._compose_uses_sudo = False
        self._compose_host_network = False

    def _instance_exec_prefix(
        self,
        cwd: str | None = None,
        *,
        instance_name: str | None = None,
        default_cwd: str | None = None,
    ) -> list[str]:
        command = [
            self._instance_runtime(),
            "exec",
            "--cleanenv",
            "--pwd",
            cwd or default_cwd or self._workdir,
        ]
        if self._shared_faked:
            command.extend(["--env", "FAKEROOTKEY=ursa-shared-faked"])
        command.append(f"instance://{instance_name or self._instance_name}")
        if self._shared_faked:
            command.extend([
                "/bin/sh",
                "-c",
                _SHARED_FAKED_SCRIPT,
                "ursa-shared-faked",
            ])
        return command

    def _instance_start_command(
        self, network_policy: NetworkPolicy | None = None
    ) -> list[str]:
        if self._staging_dir is None or self._sif_path is None:
            raise RuntimeError("Singularity instance is not prepared")
        network_policy = network_policy or self._network_policy
        command = [
            self._instance_runtime(),
            "instance",
            "start",
            "--containall",
            *self._home_options(),
            "--workdir",
            str(self._runtime_workdir("main")),
        ]
        if self._fakeroot:
            command.insert(3, "--fakeroot")
        if self._overlay_path is not None:
            command.extend(["--overlay", str(self._overlay_path)])
        for path in self._overlay_no_mounts:
            command.extend(["--no-mount", path])
        for source, target in self._harbor_writable_binds():
            command.extend(["-B", f"{source}:{target}"])
        if network_policy.network_mode == NetworkMode.NO_NETWORK:
            command.extend(["--net", "--network", "none"])
        command.extend(["-B", f"{self._staging_dir}:/staging"])
        for mount in self._host_mounts:
            if mount.get("type") != "bind":
                raise ValueError("Singularity supports only Harbor bind mounts")
            if mount.get("target") == "/staging":
                raise ValueError(
                    "/staging is reserved by the Harbor environment"
                )
            bind = f"{mount['source']}:{mount['target']}"
            if mount.get("read_only"):
                bind += ":ro"
            command.extend(["-B", bind])
        command.extend([str(self._sif_path), self._instance_name])
        return command

    @property
    def _overlay_no_mounts(self) -> tuple[str, ...]:
        # See ``_HOST_LOCALTIME`` and ``_IMAGE_TMP``. Only Apptainer, where
        # this was measured, drops the binds; SingularityCE keeps its default
        # binds.
        if self._overlay_path is None or not _runtime_is_apptainer(
            self._instance_runtime()
        ):
            return ()
        if not self._fakeroot:
            return (_HOST_LOCALTIME,)
        return (_HOST_LOCALTIME, _IMAGE_TMP)

    @property
    def _keeps_image_tmp(self) -> bool:
        return _IMAGE_TMP in self._overlay_no_mounts

    @property
    def _uses_private_home(self) -> bool:
        # Fakeroot's /root is writable only through the disk overlay; without
        # one it sits on the read-only image root filesystem.
        return not self._fakeroot or self._overlay_path is None

    def _home_options(self) -> list[str]:
        if not self._uses_private_home:
            return ["--no-home"]
        home = self._staging / "harbor-home" / "main"
        home.mkdir(mode=0o700, parents=True, exist_ok=True)
        return ["--home", f"{home}:{_CONTAINER_HOME}"]

    def _exec_home(
        self, env: dict[str, str] | None, *, merge_persistent: bool
    ) -> str | None:
        merged = self._merge_env(env) if merge_persistent else env
        if merged and "HOME" in merged:
            return merged["HOME"]
        return _CONTAINER_HOME if self._uses_private_home else None

    def _prepare_host_ca_certificates(self) -> Path | None:
        """Materialize the host OpenSSL trust store in the staging mount."""
        if self._staging_dir is None:
            raise RuntimeError("Singularity staging directory is not prepared")
        source = self._staging_dir / _HOST_CA_CERTIFICATES_FILENAME
        if not source.is_file():
            certificates = _host_ca_certificates_pem()
            if not certificates:
                return None
            source.write_text(certificates, encoding="ascii")
            source.chmod(0o444)
        return source

    def _host_ca_certificates_env(self) -> dict[str, str]:
        if self._staging_dir is None:
            return {}
        source = self._staging_dir / _HOST_CA_CERTIFICATES_FILENAME
        if not source.is_file():
            return {}
        container_path = str(
            PurePosixPath("/staging") / source.relative_to(self._staging_dir)
        )
        return {
            "CURL_CA_BUNDLE": container_path,
            "REQUESTS_CA_BUNDLE": container_path,
            "SSL_CERT_FILE": container_path,
        }

    def _harbor_writable_binds(
        self,
    ) -> tuple[tuple[Path, PurePosixPath], ...]:
        """Return host binds that provide Harbor's writable runtime paths."""
        if self._staging_dir is None:
            raise RuntimeError("Singularity staging directory is not prepared")
        root = self._staging_dir / "harbor-writable"
        candidates = [
            (root / "installed-agent", PurePosixPath("/installed-agent")),
            (root / "solution", EnvironmentPaths.solution_dir),
            (root / "tests", EnvironmentPaths.tests_dir),
            (root / "skills", EnvironmentPaths.default_skills_dir),
        ]
        workdir = PurePosixPath(self._workdir)
        # Over a disk overlay that keeps the image's /tmp, the workdir stays
        # in the overlay too: a bind there would be a separate mount, so
        # os.rename between it and /tmp would still fail with EXDEV.
        if workdir != PurePosixPath("/") and not self._keeps_image_tmp:
            candidates.append((root / "workdir", workdir))
        candidates.extend(
            (root / target.relative_to("/"), target)
            for target in (
                EnvironmentPaths.agent_dir,
                EnvironmentPaths.user_agent_dir,
                EnvironmentPaths.verifier_dir,
                EnvironmentPaths.artifacts_dir,
            )
        )
        configured_targets = {
            PurePosixPath(mount["target"])
            for mount in self._host_mounts
            if mount.get("type") == "bind"
        }
        if self._uses_compose:
            configured_targets.update(
                PurePosixPath(_volume_target(volume))
                for volume in self._load_compose_config()["services"][
                    "main"
                ].get("volumes", [])
            )
        binds: list[tuple[Path, PurePosixPath]] = []
        seen_targets: set[PurePosixPath] = set()
        for source, target in candidates:
            if any(
                target.is_relative_to(path) for path in configured_targets
            ) or any(target.is_relative_to(path) for path in seen_targets):
                continue
            binds.append((source, target))
            seen_targets.add(target)
        return tuple(binds)

    def _main_mount_sources(self) -> list[tuple[Path, PurePosixPath]]:
        mounts = [
            (Path(mount["source"]), PurePosixPath(mount["target"]))
            for mount in self._host_mounts
            if mount.get("type") == "bind"
        ]
        if self._uses_compose:
            mounts.extend(
                (
                    Path(volume.split(":")[0]),
                    PurePosixPath(_volume_target(volume)),
                )
                for volume in self._load_compose_config()["services"][
                    "main"
                ].get("volumes", [])
            )
        return mounts

    def _prepare_harbor_bind_sources(self) -> None:
        binds = self._harbor_writable_binds()
        for source, _target in binds:
            source.mkdir(parents=True, exist_ok=True)
            source.chmod(0o777)
        # A mount nested inside a writable bind, such as SciCode's
        # /app/test_data.h5 under the /app workdir, needs a mount point in
        # that bind's host directory; Singularity cannot create one there.
        for mount_source, mount_target in self._main_mount_sources():
            parents = [
                (source, target)
                for source, target in binds
                if mount_target != target
                and mount_target.is_relative_to(target)
            ]
            if not parents:
                continue
            source, target = max(parents, key=lambda bind: len(bind[1].parts))
            placeholder = source / mount_target.relative_to(target)
            if mount_source.is_dir():
                placeholder.mkdir(parents=True, exist_ok=True)
            else:
                placeholder.parent.mkdir(parents=True, exist_ok=True)
                placeholder.touch(exist_ok=True)

    def _prepare_disk_overlay(self) -> None:
        """Create a sparse ext3 overlay for the detected runtime."""
        if self._scratch_dir is None:
            raise RuntimeError("Singularity scratch directory is not prepared")
        mkfs_ext3 = shutil.which("mkfs.ext3")
        mkfs = mkfs_ext3 or shutil.which("mke2fs")
        if mkfs is None:
            raise RuntimeError(
                "Singularity --overlay requires mkfs.ext3 or mke2fs"
            )

        # The image is sparse, so the size only caps what the task can write.
        # 1 GiB was too small for agents that install a toolchain, such as
        # Codex's Node and npm, on tasks that set no storage_mb.
        storage_mb = self.task_env_config.storage_mb or _DEFAULT_OVERLAY_MB
        # SingularityCE 3.6 needs upper/ and work/ in the image, owned by the
        # invoking user. Apptainer with fakeroot creates them itself, and under
        # root-mapped fakeroot rejects pre-made ones as "upper dir is not
        # writable". Without fakeroot it runs as the invoking user, who cannot
        # create them in a bare image's root-owned root directory.
        layout = None
        if not (
            self._fakeroot and _runtime_is_apptainer(self._instance_runtime())
        ):
            layout = self._scratch_dir / "overlay-layout"
            for directory in (layout / "upper", layout / "work"):
                directory.mkdir(parents=True)
                directory.chmod(0o777)
        overlay = self._scratch_dir / "overlay.img"
        overlay.touch(mode=0o600)
        os.truncate(overlay, storage_mb * 1024 * 1024)
        try:
            command = [mkfs, "-q"]
            if mkfs_ext3 is None:
                command.extend(["-t", "ext3"])
            if layout is not None:
                command.extend(["-d", str(layout)])
            command.append(str(overlay))
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            overlay.unlink(missing_ok=True)
            detail = getattr(exc, "stderr", None) or str(exc)
            raise RuntimeError(
                f"Could not create Singularity writable overlay: {detail}"
            ) from exc
        finally:
            if layout is not None:
                shutil.rmtree(layout, ignore_errors=True)
        self._overlay_path = overlay

    def _main_seed_image(self) -> str:
        if self._sif_path is None:
            raise RuntimeError("Main Singularity image is not prepared")
        return str(self._sif_path)

    async def _fakeroot_probe_error(self) -> str | None:
        """Return why ``--fakeroot`` cannot run on this host, or ``None``."""
        command = [
            self._instance_runtime(),
            "exec",
            "--cleanenv",
            "--containall",
            "--no-home",
            "--workdir",
            str(self._runtime_workdir("fakeroot-probe")),
            "--fakeroot",
            self._main_seed_image(),
            "true",
        ]
        try:
            await self._run(*command)
        except RuntimeError as exc:
            lines = [line.strip() for line in str(exc).splitlines()]
            lines = [line for line in lines if line]
            # Report the runtime's own diagnosis, e.g. SingularityCE's
            # "FATAL: could not use fakeroot: no mapping entry found in
            # /etc/subuid", rather than the whole command line.
            reason = next(
                (line for line in reversed(lines) if "FATAL" in line),
                lines[-1] if lines else type(exc).__name__,
            )
            lowered = reason.lower()
            if not any(
                marker in lowered for marker in _FAKEROOT_FAILURE_MARKERS
            ):
                # A pull, image, or I/O failure says nothing about fakeroot;
                # fail the start rather than silently dropping container root.
                raise
            return reason
        return None

    async def _resolve_fakeroot(self) -> None:
        """Verify that this start can use ``--fakeroot``.

        Runs before the first command that would pass ``--fakeroot`` so the
        seed exec, instance start, Compose options, and user switching all
        see the resolved value.
        """
        if not self._fakeroot_setting:
            self._fakeroot = False
            self._warn_fakeroot_disabled()
            return
        runtime = self._instance_runtime()
        cached = self._fakeroot_unavailable_reasons.get(runtime)
        reason = cached or await self._fakeroot_probe_error()
        if reason is not None:
            self._fakeroot = False
            self._fakeroot_unavailable_reasons.setdefault(runtime, reason)
            self._fakeroot_unavailable(reason)
        self._fakeroot = True

    def _raise_if_fakeroot_known_unavailable(self) -> None:
        """Fail before building or pulling when an earlier probe failed."""
        if not self._fakeroot_setting:
            return
        reason = self._fakeroot_unavailable_reasons.get(
            self._instance_runtime()
        )
        if reason is not None:
            self._fakeroot_unavailable(reason)

    @staticmethod
    def _fakeroot_unavailable(reason: str) -> NoReturn:
        raise RuntimeError(
            "Singularity fakeroot is required (singularity_fakeroot=true) "
            f"but unavailable on this host: {reason}. Fakeroot needs a "
            "user-namespace mapping for the invoking user, such as "
            "/etc/subuid and /etc/subgid entries. Set "
            "singularity_fakeroot=false to run as the invoking host user "
            "instead; most tasks that install packages then fail."
        )

    def _warn_fakeroot_disabled(self) -> None:
        cls = DockerfileSingularityEnvironment
        # The warning also covers the user switch that fakeroot would allow.
        self._fakeroot_disabled_explained = True
        self.logger.log(
            logging.DEBUG if cls._fakeroot_disabled_warned else logging.WARNING,
            "Singularity fakeroot is disabled (singularity_fakeroot=false), so "
            "commands run as the invoking host user instead of container "
            "root. Root-owned image files stay read-only, and tasks that "
            "install system packages will fail.",
        )
        cls._fakeroot_disabled_warned = True

    async def _seed_harbor_bind_sources(self) -> None:
        """Copy existing image content into writable bind sources."""
        if self._staging_dir is None:
            raise RuntimeError("Singularity staging directory is not prepared")
        commands = []
        for source, target in self._harbor_writable_binds():
            destination = PurePosixPath("/staging") / source.relative_to(
                self._staging_dir
            )
            commands.append(
                f"if [ -d {shlex.quote(str(target))} ]; then "
                f"cp -a {shlex.quote(str(target) + '/.')} "
                f"{shlex.quote(str(destination) + '/')}; fi"
            )
        if not commands:
            return
        command = [
            self._instance_runtime(),
            "exec",
            "--cleanenv",
            "--containall",
            "--no-home",
            "--workdir",
            str(self._runtime_workdir("seed")),
            "--pwd",
            "/",
        ]
        if self._fakeroot:
            command.append("--fakeroot")
        if self._network_policy.network_mode == NetworkMode.NO_NETWORK:
            command.extend(["--net", "--network", "none"])
        command.extend([
            "-B",
            f"{self._staging_dir}:/staging",
            self._main_seed_image(),
            "sh",
            "-c",
            "set -e; " + "; ".join(commands),
        ])
        await self._run(*command)
        # ``cp -a source/. destination/`` also copies the source directory's
        # mode onto the bind root. Restore the writable contract after seeding.
        self._prepare_harbor_bind_sources()

    async def _list_instances(self) -> set[str]:
        command = [
            self._instance_runtime(),
            "instance",
            "list",
            "--json",
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Could not list Singularity instances: {exc}"
            ) from exc
        try:
            stdout, stderr = await process.communicate()
        except asyncio.CancelledError:
            await self._await_cleanup(self._terminate_host_process(process))
            raise
        if process.returncode:
            detail = stderr.decode(errors="replace") or stdout.decode(
                errors="replace"
            )
            raise RuntimeError(
                f"Could not list Singularity instances: {detail}"
            )
        return self._parse_instance_list(stdout)

    async def _instance_exists(self) -> bool:
        return self._instance_name in await self._list_instances()

    @staticmethod
    def _parse_instance_list(output: bytes) -> set[str]:
        try:
            instances = json.loads(output)["instances"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Could not parse Singularity instance list"
            ) from exc
        if not isinstance(instances, list) or any(
            not isinstance(instance, dict)
            or not isinstance(instance.get("instance"), str)
            for instance in instances
        ):
            raise RuntimeError("Could not parse Singularity instance list")
        return {instance["instance"] for instance in instances}

    async def _stop_instance(self, *, warn: bool) -> bool:
        try:
            exists = await self._instance_exists()
        except RuntimeError as exc:
            if warn:
                self.logger.warning(
                    "Failed to inspect Singularity instance %s: %s",
                    self._instance_name,
                    exc,
                )
                return False
            raise
        if not exists:
            self._instance_started = False
            self._instance_start_attempted = False
            return True
        try:
            await self._run(
                self._instance_runtime(),
                "instance",
                "stop",
                "--force",
                self._instance_name,
            )
        except RuntimeError as exc:
            if warn:
                self.logger.warning(
                    "Failed to stop Singularity instance %s: %s",
                    self._instance_name,
                    exc,
                )
                return False
            raise
        self._instance_started = False
        self._instance_start_attempted = False
        # Set by the first instance exec when Apptainer wraps execs in the
        # fakeroot command; see ``_SHARED_FAKED_SCRIPT``.
        self._shared_faked = False
        return True

    async def _stop_compose(self, *, warn: bool) -> bool:
        try:
            await self._run_compose("down", "--timeout", "0")
        except RuntimeError as exc:
            if self._compose_uses_sudo:
                if warn:
                    self.logger.warning(
                        "Failed to stop privileged singularity-compose "
                        "project; preserving runtime files: %s",
                        exc,
                    )
                    return False
                raise RuntimeError(
                    "Failed to stop privileged singularity-compose project"
                ) from exc
            try:
                existing = await self._list_instances()
            except RuntimeError as list_exc:
                if warn:
                    self.logger.warning(
                        "Failed to stop singularity-compose project (%s) "
                        "or inspect instances (%s)",
                        exc,
                        list_exc,
                    )
                    return False
                raise
            failures = []
            expected = {
                instance
                for instances in self._compose_instances.values()
                for instance in instances
            }
            for instance in expected & existing:
                try:
                    await self._run(
                        self._instance_runtime(),
                        "instance",
                        "stop",
                        "--force",
                        instance,
                    )
                except RuntimeError as stop_exc:
                    failures.append(str(stop_exc))
            if failures:
                if warn:
                    self.logger.warning(
                        "Failed to stop singularity-compose project (%s) and "
                        "instances (%s)",
                        exc,
                        "; ".join(failures),
                    )
                    return False
                raise
        self._instance_started = False
        self._instance_start_attempted = False
        # Set by the first instance exec when Apptainer wraps execs in the
        # fakeroot command; see ``_SHARED_FAKED_SCRIPT``.
        self._shared_faked = False
        return True

    async def _stop_prepared_runtime(self, *, warn: bool) -> bool:
        if self._uses_compose:
            return await self._stop_compose(warn=warn)
        return await self._stop_instance(warn=warn)

    async def _start_prepared_runtime(
        self, network_policy: NetworkPolicy
    ) -> None:
        if self._uses_compose:
            self._set_compose_network_policy(network_policy)
            lock_path = self._compose_start_lock_path()
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            async with AsyncFileLock(lock_path):
                compose_arguments = ["up"]
                if self._overlay_path is not None:
                    compose_arguments.append("--read_only")
                if self._compose_host_network:
                    # Its generated /etc/hosts maps the instance to a bridge
                    # address it never gets, and its resolv.conf lists only
                    # public resolvers. Keep the host's files, as without
                    # Compose.
                    compose_arguments.append("--no-resolv")
                self._instance_start_attempted = True
                compose_output = await self._run_compose(*compose_arguments)
            if not self._compose_uses_sudo:
                # Instances started through sudo belong to root and are not
                # listed for the invoking user.
                await self._verify_compose_instances(compose_output or "")
            if (
                network_policy.network_mode != NetworkMode.NO_NETWORK
                and not self._compose_host_network
            ):
                self._add_compose_service_aliases()
        else:
            self._instance_start_attempted = True
            await self._run(*self._instance_start_command(network_policy))
        self._instance_started = True
        if self._keeps_image_tmp:
            # Without its bind, the workdir that every exec starts in exists
            # only if the image has it; Docker creates a missing one.
            await self._run(
                *self._instance_exec_prefix(default_cwd="/"),
                "mkdir",
                "-p",
                self._workdir,
            )
        await self._detect_shared_faked()
        # Also starts the shared faked before any agent command.
        if self._shared_faked:
            await self._run(
                *self._instance_exec_prefix(),
                "/bin/sh",
                "-c",
                _FAKEROOT_COMMAND_SETUP,
                "sh",
                _SUDO_SHIM,
            )
        else:
            await self._run(*self._instance_exec_prefix(), "true")

    async def _detect_shared_faked(self) -> None:
        """Share one faked per instance under Apptainer's fakeroot command.

        Only Apptainer's root-mapped fakeroot, used without /etc/subuid
        entries, wraps each exec in fakeroot(1); its execs see FAKEROOTKEY.
        Subuid-mapped fakeroot and SingularityCE keep their own exec path.
        """
        self._shared_faked = False
        if not (
            self._fakeroot and _runtime_is_apptainer(self._instance_runtime())
        ):
            return
        output = await self._run(
            *self._instance_exec_prefix(),
            "/bin/sh",
            "-c",
            _FAKEROOT_COMMAND_PROBE,
        )
        self._shared_faked = _FAKEROOT_COMMAND_MARKER in output.split()

    async def _restore_prepared_runtime(
        self, network_policy: NetworkPolicy
    ) -> None:
        if self._instance_started or self._instance_start_attempted:
            stopped = await self._stop_prepared_runtime(warn=True)
            if not stopped:
                return
        try:
            async with asyncio.timeout(self._startup_timeout_sec):
                await self._start_prepared_runtime(network_policy)
        except Exception as exc:
            self.logger.warning(
                "Failed to restore Singularity network policy %s: %s",
                network_policy.network_mode.value,
                exc,
            )

    @override
    async def _apply_network_policy(
        self, network_policy: NetworkPolicy
    ) -> None:
        if self._overlay_path is None:
            raise RuntimeError(
                "Singularity network policy changes require a writable overlay"
            )
        if (
            network_policy.network_mode == NetworkMode.NO_NETWORK
            and self._uses_compose
            and set(self._load_compose_config()["services"]) - {"main"}
        ):
            raise ValueError(
                "singularity-compose cannot provide sidecar networking "
                "during a Harbor 'no-network' phase"
            )
        async with self._lifecycle_lock:
            if not self._instance_started:
                raise RuntimeError("Singularity environment is not running")
            previous_policy = self._network_policy
            try:
                await self._stop_prepared_runtime(warn=False)
                async with asyncio.timeout(self._startup_timeout_sec):
                    await self._start_prepared_runtime(network_policy)
            except BaseException:
                await self._await_cleanup(
                    self._restore_prepared_runtime(previous_policy)
                )
                raise

    async def _cleanup_failed_start(self) -> None:
        """Best-effort runtime cleanup without masking the startup error."""
        if not self._instance_start_attempted:
            self._cleanup_compose_project()
            self._cleanup_staging()
            await self._release_sifs_in_use()
            return
        stopped = True
        try:
            if self._uses_compose:
                if self._compose_file is not None:
                    stopped = await self._stop_compose(warn=True)
            else:
                stopped = await self._stop_instance(warn=True)
        except Exception as exc:
            stopped = False
            self.logger.warning(
                "Failed to clean up Singularity startup: %s", exc
            )
        if stopped:
            self._cleanup_compose_project()
            self._cleanup_staging()
            await self._release_sifs_in_use()

    @override
    async def start(self, force_build: bool) -> None:
        async with self._lifecycle_lock:
            if self._instance_started:
                raise RuntimeError("Singularity environment is already started")
            if (
                self._instance_start_attempted
                or self._runtime_workdir_temp is not None
            ):
                raise RuntimeError(
                    "Singularity environment requires stop before restart"
                )
            await self._start(force_build)

    async def _start(self, force_build: bool) -> None:
        if sys.platform == "win32":
            raise RuntimeError("Singularity is unavailable on Windows")
        try:
            self._raise_if_fakeroot_known_unavailable()
            self._raise_if_compose_bridge_unavailable()
            self._sif_path = await self._build_main_sif(force_build)
            await self._mark_sif_in_use(self._sif_path)
            self._prepare_runtime_workdir()
            self._scratch_dir = Path(
                tempfile.mkdtemp(prefix="ursa-harbor-singularity-")
            )
            self._scratch_dir.chmod(0o700)
            self._staging_dir = self._scratch_dir / "staging"
            self._staging_dir.mkdir(mode=0o755)
            self._prepare_host_ca_certificates()
            self._instance_start_attempted = False
            async with asyncio.timeout(self._startup_timeout_sec):
                await self._resolve_fakeroot()
                if self._overlay_enabled:
                    self._prepare_disk_overlay()
                self._prepare_harbor_bind_sources()
                await self._seed_harbor_bind_sources()
                if self._uses_compose:
                    await self._prepare_compose_project(
                        force_build or self._force_pull
                    )
                await self._start_prepared_runtime(self._network_policy)
            await self.ensure_dirs(self._mount_targets(writable_only=True))
            await self._upload_environment_dir_after_start()
        except TimeoutError as exc:
            await self._await_cleanup(self._cleanup_failed_start())
            raise TimeoutError(
                "Singularity instance did not become ready within "
                f"{self._startup_timeout_sec:g} seconds"
            ) from exc
        except BaseException:
            # An interrupted instance-start command can still leave an instance
            # behind, so cleanup is attempted before start reports success.
            await self._await_cleanup(self._cleanup_failed_start())
            raise

    def _prepare_runtime_workdir(self) -> None:
        if self._runtime_workdir_temp is not None:
            raise RuntimeError(
                "Singularity runtime workdir is already prepared"
            )
        temporary = tempfile.TemporaryDirectory(
            prefix="ursa-harbor-singularity-workdir-",
            ignore_cleanup_errors=True,
            delete=False,
        )
        self._runtime_workdir_temp = temporary
        self._runtime_workdir_root = Path(temporary.name)

    def _runtime_workdir(self, name: str) -> Path:
        if self._runtime_workdir_root is None:
            raise RuntimeError("Singularity runtime workdir is not prepared")
        workdir = self._runtime_workdir_root / name
        workdir.mkdir(mode=0o700, parents=True, exist_ok=True)
        return workdir

    def _cleanup_runtime_workdir(self) -> None:
        temporary = self._runtime_workdir_temp
        self._runtime_workdir_temp = None
        self._runtime_workdir_root = None
        if temporary is not None:
            temporary.cleanup()

    def _cleanup_staging(self) -> None:
        scratch_dir = getattr(self, "_scratch_dir", None)
        if scratch_dir is not None:
            shutil.rmtree(scratch_dir, ignore_errors=True)
        elif self._staging_dir is not None:
            shutil.rmtree(self._staging_dir, ignore_errors=True)
        self._scratch_dir = None
        self._staging_dir = None
        self._overlay_path = None
        self._cleanup_runtime_workdir()

    @override
    async def stop(self, delete: bool) -> None:
        async with self._lifecycle_lock:
            await self._stop(delete)

    async def _stop(self, delete: bool) -> None:
        stopped = True
        if self._uses_compose:
            if self._compose_file is not None:
                stopped = await self._stop_compose(warn=True)
        elif self._instance_started or self._instance_start_attempted:
            stopped = await self._stop_instance(warn=True)
        if stopped:
            self._cleanup_compose_project()
            self._cleanup_staging()
            await self._release_sifs_in_use()
        else:
            self.logger.warning(
                "Preserving Singularity runtime files for a later stop retry"
            )
        if delete:
            self.logger.debug(
                "Singularity image preserved at %s for reuse", self._sif_path
            )

    @override
    async def attach(self) -> None:
        """Replace this process with an interactive shell in the main instance."""
        if not self._instance_started:
            raise RuntimeError("Singularity instance is not running")
        command = [
            self._instance_runtime(),
            "shell",
            "--cleanenv",
            "--pwd",
            self._workdir,
            f"instance://{self._instance_name}",
        ]
        if self._shared_faked:
            # Run the image's shell action under the shared faked, so the
            # shell sees the ownership agent commands set and what it leaves
            # running keeps working after it exits.
            command = [
                *self._instance_exec_prefix(),
                "/.singularity.d/actions/shell",
            ]
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("SINGULARITYENV_", "APPTAINERENV_"))
        }
        os.execvpe(command[0], command, environment)

    def _exec_shell_command(
        self,
        command: str,
        cwd: str | None,
        user: str | int | None,
        pid_file: str,
        *,
        default_cwd: str | None = None,
        resolve_default_user: bool = True,
        home: str | None = None,
    ) -> str:
        command = f"cd {shlex.quote(cwd or default_cwd or self._workdir)} && {command}"
        if home is not None:
            # Singularity and Apptainer refuse to set HOME through their
            # environment prefixes, so export it in the container shell.
            command = f"export HOME={shlex.quote(home)}; {command}"
        requested_user = (
            self._resolve_user(user) if resolve_default_user else user
        )
        resolved_user = requested_user
        if not self._fakeroot:
            # A rootless Singularity instance can only execute as the invoking
            # host user. Attempting to honor Harbor's requested image user with
            # ``su`` prompts for a container password and prevents every
            # command from starting.
            resolved_user = None
            if requested_user is not None and not getattr(
                self, "_warned_user_switch_without_fakeroot", False
            ):
                self.logger.log(
                    logging.DEBUG
                    if getattr(self, "_fakeroot_disabled_explained", False)
                    else logging.WARNING,
                    "Singularity fakeroot is disabled; cannot switch to "
                    "requested container user %r, so commands will run as "
                    "the invoking host user",
                    requested_user,
                )
                self._warned_user_switch_without_fakeroot = True
        if resolved_user is not None:
            if isinstance(resolved_user, int):
                user_arg = f"$(getent passwd {resolved_user} | cut -d: -f1)"
            else:
                user_arg = shlex.quote(str(resolved_user))
            command = f"su {user_arg} -s /bin/bash -c {shlex.quote(command)}"
        quoted_pid_file = shlex.quote(pid_file)
        return (
            f"echo $$ > {quoted_pid_file}; "
            f"trap 'rm -f {quoted_pid_file}' EXIT; "
            f"{command}"
        )

    def _runtime_environment(
        self,
        env: dict[str, str] | None,
        *,
        merge_persistent: bool = True,
    ) -> dict[str, str]:
        runtime_env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("SINGULARITYENV_", "APPTAINERENV_"))
        }
        # Apptainer accepts SINGULARITYENV_ only with a deprecation warning
        # on every command's stderr.
        prefix = (
            "APPTAINERENV_"
            if _runtime_is_apptainer(self._instance_runtime())
            else "SINGULARITYENV_"
        )
        merged = self._merge_env(env) if merge_persistent else env
        merged = {
            **self._host_ca_certificates_env(),
            **(merged or {}),
        }
        for key, value in (merged or {}).items():
            if not key or "=" in key or "\0" in key or "\0" in value:
                raise ValueError(f"Invalid environment variable: {key!r}")
            # ``_exec_shell_command`` exports HOME, which the runtime rejects.
            if key != "HOME":
                runtime_env[f"{prefix}{key}"] = value
        return runtime_env

    async def _cleanup_exec_process(
        self,
        process: asyncio.subprocess.Process,
        pid_file: str,
        *,
        instance_name: str | None = None,
        default_cwd: str | None = None,
    ) -> None:
        cleanup = asyncio.create_task(
            self._run(
                *self._instance_exec_prefix(
                    instance_name=instance_name,
                    default_cwd=default_cwd,
                ),
                "bash",
                "-c",
                self._terminate_process_tree_command(pid_file),
            )
        )
        with suppress(Exception, asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(cleanup), timeout=10)
        await self._terminate_host_process(process)

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if not self._instance_started:
            raise RuntimeError("Singularity environment not started")

        return await self._exec_instance(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
        )

    async def _exec_instance(
        self,
        command: str,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        timeout_sec: int | None,
        user: str | int | None,
        instance_name: str | None = None,
        default_cwd: str | None = None,
        merge_persistent: bool = True,
    ) -> ExecResult:
        pid_file = f"/tmp/harbor-exec-{secrets.token_hex(8)}.pid"
        shell_command = self._exec_shell_command(
            command,
            cwd,
            user,
            pid_file,
            default_cwd=default_cwd,
            resolve_default_user=merge_persistent,
            home=self._exec_home(env, merge_persistent=merge_persistent),
        )
        runtime_command = [
            *self._instance_exec_prefix(
                cwd,
                instance_name=instance_name,
                default_cwd=default_cwd,
            ),
            "bash",
            "-c",
            shell_command,
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *runtime_command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                env=self._runtime_environment(
                    env, merge_persistent=merge_persistent
                ),
            )
        except OSError as exc:
            raise RuntimeError(
                f"Could not execute command in Singularity: {exc}"
            ) from exc

        communication = asyncio.create_task(process.communicate())
        timed_out = False
        try:
            if timeout_sec is None:
                stdout, stderr = await communication
            else:
                stdout, stderr = await asyncio.wait_for(
                    asyncio.shield(communication), timeout=timeout_sec
                )
        except TimeoutError:
            timed_out = True
            if instance_name is None:
                await self._cleanup_exec_process(process, pid_file)
            else:
                await self._cleanup_exec_process(
                    process,
                    pid_file,
                    instance_name=instance_name,
                    default_cwd=default_cwd,
                )
            try:
                stdout, stderr = await asyncio.wait_for(
                    communication, timeout=_EXEC_OUTPUT_DRAIN_SEC
                )
            except TimeoutError:
                # A background job the command started, e.g. ``server &``,
                # still holds the output pipes after the tree was killed.
                stdout, stderr = b"", b""
        except asyncio.CancelledError:
            cleanup_call = (
                self._cleanup_exec_process(process, pid_file)
                if instance_name is None
                else self._cleanup_exec_process(
                    process,
                    pid_file,
                    instance_name=instance_name,
                    default_cwd=default_cwd,
                )
            )
            cleanup = asyncio.create_task(cleanup_call)
            with suppress(asyncio.CancelledError):
                await asyncio.shield(cleanup)
            with suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(
                    asyncio.shield(communication),
                    timeout=_EXEC_OUTPUT_DRAIN_SEC,
                )
            communication.cancel()
            raise

        stdout_text = stdout.decode(errors="replace")
        stderr_text = stderr.decode(errors="replace")
        return_code = (
            process.returncode if process.returncode is not None else 1
        )
        if timed_out:
            return_code = 124
            timeout_message = f"Command timed out after {timeout_sec} seconds"
            stderr_text = "\n".join(
                part for part in (stderr_text.rstrip(), timeout_message) if part
            )
        if return_code != 0:
            error_output = stderr_text or stdout_text or "<no output>"
            self.logger.debug(
                "Command exited with rc=%s: %s", return_code, error_output
            )
        return ExecResult(
            stdout=stdout_text,
            stderr=stderr_text,
            return_code=return_code,
        )

    def _transfer_staging_path(self, name: str) -> Path:
        return self._staging / (
            f"harbor-transfer-{secrets.token_hex(8)}-{Path(name).name}"
        )

    @staticmethod
    def _remove_staging_path(path: Path) -> None:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)

    @staticmethod
    def _check_transfer(result: ExecResult, operation: str) -> None:
        if result.return_code != 0:
            error = result.stderr or result.stdout or "<no output>"
            raise RuntimeError(f"Failed to {operation}: {error}")

    @override
    async def upload_file(
        self, source_path: Path | str, target_path: str
    ) -> None:
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        staged = self._transfer_staging_path(source.name)
        try:
            self._remove_staging_path(staged)
            shutil.copy2(source, staged)
            result = await self.exec(
                f"cp {shlex.quote('/staging/' + staged.name)} "
                f"{shlex.quote(target_path)}"
            )
            self._check_transfer(result, "upload file")
        finally:
            self._remove_staging_path(staged)

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source = Path(source_dir)
        if not source.exists():
            raise FileNotFoundError(f"Source directory not found: {source}")

        staged = self._transfer_staging_path(source.name)
        try:
            self._remove_staging_path(staged)
            shutil.copytree(source, staged)
            result = await self.exec(f"mkdir -p {shlex.quote(target_dir)}")
            self._check_transfer(result, "prepare upload directory")
            result = await self.exec(
                f"cp -r {shlex.quote('/staging/' + staged.name + '/.')} "
                f"{shlex.quote(target_dir + '/')}"
            )
            self._check_transfer(result, "upload directory")
        finally:
            self._remove_staging_path(staged)

    def _skip_download_without_staging(self, source: str) -> bool:
        """Skip downloads when start() failed or the environment stopped.

        Harbor still collects logs after a failed start; raising here would
        log a second traceback that hides the startup error.
        """
        if self._staging_dir is not None:
            return False
        self.logger.debug(
            "Skipping download of %s; the Singularity environment is not "
            "running",
            source,
        )
        return True

    @override
    async def download_file(
        self, source_path: str, target_path: Path | str
    ) -> None:
        await self._download_file(source_path, target_path, self.exec)

    async def _download_file(
        self,
        source_path: str,
        target_path: Path | str,
        execute: Callable[[str], Awaitable[ExecResult]],
    ) -> None:
        if self._skip_download_without_staging(source_path):
            return
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = self._transfer_staging_path(Path(source_path).name)
        try:
            self._remove_staging_path(staged)
            result = await execute(
                f"cp {shlex.quote(source_path)} "
                f"{shlex.quote('/staging/' + staged.name)}"
            )
            self._check_transfer(result, "download file")
            if not staged.is_file():
                raise RuntimeError(f"File not found in staging: {staged}")
            shutil.copy2(staged, target)
        finally:
            self._remove_staging_path(staged)

    @override
    async def download_dir(
        self, source_dir: str, target_dir: Path | str
    ) -> None:
        await self._download_dir(source_dir, target_dir, self.exec)

    async def _download_dir(
        self,
        source_dir: str,
        target_dir: Path | str,
        execute: Callable[[str], Awaitable[ExecResult]],
    ) -> None:
        if self._skip_download_without_staging(source_dir):
            return
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        staged = self._transfer_staging_path(Path(source_dir).name)
        try:
            self._remove_staging_path(staged)
            result = await execute(
                f"cp -r {shlex.quote(source_dir)} "
                f"{shlex.quote('/staging/' + staged.name)}"
            )
            self._check_transfer(result, "download directory")
            if not staged.is_dir():
                raise RuntimeError(f"Directory not found in staging: {staged}")
            for item in staged.iterdir():
                destination = target / item.name
                if item.is_dir():
                    if destination.exists():
                        shutil.rmtree(destination)
                    shutil.copytree(item, destination)
                else:
                    shutil.copy2(item, destination)
        finally:
            self._remove_staging_path(staged)

    @override
    async def download_dir_filtered(
        self,
        *,
        source_dir: str,
        target_dir: Path | str,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        protect: Sequence[str] | None = None,
    ) -> None:
        if self._skip_download_without_staging(source_dir):
            return
        await super().download_dir_filtered(
            source_dir=source_dir,
            target_dir=target_dir,
            include=include,
            exclude=exclude,
            protect=protect,
        )

    def _compose_instance(self, service: str) -> str:
        instances = self._compose_instances.get(service)
        if not instances:
            raise ValueError(f"Unknown Docker Compose service: {service!r}")
        return instances[0]

    @override
    async def service_exec(
        self,
        command: str,
        *,
        service: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if self.is_main_service(service):
            return await self.exec(
                command,
                cwd=cwd,
                env=env,
                timeout_sec=timeout_sec,
                user=user,
            )
        instance = self._compose_instance(service)
        return await self._exec_instance(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
            instance_name=instance,
            default_cwd="/",
            merge_persistent=False,
        )

    @override
    async def service_download_file(
        self,
        source_path: str,
        target_path: Path | str,
        *,
        service: str | None = None,
    ) -> None:
        if self.is_main_service(service):
            await self.download_file(source_path, target_path)
            return
        await self._download_file(
            source_path,
            target_path,
            partial(self.service_exec, service=service),
        )

    @override
    async def service_download_dir(
        self,
        source_dir: str,
        target_dir: Path | str,
        *,
        service: str | None = None,
    ) -> None:
        if self.is_main_service(service):
            await self.download_dir(source_dir, target_dir)
            return
        await self._download_dir(
            source_dir,
            target_dir,
            partial(self.service_exec, service=service),
        )

    @override
    async def stop_service(self, service: str) -> None:
        async with self._lifecycle_lock:
            await self._stop_service(service)

    async def _stop_service(self, service: str) -> None:
        instances = self._compose_instances.get(service)
        if not instances:
            raise ValueError(f"Unknown Docker Compose service: {service!r}")
        await self._run_compose("stop", "--timeout", "0", *instances)
        if service == "main":
            self._instance_started = False

    @staticmethod
    def _terminate_process_tree_command(pid_file: str) -> str:
        quoted_pid_file = shlex.quote(pid_file)
        return (
            "i=0; "
            f'while [ ! -s {quoted_pid_file} ] && [ "$i" -lt 20 ]; do '
            "sleep 0.1; i=$((i + 1)); done; "
            f"if [ -s {quoted_pid_file} ]; then "
            f"pid=$(cat {quoted_pid_file}); "
            "case $pid in *[!0-9]*|'') exit 0;; esac; "
            "descendants() { for child in "
            '$(cat "/proc/$1/task/$1/children" 2>/dev/null); '
            'do descendants "$child"; echo "$child"; done; }; '
            'children=$(descendants "$pid"); '
            'kill -TERM $children "$pid" 2>/dev/null || true; '
            'i=0; while kill -0 "$pid" 2>/dev/null '
            '&& [ "$i" -lt 20 ]; do '
            "sleep 0.1; i=$((i + 1)); done; "
            'kill -KILL $children "$pid" 2>/dev/null || true; '
            f"rm -f {quoted_pid_file}; fi"
        )


__all__ = [
    "DockerfileSingularityEnvironment",
    "docker_compose_to_singularity_compose",
]
