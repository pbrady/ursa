"""A Dockerfile and singularity-compose environment for Harbor tasks."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
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
import tempfile
from collections.abc import (
    Awaitable,
    Callable,
    Generator,
    Iterable,
    Mapping,
    Sequence,
)
from contextlib import AsyncExitStack, contextmanager, suppress
from copy import deepcopy
from functools import cache, partial
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, ClassVar, NoReturn, cast, override
from urllib.parse import urlsplit

import archspec.cpu
import yaml
from filelock import AsyncFileLock
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.environments.compose_service_ops import (
    ComposeServiceOpsMixin,
    ComposeServiceTransport,
)
from harbor.environments.definition import should_use_prebuilt_docker_image
from harbor.environments.docker.compose_env import (
    ComposeInfraEnvVars,
    legacy_log_mount_env_vars,
)
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import (
    NetworkMode,
    NetworkPolicy,
    normalize_allowed_hosts,
)
from harbor.models.trial.paths import EnvironmentPaths
from harbor.utils.env import resolve_env_vars
from pathspec import GitIgnoreSpec

from ursa.integrations.harbor import network_proxy
from ursa.util.http import OpenSSLContext

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
_PROXY_CONTAINER_DIR = "/run/ursa-proxy"
_HARBOR_WRITABLE_BIND_TARGETS = (
    ("installed-agent", PurePosixPath("/installed-agent")),
    ("solution", EnvironmentPaths.solution_dir),
    ("tests", EnvironmentPaths.tests_dir),
    ("skills", EnvironmentPaths.default_skills_dir),
)
_HARBOR_WRITABLE_IMAGE_TARGETS = (
    EnvironmentPaths.agent_dir,
    EnvironmentPaths.user_agent_dir,
    EnvironmentPaths.verifier_dir,
    EnvironmentPaths.artifacts_dir,
)
_COMPOSE_VARIABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_COMPOSE_BRIDGE_REQUIREMENT = (
    "Compose projects with sidecars or published ports need a setuid "
    "Apptainer installation with /etc/subuid and /etc/subgid entries for the "
    "invoking user, or root (or Docker): an unprivileged Apptainer gives "
    "non-root users only --network none."
)
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
# Apptainer's root-mapped fakeroot bind-mounts the host's fakeroot(1) into
# builds and containers; it cannot load in images with an older glibc than
# the host's, or with musl. Such a failure is about the image, not the host,
# and ``--ignore-fakeroot-command`` avoids it.
_FAKEROOT_COMMAND_FAILURE_MARKERS = (
    "error while starting the `faked' daemon",
    "required by /.singularity.d/libs/libfakeroot",
)
_warned_unset_compose_variables: set[str] = set()
_logger = logging.getLogger(__name__)


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


def _host_ca_certificates_pem() -> str:
    """Serialize the host OpenSSL trust anchors as one PEM bundle."""
    context = OpenSSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_default_certs()
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


def _volume_target(volume: str) -> str:
    """Return the container path of a normalized ``src:dst[:ro]`` volume."""
    return volume.split(":")[1]


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
    # unset, and the argument is expanded only when it is used.
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


def _is_compose_port(value: Any) -> bool:
    """Return whether a Compose port endpoint has a supported scalar form."""
    return (type(value) is int and value >= 0) or (
        isinstance(value, str) and re.fullmatch(r"[0-9]+", value) is not None
    )


def _normalize_build(name: str, build: Any, base_dir: Path) -> Any:
    if isinstance(build, str):
        return _compose_path(build, base_dir)
    if build is None:
        return build
    if not isinstance(build, dict):
        raise ValueError(
            f"Docker Compose build on service {name!r} must be a path or mapping"
        )
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
    for field, default in (("dockerfile", "Dockerfile"), ("target", "")):
        if not isinstance(build.get(field, default), str):
            raise ValueError(
                f"Docker Compose build {field} on service {name!r} must be a string"
            )
    args = build.get("args", {})
    if not isinstance(args, (dict, list)) or not all(
        isinstance(item, str) for item in args
    ):
        raise ValueError(
            f"Docker Compose build args on service {name!r} must be a mapping "
            "or list using string names"
        )
    return build


def _normalize_env_files(name: str, entries: Any, base_dir: Path) -> list[str]:
    normalized = []
    for entry in entries if isinstance(entries, list) else [entries]:
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
            required, entry = entry.get("required", True), entry["path"]
        if not isinstance(entry, str):
            raise ValueError(
                f"Docker Compose env_file on service {name!r} must contain paths"
            )
        path = _compose_path(entry, base_dir)
        if required or Path(path).is_file():
            normalized.append(path)
    return normalized


def _normalize_depends_on(name: str, value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list) and all(
        isinstance(dependency, str) for dependency in value
    ):
        return value
    if not isinstance(value, dict):
        raise ValueError(
            f"Docker Compose depends_on on service {name!r} must be a list of names"
        )
    if not all(isinstance(dependency, str) for dependency in value):
        raise ValueError(
            f"Docker Compose depends_on on service {name!r} must use string names"
        )
    for options in value.values():
        options = options or {}
        if (
            not isinstance(options, dict)
            or set(options) - {"condition", "required", "restart"}
            or options.get("condition", "service_started") != "service_started"
            or options.get("required", True) is not True
            or options.get("restart", False) is not False
        ):
            raise ValueError(
                "singularity-compose supports only service_started "
                f"depends_on conditions on service {name!r}"
            )
    return list(value)


def _normalize_volumes(
    name: str, volumes: Any, base_dir: Path
) -> list[str] | None:
    if volumes is None:
        return None
    if not isinstance(volumes, list):
        raise ValueError(
            f"Docker Compose volumes on service {name!r} must be a list"
        )
    normalized = []
    for volume in volumes:
        error = (
            "singularity-compose supports only SOURCE:TARGET[:ro|rw] bind "
            f"mounts on service {name!r}: {volume!r}"
        )
        if isinstance(volume, dict):
            if (
                set(volume) - {"read_only", "source", "target", "type"}
                or volume.get("type") != "bind"
                or not isinstance(volume.get("read_only", False), bool)
                or not all(
                    isinstance(volume.get(field), str)
                    for field in ("source", "target")
                )
            ):
                raise ValueError(error)
            volume = f"{volume['source']}:{volume['target']}" + (
                ":ro" if volume.get("read_only") else ""
            )
        parts = volume.split(":") if isinstance(volume, str) else []
        if (
            len(parts) not in (2, 3)
            or not parts[0].startswith(("/", ".", "~"))
            or not parts[1].startswith("/")
            or parts[2:] not in ([], ["ro"], ["rw"])
        ):
            raise ValueError(error)
        # Singularity splits its bind list on commas.
        if "," in volume:
            raise ValueError(
                "Singularity bind mounts cannot contain commas on "
                f"service {name!r}: {volume!r}"
            )
        normalized.append(
            f"{_compose_path(parts[0], base_dir)}:{parts[1]}"
            + (":ro" if parts[2:] == ["ro"] else "")
        )
    return normalized


def _normalize_ports(name: str, ports: Any) -> list[str] | None:
    if ports is None:
        return None
    if not isinstance(ports, list):
        raise ValueError(
            f"Docker Compose ports on service {name!r} must be a list"
        )
    normalized = []
    for port in ports:
        if isinstance(port, dict):
            valid = (
                not (set(port) - {"protocol", "published", "target"})
                and port.get("protocol", "tcp") == "tcp"
                and all(
                    _is_compose_port(port.get(field))
                    for field in ("published", "target")
                )
            )
            port = f"{port['published']}:{port['target']}" if valid else port
        elif (
            not isinstance(port, str)
            or re.fullmatch(r"[0-9]+:[0-9]+", port) is None
        ):
            valid = False
        else:
            valid = True
        if not valid:
            raise ValueError(
                "singularity-compose supports only TCP published ports "
                f"on service {name!r}: {port!r}"
            )
        normalized.append(port)
    return normalized


def _normalize_compose_service(
    name: str, service: dict[str, Any], base_dir: Path
) -> None:
    unsupported = set(service) - _COMPOSE_SERVICE_FIELDS
    if unsupported:
        raise ValueError(
            "singularity-compose does not support these fields on service "
            f"{name!r}: {', '.join(sorted(unsupported))}"
        )
    normalizers = {
        "build": lambda value: _normalize_build(name, value, base_dir),
        "env_file": lambda value: _normalize_env_files(name, value, base_dir),
        "volumes": lambda value: _normalize_volumes(name, value, base_dir),
        "ports": lambda value: _normalize_ports(name, value),
    }
    for field, normalize in normalizers.items():
        if field in service and (
            field != "env_file" or service[field] is not None
        ):
            service[field] = normalize(service[field])
    command, environment = (
        service.get("command", ""),
        service.get("environment", {}),
    )
    if not isinstance(command, (str, list)) or (
        isinstance(command, list)
        and not all(isinstance(argument, str) for argument in command)
    ):
        raise ValueError(
            f"Docker Compose command on service {name!r} must contain only strings"
        )
    if not isinstance(environment, (dict, list)) or not all(
        isinstance(item, str) for item in environment
    ):
        raise ValueError(
            f"Docker Compose environment on service {name!r} must contain only strings"
        )
    deploy = service.get("deploy")
    if deploy is not None:
        replicas = (
            deploy.get("replicas", 1) if isinstance(deploy, dict) else None
        )
        if not isinstance(deploy, dict) or set(deploy) - {
            "replicas",
            "resources",
        }:
            raise ValueError(
                "singularity-compose supports only deploy.replicas and "
                f"deploy.resources on service {name!r}"
            )
        # Like the task's cpus and memory_mb, Compose resource limits are
        # not enforced; Harbor's Terminal-Bench adapter always emits them.
        deploy.pop("resources", None)
        if type(replicas) is not int or replicas < 1:
            raise ValueError(
                f"Docker Compose replicas on service {name!r} must be a positive integer"
            )
    service["depends_on"] = _normalize_depends_on(
        name, service.get("depends_on")
    )


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


def _resolve_compose_mapping(
    values: dict[str, Any] | list[str],
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve Compose environment/build arguments in mapping or list form."""
    if isinstance(values, dict):
        return {
            key: _resolve_compose_value(key, value, environ)
            for key, value in values.items()
        }
    resolved = {}
    for entry in values:
        key, separator, value = entry.partition("=")
        resolved[key] = _resolve_compose_value(
            key, value if separator else None, environ
        )
    return resolved


def _read_compose_env_file(
    path: Path, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    environ = os.environ if environ is None else environ
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
    environment = {}
    for path in service.get("env_file", []):
        environment.update(
            _read_compose_env_file(Path(path), interpolation_environment)
        )
    environment.update(
        _resolve_compose_mapping(
            service.get("environment", {}), interpolation_environment
        )
    )
    if service_name == "main":
        environment.update(main_environment)
    return environment


def _shell_exports(environment: dict[str, str], *, context: str) -> str:
    """Validate shell variable names and quote values without interpolation."""
    invalid = [
        key
        for key in environment
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None
    ]
    if invalid:
        raise ValueError(
            f"Invalid {context} environment variable names: "
            + ", ".join(sorted(invalid))
        )
    return "".join(
        f"export {key}={shlex.quote(value)}\n"
        for key, value in sorted(environment.items())
    )


def _write_compose_environment(
    project_dir: Path,
    identity: str,
    service_name: str,
    environment: dict[str, str],
) -> Path | None:
    if not environment:
        return None
    contents = _shell_exports(environment, context="Compose")
    path = project_dir / f"{_compose_key(identity, service_name)}.env.sh"
    path.write_text(contents)
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
            target.is_relative_to(parent) for parent in read_only_targets
        )

    return (
        [volume for volume in volumes if not under_read_only(volume)],
        [volume for volume in volumes if under_read_only(volume)],
    )


@cache
def _runtime_lacks_setuid_starter(runtime: str) -> bool:
    """Whether ``runtime`` is certainly an unprivileged (non-setuid) install.

    A setuid installation ships a root-owned setuid ``starter-suid`` next to
    ``starter`` in ``$LIBEXECDIR/{apptainer,singularity}/bin``; SingularityCE
    and Apptainer both report ``LIBEXECDIR`` from ``buildcfg``. Without that
    starter, Apptainer runs in a user namespace and gives non-root users only
    ``--network none``. The answer is ``True`` only when a ``starter`` is
    found without a setuid ``starter-suid`` beside it, so an unexpected
    layout or a failing ``buildcfg`` never rejects a setuid installation.
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
    # Relocated unprivileged installations keep libexec beside bin.
    with suppress(OSError):
        libexec_dirs.append(Path(runtime).resolve().parent.parent / "libexec")
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
    fakeroot: bool | None = None,
    ignore_fakeroot_command: bool = False,
    main_agent_layer: Path | None = None,
    main_overlay: Path | None = None,
    service_workdirs: Mapping[str, Sequence[Path]] | None = None,
    interpolation_environment: Mapping[str, str] | None = None,
) -> dict[str, list[str]]:
    """Convert one or more Docker Compose files into a singularity-compose file.

    ``interpolation_environment`` supplies the Compose variables and defaults
    to the host environment. ``ignore_fakeroot_command`` adds
    ``--ignore-fakeroot-command`` to fakeroot starts.
    """
    if fakeroot is None:
        fakeroot = _runtime_is_apptainer(
            DockerfileSingularityEnvironment._runtime()
        )
    paths = (
        [docker_compose_paths]
        if isinstance(docker_compose_paths, Path)
        else list(docker_compose_paths)
    )
    config = _load_docker_compose(paths, interpolation_environment)
    # A lone main service needs no bridge: host networking matches the
    # non-Compose path and needs no CNI networking, which unprivileged
    # runtimes lack.
    host_network = _compose_uses_host_network(config)
    service_keys = {
        name: _compose_key(identity, name) for name in config["services"]
    }
    replicas_by_service = {
        name: service.get("deploy", {}).get("replicas", 1)
        for name, service in config["services"].items()
    }
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
            if name not in service_workdirs:
                raise ValueError(
                    f"Missing Singularity workdir for service {name!r}"
                )
            workdirs = service_workdirs[name]
            if len(workdirs) != replicas:
                raise ValueError(
                    f"Expected {replicas} Singularity workdir(s) for service {name!r}, got {len(workdirs)}"
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
        start_options = ["containall", "no-home"]
        if fakeroot:
            start_options.insert(0, "fakeroot")
            if ignore_fakeroot_command:
                start_options.insert(1, "ignore-fakeroot-command")
        if name == "main" and main_agent_layer is not None:
            start_options.append(f"overlay={main_agent_layer}:ro")
        if name == "main" and main_overlay is not None:
            start_options.append(f"overlay={main_overlay}")
        elif main_overlay is not None:
            start_options.append("writable-tmpfs")
        network: dict[str, Any] = {
            "allocate_ip": network_mode == NetworkMode.PUBLIC,
            "enable": network_mode != NetworkMode.PUBLIC or not host_network,
        }
        start: dict[str, Any] = {"options": start_options}
        instance: dict[str, Any] = {
            "image": await image_resolver(name, service),
            "network": network,
            "start": start,
        }
        if network_mode != NetworkMode.PUBLIC:
            network["type"] = "none"
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
                key
                for dependency in depends_on
                for key in instance_keys[dependency]
            ]
        if ports := service.get("ports", []):
            instance["ports"] = ports
        if command := service.get("command"):
            start["args"] = (
                shlex.join(command) if isinstance(command, list) else command
            )
        replicas = replicas_by_service[name]
        if service_workdirs is None and replicas != 1:
            instance["deploy"] = {"replicas": replicas}
        if service_workdirs is None:
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


class _SingularityComposeServiceTransport:
    """Route Harbor's generic Compose operations to Singularity instances."""

    def __init__(self, environment: DockerfileSingularityEnvironment) -> None:
        self.environment = environment

    def _execute(self, service: str) -> Callable[..., Awaitable[ExecResult]]:
        return partial(
            self.environment._exec_instance,
            cwd=None,
            env=None,
            timeout_sec=None,
            user=None,
            instance_name=self.environment._compose_instance(service),
            default_cwd="/",
            merge_persistent=False,
        )

    async def service_exec(
        self,
        command: str,
        *,
        service: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        return await self._execute(service)(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
        )

    async def service_download_file(
        self, source_path: str, target_path: Path | str, *, service: str
    ) -> None:
        await self.environment._download(
            source_path, target_path, self._execute(service), directory=False
        )

    async def service_download_dir(
        self, source_dir: str, target_dir: Path | str, *, service: str
    ) -> None:
        await self.environment._download(
            source_dir, target_dir, self._execute(service), directory=True
        )

    async def stop_service(self, service: str) -> None:
        async with self.environment._lifecycle_lock:
            await self.environment._stop_service(service)


class DockerfileSingularityEnvironment(ComposeServiceOpsMixin, BaseEnvironment):
    """Run Dockerfile and supported Compose tasks with Singularity or Apptainer."""

    _compose_filename = "docker-compose.yaml"
    # Why ``--fakeroot`` failed, per runtime: later trials in the same process
    # fail before building instead of probing again.
    _fakeroot_unavailable_reasons: ClassVar[dict[str, str]] = {}
    # An explicit opt-out is explained once per process.
    _fakeroot_disabled_warned: ClassVar[bool] = False

    def __init__(
        self,
        *args,
        singularity_image_cache_dir: Path | str | None = None,
        singularity_agent: str | None = None,
        singularity_agent_kwargs: dict | str | None = None,
        singularity_agent_cache_dir: Path | str | None = None,
        singularity_force_pull: bool | int | str = False,
        singularity_fakeroot: bool | int | str | None = None,
        singularity_overlay: bool | int | str = True,
        singularity_no_mount: str | None = None,
        singularity_startup_timeout_sec: float = 300,
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
                "always isolates mounts with --containall and --no-home"
            )

        self._image_cache_dir = (
            Path(singularity_image_cache_dir)
            if singularity_image_cache_dir
            else self._default_image_cache_dir()
        )
        self._force_pull = self._parse_bool(
            "singularity_force_pull", singularity_force_pull
        )
        singularity_overlay = self._parse_bool(
            "singularity_overlay", singularity_overlay
        )
        # Harbor 0.25 forwards environment kwargs to separate verifiers too.
        # Their explicit auto-mount list contains only the verifier log mount;
        # they must use their own image without an agent installation layer.
        verifier_only = {
            PurePosixPath(mount["target"])
            for mount in kwargs.get("auto_mounts") or []
        } == {EnvironmentPaths.verifier_dir}
        self._agent_spec = None if verifier_only else singularity_agent
        if isinstance(singularity_agent_kwargs, str):
            singularity_agent_kwargs = json.loads(singularity_agent_kwargs)
        self._agent_kwargs = (
            {} if singularity_agent_kwargs is None else singularity_agent_kwargs
        )
        if not isinstance(self._agent_kwargs, dict):
            raise ValueError("singularity_agent_kwargs must be a JSON object")
        self._agent_install_key: str | None = None
        self._agent_cache_dir = (
            Path(
                singularity_agent_cache_dir
                or os.environ.get("URSA_HARBOR_AGENT_CACHE")
                or self._cache_root() / "agents"
            )
            .expanduser()
            .resolve()
        )
        self._agent_layer_path: Path | None = None
        if self._agent_spec is not None and not singularity_overlay:
            raise ValueError(
                "An agent install layer requires a writable trial overlay"
            )
        # ``None`` follows the runtime.
        self._fakeroot_setting = (
            None
            if singularity_fakeroot is None
            else self._parse_bool("singularity_fakeroot", singularity_fakeroot)
        )
        self._fakeroot = self._fakeroot_setting
        self._overlay_enabled = singularity_overlay
        self._runtime_path: str | None = None
        # Resolved by ``_prepare_workdir`` at start, or on first use.
        self._resolved_workdir: str | None = None
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
        identity = hashlib.sha256(self.session_id.encode()).hexdigest()[:16]
        self._instance_name = f"ursa{identity}{secrets.token_hex(4)}"
        self._compose_identity = f"{identity[:8]}{secrets.token_hex(4)}"
        self._instance_started = False
        self._instance_start_attempted = False
        # Set by the fakeroot probe when the host's fakeroot command cannot
        # run in this image; see ``_FAKEROOT_COMMAND_FAILURE_MARKERS``.
        self._ignore_fakeroot_command = False
        self._warned_user_switch_without_fakeroot = False
        self._fakeroot_disabled_explained = False
        self._compose_project_dir: Path | None = None
        self._compose_file: Path | None = None
        self._compose_instances: dict[str, list[str]] = {}
        self._compose_uses_sudo = False
        self._compose_host_network = False
        self._sif_use_token = f"{os.getpid()}-{secrets.token_hex(16)}"
        self._sifs_in_use: set[Path] = set()
        self._proxy_temp: tempfile.TemporaryDirectory[str] | None = None
        self._proxy_stack: AsyncExitStack | None = None
        self._proxy_process: asyncio.subprocess.Process | None = None
        self._proxy_env: dict[str, str] = {}
        self._model_forwards: dict[tuple[str, int], int] = {}

    @staticmethod
    def _parse_bool(name: str, value: bool | int | str) -> bool:
        """Parse a boolean environment kwarg such as ``singularity_overlay``."""
        if isinstance(value, bool):
            return value
        # Harbor's CLI JSON-decodes ``--ek`` values, so ``=1`` arrives as int,
        # while other configuration paths can pass the raw string, whose
        # truthiness would turn ``"false"`` on.
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no"}:
                return False
        raise ValueError(f"{name} must be true or false")

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
        # Restricted phases use an isolated namespace; allowlisted HTTP(S)
        # travels through a host-owned Unix-socket proxy. Phase changes restart
        # the instance while retaining its writable overlay.
        return EnvironmentCapabilities(
            mounted=True,
            disable_internet=True,
            network_allowlist=True,
            network_allowlist_hostnames=True,
            network_allowlist_wildcard_hostnames=True,
            network_allowlist_ipv4_addresses=True,
            network_allowlist_ipv6_addresses=True,
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
    def _cache_root() -> Path:
        cache_home = os.environ.get("XDG_CACHE_HOME")
        return (
            (
                Path(cache_home).expanduser()
                if cache_home
                else Path.home() / ".cache"
            )
            / "ursa"
            / "harbor"
        )

    @classmethod
    def _default_image_cache_dir(cls) -> Path:
        configured = os.environ.get("URSA_HARBOR_SIF_CACHE")
        return (
            Path(configured).expanduser()
            if configured
            else cls._cache_root() / "sif"
        )

    @classmethod
    def _compose_start_lock_path(cls) -> Path:
        return cls._cache_root() / "singularity-compose.lock"

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
        if self._fakeroot is None:
            self._fakeroot = _runtime_is_apptainer(runtime)
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
            mount for mount in self._mounts if mount.get("type") != "bind"
        ]
        if unsupported_mounts:
            raise ValueError("Singularity supports only Harbor bind mounts")
        if any(
            PurePosixPath(mount["target"]).is_relative_to(_PROXY_CONTAINER_DIR)
            for mount in self._mounts
        ):
            raise ValueError(
                f"{_PROXY_CONTAINER_DIR} is reserved for network access"
            )
        config = None
        if self._uses_compose:
            config = self._load_compose_config()
            sidecars = set(config["services"]) - {"main"}
            if sidecars and any(
                policy.network_mode != NetworkMode.PUBLIC
                for policy in (
                    self._network_policy,
                    *self._phase_network_policies,
                )
            ):
                raise ValueError(
                    "singularity-compose cannot provide sidecar networking "
                    "during a restricted Harbor network phase"
                )
        if not self._overlay_enabled and any(
            policy != self._network_policy
            for policy in self._phase_network_policies
        ):
            raise ValueError(
                "Singularity network policy changes require a writable overlay"
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
        environment, the task's ``[environment].env``, and persistent env.
        """
        environment = dict(os.environ)
        if self.task_env_config.env:
            environment.update(resolve_env_vars(self.task_env_config.env))
        environment.update(self._persistent_env)
        main_image_name = getattr(self, "_compose_main_image_name", None)
        if main_image_name is None:
            # ``environment_id`` hashes the environment directory.
            main_image_name = f"hb__{self.environment_id}".lower()
            self._compose_main_image_name = main_image_name
        memory_mb = self._effective_memory_mb
        environment.update(
            ComposeInfraEnvVars(
                main_image_name=main_image_name,
                context_dir=str(self.environment_dir.resolve()),
                prebuilt_image_name=self.task_env_config.docker_image,
                cpus=self._effective_cpus,
                memory=f"{memory_mb}M" if memory_mb else None,
            ).to_env_dict()
        )
        environment.update(
            legacy_log_mount_env_vars(self._host_mounts, host_value="source")
        )
        return environment

    def _load_compose_config(self) -> dict[str, Any]:
        return _load_docker_compose(
            self._compose_paths(), self._compose_interpolation_environment()
        )

    @property
    def _workdir(self) -> str:
        """The directory commands start in; see ``_resolve_workdir``."""
        if self._resolved_workdir is None:
            self._resolved_workdir = self._resolve_workdir(force_build=False)
        return self._resolved_workdir

    @_workdir.setter
    def _workdir(self, value: str | None) -> None:
        self._resolved_workdir = value

    async def _prepare_workdir(self, force_build: bool) -> None:
        """Resolve the workdir for this start without blocking the loop."""
        self._resolved_workdir = await asyncio.to_thread(
            self._resolve_workdir, force_build
        )

    def _resolve_workdir(self, force_build: bool) -> str:
        """Resolve task overrides and Dockerfile WORKDIR instructions.

        A pulled image, a Compose ``main.image`` or a prebuilt
        ``docker_image`` without a Dockerfile, starts in ``/`` with a
        warning; a task whose image uses another WORKDIR sets
        ``[environment].workdir``. A prebuilt image that comes with its
        Dockerfile takes the Dockerfile's WORKDIR. ``force_build`` is the
        trial's, which decides whether the Dockerfile is built.
        """
        if self.task_env_config.workdir is not None:
            return self.task_env_config.workdir
        dockerfile_path = self._dockerfile_path
        build = self._main_compose_build_inputs()
        if build is not None:
            dockerfile_path = build[0]
        else:
            image = self._main_image(force_build)
            if image is not None and (
                image == self._compose_main_image()
                or not dockerfile_path.is_file()
            ):
                return self._image_workdir(image)
            if not dockerfile_path.is_file():
                return "/"
        workdir = PurePosixPath("/")
        for line in dockerfile_path.read_text().splitlines():
            instruction = line.strip()
            if instruction.upper().startswith("FROM "):
                workdir = PurePosixPath("/")
            elif instruction.upper().startswith("WORKDIR "):
                value = line.split(None, 1)[1].strip()
                if "$" in value:
                    raise ValueError(
                        "Singularity cannot resolve variables in Dockerfile "
                        f"WORKDIR: {value}"
                    )
                path = PurePosixPath(value)
                workdir = path if path.is_absolute() else workdir / path
        return str(workdir)

    def _image_workdir(self, image: str) -> str:
        """Return the directory a pulled image's commands start in."""
        self.logger.warning(
            "Cannot read the WORKDIR of %s; commands start in /. Set "
            "[environment].workdir if it has another.",
            image,
        )
        return "/"

    def _dockerfile_cache_path_sync(
        self,
        dockerfile_path: Path,
        context_dir: Path,
        build_args: tuple[str, ...],
        target: str | None,
    ) -> Path:
        dockerfile_path = dockerfile_path.resolve()
        context_dir = context_dir.resolve()
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "architecture": platform.machine(),
                    "build_args": build_args,
                    "microarchitecture": archspec.cpu.host().name,
                    "target": target,
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        )
        ignore_path = context_dir / ".dockerignore"
        ignore = (
            GitIgnoreSpec.from_lines(ignore_path.read_text().splitlines())
            if ignore_path.is_file()
            else None
        )
        dockerfile_in_context = None
        with suppress(ValueError):
            dockerfile_in_context = dockerfile_path.relative_to(
                context_dir
            ).as_posix()
        for path in sorted(context_dir.rglob("*")):
            relative = path.relative_to(context_dir).as_posix()
            if (
                ignore is not None
                and relative not in {dockerfile_in_context, ".dockerignore"}
                and ignore.match_file(relative)
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
    ) -> Path:
        return await asyncio.to_thread(
            self._dockerfile_cache_path_sync,
            dockerfile_path or self._dockerfile_path,
            context_dir or self.environment_dir,
            build_args,
            target,
        )

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

    async def _set_sif_in_use(
        self, path: Path, present: bool | None, *, touch: bool = False
    ) -> None:
        async with AsyncFileLock(path.with_suffix(".lock")):
            if touch:
                os.utime(path)
            if present is None:
                return
            marker = self._sif_in_use_path(path)
            users = (
                set(marker.read_text().splitlines())
                if marker.is_file()
                else set()
            )
            (users.add if present else users.discard)(self._sif_use_token)
            if users:
                marker.write_text(
                    "".join(f"{user}\n" for user in sorted(users))
                )
            else:
                marker.unlink(missing_ok=True)

    async def _mark_sif_in_use(self, path: Path) -> None:
        """Refresh a cached SIF and record this environment as a user."""
        if not path.is_relative_to(self._image_cache_dir):
            # A task's own local SIF file is not part of the cache.
            return
        present = True if path not in self._sifs_in_use else None
        await self._set_sif_in_use(path, present=present, touch=True)
        if present:
            self._sifs_in_use.add(path)

    async def _release_sifs_in_use(self) -> None:
        """Remove this environment from each cached SIF's users."""
        for path in tuple(self._sifs_in_use):
            try:
                await self._set_sif_in_use(path, present=False)
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

    def _warn_root_failure(self, output: Iterable[str]) -> None:
        runtime = (
            self._runtime_path
            or shutil.which("apptainer")
            or shutil.which("singularity")
        )
        if runtime is None or _runtime_is_apptainer(runtime):
            return
        if any(
            re.search(
                r"\b(?:(?:must|needs? to) be (?:the )?root|requires? (?:root|superuser)|"
                r"(?:must|has to|needs? to) (?:be )?run "
                r"(?:(?:as|by|under) (?:the )?root|with (?:root|superuser) privileges)|"
                r"(?:root|superuser) privileges (?:are )?required|"
                r"are you root|(?:you are|you're) not root)\b",
                line,
                re.IGNORECASE,
            )
            for line in output
        ):
            self.logger.warning(
                "Singularity reported a root-permission failure. Consider "
                "switching to Apptainer, for which this integration enables "
                "fakeroot by default."
            )

    async def _run(
        self,
        *command: str,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        failure_log_path: Path | None = None,
    ) -> str:
        """Run a host command; return its output unless it goes to a log."""
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
                        output=cast(BinaryIO, captured),
                    )
                    detail = ""
                else:
                    stdout, stderr = cast(tuple[bytes, bytes], result)
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
            stdout, stderr = cast(tuple[bytes, bytes], result)
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
        output = await self._dockerfile_cache_path(
            dockerfile_path, context_dir, build_args, target
        )
        self._image_cache_dir.mkdir(parents=True, exist_ok=True)
        if not force_build and await self._is_valid_sif(output):
            return output
        async with AsyncFileLock(output.with_suffix(".lock")):
            if not force_build and await self._is_valid_sif(output):
                return output
            builders = [
                builder
                for name in ("buildah", "podman", "docker")
                if (builder := shutil.which(name)) is not None
            ]
            if not builders:
                raise RuntimeError(
                    "Building a Dockerfile for Singularity requires buildah, "
                    "podman, or docker"
                )
            temporary = output.with_suffix(".tmp.sif")
            build_log_path = output.with_suffix(".build.log")
            build_log_path.unlink(missing_ok=True)
            failures = []
            for builder in builders:
                try:
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
                    build_log_path.unlink(missing_ok=True)
                    return output
                except RuntimeError as exc:
                    failures.append(str(exc))
            log_note = (
                f" Build log: {build_log_path.absolute()}"
                if build_log_path.is_file()
                else ""
            )
            if build_log_path.is_file():
                self._surface_build_log(build_log_path)
                with suppress(OSError):
                    with build_log_path.open(errors="replace") as log:
                        self._warn_root_failure(log)
            raise RuntimeError(
                "No container builder succeeded: "
                + "; ".join(failures)
                + log_note
            )

    def _compose_key(self, service: str) -> str:
        identity = getattr(
            self,
            "_compose_identity",
            hashlib.sha256(self.session_id.encode()).hexdigest()[:16],
        )
        return _compose_key(identity, service)

    @staticmethod
    def _image_uri(image: str) -> str:
        if not isinstance(image, str) or not image:
            raise ValueError("Singularity image must be a non-empty string")
        return image if "://" in image else f"docker://{image}"

    async def _pull_image_sif(self, image: str, force_build: bool) -> Path:
        """Pull an image into the SIF cache, or use a local ``*.sif`` file."""
        if (
            isinstance(image, str)
            and "://" not in image
            and image.endswith(".sif")
        ):
            path = Path(image).expanduser()
            if not await self._is_valid_sif(path):
                raise FileNotFoundError(f"Singularity image not found: {path}")
            return path
        image = self._image_uri(image)
        digest = hashlib.sha256(
            json.dumps([platform.machine(), image]).encode()
        ).hexdigest()[:24]
        output = self._image_cache_dir / f"image-{digest}.sif"
        self._image_cache_dir.mkdir(parents=True, exist_ok=True)
        async with AsyncFileLock(output.with_suffix(".lock")):
            if not force_build and await self._is_valid_sif(output):
                return output
            temporary = output.with_suffix(".tmp.sif")
            temporary.unlink(missing_ok=True)
            try:
                await self._run(
                    self._instance_runtime(),
                    "pull",
                    *(["--disable-cache"] if force_build else []),
                    str(temporary),
                    image,
                )
                if not await self._is_valid_sif(temporary):
                    raise RuntimeError(
                        f"Pulled image is not a valid SIF: {image}"
                    )
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
        return output

    def _compose_build_inputs(
        self, build: str | dict[str, Any]
    ) -> tuple[Path, Path, tuple[str, ...], str | None]:
        if isinstance(build, str):
            context = Path(build)
            return context / "Dockerfile", context, (), None
        context = Path(build.get("context", "."))
        resolved_args = _resolve_compose_mapping(
            build.get("args", {}), self._compose_interpolation_environment()
        )
        return (
            context / build.get("dockerfile", "Dockerfile"),
            context,
            tuple(
                f"{key}={value}" for key, value in sorted(resolved_args.items())
            ),
            build.get("target"),
        )

    def _main_compose_build_inputs(
        self,
    ) -> tuple[Path, Path, tuple[str, ...], str | None] | None:
        build = (
            self._load_compose_config()["services"]["main"].get("build")
            if self._uses_compose
            else None
        )
        return self._compose_build_inputs(build) if build is not None else None

    async def _build_sif(
        self,
        build: tuple[Path, Path, tuple[str, ...], str | None],
        force_build: bool,
    ) -> Path:
        dockerfile, context, build_args, target = build
        return await self._build_dockerfile_sif(
            force_build,
            dockerfile_path=dockerfile,
            context_dir=context,
            build_args=build_args,
            target=target,
        )

    def _compose_main_image(self) -> str | None:
        """Return the Compose ``main.image`` when main has no build section."""
        if not self._uses_compose:
            return None
        main = self._load_compose_config()["services"]["main"]
        return None if "build" in main else main.get("image")

    def _prebuilt_image(self, force_build: bool) -> str | None:
        """Return the task's ``docker_image`` when Harbor would pull it."""
        image = self.task_env_config.docker_image
        if not should_use_prebuilt_docker_image(
            self.environment_dir, docker_image=image, force_build=force_build
        ):
            return None
        return image

    def _main_image(self, force_build: bool) -> str | None:
        """Return the image the main SIF is pulled from, or None to build it.

        As with Harbor's Docker backend, the task's prebuilt ``docker_image``
        wins over the Dockerfile unless the trial forces a build, and a
        Compose ``main.image`` replaces that prebuilt image. When Harbor
        would build, ``main.image`` only names the built image, so it is
        pulled only if there is no Dockerfile to build.
        """
        prebuilt = self._prebuilt_image(force_build)
        image = self._compose_main_image()
        if image is not None and (
            prebuilt is not None or not self._dockerfile_path.is_file()
        ):
            return image
        return prebuilt

    async def _build_main_sif(
        self, force_build: bool, *, refresh: bool = False
    ) -> Path:
        # ``refresh`` and ``singularity_force_pull`` rebuild or re-pull the
        # cached SIF but never turn a prebuilt-image task into a build.
        refresh = force_build or refresh or self._force_pull
        image = self._main_image(force_build)
        if image is not None:
            return await self._pull_image_sif(image, refresh)
        build = self._main_compose_build_inputs()
        if build is None:
            return await self._build_dockerfile_sif(refresh)
        return await self._build_sif(build, refresh)

    async def build_sifs(
        self, force_build: bool = False, *, refresh: bool = False
    ) -> list[Path]:
        """Build or pull all environment and Compose SIFs without starting them.

        ``force_build`` has Harbor's meaning and can switch a prebuilt-image
        task to a Dockerfile build. ``refresh`` only rebuilds or re-pulls the
        SIFs that a trial without ``--force-build`` would use.
        """
        main_path = await self._build_main_sif(force_build, refresh=refresh)
        paths = [main_path]
        if not self._uses_compose:
            return paths

        refresh = force_build or refresh or self._force_pull
        compose_config = self._load_compose_config()
        main_image = self._main_image(force_build)
        main_build = None
        if main_image is None:
            main_build = self._main_compose_build_inputs() or (
                self._dockerfile_path,
                self.environment_dir,
                (),
                None,
            )
        seen_builds = {main_build} if main_build is not None else set()
        seen_images = (
            {self._image_uri(main_image)} if main_image is not None else set()
        )
        for name, service in compose_config["services"].items():
            if name == "main":
                continue
            if "build" in service:
                build = self._compose_build_inputs(service["build"])
                if build in seen_builds:
                    continue
                seen_builds.add(build)
                path = await self._build_sif(build, refresh)
            else:
                image = self._image_uri(service["image"])
                if image in seen_images:
                    continue
                seen_images.add(image)
                path = await self._pull_image_sif(service["image"], refresh)
            if path not in paths:
                paths.append(path)
        return paths

    async def _compose_image(
        self,
        service_name: str,
        service: dict[str, Any],
        force_build: bool,
    ) -> str:
        if service_name == "main":
            if self._sif_path is None:
                raise RuntimeError("Main Singularity image is not prepared")
            path = self._sif_path
        elif "build" in service:
            path = await self._build_sif(
                self._compose_build_inputs(service["build"]), force_build
            )
        else:
            path = await self._pull_image_sif(service["image"], force_build)
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
        service_workdirs = {
            name: tuple(
                self._runtime_workdir(
                    f"compose-{self._compose_key(name)}-{replica}"
                )
                for replica in range(
                    1, service.get("deploy", {}).get("replicas", 1) + 1
                )
            )
            for name, service in compose_config["services"].items()
        }
        main_mounts = [
            *self._mounts,
            *(
                {
                    "type": "bind",
                    "source": str(source),
                    "target": str(target),
                }
                for source, target in self._harbor_writable_binds()
            ),
        ]
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
            ignore_fakeroot_command=self._ignore_fakeroot_command,
            main_agent_layer=self._agent_layer_path,
            main_overlay=self._overlay_path,
            service_workdirs=service_workdirs,
            interpolation_environment=self._compose_interpolation_environment(),
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
        isolated = network_policy.network_mode != NetworkMode.PUBLIC
        instances = compose["instances"].values()
        # As in the conversion, a lone main instance without published ports
        # uses host networking for public phases.
        self._compose_host_network = len(instances) == 1 and not any(
            instance.get("ports") for instance in instances
        )
        for instance in instances:
            network = instance.setdefault("network", {})
            network.update(
                allocate_ip=not isolated,
                enable=isolated or not self._compose_host_network,
            )
            if isolated:
                network["type"] = "none"
            else:
                network.pop("type", None)
            options = instance.setdefault("start", {}).setdefault("options", [])
            options[:] = [
                value
                for value in options
                if value != "userns"
                and (
                    not value.startswith("bind=")
                    or f":{_PROXY_CONTAINER_DIR}:" not in value
                )
            ]
            if isolated and not self._fakeroot:
                options.append("userns")
            if network_policy.network_mode == NetworkMode.ALLOWLIST:
                options.append(f"bind={self._proxy_bind()}")
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
        environment = self._clean_host_environment()
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
        # Restricted phases start every instance with ``--network none``.
        return any(
            policy.network_mode == NetworkMode.PUBLIC
            for policy in (self._network_policy, *self._phase_network_policies)
        )

    def _raise_if_compose_bridge_unavailable(self) -> None:
        """Fail before building when the runtime offers only --network none.

        Detection is conservative: root, ``singularity_fakeroot=false``, and
        any installation with a setuid starter are never rejected here; an
        undetected failure is still caught after ``up``.
        """
        if not self._uses_compose:
            return
        runtime = self._instance_runtime()
        if not self._fakeroot or os.geteuid() == 0:
            # Without fakeroot, singularity-compose never asks the runtime for
            # a bridge: it uses sudo for published ports and otherwise adds
            # ``--network none``.
            return
        if not self._compose_needs_bridge():
            return
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
        return [
            self._instance_runtime(),
            "exec",
            "--cleanenv",
            *self._fakeroot_command_options(),
            "--pwd",
            cwd or default_cwd or self._workdir,
            f"instance://{instance_name or self._instance_name}",
        ]

    def _instance_start_command(
        self, network_policy: NetworkPolicy | None = None
    ) -> list[str]:
        if self._staging_dir is None or self._sif_path is None:
            raise RuntimeError("Singularity instance is not prepared")
        command = [self._instance_runtime(), "instance", "start"]
        command.extend(
            self._isolated_runtime_args(
                self._runtime_workdir("main"),
                network_policy=network_policy,
                include_overlay=True,
            )
        )
        for source, target in self._harbor_writable_binds():
            command.extend(["-B", f"{source}:{target}"])
        command.extend(["-B", f"{self._staging_dir}:/staging"])
        for mount in self._mounts:
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
        if (
            network_policy or self._network_policy
        ).network_mode == NetworkMode.ALLOWLIST:
            command.extend(["-B", self._proxy_bind()])
        command.extend([str(self._sif_path), self._instance_name])
        return command

    def _isolated_runtime_args(
        self,
        workdir: Path,
        *,
        network_policy: NetworkPolicy | None = None,
        include_overlay: bool = False,
        pwd: str | None = None,
    ) -> list[str]:
        """Return the isolation options shared by starts and seed copies."""
        args = ["--containall", "--no-home", "--workdir", str(workdir)]
        if pwd is not None:
            args.extend(["--pwd", pwd])
        if self._fakeroot:
            args.append("--fakeroot")
            args.extend(self._fakeroot_command_options())
        if self._agent_layer_path is not None:
            args.extend(["--overlay", f"{self._agent_layer_path}:ro"])
        if include_overlay and self._overlay_path is not None:
            args.extend(["--overlay", str(self._overlay_path)])
        if (
            network_policy or self._network_policy
        ).network_mode != NetworkMode.PUBLIC:
            if not self._fakeroot:
                args.append("--userns")
            args.extend(["--net", "--network", "none"])
        return args

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
            (root / name, target)
            for name, target in _HARBOR_WRITABLE_BIND_TARGETS
        ]
        workdir = PurePosixPath(self._workdir)
        if workdir != PurePosixPath("/"):
            candidates.append((root / "workdir", workdir))
        candidates.extend(
            (root / target.relative_to("/"), target)
            for target in _HARBOR_WRITABLE_IMAGE_TARGETS
        )
        configured_targets = {
            PurePosixPath(mount["target"])
            for mount in self._mounts
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
                target.is_relative_to(path)
                for path in configured_targets | seen_targets
            ):
                continue
            binds.append((source, target))
            seen_targets.add(target)
        return tuple(binds)

    def _prepare_harbor_bind_sources(self) -> None:
        # Harbor 0.25 mirrors /logs/artifacts below artifacts/logs/artifacts.
        # TrialPaths.mkdir() creates only the collection root. Unlike Docker,
        # Singularity requires automatic bind sources to exist at startup.
        for mount in self._mounts:
            if mount["target"] == str(EnvironmentPaths.artifacts_dir):
                source = Path(mount["source"])
                expected = self.trial_paths.host_artifact_path(
                    "main", str(EnvironmentPaths.artifacts_dir)
                )
                if source == expected:
                    source.mkdir(parents=True, exist_ok=True)
                    source.chmod(0o777)
        for source, _target in self._harbor_writable_binds():
            source.mkdir(parents=True, exist_ok=True)
            source.chmod(0o777)

    def _prepare_disk_overlay(self) -> None:
        """Create a sparse overlay with ownership suited to the runtime."""
        if self._scratch_dir is None:
            raise RuntimeError("Singularity scratch directory is not prepared")
        runtime = self._instance_runtime()
        storage_mb = self.task_env_config.storage_mb or 1024
        layout = self._scratch_dir / "overlay-layout"
        overlay = self._scratch_dir / "overlay.img"
        try:
            if self._fakeroot and _runtime_is_apptainer(runtime):
                # Let Apptainer map overlay ownership, including on hosts
                # without subordinate UID/GID ranges.
                command = [
                    runtime,
                    "overlay",
                    "create",
                    "--fakeroot",
                    "--sparse",
                    "--size",
                    str(storage_mb),
                    str(overlay),
                ]
            else:
                # Singularity 3.6 has no overlay-create command.
                mkfs_ext3 = shutil.which("mkfs.ext3")
                mkfs = mkfs_ext3 or shutil.which("mke2fs")
                if mkfs is None:
                    raise RuntimeError(
                        "Singularity --overlay requires mkfs.ext3 or mke2fs"
                    )
                for directory in (layout / "upper", layout / "work"):
                    directory.mkdir(parents=True)
                    directory.chmod(0o777)
                overlay.touch(mode=0o600)
                os.truncate(overlay, storage_mb * 1024 * 1024)
                command = [mkfs, "-q"]
                if mkfs_ext3 is None:
                    command.extend(["-t", "ext3"])
                command.extend(["-d", str(layout), str(overlay)])
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
            shutil.rmtree(layout, ignore_errors=True)
        self._overlay_path = overlay

    def _main_seed_image(self) -> str:
        if self._sif_path is not None:
            return str(self._sif_path)
        main = self._load_compose_config()["services"]["main"]
        image = main.get("image")
        if not isinstance(image, str) or not image:
            raise RuntimeError("Main Singularity image is not prepared")
        return image if "://" in image else f"docker://{image}"

    def _fakeroot_command_options(self) -> list[str]:
        """Return the options that keep the fakeroot command out of execs."""
        if self._fakeroot and self._ignore_fakeroot_command:
            return ["--ignore-fakeroot-command"]
        return []

    async def _fakeroot_probe_error(self) -> str | None:
        """Return why ``--fakeroot`` cannot run on this host, or ``None``.

        When only the host's fakeroot command fails to load in this image,
        the host itself can run fakeroot: the probe is repeated with
        ``--ignore-fakeroot-command``, which this trial's starts and execs
        then also pass, as the Apptainer build does. A failure of that
        retry is this image's, so it fails the start without a reason that
        later trials would reuse.
        """

        async def probe(*extra: str) -> None:
            await self._run(
                self._instance_runtime(),
                "exec",
                "--cleanenv",
                "--containall",
                "--no-home",
                "--workdir",
                str(self._runtime_workdir("fakeroot-probe")),
                "--fakeroot",
                *extra,
                self._main_seed_image(),
                "true",
            )

        self._ignore_fakeroot_command = False
        try:
            await probe()
        except RuntimeError as exc:
            if any(
                marker in str(exc)
                for marker in _FAKEROOT_COMMAND_FAILURE_MARKERS
            ):
                await probe("--ignore-fakeroot-command")
                self.logger.warning(
                    "The host's fakeroot command cannot run in %s; using "
                    "--ignore-fakeroot-command",
                    self._main_seed_image(),
                )
                self._ignore_fakeroot_command = True
                return None
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

        Runs before the first command that passes ``--fakeroot``, including
        Apptainer's ``overlay create``, so a host without a usable fakeroot
        fails with the runtime's reason instead of a later raw error.
        """
        runtime = self._instance_runtime()
        if not self._fakeroot:
            if self._fakeroot_setting is False:
                self._warn_fakeroot_disabled()
            return
        reason = (
            self._fakeroot_unavailable_reasons.get(runtime)
            or await self._fakeroot_probe_error()
        )
        if reason is not None:
            self._fakeroot = False
            self._fakeroot_unavailable_reasons.setdefault(runtime, reason)
            self._fakeroot_unavailable(reason)

    def _raise_if_fakeroot_known_unavailable(self) -> None:
        """Fail before building or pulling when an earlier probe failed."""
        if not self._fakeroot_unavailable_reasons:
            return
        runtime = self._instance_runtime()
        reason = self._fakeroot_unavailable_reasons.get(runtime)
        if self._fakeroot and reason is not None:
            self._fakeroot_unavailable(reason)

    @staticmethod
    def _fakeroot_unavailable(reason: str) -> NoReturn:
        raise RuntimeError(
            "Singularity fakeroot is enabled (the default with Apptainer, or "
            "singularity_fakeroot=true) but unavailable on this host: "
            f"{reason}. Fakeroot needs unprivileged user namespaces, and "
            "outside Apptainer's root-mapped mode a user-namespace mapping "
            "for the invoking user, such as /etc/subuid and /etc/subgid "
            "entries. Set singularity_fakeroot=false to run as the invoking "
            "host user instead; most tasks that install packages then fail."
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
        ]
        command.extend(
            self._isolated_runtime_args(self._runtime_workdir("seed"), pwd="/")
        )
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
        operation = "inspect"
        try:
            if await self._instance_exists():
                operation = "stop"
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
                    "Failed to %s Singularity instance %s: %s",
                    operation,
                    self._instance_name,
                    exc,
                )
                return False
            raise
        self._instance_started = False
        self._instance_start_attempted = False
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
        return True

    async def _stop_prepared_runtime(self, *, warn: bool) -> bool:
        await self._stop_network_proxy()
        if self._uses_compose:
            return await self._stop_compose(warn=warn)
        return await self._stop_instance(warn=warn)

    async def _start_prepared_runtime(
        self, network_policy: NetworkPolicy
    ) -> None:
        await self._prepare_network_proxy(network_policy)
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
                network_policy.network_mode == NetworkMode.PUBLIC
                and not self._compose_host_network
            ):
                self._add_compose_service_aliases()
        else:
            self._instance_start_attempted = True
            await self._run(*self._instance_start_command(network_policy))
        self._instance_started = True
        await self._run(*self._instance_exec_prefix(), "true")
        if network_policy.network_mode == NetworkMode.ALLOWLIST:
            await self._start_proxy_bridge()

    def _proxy_bind(self) -> str:
        if self._proxy_temp is None:
            raise RuntimeError("Network proxy is not prepared")
        return f"{self._proxy_temp.name}:{_PROXY_CONTAINER_DIR}:ro"

    async def _prepare_network_proxy(self, policy: NetworkPolicy) -> None:
        await self._stop_network_proxy()
        if policy.network_mode != NetworkMode.ALLOWLIST:
            return
        if self._proxy_temp is None:
            # Short paths work with AF_UNIX even when the job lives on a long
            # shared-filesystem path. This directory is private to this trial.
            self._proxy_temp = tempfile.TemporaryDirectory(
                prefix="ursa-net-", dir="/tmp"
            )
            shutil.copyfile(
                network_proxy.__file__,
                Path(self._proxy_temp.name) / "bridge.py",
            )
        self._proxy_stack = AsyncExitStack()
        await self._proxy_stack.enter_async_context(
            network_proxy.host_server(
                Path(self._proxy_temp.name) / "host.sock",
                allowed_hosts=normalize_allowed_hosts(policy.allowed_hosts),
            )
        )

    async def _start_proxy_bridge(self) -> None:
        port, self._proxy_process = await self._launch_proxy_bridge()
        endpoint = f"http://127.0.0.1:{port}"
        bypass = "localhost,localhost.,127.0.0.1,::1"
        self._proxy_env = {"NO_PROXY": bypass, "no_proxy": bypass}
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            self._proxy_env[key] = self._proxy_env[key.lower()] = endpoint

    async def _launch_proxy_bridge(
        self, *, target: str | None = None, host: str = "127.0.0.1"
    ) -> tuple[int, asyncio.subprocess.Process]:
        if self._proxy_stack is None:
            raise RuntimeError("Network proxy is not prepared")
        # Only the stdlib is needed. An installed URSA runtime is a fallback
        # for task images without a system Python interpreter.
        command = (
            "if command -v python3 >/dev/null 2>&1; then python=$(command -v python3); "
            "elif test -x /installed-agent/tools/ursa-ai/bin/python; then "
            "python=/installed-agent/tools/ursa-ai/bin/python; else "
            "echo 'Allowlisted networking requires Python 3.8+ in the task image' >&2; exit 1; fi; "
            f'exec "$python" {_PROXY_CONTAINER_DIR}/bridge.py '
            f"--socket {_PROXY_CONTAINER_DIR}/host.sock --port 0"
        )
        if target is not None:
            command += " " + shlex.join(["--host", host, "--target", target])
        log_path = (
            Path(self._proxy_temp.name) / f"bridge-{secrets.token_hex(4)}.log"
        )
        log = self._proxy_stack.enter_context(log_path.open("wb"))
        process = await asyncio.create_subprocess_exec(
            *self._instance_exec_prefix(cwd="/"),
            "sh",
            "-c",
            command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=log,
            start_new_session=True,
            env=self._runtime_environment({}, merge_persistent=False),
        )
        self._proxy_stack.push_async_callback(
            self._terminate_host_process, process
        )
        line = await asyncio.wait_for(
            process.stdout.readline(), self._startup_timeout_sec
        )
        try:
            port = json.loads(line)["port"]
        except (ValueError, KeyError, TypeError):
            port = None
        if (
            type(port) is not int
            or not 1 <= port <= 65535
            or process.returncode is not None
        ):
            details = log_path.read_text(errors="replace")
            raise RuntimeError(
                f"Could not start network proxy bridge: {details.strip()}"
            )
        return port, process

    async def _forward_model_url(self, url: str) -> str:
        """Forward a host-local model through its own container loopback port."""
        if self._proxy_stack is None:
            return url
        host, port = network_proxy.destination(url)
        if host == "localhost":
            bind_host = "127.0.0.1"
        else:
            try:
                if not ipaddress.ip_address(host).is_loopback:
                    return url
            except ValueError:
                return url
            bind_host = host
        async with self._lifecycle_lock:
            if self._proxy_stack is None:
                raise RuntimeError(
                    "Network proxy stopped while preparing model access"
                )
            if not network_proxy.HostProxy(
                None,
                normalize_allowed_hosts(self._network_policy.allowed_hosts),
            ).permits((host, port)):
                raise ValueError(f"Model host {host!r} is not allowlisted")
            key = (host, port)
            authority = f"[{host}]" if ":" in host else host
            if key not in self._model_forwards:
                forwarded, _process = await self._launch_proxy_bridge(
                    target=f"{authority}:{port}", host=bind_host
                )
                self._model_forwards[key] = forwarded
                bypass = self._proxy_env["NO_PROXY"].split(",")
                if host not in bypass:
                    bypass.append(host)
                    self._proxy_env["NO_PROXY"] = self._proxy_env[
                        "no_proxy"
                    ] = ",".join(bypass)
            # Retain the hostname for HTTPS certificate verification and SNI.
            # Only the port changes; model API paths and query strings survive.
            parsed = urlsplit(url)
            return parsed._replace(
                netloc=f"{authority}:{self._model_forwards[key]}"
            ).geturl()

    async def _stop_network_proxy(self) -> None:
        self._proxy_env = {}
        self._model_forwards.clear()
        self._proxy_process = None
        stack, self._proxy_stack = self._proxy_stack, None
        if stack is not None:
            await stack.aclose()

    async def _restart_prepared_runtime(
        self, network_policy: NetworkPolicy, *, warn_stop: bool
    ) -> None:
        if self._instance_started or self._instance_start_attempted:
            stopped = await self._stop_prepared_runtime(warn=warn_stop)
            if warn_stop and not stopped:
                return
        async with asyncio.timeout(self._startup_timeout_sec):
            await self._start_prepared_runtime(network_policy)

    async def _restore_prepared_runtime(
        self, network_policy: NetworkPolicy
    ) -> None:
        try:
            await self._restart_prepared_runtime(network_policy, warn_stop=True)
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
            network_policy.network_mode != NetworkMode.PUBLIC
            and self._uses_compose
            and set(self._load_compose_config()["services"]) - {"main"}
        ):
            raise ValueError(
                "singularity-compose cannot provide sidecar networking "
                "during a restricted Harbor network phase"
            )
        async with self._lifecycle_lock:
            if not self._instance_started:
                raise RuntimeError("Singularity environment is not running")
            previous_policy = self._network_policy
            try:
                await self._restart_prepared_runtime(
                    network_policy, warn_stop=False
                )
            except BaseException:
                await self._await_cleanup(
                    self._restore_prepared_runtime(previous_policy)
                )
                raise

    async def _cleanup_failed_start(self) -> None:
        """Best-effort runtime cleanup without masking the startup error."""
        needs_stop = self._instance_start_attempted and (
            not self._uses_compose or self._compose_file is not None
        )
        if needs_stop:
            try:
                if not await self._stop_prepared_runtime(warn=True):
                    return
            except Exception as exc:
                self.logger.warning(
                    "Failed to clean up Singularity startup: %s", exc
                )
                return
        await self._cleanup_runtime_resources()

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
            await self._prepare_workdir(force_build)
            self._sif_path = await self._build_main_sif(force_build)
            if self._sif_path is not None:
                await self._mark_sif_in_use(self._sif_path)
            if self._agent_spec is not None:
                from ursa.integrations.harbor.agent_layers import (
                    build_agent_layer,
                )

                self._agent_layer_path = await build_agent_layer(
                    self, self._agent_spec, require_cached=True
                )
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
                    # A forced pull also refreshes the sidecar SIFs.
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
        if self._agent_layer_path is not None:
            self._agent_layer_path.unlink(missing_ok=True)
        self._agent_layer_path = None
        self._cleanup_runtime_workdir()

    async def _cleanup_runtime_resources(self) -> None:
        await self._stop_network_proxy()
        if self._proxy_temp is not None:
            self._proxy_temp.cleanup()
            self._proxy_temp = None
        self._cleanup_compose_project()
        self._cleanup_staging()
        await self._release_sifs_in_use()

    @override
    async def stop(self, delete: bool) -> None:
        async with self._lifecycle_lock:
            await self._stop(delete)

    async def _stop(self, delete: bool) -> None:
        needs_stop = (
            self._compose_file is not None
            if self._uses_compose
            else self._instance_started or self._instance_start_attempted
        )
        if not needs_stop or await self._stop_prepared_runtime(warn=True):
            await self._cleanup_runtime_resources()
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
        """Open a shell, retaining the host process when it owns a proxy."""
        if not self._instance_started:
            raise RuntimeError("Singularity instance is not running")
        command = [
            self._instance_runtime(),
            "shell",
            "--cleanenv",
            *self._fakeroot_command_options(),
            "--pwd",
            self._workdir,
            f"instance://{self._instance_name}",
        ]
        if self._proxy_process is not None:
            process = await asyncio.create_subprocess_exec(
                *command, env=self._runtime_environment(None)
            )
            await process.wait()
        else:
            os.execvpe(command[0], command, self._clean_host_environment())

    @staticmethod
    def _clean_host_environment() -> dict[str, str]:
        return {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("SINGULARITYENV_", "APPTAINERENV_"))
        }

    def _exec_shell_command(
        self,
        command: str,
        cwd: str | None,
        user: str | int | None,
        pid_file: str,
        *,
        default_cwd: str | None = None,
        resolve_default_user: bool = True,
        environment_file: str | None = None,
    ) -> str:
        command = f"cd {shlex.quote(cwd or default_cwd or self._workdir)} && {command}"
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
        source_environment = (
            f". {shlex.quote(environment_file)}; "
            if environment_file is not None
            else ""
        )
        return (
            f"echo $$ > {quoted_pid_file}; "
            f"trap 'rm -f {quoted_pid_file}' EXIT; "
            f"{source_environment}{command}"
        )

    def _runtime_environment(
        self,
        env: dict[str, str] | None,
        *,
        merge_persistent: bool = True,
    ) -> dict[str, str]:
        runtime_env = self._clean_host_environment()
        # Apptainer accepts SINGULARITYENV_ only with a deprecation warning
        # on every command's stderr.
        prefix = (
            "APPTAINERENV_"
            if _runtime_is_apptainer(self._instance_runtime())
            else "SINGULARITYENV_"
        )
        configured = (self._merge_env(env) if merge_persistent else env) or {}
        merged = {
            **self._host_ca_certificates_env(),
            **configured,
            **self._proxy_env,
        }
        for key, value in merged.items():
            if not key or "=" in key or "\0" in key or "\0" in value:
                raise ValueError(f"Invalid environment variable: {key!r}")
            runtime_env[f"{prefix}{key}"] = value
        return runtime_env

    @contextmanager
    def _staged_multiline_environment(
        self, command: str, env: dict[str, str] | None
    ) -> Generator[tuple[str, dict[str, str] | None, str | None], None, None]:
        """Keep multiline values out of Singularity's environment injection."""
        multiline = {
            key: value for key, value in (env or {}).items() if "\n" in value
        }
        if not multiline:
            yield command, env, None
            return
        contents = _shell_exports(multiline, context="multiline")
        with self._staged_transfer("exec-env") as staged:
            staged.write_text(contents)
            staged.chmod(0o600)
            yield (
                command,
                {
                    key: value
                    for key, value in (env or {}).items()
                    if key not in multiline
                },
                f"/staging/{staged.name}",
            )

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
        if (
            self._proxy_process is not None
            and self._proxy_process.returncode is not None
        ):
            raise RuntimeError("The network proxy bridge exited unexpectedly")

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
        configured_env = {
            **((self._merge_env(env) if merge_persistent else env) or {}),
            **self._proxy_env,
        }
        with self._staged_multiline_environment(command, configured_env) as (
            command,
            env,
            environment_file,
        ):
            pid_file = f"/tmp/harbor-exec-{secrets.token_hex(8)}.pid"
            shell_command = self._exec_shell_command(
                command,
                cwd,
                user,
                pid_file,
                default_cwd=default_cwd,
                resolve_default_user=merge_persistent,
                environment_file=environment_file,
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
                    env=self._runtime_environment(env, merge_persistent=False),
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
                await self._cleanup_exec_process(
                    process,
                    pid_file,
                    instance_name=instance_name,
                    default_cwd=default_cwd,
                )
                stdout, stderr = await communication
            except asyncio.CancelledError:
                cleanup = asyncio.create_task(
                    self._cleanup_exec_process(
                        process,
                        pid_file,
                        instance_name=instance_name,
                        default_cwd=default_cwd,
                    )
                )
                with suppress(asyncio.CancelledError):
                    await asyncio.shield(cleanup)
                with suppress(Exception, asyncio.CancelledError):
                    await asyncio.shield(communication)
                raise

            stdout_text = stdout.decode(errors="replace")
            stderr_text = stderr.decode(errors="replace")
            return_code = (
                process.returncode if process.returncode is not None else 1
            )
            if timed_out:
                return_code = 124
                timeout_message = (
                    f"Command timed out after {timeout_sec} seconds"
                )
                stderr_text = "\n".join(
                    part
                    for part in (stderr_text.rstrip(), timeout_message)
                    if part
                )
            if return_code != 0:
                error_output = stderr_text or stdout_text or "<no output>"
                self.logger.debug(
                    "Command exited with rc=%s: %s", return_code, error_output
                )
                if not timed_out:
                    self._warn_root_failure((stdout_text, stderr_text))
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

    @contextmanager
    def _staged_transfer(self, name: str) -> Generator[Path, None, None]:
        staged = self._transfer_staging_path(name)
        try:
            self._remove_staging_path(staged)
            yield staged
        finally:
            self._remove_staging_path(staged)

    @staticmethod
    def _check_transfer(result: ExecResult, operation: str) -> None:
        if result.return_code != 0:
            error = result.stderr or result.stdout or "<no output>"
            raise RuntimeError(f"Failed to {operation}: {error}")

    @override
    async def upload_file(
        self, source_path: Path | str, target_path: str
    ) -> None:
        await self._upload(source_path, target_path, directory=False)

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self._upload(source_dir, target_dir, directory=True)

    @staticmethod
    def _stage_upload_directory(source: Path, destination: Path) -> None:
        """Copy a directory while keeping its relative links usable."""
        shutil.copytree(source, destination, symlinks=True)
        source_root = source.resolve()
        transfer_root = source_root.parent
        for source_link in source.rglob("*"):
            if (
                not source_link.is_symlink()
                or source_link.readlink().is_absolute()
            ):
                continue
            try:
                target = source_link.resolve(strict=True)
            except OSError:
                continue
            if target.is_relative_to(source_root) or not target.is_relative_to(
                transfer_root
            ):
                continue
            staged_link = destination / source_link.relative_to(source)
            staged_link.unlink()
            if target.is_dir():
                shutil.copytree(target, staged_link, symlinks=True)
            else:
                shutil.copy2(target, staged_link)

    async def _upload(
        self, source_path: Path | str, target_path: str, *, directory: bool
    ) -> None:
        source = Path(source_path)
        kind = "directory" if directory else "file"
        if not source.exists():
            raise FileNotFoundError(f"Source {kind} not found: {source}")
        with self._staged_transfer(source.name) as staged:
            if directory:
                self._stage_upload_directory(source, staged)
                result = await self.exec(f"mkdir -p {shlex.quote(target_path)}")
                self._check_transfer(result, "prepare upload directory")
                source = f"/staging/{staged.name}/."
                target = f"{target_path}/"
                command = f"cp -rP {shlex.quote(source)} {shlex.quote(target)}"
            else:
                shutil.copy2(source, staged)
                command = (
                    f"cp {shlex.quote('/staging/' + staged.name)} "
                    f"{shlex.quote(target_path)}"
                )
            self._check_transfer(await self.exec(command), f"upload {kind}")

    @override
    async def download_file(
        self, source_path: str, target_path: Path | str
    ) -> None:
        await self._download(
            source_path, target_path, self.exec, directory=False
        )

    async def _download(
        self,
        source_path: str,
        target_path: Path | str,
        execute: Callable[[str], Awaitable[ExecResult]],
        *,
        directory: bool,
    ) -> None:
        target = Path(target_path)
        kind = "directory" if directory else "file"
        (target.mkdir if directory else target.parent.mkdir)(
            parents=True, exist_ok=True
        )
        with self._staged_transfer(Path(source_path).name) as staged:
            recursive = "-rP " if directory else ""
            command = (
                f"cp {recursive}{shlex.quote(source_path)} "
                f"{shlex.quote('/staging/' + staged.name)}"
            )
            self._check_transfer(await execute(command), f"download {kind}")
            if directory:
                if not staged.is_dir():
                    raise RuntimeError(
                        f"Directory not found in staging: {staged}"
                    )
                for item in staged.iterdir():
                    destination = target / item.name
                    self._remove_staging_path(destination)
                    if item.is_symlink():
                        shutil.copy2(item, destination, follow_symlinks=False)
                    elif item.is_dir():
                        shutil.copytree(item, destination, symlinks=True)
                    else:
                        shutil.copy2(item, destination)
            else:
                if not staged.is_file():
                    raise RuntimeError(f"File not found in staging: {staged}")
                shutil.copy2(staged, target)

    @override
    async def download_dir(
        self, source_dir: str, target_dir: Path | str
    ) -> None:
        await self._download(source_dir, target_dir, self.exec, directory=True)

    def _compose_instance(self, service: str) -> str:
        instances = self._compose_instances.get(service)
        if not instances:
            raise ValueError(f"Unknown Docker Compose service: {service!r}")
        return instances[0]

    @override
    def _compose_service_transport(
        self, service: str | None
    ) -> ComposeServiceTransport:
        del service
        return _SingularityComposeServiceTransport(self)

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
