import asyncio
import gc
import inspect
import io
import logging
import sys
import weakref
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
import yaml

pytest.importorskip("harbor")

from harbor.cli.utils import parse_kwargs  # noqa: E402
from harbor.mappers.terminal_bench import HARBOR_MAIN_SERVICE  # noqa: E402
from harbor.models.task.config import (  # noqa: E402
    EnvironmentConfig,
    NetworkMode,
    NetworkPolicy,
)
from harbor.models.trial.paths import TrialPaths  # noqa: E402

from ursa.integrations import harbor_singularity  # noqa: E402
from ursa.integrations.harbor_singularity import (  # noqa: E402
    DockerfileSingularityEnvironment,
    docker_compose_to_singularity_compose,
)


def _environment(
    tmp_path: Path,
    *,
    compose: bool = False,
    network_mode: NetworkMode = NetworkMode.PUBLIC,
    phase_network_modes: tuple[NetworkMode, ...] = (),
    overlay: bool = False,
    auto_mounting: bool = False,
    fakeroot: bool | str = False,
) -> DockerfileSingularityEnvironment:
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nWORKDIR /workspace\n"
    )
    if compose:
        (environment_dir / "docker-compose.yaml").write_text(
            "services:\n"
            "  main:\n"
            "    image: ubuntu:24.04\n"
            "  helper:\n"
            "    image: ubuntu:24.04\n"
        )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    environment = DockerfileSingularityEnvironment(
        environment_dir=environment_dir,
        environment_name="unit-test",
        session_id=f"{tmp_path.name}__env",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(workdir="/workspace"),
        mounts=[
            {
                "type": "bind",
                "source": str(workspace),
                "target": "/workspace",
                "read_only": False,
            }
        ],
        network_policy=NetworkPolicy(network_mode=network_mode),
        phase_network_policies=[
            NetworkPolicy(network_mode=mode) for mode in phase_network_modes
        ],
        persistent_env={},
        singularity_fakeroot=fakeroot,
        singularity_overlay=overlay,
        auto_mounting=auto_mounting,
    )
    environment._runtime_path = str(tmp_path / "fake-singularity")
    return environment


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def _prepare_started_environment(
    environment: DockerfileSingularityEnvironment, tmp_path: Path
) -> Path:
    environment._prepare_runtime_workdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    environment._scratch_dir = scratch
    staging = scratch / "staging"
    staging.mkdir()
    environment._staging_dir = staging
    overlay = scratch / "overlay.img"
    overlay.touch()
    environment._overlay_path = overlay
    environment._sif_path = tmp_path / "image.sif"
    environment._instance_started = True
    environment._instance_start_attempted = True
    return overlay


def test_host_ca_certificates_pem_uses_ssl(
    monkeypatch: pytest.MonkeyPatch,
):
    class Context:
        @staticmethod
        def get_ca_certs(*, binary_form: bool) -> list[bytes]:
            assert binary_form
            return [b"first", b"second"]

    monkeypatch.setattr(
        harbor_singularity.ssl,
        "create_default_context",
        lambda: Context(),
    )
    monkeypatch.setattr(
        harbor_singularity.ssl,
        "get_default_verify_paths",
        lambda: type("VerifyPaths", (), {"cafile": None, "capath": None})(),
    )
    monkeypatch.setattr(
        harbor_singularity.ssl,
        "DER_cert_to_PEM_cert",
        lambda certificate: f"PEM:{certificate.decode()}\n",
    )

    assert harbor_singularity._host_ca_certificates_pem() == (
        "PEM:first\nPEM:second\n"
    )


def test_host_ca_certificates_pem_loads_hashed_capath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    capath = tmp_path / "certs"
    capath.mkdir()
    certificate = capath / "1234abcd.0"
    certificate.write_bytes(b"certificate")
    (capath / "ignored.pem").write_bytes(b"ignored")

    class Context:
        loaded: list[bytes] = []

        def load_verify_locations(self, *, cafile: str) -> None:
            self.loaded.append(Path(cafile).read_bytes())

        def get_ca_certs(self, *, binary_form: bool) -> list[bytes]:
            assert binary_form
            return self.loaded

    monkeypatch.setattr(
        harbor_singularity.ssl,
        "create_default_context",
        lambda: Context(),
    )
    monkeypatch.setattr(
        harbor_singularity.ssl,
        "get_default_verify_paths",
        lambda: type(
            "VerifyPaths",
            (),
            {"cafile": None, "capath": str(capath)},
        )(),
    )
    monkeypatch.setattr(
        harbor_singularity.ssl,
        "DER_cert_to_PEM_cert",
        lambda value: f"PEM:{value.decode()}\n",
    )

    assert harbor_singularity._host_ca_certificates_pem() == (
        "PEM:certificate\n"
    )


def test_phase_network_change_requires_overlay(tmp_path: Path):
    # Without an overlay the environment reports a static network policy, so
    # Harbor's BaseEnvironment rejects the phase change before startup.
    with pytest.raises(
        ValueError, match="cannot change network policy after start"
    ):
        _environment(
            tmp_path,
            phase_network_modes=(NetworkMode.NO_NETWORK,),
        )


def test_overlay_enables_dynamic_network_policy(tmp_path: Path):
    environment = _environment(
        tmp_path,
        phase_network_modes=(NetworkMode.NO_NETWORK,),
        overlay=True,
    )

    assert environment.capabilities.dynamic_network_policy


def test_compose_sidecars_reject_no_network_phase(tmp_path: Path):
    with pytest.raises(ValueError, match="sidecar networking"):
        _environment(
            tmp_path,
            compose=True,
            phase_network_modes=(NetworkMode.NO_NETWORK,),
            overlay=True,
        )


def test_compose_network_policy_is_rewritten(tmp_path: Path):
    environment = _environment(tmp_path)
    compose_file = tmp_path / "singularity-compose.yml"
    compose_file.write_text(
        "instances:\n"
        "  main:\n"
        "    network:\n"
        "      allocate_ip: true\n"
        "      enable: true\n"
    )
    environment._compose_file = compose_file

    environment._set_compose_network_policy(
        NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
    )
    isolated = yaml.safe_load(compose_file.read_text())["instances"]["main"]
    assert isolated["network"] == {
        "allocate_ip": False,
        "enable": True,
        "type": "none",
    }

    environment._set_compose_network_policy(
        NetworkPolicy(network_mode=NetworkMode.PUBLIC)
    )
    public = yaml.safe_load(compose_file.read_text())["instances"]["main"]
    # A lone main instance uses host networking, as without Compose.
    assert public["network"] == {
        "allocate_ip": True,
        "enable": False,
    }

    compose = yaml.safe_load(compose_file.read_text())
    compose["instances"]["helper"] = {"network": {}}
    compose_file.write_text(yaml.safe_dump(compose))
    environment._set_compose_network_policy(
        NetworkPolicy(network_mode=NetworkMode.PUBLIC)
    )
    instances = yaml.safe_load(compose_file.read_text())["instances"]
    assert all(
        instance["network"] == {"allocate_ip": True, "enable": True}
        for instance in instances.values()
    )


@pytest.mark.parametrize(
    ("instances", "up", "aliases"),
    [
        ({"main": {}}, ["up", "--read_only", "--no-resolv"], False),
        ({"main": {}, "helper": {}}, ["up", "--read_only"], True),
    ],
)
async def test_lone_main_compose_keeps_host_resolver_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    instances: dict[str, dict[str, object]],
    up: list[str],
    aliases: bool,
):
    environment = _environment(tmp_path)
    _prepare_started_environment(environment, tmp_path)
    environment._compose_file = tmp_path / "singularity-compose.yml"
    environment._compose_file.write_text(
        yaml.safe_dump({"instances": instances})
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    calls: list[tuple[str, ...]] = []
    alias_calls: list[bool] = []

    async def run_compose(*arguments: str) -> None:
        calls.append(arguments)

    async def run(*_command: str, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(environment, "_run_compose", run_compose)
    monkeypatch.setattr(environment, "_run", run)
    monkeypatch.setattr(
        environment,
        "_add_compose_service_aliases",
        lambda: alias_calls.append(True),
    )
    monkeypatch.setattr(
        type(environment), "_uses_compose", property(lambda _self: True)
    )

    await environment._start_prepared_runtime(
        NetworkPolicy(network_mode=NetworkMode.PUBLIC)
    )

    assert calls == [tuple(up)]
    assert bool(alias_calls) is aliases
    environment._cleanup_staging()


def test_host_ca_certificates_are_staged_and_exported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    _prepare_started_environment(environment, tmp_path)
    monkeypatch.setattr(
        harbor_singularity,
        "_host_ca_certificates_pem",
        lambda: "test host CA bundle\n",
    )
    staged = environment._prepare_host_ca_certificates()
    environment._persistent_env["SSL_CERT_FILE"] = "/task/ca-bundle.pem"

    runtime_env = environment._runtime_environment(None)

    assert staged is not None
    assert staged.read_text() == "test host CA bundle\n"
    assert staged.stat().st_mode & 0o222 == 0
    assert runtime_env["SINGULARITYENV_SSL_CERT_FILE"] == (
        "/task/ca-bundle.pem"
    )
    assert runtime_env["SINGULARITYENV_REQUESTS_CA_BUNDLE"] == (
        "/staging/host-ca-certificates.crt"
    )
    assert runtime_env["SINGULARITYENV_CURL_CA_BUNDLE"] == (
        "/staging/host-ca-certificates.crt"
    )


@pytest.mark.parametrize(
    ("name", "version", "prefix"),
    [
        ("singularity", "singularity-ce version 3.6.2", "SINGULARITYENV_"),
        ("singularity", "apptainer version 1.3.4", "APPTAINERENV_"),
        ("apptainer", "", "APPTAINERENV_"),
    ],
)
def test_runtime_environment_prefix_follows_the_runtime(
    tmp_path: Path, name: str, version: str, prefix: str
):
    environment = _environment(tmp_path)
    runtime = tmp_path / "bin" / name
    runtime.parent.mkdir()
    runtime.write_text(f"#!/bin/sh\necho '{version}'\n")
    runtime.chmod(0o755)
    environment._runtime_path = str(runtime)

    runtime_env = environment._runtime_environment({"TASK_VALUE": "1"})

    assert runtime_env[f"{prefix}TASK_VALUE"] == "1"
    other = "SINGULARITYENV_" if prefix == "APPTAINERENV_" else "APPTAINERENV_"
    assert f"{other}TASK_VALUE" not in runtime_env


@pytest.mark.parametrize(
    ("name", "version", "fakeroot", "layout"),
    [
        ("singularity", "singularity-ce version 3.6.2", True, True),
        ("singularity", "singularity-ce version 3.6.2", False, True),
        ("singularity", "apptainer version 1.5.5", True, False),
        ("apptainer", "", True, False),
        # Without fakeroot the invoking user needs upper/ and work/ it owns.
        ("apptainer", "", False, True),
    ],
)
def test_disk_overlay_layout_follows_the_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    version: str,
    fakeroot: bool,
    layout: bool,
):
    environment = _environment(tmp_path, overlay=True)
    environment._fakeroot = fakeroot
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    runtime = bin_dir / name
    runtime.write_text(f"#!/bin/sh\necho '{version}'\n")
    runtime.chmod(0o755)
    environment._runtime_path = str(runtime)
    # The fake mkfs records its argv and the layout it was handed, which the
    # environment removes once the image is made.
    record = tmp_path / "mkfs.log"
    mkfs = bin_dir / "mkfs.ext3"
    mkfs.write_text(
        "#!/bin/sh\n"
        f'echo "$@" > {record}\n'
        'if [ "$2" = -d ]; then ls "$3" >> ' + str(record) + "; fi\n"
    )
    mkfs.chmod(0o755)
    monkeypatch.setattr(
        harbor_singularity.shutil,
        "which",
        lambda command: str(mkfs) if command == "mkfs.ext3" else None,
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    environment._scratch_dir = scratch

    environment._prepare_disk_overlay()

    overlay = scratch / "overlay.img"
    assert environment._overlay_path == overlay
    assert overlay.stat().st_size == 1024 * 1024 * 1024
    argv, *listing = record.read_text().splitlines()
    if layout:
        assert argv == f"-q -d {scratch / 'overlay-layout'} {overlay}"
        assert listing == ["upper", "work"]
    else:
        assert argv == f"-q {overlay}"
        assert listing == []
    assert not (scratch / "overlay-layout").exists()


def test_disk_overlay_uses_mke2fs_ext3_without_mkfs_ext3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path, overlay=True)
    environment._fakeroot = True
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    runtime = bin_dir / "apptainer"
    runtime.write_text("#!/bin/sh\n")
    runtime.chmod(0o755)
    environment._runtime_path = str(runtime)
    record = tmp_path / "mkfs.log"
    mke2fs = bin_dir / "mke2fs"
    mke2fs.write_text(f'#!/bin/sh\necho "$@" > {record}\n')
    mke2fs.chmod(0o755)
    monkeypatch.setattr(
        harbor_singularity.shutil,
        "which",
        lambda command: str(mke2fs) if command == "mke2fs" else None,
    )
    environment._scratch_dir = tmp_path

    environment._prepare_disk_overlay()

    assert record.read_text().split() == [
        "-q",
        "-t",
        "ext3",
        str(tmp_path / "overlay.img"),
    ]


async def test_network_policy_change_restarts_with_existing_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(
        tmp_path,
        phase_network_modes=(NetworkMode.NO_NETWORK,),
        overlay=True,
    )
    overlay = _prepare_started_environment(environment, tmp_path)
    commands: list[tuple[str, ...]] = []

    async def stop(*, warn: bool) -> bool:
        assert not warn
        environment._instance_started = False
        environment._instance_start_attempted = False
        return True

    async def run(*command: str, **_kwargs: object) -> None:
        commands.append(command)

    monkeypatch.setattr(environment, "_stop_instance", stop)
    monkeypatch.setattr(environment, "_run", run)

    await environment.set_network_policy(
        NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
    )

    start = next(command for command in commands if "start" in command)
    assert start[start.index("--overlay") + 1] == str(overlay)
    assert start[start.index("--network") + 1] == "none"
    assert environment.network_policy.network_mode == NetworkMode.NO_NETWORK
    assert environment._instance_started


async def test_failed_network_policy_change_restores_previous_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(
        tmp_path,
        phase_network_modes=(NetworkMode.NO_NETWORK,),
        overlay=True,
    )
    _prepare_started_environment(environment, tmp_path)
    starts: list[tuple[str, ...]] = []

    async def stop(*, warn: bool) -> bool:
        environment._instance_started = False
        environment._instance_start_attempted = False
        return True

    async def run(*command: str, **_kwargs: object) -> None:
        if "start" not in command:
            return
        starts.append(command)
        if "--network" in command:
            raise RuntimeError("isolated start failed")

    monkeypatch.setattr(environment, "_stop_instance", stop)
    monkeypatch.setattr(environment, "_run", run)

    with pytest.raises(RuntimeError, match="isolated start failed"):
        await environment.set_network_policy(
            NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
        )

    assert len(starts) == 2
    assert "--network" in starts[0]
    assert "--network" not in starts[1]
    assert environment.network_policy.network_mode == NetworkMode.PUBLIC
    assert environment._instance_started


async def test_run_only_records_failed_build_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    environment = _environment(tmp_path)
    build_log = tmp_path / "cached-image.build.log"

    with pytest.raises(RuntimeError) as raised:
        await environment._run(
            sys.executable,
            "-c",
            "import sys; print('builder stdout'); "
            "print('builder stderr', file=sys.stderr); raise SystemExit(7)",
            failure_log_path=build_log,
        )

    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err
    message = str(raised.value)
    assert f"Build log: {build_log}" in message
    log = build_log.read_text()
    assert "exit code: 7" in log
    assert "builder stdout" in log
    assert "builder stderr" in log
    assert build_log.stat().st_mode & 0o777 == 0o600


async def test_run_discards_successful_build_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    environment = _environment(tmp_path)
    build_log = tmp_path / "cached-image.build.log"

    await environment._run(
        sys.executable,
        "-c",
        "import sys; print('builder stdout'); "
        "print('builder stderr', file=sys.stderr)",
        failure_log_path=build_log,
    )

    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err
    assert not build_log.exists()


async def test_build_surfaces_log_after_all_builders_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    environment = _environment(tmp_path)
    output = tmp_path / "cached-image.sif"

    async def cache_path(*_args: object, **_kwargs: object) -> Path:
        return output

    async def invalid(_path: Path) -> bool:
        return False

    async def fail_build(
        _builder: str,
        _output: Path,
        _temporary: Path,
        _force_build: bool,
        _dockerfile_path: Path,
        _context_dir: Path,
        _build_args: tuple[str, ...],
        _target: str | None,
        build_log_path: Path,
    ) -> None:
        build_log_path.write_text("builder failed\n")
        raise RuntimeError("failed")

    monkeypatch.setattr(environment, "_dockerfile_cache_path", cache_path)
    monkeypatch.setattr(environment, "_is_valid_sif", invalid)
    monkeypatch.setattr(environment, "_build_with", fail_build)
    monkeypatch.setattr(
        harbor_singularity.shutil,
        "which",
        lambda name: "/usr/bin/docker" if name == "docker" else None,
    )

    with pytest.raises(RuntimeError, match="No container builder succeeded"):
        await environment._build_dockerfile_sif(False)

    captured = capsys.readouterr()
    assert not captured.out
    assert "builder failed" in captured.err


async def test_build_failure_names_shared_log_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    output = tmp_path / "cached-image.sif"

    async def cache_path(*_args: object, **_kwargs: object) -> Path:
        return output

    async def invalid(_path: Path) -> bool:
        return False

    async def fail_build(
        builder: str,
        _output: Path,
        _temporary: Path,
        _force_build: bool,
        _dockerfile_path: Path,
        _context_dir: Path,
        _build_args: tuple[str, ...],
        _target: str | None,
        build_log_path: Path,
    ) -> None:
        log_note = environment._record_command_failure(
            build_log_path, (builder, "build"), return_code=1
        )
        raise RuntimeError(f"{Path(builder).name} build failed{log_note}")

    monkeypatch.setattr(environment, "_dockerfile_cache_path", cache_path)
    monkeypatch.setattr(environment, "_is_valid_sif", invalid)
    monkeypatch.setattr(environment, "_build_with", fail_build)
    monkeypatch.setattr(
        harbor_singularity.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name != "buildah" else None,
    )

    with pytest.raises(RuntimeError) as raised:
        await environment._build_dockerfile_sif(False)

    message = str(raised.value)
    build_log = output.with_suffix(".build.log").absolute()
    assert "podman build failed" in message
    assert "docker build failed" in message
    assert message.count(f"Build log: {build_log}") == 1
    assert message.endswith(f"Build log: {build_log}")


async def test_build_discards_failed_attempt_log_after_fallback_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    environment = _environment(tmp_path)
    output = tmp_path / "cached-image.sif"
    attempts = 0

    async def cache_path(*_args: object, **_kwargs: object) -> Path:
        return output

    async def invalid(_path: Path) -> bool:
        return False

    async def build(
        _builder: str,
        output: Path,
        _temporary: Path,
        _force_build: bool,
        _dockerfile_path: Path,
        _context_dir: Path,
        _build_args: tuple[str, ...],
        _target: str | None,
        build_log_path: Path,
    ) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            build_log_path.write_text("first builder failed\n")
            raise RuntimeError("failed")
        output.write_bytes(b"sif")

    monkeypatch.setattr(environment, "_dockerfile_cache_path", cache_path)
    monkeypatch.setattr(environment, "_is_valid_sif", invalid)
    monkeypatch.setattr(environment, "_build_with", build)
    monkeypatch.setattr(
        harbor_singularity.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name != "podman" else None,
    )

    assert await environment._build_dockerfile_sif(False) == output

    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err
    assert not output.with_suffix(".build.log").exists()


async def test_build_with_records_builder_and_apptainer_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    output = tmp_path / "cached-image.sif"
    temporary = tmp_path / "cached-image.tmp.sif"
    build_log = tmp_path / "cached-image.build.log"
    observed: list[tuple[tuple[str, ...], dict[str, object]]] = []

    async def capture(*command: str, **kwargs: object) -> None:
        observed.append((command, kwargs))
        if command[:2] == (environment._runtime_path, "build"):
            Path(command[2]).write_bytes(b"sif")

    monkeypatch.setattr(environment, "_run", capture)

    await environment._build_with(
        "docker",
        output,
        temporary,
        False,
        environment._dockerfile_path,
        environment.environment_dir,
        (),
        None,
        build_log,
    )

    builder_build = observed[0]
    assert builder_build[0][:2] == ("docker", "build")
    assert builder_build[1] == {"failure_log_path": build_log}
    apptainer_build = next(
        item
        for item in observed
        if item[0][:2] == (environment._runtime_path, "build")
    )
    assert apptainer_build[1] == {"failure_log_path": build_log}
    save = next(item for item in observed if item[0][:2] == ("docker", "save"))
    assert save[1] == {"failure_log_path": build_log}
    assert output.read_bytes() == b"sif"


def _fake_apptainer(
    environment: DockerfileSingularityEnvironment,
    tmp_path: Path,
    *,
    build: str = 'echo sif > "$output"',
    version: str = "apptainer version 1.5.5",
    name: str = "apptainer",
) -> Path:
    """Install a fake runtime that records its calls in ``calls``."""
    runtime = tmp_path / "bin" / name
    runtime.parent.mkdir(exist_ok=True)
    calls = tmp_path / "calls"
    runtime.write_text(
        "#!/bin/sh\n"
        f'printf \'%s|%s|%s\\n\' "$PWD" "${{APPTAINER_BINDPATH-}}" "$*" '
        f">> {calls}\n"
        "case $1 in\n"
        f"--version) echo '{version}' ;;\n"
        'sif) [ -s "$3" ] ;;\n'
        "build)\n"
        '    eval "output=\\${$(($# - 1))}"\n'
        '    eval "definition=\\${$#}"\n'
        f'    cp "$definition" {tmp_path}/last.def\n'
        f"    {build}\n"
        "    ;;\n"
        "esac\n"
    )
    runtime.chmod(0o755)
    environment._runtime_path = str(runtime)
    environment._image_cache_dir = tmp_path / "cache"
    return calls


def _builds(calls: Path) -> list[list[str]]:
    if not calls.is_file():
        return []
    return [
        line.split("|")
        for line in calls.read_text().splitlines()
        if line.split("|")[2].startswith("build ")
    ]


def _oci_builders(
    monkeypatch: pytest.MonkeyPatch, *names: str
) -> tuple[list[tuple[str, Path]], Callable[..., Awaitable[None]]]:
    """Make ``names`` the installed OCI builders.

    Returns the builds recorded by the returned ``_build_with`` stand-in.
    """
    built: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        harbor_singularity.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in names else None,
    )

    async def build(builder: str, output: Path, *_args: object) -> None:
        built.append((builder, output))
        output.write_text("sif")

    return built, build


async def test_auto_builds_dockerfile_from_apptainer_definition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    calls = _fake_apptainer(environment, tmp_path)
    built, build = _oci_builders(monkeypatch, "docker")
    monkeypatch.setattr(environment, "_build_with", build)
    monkeypatch.setenv("APPTAINER_BINDPATH", "/host:/host")

    output = await environment._build_dockerfile_sif(False)

    assert not built
    [(cwd, bindpath, command)] = _builds(calls)
    assert cwd == str(environment.environment_dir)
    # The caller's binds would otherwise apply to %post.
    assert bindpath == ""
    arguments = command.split()
    assert arguments[:2] == ["build", "--fakeroot"]
    assert arguments[2] == str(output.with_suffix(".tmp.sif"))
    assert arguments[3].endswith("/build.def")
    assert (
        (tmp_path / "last.def")
        .read_text()
        .startswith("Bootstrap: docker\nFrom: ubuntu:24.04\n")
    )
    assert output.read_text() == "sif\n"
    assert output != await environment._dockerfile_cache_path()
    assert not output.with_suffix(".build.log").exists()
    assert not output.with_suffix(".tmp.sif").exists()

    assert await environment._build_dockerfile_sif(False) == output
    assert len(_builds(calls)) == 1


async def test_auto_reuses_cached_oci_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    calls = _fake_apptainer(environment, tmp_path)
    built, build = _oci_builders(monkeypatch, "buildah")
    monkeypatch.setattr(environment, "_build_with", build)
    cached = await environment._dockerfile_cache_path()
    cached.parent.mkdir(parents=True)
    cached.write_text("sif")

    assert await environment._build_dockerfile_sif(False) == cached
    assert not built
    assert not _builds(calls)


async def test_auto_builds_unsupported_dockerfile_with_oci_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    (environment.environment_dir / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nRUN --mount=type=secret,id=token true\n"
    )
    calls = _fake_apptainer(environment, tmp_path)
    built, build = _oci_builders(monkeypatch, "podman")
    monkeypatch.setattr(environment, "_build_with", build)

    output = await environment._build_dockerfile_sif(False)

    assert built == [("/usr/bin/podman", output)]
    assert output == await environment._dockerfile_cache_path()
    assert not _builds(calls)


async def test_explicit_apptainer_builder_names_unsupported_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    environment._builder_setting = "apptainer"
    (environment.environment_dir / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nRUN --mount=type=secret,id=token true\n"
    )
    calls = _fake_apptainer(environment, tmp_path)
    _oci_builders(monkeypatch, "docker")

    with pytest.raises(RuntimeError, match="line 2: RUN --mount=type=secret"):
        await environment._build_dockerfile_sif(False)
    assert not _builds(calls)


async def test_apptainer_host_failure_is_skipped_for_later_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        DockerfileSingularityEnvironment, "_unusable_builders", {}
    )
    environment = _environment(tmp_path)
    calls = _fake_apptainer(
        environment,
        tmp_path,
        build="echo 'FATAL:   could not use fakeroot: no user namespace'; "
        "exit 255",
    )
    built, build = _oci_builders(monkeypatch, "docker")
    monkeypatch.setattr(environment, "_build_with", build)

    output = await environment._build_dockerfile_sif(False)

    assert built == [("/usr/bin/docker", output)]
    assert environment._unusable_builders == {
        environment._runtime_path: (
            "FATAL:   could not use fakeroot: no user namespace"
        )
    }
    # The Apptainer log stays for diagnosis; the OCI build left none.
    [log] = output.parent.glob("*.build.log")
    assert "could not use fakeroot" in log.read_text()
    assert log.stem != output.stem
    assert environment._dockerfile_builders() == ["/usr/bin/docker"]
    await environment._build_dockerfile_sif(True)
    assert len(_builds(calls)) == 1


async def test_apptainer_step_failure_is_not_a_host_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        DockerfileSingularityEnvironment, "_unusable_builders", {}
    )
    environment = _environment(tmp_path)
    _fake_apptainer(
        environment,
        tmp_path,
        build="echo 'E: Unable to locate package nope'; echo 'FATAL:   "
        "While performing build: while running %post section: exit status "
        "100'; exit 255",
    )
    _oci_builders(monkeypatch)

    with pytest.raises(RuntimeError) as raised:
        await environment._build_dockerfile_sif(False)

    message = str(raised.value)
    assert message.startswith("No container builder succeeded: ")
    log = next((tmp_path / "cache").glob("*.build.log"))
    assert message.endswith(f"Build log: {log.absolute()}")
    assert "Unable to locate package" in log.read_text()
    assert not environment._unusable_builders


async def test_fakeroot_command_failure_retries_without_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    calls = _fake_apptainer(
        environment,
        tmp_path,
        build='case "$*" in *--ignore-fakeroot-command*) '
        'echo sif > "$output" ;; *) echo "fakeroot: error while starting the '
        "\\`faked' daemon.\"; exit 1 ;; esac",
    )
    _oci_builders(monkeypatch)

    output = await environment._build_dockerfile_sif(False)

    assert output.read_text() == "sif\n"
    builds = [command.split()[:3] for _, _, command in _builds(calls)]
    assert builds == [
        ["build", "--fakeroot", str(output.with_suffix(".tmp.sif"))],
        ["build", "--fakeroot", "--ignore-fakeroot-command"],
    ]


async def test_oci_run_failure_is_skipped_for_later_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        DockerfileSingularityEnvironment, "_unusable_builders", {}
    )
    environment = _environment(tmp_path)
    _fake_apptainer(
        environment,
        tmp_path,
        name="singularity",
        version="singularity-ce version 3.6.2",
    )
    built: list[str] = []
    monkeypatch.setattr(
        harbor_singularity.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name != "docker" else None,
    )

    async def build(builder: str, output: Path, *args: object) -> None:
        built.append(builder)
        build_log_path = args[-1]
        if builder.endswith("buildah"):
            environment._record_command_failure(
                build_log_path,
                (builder, "build"),
                return_code=1,
                output=io.BytesIO(
                    b"STEP 2/2: RUN true\nerror running container: error "
                    b"setting supplemental groups list: operation not "
                    b"permitted\n"
                ),
            )
            raise RuntimeError("buildah build failed")
        output.write_text("sif")

    monkeypatch.setattr(environment, "_build_with", build)

    # SingularityCE cannot build definitions without fakeroot here, so it
    # keeps the OCI builders.
    assert environment._dockerfile_builders() == [
        "/usr/bin/buildah",
        "/usr/bin/podman",
    ]
    await environment._build_dockerfile_sif(False)
    await environment._build_dockerfile_sif(True)

    assert built == ["/usr/bin/buildah", "/usr/bin/podman", "/usr/bin/podman"]
    assert (
        "supplemental groups"
        in environment._unusable_builders["/usr/bin/buildah"]
    )


@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        ("auto", ["apptainer", "/usr/bin/buildah", "/usr/bin/docker"]),
        ("apptainer", ["apptainer"]),
        ("buildah", ["/usr/bin/buildah"]),
        ("docker", ["/usr/bin/docker"]),
    ],
)
def test_builder_setting_selects_builders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
    expected: list[str],
):
    monkeypatch.setattr(
        DockerfileSingularityEnvironment, "_unusable_builders", {}
    )
    environment = _environment(tmp_path)
    environment._builder_setting = setting
    _fake_apptainer(environment, tmp_path)
    _oci_builders(monkeypatch, "buildah", "docker")

    assert environment._dockerfile_builders() == expected


def test_explicit_builder_must_be_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    _fake_apptainer(
        environment,
        tmp_path,
        name="singularity",
        version="singularity-ce version 3.6.2",
    )
    _oci_builders(monkeypatch)

    environment._builder_setting = "apptainer"
    with pytest.raises(RuntimeError, match="is not Apptainer"):
        environment._dockerfile_builders()
    environment._builder_setting = "podman"
    with pytest.raises(RuntimeError, match="requires podman"):
        environment._dockerfile_builders()
    environment._builder_setting = "auto"
    with pytest.raises(RuntimeError, match="requires Apptainer, buildah"):
        environment._dockerfile_builders()


@pytest.mark.parametrize(
    ("value", "environ", "expected"),
    [
        (None, None, "auto"),
        (None, "buildah", "buildah"),
        (" Apptainer ", "buildah", "apptainer"),
        ("docker", None, "docker"),
    ],
)
def test_builder_setting_accepts_kwarg_or_environment(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
    environ: str | None,
    expected: str,
):
    if environ is None:
        monkeypatch.delenv("URSA_HARBOR_SIF_BUILDER", raising=False)
    else:
        monkeypatch.setenv("URSA_HARBOR_SIF_BUILDER", environ)

    assert DockerfileSingularityEnvironment._parse_builder(value) == expected


def test_builder_setting_rejects_unknown_builders():
    with pytest.raises(ValueError, match="singularity_builder must be one of"):
        DockerfileSingularityEnvironment._parse_builder("kaniko")


async def test_definition_cache_key_includes_translator_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    oci = await environment._dockerfile_cache_path()
    definition = await environment._dockerfile_cache_path(builder="apptainer")

    monkeypatch.setattr(
        harbor_singularity, "DOCKERFILE_TRANSLATOR_VERSION", 999
    )

    assert definition != oci
    assert await environment._dockerfile_cache_path() == oci
    assert (
        await environment._dockerfile_cache_path(builder="apptainer")
        != definition
    )


def test_workdir_uses_compose_build_target_and_args(tmp_path: Path):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text(
        "FROM ubuntu AS base\n"
        "ARG APP=/default\n"
        "WORKDIR $APP\n"
        "FROM base\n"
        "WORKDIR /final\n"
    )
    (environment_dir / "docker-compose.yaml").write_text(
        "services:\n"
        "  main:\n"
        "    build:\n"
        "      context: .\n"
        "      target: base\n"
        "      args:\n"
        "        APP: /from-compose\n"
    )
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()

    environment = DockerfileSingularityEnvironment(
        environment_dir=environment_dir,
        environment_name="unit-test",
        session_id=f"{tmp_path.name}__env",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(),
        network_policy=NetworkPolicy(network_mode=NetworkMode.PUBLIC),
        persistent_env={},
    )

    assert environment._workdir == "/from-compose"


@pytest.mark.parametrize("auto_mounting", [False, True])
def test_log_dirs_bind_host_trial_only_with_auto_mounting(
    tmp_path: Path, auto_mounting: bool
):
    environment = _environment(tmp_path, auto_mounting=auto_mounting)
    environment._prepare_runtime_workdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    environment._staging_dir = staging
    environment._sif_path = tmp_path / "image.sif"
    trial_agent_dir = environment.trial_paths.agent_dir

    start = environment._instance_start_command()
    binds = [start[i + 1] for i, arg in enumerate(start) if arg == "-B"]
    agent_binds = [bind for bind in binds if bind.endswith(":/logs/agent")]

    assert f"{tmp_path / 'workspace'}:/workspace" in binds
    if auto_mounting:
        assert agent_binds == [f"{trial_agent_dir}:/logs/agent"]
    else:
        # Harbor downloads unmounted logs; staging keeps /logs writable.
        assert agent_binds == [
            f"{staging / 'harbor-writable' / 'logs' / 'agent'}:/logs/agent"
        ]
    assert (
        environment.is_path_mounted("/logs/agent", trial_agent_dir)
        is auto_mounting
    )
    environment._cleanup_staging()


@pytest.mark.parametrize(
    ("fakeroot", "overlay"),
    [(False, False), (False, True), (True, True), (True, False)],
)
def test_instance_home_is_private_unless_fakeroot_has_an_overlay(
    tmp_path: Path, fakeroot: bool, overlay: bool
):
    environment = _environment(tmp_path, fakeroot=fakeroot)
    environment._prepare_runtime_workdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    environment._staging_dir = staging
    environment._sif_path = tmp_path / "image.sif"
    if overlay:
        environment._overlay_path = tmp_path / "overlay.img"

    start = environment._instance_start_command()

    home = staging / "harbor-home" / "main"
    if fakeroot and overlay:
        # Fakeroot runs as root with a writable /root in the overlay.
        assert "--no-home" in start
        assert "--home" not in start
        assert not home.exists()
    else:
        assert "--no-home" not in start
        assert _option_value(start, "--home") == f"{home}:/harbor-home"
        assert home.is_dir()
        assert "--containall" in start
    environment._cleanup_staging()


@pytest.mark.parametrize(
    ("fakeroot", "overlay", "env", "expected"),
    [
        (False, False, None, "/harbor-home"),
        (False, False, {"HOME": "/custom"}, "/custom"),
        (True, True, None, None),
        (True, True, {"HOME": "/custom"}, "/custom"),
        (True, False, None, "/harbor-home"),
    ],
)
async def test_exec_exports_home_instead_of_runtime_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fakeroot: bool,
    overlay: bool,
    env: dict[str, str] | None,
    expected: str | None,
):
    environment = _environment(tmp_path, fakeroot=fakeroot)
    _prepare_started_environment(environment, tmp_path)
    if not overlay:
        environment._overlay_path = None
    observed: dict[str, object] = {}

    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def spawn(*command: str, **kwargs: object) -> Process:
        observed["command"] = command
        observed["env"] = kwargs["env"]
        return Process()

    monkeypatch.setattr(
        harbor_singularity.asyncio, "create_subprocess_exec", spawn
    )

    await environment.exec("true", env=env, user="root")

    shell = observed["command"][-1]
    runtime_env = observed["env"]
    assert isinstance(shell, str) and isinstance(runtime_env, dict)
    assert "SINGULARITYENV_HOME" not in runtime_env
    assert "APPTAINERENV_HOME" not in runtime_env
    if expected is None:
        assert "export HOME" not in shell
    else:
        assert f"export HOME={expected}; cd " in shell
    environment._cleanup_staging()


async def test_compose_home_option_replaces_no_home_without_fakeroot(
    tmp_path: Path,
):
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text("services:\n  main: {}\n  helper:\n    image: x\n")
    staging = tmp_path / "staging"
    staging.mkdir()
    output = tmp_path / "singularity-compose.yml"

    async def resolve(_name: str, _service: dict[str, object]) -> str:
        return "image.sif"

    for fakeroot in (True, False):
        await docker_compose_to_singularity_compose(
            compose,
            output,
            identity="unit",
            image_resolver=resolve,
            staging_dir=staging,
            fakeroot=fakeroot,
            home_root=None if fakeroot else staging / "harbor-home",
        )
        instances = yaml.safe_load(output.read_text())["instances"]
        for key, instance in instances.items():
            options = instance["start"]["options"]
            homes = [option for option in options if option.startswith("home=")]
            if fakeroot:
                assert "no-home" in options
                assert not homes
            else:
                assert "no-home" not in options
                home = staging / "harbor-home" / key
                assert homes == [f"home={home}:/harbor-home"]
                assert home.is_dir()


async def test_direct_runtime_commands_use_isolated_host_workdirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    environment._prepare_runtime_workdir()
    root = environment._runtime_workdir_root
    assert root is not None
    staging = tmp_path / "staging"
    staging.mkdir()
    environment._staging_dir = staging
    environment._sif_path = tmp_path / "image.sif"

    start = environment._instance_start_command()
    observed: list[str] = []

    async def capture(*command: str, **_kwargs: object) -> None:
        observed.extend(command)

    monkeypatch.setattr(environment, "_run", capture)
    environment._prepare_harbor_bind_sources()
    await environment._seed_harbor_bind_sources()

    main_workdir = Path(_option_value(start, "--workdir"))
    seed_workdir = Path(_option_value(observed, "--workdir"))
    assert main_workdir == root / "main"
    assert seed_workdir == root / "seed"
    assert main_workdir.is_dir()
    assert seed_workdir.is_dir()
    assert main_workdir != seed_workdir

    environment._cleanup_staging()
    assert not root.exists()


FAKEROOT_FATAL = (
    "FATAL:   could not use fakeroot: no mapping entry found in "
    "/etc/subuid for alice"
)


def _write_fake_runtime(
    environment: DockerfileSingularityEnvironment, tmp_path: Path, script: str
) -> None:
    runtime = tmp_path / "fake-singularity"
    runtime.write_text(f"#!/bin/sh\n{script}\n")
    runtime.chmod(0o755)
    environment._runtime_path = str(runtime)
    environment._prepare_runtime_workdir()
    environment._sif_path = tmp_path / "image.sif"


def test_fakeroot_is_required_by_default():
    signature = inspect.signature(DockerfileSingularityEnvironment.__init__)

    assert signature.parameters["singularity_fakeroot"].default is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        (" True ", True),
        ("false", False),
        ("1", True),
        ("no", False),
        (1, True),
        (0, False),
    ],
)
def test_fakeroot_setting_accepts_cli_values(
    value: bool | int | str, expected: bool
):
    assert DockerfileSingularityEnvironment._parse_fakeroot(value) is expected


def test_fakeroot_setting_accepts_harbor_cli_ints():
    parsed = parse_kwargs(["singularity_fakeroot=1", "other=0"])

    assert DockerfileSingularityEnvironment._parse_fakeroot(
        parsed["singularity_fakeroot"]
    )
    assert not DockerfileSingularityEnvironment._parse_fakeroot(parsed["other"])


@pytest.mark.parametrize("value", ["auto", "sometimes", 2, -1, None])
def test_fakeroot_setting_rejects_unknown_values(value: object):
    with pytest.raises(ValueError, match="singularity_fakeroot"):
        DockerfileSingularityEnvironment._parse_fakeroot(value)


async def test_fakeroot_reports_cause_when_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        DockerfileSingularityEnvironment, "_fakeroot_unavailable_reasons", {}
    )
    environment = _environment(tmp_path, fakeroot=True)
    _write_fake_runtime(
        environment,
        tmp_path,
        f'echo "{FAKEROOT_FATAL}" >&2; exit 255',
    )

    with pytest.raises(RuntimeError) as raised:
        await environment._resolve_fakeroot()

    message = str(raised.value)
    assert "singularity_fakeroot=true" in message
    assert FAKEROOT_FATAL in message
    assert "singularity_fakeroot=false" in message
    assert "auto" not in message
    assert "Command failed" not in message
    assert not environment._fakeroot
    assert environment._fakeroot_unavailable_reasons == {
        environment._instance_runtime(): FAKEROOT_FATAL
    }
    environment._cleanup_runtime_workdir()


async def test_fakeroot_reraises_unrelated_probe_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    monkeypatch.setattr(
        DockerfileSingularityEnvironment, "_fakeroot_unavailable_reasons", {}
    )
    environment = _environment(tmp_path, fakeroot=True)
    _write_fake_runtime(
        environment,
        tmp_path,
        'echo "FATAL:   Unable to handle docker://example/image: '
        'registry unavailable" >&2; exit 255',
    )

    with (
        caplog.at_level("WARNING"),
        pytest.raises(RuntimeError, match="registry unavailable") as raised,
    ):
        await environment._resolve_fakeroot()

    assert "singularity_fakeroot" not in str(raised.value)
    assert not any("fakeroot" in r.getMessage() for r in caplog.records)
    assert not environment._fakeroot_unavailable_reasons
    environment._cleanup_runtime_workdir()


async def test_fakeroot_failure_is_cached_per_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        DockerfileSingularityEnvironment, "_fakeroot_unavailable_reasons", {}
    )
    probes: list[str] = []

    async def probe() -> str:
        probes.append("probe")
        return FAKEROOT_FATAL

    for name in ("agent", "verifier"):
        (tmp_path / name).mkdir()
    environments = [
        _environment(tmp_path / name, fakeroot=True)
        for name in ("agent", "verifier")
    ]
    for environment in environments:
        monkeypatch.setattr(environment, "_runtime_path", "/fake/singularity")
        monkeypatch.setattr(environment, "_fakeroot_probe_error", probe)
        with pytest.raises(RuntimeError, match="no mapping entry"):
            await environment._resolve_fakeroot()

    assert probes == ["probe"]
    assert all(not environment._fakeroot for environment in environments)


async def test_start_fails_before_building_when_fakeroot_is_known_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path, fakeroot=True)
    monkeypatch.setattr(
        DockerfileSingularityEnvironment,
        "_fakeroot_unavailable_reasons",
        {environment._instance_runtime(): FAKEROOT_FATAL},
    )

    async def unexpected(*_args: object, **_kwargs: object) -> Path:
        pytest.fail("the image must not be built without usable fakeroot")

    monkeypatch.setattr(environment, "_build_main_sif", unexpected)

    with pytest.raises(RuntimeError, match="no mapping entry"):
        await environment.start(force_build=False)

    assert environment._runtime_workdir_root is None


async def test_fakeroot_is_enabled_when_probe_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        DockerfileSingularityEnvironment, "_fakeroot_unavailable_reasons", {}
    )
    environment = _environment(tmp_path, fakeroot=True)
    environment._prepare_runtime_workdir()
    environment._sif_path = tmp_path / "image.sif"
    probes: list[tuple[str, ...]] = []

    async def run(*command: str, **_kwargs: object) -> None:
        probes.append(command)

    monkeypatch.setattr(environment, "_run", run)

    await environment._resolve_fakeroot()

    assert environment._fakeroot
    assert len(probes) == 1
    assert "--fakeroot" in probes[0]
    assert probes[0][-2:] == (str(tmp_path / "image.sif"), "true")
    environment._cleanup_runtime_workdir()


async def test_disabled_fakeroot_skips_probe_and_warns_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    monkeypatch.setattr(
        DockerfileSingularityEnvironment, "_fakeroot_disabled_warned", False
    )

    async def unexpected(*_command: str, **_kwargs: object) -> None:
        pytest.fail("fakeroot probe should not run when disabled")

    for name in ("agent", "verifier"):
        (tmp_path / name).mkdir()
    environments = [
        _environment(tmp_path / name, fakeroot="false")
        for name in ("agent", "verifier")
    ]
    with caplog.at_level("DEBUG"):
        for environment in environments:
            monkeypatch.setattr(environment, "_run", unexpected)
            await environment._resolve_fakeroot()
            environment._exec_shell_command("true", None, "root", "/tmp/pid")

    assert all(not environment._fakeroot for environment in environments)
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING"
    ]
    assert len(warnings) == 1
    assert "singularity_fakeroot=false" in warnings[0]
    assert "read-only" in warnings[0]


async def test_start_resolves_fakeroot_before_seed_and_compose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        DockerfileSingularityEnvironment, "_fakeroot_unavailable_reasons", {}
    )
    environment = _environment(tmp_path, compose=True, fakeroot=True)
    environment._fakeroot = False
    observed: list[object] = []

    async def build(_force_build: bool) -> Path:
        return tmp_path / "image.sif"

    async def mark(_path: Path) -> None:
        pass

    async def probe() -> None:
        observed.append("probe")

    async def seed() -> None:
        observed.append(("seed", environment._fakeroot))

    async def convert(*_args: object, **kwargs: object) -> dict[str, list[str]]:
        observed.append(("compose", kwargs["fakeroot"]))
        return {"main": ["main1"], "helper": ["helper1"]}

    async def start_runtime(_network_policy: NetworkPolicy) -> None:
        raise RuntimeError("stop after compose preparation")

    monkeypatch.setattr(environment, "_build_main_sif", build)
    monkeypatch.setattr(environment, "_mark_sif_in_use", mark)
    monkeypatch.setattr(environment, "_fakeroot_probe_error", probe)
    monkeypatch.setattr(environment, "_seed_harbor_bind_sources", seed)
    monkeypatch.setattr(
        harbor_singularity, "docker_compose_to_singularity_compose", convert
    )
    monkeypatch.setattr(environment, "_start_prepared_runtime", start_runtime)

    with pytest.raises(RuntimeError, match="stop after compose preparation"):
        await environment.start(force_build=False)

    assert observed == ["probe", ("seed", True), ("compose", True)]


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        ("plain", "plain"),
        ("$$HOME and $${SET}", "$HOME and ${SET}"),
        ("$SET/x", "value/x"),
        ("${SET}", "value"),
        ("a${SET}b", "avalueb"),
        ("${UNSET:-fallback}", "fallback"),
        ("${EMPTY:-fallback}", "fallback"),
        ("${UNSET-fallback}", "fallback"),
        ("${EMPTY-fallback}", ""),
        ("${SET:-fallback}", "value"),
        ("${SET:?unused}", "value"),
        ("${EMPTY?unused}", ""),
        ("${SET:+alt}", "alt"),
        ("${EMPTY:+alt}", ""),
        ("${EMPTY+alt}", "alt"),
        ("${UNSET+alt}", ""),
        ("${UNSET:-${SET}}", "value"),
        ("${UNSET:-${ALSO_UNSET:-deep}}/x", "deep/x"),
        ("${UNSET:-a:b c}", "a:b c"),
        ("${UNSET:-$$literal}", "$literal"),
        ("${SET:-${MISSING:?not evaluated}}", "value"),
        # Docker Compose substitutes an empty string for a plain reference.
        ("${MISSING}/x", "/x"),
        ("$MISSING", ""),
    ],
)
def test_compose_interpolation(template: str, expected: str):
    environ = {"SET": "value", "EMPTY": ""}

    assert harbor_singularity._interpolate_compose(template, environ) == (
        expected
    )


@pytest.mark.parametrize(
    ("template", "match"),
    [
        ("${MISSING:?set MISSING first}", "'MISSING' is required: set"),
        ("${EMPTY:?must not be empty}", "'EMPTY' is required: must not"),
        ("${MISSING?}", "'MISSING' is required$"),
        ("${UNTERMINATED", "Unterminated"),
        ("${1BAD}", "Invalid"),
        ("${BAD:x}", "Invalid"),
        ("cost $5", "Invalid"),
    ],
)
def test_compose_interpolation_errors(template: str, match: str):
    with pytest.raises(ValueError, match=match):
        harbor_singularity._interpolate_compose(template, {"EMPTY": ""})


def test_unset_compose_variable_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setattr(
        harbor_singularity, "_warned_unset_compose_variables", set()
    )

    with caplog.at_level(logging.WARNING, harbor_singularity.__name__):
        harbor_singularity._interpolate_compose("${MISSING}", {})
        harbor_singularity._interpolate_compose("$MISSING", {})

    assert [record.getMessage() for record in caplog.records] == [
        "Compose variable 'MISSING' in '${MISSING}' is not set; defaulting "
        "to an empty string"
    ]


def test_compose_interpolation_includes_harbor_infra_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Harbor's Terminal-Bench mapper writes this main service.
    monkeypatch.delenv("TEST_DIR", raising=False)
    monkeypatch.setenv("CONTEXT_DIR", "/host/value/is/overridden")
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (environment_dir / "docker-compose.yaml").write_text(
        yaml.safe_dump({"services": {"main": HARBOR_MAIN_SERVICE}})
    )
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    environment = DockerfileSingularityEnvironment(
        environment_dir=environment_dir,
        environment_name="unit-test",
        session_id=f"{tmp_path.name}__env",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(
            docker_image="alexgshaw/x:20251031", cpus=2, memory_mb=4096
        ),
        persistent_env={},
        singularity_image_cache_dir=tmp_path / "cache",
        singularity_fakeroot=False,
    )

    variables = environment._compose_interpolation_environment()
    main = environment._load_compose_config()["services"]["main"]

    assert variables["CONTEXT_DIR"] == str(environment_dir.resolve())
    assert variables["MAIN_IMAGE_NAME"] == (f"hb__{environment.environment_id}")
    assert variables["PREBUILT_IMAGE_NAME"] == "alexgshaw/x:20251031"
    assert variables["CPUS"] == "2"
    assert variables["MEMORY"] == "4096M"
    assert main["build"] == {"context": str(environment_dir.resolve())}
    assert main["environment"] == ["TEST_DIR="]
    # Resource limits are accepted but not enforced.
    assert main["deploy"] == {}


SCICODE_COMPOSE = (
    "services:\n"
    "  main:\n"
    "    volumes:\n"
    '      - "${HARBOR_SCICODE_TEST_DATA_PATH:?'
    "set HARBOR_SCICODE_TEST_DATA_PATH to an absolute path to a local "
    'test_data.h5 before running harbor}:/app/test_data.h5:ro"\n'
)


def test_scicode_compose_requires_test_data_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("HARBOR_SCICODE_TEST_DATA_PATH", raising=False)
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text(SCICODE_COMPOSE)

    with pytest.raises(ValueError) as raised:
        harbor_singularity._load_docker_compose([compose])

    message = str(raised.value)
    assert str(compose) in message
    assert "'HARBOR_SCICODE_TEST_DATA_PATH' is required" in message
    assert message.endswith(
        "set HARBOR_SCICODE_TEST_DATA_PATH to an absolute path to a local "
        "test_data.h5 before running harbor"
    )


def _scicode_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[DockerfileSingularityEnvironment, Path]:
    test_data = tmp_path / "data" / "test_data.h5"
    test_data.parent.mkdir()
    test_data.write_bytes(b"hdf5")
    monkeypatch.setenv("HARBOR_SCICODE_TEST_DATA_PATH", str(test_data))
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    # The Dockerfile and compose file generated by Harbor's SciCode adapter.
    (environment_dir / "Dockerfile").write_text(
        "FROM python:3.11-slim\n"
        "RUN apt-get update && apt-get install -y build-essential\n"
        "WORKDIR /app\n"
    )
    (environment_dir / "docker-compose.yaml").write_text(SCICODE_COMPOSE)
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    environment = DockerfileSingularityEnvironment(
        environment_dir=environment_dir,
        environment_name="scicode",
        session_id=f"{tmp_path.name}__env",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(),
        network_policy=NetworkPolicy(network_mode=NetworkMode.PUBLIC),
        persistent_env={},
        singularity_fakeroot=False,
        singularity_overlay=False,
    )
    runtime = tmp_path / "fake-singularity"
    runtime.write_text("#!/bin/sh\nexit 0\n")
    runtime.chmod(0o755)
    environment._runtime_path = str(runtime)
    environment._prepare_runtime_workdir()
    environment._staging_dir = tmp_path / "staging"
    environment._staging_dir.mkdir()
    environment._sif_path = tmp_path / "image.sif"
    environment._sif_path.write_text("sif")
    return environment, test_data


async def test_scicode_compose_binds_test_data_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment, test_data = _scicode_environment(tmp_path, monkeypatch)
    assert environment._workdir == "/app"

    environment._prepare_harbor_bind_sources()
    await environment._prepare_compose_project(force_build=False)

    assert environment._compose_file is not None
    instances = yaml.safe_load(environment._compose_file.read_text())[
        "instances"
    ]
    (main,) = instances.values()
    workdir_source = tmp_path / "staging" / "harbor-writable" / "workdir"
    assert not any(volume.endswith(":ro") for volume in main["volumes"])
    assert f"{workdir_source}:/app" in main["volumes"]
    assert f"bind={test_data}:/app/test_data.h5:ro" in main["start"]["options"]
    # The writable /app bind needs a mount point for the nested file bind.
    assert (workdir_source / "test_data.h5").is_file()
    assert main["network"]["enable"] is False
    environment._cleanup_compose_project()
    await environment._release_sifs_in_use()
    environment._cleanup_staging()


async def test_compose_rejects_missing_bind_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment, test_data = _scicode_environment(tmp_path, monkeypatch)
    test_data.unlink()

    with pytest.raises(FileNotFoundError, match="test_data.h5"):
        await environment._prepare_compose_project(force_build=False)
    environment._cleanup_compose_project()
    await environment._release_sifs_in_use()
    environment._cleanup_staging()


@pytest.mark.parametrize(
    ("volume", "expected"),
    [
        ("./data:/data", "{base}/data:/data"),
        ("./data:/data:rw", "{base}/data:/data"),
        ("./data:/data:ro", "{base}/data:/data:ro"),
        (
            {"type": "bind", "source": "./data", "target": "/data"},
            "{base}/data:/data",
        ),
        (
            {
                "type": "bind",
                "source": "./data",
                "target": "/data",
                "read_only": True,
            },
            "{base}/data:/data:ro",
        ),
    ],
)
def test_compose_volume_modes_are_normalized(
    tmp_path: Path, volume: object, expected: str
):
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text(
        yaml.safe_dump({"services": {"main": {"volumes": [volume]}}})
    )

    config = harbor_singularity._load_docker_compose([compose])

    assert config["services"]["main"]["volumes"] == [
        expected.format(base=tmp_path.resolve())
    ]


@pytest.mark.parametrize(
    "volume",
    ["./data:/data:z", "./data:/data:ro:z", "named:/data", "./a,b:/data"],
)
def test_compose_rejects_unsupported_volumes(tmp_path: Path, volume: str):
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text(
        yaml.safe_dump({"services": {"main": {"volumes": [volume]}}})
    )

    with pytest.raises(ValueError, match="service 'main'"):
        harbor_singularity._load_docker_compose([compose])


def test_compose_overlay_replaces_volume_by_target(tmp_path: Path):
    base = tmp_path / "docker-compose.yaml"
    base.write_text("services:\n  main:\n    volumes: [/a:/data, /b:/keep]\n")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("services:\n  main:\n    volumes: [/c:/data:ro]\n")

    config = harbor_singularity._load_docker_compose([base, overlay])

    assert config["services"]["main"]["volumes"] == ["/c:/data:ro", "/b:/keep"]


async def test_compose_interpolates_environment_env_file_and_build_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("URSA_TEST_TOKEN", "secret")
    monkeypatch.delenv("URSA_TEST_UNSET", raising=False)
    (tmp_path / "service.env").write_text(
        "FROM_FILE=${URSA_TEST_TOKEN}-file\n"
        "LITERAL='${URSA_TEST_TOKEN}'\n"
        'QUOTED="${URSA_TEST_UNSET:-quoted}"\n'
        "CHAINED=${FROM_FILE}+\n"
    )
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text(
        "services:\n"
        "  main:\n"
        "    build:\n"
        "      context: .\n"
        "      args:\n"
        '        TOKEN: "${URSA_TEST_TOKEN}"\n'
        '        PRICE: "$$5"\n'
        "    env_file: service.env\n"
        "    environment:\n"
        '      COMBINED: "pre-${URSA_TEST_TOKEN}-${URSA_TEST_UNSET:-none}"\n'
        '      ESCAPED: "$${URSA_TEST_TOKEN}"\n'
    )
    (tmp_path / "Dockerfile").write_text("FROM busybox\n")
    config = harbor_singularity._load_docker_compose([compose])
    main = config["services"]["main"]

    environment = harbor_singularity._compose_environment("main", main, {})
    assert environment == {
        "FROM_FILE": "secret-file",
        "LITERAL": "${URSA_TEST_TOKEN}",
        "QUOTED": "quoted",
        "CHAINED": "secret-file+",
        "COMBINED": "pre-secret-none",
        "ESCAPED": "${URSA_TEST_TOKEN}",
    }
    assert main["build"]["args"] == {"TOKEN": "secret", "PRICE": "$5"}


def test_compose_env_file_uses_the_interpolation_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("URSA_TASK_ONLY", raising=False)
    env_file = tmp_path / "service.env"
    env_file.write_text(
        "FROM_TASK=${URSA_TASK_ONLY:?set by the task}\nURSA_TASK_ONLY\n"
    )

    environment = harbor_singularity._compose_environment(
        "main",
        {"env_file": [str(env_file)], "environment": ["URSA_TASK_ONLY"]},
        {},
        {"URSA_TASK_ONLY": "task-value"},
    )

    assert environment == {
        "FROM_TASK": "task-value",
        "URSA_TASK_ONLY": "task-value",
    }


async def test_compose_keeps_read_only_harbor_mounts(tmp_path: Path):
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text("services:\n  main: {}\n")
    staging = tmp_path / "staging"
    staging.mkdir()
    output = tmp_path / "singularity-compose.yml"

    async def resolve(_name: str, _service: dict[str, object]) -> str:
        return "image.sif"

    await docker_compose_to_singularity_compose(
        compose,
        output,
        identity="unit",
        image_resolver=resolve,
        staging_dir=staging,
        fakeroot=False,
        main_mounts=[
            {
                "type": "bind",
                "source": "/host/data",
                "target": "/data",
                "read_only": True,
            },
            {"type": "bind", "source": "/host/app", "target": "/app"},
        ],
    )

    (main,) = yaml.safe_load(output.read_text())["instances"].values()
    assert "bind=/host/data:/data:ro" in main["start"]["options"]
    assert "/host/app:/app" in main["volumes"]
    assert not any(volume.endswith(":ro") for volume in main["volumes"])


async def test_compose_read_only_parent_does_not_hide_writable_child(
    tmp_path: Path,
):
    for name in ("data", "out", "other"):
        (tmp_path / name).mkdir()
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text(
        "services:\n"
        "  main:\n"
        "    volumes:\n"
        f"      - {tmp_path}/out:/data/out\n"
        f"      - {tmp_path}/data:/data:ro\n"
        f"      - {tmp_path}/other:/other\n"
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    output = tmp_path / "singularity-compose.yml"

    async def resolve(_name: str, _service: dict[str, object]) -> str:
        return "image.sif"

    await docker_compose_to_singularity_compose(
        compose,
        output,
        identity="unit",
        image_resolver=resolve,
        staging_dir=staging,
        fakeroot=False,
    )

    (main,) = yaml.safe_load(output.read_text())["instances"].values()
    # singularity-compose emits volumes first, then start options in order.
    assert main["volumes"] == [
        f"{tmp_path}/other:/other",
        f"{staging}:/staging",
    ]
    binds = [
        option
        for option in main["start"]["options"]
        if option.startswith("bind=")
    ]
    assert binds == [
        f"bind={tmp_path}/data:/data:ro",
        f"bind={tmp_path}/out:/data/out",
    ]


async def test_compose_writable_volumes_mount_parents_first(tmp_path: Path):
    (tmp_path / "nested").mkdir()
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text(
        "services:\n  main:\n    volumes: [./nested:/app/nested]\n"
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    output = tmp_path / "singularity-compose.yml"

    async def resolve(_name: str, _service: dict[str, object]) -> str:
        return "image.sif"

    await docker_compose_to_singularity_compose(
        compose,
        output,
        identity="unit",
        image_resolver=resolve,
        staging_dir=staging,
        fakeroot=False,
        main_mounts=[{"type": "bind", "source": "/host/app", "target": "/app"}],
    )

    (main,) = yaml.safe_load(output.read_text())["instances"].values()
    targets = [volume.split(":")[1] for volume in main["volumes"]]
    assert targets.index("/app") < targets.index("/app/nested")


async def test_compose_sidecars_keep_bridge_networking(tmp_path: Path):
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text("services:\n  main: {}\n  helper:\n    image: redis:7\n")
    staging = tmp_path / "staging"
    staging.mkdir()
    output = tmp_path / "singularity-compose.yml"

    async def resolve(_name: str, _service: dict[str, object]) -> str:
        return "image.sif"

    await docker_compose_to_singularity_compose(
        compose,
        output,
        identity="unit",
        image_resolver=resolve,
        staging_dir=staging,
        fakeroot=False,
    )

    instances = yaml.safe_load(output.read_text())["instances"]
    assert all(
        instance["network"]["enable"] is True for instance in instances.values()
    )


def test_non_compose_nested_mount_gets_a_mount_point(tmp_path: Path):
    environment = _environment(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    environment._host_mounts.append({
        "type": "bind",
        "source": str(data),
        "target": "/logs/agent/data",
        "read_only": True,
    })
    staging = tmp_path / "staging"
    staging.mkdir()
    environment._staging_dir = staging

    environment._prepare_harbor_bind_sources()

    assert (staging / "harbor-writable" / "logs" / "agent" / "data").is_dir()


async def test_compose_services_get_distinct_host_workdirs(tmp_path: Path):
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text(
        "services:\n"
        "  main:\n"
        "    image: ubuntu:24.04\n"
        "  helper:\n"
        "    image: redis:7\n"
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    output = tmp_path / "singularity-compose.yml"
    workdirs = {
        "main": (tmp_path / "workdirs" / "main",),
        "helper": (tmp_path / "workdirs" / "helper",),
    }
    for service_workdirs in workdirs.values():
        for workdir in service_workdirs:
            workdir.mkdir(parents=True)

    async def resolve(_name: str, service: dict[str, object]) -> str:
        return str(service["image"])

    await docker_compose_to_singularity_compose(
        compose,
        output,
        identity="unit",
        image_resolver=resolve,
        staging_dir=staging,
        fakeroot=False,
        service_workdirs=workdirs,
    )

    instances = yaml.safe_load(output.read_text())["instances"]
    assert len(instances) == 2
    assert not any(key.endswith("-r1") for key in instances)
    options = {
        key: instance["start"]["options"] for key, instance in instances.items()
    }
    configured = {
        option.removeprefix("workdir=")
        for instance_options in options.values()
        for option in instance_options
        if option.startswith("workdir=")
    }
    assert configured == {
        str(path) for paths in workdirs.values() for path in paths
    }
    assert all("containall" in value for value in options.values())


async def test_compose_conversion_rejects_missing_service_workdir(
    tmp_path: Path,
):
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text(
        "services:\n"
        "  main:\n"
        "    image: ubuntu:24.04\n"
        "  helper:\n"
        "    image: redis:7\n"
    )
    staging = tmp_path / "staging"
    staging.mkdir()

    async def resolve(_name: str, service: dict[str, object]) -> str:
        return str(service["image"])

    with pytest.raises(ValueError, match="workdir.*helper"):
        await docker_compose_to_singularity_compose(
            compose,
            tmp_path / "singularity-compose.yml",
            identity="unit",
            image_resolver=resolve,
            staging_dir=staging,
            fakeroot=False,
            service_workdirs={"main": (tmp_path / "main",)},
        )


async def test_compose_replicas_get_distinct_host_workdirs(tmp_path: Path):
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text(
        "services:\n"
        "  main:\n"
        "    image: ubuntu:24.04\n"
        "    depends_on: [helper]\n"
        "  helper:\n"
        "    image: redis:7\n"
        "    deploy:\n"
        "      replicas: 2\n"
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    workdirs = {
        "main": (tmp_path / "workdirs" / "main",),
        "helper": (
            tmp_path / "workdirs" / "helper-1",
            tmp_path / "workdirs" / "helper-2",
        ),
    }

    async def resolve(_name: str, service: dict[str, object]) -> str:
        return str(service["image"])

    names = await docker_compose_to_singularity_compose(
        compose,
        tmp_path / "singularity-compose.yml",
        identity="unit",
        image_resolver=resolve,
        staging_dir=staging,
        fakeroot=False,
        service_workdirs=workdirs,
    )

    instances = yaml.safe_load(
        (tmp_path / "singularity-compose.yml").read_text()
    )["instances"]
    helper_workdirs = {
        option.removeprefix("workdir=")
        for key, instance in instances.items()
        if "helper" in key
        for option in instance["start"]["options"]
        if option.startswith("workdir=")
    }
    assert helper_workdirs == {str(path) for path in workdirs["helper"]}
    assert len(names["helper"]) == 2
    assert len(set(names["helper"])) == 2
    assert all("deploy" not in instance for instance in instances.values())
    main = next(
        instance for key, instance in instances.items() if "main" in key
    )
    assert len(main["depends_on"]) == 2


async def test_prepare_compose_project_wires_runtime_workdirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path, compose=True)
    environment._prepare_runtime_workdir()
    environment._staging_dir = tmp_path / "staging"
    environment._staging_dir.mkdir()
    monkeypatch.setattr(
        harbor_singularity,
        "_host_ca_certificates_pem",
        lambda: "test host CA bundle\n",
    )
    staged_ca_certificates = environment._prepare_host_ca_certificates()
    observed: dict[str, object] = {}

    async def convert(*_args: object, **kwargs: object) -> dict[str, list[str]]:
        observed.update(kwargs)
        return {"main": ["main1"], "helper": ["helper1"]}

    monkeypatch.setattr(
        harbor_singularity,
        "docker_compose_to_singularity_compose",
        convert,
    )
    await environment._prepare_compose_project(force_build=False)

    workdirs = observed["service_workdirs"]
    assert isinstance(workdirs, dict)
    assert set(workdirs) == {"main", "helper"}
    assert all(
        path.is_dir()
        for service_workdirs in workdirs.values()
        for path in service_workdirs
    )
    assert environment._runtime_workdir_root is not None
    assert all(
        path.is_relative_to(environment._runtime_workdir_root)
        for service_workdirs in workdirs.values()
        for path in service_workdirs
    )
    assert staged_ca_certificates is not None
    assert staged_ca_certificates.read_text() == "test host CA bundle\n"
    assert staged_ca_certificates.stat().st_mode & 0o222 == 0
    assert observed["main_environment"] == {
        "CURL_CA_BUNDLE": "/staging/host-ca-certificates.crt",
        "REQUESTS_CA_BUNDLE": "/staging/host-ca-certificates.crt",
        "SSL_CERT_FILE": "/staging/host-ca-certificates.crt",
    }
    environment._cleanup_compose_project()
    environment._cleanup_staging()


async def test_build_sifs_builds_main_and_unique_compose_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path, compose=True)
    helper = environment.environment_dir / "helper"
    helper.mkdir()
    (helper / "Dockerfile").write_text("FROM busybox:latest\n")
    (environment.environment_dir / "docker-compose.yaml").write_text(
        "services:\n"
        "  main:\n"
        "    build: .\n"
        "  helper:\n"
        "    build: helper\n"
        "  duplicate:\n"
        "    build: helper\n"
        "  external:\n"
        "    image: redis:7\n"
    )
    observed: list[tuple[bool, Path, Path]] = []

    async def build(
        force_build: bool,
        *,
        dockerfile_path: Path | None = None,
        context_dir: Path | None = None,
        **_kwargs: object,
    ) -> Path:
        dockerfile = dockerfile_path or environment._dockerfile_path
        context = context_dir or environment.environment_dir
        observed.append((force_build, dockerfile, context))
        return tmp_path / f"{context.name}.sif"

    async def pull(image: str, force_pull: bool) -> Path:
        observed.append((force_pull, Path(image), Path()))
        return tmp_path / "redis.sif"

    monkeypatch.setattr(environment, "_build_dockerfile_sif", build)
    monkeypatch.setattr(environment, "_pull_image_sif", pull)

    paths = await environment.build_sifs(force_build=True)

    assert paths == [
        tmp_path / "environment.sif",
        tmp_path / "helper.sif",
        tmp_path / "redis.sif",
    ]
    assert observed == [
        (
            True,
            environment.environment_dir / "Dockerfile",
            environment.environment_dir,
        ),
        (True, helper / "Dockerfile", helper),
        (True, Path("redis:7"), Path()),
    ]


def _image_environment(
    tmp_path: Path,
    docker_image: str | None,
    *,
    dockerfile: bool = True,
    compose: str | None = None,
    force_pull: bool = False,
) -> DockerfileSingularityEnvironment:
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    if dockerfile:
        (environment_dir / "Dockerfile").write_text(
            "FROM ubuntu:24.04\nRUN apt-get update\nWORKDIR /app\n"
        )
    if compose is not None:
        (environment_dir / "docker-compose.yaml").write_text(compose)
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    environment = DockerfileSingularityEnvironment(
        environment_dir=environment_dir,
        environment_name="unit-test",
        session_id=f"{tmp_path.name}__env",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(docker_image=docker_image),
        network_policy=NetworkPolicy(network_mode=NetworkMode.PUBLIC),
        persistent_env={},
        singularity_image_cache_dir=tmp_path / "cache",
        singularity_force_pull=force_pull,
        singularity_fakeroot=False,
    )
    return environment


def _install_pull_runtime(
    environment: DockerfileSingularityEnvironment,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str = "singularity",
) -> Path:
    """Fake a runtime that records argv, pulls, and validates SIF files."""
    log = tmp_path / "runtime.log"
    runtime = tmp_path / "bin" / name
    runtime.parent.mkdir(exist_ok=True)
    runtime.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> {log}\n'
        'case "$1" in\n'
        '  pull) printf sif > "$2" ;;\n'
        '  sif) test -s "$3" ;;\n'
        "  *) exit 1 ;;\n"
        "esac\n"
    )
    runtime.chmod(0o755)
    environment._runtime_path = str(runtime)

    def no_builders(command: str) -> str | None:
        if command in {"buildah", "podman", "docker"}:
            pytest.fail(f"a prebuilt image must not look up {command}")
        return None

    monkeypatch.setattr(harbor_singularity.shutil, "which", no_builders)
    return log


def _pulls(log: Path) -> list[list[str]]:
    if not log.is_file():
        return []
    return [
        line.split()
        for line in log.read_text().splitlines()
        if line.startswith("pull ")
    ]


async def test_prebuilt_image_is_pulled_instead_of_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _image_environment(tmp_path, "alexgshaw/x:20251031")
    log = _install_pull_runtime(environment, tmp_path, monkeypatch)

    first = await environment._build_main_sif(force_build=False)
    second = await environment._build_main_sif(force_build=False)

    pulls = _pulls(log)
    assert len(pulls) == 1
    assert pulls[0][0] == "pull"
    assert pulls[0][1] == str(first.with_suffix(".tmp.sif"))
    assert pulls[0][2] == "docker://alexgshaw/x:20251031"
    assert first == second
    assert first.parent == tmp_path / "cache"
    assert first.name.startswith("image-")
    assert first.read_text() == "sif"
    assert not first.with_suffix(".tmp.sif").exists()
    # The Dockerfile still provides the workdir, as for Harbor's Docker
    # backend, but is never built.
    assert environment._workdir == "/app"


async def test_force_build_builds_dockerfile_despite_prebuilt_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _image_environment(tmp_path, "alexgshaw/x:20251031")
    built: list[bool] = []

    async def build(force_build: bool, **_kwargs: object) -> Path:
        built.append(force_build)
        return tmp_path / "built.sif"

    async def unexpected(*_args: object) -> Path:
        pytest.fail("force_build must build the task Dockerfile")

    monkeypatch.setattr(environment, "_build_dockerfile_sif", build)
    monkeypatch.setattr(environment, "_pull_image_sif", unexpected)

    assert await environment._build_main_sif(force_build=True) == (
        tmp_path / "built.sif"
    )
    assert built == [True]


async def test_prebuild_refresh_repulls_prebuilt_image_without_building(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # ``ursa harbor prebuild --force`` must refresh the SIF a normal trial
    # pulls, not build the Dockerfile that every Terminal-Bench task ships.
    environment = _image_environment(tmp_path, "alexgshaw/x:20251031")
    log = _install_pull_runtime(environment, tmp_path, monkeypatch)

    first = await environment.build_sifs()
    second = await environment.build_sifs(refresh=True)

    assert first == second
    assert first[0].name.startswith("image-")
    assert [pull[2] for pull in _pulls(log)] == [
        "docker://alexgshaw/x:20251031",
        "docker://alexgshaw/x:20251031",
    ]


async def test_force_build_without_dockerfile_refreshes_the_pull(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _image_environment(
        tmp_path, "alexgshaw/x:20251031", dockerfile=False
    )
    log = _install_pull_runtime(environment, tmp_path, monkeypatch)

    await environment._build_main_sif(force_build=False)
    await environment._build_main_sif(force_build=True)

    assert len(_pulls(log)) == 2
    assert environment._workdir == "/"


async def test_force_pull_refreshes_prebuilt_image_without_building(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _image_environment(
        tmp_path, "alexgshaw/x:20251031", force_pull=True
    )
    log = _install_pull_runtime(environment, tmp_path, monkeypatch)

    await environment._build_main_sif(force_build=False)
    await environment._build_main_sif(force_build=False)

    assert len(_pulls(log)) == 2


async def test_failed_pull_leaves_no_cached_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _image_environment(tmp_path, "example/missing:1")
    _install_pull_runtime(environment, tmp_path, monkeypatch)
    runtime = Path(environment._instance_runtime())
    runtime.write_text(
        '#!/bin/sh\nprintf partial > "$2"\n'
        'echo "FATAL:   manifest unknown" >&2\nexit 255\n'
    )

    with pytest.raises(RuntimeError, match="manifest unknown"):
        await environment._build_main_sif(force_build=False)

    assert not [
        path for path in (tmp_path / "cache").iterdir() if path.suffix == ".sif"
    ]


async def test_prebuilt_image_pull_uses_apptainer_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _image_environment(tmp_path, "ubuntu", dockerfile=False)
    log = _install_pull_runtime(
        environment, tmp_path, monkeypatch, name="apptainer"
    )

    await environment._build_main_sif(force_build=False)

    assert _pulls(log)[0][2] == "docker://ubuntu:latest"
    assert Path(environment._instance_runtime()).name == "apptainer"


async def test_prebuilt_sif_path_is_used_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image = tmp_path / "images" / "task.sif"
    image.parent.mkdir()
    image.write_text("sif")
    environment = _image_environment(tmp_path, str(image), dockerfile=False)
    log = _install_pull_runtime(environment, tmp_path, monkeypatch)

    assert await environment._build_main_sif(force_build=False) == image
    assert not _pulls(log)

    image.unlink()
    with pytest.raises(FileNotFoundError, match="task.sif"):
        await environment._build_main_sif(force_build=False)


@pytest.mark.parametrize(
    ("image", "uri"),
    [
        ("ubuntu", "docker://ubuntu:latest"),
        ("alexgshaw/x:20251031", "docker://alexgshaw/x:20251031"),
        ("registry:5000/team/img", "docker://registry:5000/team/img:latest"),
        ("registry:5000/team/img:v1", "docker://registry:5000/team/img:v1"),
        ("img@sha256:" + "a" * 64, "docker://img@sha256:" + "a" * 64),
        ("library://alpine:3.20", "library://alpine:3.20"),
        ("docker://ubuntu:24.04", "docker://ubuntu:24.04"),
    ],
)
def test_image_uri_normalization(image: str, uri: str):
    assert DockerfileSingularityEnvironment._image_uri(image) == uri


def test_pulled_image_cache_is_keyed_by_reference(tmp_path: Path):
    environment = _image_environment(tmp_path, "ubuntu:24.04")
    key = environment._pulled_image_cache_path

    assert key("docker://ubuntu:24.04") == key("docker://ubuntu:24.04")
    assert key("docker://ubuntu:24.04") != key("docker://ubuntu:22.04")
    assert key("docker://img@sha256:" + "a" * 64) != key(
        "docker://img@sha256:" + "b" * 64
    )


def test_prebuilt_image_does_not_require_a_dockerfile(tmp_path: Path):
    environment = _image_environment(tmp_path, "ubuntu:24.04", dockerfile=False)

    assert environment._main_image(force_build=False) == "ubuntu:24.04"
    (tmp_path / "other").mkdir()
    with pytest.raises(FileNotFoundError, match="Dockerfile"):
        _image_environment(tmp_path / "other", None, dockerfile=False)


async def test_separate_verifier_image_is_pulled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Harbor starts a separate verifier with its own EnvironmentConfig and
    # force_build=False, so its docker_image is pulled like the agent's.
    verifier = _image_environment(
        tmp_path, "alexgshaw/x-tests:20251031", dockerfile=False
    )
    log = _install_pull_runtime(verifier, tmp_path, monkeypatch)

    paths = await verifier.build_sifs()

    assert len(paths) == 1
    assert _pulls(log)[0][2] == "docker://alexgshaw/x-tests:20251031"


@pytest.mark.parametrize(
    ("main", "expected"),
    [
        ("    volumes: []\n", "docker://alexgshaw/x:20251031"),
        ("    image: ubuntu:24.04\n", "docker://ubuntu:24.04"),
    ],
)
async def test_compose_main_image_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    main: str,
    expected: str,
):
    environment = _image_environment(
        tmp_path,
        "alexgshaw/x:20251031",
        compose=f"services:\n  main:\n{main}",
    )
    log = _install_pull_runtime(environment, tmp_path, monkeypatch)

    await environment._build_main_sif(force_build=False)

    assert [pull[2] for pull in _pulls(log)] == [expected]


@pytest.mark.parametrize(
    ("docker_image", "force_build", "dockerfile", "pulled"),
    [
        # Harbor builds the Dockerfile; main.image only names the result.
        (None, False, True, False),
        ("alexgshaw/x:20251031", True, True, False),
        # Nothing to build, so main.image is the only source.
        (None, False, False, True),
        ("alexgshaw/x:20251031", True, False, True),
    ],
)
async def test_compose_main_image_follows_harbor_build_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    docker_image: str | None,
    force_build: bool,
    dockerfile: bool,
    pulled: bool,
):
    environment = _image_environment(
        tmp_path,
        docker_image,
        dockerfile=dockerfile,
        compose="services:\n  main:\n    image: ubuntu:24.04\n",
    )
    observed: list[str] = []

    async def build(_force_build: bool, **_kwargs: object) -> Path:
        observed.append("build")
        return tmp_path / "built.sif"

    async def pull(image: str, _force_pull: bool) -> Path:
        observed.append(image)
        return tmp_path / "pulled.sif"

    monkeypatch.setattr(environment, "_build_dockerfile_sif", build)
    monkeypatch.setattr(environment, "_pull_image_sif", pull)

    await environment._build_main_sif(force_build=force_build)

    assert observed == (["ubuntu:24.04"] if pulled else ["build"])
    if docker_image is None:
        assert environment._workdir == ("/" if pulled else "/app")


async def test_compose_sidecar_images_resolve_to_cached_sifs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _image_environment(
        tmp_path,
        "alexgshaw/x:20251031",
        compose="services:\n  main: {}\n  helper:\n    image: redis:7\n",
    )
    log = _install_pull_runtime(environment, tmp_path, monkeypatch)
    environment._sif_path = await environment._build_main_sif(False)

    main = await environment._compose_image("main", {}, False)
    helper = await environment._compose_image(
        "helper", {"image": "redis:7"}, False
    )
    paths = await environment.build_sifs()

    assert main == str(environment._sif_path)
    assert Path(helper).parent == tmp_path / "cache"
    assert paths == [environment._sif_path, Path(helper)]
    assert [pull[2] for pull in _pulls(log)] == [
        "docker://alexgshaw/x:20251031",
        "docker://redis:7",
    ]
    assert environment._sif_in_use_path(Path(helper)).is_file()
    await environment._release_sifs_in_use()


async def test_stop_removes_runtime_workdir_and_restart_uses_a_new_one(
    tmp_path: Path,
):
    environment = _environment(tmp_path)
    environment._prepare_runtime_workdir()
    first = environment._runtime_workdir_root
    assert first is not None and first.is_dir()

    await environment.stop(delete=False)
    assert not first.exists()
    assert environment._runtime_workdir_root is None

    environment._prepare_runtime_workdir()
    second = environment._runtime_workdir_root
    assert second is not None and second.is_dir()
    assert second != first
    environment._cleanup_staging()
    assert not second.exists()


async def test_cleanup_removes_runtime_workdir_before_temporary_is_released(
    tmp_path: Path,
):
    environment = _environment(tmp_path)
    environment._prepare_runtime_workdir()
    temporary = environment._runtime_workdir_temp
    root = environment._runtime_workdir_root
    assert temporary is not None
    assert root is not None and root.is_dir()

    environment._cleanup_staging()

    assert not root.exists()
    assert temporary.name == str(root)


async def test_abandoned_environment_does_not_remove_preserved_workdir(
    tmp_path: Path,
):
    environment = _environment(tmp_path)
    environment._prepare_runtime_workdir()
    temporary = environment._runtime_workdir_temp
    root = environment._runtime_workdir_root
    assert temporary is not None
    assert root is not None and root.is_dir()
    temporary_reference = weakref.ref(temporary)
    del temporary
    del environment

    gc.collect()

    assert temporary_reference() is None
    assert root.is_dir()
    root.rmdir()


async def test_concurrent_start_cannot_clean_up_running_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    seed_started = asyncio.Event()
    release_seed = asyncio.Event()

    async def build(_force_build: bool) -> Path:
        return tmp_path / "image.sif"

    async def no_op(*_args: object, **_kwargs: object) -> None:
        pass

    async def seed() -> None:
        seed_started.set()
        await release_seed.wait()

    monkeypatch.setattr(environment, "_build_main_sif", build)
    monkeypatch.setattr(environment, "_mark_sif_in_use", no_op)
    monkeypatch.setattr(environment, "_seed_harbor_bind_sources", seed)
    monkeypatch.setattr(environment, "_run", no_op)
    monkeypatch.setattr(environment, "ensure_dirs", no_op)
    monkeypatch.setattr(
        environment, "_upload_environment_dir_after_start", no_op
    )

    first_start = asyncio.create_task(environment.start(force_build=False))
    await seed_started.wait()
    root = environment._runtime_workdir_root
    assert root is not None and root.is_dir()

    second_start = asyncio.create_task(environment.start(force_build=False))
    await asyncio.sleep(0)
    assert not second_start.done()

    release_seed.set()
    await first_start
    with pytest.raises(RuntimeError, match="already started"):
        await second_start

    assert environment._runtime_workdir_root == root
    assert root.is_dir()
    environment._instance_started = False
    environment._instance_start_attempted = False
    environment._cleanup_staging()


async def test_start_rejects_preserved_failed_runtime_state(tmp_path: Path):
    environment = _environment(tmp_path)
    environment._prepare_runtime_workdir()
    root = environment._runtime_workdir_root
    assert root is not None
    environment._instance_start_attempted = True

    with pytest.raises(RuntimeError, match="requires stop"):
        await environment.start(force_build=False)

    assert environment._runtime_workdir_root == root
    assert root.is_dir()
    environment._instance_start_attempted = False
    environment._cleanup_staging()


async def test_cleanup_completion_survives_repeated_cancellation():
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def cleanup() -> None:
        cleanup_started.set()
        await release_cleanup.wait()
        cleanup_finished.set()

    waiter = asyncio.create_task(
        DockerfileSingularityEnvironment._await_cleanup(cleanup())
    )
    await cleanup_started.wait()
    waiter.cancel()
    await asyncio.sleep(0)
    waiter.cancel()
    await asyncio.sleep(0)
    assert not waiter.done()

    release_cleanup.set()
    await waiter
    assert cleanup_finished.is_set()


async def test_compose_cleanup_accepts_confirmed_absent_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path, compose=True)
    environment._compose_file = tmp_path / "singularity-compose.yml"
    environment._compose_instances = {
        "main": ["main1"],
        "helper": ["helper1", "helper2"],
    }
    environment._instance_started = True
    environment._instance_start_attempted = True

    async def fail_down(*_args: str) -> None:
        raise RuntimeError("compose unavailable")

    async def no_instances() -> set[str]:
        return set()

    async def unexpected_stop(*_args: str, **_kwargs: object) -> None:
        pytest.fail("absent instances must not be stopped")

    monkeypatch.setattr(environment, "_run_compose", fail_down)
    monkeypatch.setattr(environment, "_list_instances", no_instances)
    monkeypatch.setattr(environment, "_run", unexpected_stop)

    assert await environment._stop_compose(warn=True)
    assert await environment._stop_compose(warn=True)
    assert not environment._instance_started
    assert not environment._instance_start_attempted


async def test_compose_cleanup_preserves_privileged_runtime_on_down_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path, compose=True)
    environment._compose_file = tmp_path / "singularity-compose.yml"
    environment._compose_instances = {"main": ["main1"]}
    environment._compose_uses_sudo = True
    environment._instance_started = True
    environment._instance_start_attempted = True

    async def fail_down(*_args: str) -> None:
        raise RuntimeError("compose unavailable")

    async def unexpected_list() -> set[str]:
        pytest.fail("an unprivileged list cannot prove privileged absence")

    monkeypatch.setattr(environment, "_run_compose", fail_down)
    monkeypatch.setattr(environment, "_list_instances", unexpected_list)

    assert not await environment._stop_compose(warn=True)
    assert environment._instance_started
    assert environment._instance_start_attempted


async def test_failed_start_before_instance_attempt_removes_runtime_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    observed: list[Path] = []

    async def build(_force_build: bool) -> Path:
        return tmp_path / "image.sif"

    async def mark(_path: Path) -> None:
        pass

    async def fail_seed() -> None:
        assert environment._runtime_workdir_root is not None
        observed.append(environment._runtime_workdir_root)
        raise RuntimeError("seed failed")

    monkeypatch.setattr(environment, "_build_main_sif", build)
    monkeypatch.setattr(environment, "_mark_sif_in_use", mark)
    monkeypatch.setattr(environment, "_seed_harbor_bind_sources", fail_seed)

    with pytest.raises(RuntimeError, match="seed failed"):
        await environment.start(force_build=False)

    assert len(observed) == 1
    assert not observed[0].exists()
    assert environment._runtime_workdir_root is None


async def test_failed_stop_preserves_runtime_workdir_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    environment._prepare_runtime_workdir()
    root = environment._runtime_workdir_root
    assert root is not None
    environment._instance_start_attempted = True

    async def exists() -> bool:
        return True

    async def fail(*_command: str, **_kwargs: object) -> None:
        raise RuntimeError("stop failed")

    monkeypatch.setattr(environment, "_instance_exists", exists)
    monkeypatch.setattr(environment, "_run", fail)
    await environment.stop(delete=False)

    assert root.is_dir()
    assert environment._runtime_workdir_root == root

    async def succeed(*_command: str, **_kwargs: object) -> None:
        pass

    monkeypatch.setattr(environment, "_run", succeed)
    await environment.stop(delete=False)
    assert not root.exists()


async def test_stop_cleans_failed_start_when_instance_was_never_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    environment._prepare_runtime_workdir()
    root = environment._runtime_workdir_root
    assert root is not None
    environment._instance_start_attempted = True

    async def missing() -> bool:
        return False

    async def unexpected(*_command: str, **_kwargs: object) -> None:
        pytest.fail("stop command should not run for an absent instance")

    monkeypatch.setattr(environment, "_instance_exists", missing)
    monkeypatch.setattr(environment, "_run", unexpected)
    await environment.stop(delete=False)

    assert not root.exists()
    assert not environment._instance_start_attempted


async def test_downloads_after_failed_start_are_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    environment = _environment(tmp_path)

    async def fail_build(_force_build: bool) -> Path:
        raise RuntimeError("build failed")

    async def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("download should not run commands after a failed start")

    monkeypatch.setattr(environment, "_build_main_sif", fail_build)
    with pytest.raises(RuntimeError, match="build failed"):
        await environment.start(force_build=False)
    monkeypatch.setattr(environment, "exec", unexpected)
    monkeypatch.setattr(environment, "service_exec", unexpected)

    with caplog.at_level("DEBUG"):
        await environment.download_dir("/logs/agent", tmp_path / "agent")
        await environment.download_file(
            "/logs/agent/ursa.log", tmp_path / "file" / "ursa.log"
        )
        await environment.download_dir_filtered(
            source_dir="/logs/agent",
            target_dir=tmp_path / "filtered",
            include=["*.log"],
        )
        await environment.service_download_dir(
            "/logs", tmp_path / "helper", service="helper"
        )

    assert not (tmp_path / "agent").exists()
    assert not (tmp_path / "file").exists()
    assert not (tmp_path / "filtered").exists()
    assert not (tmp_path / "helper").exists()
    assert "Skipping download of /logs/agent" in caplog.text
    assert not [
        record for record in caplog.records if record.levelname == "ERROR"
    ]


@pytest.mark.parametrize(
    "output",
    [
        b'{"instances": {}}',
        b'{"instances": "bad"}',
        b'{"instances": [1]}',
        b'{"instances": [{}]}',
        b'{"instances": [{"instance": 1}]}',
    ],
)
def test_instance_list_parser_rejects_malformed_schema(output: bytes):
    with pytest.raises(RuntimeError, match="parse Singularity instance list"):
        DockerfileSingularityEnvironment._parse_instance_list(output)


def test_instance_list_parser_returns_instance_names():
    output = b'{"instances": [{"instance": "first", "pid": 1}]}'
    assert DockerfileSingularityEnvironment._parse_instance_list(output) == {
        "first"
    }
