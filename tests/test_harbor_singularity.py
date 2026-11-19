import asyncio
import hashlib
import io
import json
import logging
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import archspec.cpu
import pytest
import yaml

pytest.importorskip("harbor")

from harbor.cli.utils import parse_kwargs
from harbor.environments.base import ExecResult
from harbor.mappers.terminal_bench import HARBOR_MAIN_SERVICE
from harbor.models.task.config import (
    EnvironmentConfig,
    NetworkMode,
    NetworkPolicy,
)
from harbor.models.trial.paths import TrialPaths

from ursa.integrations.harbor import singularity as harbor_singularity
from ursa.integrations.harbor.singularity import (
    DockerfileSingularityEnvironment,
    docker_compose_to_singularity_compose,
)

_runtime_is_apptainer = harbor_singularity._runtime_is_apptainer


@pytest.fixture(autouse=True)
def _runtime_detected_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests from running ``--version`` on their fake runtime paths."""
    monkeypatch.setattr(
        harbor_singularity,
        "_runtime_is_apptainer",
        lambda runtime: Path(runtime).name == "apptainer",
    )


@pytest.fixture(autouse=True)
def _no_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep definition builds from reading base image configs online."""
    monkeypatch.setattr(
        harbor_singularity, "image_config", lambda image, machine: {}
    )


@pytest.fixture(autouse=True)
def _fresh_builder_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with the default builder and none marked unusable."""
    monkeypatch.delenv("URSA_HARBOR_SIF_BUILDER", raising=False)
    monkeypatch.setattr(
        DockerfileSingularityEnvironment, "_unusable_builders", {}
    )


def _environment(
    tmp_path: Path,
    *,
    compose: bool = False,
    network_mode: NetworkMode = NetworkMode.PUBLIC,
    allowed_hosts: tuple[str, ...] = (),
    phase_network_modes: tuple[NetworkMode, ...] = (),
    overlay: bool = False,
    workdir: str | None = "/workspace",
    fakeroot: bool | str | None = False,
    agent: str | None = None,
    separate_verifier: bool = False,
    auto_mounting: bool = False,
    extra_kwargs: dict[str, object] | None = None,
) -> DockerfileSingularityEnvironment:
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nWORKDIR /workspace\n"
    )
    if compose:
        (environment_dir / "docker-compose.yaml").write_text(
            yaml.safe_dump({
                "services": {
                    "main": {"image": "ubuntu:24.04"},
                    "helper": {"image": "ubuntu:24.04"},
                }
            })
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
        task_env_config=EnvironmentConfig(workdir=workdir),
        mounts=[
            dict(
                type="bind",
                source=str(workspace),
                target="/workspace",
                read_only=False,
            )
        ],
        network_policy=NetworkPolicy(
            network_mode=network_mode, allowed_hosts=list(allowed_hosts)
        ),
        phase_network_policies=[
            NetworkPolicy(network_mode=mode) for mode in phase_network_modes
        ],
        persistent_env={},
        auto_mounts=[
            dict(
                type="bind",
                source=str(trial_paths.verifier_dir),
                target="/logs/verifier",
            )
        ]
        if separate_verifier
        else None,
        auto_mounting=auto_mounting,
        **{
            "singularity_agent": agent,
            "singularity_fakeroot": fakeroot,
            "singularity_overlay": overlay,
            **(extra_kwargs or {}),
        },
    )
    environment._runtime_path = str(tmp_path / "fake-singularity")
    return environment


def test_agent_layer_selection_does_not_apply_to_separate_verifiers(tmp_path):
    agent_path = tmp_path / "agent"
    agent_path.mkdir()
    verifier_path = tmp_path / "verifier"
    verifier_path.mkdir()
    agent = _environment(agent_path, agent="goose", overlay=True)
    verifier = _environment(
        verifier_path, agent="goose", separate_verifier=True
    )
    assert agent._agent_spec == "goose"
    assert verifier._agent_spec is None


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


async def _await(value: object) -> object:
    return value


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


async def _convert_compose(
    tmp_path: Path,
    services: dict[str, object],
    workdirs: dict[str, tuple[Path, ...]],
    fakeroot: bool | None = False,
    ignore_fakeroot_command: bool = False,
) -> tuple[dict[str, list[str]], dict[str, object]]:
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text(yaml.safe_dump({"services": services}))
    staging = tmp_path / "staging"
    staging.mkdir()

    async def resolve(_name: str, service: dict[str, object]) -> str:
        return str(service["image"])

    names = await docker_compose_to_singularity_compose(
        compose,
        tmp_path / "singularity-compose.yml",
        identity="unit",
        image_resolver=resolve,
        staging_dir=staging,
        fakeroot=fakeroot,
        ignore_fakeroot_command=ignore_fakeroot_command,
        service_workdirs=workdirs,
    )
    instances = yaml.safe_load(
        (tmp_path / "singularity-compose.yml").read_text()
    )["instances"]
    return names, instances


@pytest.mark.parametrize("runtime", ["singularity", "apptainer"])
@pytest.mark.parametrize("fakeroot", [None, False, True])
async def test_fakeroot_defaults_and_overrides(
    tmp_path, monkeypatch, runtime, fakeroot
):
    environment = _environment(tmp_path, fakeroot=fakeroot)
    environment._runtime_path = None
    runtime_path = f"/usr/bin/{runtime}"
    runtime_lookup = Mock(return_value=runtime_path)
    monkeypatch.setattr(
        DockerfileSingularityEnvironment,
        "_runtime",
        staticmethod(runtime_lookup),
    )
    expected = runtime == "apptainer" if fakeroot is None else fakeroot
    assert environment._instance_runtime() == runtime_path
    assert environment._fakeroot is expected
    assert environment._instance_runtime() == runtime_path
    runtime_lookup.assert_called_once_with()
    assert (
        "--fakeroot" in environment._isolated_runtime_args(tmp_path)
    ) is expected

    _, instances = await _convert_compose(
        tmp_path,
        {"main": {"image": "ubuntu:24.04"}},
        {"main": (tmp_path / "workdir",)},
        fakeroot=fakeroot,
    )
    assert (
        "fakeroot" in next(iter(instances.values()))["start"]["options"]
    ) is expected


async def test_compose_start_options_can_ignore_the_fakeroot_command(tmp_path):
    _, instances = await _convert_compose(
        tmp_path,
        {"main": {"image": "alpine:3"}},
        {"main": (tmp_path / "workdir",)},
        fakeroot=True,
        ignore_fakeroot_command=True,
    )

    options = next(iter(instances.values()))["start"]["options"]
    assert options[:2] == ["fakeroot", "ignore-fakeroot-command"]


def test_automatic_fakeroot_allows_offline_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(harbor_singularity.shutil, "which", lambda _name: None)
    _environment(tmp_path, fakeroot=None)


@pytest.mark.parametrize("runtime", ["singularity", "apptainer"])
@pytest.mark.parametrize("fakeroot", [None, False, True])
@pytest.mark.parametrize("storage_mb", [None, 64])
def test_overlay_creation_matches_runtime_and_fakeroot(
    tmp_path, monkeypatch, runtime, fakeroot, storage_mb
):
    environment = _environment(tmp_path, fakeroot=fakeroot)
    environment._runtime_path = f"/usr/bin/{runtime}"
    environment._scratch_dir = tmp_path / "scratch"
    environment._scratch_dir.mkdir()
    environment.task_env_config.storage_mb = storage_mb
    overlay = environment._scratch_dir / "overlay.img"
    layout = environment._scratch_dir / "overlay-layout"
    native = runtime == "apptainer" and fakeroot is not False
    formatter = Mock(return_value="/usr/bin/mkfs.ext3")
    monkeypatch.setattr(harbor_singularity.shutil, "which", formatter)

    def create(command, **_kwargs):
        assert command[-1] == str(overlay)
        if native:
            assert command[:3] == ["/usr/bin/apptainer", "overlay", "create"]
            assert {"--fakeroot", "--sparse"} <= set(command)
            assert _option_value(command, "--size") == str(storage_mb or 10240)
            assert not overlay.exists() and not layout.exists()
            overlay.touch()
        else:
            assert command[0] == formatter.return_value
            assert _option_value(command, "-d") == str(layout)
            assert overlay.stat().st_size == (storage_mb or 10240) * 1024**2
            for name in ("upper", "work"):
                assert (layout / name).stat().st_mode & 0o777 == 0o777

    monkeypatch.setattr(harbor_singularity.subprocess, "run", create)
    environment._prepare_disk_overlay()

    assert environment._overlay_path == overlay and overlay.is_file()
    assert not layout.exists()
    if native:
        formatter.assert_not_called()


@pytest.mark.parametrize("runtime", ["singularity", "apptainer"])
@pytest.mark.parametrize("failed_to_start", [False, True])
def test_failed_overlay_creation_removes_partial_files(
    tmp_path, monkeypatch, runtime, failed_to_start
):
    environment = _environment(tmp_path, fakeroot=True)
    environment._runtime_path = f"/usr/bin/{runtime}"
    environment._scratch_dir = tmp_path / "scratch"
    environment._scratch_dir.mkdir()
    monkeypatch.setattr(
        harbor_singularity.shutil, "which", lambda _name: "/usr/bin/mkfs.ext3"
    )

    def fail(command, **_kwargs):
        Path(command[-1]).write_bytes(b"partial overlay")
        if failed_to_start:
            raise OSError("overlay unavailable")
        raise subprocess.CalledProcessError(
            1, command, stderr="overlay unavailable"
        )

    monkeypatch.setattr(harbor_singularity.subprocess, "run", fail)
    with pytest.raises(
        RuntimeError, match="writable overlay: overlay unavailable"
    ):
        environment._prepare_disk_overlay()

    assert environment._overlay_path is None
    assert not list(environment._scratch_dir.iterdir())


def test_compose_main_image_workdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # With a Dockerfile, Harbor builds main and main.image only names the
    # result, so the Dockerfile's WORKDIR applies.
    (tmp_path / "built").mkdir()
    built = _environment(tmp_path / "built", compose=True, workdir=None)
    assert built._workdir == "/workspace"

    # Without one, main.image is pulled and starts in its WorkingDir, as
    # under Docker, or in "/" with a warning when its config is unreadable.
    configs: list[dict[str, str] | None] = [{"WorkingDir": "/srv/app"}]
    lookups: list[tuple[str, str]] = []

    def lookup(image: str, machine: str) -> dict[str, str] | None:
        lookups.append((image, machine))
        return configs[0]

    monkeypatch.setattr(harbor_singularity, "image_config", lookup)
    pulled = tmp_path / "pulled" / "environment"
    pulled.mkdir(parents=True)
    (pulled / "docker-compose.yaml").write_text(
        "services:\n  main:\n    image: ubuntu:24.04\n"
    )
    trial_paths = TrialPaths(tmp_path / "pulled" / "trial")
    trial_paths.mkdir()
    environment = DockerfileSingularityEnvironment(
        environment_dir=pulled,
        environment_name="unit-test",
        session_id="pulled__env",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(),
        network_policy=NetworkPolicy(network_mode=NetworkMode.PUBLIC),
        persistent_env={},
        singularity_fakeroot=False,
    )
    # Construction reads no registry; the start resolves the workdir.
    assert lookups == []
    assert environment._workdir == "/srv/app"
    assert lookups == [("ubuntu:24.04", platform.machine())]

    configs[0] = None
    environment._workdir = None
    assert environment._workdir == "/"
    assert "Cannot read the config of ubuntu:24.04" in caplog.text
    assert "[environment].workdir" in caplog.text


@pytest.mark.parametrize(
    "values",
    [
        {"INHERITED": None, "EMPTY": "", "VALUE": "${INHERITED}", "BOOL": True},
        ["INHERITED", "EMPTY=", "VALUE=old", "VALUE=${INHERITED}", "BOOL=true"],
    ],
)
def test_compose_environment_and_build_arguments(tmp_path, monkeypatch, values):
    monkeypatch.setenv("INHERITED", "host=value")
    env_file = tmp_path / "service.env"
    env_file.write_text("VALUE=file\nFILE_ONLY=preserved\n")
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text(
        yaml.safe_dump({
            "services": {
                "main": {
                    "build": {"context": ".", "args": values},
                    "environment": values,
                    "env_file": [str(env_file)],
                }
            }
        })
    )
    service = harbor_singularity._load_docker_compose([compose])["services"][
        "main"
    ]
    expected = {
        "BOOL": "true",
        "EMPTY": "",
        "INHERITED": "host=value",
        "VALUE": "host=value",
    }
    environment = _environment(tmp_path)
    assert environment._compose_build_inputs(service["build"])[2] == tuple(
        f"{key}={value}" for key, value in expected.items()
    )
    assert harbor_singularity._compose_environment("helper", service, {}) == {
        "FILE_ONLY": "preserved",
        **expected,
    }
    assert harbor_singularity._compose_environment(
        "main", service, {"VALUE": "runtime"}
    ) == {"FILE_ONLY": "preserved", **expected, "VALUE": "runtime"}


@pytest.mark.parametrize("hashed_capath", [False, True])
def test_host_ca_certificates_pem(
    tmp_path: Path, monkeypatch, hashed_capath: bool
):
    capath = tmp_path / "certs" if hashed_capath else None
    if capath:
        capath.mkdir()
        (capath / "1234abcd.0").write_bytes(b"certificate")
        (capath / "ignored.pem").write_bytes(b"ignored")
    loaded: list[bytes] = []

    class Context:
        def load_default_certs(self) -> None:
            pass

        def load_verify_locations(self, *, cafile: str) -> None:
            loaded.append(Path(cafile).read_bytes())

        def get_ca_certs(self, *, binary_form: bool) -> list[bytes]:
            assert binary_form
            return loaded if hashed_capath else [b"first", b"second"]

    monkeypatch.setattr(
        harbor_singularity, "OpenSSLContext", lambda protocol: Context()
    )
    monkeypatch.setattr(
        harbor_singularity.ssl,
        "get_default_verify_paths",
        lambda: type(
            "Paths",
            (),
            {"cafile": None, "capath": str(capath) if capath else None},
        )(),
    )
    monkeypatch.setattr(
        harbor_singularity.ssl,
        "DER_cert_to_PEM_cert",
        lambda value: f"PEM:{value.decode()}\n",
    )
    assert harbor_singularity._host_ca_certificates_pem() == (
        "PEM:certificate\n" if hashed_capath else "PEM:first\nPEM:second\n"
    )


@pytest.mark.parametrize("import_first", [False, True])
def test_host_ca_export_with_truststore(import_first):
    # Isolate the process-wide injection and test both Harbor import orders.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import ssl
import sys
import truststore
if {import_first!r}:
    from ursa.integrations.harbor.singularity import _host_ca_certificates_pem
from ursa.util.http import inject_truststore_into_ssl
inject_truststore_into_ssl()
from ursa.integrations.harbor.singularity import _host_ca_certificates_pem
actual = _host_ca_certificates_pem()
assert actual == sys.stdin.read()
assert ssl.SSLContext is truststore.SSLContext
""",
        ],
        capture_output=True,
        text=True,
        input=harbor_singularity._host_ca_certificates_pem(),
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("compose", "overlay", "error"),
    [
        (False, False, "cannot change network policy after start"),
        (False, True, None),
        (True, True, "sidecar networking"),
    ],
)
def test_network_phase_requirements(
    tmp_path: Path, compose: bool, overlay: bool, error: str | None
):
    kwargs = {
        "compose": compose,
        "overlay": overlay,
        "phase_network_modes": (NetworkMode.NO_NETWORK,),
    }
    if error:
        with pytest.raises(ValueError, match=error):
            _environment(tmp_path, **kwargs)
    else:
        assert _environment(
            tmp_path, **kwargs
        ).capabilities.dynamic_network_policy


@pytest.mark.parametrize(
    ("port", "valid"),
    [
        ({"published": 8080, "target": 80}, True),
        ({"published": True, "target": 80}, False),
        ({"published": 8080, "target": False}, False),
        ({"published": [], "target": 80}, False),
        ({"published": 8080, "target": {}}, False),
        ({"published": -1, "target": 80}, False),
        ({"published": 8080, "target": "http"}, False),
    ],
)
def test_compose_parser_normalizes_null_dependencies_and_validates_ports(
    tmp_path: Path, port: dict[str, object], valid: bool
):
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text(
        yaml.safe_dump({
            "services": {
                "main": {
                    "image": "ubuntu:24.04",
                    "depends_on": None,
                    "ports": [port],
                }
            }
        })
    )

    if not valid:
        with pytest.raises(ValueError, match="TCP published ports"):
            harbor_singularity._load_docker_compose([compose])
        return

    config = harbor_singularity._load_docker_compose([compose])
    assert config["services"]["main"]["depends_on"] == []


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

    for mode, expected in (
        (
            NetworkMode.NO_NETWORK,
            {"allocate_ip": False, "enable": True, "type": "none"},
        ),
        # A lone main instance uses host networking, as without Compose.
        (NetworkMode.PUBLIC, {"allocate_ip": True, "enable": False}),
    ):
        environment._set_compose_network_policy(
            NetworkPolicy(network_mode=mode)
        )
        assert (
            yaml.safe_load(compose_file.read_text())["instances"]["main"][
                "network"
            ]
            == expected
        )
    assert environment._compose_host_network

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
    assert not environment._compose_host_network


@pytest.mark.parametrize("fail_isolated_start", [False, True])
async def test_network_policy_change_restarts_or_restores_previous_policy(
    tmp_path: Path, monkeypatch, fail_isolated_start: bool
):
    environment = _environment(
        tmp_path, phase_network_modes=(NetworkMode.NO_NETWORK,), overlay=True
    )
    overlay = _prepare_started_environment(environment, tmp_path)
    starts: list[tuple[str, ...]] = []

    async def stop(*, warn: bool) -> bool:
        environment._instance_started = (
            environment._instance_start_attempted
        ) = False
        return True

    async def run(*command: str, **_kwargs: object) -> None:
        if "start" in command:
            starts.append(command)
        if fail_isolated_start and "--network" in command:
            raise RuntimeError("isolated start failed")

    monkeypatch.setattr(environment, "_stop_instance", stop)
    monkeypatch.setattr(environment, "_run", run)

    if fail_isolated_start:
        with pytest.raises(RuntimeError, match="isolated start failed"):
            await environment.set_network_policy(
                NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
            )
        assert (
            len(starts) == 2
            and "--network" in starts[0]
            and "--network" not in starts[1]
        )
        assert environment.network_policy.network_mode == NetworkMode.PUBLIC
    else:
        await environment.set_network_policy(
            NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
        )
        start = starts[0]
        assert (
            _option_value(list(start), "--overlay"),
            _option_value(list(start), "--network"),
        ) == (str(overlay), "none")
        assert environment.network_policy.network_mode == NetworkMode.NO_NETWORK
    assert environment._instance_started


@pytest.mark.parametrize("fails", [False, True])
async def test_run_build_log_is_retained_only_on_failure(
    tmp_path: Path, capsys, fails: bool
):
    environment = _environment(tmp_path)
    build_log = tmp_path / "cached-image.build.log"
    program = "import sys; print('builder stdout'); print('builder stderr', file=sys.stderr)"
    if fails:
        program += "; raise SystemExit(7)"

    if fails:
        with pytest.raises(RuntimeError) as raised:
            await environment._run(
                sys.executable, "-c", program, failure_log_path=build_log
            )
        message = str(raised.value)
        assert f"Build log: {build_log}" in message
        log = build_log.read_text()
        assert all(
            value in log
            for value in ("exit code: 7", "builder stdout", "builder stderr")
        )
        assert build_log.stat().st_mode & 0o777 == 0o600
    else:
        await environment._run(
            sys.executable, "-c", program, failure_log_path=build_log
        )
    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err
    assert build_log.exists() is fails


@pytest.mark.parametrize(
    "fallback_succeeds,runtime,error,warning",
    [
        (False, "singularity", "builder failed", False),
        (True, "singularity", "You must be the root user", False),
        (False, "singularity", "You must be the root user", True),
        (False, "apptainer", "You must be the root user", False),
    ],
)
async def test_build_attempt_logs_are_retained_only_after_all_builders_fail(
    tmp_path: Path,
    monkeypatch,
    capsys,
    caplog,
    fallback_succeeds,
    runtime,
    error,
    warning,
):
    environment = _environment(tmp_path)
    environment._runtime_path = f"/usr/bin/{runtime}"
    # Under auto, Apptainer builds a definition first; this test covers the
    # OCI builders' logs.
    environment._builder_setting = (
        "docker" if runtime == "apptainer" else "auto"
    )
    output = tmp_path / "cached-image.sif"

    attempts = 0

    async def build(*args: object) -> None:
        nonlocal attempts
        attempts += 1
        build_log_path = args[-1]
        assert isinstance(build_log_path, Path)
        if not fallback_succeeds or attempts == 1:
            build_log_path.write_text(f"{error}\n")
            raise RuntimeError("failed")
        output.write_bytes(b"sif")

    monkeypatch.setattr(
        environment,
        "_dockerfile_cache_path",
        lambda *_args, **_kwargs: _await(output),
    )
    monkeypatch.setattr(
        environment, "_is_valid_sif", lambda _path: _await(False)
    )
    monkeypatch.setattr(environment, "_build_with", build)
    monkeypatch.setattr(
        harbor_singularity.shutil,
        "which",
        lambda name: (
            f"/usr/bin/{name}"
            if fallback_succeeds and name != "podman"
            else "/usr/bin/docker"
            if name == "docker"
            else None
        ),
    )

    if fallback_succeeds:
        assert await environment._build_dockerfile_sif(False) == output
    else:
        with pytest.raises(
            RuntimeError, match="No container builder succeeded"
        ):
            await environment._build_dockerfile_sif(False)
    captured = capsys.readouterr()
    assert not captured.out
    assert (error in captured.err) is not fallback_succeeds
    assert output.with_suffix(".build.log").exists() is not fallback_succeeds
    assert ("switching to Apptainer" in caplog.text) is warning


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
        f"    rm -rf {tmp_path}/last-staging\n"
        f'    cp -R "${{definition%/*}}" {tmp_path}/last-staging\n'
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


async def test_definition_build_starts_in_base_image_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    (environment.environment_dir / "Dockerfile").write_text(
        "FROM swebench/sweb.eval:latest\nRUN git add -A\n"
    )
    _fake_apptainer(environment, tmp_path)
    _oci_builders(monkeypatch)
    lookups = []

    def config(image: str, machine: str) -> dict[str, str]:
        lookups.append(image)
        return {"WorkingDir": "/testbed", "User": "root"}

    monkeypatch.setattr(harbor_singularity, "image_config", config)

    await environment._build_dockerfile_sif(False)

    assert lookups == ["swebench/sweb.eval:latest"]
    script = (tmp_path / "last-staging" / "s0" / "build.sh").read_text()
    assert script.endswith(
        "_ursa_private_tmp\n"
        "_ursa_workdir /testbed\n"
        "_ursa_user=root\n"
        "_ursa_step 2 'RUN git add -A'\n"
        "_ursa_run /bin/sh -c 'git add -A'\n"
    )


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


async def test_cached_sif_is_reused_without_its_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # A SIF prebuilt on a login node serves a node with no builder at all.
    environment = _environment(tmp_path)
    _fake_apptainer(
        environment,
        tmp_path,
        name="singularity",
        version="singularity-ce version 3.6.2",
    )
    _oci_builders(monkeypatch)
    cached = await environment._dockerfile_cache_path()
    cached.parent.mkdir(parents=True)
    cached.write_text("sif")

    assert await environment._build_dockerfile_sif(False) == cached

    # Apptainer with no OCI builder reuses an OCI build too ...
    calls = _fake_apptainer(environment, tmp_path)
    assert await environment._build_dockerfile_sif(False) == cached
    assert not _builds(calls)

    # ... and an explicit Apptainer builder its own build, even after the
    # runtime failed for a host reason.
    environment._builder_setting = "apptainer"
    monkeypatch.setattr(
        DockerfileSingularityEnvironment,
        "_unusable_builders",
        {environment._runtime_path: "FATAL:   no fakeroot"},
    )
    definition = await environment._dockerfile_cache_path(builder="apptainer")
    definition.write_text("sif")
    assert await environment._build_dockerfile_sif(False) == definition
    assert not _builds(calls)


async def test_build_without_base_config_is_cached_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    (environment.environment_dir / "Dockerfile").write_text(
        "FROM swebench/sweb.eval:latest\nRUN git add -A\n"
    )
    calls = _fake_apptainer(environment, tmp_path)
    _oci_builders(monkeypatch)
    configs: list[dict[str, str] | None] = [None]
    monkeypatch.setattr(
        harbor_singularity, "image_config", lambda image, machine: configs[0]
    )

    degraded = await environment._build_dockerfile_sif(False)

    assert degraded != await environment._dockerfile_cache_path(
        builder="apptainer"
    )
    # The same failure reuses that build ...
    assert await environment._build_dockerfile_sif(False) == degraded
    assert len(_builds(calls)) == 1

    # ... but once the config can be read, the build is redone with it.
    configs[0] = {"WorkingDir": "/testbed"}
    output = await environment._build_dockerfile_sif(False)

    assert output == await environment._dockerfile_cache_path(
        builder="apptainer"
    )
    assert len(_builds(calls)) == 2
    script = (tmp_path / "last-staging" / "s0" / "build.sh").read_text()
    assert "_ursa_workdir /testbed\n" in script


async def test_definition_cache_key_uses_docker_ignore_matching(
    tmp_path: Path,
):
    environment = _environment(tmp_path)
    (environment.environment_dir / ".dockerignore").write_text("*.json\n")
    nested = environment.environment_dir / "src" / "config.json"
    nested.parent.mkdir()
    nested.write_text("1")
    oci = await environment._dockerfile_cache_path()
    definition = await environment._dockerfile_cache_path(builder="apptainer")

    nested.write_text("2")

    # Docker copies src/config.json, so the definition build depends on it.
    assert await environment._dockerfile_cache_path() == oci
    assert (
        await environment._dockerfile_cache_path(builder="apptainer")
        != definition
    )


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


async def test_oci_cache_key_is_unchanged_by_definition_builds(
    tmp_path: Path,
):
    # SIFs that OCI builders cached before the definition builder existed
    # must keep their names, so the OCI key is recomputed here by hand.
    environment = _environment(tmp_path)
    context = environment.environment_dir
    (context / ".dockerignore").write_text("ignored.txt\n")
    (context / "ignored.txt").write_text("skip")
    (context / "kept.txt").write_text("keep")
    digest = hashlib.sha256(
        json.dumps(
            {
                "architecture": platform.machine(),
                "build_args": [],
                "microarchitecture": archspec.cpu.host().name,
                "target": None,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    for path in sorted(context.resolve().rglob("*")):
        relative = path.relative_to(context.resolve()).as_posix()
        if relative == "ignored.txt":
            continue
        digest.update(relative.encode())
        digest.update(str(path.lstat().st_mode).encode())
        digest.update(path.read_bytes())

    assert await environment._dockerfile_cache_path() == (
        environment._image_cache_dir
        / f"dockerfile-{digest.hexdigest()[:24]}.sif"
    )


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


def test_relative_workdir_starts_in_base_image_workdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text("FROM base\nWORKDIR sub\n")
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    configs: list[dict[str, str] | None] = [{"WorkingDir": "/app"}]
    monkeypatch.setattr(
        harbor_singularity, "image_config", lambda image, machine: configs[0]
    )

    def workdir() -> str:
        return DockerfileSingularityEnvironment(
            environment_dir=environment_dir,
            environment_name="unit-test",
            session_id=f"{tmp_path.name}__env",
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(),
            network_policy=NetworkPolicy(network_mode=NetworkMode.PUBLIC),
            persistent_env={},
        )._workdir

    assert workdir() == "/app/sub"
    configs[0] = None
    assert workdir() == "/sub"
    assert "cannot read the config of base" in caplog.text


async def test_marking_a_live_sif_twice_keeps_its_usage_marker(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    environment._image_cache_dir = tmp_path
    image = tmp_path / "image.sif"
    image.touch()

    await environment._mark_sif_in_use(image)
    await environment._mark_sif_in_use(image)

    marker = environment._sif_in_use_path(image)
    assert marker.read_text().splitlines() == [environment._sif_use_token]


async def test_image_pull_cache_reuses_valid_images_and_refreshes_on_request(
    tmp_path, monkeypatch
):
    environment = _environment(tmp_path)
    environment._image_cache_dir = tmp_path / "cache"
    pulls = []

    async def run(*command):
        if command[1] == "sif":
            if Path(command[-1]).read_bytes() != b"valid sif":
                raise RuntimeError("invalid SIF")
        else:
            assert command[1] == "pull"
            pulls.append(command)
            await asyncio.sleep(0.01)
            Path(command[-2]).write_bytes(b"valid sif")

    monkeypatch.setattr(environment, "_run", run)
    first, second = await asyncio.gather(
        environment._pull_image_sif("ubuntu:24.04", False),
        environment._pull_image_sif("docker://ubuntu:24.04", False),
    )
    assert first == second and len(pulls) == 1
    assert pulls[0][-1] == "docker://ubuntu:24.04"
    assert await environment._pull_image_sif("ubuntu:24.04", True) == first
    assert len(pulls) == 2 and "--disable-cache" in pulls[-1]

    first.write_bytes(b"corrupt sif")
    assert await environment._pull_image_sif("ubuntu:24.04", False) == first
    assert len(pulls) == 3 and first.read_bytes() == b"valid sif"
    assert await environment._pull_image_sif("ubuntu:22.04", False) != first
    monkeypatch.setattr(harbor_singularity.platform, "machine", lambda: "other")
    assert await environment._pull_image_sif("ubuntu:24.04", False) != first
    assert len(pulls) == 5
    assert not list(environment._image_cache_dir.glob("*.tmp.sif"))


@pytest.mark.parametrize("failure", ["download", "invalid", "cancelled"])
async def test_failed_image_refresh_preserves_cache(
    tmp_path, monkeypatch, failure
):
    environment = _environment(tmp_path)
    environment._image_cache_dir = tmp_path / "cache"
    refreshing = False

    async def run(*command):
        if command[1] == "sif":
            if Path(command[-1]).read_bytes() != b"valid sif":
                raise RuntimeError("invalid SIF")
            return
        Path(command[-2]).write_bytes(
            b"partial" if refreshing else b"valid sif"
        )
        if refreshing and failure == "download":
            raise RuntimeError("download failed")
        if refreshing and failure == "cancelled":
            raise asyncio.CancelledError

    monkeypatch.setattr(environment, "_run", run)
    cached = await environment._pull_image_sif("ubuntu:24.04", False)
    refreshing = True
    error = asyncio.CancelledError if failure == "cancelled" else RuntimeError
    with pytest.raises(error):
        await environment._pull_image_sif("ubuntu:24.04", True)
    assert cached.read_bytes() == b"valid sif"
    assert not list(environment._image_cache_dir.glob("*.tmp.sif"))
    assert await environment._pull_image_sif("ubuntu:24.04", False) == cached


@pytest.mark.parametrize("scenario", ["simple", "missing", "replicas"])
async def test_compose_conversion_assigns_each_service_a_workdir(
    tmp_path: Path, scenario: str
):
    services: dict[str, object] = {
        "main": {"image": "ubuntu:24.04"},
        "helper": {"image": "redis:7"},
    }
    workdirs = {
        "main": (tmp_path / "workdirs" / "main",),
        "helper": (tmp_path / "workdirs" / "helper",),
    }
    if scenario == "missing":
        with pytest.raises(ValueError, match="workdir.*helper"):
            await _convert_compose(
                tmp_path, services, {"main": workdirs["main"]}
            )
        return
    if scenario == "replicas":
        services["main"] = {"image": "ubuntu:24.04", "depends_on": ["helper"]}
        services["helper"] = {"image": "redis:7", "deploy": {"replicas": 2}}
        workdirs["helper"] = (
            tmp_path / "workdirs" / "helper-1",
            tmp_path / "workdirs" / "helper-2",
        )
    for paths in workdirs.values():
        for path in paths:
            path.mkdir(parents=True)
    names, instances = await _convert_compose(tmp_path, services, workdirs)
    options = [item["start"]["options"] for item in instances.values()]
    configured = {
        option.removeprefix("workdir=")
        for values in options
        for option in values
        if option.startswith("workdir=")
    }
    assert configured == {
        str(path) for paths in workdirs.values() for path in paths
    }
    assert all("containall" in values for values in options)
    if scenario == "simple":
        assert len(instances) == 2
        assert not any(key.endswith("-r1") for key in instances)
        return
    assert len(names["helper"]) == len(set(names["helper"])) == 2
    assert all("deploy" not in instance for instance in instances.values())
    assert (
        len(
            next(v for key, v in instances.items() if "main" in key)[
                "depends_on"
            ]
        )
        == 2
    )


@pytest.mark.parametrize("runtime", ["singularity", "apptainer"])
async def test_prepare_compose_project_wires_runtime_workdirs(
    tmp_path: Path, monkeypatch, runtime
):
    environment = _environment(tmp_path, compose=True, fakeroot=None)
    environment._runtime_path = f"/usr/bin/{runtime}"
    environment._prepare_runtime_workdir()
    environment._staging_dir = tmp_path / "staging"
    environment._staging_dir.mkdir()
    environment._agent_layer_path = tmp_path / "agent-layer.img"
    environment._agent_layer_path.touch()
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
        harbor_singularity, "docker_compose_to_singularity_compose", convert
    )
    await environment._prepare_compose_project(force_build=False)

    assert observed["fakeroot"] is (runtime == "apptainer")
    assert observed["main_agent_layer"] == environment._agent_layer_path
    workdirs = observed["service_workdirs"]
    assert isinstance(workdirs, dict) and set(workdirs) == {"main", "helper"}
    paths = [path for values in workdirs.values() for path in values]
    assert all(path.is_dir() for path in paths)
    assert environment._runtime_workdir_root and all(
        path.is_relative_to(environment._runtime_workdir_root) for path in paths
    )
    assert (
        staged_ca_certificates
        and staged_ca_certificates.read_text() == "test host CA bundle\n"
        and not staged_ca_certificates.stat().st_mode & 0o222
    )
    assert observed["main_environment"] == dict.fromkeys(
        ("CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"),
        "/staging/host-ca-certificates.crt",
    )
    environment._persistent_env["SSL_CERT_FILE"] = "/task/ca-bundle.pem"
    runtime_environment = environment._runtime_environment(None)
    prefix = f"{runtime.upper()}ENV_"
    assert {
        key.removeprefix(prefix): value
        for key, value in runtime_environment.items()
        if key.startswith(prefix)
    } == {
        "SSL_CERT_FILE": "/task/ca-bundle.pem",
        "REQUESTS_CA_BUNDLE": "/staging/host-ca-certificates.crt",
        "CURL_CA_BUNDLE": "/staging/host-ca-certificates.crt",
    }
    environment._cleanup_compose_project()
    environment._cleanup_staging()


async def test_agent_layer_is_read_only_during_seeding_and_below_trial_overlay(
    tmp_path, monkeypatch
):
    environment = _environment(tmp_path)
    _prepare_started_environment(environment, tmp_path)
    environment._agent_layer_path = tmp_path / "agent-layer.img"
    environment._agent_layer_path.touch()
    command = environment._instance_start_command()
    overlays = [
        command[index + 1]
        for index, item in enumerate(command)
        if item == "--overlay"
    ]
    assert overlays == [
        f"{environment._agent_layer_path}:ro",
        str(environment._overlay_path),
    ]
    run = AsyncMock()
    monkeypatch.setattr(environment, "_run", run)
    await environment._seed_harbor_bind_sources()
    seed_command = run.call_args.args
    assert f"{environment._agent_layer_path}:ro" in seed_command
    assert str(environment._overlay_path) not in seed_command
    environment._cleanup_staging()


async def test_start_failure_cleans_runtime_workdir(tmp_path, monkeypatch):
    environment = _environment(tmp_path)
    roots = []

    async def fail_seed():
        roots.append(environment._runtime_workdir_root)
        raise RuntimeError("seed failed")

    monkeypatch.setattr(
        environment, "_build_main_sif", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(environment, "_seed_harbor_bind_sources", fail_seed)
    with pytest.raises(RuntimeError, match="seed failed"):
        await environment.start(force_build=False)

    assert len(roots) == 1 and roots[0] is not None and not roots[0].exists()
    assert environment._runtime_workdir_root is None


async def test_exec_stages_multiline_environment_values(tmp_path, monkeypatch):
    environment = _environment(tmp_path)
    _prepare_started_environment(environment, tmp_path)
    environment._fakeroot = True
    environment._persistent_env["PERSISTENT"] = "persistent\nsecret"
    observed: dict[str, object] = {}

    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"ok", b""

    async def create_process(*command: str, **kwargs: object) -> Process:
        observed["command"] = command
        observed["env"] = kwargs["env"]
        staged = next(environment._staging.glob("*exec-env"))
        observed["contents"] = staged.read_text()
        observed["mode"] = staged.stat().st_mode & 0o777
        return Process()

    monkeypatch.setattr(
        harbor_singularity.asyncio, "create_subprocess_exec", create_process
    )
    result = await environment._exec_instance(
        "printf ready",
        cwd=None,
        env={"PROMPT": "private line\nsecond line", "SAFE": "value"},
        timeout_sec=None,
        user="taskuser",
    )

    assert result.stdout == "ok"
    command = " ".join(observed["command"])
    assert "private line" not in command
    assert ". /staging/" in command
    assert command.index(". /staging/") < command.index("su taskuser")
    assert observed["contents"] == (
        "export PERSISTENT='persistent\nsecret'\n"
        "export PROMPT='private line\nsecond line'\n"
    )
    assert observed["mode"] == 0o600
    runtime_env = observed["env"]
    assert isinstance(runtime_env, dict)
    assert runtime_env["SINGULARITYENV_SAFE"] == "value"
    assert "SINGULARITYENV_PROMPT" not in runtime_env
    assert "SINGULARITYENV_PERSISTENT" not in runtime_env
    assert not list(environment._staging.glob("*exec-env"))


@pytest.mark.parametrize(
    "runtime,return_code,stdout,stderr,warning",
    [
        (
            "singularity",
            1,
            b"",
            b"E: Could not open lock file (are you root?)",
            True,
        ),
        (
            "singularity",
            1,
            b"This command has to be run under the root user",
            b"",
            True,
        ),
        ("singularity", 1, b"", b"This command requires root privileges", True),
        (
            "singularity",
            1,
            b"",
            b"You need to be root to perform this command",
            True,
        ),
        ("singularity", 0, b"", b"requires root privileges", False),
        ("singularity", 1, b"", b"Connection refused", False),
        ("singularity", 1, b"", b"Permission denied", False),
        ("singularity", 1, b"", b"This command must not run as root", False),
        ("apptainer", 1, b"", b"requires root privileges", False),
    ],
)
async def test_failed_root_commands_suggest_apptainer(
    tmp_path, monkeypatch, caplog, runtime, return_code, stdout, stderr, warning
):
    environment = _environment(tmp_path)
    environment._runtime_path = f"/usr/bin/{runtime}"
    _prepare_started_environment(environment, tmp_path)

    class Process:
        returncode = return_code

        async def communicate(self):
            return stdout, stderr

    monkeypatch.setattr(
        harbor_singularity.asyncio,
        "create_subprocess_exec",
        AsyncMock(return_value=Process()),
    )
    result = await environment._exec_instance(
        "build-command", cwd=None, env=None, timeout_sec=None, user=None
    )
    assert result.return_code == return_code
    assert result.stdout == stdout.decode()
    assert result.stderr == stderr.decode()
    assert ("switching to Apptainer" in caplog.text) is warning


async def test_directory_transfers_preserve_symlinks(tmp_path, monkeypatch):
    environment = _environment(tmp_path)
    _prepare_started_environment(environment, tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "target").write_text("data")
    (source / "link").symlink_to("target")
    external = tmp_path / "external"
    external.write_text("fixture")
    (source / "external-link").symlink_to("../external")
    commands: list[str] = []

    async def upload_execute(command: str, **_kwargs: object) -> ExecResult:
        commands.append(command)
        if command.startswith("cp -rP"):
            staged = next(environment._staging.glob("*source"))
            assert (staged / "link").is_symlink()
            assert not (staged / "external-link").is_symlink()
            assert (staged / "external-link").read_text() == "fixture"
        return ExecResult(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(environment, "exec", upload_execute)
    await environment.upload_dir(source, "/workspace")
    assert any(command.startswith("cp -rP") for command in commands)

    async def download_execute(command: str) -> ExecResult:
        staged = environment._staging / Path(shlex.split(command)[-1]).name
        staged.mkdir()
        os.symlink("/container-only/path", staged / "link")
        return ExecResult(stdout="", stderr="", return_code=0)

    destination = tmp_path / "download"
    await environment._download(
        "/workspace", destination, download_execute, directory=True
    )
    assert (destination / "link").is_symlink()
    assert os.readlink(destination / "link") == "/container-only/path"


@pytest.mark.parametrize(
    "scenario", ["instance-retry", "compose-fallback", "compose-privileged"]
)
async def test_stop_failure_cleanup_contracts(tmp_path, monkeypatch, scenario):
    compose = scenario.startswith("compose")
    environment = _environment(tmp_path, compose=compose)
    environment._prepare_runtime_workdir()
    root = environment._runtime_workdir_root
    environment._instance_started = environment._instance_start_attempted = True

    if compose:
        environment._compose_file = tmp_path / "singularity-compose.yml"
        environment._compose_instances = {"main": ["main"]}
        environment._compose_uses_sudo = scenario == "compose-privileged"
        monkeypatch.setattr(
            environment,
            "_run_compose",
            AsyncMock(side_effect=RuntimeError("compose unavailable")),
        )
        if scenario == "compose-fallback":
            monkeypatch.setattr(
                environment, "_list_instances", AsyncMock(return_value=set())
            )
    else:
        monkeypatch.setattr(
            environment, "_instance_exists", AsyncMock(return_value=True)
        )
        monkeypatch.setattr(
            environment,
            "_run",
            AsyncMock(side_effect=RuntimeError("stop failed")),
        )

    await environment.stop(delete=False)
    preserved = scenario != "compose-fallback"
    assert (environment._runtime_workdir_root is not None) is preserved

    if scenario == "instance-retry":
        assert (
            root is not None and root.exists() and environment._instance_started
        )
        monkeypatch.setattr(environment, "_run", AsyncMock(return_value=None))
        await environment.stop(delete=False)
        assert not root.exists() and environment._runtime_workdir_root is None


async def test_start_lock_rejects_a_concurrent_start(tmp_path):
    environment = _environment(tmp_path)
    async with environment._lifecycle_lock:
        pending = asyncio.create_task(environment.start(force_build=False))
        await asyncio.sleep(0)
        assert not pending.done()
        environment._instance_started = True
    with pytest.raises(RuntimeError, match="already started"):
        await pending


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (b'{"instances": {}}', None),
        (b'{"instances": "bad"}', None),
        (b'{"instances": [1]}', None),
        (b'{"instances": [{}]}', None),
        (b'{"instances": [{"instance": 1}]}', None),
        (b'{"instances": [{"instance": "first", "pid": 1}]}', {"first"}),
    ],
)
def test_instance_list_parser(output: bytes, expected: set[str] | None):
    if expected is None:
        with pytest.raises(
            RuntimeError, match="parse Singularity instance list"
        ):
            DockerfileSingularityEnvironment._parse_instance_list(output)
    else:
        assert (
            DockerfileSingularityEnvironment._parse_instance_list(output)
            == expected
        )


@pytest.mark.parametrize("inferred", [False, True])
@pytest.mark.parametrize("fail_bridge", [False, True])
async def test_allowlist_automatically_proxies_and_revokes_phase_access(
    tmp_path, monkeypatch, inferred, fail_bridge
):
    from harbor.trial.network_policy import (
        add_inference_hosts,
        inference_endpoint_host,
    )

    offline = NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
    policy = (
        add_inference_hosts(
            offline, [inference_endpoint_host("http://localhost:11434/v1")]
        )
        if inferred
        else NetworkPolicy(
            network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["localhost"]
        )
    )
    environment = _environment(
        tmp_path, network_mode=NetworkMode.NO_NETWORK, overlay=True
    )
    _prepare_started_environment(environment, tmp_path)
    starts = []
    sockets = []

    async def run(*command, **kwargs):
        if "start" in command:
            starts.append(command)

    async def stop(*, warn):
        environment._instance_started = False
        environment._instance_start_attempted = False
        return True

    async def bridge():
        path = Path(environment._proxy_temp.name) / "host.sock"
        assert path.is_socket()
        sockets.append(path)
        if fail_bridge:
            raise RuntimeError("bridge unavailable")

    monkeypatch.setattr(environment, "_run", run)
    monkeypatch.setattr(environment, "_stop_instance", stop)
    monkeypatch.setattr(environment, "_start_proxy_bridge", bridge)
    try:
        if fail_bridge:
            with pytest.raises(RuntimeError, match="bridge unavailable"):
                await environment.set_network_policy(policy)
        else:
            await environment.set_network_policy(policy)
            assert environment.network_policy == policy
            await environment.set_network_policy(offline)
        assert sockets and not sockets[0].exists()
        assert environment.network_policy == offline
        assert environment._proxy_stack is None
        start = starts[0]
        assert "--userns" in start and "--fakeroot" not in start
        assert _option_value(start, "--network") == "none"
        assert any(":/run/ursa-proxy:ro" in arg for arg in start)
        assert not any(":/run/ursa-proxy:" in arg for arg in starts[-1])
    finally:
        directory = Path(environment._proxy_temp.name)
        await environment.stop(True)
        assert not directory.exists()


@pytest.mark.parametrize("ready", [True, False])
async def test_proxy_bridge_environment_and_startup_failure(
    tmp_path, monkeypatch, ready
):
    environment = _environment(tmp_path)
    environment._runtime_path = "/usr/bin/apptainer"
    environment._instance_started = True
    policy = NetworkPolicy(
        network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["localhost"]
    )
    await environment._prepare_network_proxy(policy)
    process = Mock(returncode=None if ready else 1)
    process.stdout = asyncio.StreamReader()
    process.stdout.feed_data(b'{"port": 12345}\n' if ready else b"")
    process.stdout.feed_eof()
    spawn = AsyncMock(return_value=process)
    terminate = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(environment, "_terminate_host_process", terminate)
    try:
        if ready:
            await environment._start_proxy_bridge()
            env = environment._runtime_environment({
                "NO_PROXY": "*",
                "http_proxy": "invalid",
            })
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
                assert env[f"APPTAINERENV_{key}"] == "http://127.0.0.1:12345"
                assert (
                    env[f"APPTAINERENV_{key.lower()}"]
                    == "http://127.0.0.1:12345"
                )
            assert (
                env["APPTAINERENV_NO_PROXY"]
                == env["APPTAINERENV_no_proxy"]
                == "localhost,localhost.,127.0.0.1,::1"
            )
            process.returncode = 1
            with pytest.raises(RuntimeError, match="bridge exited"):
                await environment.exec("true")
        else:
            with pytest.raises(
                RuntimeError, match="Could not start network proxy bridge"
            ):
                await environment._start_proxy_bridge()
            assert not environment._proxy_env
    finally:
        directory = Path(environment._proxy_temp.name)
        await environment._cleanup_runtime_resources()
        terminate.assert_awaited_once_with(process)
        assert not directory.exists()
        assert not environment._proxy_env


@pytest.mark.parametrize("host", ["10.0.0.0/8", "::1/128"])
def test_allowlist_rejects_unsupported_cidrs(tmp_path, host):
    with pytest.raises(ValueError, match="CIDR"):
        _environment(
            tmp_path, network_mode=NetworkMode.ALLOWLIST, allowed_hosts=(host,)
        )


def test_allowlist_rejects_compose_sidecars(tmp_path):
    with pytest.raises(ValueError, match="restricted Harbor network phase"):
        _environment(tmp_path, compose=True, network_mode=NetworkMode.ALLOWLIST)


async def test_compose_proxy_mount_is_removed_when_policy_changes(tmp_path):
    environment = _environment(tmp_path)
    compose = tmp_path / "compose.yml"
    compose.write_text(
        "instances:\n  main:\n    start:\n      options: [containall]\n"
    )
    environment._compose_file = compose
    policy = NetworkPolicy(
        network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["localhost"]
    )
    try:
        await environment._prepare_network_proxy(policy)
        environment._set_compose_network_policy(policy)
        instance = yaml.safe_load(compose.read_text())["instances"]["main"]
        # Restricted phases keep the isolated namespace, even for a lone main.
        assert instance["network"] == {
            "allocate_ip": False,
            "enable": True,
            "type": "none",
        }
        assert "userns" in instance["start"]["options"]
        assert any(
            ":/run/ursa-proxy:ro" in option
            for option in instance["start"]["options"]
        )
        environment._set_compose_network_policy(NetworkPolicy())
        instance = yaml.safe_load(compose.read_text())["instances"]["main"]
        assert instance["start"]["options"] == ["containall"]
        # A public lone main instance uses host networking.
        assert instance["network"] == {"allocate_ip": True, "enable": False}
    finally:
        await environment._cleanup_runtime_resources()


async def test_model_forwards_are_scoped_to_policy_and_preserve_url_identity(
    tmp_path, monkeypatch
):
    environment = _environment(
        tmp_path,
        network_mode=NetworkMode.ALLOWLIST,
        allowed_hosts=("localhost", "127.0.0.1", "::1"),
    )
    policy = environment.network_policy
    launch = AsyncMock(
        side_effect=[(12345, Mock()), (12346, Mock()), (12347, Mock())]
    )
    monkeypatch.setattr(environment, "_launch_proxy_bridge", launch)
    try:
        await environment._prepare_network_proxy(policy)
        await environment._start_proxy_bridge()
        url = "https://localhost:8000/v1?mode=test"
        assert (
            await environment._forward_model_url(url)
            == "https://localhost:12346/v1?mode=test"
        )
        assert (
            await environment._forward_model_url(url)
            == "https://localhost:12346/v1?mode=test"
        )
        assert (
            await environment._forward_model_url("https://[::1]:8443/v1")
            == "https://[::1]:12347/v1"
        )
        assert (
            await environment._forward_model_url("https://api.example/v1")
            == "https://api.example/v1"
        )
        assert launch.await_count == 3
        assert launch.await_args_list[1].kwargs == {
            "target": "localhost:8000",
            "host": "127.0.0.1",
        }
        assert launch.await_args_list[2].kwargs == {
            "target": "[::1]:8443",
            "host": "::1",
        }
        with pytest.raises(ValueError, match="not allowlisted"):
            await environment._forward_model_url("http://127.0.0.2:8080/v1")
        await environment._stop_network_proxy()
        assert not environment._model_forwards
        assert await environment._forward_model_url(url) == url
    finally:
        await environment._cleanup_runtime_resources()


def _image_environment(
    tmp_path: Path,
    docker_image: str | None,
    *,
    dockerfile: bool = True,
    compose: str | None = None,
    force_pull: bool = False,
) -> DockerfileSingularityEnvironment:
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir(parents=True)
    if dockerfile:
        (environment_dir / "Dockerfile").write_text(
            "FROM ubuntu:24.04\nRUN apt-get update\nWORKDIR /app\n"
        )
    if compose is not None:
        (environment_dir / "docker-compose.yaml").write_text(compose)
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    return DockerfileSingularityEnvironment(
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


def _install_pull_runtime(
    environment: DockerfileSingularityEnvironment,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Fake a runtime that records argv, pulls, and validates SIF files."""
    log = tmp_path / "runtime.log"
    runtime = tmp_path / "bin" / "apptainer"
    runtime.parent.mkdir(exist_ok=True)
    runtime.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> {log}\n'
        'case "$1" in\n'
        "  pull) for arg; do output=$image; image=$arg; done\n"
        '        printf sif > "$output" ;;\n'
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
    environment = _image_environment(tmp_path, "example/task:20251031")
    log = _install_pull_runtime(environment, tmp_path, monkeypatch)

    first = await environment._build_main_sif(force_build=False)
    second = await environment._build_main_sif(force_build=False)

    pulls = _pulls(log)
    assert len(pulls) == 1
    assert pulls[0][-2:] == [
        str(first.with_suffix(".tmp.sif")),
        "docker://example/task:20251031",
    ]
    assert first == second
    assert first.parent == tmp_path / "cache"
    assert first.read_text() == "sif"
    assert not first.with_suffix(".tmp.sif").exists()
    # The Dockerfile still provides the workdir, as for Harbor's Docker
    # backend, but is never built.
    assert environment._workdir == "/app"


async def test_force_build_builds_dockerfile_despite_prebuilt_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _image_environment(tmp_path, "example/task:20251031")
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
    # ``prebuild --force`` refreshes the SIF a normal trial pulls instead of
    # building the Dockerfile that a prebuilt-image task may also ship.
    environment = _image_environment(tmp_path, "example/task:20251031")
    log = _install_pull_runtime(environment, tmp_path, monkeypatch)

    first = await environment.build_sifs()
    second = await environment.build_sifs(refresh=True)

    assert first == second
    pulls = _pulls(log)
    assert [pull[-1] for pull in pulls] == [
        "docker://example/task:20251031",
        "docker://example/task:20251031",
    ]
    assert "--disable-cache" in pulls[1]


async def test_force_build_without_dockerfile_refreshes_the_pull(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _image_environment(
        tmp_path, "example/task:20251031", dockerfile=False
    )
    log = _install_pull_runtime(environment, tmp_path, monkeypatch)

    await environment._build_main_sif(force_build=False)
    await environment._build_main_sif(force_build=True)

    assert len(_pulls(log)) == 2
    # The pulled image's own WorkingDir, as under Docker.
    monkeypatch.setattr(
        harbor_singularity,
        "image_config",
        lambda image, machine: {"WorkingDir": "/task"}
        if image == "example/task:20251031"
        else None,
    )
    await environment._prepare_workdir(force_build=True)
    assert environment._workdir == "/task"


async def test_force_pull_refreshes_prebuilt_image_without_building(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _image_environment(
        tmp_path, "example/task:20251031", force_pull=True
    )
    log = _install_pull_runtime(environment, tmp_path, monkeypatch)

    await environment._build_main_sif(force_build=False)
    await environment._build_main_sif(force_build=False)

    assert len(_pulls(log)) == 2


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
    # A local SIF is not part of the cache, so it gets no usage marker.
    await environment._mark_sif_in_use(image)
    assert sorted(image.parent.iterdir()) == [image]

    image.unlink()
    with pytest.raises(FileNotFoundError, match="task.sif"):
        await environment._build_main_sif(force_build=False)


def test_prebuilt_image_does_not_require_a_dockerfile(tmp_path: Path):
    environment = _image_environment(
        tmp_path / "prebuilt", "ubuntu:24.04", dockerfile=False
    )

    assert environment._main_image(force_build=False) == "ubuntu:24.04"
    with pytest.raises(FileNotFoundError, match="Dockerfile"):
        _image_environment(tmp_path / "other", None, dockerfile=False)


@pytest.mark.parametrize(
    ("main", "expected"),
    [
        ("    volumes: []\n", "docker://example/task:20251031"),
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
        "example/task:20251031",
        compose=f"services:\n  main:\n{main}",
    )
    log = _install_pull_runtime(environment, tmp_path, monkeypatch)

    await environment._build_main_sif(force_build=False)

    assert [pull[-1] for pull in _pulls(log)] == [expected]


@pytest.mark.parametrize(
    ("docker_image", "force_build", "dockerfile", "pulled"),
    [
        # Harbor builds the Dockerfile; main.image only names the result.
        (None, False, True, False),
        ("example/task:20251031", True, True, False),
        # A prebuilt image is pulled, and main.image replaces it.
        ("example/task:20251031", False, True, True),
        # Nothing to build, so main.image is the only source.
        (None, False, False, True),
        ("example/task:20251031", True, False, True),
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
    # The workdir follows the image the trial uses: the pulled image's
    # config, or the Dockerfile that --force-build builds.
    monkeypatch.setattr(
        harbor_singularity,
        "image_config",
        lambda image, machine: {"WorkingDir": f"/from/{image}"},
    )
    await environment._prepare_workdir(force_build)
    assert environment._workdir == ("/from/ubuntu:24.04" if pulled else "/app")


async def test_compose_sidecars_follow_prebuilt_main_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # The main service is pulled, so a sidecar that builds the task
    # Dockerfile is not skipped as a duplicate of the main build.
    environment = _image_environment(
        tmp_path,
        "example/task:20251031",
        compose=(
            "services:\n  main: {}\n  helper:\n    build: .\n"
            "  cache:\n    image: redis:7\n"
        ),
    )
    observed: list[tuple[str, bool]] = []

    async def build(force_build: bool, **_kwargs: object) -> Path:
        observed.append(("build", force_build))
        return tmp_path / "built.sif"

    async def pull(image: str, force_pull: bool) -> Path:
        observed.append((image, force_pull))
        return tmp_path / f"{image.replace('/', '-')}.sif"

    monkeypatch.setattr(environment, "_build_dockerfile_sif", build)
    monkeypatch.setattr(environment, "_pull_image_sif", pull)

    paths = await environment.build_sifs(refresh=True)

    assert sorted(observed) == [
        ("build", True),
        ("example/task:20251031", True),
        ("redis:7", True),
    ]
    assert len(paths) == 3


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


def test_unset_compose_variable_warns_once(monkeypatch, caplog):
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
    tmp_path: Path, monkeypatch
):
    # Harbor's Terminal-Bench mapper writes this main service.
    monkeypatch.delenv("TEST_DIR", raising=False)
    monkeypatch.setenv("CONTEXT_DIR", "/host/value/is/overridden")
    monkeypatch.setenv("URSA_TEST_HOST_ONLY", "host")
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
            docker_image="example/task:20251031",
            cpus=2,
            memory_mb=4096,
            env={"URSA_TEST_TASK": "task", "URSA_TEST_HOST_ONLY": "task"},
        ),
        persistent_env={"URSA_TEST_TASK": "persistent"},
        singularity_image_cache_dir=tmp_path / "cache",
        singularity_fakeroot=False,
    )

    variables = environment._compose_interpolation_environment()
    main = environment._load_compose_config()["services"]["main"]

    assert variables["CONTEXT_DIR"] == str(environment_dir.resolve())
    assert variables["MAIN_IMAGE_NAME"] == f"hb__{environment.environment_id}"
    assert variables["PREBUILT_IMAGE_NAME"] == "example/task:20251031"
    assert variables["CPUS"] == "2"
    assert variables["MEMORY"] == "4096M"
    # Task env overrides the host; persistent env overrides the task.
    assert variables["URSA_TEST_HOST_ONLY"] == "task"
    assert variables["URSA_TEST_TASK"] == "persistent"
    assert main["build"] == {"context": str(environment_dir.resolve())}
    assert main["image"] == f"hb__{environment.environment_id}"
    assert main["environment"] == ["TEST_DIR="]
    # Resource limits are accepted but not enforced.
    assert main["deploy"] == {}


def test_compose_legacy_log_mount_variables(tmp_path: Path):
    environment = _environment(tmp_path)
    # As for Harbor's Docker backend, only host-visible mounts count.
    assert "ENV_VERIFIER_LOGS_PATH" not in (
        environment._compose_interpolation_environment()
    )
    environment._host_mounts.append({
        "type": "bind",
        "source": str(environment.trial_paths.verifier_dir),
        "target": "/logs/verifier",
    })

    variables = environment._compose_interpolation_environment()

    assert variables["ENV_VERIFIER_LOGS_PATH"] == "/logs/verifier"
    assert variables["HOST_VERIFIER_LOGS_PATH"] == str(
        environment.trial_paths.verifier_dir
    )


@pytest.mark.parametrize(
    ("deploy", "error"),
    [
        ({"resources": {"limits": {"cpus": "2", "memory": "4G"}}}, None),
        ({"replicas": 2, "resources": {"reservations": {}}}, None),
        ({"restart_policy": {"condition": "any"}}, "deploy.replicas and"),
        ({"replicas": 0, "resources": {}}, "positive integer"),
    ],
)
def test_compose_deploy_resources_are_accepted_but_not_enforced(
    tmp_path: Path, deploy: dict[str, object], error: str | None
):
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text(
        yaml.safe_dump({"services": {"main": {"image": "x", "deploy": deploy}}})
    )

    if error is not None:
        with pytest.raises(ValueError, match=error):
            harbor_singularity._load_docker_compose([compose])
        return
    main = harbor_singularity._load_docker_compose([compose])["services"][
        "main"
    ]
    assert main["deploy"] == {
        key: value for key, value in deploy.items() if key != "resources"
    }


async def test_compose_interpolates_environment_env_file_and_build_args(
    tmp_path: Path, monkeypatch
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
        "    volumes:\n"
        '      - "${URSA_TEST_UNSET:-.}/data:/data"\n'
    )
    config = harbor_singularity._load_docker_compose([compose])
    main = config["services"]["main"]

    assert harbor_singularity._compose_environment("main", main, {}) == {
        "FROM_FILE": "secret-file",
        "LITERAL": "${URSA_TEST_TOKEN}",
        "QUOTED": "quoted",
        "CHAINED": "secret-file+",
        "COMBINED": "pre-secret-none",
        "ESCAPED": "${URSA_TEST_TOKEN}",
    }
    assert main["build"]["args"] == {"TOKEN": "secret", "PRICE": "$5"}
    assert main["volumes"] == [f"{tmp_path.resolve()}/data:/data"]


def test_compose_env_file_uses_the_interpolation_environment(
    tmp_path: Path, monkeypatch
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


SCICODE_COMPOSE = (
    "services:\n"
    "  main:\n"
    "    volumes:\n"
    '      - "${HARBOR_SCICODE_TEST_DATA_PATH:?'
    "set HARBOR_SCICODE_TEST_DATA_PATH to an absolute path to a local "
    'test_data.h5 before running harbor}:/app/test_data.h5:ro"\n'
)


def test_scicode_compose_requires_test_data_variable(
    tmp_path: Path, monkeypatch
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
    tmp_path: Path, monkeypatch
) -> tuple[DockerfileSingularityEnvironment, Path]:
    test_data = tmp_path / "data" / "test_data.h5"
    test_data.parent.mkdir()
    test_data.write_bytes(b"hdf5")
    monkeypatch.setenv("HARBOR_SCICODE_TEST_DATA_PATH", str(test_data))
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    # The Dockerfile and Compose file of Harbor's SciCode adapter.
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
    tmp_path: Path, monkeypatch
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
    assert main["network"]["enable"] is False
    environment._cleanup_compose_project()
    await environment._release_sifs_in_use()
    environment._cleanup_staging()


async def test_compose_rejects_missing_bind_source(tmp_path: Path, monkeypatch):
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
    [
        "./data:/data:z",
        "./data:/data:ro:z",
        "named:/data",
        "./a,b:/data",
        {"type": "bind", "source": "./data", "target": "/d", "read_only": 1},
        {"type": "volume", "source": "named", "target": "/data"},
    ],
)
def test_compose_rejects_unsupported_volumes(tmp_path: Path, volume: object):
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


async def _convert_main_volumes(
    tmp_path: Path,
    compose_text: str,
    main_mounts: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text(compose_text)
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
        main_mounts=main_mounts or [],
    )
    (main,) = yaml.safe_load(output.read_text())["instances"].values()
    return main


async def test_compose_keeps_read_only_harbor_mounts(tmp_path: Path):
    main = await _convert_main_volumes(
        tmp_path,
        "services:\n  main: {}\n",
        [
            {
                "type": "bind",
                "source": "/host/data",
                "target": "/data",
                "read_only": True,
            },
            {"type": "bind", "source": "/host/app", "target": "/app"},
        ],
    )

    assert "bind=/host/data:/data:ro" in main["start"]["options"]
    assert "/host/app:/app" in main["volumes"]
    assert not any(volume.endswith(":ro") for volume in main["volumes"])


def test_compose_accepts_read_only_harbor_mounts(tmp_path: Path):
    environment = _environment(tmp_path, compose=True)
    environment._mounts[0]["read_only"] = True

    environment._validate_definition()


async def test_compose_read_only_parent_does_not_hide_writable_child(
    tmp_path: Path,
):
    for name in ("data", "out", "other"):
        (tmp_path / name).mkdir()
    main = await _convert_main_volumes(
        tmp_path,
        "services:\n"
        "  main:\n"
        "    volumes:\n"
        f"      - {tmp_path}/out:/data/out\n"
        f"      - {tmp_path}/data:/data:ro\n"
        f"      - {tmp_path}/other:/other\n",
    )

    # singularity-compose emits volumes first, then start options in order.
    assert main["volumes"] == [
        f"{tmp_path}/other:/other",
        f"{tmp_path / 'staging'}:/staging",
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
    main = await _convert_main_volumes(
        tmp_path,
        "services:\n  main:\n    volumes: [./nested:/app/nested]\n",
        [{"type": "bind", "source": "/host/app", "target": "/app"}],
    )

    targets = [volume.split(":")[1] for volume in main["volumes"]]
    assert targets.index("/app") < targets.index("/app/nested")


@pytest.mark.parametrize(
    ("services", "network_mode", "network"),
    [
        (
            {"main": {}},
            NetworkMode.PUBLIC,
            {"allocate_ip": True, "enable": False},
        ),
        (
            {"main": {}},
            NetworkMode.NO_NETWORK,
            {"allocate_ip": False, "enable": True, "type": "none"},
        ),
        (
            {"main": {"ports": ["8080:80"]}},
            NetworkMode.PUBLIC,
            {"allocate_ip": True, "enable": True},
        ),
        (
            {"main": {}, "helper": {"image": "redis:7"}},
            NetworkMode.PUBLIC,
            {"allocate_ip": True, "enable": True},
        ),
    ],
)
async def test_compose_conversion_uses_host_networking_for_a_lone_main(
    tmp_path: Path,
    services: dict[str, object],
    network_mode: NetworkMode,
    network: dict[str, object],
):
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text(yaml.safe_dump({"services": services}))
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
        network_mode=network_mode,
        fakeroot=False,
    )

    instances = yaml.safe_load(output.read_text())["instances"]
    assert all(
        instance["network"] == network for instance in instances.values()
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
    monkeypatch,
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

    async def run_compose(*arguments: str) -> str:
        calls.append(arguments)
        return ""

    monkeypatch.setattr(environment, "_run_compose", run_compose)
    monkeypatch.setattr(environment, "_run", AsyncMock(return_value=""))
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


NETWORK_FATAL = (
    "FATAL:   container creation failed: network requires root or a suid "
    "installation with /etc/subuid --fakeroot; non-root users can only use "
    "--network=none unless permitted by the administrator"
)


def _compose_up_environment(
    tmp_path: Path, monkeypatch, *, running: list[str]
) -> DockerfileSingularityEnvironment:
    """Prepare a sidecar project whose singularity-compose hides a FATAL."""
    environment = _environment(tmp_path, compose=True)
    _prepare_started_environment(environment, tmp_path)
    environment._instance_started = False
    environment._instance_start_attempted = False
    project = tmp_path / "project"
    (project / "bin").mkdir(parents=True)
    environment._compose_project_dir = project
    environment._compose_file = project / "singularity-compose.yml"
    environment._compose_file.write_text(
        yaml.safe_dump({
            "instances": {
                "unitmain": {"network": {}},
                "unithelper": {"network": {}},
            }
        })
    )
    environment._compose_instances = {
        "main": ["unitmain1"],
        "helper": ["unithelper1"],
    }
    tools = tmp_path / "tools"
    tools.mkdir()
    # singularity-compose 0.1.19 logs the failed instance start and exits 0.
    compose = tools / "singularity-compose"
    compose.write_text(
        "#!/bin/sh\n"
        f"echo '{NETWORK_FATAL}' >&2\n"
        'echo \'ERROR ["singularity", "instance", "start"] '
        "return code 255'\n"
        "exit 0\n"
    )
    compose.chmod(0o755)
    listed = ", ".join(f'{{"instance": "{name}"}}' for name in running)
    runtime = tmp_path / "fake-singularity"
    runtime.write_text(
        "#!/bin/sh\n"
        'if [ "$1 $2" = "instance list" ]; then\n'
        f"  echo '{{\"instances\": [{listed}]}}'\n"
        "fi\n"
        "exit 0\n"
    )
    runtime.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tools}:{Path(sys.executable).parent}")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(
        environment, "_add_compose_service_aliases", lambda: None
    )
    return environment


async def test_compose_up_that_exits_zero_without_instances_fails(
    tmp_path: Path, monkeypatch
):
    environment = _compose_up_environment(
        tmp_path, monkeypatch, running=["unitmain1"]
    )

    with pytest.raises(RuntimeError) as raised:
        await environment._start_prepared_runtime(
            NetworkPolicy(network_mode=NetworkMode.PUBLIC)
        )

    message = str(raised.value)
    assert "did not start instance(s): unithelper1" in message
    assert "unitmain1" not in message.split("\n")[0]
    assert NETWORK_FATAL in message
    assert "return code 255" in message
    assert "setuid Apptainer" in message
    assert "/etc/subuid" in message
    assert not environment._instance_started
    assert environment._instance_start_attempted
    environment._cleanup_staging()


async def test_compose_up_keeps_its_output_when_instances_cannot_be_listed(
    tmp_path: Path, monkeypatch
):
    environment = _compose_up_environment(tmp_path, monkeypatch, running=[])

    async def failing_list() -> set[str]:
        raise RuntimeError("Could not list Singularity instances: denied")

    monkeypatch.setattr(environment, "_list_instances", failing_list)

    with pytest.raises(RuntimeError) as raised:
        await environment._start_prepared_runtime(
            NetworkPolicy(network_mode=NetworkMode.PUBLIC)
        )

    message = str(raised.value)
    assert "Could not list Singularity instances: denied" in message
    assert NETWORK_FATAL in message
    assert not environment._instance_started
    environment._cleanup_staging()


async def test_compose_up_checks_every_instance_is_running(
    tmp_path: Path, monkeypatch
):
    environment = _compose_up_environment(
        tmp_path, monkeypatch, running=["unitmain1", "unithelper1", "other"]
    )

    await environment._start_prepared_runtime(
        NetworkPolicy(network_mode=NetworkMode.PUBLIC)
    )

    assert environment._instance_started
    environment._cleanup_staging()


async def test_compose_up_through_sudo_skips_the_user_instance_list(
    tmp_path: Path, monkeypatch
):
    environment = _compose_up_environment(tmp_path, monkeypatch, running=[])
    environment._compose_uses_sudo = True

    async def unexpected_list() -> set[str]:
        pytest.fail("root's instances are not listed for the invoking user")

    monkeypatch.setattr(environment, "_list_instances", unexpected_list)

    await environment._start_prepared_runtime(
        NetworkPolicy(network_mode=NetworkMode.PUBLIC)
    )

    assert environment._instance_started
    environment._cleanup_staging()


async def test_run_returns_successful_output(tmp_path: Path):
    environment = _environment(tmp_path)

    output = await environment._run(
        sys.executable,
        "-c",
        "import sys; print('out'); print('err', file=sys.stderr)",
    )

    assert output == "out\nerr\n"


def _fake_install(
    tmp_path: Path, *, buildcfg: bool = True, starter: bool = True
) -> str:
    libexec = tmp_path / "install" / "libexec"
    if starter:
        (libexec / "apptainer" / "bin").mkdir(parents=True)
        (libexec / "apptainer" / "bin" / "starter").write_text("")
    runtime = tmp_path / "install" / "bin" / "apptainer"
    runtime.parent.mkdir(parents=True)
    script = "exit 1"
    if buildcfg:
        script = (
            'if [ "$1" = buildcfg ]; then\n'
            "  echo PREFIX=/usr\n"
            f"  echo LIBEXECDIR={libexec}\n"
            "fi"
        )
    runtime.write_text(f"#!/bin/sh\n{script}\n")
    runtime.chmod(0o755)
    harbor_singularity._runtime_lacks_setuid_starter.cache_clear()
    return str(runtime)


def test_unprivileged_runtime_is_detected_from_buildcfg(tmp_path: Path):
    runtime = _fake_install(tmp_path)

    assert harbor_singularity._runtime_lacks_setuid_starter(runtime)


def test_relocated_unprivileged_runtime_is_detected_beside_its_binary(
    tmp_path: Path,
):
    runtime = _fake_install(tmp_path, buildcfg=False)

    assert harbor_singularity._runtime_lacks_setuid_starter(runtime)


def test_runtime_without_a_known_layout_is_not_called_unprivileged(
    tmp_path: Path,
):
    runtime = _fake_install(tmp_path, buildcfg=False, starter=False)

    assert not harbor_singularity._runtime_lacks_setuid_starter(runtime)
    assert not harbor_singularity._runtime_lacks_setuid_starter(
        str(tmp_path / "missing" / "apptainer")
    )


@pytest.mark.parametrize("name", ["apptainer", "singularity"])
def test_setuid_starter_marks_the_runtime_privileged(
    tmp_path: Path, monkeypatch, name: str
):
    runtime = _fake_install(tmp_path)
    bin_dir = tmp_path / "install" / "libexec" / name / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "starter").write_text("")
    starter_suid = bin_dir / "starter-suid"
    starter_suid.write_text("")
    real_stat = Path.stat

    class RootSetuid:
        st_uid = 0
        st_mode = 0o104755

    def fake_stat(path: Path, **kwargs: object):
        if path == starter_suid:
            return RootSetuid()
        return real_stat(path, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)

    assert not harbor_singularity._runtime_lacks_setuid_starter(runtime)


def test_user_owned_starter_suid_is_not_setuid(tmp_path: Path):
    runtime = _fake_install(tmp_path)
    starter_suid = (
        tmp_path / "install" / "libexec" / "apptainer" / "bin" / "starter-suid"
    )
    starter_suid.write_text("")
    starter_suid.chmod(0o4755)

    assert harbor_singularity._runtime_lacks_setuid_starter(runtime)


async def test_start_fails_before_building_without_bridge_networking(
    tmp_path: Path, monkeypatch
):
    environment = _environment(tmp_path, compose=True, fakeroot=True)
    checked: list[str] = []

    def unprivileged(runtime: str) -> bool:
        checked.append(runtime)
        return True

    async def unexpected(*_args: object, **_kwargs: object) -> Path:
        pytest.fail("the image must not be built without bridge networking")

    monkeypatch.setattr(harbor_singularity.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        harbor_singularity, "_runtime_lacks_setuid_starter", unprivileged
    )
    monkeypatch.setattr(environment, "_build_main_sif", unexpected)

    with pytest.raises(RuntimeError) as raised:
        await environment.start(force_build=False)

    message = str(raised.value)
    assert "bridge networking" in message
    assert "setuid Apptainer" in message
    assert "/etc/subuid" in message
    assert "root" in message
    assert "Docker" in message
    assert checked == [environment._instance_runtime()]
    assert environment._runtime_workdir_root is None


@pytest.mark.parametrize(
    ("compose", "euid", "unprivileged", "network_mode", "fakeroot"),
    [
        # Root and setuid installations are never rejected.
        ("sidecar", 0, True, NetworkMode.PUBLIC, True),
        ("sidecar", 1000, False, NetworkMode.PUBLIC, True),
        # A lone main uses host networking; restricted phases use
        # --network none.
        ("main", 1000, True, NetworkMode.PUBLIC, True),
        ("ports-main", 1000, True, NetworkMode.NO_NETWORK, True),
        ("ports-main", 1000, True, NetworkMode.ALLOWLIST, True),
        # Without fakeroot, singularity-compose uses sudo for published ports
        # and otherwise --network none, so it never asks for a bridge.
        ("ports-main", 1000, True, NetworkMode.PUBLIC, False),
        ("sidecar", 1000, True, NetworkMode.PUBLIC, False),
    ],
)
def test_bridge_check_allows_projects_that_can_start(
    tmp_path: Path,
    monkeypatch,
    compose: str,
    euid: int,
    unprivileged: bool,
    network_mode: NetworkMode,
    fakeroot: bool,
):
    environment = _environment(
        tmp_path, network_mode=network_mode, fakeroot=fakeroot
    )
    services = {
        "sidecar": "  main: {}\n  helper:\n    image: redis:7\n",
        "main": "  main: {}\n",
        "ports-main": "  main:\n    ports:\n      - '8080:80'\n",
    }[compose]
    (environment.environment_dir / "docker-compose.yaml").write_text(
        f"services:\n{services}"
    )
    monkeypatch.setattr(harbor_singularity.os, "geteuid", lambda: euid)
    monkeypatch.setattr(
        harbor_singularity,
        "_runtime_lacks_setuid_starter",
        lambda _runtime: unprivileged,
    )

    environment._raise_if_compose_bridge_unavailable()


def test_bridge_check_covers_public_phases_and_published_ports(
    tmp_path: Path, monkeypatch
):
    environment = _environment(
        tmp_path,
        network_mode=NetworkMode.NO_NETWORK,
        phase_network_modes=(NetworkMode.PUBLIC,),
        overlay=True,
        fakeroot=True,
    )
    (environment.environment_dir / "docker-compose.yaml").write_text(
        "services:\n  main:\n    ports:\n      - '8080:80'\n"
    )
    monkeypatch.setattr(harbor_singularity.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        harbor_singularity, "_runtime_lacks_setuid_starter", lambda _r: True
    )

    with pytest.raises(RuntimeError, match="setuid Apptainer"):
        environment._raise_if_compose_bridge_unavailable()


@pytest.mark.parametrize(
    ("name", "version", "prefix"),
    [
        ("singularity", "singularity-ce version 3.6.2", "SINGULARITYENV_"),
        ("singularity", "apptainer version 1.3.4", "APPTAINERENV_"),
        ("apptainer", "", "APPTAINERENV_"),
    ],
)
def test_runtime_environment_prefix_follows_the_runtime(
    tmp_path: Path, monkeypatch, name: str, version: str, prefix: str
):
    monkeypatch.setattr(
        harbor_singularity, "_runtime_is_apptainer", _runtime_is_apptainer
    )
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


def test_apptainer_behind_a_singularity_alias_enables_fakeroot(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        harbor_singularity, "_runtime_is_apptainer", _runtime_is_apptainer
    )
    runtime = tmp_path / "bin" / "singularity"
    runtime.parent.mkdir()
    runtime.write_text("#!/bin/sh\necho 'apptainer version 1.5.5'\n")
    runtime.chmod(0o755)
    environment = _environment(tmp_path, fakeroot=None)
    environment._runtime_path = str(runtime)

    environment._instance_runtime()

    assert environment._fakeroot is True


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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (True, True),
        (False, False),
        ("true", True),
        (" True ", True),
        ("false", False),
        ("1", True),
        ("0", False),
        ("yes", True),
        ("no", False),
        (1, True),
        (0, False),
    ],
)
def test_fakeroot_setting_accepts_cli_values(
    tmp_path: Path, value: bool | int | str | None, expected: bool | None
):
    environment = _environment(tmp_path, fakeroot=value)
    assert environment._fakeroot_setting is expected


def test_fakeroot_setting_accepts_harbor_cli_values(tmp_path: Path):
    parsed = parse_kwargs(["singularity_fakeroot=1", "other=0"])

    assert DockerfileSingularityEnvironment._parse_bool(
        "singularity_fakeroot", parsed["singularity_fakeroot"]
    )
    assert not DockerfileSingularityEnvironment._parse_bool(
        "other", parsed["other"]
    )
    # A non-JSON string must not turn fakeroot on by being truthy.
    assert _environment(tmp_path, fakeroot="false")._fakeroot is False


@pytest.mark.parametrize("value", ["auto", "sometimes", 2, -1, 1.0])
def test_fakeroot_setting_rejects_unknown_values(tmp_path: Path, value: object):
    with pytest.raises(ValueError, match="singularity_fakeroot"):
        _environment(tmp_path, fakeroot=value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("false", False), ("0", False), (0, False), ("true", True), (1, True)],
)
def test_overlay_and_force_pull_settings_parse_cli_values(
    tmp_path: Path, value: int | str, expected: bool
):
    environment = _environment(
        tmp_path,
        extra_kwargs={
            "singularity_overlay": value,
            "singularity_force_pull": value,
        },
    )

    assert environment._overlay_enabled is expected
    assert environment._force_pull is expected


@pytest.mark.parametrize(
    "name", ["singularity_overlay", "singularity_force_pull"]
)
def test_overlay_and_force_pull_settings_reject_unknown_values(
    tmp_path: Path, name: str
):
    with pytest.raises(ValueError, match=name):
        _environment(tmp_path, extra_kwargs={name: "sometimes"})


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


FAKED_DAEMON_ERROR = (
    "/.singularity.d/libs/fakeroot: error while starting the `faked' daemon."
)
_FAKED_DAEMON_ERROR_SH = FAKED_DAEMON_ERROR.replace("`", "\\`")


async def test_fakeroot_command_failure_in_one_image_is_not_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    # The host's fakeroot(1) cannot load in a musl or older-glibc image; the
    # host's fakeroot still works without it, and other images keep it.
    monkeypatch.setattr(
        DockerfileSingularityEnvironment, "_fakeroot_unavailable_reasons", {}
    )
    environment = _environment(tmp_path, fakeroot=True)
    _write_fake_runtime(
        environment,
        tmp_path,
        'case " $* " in *" --ignore-fakeroot-command "*) exit 0 ;; esac; '
        f'echo "{_FAKED_DAEMON_ERROR_SH}" >&2; exit 1',
    )

    with caplog.at_level("WARNING"):
        await environment._resolve_fakeroot()

    assert environment._fakeroot
    assert environment._ignore_fakeroot_command
    assert not environment._fakeroot_unavailable_reasons
    assert "--ignore-fakeroot-command" in caplog.text
    args = environment._isolated_runtime_args(tmp_path / "workdir")
    assert args[args.index("--fakeroot") + 1] == "--ignore-fakeroot-command"
    assert "--ignore-fakeroot-command" in environment._instance_exec_prefix()
    environment._cleanup_runtime_workdir()


async def test_fakeroot_command_failure_that_persists_fails_only_the_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        DockerfileSingularityEnvironment, "_fakeroot_unavailable_reasons", {}
    )
    environment = _environment(tmp_path, fakeroot=True)
    _write_fake_runtime(
        environment,
        tmp_path,
        'case " $* " in *" --ignore-fakeroot-command "*) '
        'echo "FATAL: broken image" >&2; exit 1 ;; esac; '
        f'echo "{_FAKED_DAEMON_ERROR_SH}" >&2; exit 1',
    )

    with pytest.raises(RuntimeError, match="broken image") as raised:
        await environment._resolve_fakeroot()

    assert "singularity_fakeroot=false" not in str(raised.value)
    assert not environment._fakeroot_unavailable_reasons
    assert not environment._ignore_fakeroot_command
    environment._cleanup_runtime_workdir()


@pytest.mark.parametrize("fakeroot", [None, True])
async def test_start_fails_before_building_when_fakeroot_is_known_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fakeroot: bool | None
):
    environment = _environment(tmp_path, fakeroot=fakeroot)
    environment._runtime_path = "/usr/bin/apptainer"
    monkeypatch.setattr(
        DockerfileSingularityEnvironment,
        "_fakeroot_unavailable_reasons",
        {"/usr/bin/apptainer": FAKEROOT_FATAL},
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

    async def run(*command: str, **_kwargs: object) -> str:
        probes.append(command)
        return ""

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

    for name in ("agent", "verifier", "default"):
        (tmp_path / name).mkdir()
    environments = [
        _environment(tmp_path / name, fakeroot="false")
        for name in ("agent", "verifier")
    ]
    # SingularityCE's default is no fakeroot, which needs no warning.
    environments.append(_environment(tmp_path / "default", fakeroot=None))
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
    assert len(warnings) == 2
    assert "singularity_fakeroot=false" in warnings[0]
    assert "read-only" in warnings[0]
    assert "cannot switch to requested container user" in warnings[1]


async def test_start_resolves_fakeroot_before_overlay_seed_and_compose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        DockerfileSingularityEnvironment, "_fakeroot_unavailable_reasons", {}
    )
    environment = _environment(
        tmp_path, compose=True, fakeroot=None, overlay=True
    )
    environment._runtime_path = "/usr/bin/apptainer"
    observed: list[object] = []

    async def build(_force_build: bool) -> Path:
        return tmp_path / "image.sif"

    async def mark(_path: Path) -> None:
        pass

    async def probe() -> None:
        observed.append("probe")

    def overlay() -> None:
        observed.append("overlay")

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
    monkeypatch.setattr(environment, "_prepare_disk_overlay", overlay)
    monkeypatch.setattr(environment, "_seed_harbor_bind_sources", seed)
    monkeypatch.setattr(
        harbor_singularity, "docker_compose_to_singularity_compose", convert
    )
    monkeypatch.setattr(environment, "_start_prepared_runtime", start_runtime)

    with pytest.raises(RuntimeError, match="stop after compose preparation"):
        await environment.start(force_build=False)

    assert observed == [
        "probe",
        "overlay",
        ("seed", True),
        ("compose", True),
    ]


@pytest.mark.parametrize(
    ("runtime", "fakeroot", "wrapped", "shared"),
    [
        # Root-mapped fakeroot: Apptainer's fakeroot command wraps execs.
        ("apptainer", True, True, True),
        # Subuid-mapped fakeroot: no wrapper, so no FAKEROOTKEY.
        ("apptainer", True, False, False),
        ("apptainer", False, True, False),
        ("singularity", True, True, False),
    ],
)
async def test_instance_start_detects_apptainer_fakeroot_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime: str,
    fakeroot: bool,
    wrapped: bool,
    shared: bool,
):
    environment = _environment(tmp_path, fakeroot=fakeroot)
    _prepare_started_environment(environment, tmp_path)
    environment._instance_started = False
    environment._runtime_path = str(tmp_path / runtime)
    commands: list[list[str]] = []

    async def run(*command: str, **_kwargs: object) -> str:
        commands.append(list(command))
        if command[-1] == harbor_singularity._FAKEROOT_COMMAND_PROBE:
            # Apptainer's own warnings share the captured output.
            output = "WARNING: Skipping mount /etc/localtime\n"
            if wrapped:
                output += harbor_singularity._FAKEROOT_COMMAND_MARKER + "\n"
            return output
        return ""

    monkeypatch.setattr(environment, "_run", run)
    monkeypatch.setattr(
        environment, "_instance_start_command", lambda _policy: ["start"]
    )

    await environment._start_prepared_runtime(
        NetworkPolicy(network_mode=NetworkMode.PUBLIC)
    )

    instance = f"instance://{environment._instance_name}"
    probed = [
        command
        for command in commands
        if command[-1] == harbor_singularity._FAKEROOT_COMMAND_PROBE
    ]
    assert len(probed) == (runtime == "apptainer" and fakeroot)
    for command in probed:
        assert "--env" not in command
        assert command[command.index(instance) + 1] == "/bin/sh"
    assert environment._shared_faked is shared
    # The startup exec that follows starts the shared faked, and under it
    # sets up apt and the sudo shim.
    if shared:
        assert commands[-1][-4:] == [
            "-c",
            harbor_singularity._FAKEROOT_COMMAND_SETUP,
            "sh",
            harbor_singularity._SUDO_SHIM,
        ]
    else:
        assert commands[-1][-1] == "true"
    assert ("--env" in commands[-1]) is shared

    # A stop forgets the faked; the next start probes the new instance.
    monkeypatch.setattr(environment, "_instance_exists", lambda: _await(True))
    assert await environment._stop_instance(warn=False)
    assert environment._shared_faked is False
    environment._cleanup_staging()


@pytest.mark.parametrize("shared", [True, False])
async def test_exec_shares_one_faked_only_when_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared: bool
):
    environment = _environment(tmp_path, fakeroot=True)
    _prepare_started_environment(environment, tmp_path)
    environment._runtime_path = str(tmp_path / "apptainer")
    environment._shared_faked = shared
    observed: dict[str, object] = {}

    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def spawn(*command: str, **kwargs: object) -> Process:
        observed["command"] = list(command)
        return Process()

    monkeypatch.setattr(
        harbor_singularity.asyncio, "create_subprocess_exec", spawn
    )

    await environment.exec("true")

    command = observed["command"]
    assert isinstance(command, list)
    instance = command.index(f"instance://{environment._instance_name}")
    assert command[instance + 1 : instance + 3] == (
        ["/bin/sh", "-c"] if shared else ["bash", "-c"]
    )
    if shared:
        assert command[instance - 2 : instance] == [
            "--env",
            "FAKEROOTKEY=ursa-shared-faked",
        ]
        assert command[instance + 3] == harbor_singularity._SHARED_FAKED_SCRIPT
        assert command[instance + 5 : instance + 7] == ["bash", "-c"]
    else:
        assert "--env" not in command
    environment._cleanup_staging()


@pytest.mark.parametrize("shared", [True, False])
async def test_proxy_bridge_does_not_depend_on_the_shared_faked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared: bool
):
    # A bridge under the shared faked would block once an agent kills it.
    environment = _environment(tmp_path, fakeroot=True)
    environment._runtime_path = str(tmp_path / "apptainer")
    environment._shared_faked = shared

    command = environment._instance_exec_prefix(cwd="/", shared_faked=False)

    instance = command.index(f"instance://{environment._instance_name}")
    assert command[instance + 1 :] == []
    assert "FAKEROOTKEY=ursa-shared-faked" not in command
    assert ("FAKEROOTKEY=ursa-no-fakeroot" in command) is shared

    observed: dict[str, list[str]] = {}

    async def spawn(*command: str, **_kwargs: object) -> None:
        observed["command"] = list(command)
        raise RuntimeError("stop after spawning the bridge")

    environment._proxy_stack = AsyncExitStack()
    environment._proxy_temp = tempfile.TemporaryDirectory()
    monkeypatch.setattr(
        harbor_singularity.asyncio, "create_subprocess_exec", spawn
    )
    try:
        with pytest.raises(RuntimeError, match="stop after spawning"):
            await environment._launch_proxy_bridge()
    finally:
        await environment._proxy_stack.aclose()
        environment._proxy_temp.cleanup()

    assert harbor_singularity._SHARED_FAKED_SCRIPT not in observed["command"]
    assert "FAKEROOTKEY=ursa-shared-faked" not in observed["command"]


@pytest.mark.parametrize("key", ["1151152171", "", None])
async def test_fakeroot_command_probe_reports_only_a_wrapped_exec(
    tmp_path: Path, key: str | None
):
    env = {"PATH": "/usr/bin:/bin"}
    if key is not None:
        env["FAKEROOTKEY"] = key
    # A record left by the faked of an instance that was restarted.
    state = tmp_path / "ursa-faked"
    state.write_text("1151152171:68\n")
    process = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        harbor_singularity._FAKEROOT_COMMAND_PROBE.replace(
            harbor_singularity._SHARED_FAKED_STATE, str(state)
        ),
        stdout=asyncio.subprocess.PIPE,
        env=env,
    )
    stdout, _ = await process.communicate()

    assert process.returncode == 0
    assert stdout.decode().split() == (
        [harbor_singularity._FAKEROOT_COMMAND_MARKER] if key else []
    )
    assert not state.exists()


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("proxy", [True, False])
async def test_attach_shell_uses_the_shared_faked_when_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared: bool, proxy: bool
):
    environment = _environment(tmp_path, fakeroot=True)
    _prepare_started_environment(environment, tmp_path)
    environment._runtime_path = str(tmp_path / "apptainer")
    environment._shared_faked = shared
    if proxy:
        environment._proxy_process = Mock(returncode=None)
    observed: list[list[str]] = []

    def execvpe(_file: str, args: list[str], _env: dict[str, str]) -> None:
        observed.append(list(args))

    class Process:
        async def wait(self) -> int:
            return 0

    async def spawn(*command: str, **_kwargs: object) -> Process:
        observed.append(list(command))
        return Process()

    monkeypatch.setattr(harbor_singularity.os, "execvpe", execvpe)
    monkeypatch.setattr(
        harbor_singularity.asyncio, "create_subprocess_exec", spawn
    )

    await environment.attach()

    [command] = observed
    instance = command.index(f"instance://{environment._instance_name}")
    if shared:
        assert command == [
            *environment._instance_exec_prefix(),
            "/.singularity.d/actions/shell",
        ]
    else:
        assert command[1] == "shell"
        assert command[instance + 1 :] == []
    environment._proxy_process = None
    environment._cleanup_staging()


async def test_shared_faked_script_runs_the_command_without_fakeroot():
    # Outside Apptainer's fakeroot command there is no faked to share: the
    # script clears the key that stopped Apptainer's wrapper and runs as is.
    process = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        harbor_singularity._SHARED_FAKED_SCRIPT,
        "ursa-shared-faked",
        "sh",
        "-c",
        'echo "${FAKEROOTKEY-unset} ${LD_PRELOAD-unset} $1"',
        "sh",
        "argument with spaces",
        stdout=asyncio.subprocess.PIPE,
        env={"FAKEROOTKEY": "ursa-shared-faked", "PATH": "/usr/bin:/bin"},
    )
    stdout, _ = await process.communicate()

    assert process.returncode == 0
    assert stdout.decode().strip() == "unset unset argument with spaces"


def _sudo_shim_bin(tmp_path: Path) -> dict[str, str]:
    """Stub the image's sudo, id, getent, and su around the sudo shim."""
    shim_dir = tmp_path / "sbin"
    stub_dir = tmp_path / "bin"
    home = tmp_path / "home" / "bob"
    for directory in (shim_dir, stub_dir, home):
        directory.mkdir(parents=True)
    scripts = {
        shim_dir / "sudo": harbor_singularity._SUDO_SHIM,
        stub_dir / "sudo": '#!/bin/sh\necho "image sudo: $*"\n',
        stub_dir / "id": (
            "#!/bin/sh\n"
            'case $1 in -un) [ "$TEST_UID" = 0 ] && echo root || echo bob ;;'
            ' *) echo "$TEST_UID" ;; esac\n'
        ),
        stub_dir / "getent": (
            "#!/bin/sh\n"
            "case $2 in root | 0) echo root:x:0:0:root:/root:/bin/sh ;;"
            f" bob | 1000) echo bob:x:1000:1000::{home}:/bin/sh ;;"
            " *) exit 2 ;; esac\n"
        ),
        # su runs the command as the shim built it and reports the user.
        stub_dir / "su": (
            '#!/bin/sh\necho "su $1 $2 $3 $4"\nexec /bin/sh -c "$5"\n'
        ),
    }
    for path, script in scripts.items():
        path.write_text(script)
        path.chmod(0o755)
    return {"PATH": f"{shim_dir}:{stub_dir}:/usr/bin:/bin", "TEST_UID": "0"}


@pytest.mark.parametrize(
    ("arguments", "uid", "expected"),
    [
        # Root keeps its environment and needs no sudo to stay root.
        (
            ["-E", "MODE=kept", "sh", "-c", 'echo "$MODE $SUDO_USER $PRELOAD"'],
            "0",
            ["kept root fakeroot"],
        ),
        (["-nH", "--", "sh", "-c", 'echo "$HOME"'], "0", ["/root"]),
        (
            ["-u", "bob", "printf", "[%s]", "a b", "c'd", "$x"],
            "0",
            ["su bob -s /bin/sh -c", "[a b][c'd][$x]"],
        ),
        (
            ["-Eu#1000", "sh", "-c", 'echo "$PRELOAD"'],
            "0",
            ["su bob -s /bin/sh -c", "fakeroot"],
        ),
        (["--user=bob", "-i", "pwd"], "0", ["su bob -s /bin/sh -c", "{home}"]),
        # Other modes and non-root callers get the image's sudo.
        (["-l"], "0", ["image sudo: -l"]),
        (["-u", "bob", "-V"], "0", ["image sudo: -u bob -V"]),
        ([], "0", ["image sudo:"]),
        (["-n", "true"], "1000", ["image sudo: -n true"]),
    ],
)
async def test_sudo_shim_keeps_the_environment_of_root_commands(
    tmp_path: Path, arguments: list[str], uid: str, expected: list[str]
):
    env = {**_sudo_shim_bin(tmp_path), "TEST_UID": uid, "PRELOAD": "fakeroot"}
    process = await asyncio.create_subprocess_exec(
        str(tmp_path / "sbin" / "sudo"),
        *arguments,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    stdout, _ = await process.communicate()

    assert process.returncode == 0, stdout.decode()
    home = str(tmp_path / "home" / "bob")
    assert [line.rstrip() for line in stdout.decode().splitlines()] == [
        line.format(home=home) for line in expected
    ]


@pytest.mark.parametrize(
    ("image_sudo", "other_sudo"),
    [(True, False), (False, False), (True, True)],
)
async def test_fakeroot_command_setup_shadows_only_an_image_sudo(
    tmp_path: Path, image_sudo: bool, other_sudo: bool
):
    root = tmp_path / "root"
    (root / "etc" / "apt").mkdir(parents=True)
    (root / "usr" / "bin").mkdir(parents=True)
    if image_sudo:
        (root / "usr" / "bin" / "sudo").write_text("#!/bin/sh\n")
        (root / "usr" / "bin" / "sudo").chmod(0o755)
    shim = root / harbor_singularity._SUDO_SHIM_PATH.lstrip("/")
    if other_sudo:
        shim.parent.mkdir(parents=True)
        shim.write_text("#!/bin/sh\necho another sudo\n")
    script = (
        harbor_singularity._FAKEROOT_COMMAND_SETUP.replace(
            "/etc/apt", f"{root}/etc/apt"
        )
        .replace("/usr/local", f"{root}/usr/local")
        .replace("/usr/bin/sudo", f"{root}/usr/bin/sudo")
    )
    for _ in range(2):  # An instance restart runs it again.
        process = await asyncio.create_subprocess_exec(
            "/bin/sh", "-c", script, "sh", harbor_singularity._SUDO_SHIM
        )
        assert await process.wait() == 0

    config = root / harbor_singularity._APT_FAKEROOT_CONFIG.lstrip("/")
    lines = config.read_text().splitlines()
    assert 'APT::Sandbox::User "root";' in lines
    # Immediate retries and timeouts, since apt 2.8 can wait forever on a
    # delayed retry after a transient fetch failure; no address-family pin.
    assert 'Acquire::Retries "5";' in lines
    assert 'Acquire::Retries::Delay "false";' in lines
    assert 'Acquire::http::Timeout "20";' in lines
    assert 'Acquire::https::Timeout "20";' in lines
    assert not any("ForceIPv4" in line for line in lines)
    stash = root / harbor_singularity._SUDO_SHIM_STASH.lstrip("/")
    assert stash.read_text() == harbor_singularity._SUDO_SHIM + "\n"
    if other_sudo:
        assert shim.read_text() == "#!/bin/sh\necho another sudo\n"
    elif image_sudo:
        assert shim.read_text() == stash.read_text()
        assert shim.stat().st_mode & 0o111
    else:
        assert not shim.exists()
    assert harbor_singularity._SUDO_SHIM_MARKER in stash.read_text()


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
    ("fakeroot", "overlay", "agent"),
    [
        (False, False, None),
        (False, True, None),
        (True, True, None),
        (True, False, None),
        # Without fakeroot an agent layer's trial still needs a writable
        # HOME; with fakeroot it keeps the layer's /root.
        (False, True, "goose"),
        (True, True, "goose"),
    ],
)
def test_instance_home_is_private_unless_fakeroot_has_an_overlay(
    tmp_path: Path, fakeroot: bool, overlay: bool, agent: str | None
):
    environment = _environment(
        tmp_path, fakeroot=fakeroot, overlay=overlay, agent=agent
    )
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
    # Seed copies keep no home at all.
    assert "--no-home" in environment._isolated_runtime_args(tmp_path)
    environment._cleanup_staging()


@pytest.mark.parametrize(
    ("runtime", "overlay", "fakeroot", "expected"),
    [
        ("apptainer", True, True, ["/etc/localtime", "tmp"]),
        # Without fakeroot the invoking user cannot write the image's
        # root-owned /tmp or workdir through the overlay.
        ("apptainer", True, False, ["/etc/localtime"]),
        ("apptainer", False, True, []),
        ("singularity", True, True, []),
    ],
)
def test_apptainer_overlay_keeps_the_image_localtime_and_tmp(
    tmp_path: Path,
    runtime: str,
    overlay: bool,
    fakeroot: bool,
    expected: list[str],
):
    environment = _environment(tmp_path, fakeroot=fakeroot)
    environment._runtime_path = str(tmp_path / runtime)
    environment._workdir = "/app"
    environment._prepare_runtime_workdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    environment._staging_dir = staging
    environment._sif_path = tmp_path / "image.sif"
    if overlay:
        environment._overlay_path = tmp_path / "overlay.img"

    start = environment._instance_start_command()

    # tzdata's postinst cannot replace a bind-mounted /etc/localtime, and
    # os.rename between /app and the workdir's /tmp fails with EXDEV.
    no_mounts = [
        start[index + 1]
        for index, option in enumerate(start)
        if option == "--no-mount"
    ]
    assert no_mounts == expected
    # A bind over /app would be a mount of its own, and renames from it to
    # the overlay's /tmp fail with EXDEV too.
    workdir_bind = f"{staging / 'harbor-writable' / 'workdir'}:/app"
    assert (workdir_bind in start) is ("tmp" not in expected)
    environment._cleanup_staging()


@pytest.mark.parametrize("keep", [True, False])
async def test_compose_main_overlay_keeps_the_image_localtime_and_tmp(
    tmp_path: Path, keep: bool
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
    output = tmp_path / "singularity-compose.yml"

    async def resolve(_name: str, service: dict[str, object]) -> str:
        return str(service["image"])

    await docker_compose_to_singularity_compose(
        compose,
        output,
        identity="unit",
        image_resolver=resolve,
        staging_dir=staging,
        fakeroot=True,
        main_overlay=tmp_path / "overlay.img",
        main_no_mounts=("/etc/localtime", "tmp") if keep else (),
    )

    instances = yaml.safe_load(output.read_text())["instances"]
    options = {
        key.split("-")[1]: instance["start"]["options"]
        for key, instance in instances.items()
    }
    for option in ("no-mount=/etc/localtime", "no-mount=tmp"):
        assert (option in options["main"]) is keep
        assert option not in options["helper"]


@pytest.mark.parametrize(
    ("runtime", "fakeroot", "creates"),
    [
        ("apptainer", True, True),
        ("apptainer", False, False),
        ("singularity", True, False),
    ],
)
async def test_instance_start_creates_an_unbound_workdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime: str,
    fakeroot: bool,
    creates: bool,
):
    environment = _environment(tmp_path, fakeroot=fakeroot)
    _prepare_started_environment(environment, tmp_path)
    environment._instance_started = False
    environment._runtime_path = str(tmp_path / runtime)
    commands: list[list[str]] = []

    async def run(*command: str, **_kwargs: object) -> str:
        commands.append(list(command))
        return ""

    monkeypatch.setattr(environment, "_run", run)
    monkeypatch.setattr(
        environment, "_instance_start_command", lambda _policy: ["start"]
    )

    await environment._start_prepared_runtime(
        NetworkPolicy(network_mode=NetworkMode.PUBLIC)
    )

    # Over Apptainer's overlay with fakeroot the workdir has no bind, so the
    # start creates it before the first exec that starts in it.
    created = [
        index
        for index, command in enumerate(commands)
        if command[-3:] == ["mkdir", "-p", environment._workdir]
    ]
    assert created == ([1] if creates else [])
    for index in created:
        assert _option_value(commands[index], "--pwd") == "/"
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
    # Both runtimes refuse HOME through their environment prefixes.
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
    compose.write_text(
        "services:\n"
        "  main:\n    image: x\n"
        "  helper:\n    image: x\n    deploy:\n      replicas: 2\n"
    )
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
            service_workdirs={
                "main": (tmp_path / "main",),
                "helper": (tmp_path / "helper-1", tmp_path / "helper-2"),
            },
            home_root=None if fakeroot else staging / "harbor-home",
        )
        instances = yaml.safe_load(output.read_text())["instances"]
        assert len(instances) == 3
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


def test_nested_mount_gets_a_mount_point(tmp_path: Path):
    environment = _environment(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    data_file = tmp_path / "data.h5"
    data_file.touch()
    environment._host_mounts.extend([
        {
            "type": "bind",
            "source": str(data),
            "target": "/logs/agent/data",
            "read_only": True,
        },
        {
            "type": "bind",
            "source": str(data_file),
            "target": "/logs/verifier/data.h5",
            "read_only": True,
        },
        {
            "type": "bind",
            "source": str(data_file),
            "target": "/logs/artifacts/seeded",
            "read_only": True,
        },
    ])
    staging = tmp_path / "staging"
    staging.mkdir()
    environment._staging_dir = staging
    environment._prepare_harbor_bind_sources()
    writable = staging / "harbor-writable" / "logs"
    # Content the seed copied from the image at a mount target is kept.
    (writable / "artifacts" / "seeded").mkdir()

    environment._prepare_nested_mount_points()
    environment._prepare_nested_mount_points()

    assert (writable / "agent" / "data").is_dir()
    assert (writable / "verifier" / "data.h5").is_file()
    assert (writable / "artifacts" / "seeded").is_dir()


@pytest.mark.parametrize("cancel", [False, True])
async def test_exec_returns_while_a_background_job_holds_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool
):
    environment = _environment(tmp_path)
    _prepare_started_environment(environment, tmp_path)
    monkeypatch.setattr(harbor_singularity, "_EXEC_OUTPUT_DRAIN_SEC", 0.05)
    cleaned: list[str] = []
    started = asyncio.Event()

    class Process:
        returncode = None

        async def communicate(self) -> tuple[bytes, bytes]:
            # Output never ends, as when ``server &`` keeps the pipes open.
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def spawn(*_command: str, **_kwargs: object) -> Process:
        return Process()

    async def cleanup(_process: Process, pid_file: str, **_kwargs: object):
        cleaned.append(pid_file)

    monkeypatch.setattr(
        harbor_singularity.asyncio, "create_subprocess_exec", spawn
    )
    monkeypatch.setattr(environment, "_cleanup_exec_process", cleanup)

    if cancel:
        task = asyncio.create_task(environment.exec("server &"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
    else:
        result = await asyncio.wait_for(
            environment.exec("server &", timeout_sec=0.05), timeout=5
        )
        assert result.return_code == 124
        assert "timed out" in result.stderr
    assert len(cleaned) == 1
    environment._cleanup_staging()


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
    monkeypatch.setattr(environment, "_exec_instance", unexpected)

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
        await environment.service_download_file(
            "/logs/x", tmp_path / "helper-file" / "x", service="helper"
        )

    for name in ("agent", "file", "filtered", "helper", "helper-file"):
        assert not (tmp_path / name).exists()
    assert "Skipping download of /logs/agent" in caplog.text
    assert not [
        record for record in caplog.records if record.levelname == "ERROR"
    ]
