import asyncio
import base64
import json
import shlex
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("harbor")
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import (
    MCPServerConfig,
    NetworkMode,
    NetworkPolicy,
)
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
)
from langchain_core.tools import tool
from langgraph.checkpoint.base import empty_checkpoint
from packaging.requirements import Requirement

from ursa.agents import BaseAgent
from ursa.cli import agent_management
from ursa.integrations import harbor as harbor_integration
from ursa.integrations.harbor import (
    UrsaHarborAgent,
    _jsonl_token_usage,
)
from ursa.integrations.harbor.runner import _agent_config
from ursa.integrations.harbor.runner import _run as _runner_run
from ursa.util import Checkpointer
from ursa.util.events import AgentEvents


@tool
async def runner_echo(text: str) -> dict[str, str]:
    """Echo text through the runner's tool lifecycle."""
    return {"echo": text}


class HarborLifecycleAgent(BaseAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.lifecycle_model = FakeMessagesListChatModel(
            responses=[
                AIMessage(
                    content="runner answer",
                    usage_metadata={
                        "input_tokens": 12,
                        "output_tokens": 7,
                        "total_tokens": 19,
                    },
                    response_metadata={
                        "token_usage": {
                            "prompt_tokens": 12,
                            "completion_tokens": 7,
                            "completion_tokens_details": {
                                "reasoning_tokens": 4
                            },
                            "prompt_tokens_details": {"cached_tokens": 6},
                        }
                    },
                )
            ]
        )

    def _build_graph(self) -> None:
        pass

    async def _ainvoke(self, inputs: Any, **config: Any) -> dict[str, Any]:
        async def lifecycle(
            _inputs: Any, config: RunnableConfig
        ) -> dict[str, Any]:
            await AgentEvents(agent="ExecutionAgent", config=config).aemit(
                "Inspecting the workspace", stage="work"
            )
            answer = await self.lifecycle_model.ainvoke([], config=config)
            if any(
                getattr(message, "content", None) == "fail after usage"
                for message in _inputs.get("messages", [])
            ):
                raise RuntimeError("agent failed after chat")
            echoed = await runner_echo.ainvoke(
                {"text": "needle"}, config=config
            )
            await self.checkpointer.conn.execute(
                "CREATE TABLE runner_probe (value TEXT)"
            )
            await self.checkpointer.conn.execute(
                "INSERT INTO runner_probe VALUES ('persisted')"
            )
            return {"answer": answer.content, "tool": echoed}

        return (
            await RunnableLambda(lifecycle)
            .with_config({"run_name": "runner_lifecycle"})
            .ainvoke(inputs, config=self.build_config(**config))
        )

    def format_result(self, result: dict[str, Any]) -> str:
        return str(result["answer"])


class SetupFailingAgent(BaseAgent):
    connection: ClassVar[Any | None] = None

    def __init__(self, *, checkpointer: Any, **kwargs: Any) -> None:
        type(self).connection = checkpointer.conn
        raise RuntimeError("agent setup failed")

    def _build_graph(self) -> None:
        pass


def _native_config_file(path: Path) -> Path:
    path.write_text(
        "llm_model:\n  model: gpt-4.1-nano\n  model_provider: openai\nagent_config:\n  harbor_lifecycle:\n    enable_metrics: false\n"
    )
    return path


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _checkpoint(trial: Path) -> Path:
    path = trial / "agent/db/checkpointer.db"
    checkpointer = Checkpointer.from_workspace(trial / "agent")
    checkpointer.put(
        {"configurable": {"thread_id": "trial-thread", "checkpoint_ns": ""}},
        empty_checkpoint(),
        {},
        {},
    )
    checkpointer.conn.close()
    return path


@pytest.fixture
def cli(monkeypatch):
    monkeypatch.setattr(
        harbor_integration.shutil, "which", lambda command: f"/bin/{command}"
    )


def test_native_config_records_evaluation_group(tmp_path):
    config = _native_config_file(tmp_path / "ursa.yaml")
    config.write_text(config.read_text() + "group: science\n")
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent",
        environment_logs_dir=tmp_path / "agent",
        config=config,
    )
    runtime, _ = agent._materialize_native_config()
    assert runtime["group"] == agent._runtime_group == "science"


@pytest.mark.parametrize("inline", [False, True])
def test_native_config_accepts_file_and_inline_sources(tmp_path, inline):
    native = {
        "llm_model": {
            "model": "gpt-4.1-nano",
            "model_provider": "openai",
        }
    }
    source: Path | dict[str, Any]
    if inline:
        source = native
    else:
        source = _native_config_file(tmp_path / "ursa.yaml")
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent",
        environment_logs_dir=tmp_path / "agent",
        config=source,
    )

    runtime, _ = agent._materialize_native_config()

    assert agent.capabilities.native_config
    assert runtime["llm_model"]["model"] == "gpt-4.1-nano"


@pytest.mark.parametrize(
    ("native_config", "provider"),
    [
        (
            {
                "inference_providers": {
                    "research-gateway": {
                        "model_provider": "openai",
                        "api_key": {"keyring": True},
                    }
                },
                "llm_model": {
                    "model": "gpt-4.1-nano",
                    "inference_provider": "research-gateway",
                },
            },
            "research-gateway",
        ),
        (
            {
                "llm_model": {
                    "model": "anthropic:claude",
                    "api_key": {"keyring": True},
                }
            },
            "openai",
        ),
    ],
)
def test_native_config_keyring_uses_inference_provider(
    tmp_path, monkeypatch, native_config, provider
):
    calls = []
    monkeypatch.setattr(
        "ursa.util.secrets.get_keyring",
        lambda: SimpleNamespace(
            get_password=lambda service, username: (
                calls.append((service, username)) or "key"
            )
        ),
    )
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent",
        environment_logs_dir=tmp_path / "agent",
        config=native_config,
    )

    _, secret_env = agent._materialize_native_config()

    assert calls == [("ursa", provider)]
    assert secret_env == {"URSA_HARBOR_SECRET_0": "key"}


@pytest.mark.parametrize(
    "source_kind",
    ["implicit", "project", "wheel", "sdist", "version", "git", "url"],
)
@pytest.mark.asyncio
async def test_install_preserves_requirement_and_uploads_local_sources(
    tmp_path, monkeypatch, source_kind
):
    config = _native_config_file(tmp_path / "ursa.yaml")
    checkout = tmp_path / "checkout with spaces"
    (checkout / "pyproject.toml").parent.mkdir()
    (checkout / "pyproject.toml").write_text("[project]\nname = 'ursa-ai'\n")
    wheel = tmp_path / "ursa_ai-1.0.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    sdist = tmp_path / "ursa_ai-1.0.0.tar.gz"
    sdist.write_bytes(b"sdist")
    targets = {
        "implicit": f" @ {checkout.as_uri()}",
        "project": f" @ {checkout.as_uri()}",
        "wheel": f" @ {wheel.as_uri()}#sha256=abc",
        "sdist": f" @ {sdist.as_uri()}",
        "version": ">=0.17.3,<1",
        "git": " @ git+https://github.com/lanl/ursa.git@feature/harbor",
        "url": " @ https://example.org/ursa.whl#sha256=abc",
    }
    spec = f'ursa-ai[office_readers]{targets[source_kind]} ; python_version >= "3.12"'
    monkeypatch.setattr(
        UrsaHarborAgent, "_default_install_spec", staticmethod(lambda: spec)
    )
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent",
        environment_logs_dir=tmp_path / "agent",
        config=config,
        ursa_install_spec=None if source_kind == "implicit" else spec,
        ursa_extras="image,harbor",
    )
    built: list[Path] = []

    def build(source: Path, destination: Path) -> Path:
        assert source == checkout
        wheel = destination / "ursa_ai-1.0.0-py3-none-any.whl"
        wheel.write_bytes(b"wheel")
        built.append(wheel)
        return wheel

    class Environment:
        def __init__(self) -> None:
            self.files: list[tuple[Path, str]] = []
            self.directories: list[str] = []

        async def upload_file(self, source: Path, destination: str) -> None:
            self.files.append((source, destination))

        async def upload_dir(self, source: Path, destination: str) -> None:
            assert (source / "pyproject.toml").read_text() == (
                checkout / "pyproject.toml"
            ).read_text()
            self.directories.append(destination)

    environment = Environment()
    monkeypatch.setattr(
        agent, "_materialize_native_config", lambda **_kwargs: ({}, {})
    )
    monkeypatch.setattr(agent, "_network_use_web", lambda _env: None)
    monkeypatch.setattr(
        agent,
        "exec_as_agent",
        AsyncMock(return_value=SimpleNamespace(stdout="/")),
    )
    root = AsyncMock(return_value=SimpleNamespace(stdout="x86_64"))
    monkeypatch.setattr(agent, "exec_as_root", root)
    monkeypatch.setattr(agent, "_upload_native_config", AsyncMock())
    monkeypatch.setattr(agent, "_build_wheel", build)
    monkeypatch.setattr(
        agent,
        "_stage_uv_binary",
        lambda _machine, destination: destination.write_bytes(b"uv"),
    )

    await agent.install(environment)

    tokens = shlex.split(root.call_args_list[-1].kwargs["command"])
    installed = Requirement(tokens[tokens.index("&&") - 1])
    original = Requirement(spec)
    assert installed.extras == {"harbor", "image", "office-readers"}
    assert installed.marker == original.marker
    assert installed.specifier == original.specifier
    assert str(agent.ursa_install_spec) == str(original)
    assert len(built) == (source_kind == "implicit")
    assert environment.directories == (
        ["/installed-agent/tmp/ursa-source"] if source_kind == "project" else []
    )
    if source_kind in {"implicit", "wheel", "sdist"}:
        filename = sdist.name if source_kind == "sdist" else wheel.name
        destination = f"/installed-agent/tmp/{filename}"
        fragment = "#sha256=abc" if source_kind == "wheel" else ""
        assert installed.url == f"file://{destination}{fragment}"
        assert any(remote == destination for _, remote in environment.files)
    elif source_kind == "project":
        assert installed.url == "file:///installed-agent/tmp/ursa-source"
    elif source_kind == "git":
        assert installed.url == (
            "https://github.com/lanl/ursa/archive/feature%2Fharbor.tar.gz"
        )
    else:
        assert installed.url == original.url


@pytest.mark.parametrize(
    "value",
    [
        "",
        "./checkout",
        "/tmp/ursa.whl",
        "git+https://github.com/lanl/ursa.git",
        "ursa-ai=>1",
        Path("checkout"),
    ],
)
def test_install_spec_requires_pep508(value):
    with pytest.raises(ValueError, match="PEP 508"):
        UrsaHarborAgent._parse_install_spec(value)


def test_install_spec_requires_ursa_package():
    with pytest.raises(ValueError, match="ursa-ai package"):
        UrsaHarborAgent._parse_install_spec("another-package>=1")


def test_install_spec_validates_local_sources(tmp_path):
    missing = tmp_path / "missing.whl"
    with pytest.raises(ValueError, match="does not exist"):
        UrsaHarborAgent._local_install_source(
            Requirement(f"ursa-ai @ {missing.as_uri()}")
        )
    with pytest.raises(ValueError, match="not a Python project"):
        UrsaHarborAgent._local_install_source(
            Requirement(f"ursa-ai @ {tmp_path.as_uri()}")
        )
    for url in ("file:relative/path", "file://remote-host/path"):
        with pytest.raises(ValueError, match="absolute local file URL"):
            UrsaHarborAgent._local_install_source(
                Requirement(f"ursa-ai @ {url}")
            )


def _provider_config(path: Path) -> Path:
    path.write_text(
        "llm_model:\n"
        "  model: gpt-4.1-nano\n"
        "  model_provider: openai\n"
        "inference_providers:\n"
        "  research:\n"
        "    model_provider: openai\n"
        "  azure:\n"
        "    model_provider: azure_openai\n"
        "  bedrock:\n"
        "    model_provider: bedrock_converse\n"
    )
    return path


@pytest.mark.parametrize(
    ("provider", "key_name", "custom_key_name"),
    [
        ("openai", "OPENAI_API_KEY", False),
        ("openai", "GATEWAY_TOKEN", True),
        # Harbor keeps a custom key name for a provider it does not know.
        ("research", "GATEWAY_TOKEN", True),
    ],
)
def test_model_connection_credentials_stay_out_of_config(
    tmp_path, monkeypatch, provider, key_name, custom_key_name
):
    monkeypatch.setenv(key_name, "sk-host-secret")
    monkeypatch.setenv(
        f"{provider.upper()}_BASE_URL", "https://gateway.example/v1"
    )
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent",
        model_name=f"{provider}/gpt-4.1-nano",
        config=_provider_config(tmp_path / "ursa.yaml"),
        model_api_key_name=key_name if custom_key_name else None,
    )

    native_config, secret_env = agent._materialize_native_config()

    provider_config = native_config["inference_providers"][provider]
    assert provider_config["base_url"] == "https://gateway.example/v1"
    assert secret_env == {provider_config["api_key"]["env"]: "sk-host-secret"}
    assert "sk-host-secret" not in json.dumps(native_config)
    # The key travels only as a secret reference, never as a plain variable.
    assert agent._model_env == {}


def test_custom_inference_provider_keeps_its_base_url(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_BASE_URL", "https://research.example/v1")
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent",
        model_name="research/gpt-4.1-nano",
        config=_provider_config(tmp_path / "ursa.yaml"),
    )

    native_config, _ = agent._materialize_native_config()

    # LiteLLM does not know "research"; URSA's provider/model split applies.
    assert agent.model_connection.provider == "research"
    assert agent.to_agent_info().model_info.provider == "research"
    assert native_config["llm_model"] == {
        "model": "gpt-4.1-nano",
        "model_provider": "openai",
        "inference_provider": "research",
    }
    assert native_config["inference_providers"]["research"]["base_url"] == (
        "https://research.example/v1"
    )


def test_model_connection_forwards_provider_extra_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_API_KEY", "azure-secret")
    monkeypatch.setenv("AZURE_API_VERSION", "2024-10-21")
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent",
        model_name="azure/gpt-4.1-nano",
        config=_provider_config(tmp_path / "ursa.yaml"),
    )

    native_config, secret_env = agent._materialize_native_config()

    provider = native_config["inference_providers"]["azure"]
    assert secret_env == {provider["api_key"]["env"]: "azure-secret"}
    assert agent._model_env == {"AZURE_API_VERSION": "2024-10-21"}


def test_env_authenticated_provider_keeps_credentials_in_env(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-token")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent",
        model_name="bedrock/anthropic.claude-sonnet",
        config=_provider_config(tmp_path / "ursa.yaml"),
    )

    native_config, secret_env = agent._materialize_native_config()

    assert agent.model_connection.provider == "amazon-bedrock"
    assert "api_key" not in native_config["inference_providers"]["bedrock"]
    assert secret_env == {}
    assert agent._model_env == {
        "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
        "AWS_REGION": "us-west-2",
    }


@pytest.mark.parametrize(
    ("network_mode", "use_web"),
    [
        (NetworkMode.PUBLIC, True),
        (NetworkMode.ALLOWLIST, False),
        (NetworkMode.NO_NETWORK, False),
    ],
)
def test_web_tools_require_public_network(network_mode, use_web):
    environment = SimpleNamespace(
        network_policy=NetworkPolicy(
            network_mode=network_mode,
            allowed_hosts=(
                ["api.openai.com"]
                if network_mode == NetworkMode.ALLOWLIST
                else []
            ),
        )
    )

    assert UrsaHarborAgent._network_use_web(environment) is use_web
    assert UrsaHarborAgent._network_use_web(object()) is None


@pytest.mark.parametrize(
    ("configured", "network_mode", "override", "expected"),
    [
        (None, NetworkMode.PUBLIC, True, True),
        # An explicit setting is left to the config...
        ("false", NetworkMode.PUBLIC, None, False),
        ("true", NetworkMode.PUBLIC, None, True),
        # ...but a restricted network always turns web tools off.
        ("true", NetworkMode.NO_NETWORK, False, False),
        ("true", NetworkMode.ALLOWLIST, False, False),
        (None, NetworkMode.NO_NETWORK, False, False),
    ],
)
def test_public_network_enables_web_tools_only_when_not_configured(
    tmp_path, configured, network_mode, override, expected
):
    config_file = _native_config_file(tmp_path / "ursa.yaml")
    if configured is not None:
        config_file.write_text(
            config_file.read_text() + f"use_web: {configured}\n"
        )
    agent = UrsaHarborAgent(logs_dir=tmp_path / "agent", config=config_file)
    environment = SimpleNamespace(
        network_policy=NetworkPolicy(network_mode=network_mode)
    )

    use_web = agent._phase_use_web(agent._network_use_web(environment))
    native_config, _ = agent._materialize_native_config(use_web=use_web)

    assert use_web is override
    assert native_config.get("use_web", False) is expected


@pytest.mark.parametrize(
    ("disable_web_search", "network_mode", "expected"),
    [
        (True, NetworkMode.PUBLIC, False),
        ("true", NetworkMode.PUBLIC, False),
        # An explicit opt-in replaces the network decision, as in Harbor.
        (False, NetworkMode.NO_NETWORK, True),
        ("false", NetworkMode.ALLOWLIST, True),
    ],
)
def test_disable_web_search_kwarg_overrides_network_decision(
    tmp_path, disable_web_search, network_mode, expected
):
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent",
        config=_native_config_file(tmp_path / "ursa.yaml"),
        disable_web_search=disable_web_search,
    )
    environment = SimpleNamespace(
        network_policy=NetworkPolicy(network_mode=network_mode)
    )

    assert agent._phase_use_web(agent._network_use_web(environment)) is (
        expected
    )
    assert agent._phase_use_web(None) is expected


@pytest.mark.parametrize(
    ("configured", "use_web", "expected"),
    [
        (None, True, True),
        (False, True, False),
        (True, False, False),
        (False, None, False),
    ],
)
def test_runner_turns_web_tools_on_only_for_unconfigured_agents(
    configured, use_web, expected
):
    class WebAgent:
        def __init__(self, use_web: bool = False):
            self.use_web = use_web

    options = {} if configured is None else {"use_web": configured}
    config = SimpleNamespace(agent_config={"web": options})

    selected = _agent_config(config, WebAgent, use_web=use_web)

    assert selected.get("use_web", False) is expected


def _recording_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: Path,
    **kwargs: Any,
) -> tuple[UrsaHarborAgent, list[Any], list[dict[str, Any]]]:
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent", config=config, **kwargs
    )
    agent._remote_config_file = "/installed-agent/tmp/ursa-config.json"
    uploaded: list[Any] = []
    payloads: list[dict[str, Any]] = []

    async def upload(environment: Any, native_config: dict[str, Any]) -> None:
        uploaded.append(native_config.get("use_web"))
        agent._runtime_proxy = getattr(environment, "_proxy_stack", None)

    async def run(_environment: Any, **kwargs: Any) -> Any:
        encoded = shlex.split(kwargs["command"])[-1]
        payloads.append(json.loads(base64.urlsafe_b64decode(encoded)))
        return SimpleNamespace(stdout=None)

    monkeypatch.setattr(agent, "_upload_native_config", upload)
    monkeypatch.setattr(agent, "exec_as_agent", run)
    monkeypatch.setattr(agent, "_best_effort_root_exec", AsyncMock())
    return agent, uploaded, payloads


class _PhaseEnvironment(SimpleNamespace):
    def is_path_mounted(self, _environment_path: Any, _host_path: Path) -> bool:
        return True


@pytest.mark.asyncio
async def test_harbor_web_tool_decision_updates_native_config(
    tmp_path, monkeypatch
):
    agent, uploaded, payloads = _recording_agent(
        tmp_path, monkeypatch, _native_config_file(tmp_path / "ursa.yaml")
    )
    agent._runtime_use_web = True
    agent._runtime_proxy = None
    environment = _PhaseEnvironment(
        network_policy=NetworkPolicy(network_mode=NetworkMode.PUBLIC)
    )

    assert UrsaHarborAgent.capabilities.native_web_tools is True
    await agent.set_web_tools_disabled(True, environment)
    await agent.run("solve it", environment, AgentContext())
    await agent.run("solve it again", environment, AgentContext())

    assert uploaded == [False]
    assert [payload["use_web"] for payload in payloads] == [False, False]


@pytest.mark.asyncio
async def test_configured_use_web_returns_after_a_forced_off_phase(
    tmp_path, monkeypatch
):
    config_file = _native_config_file(tmp_path / "ursa.yaml")
    config_file.write_text(config_file.read_text() + "use_web: true\n")
    agent, uploaded, payloads = _recording_agent(
        tmp_path, monkeypatch, config_file
    )
    # A no-network setup phase turned web tools off.
    agent._runtime_use_web = False
    agent._runtime_proxy = None
    environment = _PhaseEnvironment(
        network_policy=NetworkPolicy(network_mode=NetworkMode.PUBLIC)
    )

    await agent.run("solve it", environment, AgentContext())

    # None removes the override, so the configured use_web applies again.
    assert uploaded == [True]
    assert [payload["use_web"] for payload in payloads] == [None]


@pytest.mark.asyncio
async def test_web_and_proxy_change_upload_the_config_once(
    tmp_path, monkeypatch
):
    agent, uploaded, _ = _recording_agent(
        tmp_path, monkeypatch, _native_config_file(tmp_path / "ursa.yaml")
    )
    agent._runtime_use_web = True
    agent._runtime_proxy = None
    environment = _PhaseEnvironment(
        network_policy=NetworkPolicy(network_mode=NetworkMode.PUBLIC),
        _proxy_stack=object(),
    )

    await agent.set_web_tools_disabled(True, environment)
    await agent.run("solve it", environment, AgentContext())
    await agent.run("solve it again", environment, AgentContext())

    assert uploaded == [False]


def test_mcp_servers_are_forwarded_without_headers(tmp_path):
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent",
        config=_native_config_file(tmp_path / "ursa.yaml"),
        mcp_servers=[
            MCPServerConfig(name="docs", url="http://mcp.example/sse")
        ],
    )

    assert UrsaHarborAgent.capabilities.mcp_servers is True
    assert UrsaHarborAgent.capabilities.mcp_http_headers is False
    assert agent._mcp_config() == {
        "docs": {
            "transport": "sse",
            "url": "http://mcp.example/sse",
            "args": [],
        }
    }

    agent.mcp_servers = [
        MCPServerConfig(
            name="docs",
            transport="streamable-http",
            url="http://mcp.example/mcp",
            headers={"Authorization": "Bearer ${MCP_TOKEN}"},
        )
    ]
    with pytest.raises(ValueError, match="does not forward MCP HTTP headers"):
        agent._mcp_config()


def test_skills_are_not_declared():
    # Harbor 0.25 rejects tasks with skills_dir or --skills for this adapter.
    assert UrsaHarborAgent.capabilities.skills is False


def _usage_record(input_tokens: int, cached: int, output: int) -> str:
    return (
        json.dumps({
            "type": "chat",
            "event": "end",
            "usage": {
                "input_tokens": input_tokens,
                "cached_tokens": cached,
                "output_tokens": output,
            },
        })
        + "\n"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mounted", "download", "expected"),
    [
        (False, "ok", (3, 1, 2)),
        (False, "error", (None, None, None)),
        (False, "hang", (None, None, None)),
        (True, "unexpected", (5, 0, 4)),
    ],
)
async def test_run_reads_token_usage_without_copying_log_to_host(
    tmp_path, monkeypatch, mounted, download, expected
):
    logs = tmp_path / "agent"
    agent = UrsaHarborAgent(
        logs_dir=logs, config=_native_config_file(tmp_path / "ursa.yaml")
    )
    agent._remote_config_file = "/installed-agent/tmp/ursa-config.json"
    if mounted:
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "ursa.jsonl").write_text(_usage_record(5, 0, 4))
    downloads: list[str] = []

    class Environment:
        network_policy = None

        def is_path_mounted(self, environment_path: Any, host_path: Path):
            assert str(environment_path) == "/logs/agent"
            assert host_path == logs
            return mounted

        async def download_file(self, source: str, target: Path) -> None:
            downloads.append(source)
            assert not target.is_relative_to(logs)
            if download == "ok":
                target.write_text(_usage_record(3, 1, 2))
            elif download == "hang":
                await asyncio.Event().wait()
            else:
                raise RuntimeError("download failed")

    async def run(_environment: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(stdout=None)

    wait_for = asyncio.wait_for
    timeouts: list[float] = []

    async def short_wait_for(awaitable: Any, timeout: float) -> Any:
        timeouts.append(timeout)
        return await wait_for(awaitable, timeout=0.01)

    monkeypatch.setattr(agent, "exec_as_agent", run)
    monkeypatch.setattr(agent, "_best_effort_root_exec", AsyncMock())
    monkeypatch.setattr(harbor_integration.asyncio, "wait_for", short_wait_for)
    context = AgentContext()

    await agent.run("solve it", Environment(), context)

    assert downloads == ([] if mounted else ["/logs/agent/ursa.jsonl"])
    assert timeouts == ([] if mounted else [10])
    # Harbor's own log sync decides whether ursa.jsonl reaches the host.
    assert (logs / "ursa.jsonl").exists() is mounted
    assert (
        context.n_input_tokens,
        context.n_cache_tokens,
        context.n_output_tokens,
    ) == expected


def test_handoff_imports_checkpoint_as_local_named_agent(
    tmp_path, monkeypatch, cli
):
    cache, trial = tmp_path / "cache", tmp_path / "install pytorch__abc123"
    monkeypatch.setattr(agent_management, "URSA_CACHE_DIR", cache)
    (cache / "science").mkdir(parents=True)
    source = _checkpoint(trial)
    (trial / "result.json").write_text(
        json.dumps({"agent_result": {"metadata": {"group": "science"}}})
    )
    assert UrsaHarborAgent.handoff(trial, tmp_path / "workspace") == [
        "ursa",
        "--name",
        "harbor-install-pytorch__abc123",
        "--group",
        "science",
    ]
    copied = (
        cache
        / "science/agents/harbor-install-pytorch__abc123/db/checkpointer.db"
    )
    with sqlite3.connect(copied) as db:
        assert db.execute(
            "SELECT DISTINCT thread_id FROM checkpoints"
        ).fetchall() == [("ursa",)]
    with sqlite3.connect(source) as db:
        assert db.execute(
            "SELECT DISTINCT thread_id FROM checkpoints"
        ).fetchall() == [("trial-thread",)]


def test_handoff_validates_cli_checkpoint_group_and_destination(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        harbor_integration.shutil, "which", lambda _command: None
    )
    with pytest.raises(ValueError, match="ursa CLI not found"):
        UrsaHarborAgent.handoff(tmp_path / "trial", tmp_path)
    monkeypatch.setattr(
        harbor_integration.shutil, "which", lambda command: f"/bin/{command}"
    )
    with pytest.raises(ValueError, match="URSA checkpoint not found"):
        UrsaHarborAgent.handoff(tmp_path / "trial", tmp_path)
    trial = tmp_path / "trial"
    monkeypatch.setattr(agent_management, "URSA_CACHE_DIR", tmp_path / "cache")
    _checkpoint(trial)
    with pytest.raises(
        ValueError, match="does not record the URSA evaluation group"
    ):
        UrsaHarborAgent.handoff(trial, tmp_path)
    (trial / "result.json").write_text(
        json.dumps({"agent_result": {"metadata": {"group": "default"}}})
    )
    UrsaHarborAgent.handoff(trial, tmp_path)
    with pytest.raises(ValueError, match="Destination agent already exists"):
        UrsaHarborAgent.handoff(trial, tmp_path)


def _lifecycle_agent(tmp_path):
    logs, workspace, config = (
        tmp_path / "agent",
        tmp_path / "workspace",
        _native_config_file(tmp_path / "ursa.yaml"),
    )
    workspace.mkdir()
    agent = UrsaHarborAgent(
        logs_dir=logs,
        environment_logs_dir=logs,
        model_name="openai/gpt-4.1-nano",
        config=config,
        agent_import_path=f"{__name__}:HarborLifecycleAgent",
    )
    agent._remote_config_file, agent._workspace = str(config), str(workspace)
    return agent, logs, workspace


async def _local_runner(agent, *, check_payload=False):
    async def run(_environment, **kwargs):
        if check_payload:
            assert (
                kwargs["timeout_sec"] is None
                and kwargs["cwd"] == agent._workspace
            )
        payload = json.loads(
            base64.urlsafe_b64decode(
                shlex.split(kwargs["command"])[-1]
            ).decode()
        )
        if check_payload:
            assert (
                payload["log_dir"] == str(agent.logs_dir)
                and payload["workspace"] == agent._workspace
            )
        await _runner_run(payload)
        return SimpleNamespace(stdout=None)

    return run


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("instruction", "error"),
    [
        ("solve it", None),
        ("fail after usage", "agent failed after chat"),
    ],
)
async def test_adapter_runner_preserves_usage(
    tmp_path, monkeypatch, instruction, error
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent, logs, _ = _lifecycle_agent(tmp_path)
    monkeypatch.setattr(
        agent, "exec_as_agent", await _local_runner(agent, check_payload=True)
    )
    context = AgentContext()
    if error:
        with pytest.raises(RuntimeError, match=error):
            await agent.run(instruction, object(), context)
    else:
        await agent.run(instruction, object(), context)
    assert (
        context.n_input_tokens,
        context.n_cache_tokens,
        context.n_output_tokens,
    ) == (12, 6, 7)
    result = logs / "ursa_result.out"
    if error:
        assert not result.exists()
        return
    assert context.metadata["group"] == "default"
    assert (
        result.read_text() == "runner answer"
        and "Inspecting the workspace" in (logs / "ursa.log").read_text()
    )
    records = _records(logs / "ursa.jsonl")
    events = {(record["type"], record["event"]) for record in records}
    assert {
        ("chain", "start"),
        ("agent", "progress"),
        ("chat", "end"),
        ("tool", "end"),
        ("chain", "end"),
    } <= events
    assert next(
        record
        for record in records
        if record["type"] == "chat" and record["event"] == "end"
    )["usage"] == {
        "input_tokens": 12,
        "output_tokens": 7,
        "reasoning_tokens": 4,
        "cached_tokens": 6,
    }
    assert [record["timestamp_monotonic_ns"] for record in records] == sorted(
        record["timestamp_monotonic_ns"] for record in records
    )
    with sqlite3.connect(logs / "db/checkpointer.db") as db:
        assert db.execute("PRAGMA integrity_check").fetchone() == (
            "ok",
        ) and db.execute("SELECT value FROM runner_probe").fetchall() == [
            ("persisted",)
        ]


@pytest.mark.asyncio
async def test_runner_closes_setup_connection_on_agent_setup_failure(
    tmp_path, monkeypatch
):
    SetupFailingAgent.connection = None
    with pytest.raises(RuntimeError, match="agent setup failed"):
        await _runner_run({
            "agent_import_path": f"{__name__}:SetupFailingAgent",
            "config_file": str(_native_config_file(tmp_path / "bad.yaml")),
            "workspace": str(tmp_path / "workspace"),
            "log_dir": str(tmp_path / "failed"),
            "instruction": "solve it",
        })
    with pytest.raises(ValueError, match="no active connection"):
        await SetupFailingAgent.connection.execute("SELECT 1")


def test_jsonl_token_usage_ignores_invalid_records(tmp_path):
    path = tmp_path / "ursa.jsonl"
    path.write_bytes(
        b'\xffnot-json\n{"type":"chat","event":"end","usage":{}}\n{"type":"chat","event":"end","usage":{"input_tokens":true,"cached_tokens":-1,"output_tokens":1.5}}\n'
    )
    assert _jsonl_token_usage(path) == (None, None, None)
    path.write_text(
        path.read_text(errors="ignore")
        + '{"type":"chat","event":"end","usage":{"input_tokens":9007199254740993}}\n'
    )
    assert _jsonl_token_usage(path) == (9007199254740993, None, None)
