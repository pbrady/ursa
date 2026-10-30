# ruff: noqa: E402

import asyncio
import base64
import json
import shlex
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

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
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.base import empty_checkpoint

from ursa.agents import BaseAgent
from ursa.cli import agent_management
from ursa.integrations import harbor as harbor_integration
from ursa.integrations.harbor import UrsaHarborAgent, _jsonl_token_usage
from ursa.integrations.harbor_runner import _agent_config
from ursa.integrations.harbor_runner import _run as _runner_run
from ursa.util import Checkpointer
from ursa.util.events import AgentEvents


@tool
async def runner_echo(text: str) -> dict[str, str]:
    """Echo text for the runner integration test."""
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
        runnable_config = self.build_config(**config)

        async def lifecycle(
            _inputs: Any, config: RunnableConfig
        ) -> dict[str, Any]:
            await AgentEvents(agent="ExecutionAgent", config=config).aemit(
                "Inspecting the workspace",
                stage="work",
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

        runnable = RunnableLambda(lifecycle).with_config({
            "run_name": "runner_lifecycle"
        })
        return await runnable.ainvoke(inputs, config=runnable_config)

    def format_result(self, result: dict[str, Any]) -> str:
        return str(result["answer"])


class SetupFailingAgent(BaseAgent):
    connection: ClassVar[Any | None] = None

    def __init__(self, *, checkpointer: Any, **kwargs: Any) -> None:
        type(self).connection = checkpointer.conn
        raise RuntimeError("agent setup failed")

    def _build_graph(self) -> None:
        pass


def _runtime_config(path: Path) -> Path:
    path.write_text(
        "llm_model:\n"
        "  model: gpt-4.1-nano\n"
        "  model_provider: openai\n"
        "agent_config:\n"
        "  harbor_lifecycle:\n"
        "    enable_metrics: false\n"
    )
    return path


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_runtime_config_records_evaluation_group(tmp_path):
    config_file = _runtime_config(tmp_path / "ursa.yaml")
    config_file.write_text(config_file.read_text() + "group: science\n")
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent",
        environment_logs_dir=tmp_path / "agent",
        config_file=config_file,
    )

    runtime_config, _ = agent._runtime_config()

    assert runtime_config["group"] == "science"
    assert agent._runtime_group == "science"


def _provider_config(path: Path) -> Path:
    path.write_text(
        "llm_model:\n"
        "  model: gpt-4.1-nano\n"
        "  model_provider: openai\n"
        "inference_providers:\n"
        "  lanl:\n"
        "    model_provider: openai\n"
        "  azure:\n"
        "    model_provider: azure_openai\n"
        "  bedrock:\n"
        "    model_provider: bedrock_converse\n"
    )
    return path


def test_model_connection_credentials_stay_out_of_config(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-host-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent",
        model_name="openai/gpt-4.1-nano",
        config_file=_provider_config(tmp_path / "ursa.yaml"),
    )

    runtime_config, secret_env = agent._runtime_config()

    assert UrsaHarborAgent.capabilities.base_url is False
    provider = runtime_config["inference_providers"]["openai"]
    assert provider["base_url"] == "https://gateway.example/v1"
    assert secret_env == {provider["api_key"]["env"]: "sk-host-secret"}
    assert "sk-host-secret" not in json.dumps(runtime_config)
    assert agent._model_env == {}


def test_custom_inference_provider_keeps_its_base_url(tmp_path, monkeypatch):
    monkeypatch.setenv("LANL_BASE_URL", "https://lanl.example/v1")
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent",
        model_name="lanl/gpt-4.1-nano",
        config_file=_provider_config(tmp_path / "ursa.yaml"),
    )

    runtime_config, _ = agent._runtime_config()

    # LiteLLM does not know "lanl"; URSA's provider/model split still applies.
    assert agent.model_connection.provider == "lanl"
    assert agent.to_agent_info().model_info.provider == "lanl"
    assert runtime_config["llm_model"] == {
        "model": "gpt-4.1-nano",
        "model_provider": "openai",
        "inference_provider": "lanl",
    }
    assert runtime_config["inference_providers"]["lanl"]["base_url"] == (
        "https://lanl.example/v1"
    )


def test_model_connection_forwards_provider_extra_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_API_KEY", "azure-secret")
    monkeypatch.setenv("AZURE_API_VERSION", "2024-10-21")
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent",
        model_name="azure/gpt-4.1-nano",
        config_file=_provider_config(tmp_path / "ursa.yaml"),
    )

    runtime_config, secret_env = agent._runtime_config()

    provider = runtime_config["inference_providers"]["azure"]
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
        config_file=_provider_config(tmp_path / "ursa.yaml"),
    )

    runtime_config, secret_env = agent._runtime_config()

    assert agent.model_connection.provider == "amazon-bedrock"
    assert "api_key" not in runtime_config["inference_providers"]["bedrock"]
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
        # ...but no network always turns web tools off.
        ("true", NetworkMode.NO_NETWORK, False, False),
        (None, NetworkMode.NO_NETWORK, False, False),
    ],
)
def test_public_network_enables_web_tools_only_when_not_configured(
    tmp_path, configured, network_mode, override, expected
):
    config_file = _runtime_config(tmp_path / "ursa.yaml")
    if configured is not None:
        config_file.write_text(
            config_file.read_text() + f"use_web: {configured}\n"
        )
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent", config_file=config_file
    )
    environment = SimpleNamespace(
        network_policy=NetworkPolicy(network_mode=network_mode)
    )

    use_web = agent._phase_use_web(agent._network_use_web(environment))
    runtime_config, _ = agent._runtime_config(use_web=use_web)

    assert use_web is override
    assert runtime_config.get("use_web", False) is expected


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


@pytest.mark.asyncio
async def test_harbor_web_tool_decision_updates_runtime_config(
    tmp_path, monkeypatch
):
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent",
        config_file=_runtime_config(tmp_path / "ursa.yaml"),
    )
    agent._remote_config_file = "/installed-agent/tmp/ursa-config.json"
    agent._runtime_use_web = True
    uploaded: list[bool] = []
    payloads: list[dict[str, Any]] = []

    async def upload(_environment: Any, runtime_config: dict[str, Any]):
        uploaded.append(runtime_config["use_web"])

    async def run(_environment: Any, **kwargs: Any) -> Any:
        encoded = shlex.split(kwargs["command"])[-1]
        payloads.append(json.loads(base64.urlsafe_b64decode(encoded)))
        return SimpleNamespace(stdout=None)

    monkeypatch.setattr(agent, "_upload_runtime_config", upload)
    monkeypatch.setattr(agent, "exec_as_agent", run)
    environment = SimpleNamespace(
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
    config_file = _runtime_config(tmp_path / "ursa.yaml")
    config_file.write_text(config_file.read_text() + "use_web: true\n")
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent", config_file=config_file
    )
    agent._remote_config_file = "/installed-agent/tmp/ursa-config.json"
    # A no-network setup phase turned web tools off.
    agent._runtime_use_web = False
    uploaded: list[bool] = []
    payloads: list[dict[str, Any]] = []

    async def upload(_environment: Any, runtime_config: dict[str, Any]):
        uploaded.append(runtime_config["use_web"])

    async def run(_environment: Any, **kwargs: Any) -> Any:
        encoded = shlex.split(kwargs["command"])[-1]
        payloads.append(json.loads(base64.urlsafe_b64decode(encoded)))
        return SimpleNamespace(stdout=None)

    monkeypatch.setattr(agent, "_upload_runtime_config", upload)
    monkeypatch.setattr(agent, "exec_as_agent", run)
    environment = SimpleNamespace(
        network_policy=NetworkPolicy(network_mode=NetworkMode.PUBLIC)
    )

    await agent.run("solve it", environment, AgentContext())

    assert uploaded == [True]
    assert [payload["use_web"] for payload in payloads] == [None]


def test_mcp_servers_are_forwarded_without_headers(tmp_path):
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent",
        config_file=_runtime_config(tmp_path / "ursa.yaml"),
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


def test_skills_are_not_declared():
    # Harbor 0.25 rejects tasks with skills_dir or --skills for this adapter.
    assert UrsaHarborAgent.capabilities.skills is False


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
        logs_dir=logs,
        config_file=_runtime_config(tmp_path / "ursa.yaml"),
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


def test_handoff_imports_checkpoint_as_local_named_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(
        harbor_integration.shutil, "which", lambda command: f"/bin/{command}"
    )
    cache = tmp_path / "cache"
    monkeypatch.setattr(agent_management, "URSA_CACHE_DIR", cache)
    trial = tmp_path / "install pytorch__abc123"
    (cache / "science").mkdir(parents=True)
    checkpoint = trial / "agent" / "db" / "checkpointer.db"
    checkpointer = Checkpointer.from_workspace(trial / "agent")
    checkpointer.put(
        {"configurable": {"thread_id": "trial-thread", "checkpoint_ns": ""}},
        empty_checkpoint(),
        {},
        {},
    )
    checkpointer.conn.close()
    (trial / "result.json").write_text(
        json.dumps({"agent_result": {"metadata": {"group": "science"}}})
    )

    command = UrsaHarborAgent.handoff(trial, tmp_path / "workspace")

    assert UrsaHarborAgent.capabilities.handoff is True
    assert command == [
        "ursa",
        "--name",
        "harbor-install-pytorch__abc123",
        "--group",
        "science",
    ]
    imported = (
        cache
        / "science"
        / "agents"
        / "harbor-install-pytorch__abc123"
        / "db"
        / "checkpointer.db"
    )
    with sqlite3.connect(imported) as database:
        assert database.execute(
            "SELECT DISTINCT thread_id FROM checkpoints"
        ).fetchall() == [("ursa",)]
    with sqlite3.connect(checkpoint) as database:
        assert database.execute(
            "SELECT DISTINCT thread_id FROM checkpoints"
        ).fetchall() == [("trial-thread",)]


def test_handoff_rejects_missing_cli_or_checkpoint(tmp_path, monkeypatch):
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


def test_handoff_requires_group_and_does_not_overwrite_existing_agent(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        harbor_integration.shutil, "which", lambda command: f"/bin/{command}"
    )
    monkeypatch.setattr(agent_management, "URSA_CACHE_DIR", tmp_path / "cache")
    trial = tmp_path / "trial"
    checkpointer = Checkpointer.from_workspace(trial / "agent")
    checkpointer.put(
        {"configurable": {"thread_id": "trial-thread", "checkpoint_ns": ""}},
        empty_checkpoint(),
        {},
        {},
    )
    checkpointer.conn.close()

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


@pytest.mark.asyncio
async def test_harbor_adapter_runs_real_runner_and_populates_context(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    logs = tmp_path / "agent"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_file = _runtime_config(tmp_path / "ursa.yaml")
    agent = UrsaHarborAgent(
        logs_dir=logs,
        environment_logs_dir=logs,
        model_name="openai/gpt-4.1-nano",
        config_file=config_file,
        agent_import_path=f"{__name__}:HarborLifecycleAgent",
    )
    agent._remote_config_file = str(config_file)
    agent._workspace = str(workspace)

    async def run_locally(_environment: Any, **kwargs: Any) -> Any:
        assert kwargs["timeout_sec"] is None
        assert kwargs["cwd"] == str(workspace)
        encoded = shlex.split(kwargs["command"])[-1]
        payload = json.loads(base64.urlsafe_b64decode(encoded).decode())
        assert payload["log_dir"] == str(logs)
        assert payload["workspace"] == str(workspace)
        await _runner_run(payload)
        return SimpleNamespace(stdout=None)

    monkeypatch.setattr(agent, "exec_as_agent", run_locally)
    context = AgentContext()

    await agent.run("solve it", object(), context)

    assert context.n_input_tokens == 12
    assert context.n_cache_tokens == 6
    assert context.n_output_tokens == 7
    assert context.metadata is not None
    assert context.metadata["group"] == "default"
    assert (logs / "ursa_result.out").read_text() == "runner answer"
    assert "Inspecting the workspace" in (logs / "ursa.log").read_text()

    records = _records(logs / "ursa.jsonl")
    assert [(record["type"], record["event"]) for record in records] == [
        ("chain", "start"),
        ("agent", "progress"),
        ("chat", "start"),
        ("chat", "end"),
        ("tool", "start"),
        ("tool", "end"),
        ("chain", "end"),
    ]
    assert records[0]["name"] == "runner_lifecycle"
    assert records[3]["usage"] == {
        "input_tokens": 12,
        "output_tokens": 7,
        "reasoning_tokens": 4,
        "cached_tokens": 6,
    }
    timestamps = [record["timestamp_monotonic_ns"] for record in records]
    assert timestamps == sorted(timestamps)

    checkpoint_path = logs / "db" / "checkpointer.db"
    with sqlite3.connect(checkpoint_path) as database:
        assert database.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert database.execute(
            "SELECT value FROM runner_probe"
        ).fetchall() == [("persisted",)]


@pytest.mark.asyncio
async def test_harbor_context_keeps_usage_when_runner_fails(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    logs = tmp_path / "agent"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_file = _runtime_config(tmp_path / "ursa.yaml")
    agent = UrsaHarborAgent(
        logs_dir=logs,
        environment_logs_dir=logs,
        model_name="openai/gpt-4.1-nano",
        config_file=config_file,
        agent_import_path=f"{__name__}:HarborLifecycleAgent",
    )
    agent._remote_config_file = str(config_file)
    agent._workspace = str(workspace)

    async def run_locally(_environment: Any, **kwargs: Any) -> Any:
        encoded = shlex.split(kwargs["command"])[-1]
        payload = json.loads(base64.urlsafe_b64decode(encoded).decode())
        await _runner_run(payload)
        return SimpleNamespace(stdout=None)

    monkeypatch.setattr(agent, "exec_as_agent", run_locally)
    context = AgentContext()

    with pytest.raises(RuntimeError, match="agent failed after chat"):
        await agent.run("fail after usage", object(), context)

    assert context.n_input_tokens == 12
    assert context.n_cache_tokens == 6
    assert context.n_output_tokens == 7
    assert not (logs / "ursa_result.out").exists()


@pytest.mark.asyncio
async def test_runner_closes_real_checkpointer_when_agent_setup_fails(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    SetupFailingAgent.connection = None

    with pytest.raises(RuntimeError, match="agent setup failed"):
        await _runner_run({
            "agent_import_path": f"{__name__}:SetupFailingAgent",
            "config_file": str(_runtime_config(tmp_path / "ursa.yaml")),
            "workspace": str(tmp_path / "workspace"),
            "log_dir": str(tmp_path / "agent"),
            "instruction": "solve it",
        })

    assert SetupFailingAgent.connection is not None
    with pytest.raises(ValueError, match="no active connection"):
        await SetupFailingAgent.connection.execute("SELECT 1")


def test_jsonl_token_usage_streams_and_ignores_invalid_records(tmp_path):
    path = tmp_path / "ursa.jsonl"
    path.write_bytes(
        b"\xffnot-json\n"
        b'{"type":"chat","event":"end","usage":{}}\n'
        b'{"type":"chat","event":"end","usage":'
        b'{"input_tokens":true,"cached_tokens":-1,"output_tokens":1.5}}\n'
    )

    assert _jsonl_token_usage(path) == (None, None, None)

    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            '{"type":"chat","event":"end","usage":'
            '{"input_tokens":9007199254740993}}\n'
        )
    assert _jsonl_token_usage(path) == (9007199254740993, None, None)


@pytest.mark.asyncio
async def test_runner_plan_execute_uses_async_agents_and_separate_threads(
    tmp_path, monkeypatch
):
    from ursa import agents

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    created = []
    calls = []

    class Planner:
        def __init__(self, **kwargs):
            created.append(("planner", kwargs))

        async def ainvoke(self, prompt, *, config):
            calls.append(("plan", prompt, config))
            return {"plan": SimpleNamespace(steps=["build", "verify"])}

    class Executor:
        def __init__(self, **kwargs):
            created.append(("executor", kwargs))

        async def ainvoke(self, prompt, *, config):
            calls.append(("execute", prompt, config))
            return {"answer": f"completed {len(calls)}"}

        def format_result(self, result):
            return result["answer"]

    monkeypatch.setattr(agents, "PlanningAgent", Planner)
    monkeypatch.setattr(agents, "ExecutionAgent", Executor)
    await _runner_run({
        "agent_import_path": "ursa.workflows:PlanningExecutorWorkflow",
        "config_file": str(_runtime_config(tmp_path / "ursa.yaml")),
        "workspace": str(tmp_path / "workspace"),
        "log_dir": str(tmp_path / "agent"),
        "instruction": "produce the simulation",
    })
    assert [call[0] for call in calls] == ["plan", "execute", "execute"]
    assert all("produce the simulation" in call[1] for call in calls)
    assert "completed 2" in calls[-1][1]
    assert all(call[2]["callbacks"] for call in calls)
    assert created[0][1]["checkpointer"] is created[1][1]["checkpointer"]
    assert created[0][1]["thread_id"] != created[1][1]["thread_id"]
    assert (tmp_path / "agent/ursa_result.out").read_text() == "completed 3"
    with pytest.raises(ValueError, match="no active connection"):
        await created[0][1]["checkpointer"].conn.execute("SELECT 1")
