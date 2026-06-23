
import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler

import pytest

from pycodex.compat import ThreadingHTTPServer
from pycodex import (
    AgentEvent,
    Agent,
    CliSubmissionQueue,
    AssistantMessage,
    BaseTool,
    ContextMessage,
    ModelResponse,
    ModelStreamEvent,
    ReasoningItem,
    ResponsesProviderConfig,
    ToolCall,
    ToolResult,
    ToolRegistry,
    UserMessage,
)
from pycodex.cli import (
    CliSessionView,
    EXTRA_COMMANDS_LINE,
    LOCAL_RESPONSES_SERVER_API_KEY_ENV,
    build_agent,
    build_model,
    build_parser,
    build_cli_queue,
    get_subagent_tools,
    get_tools,
    launch_chat_completion_compat_server,
    load_codex_dotenv,
    prompt_request_permissions,
    prompt_request_user_input,
    resolve_prompt_text,
    run_cli,
    run_interactive_session,
    should_run_interactive,
)
from pycodex.portable import DEFAULT_ENTRY_CONFIG, upload_codex_home
from pycodex.portable_server import CodexStorageServer
from pycodex.utils.compactor import SUMMARY_PREFIX
from pycodex.utils.session_persist import load_resumed_session
from pycodex.utils.toolcall_visualize import colorize_tool_message, tool_summary
from pycodex.utils.visualize import colorize_cli_message
from tests.fake_responses_server import CaptureStore, build_handler
from tests.fakes import ScriptedModelClient
import typing


def _normalized_headers(headers: 'typing.Dict[str, str]') -> 'typing.Dict[str, str]':
    return {key.lower(): value for key, value in headers.items()}


def _configure_cli_view_output(
    view: 'CliSessionView',
    line_output: 'typing.List[str]',
    stream_chunks: 'typing.Union[typing.List[str], None]' = None,
    line_callback=None,
) -> 'CliSessionView':
    def write_line(text: 'str') -> 'None':
        line_output.append(text)
        if line_callback is not None:
            line_callback(text)

    view._line_output = write_line
    view._color_enabled = False
    return view


def _build_cli_view(
    line_output: 'typing.List[str]',
    stream_chunks: 'typing.Union[typing.List[str], None]' = None,
) -> 'CliSessionView':
    return _configure_cli_view_output(CliSessionView(), line_output, stream_chunks)


def _write_stored_codex_home(root: 'Path') -> 'None':
    (root / "skills" / "demo").mkdir(parents=True)
    (root / "skills" / "demo" / "SKILL.md").write_text(
        "# Demo\n\nStored skill.\n"
    )
    (root / "AGENTS.md").write_text("stored agents instructions\n")
    (root / ".env").write_text('PORTABLE_API_KEY="from-storage-dotenv"\n')
    (root / DEFAULT_ENTRY_CONFIG).write_text(
        "\n".join(
            [
                'model = "demo-model"',
                'model_provider = "demo"',
                '[model_providers.demo]',
                'base_url = "https://example.com/v1"',
                'env_key = "PORTABLE_API_KEY"',
            ]
        )
    )


def _write_test_rollout(
    codex_home: 'Path',
    thread_id: 'str',
    items: 'typing.List[typing.Dict[str, object]]',
) -> 'Path':
    rollout_path = (
        codex_home
        / "sessions"
        / "2026"
        / "04"
        / "07"
        / f"rollout-2026-04-07T00-00-00-{thread_id}.jsonl"
    )
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    rollout_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False, indent=2) for item in items)
        + "\n{\"timestamp\": \"truncated"
    )
    return rollout_path


def _install_test_cli_view(
    monkeypatch: 'pytest.MonkeyPatch',
    inputs,
    line_output: 'typing.List[str]',
    stream_chunks: 'typing.List[str]',
    prompt_hook=None,
    line_callback=None,
    capture_view: 'typing.Union[typing.Dict[str, CliSessionView], None]' = None,
) -> 'None':
    input_iter = iter(inputs)
    real_view_class = CliSessionView

    class _TestView(real_view_class):
        def __init__(self) -> 'None':
            super().__init__()
            _configure_cli_view_output(
                self,
                line_output,
                stream_chunks,
                line_callback=line_callback,
            )
            if capture_view is not None:
                capture_view["view"] = self

        async def poll_prompt(self, prompt: 'str' = None) -> 'str':
            value = next(input_iter)
            if prompt_hook is not None:
                await prompt_hook(self, prompt, value)
            return value

        async def get_prompt(self, prompt: 'str' = None) -> 'str':
            return await self.poll_prompt(prompt)

    monkeypatch.setattr("pycodex.cli.CliSessionView", _TestView)


class _ScriptedResponsesClient(ScriptedModelClient):
    def __init__(
        self,
        responses=None,
        response_factory=None,
        override_factory=None,
    ) -> 'None':
        super().__init__(responses=responses, response_factory=response_factory)
        self._override_factory = (
            override_factory
            or (
                lambda _model,
                _reasoning,
                _session_id=None,
                _openai_subagent=None: self
            )
        )

    def with_overrides(
        self,
        model=None,
        reasoning_effort=None,
        session_id=None,
        openai_subagent=None,
    ):
        return self._override_factory(
            model,
            reasoning_effort,
            session_id,
            openai_subagent,
        )


class _DummyProviderConfig:
    provider_name = "demo"


def _build_scripted_tui_runtime(
    config_path: 'Path',
    responses,
) -> 'typing.Tuple[CliSubmissionQueue, _ScriptedResponsesClient]':
    client = _ScriptedResponsesClient(responses=responses)
    client._session_id = None
    client._originator = "codex-tui"
    client._config = _DummyProviderConfig()
    agent = build_agent(
        client,
        config_path=str(config_path),
        session_mode="tui",
    )
    runtime = build_cli_queue(agent)
    return runtime, client


def test_resolve_prompt_text_prefers_argv() -> 'None':
    assert resolve_prompt_text(["hello", "world"]) == "hello world"


def test_resolve_prompt_text_falls_back_to_stdin(monkeypatch) -> 'None':
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdin.read", lambda: "  hello from stdin  ")
    assert resolve_prompt_text([]) == "hello from stdin"


def test_resolve_prompt_text_rejects_missing_prompt(monkeypatch) -> 'None':
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    with pytest.raises(ValueError, match="prompt is required"):
        resolve_prompt_text([])


def test_build_parser_recognizes_json_flag() -> 'None':
    parser = build_parser()
    args = parser.parse_args(["--json", "hello"])
    assert args.json is True
    assert args.prompt == ["hello"]


def test_build_parser_recognizes_vllm_endpoint() -> 'None':
    parser = build_parser()
    args = parser.parse_args(["--vllm-endpoint", "http://127.0.0.1:18000", "hello"])
    assert args.vllm_endpoint == "http://127.0.0.1:18000"
    assert args.prompt == ["hello"]


def test_build_parser_recognizes_use_messages_flag() -> 'None':
    parser = build_parser()
    args = parser.parse_args(["--use-messages", "hello"])
    assert args.use_messages is True
    assert args.prompt == ["hello"]


def test_build_parser_recognizes_put_and_call_flags() -> 'None':
    parser = build_parser()
    args = parser.parse_args(
        [
            "--put",
            "/tmp/.codex@127.0.0.1:5577",
        ]
    )

    assert args.put == "/tmp/.codex@127.0.0.1:5577"
    assert args.call is None
    assert args.prompt == []

    bare_server_args = parser.parse_args(
        [
            "--put",
            "@127.0.0.1:5577",
        ]
    )
    assert bare_server_args.put == "@127.0.0.1:5577"


def test_build_parser_rejects_put_without_argument() -> 'None':
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--put"])


def test_vllm_base_url_normalizes_empty_path_to_v1() -> 'None':
    from responses_server import CompatServerConfig

    config = CompatServerConfig.from_base_url("http://127.0.0.1:18000")

    assert config.outcomming_base_url == "http://127.0.0.1:18000/v1"


def test_vllm_base_url_preserves_existing_v1_path() -> 'None':
    from responses_server import CompatServerConfig

    config = CompatServerConfig.from_base_url("http://127.0.0.1:18000/v1/")

    assert config.outcomming_base_url == "http://127.0.0.1:18000/v1"


def test_launch_chat_completion_compat_server_normalizes_vllm_base_url(
    monkeypatch,
) -> 'None':
    seen = {}

    class _FakeManagedServer:
        def __init__(self, config):
            seen["server_config"] = config

        def start(self):
            seen["started"] = True

    monkeypatch.setattr("responses_server.app.ManagedResponseServer", _FakeManagedServer)
    launch_chat_completion_compat_server(
        "http://127.0.0.1:18000",
        model_provider="vllm",
    )
    assert seen["server_config"].outcomming_base_url == "http://127.0.0.1:18000/v1"
    assert seen["server_config"].model_provider == "vllm"
    assert seen["started"] is True


def test_build_agent_overrides_provider_for_managed_responses_url(
    tmp_path,
    monkeypatch,
) -> 'None':
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'model = "demo-model"',
                'model_provider = "demo"',
                '[model_providers.demo]',
                'base_url = "https://example.com/v1"',
                'env_key = "DUMMY_KEY"',
            ]
        )
    )
    seen = {}

    class _FakeResponsesModelClient:
        def __init__(
            self,
            config,
            timeout_seconds,
            session_id=None,
            originator=None,
            user_agent=None,
            openai_subagent=None,
        ):
            seen["config"] = config
            seen["timeout_seconds"] = timeout_seconds
            seen["session_id"] = session_id
            seen["originator"] = originator
            seen["user_agent"] = user_agent
            seen["openai_subagent"] = openai_subagent

    monkeypatch.setattr("pycodex.cli.ResponsesModelClient", _FakeResponsesModelClient)

    model = build_model(
        config_path=str(config_path),
        profile=None,
        timeout_seconds=120.0,
        managed_responses_base_url="http://127.0.0.1:18001/v1",
    )
    build_agent(
        model,
        config_path=str(config_path),
        profile=None,
        system_prompt=None,
        session_mode="tui",
    )

    assert seen["originator"] == "codex-tui"
    assert seen["config"].base_url == "http://127.0.0.1:18001/v1"
    assert seen["config"].api_key_env == LOCAL_RESPONSES_SERVER_API_KEY_ENV
    assert os.environ[LOCAL_RESPONSES_SERVER_API_KEY_ENV] == "dummy"


def test_build_model_can_be_called_without_arguments(monkeypatch) -> 'None':
    captured = {}
    fake_client = object()

    def fake_from_codex_config(config_path, profile=None):
        captured["provider_args"] = {
            "config_path": str(config_path),
            "profile": profile,
        }
        return ResponsesProviderConfig(
            model="demo-model",
            provider_name="demo",
            base_url="https://example.com/v1",
            api_key_env="DUMMY_KEY",
        )

    class _FakeResponsesModelClient:
        def __new__(cls, config, timeout_seconds, originator=None):
            captured["client_args"] = {
                "config": config,
                "timeout_seconds": timeout_seconds,
                "originator": originator,
            }
            return fake_client

    monkeypatch.setattr("pycodex.cli.load_codex_dotenv", lambda _config_path: None)
    monkeypatch.setattr(
        "pycodex.cli.ResponsesProviderConfig.from_codex_config",
        fake_from_codex_config,
    )
    monkeypatch.setattr("pycodex.cli.ResponsesModelClient", _FakeResponsesModelClient)

    model = build_model()

    default_config_path = str(Path.home() / ".codex" / "config.toml")
    assert model is fake_client
    assert captured["provider_args"] == {
        "config_path": default_config_path,
        "profile": None,
    }
    assert captured["client_args"]["timeout_seconds"] == 120.0
    assert captured["client_args"]["originator"] == "codex-tui"


def test_build_model_client_respects_use_chat_completion_from_config(
    tmp_path,
    monkeypatch,
) -> 'None':
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'model = "demo-model"',
                'model_provider = "demo"',
                '[model_providers.demo]',
                'base_url = "https://example.com/v1"',
                'env_key = "DUMMY_KEY"',
                'use_chat_completion = true',
            ]
        )
    )
    seen = {}

    class _FakeManagedServer:
        base_url = "http://127.0.0.1:18083/v1"

        def stop(self):
            seen["stopped"] = True

    def fake_launch(
        base_url,
        api_key_env=None,
        model_provider=None,
        outcomming_api="chat_completions",
    ):
        seen["endpoint"] = base_url
        seen["api_key_env"] = api_key_env
        seen["model_provider"] = model_provider
        seen["outcomming_api"] = outcomming_api
        return _FakeManagedServer()

    monkeypatch.setattr("pycodex.cli.launch_chat_completion_compat_server", fake_launch)
    monkeypatch.setattr("pycodex.cli.configure_loguru", lambda: None)
    monkeypatch.setenv("DUMMY_KEY", "test-key")

    client = build_model(
        str(config_path),
        None,
        60.0,
        use_chat_completion=None,
    )

    assert seen["endpoint"] == "https://example.com/v1"
    assert seen["api_key_env"] == "DUMMY_KEY"
    assert seen["model_provider"] == "demo"
    assert seen["outcomming_api"] == "chat_completions"
    assert client._config.base_url == "http://127.0.0.1:18083/v1"
    assert client._config.api_key_env == "PYCODEX_LOCAL_RESPONSES_SERVER_KEY"


@pytest.mark.asyncio
async def test_run_cli_launches_managed_responses_server_for_vllm_endpoint(
    monkeypatch,
    tmp_path,
) -> 'None':
    started = {}
    registered = {}
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'model = "demo-model"',
                'model_provider = "demo"',
                '[model_providers.demo]',
                'base_url = "https://example.com/v1"',
                'env_key = "DUMMY_KEY"',
            ]
        )
    )
    monkeypatch.setenv("DUMMY_KEY", "test-key")

    class _FakeManagedServer:
        base_url = "http://127.0.0.1:18001/v1"

        def stop(self):
            started["stopped"] = True

    class _FakeRuntime:
        def __init__(self):
            self._stopped = asyncio.Event()

        def set_event_handler(self, _handler=None):
            return None

        async def run_forever(self):
            await self._stopped.wait()

        async def submit_user_turn(self, prompt_text):
            started["prompt_text"] = prompt_text
            return type(
                "_Result",
                (),
                {
                    "output_text": "OK",
                    "turn_id": "turn_1",
                    "iterations": 1,
                    "response_items": (),
                    "history": (),
                },
            )()

        async def shutdown(self):
            self._stopped.set()

    def fake_launch(
        base_url,
        api_key_env=None,
        model_provider=None,
        outcomming_api="chat_completions",
    ):
        started["endpoint"] = base_url
        started["api_key_env"] = api_key_env
        started["model_provider"] = model_provider
        started["outcomming_api"] = outcomming_api
        return _FakeManagedServer()

    def fake_build_cli_queue(agent):
        started["session_mode"] = "tui"
        started["base_url_override"] = agent._model_client._config.base_url
        return _FakeRuntime()

    monkeypatch.setattr("pycodex.cli.launch_chat_completion_compat_server", fake_launch)
    monkeypatch.setattr("pycodex.cli.build_cli_queue", fake_build_cli_queue)
    monkeypatch.setattr(
        "pycodex.cli.atexit.register",
        lambda callback: registered.setdefault("callback", callback),
    )
    monkeypatch.setattr("pycodex.cli.configure_loguru", lambda: None)
    monkeypatch.setattr("sys.stdin.read", lambda: "")

    args = build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--vllm-endpoint",
            "http://127.0.0.1:18000",
            "Reply with exactly OK.",
        ]
    )
    exit_code = await run_cli(args)

    assert exit_code == 0
    assert started["endpoint"] == "http://127.0.0.1:18000"
    assert started["model_provider"] == "vllm"
    assert started["session_mode"] == "tui"
    assert started["prompt_text"] == "Reply with exactly OK."
    assert started["base_url_override"] == "http://127.0.0.1:18001/v1"
    assert callable(registered["callback"])
    registered["callback"]()
    assert started["stopped"] is True


@pytest.mark.asyncio
async def test_run_cli_launches_managed_responses_server_for_messages_backend(
    monkeypatch,
    tmp_path,
) -> 'None':
    started = {}
    registered = {}
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'model = "claude-opus-4-6"',
                'model_provider = "opus"',
                '[profiles.opus]',
                'model = "claude-opus-4-6"',
                'model_provider = "opus"',
                '[model_providers.opus]',
                'base_url = "https://models-proxy.stepfun-inc.com/v1"',
                'env_key = "MODELS_PROXY_API_KEY"',
            ]
        )
    )
    monkeypatch.setenv("MODELS_PROXY_API_KEY", "test-key")

    class _FakeManagedServer:
        base_url = "http://127.0.0.1:18082/v1"

        def stop(self):
            started["stopped"] = True

    class _FakeRuntime:
        def __init__(self):
            self._stopped = asyncio.Event()

        def set_event_handler(self, _handler=None):
            return None

        async def run_forever(self):
            await self._stopped.wait()

        async def submit_user_turn(self, prompt_text):
            started["prompt_text"] = prompt_text
            return type(
                "_Result",
                (),
                {
                    "output_text": "OK",
                    "turn_id": "turn_1",
                    "iterations": 1,
                    "response_items": (),
                    "history": (),
                },
            )()

        async def shutdown(self):
            self._stopped.set()

    def fake_launch(
        base_url,
        api_key_env=None,
        model_provider=None,
        outcomming_api="chat_completions",
    ):
        started["endpoint"] = base_url
        started["api_key_env"] = api_key_env
        started["model_provider"] = model_provider
        started["outcomming_api"] = outcomming_api
        return _FakeManagedServer()

    def fake_build_cli_queue(agent):
        started["session_mode"] = "tui"
        started["base_url_override"] = agent._model_client._config.base_url
        return _FakeRuntime()

    monkeypatch.setattr("pycodex.cli.launch_chat_completion_compat_server", fake_launch)
    monkeypatch.setattr("pycodex.cli.build_cli_queue", fake_build_cli_queue)
    monkeypatch.setattr(
        "pycodex.cli.atexit.register",
        lambda callback: registered.setdefault("callback", callback),
    )
    monkeypatch.setattr("pycodex.cli.configure_loguru", lambda: None)
    monkeypatch.setattr("sys.stdin.read", lambda: "")

    args = build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--profile",
            "opus",
            "--use-messages",
            "Reply with exactly OK.",
        ]
    )
    exit_code = await run_cli(args)

    assert exit_code == 0
    assert started["endpoint"] == "https://models-proxy.stepfun-inc.com/v1"
    assert started["api_key_env"] == "MODELS_PROXY_API_KEY"
    assert started["model_provider"] == "opus"
    assert started["outcomming_api"] == "messages"
    assert started["session_mode"] == "tui"
    assert started["prompt_text"] == "Reply with exactly OK."
    assert started["base_url_override"] == "http://127.0.0.1:18082/v1"
    assert callable(registered["callback"])
    registered["callback"]()
    assert started["stopped"] is True


@pytest.mark.asyncio
async def test_run_cli_put_uploads_codex_home_and_prints_call_spec(
    tmp_path,
    monkeypatch,
) -> 'None':
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    _write_stored_codex_home(codex_home)
    server = CodexStorageServer(tmp_path / "storage-server", port=0)
    server.start()
    line_output: 'typing.List[str]' = []
    monkeypatch.setattr("pycodex.cli.configure_loguru", lambda: None)
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: line_output.append(" ".join(str(arg) for arg in args)),
    )

    try:
        args = build_parser().parse_args(
            [
                "--put",
                f"{codex_home}@{server.server_address}",
            ]
        )
        exit_code = await run_cli(args)
    finally:
        server.stop()

    assert exit_code == 0
    assert any(line.startswith("[put] source: ") for line in line_output)
    assert "[put] file: config.toml" in line_output
    assert "[put] file: AGENTS.md" in line_output
    assert any(line.startswith("[put] uploaded: ") for line in line_output)
    assert any(line.startswith("[put] testing call: ") for line in line_output)
    assert "[put] call test ok: config.toml" in line_output
    assert "[put] one-click start:" in line_output
    assert line_output[-1].startswith("pycodex --call ")
    assert line_output[-1].endswith(f"@{server.server_address}")


@pytest.mark.asyncio
async def test_run_cli_bootstraps_called_home_before_loading_config(
    tmp_path,
    monkeypatch,
) -> 'None':
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    _write_stored_codex_home(codex_home)
    server = CodexStorageServer(tmp_path / "storage-server", port=0)
    server.start()
    stored_call = upload_codex_home(f"{codex_home}@{server.server_address}")
    captured = {}

    class _FakeResponsesModelClient:
        def __init__(
            self,
            config,
            timeout_seconds,
            session_id=None,
            originator=None,
            user_agent=None,
            openai_subagent=None,
        ):
            captured["provider_config"] = config
            captured["timeout_seconds"] = timeout_seconds
            captured["session_id"] = session_id
            captured["originator"] = originator
            captured["user_agent"] = user_agent
            captured["openai_subagent"] = openai_subagent
            self._config = config
            self.model = config.model

        def with_overrides(
            self,
            model=None,
            reasoning_effort=None,
            session_id=None,
            openai_subagent=None,
        ):
            del model, reasoning_effort, session_id, openai_subagent
            return self

    class _FakeRuntime:
        def __init__(self):
            self._stopped = asyncio.Event()

        def set_event_handler(self, _handler=None):
            return None

        async def run_forever(self):
            await self._stopped.wait()

        async def submit_user_turn(self, prompt_text):
            captured["prompt_text"] = prompt_text
            return type(
                "_Result",
                (),
                {
                    "output_text": "OK",
                    "turn_id": "turn_1",
                    "iterations": 1,
                    "response_items": (),
                    "history": (),
                },
            )()

        async def shutdown(self):
            self._stopped.set()

    def fake_build_cli_queue(agent):
        captured["config_path"] = str(agent._context_manager._config.codex_home / "config.toml")
        captured["client_base_url"] = agent._model_client._config.base_url
        captured["session_mode"] = "tui"
        return _FakeRuntime()

    monkeypatch.setattr("pycodex.cli.ResponsesModelClient", _FakeResponsesModelClient)
    monkeypatch.setattr("pycodex.cli.build_cli_queue", fake_build_cli_queue)
    monkeypatch.setattr("pycodex.cli.configure_loguru", lambda: None)
    monkeypatch.setattr("sys.stdin.read", lambda: "")
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("PORTABLE_API_KEY", raising=False)

    try:
        args = build_parser().parse_args(
            [
                "--call",
                stored_call,
                "Reply with exactly OK.",
            ]
        )
        exit_code = await run_cli(args)
    finally:
        server.stop()

    assert exit_code == 0
    assert captured["prompt_text"] == "Reply with exactly OK."
    assert captured["session_mode"] == "tui"
    assert Path(captured["config_path"]).name == "config.toml"
    assert Path(captured["config_path"]).is_file()
    assert os.environ["CODEX_HOME"] == str(Path(captured["config_path"]).parent)
    assert os.environ["PORTABLE_API_KEY"] == "from-storage-dotenv"
    assert captured["provider_config"].base_url == "https://example.com/v1"
    assert captured["provider_config"].api_key_env == "PORTABLE_API_KEY"


@pytest.mark.asyncio
async def test_run_cli_call_reads_called_home_text_as_utf8(
    tmp_path,
    monkeypatch,
) -> 'None':
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    _write_stored_codex_home(codex_home)
    (codex_home / "AGENTS.md").write_text(
        "stored rules with unicode: \u22645 min \u2014 \u4e2d\u6587\n",
        encoding="utf-8",
    )
    (codex_home / "skills" / "demo" / "SKILL.md").write_text(
        "---\n"
        "name: demo\n"
        "description: unicode \u2264 \u2014 \u4e2d\u6587\n"
        "---\n"
        "Stored skill.\n",
        encoding="utf-8",
    )
    server = CodexStorageServer(tmp_path / "storage-server", port=0)
    server.start()

    original_read_text = Path.read_text

    def read_text_as_gbk_by_default(path, *args, **kwargs):
        encoding = kwargs.get("encoding")
        if args:
            encoding = args[0]
        if encoding is None:
            return path.read_bytes().decode("gbk")
        return original_read_text(path, *args, **kwargs)

    class _FakeResponsesModelClient(_ScriptedResponsesClient):
        def __init__(
            self,
            config,
            timeout_seconds,
            session_id=None,
            originator=None,
            user_agent=None,
            openai_subagent=None,
        ) -> 'None':
            del timeout_seconds, user_agent, openai_subagent
            super().__init__([ModelResponse(items=[AssistantMessage(text="OK")])])
            self._config = config
            self.model = config.model
            self._session_id = session_id
            self._originator = originator or "pycodex"

    monkeypatch.setattr(Path, "read_text", read_text_as_gbk_by_default)
    monkeypatch.setattr("pycodex.cli.ResponsesModelClient", _FakeResponsesModelClient)
    monkeypatch.setattr("pycodex.cli.configure_loguru", lambda: None)
    monkeypatch.setattr("sys.stdin.read", lambda: "")
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("PORTABLE_API_KEY", raising=False)

    try:
        stored_call = upload_codex_home(f"{codex_home}@{server.server_address}")
        args = build_parser().parse_args(["--call", stored_call, "say ok"])
        exit_code = await run_cli(args)
    finally:
        server.stop()

    assert exit_code == 0


def test_get_tools_registers_expected_builtin_tools() -> 'None':
    registry = get_tools()
    assert registry.names() == (
        "shell",
        "shell_command",
        "exec_command",
        "write_stdin",
        "exec",
        "wait",
        "web_search",
        "update_plan",
        "request_user_input",
        "request_permissions",
        "spawn_agent",
        "send_input",
        "resume_agent",
        "wait_agent",
        "close_agent",
        "apply_patch",
        "grep_files",
        "read_file",
        "list_dir",
        "view_image",
    )


def test_get_tools_exec_mode_matches_codex_exec_subset() -> 'None':
    registry = get_tools(exec_mode=True)
    assert registry.names() == (
        "exec_command",
        "write_stdin",
        "update_plan",
        "request_user_input",
        "apply_patch",
        "web_search",
        "view_image",
        "spawn_agent",
        "send_input",
        "resume_agent",
        "wait_agent",
        "close_agent",
    )


def _tool_payload_by_name(registry: 'ToolRegistry') -> 'typing.Dict[str, typing.Dict[str, object]]':
    payloads: 'typing.Dict[str, typing.Dict[str, object]]' = {}
    for payload in [spec.serialize() for spec in registry.model_visible_specs()]:
        name = payload.get("name") or payload.get("type")
        assert isinstance(name, str)
        payloads[name] = payload
    return payloads


def test_get_tools_exec_mode_serialization_comes_from_class_specs() -> 'None':
    registry = get_tools(exec_mode=True)
    payloads = _tool_payload_by_name(registry)

    exec_command = payloads["exec_command"]
    assert "output_schema" not in exec_command
    assert "in a PTY" in exec_command["description"]
    assert "reply first" in exec_command["description"]
    assert "you will be invoked to continue" in exec_command["description"]
    exec_command_tool = registry.get_tool("exec_command")
    assert exec_command_tool is not None
    assert exec_command_tool.output_schema["required"] == ["wall_time_seconds", "output"]
    assert "sandbox_permissions" not in exec_command["parameters"]["properties"]
    assert "justification" not in exec_command["parameters"]["properties"]
    assert "prefix_rule" not in exec_command["parameters"]["properties"]

    write_stdin = payloads["write_stdin"]
    assert "output_schema" not in write_stdin
    write_stdin_tool = registry.get_tool("write_stdin")
    assert write_stdin_tool is not None
    assert write_stdin_tool.output_schema == exec_command_tool.output_schema
    assert write_stdin["parameters"]["properties"]["session_id"]["type"] == "number"

    request_user_input = payloads["request_user_input"]
    assert "autoResolutionMs" in request_user_input["parameters"]["properties"]

    view_image = payloads["view_image"]
    assert view_image["parameters"]["properties"]["detail"]["enum"] == ["high", "original"]
    assert "output_schema" not in view_image
    view_image_tool = registry.get_tool("view_image")
    assert view_image_tool is not None
    assert view_image_tool.output_schema["required"] == ["image_url", "detail"]

    close_agent = payloads["close_agent"]
    assert "output_schema" not in close_agent
    close_agent_tool = registry.get_tool("close_agent")
    assert close_agent_tool is not None
    assert close_agent_tool.output_schema["required"] == ["previous_status"]
    assert "previous_status" in close_agent_tool.output_schema["properties"]


def test_get_subagent_tools_matches_upstream_subset() -> 'None':
    registry = get_subagent_tools()
    assert registry.names() == (
        "exec_command",
        "write_stdin",
        "update_plan",
        "apply_patch",
        "web_search",
        "view_image",
    )


def test_get_subagent_tools_serialization_comes_from_class_specs() -> 'None':
    registry = get_subagent_tools()
    payloads = _tool_payload_by_name(registry)

    assert set(payloads) == {
        "exec_command",
        "write_stdin",
        "update_plan",
        "apply_patch",
        "web_search",
        "view_image",
    }
    assert "output_schema" not in payloads["exec_command"]
    exec_command_tool = registry.get_tool("exec_command")
    assert exec_command_tool is not None
    assert exec_command_tool.output_schema["required"] == ["wall_time_seconds", "output"]
    assert payloads["write_stdin"]["parameters"]["properties"]["session_id"]["type"] == "number"
    assert payloads["view_image"]["parameters"]["properties"]["detail"]["enum"] == [
        "high",
        "original",
    ]


def test_load_codex_dotenv_reads_env_file_and_filters_codex_prefix(tmp_path, monkeypatch) -> 'None':
    config_path = tmp_path / "config.toml"
    config_path.write_text('model = "demo"\nmodel_provider = "demo"\n')
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                'TEST_KEY="loaded-from-dotenv"',
                "export OTHER_KEY=second-value",
                "PLAIN_KEY=plain-value # inline comment",
                "CODEX_INTERNAL_ORIGINATOR_OVERRIDE=blocked",
            ]
        )
    )

    monkeypatch.delenv("TEST_KEY", raising=False)
    monkeypatch.delenv("OTHER_KEY", raising=False)
    monkeypatch.delenv("PLAIN_KEY", raising=False)
    monkeypatch.delenv("CODEX_INTERNAL_ORIGINATOR_OVERRIDE", raising=False)

    load_codex_dotenv(config_path)

    assert os.environ["TEST_KEY"] == "loaded-from-dotenv"
    assert os.environ["OTHER_KEY"] == "second-value"
    assert os.environ["PLAIN_KEY"] == "plain-value"
    assert "CODEX_INTERNAL_ORIGINATOR_OVERRIDE" not in os.environ


def test_should_run_interactive_only_without_prompt_on_tty() -> 'None':
    assert should_run_interactive([], True) is True
    assert should_run_interactive(["hello"], True) is False
    assert should_run_interactive([], False) is False


@pytest.mark.asyncio
async def test_run_interactive_session_steer_mode_restarts_at_request_boundary(
    monkeypatch,
) -> 'None':
    first_turn_started = asyncio.Event()
    release_first_turn = asyncio.Event()

    class _DelayedModelClient:
        def __init__(self) -> 'None':
            self.call_count = 0

        async def complete(self, prompt, event_handler):
            del prompt
            self.call_count += 1
            if self.call_count == 1:
                first_turn_started.set()
                await release_first_turn.wait()
            text = "first" if self.call_count == 1 else "second"
            event_handler(ModelStreamEvent(kind="assistant_delta", payload={"delta": text}))
            return ModelResponse(items=[AssistantMessage(text=text)])

    model = _DelayedModelClient()
    runtime = CliSubmissionQueue(Agent(model, ToolRegistry()))
    line_output: 'typing.List[str]' = []
    stream_chunks: 'typing.List[str]' = []
    async def prompt_hook(_view, _prompt: 'str', value: 'str') -> 'None':
        if value == "again":
            await first_turn_started.wait()
        if value == "/exit":
            release_first_turn.set()

    _install_test_cli_view(
        monkeypatch,
        ["hello", "again", "/exit"],
        line_output,
        stream_chunks,
        prompt_hook=prompt_hook,
    )

    code = await run_interactive_session(
        runtime,
        False,
    )

    assert code == 0
    assert model.call_count == 2
    assert line_output[:2] == [
        "pycodex interactive mode. Type /exit to quit.",
        EXTRA_COMMANDS_LINE,
    ]
    assert "Session: hello" in line_output
    assert "[steer] inserted: again" in line_output
    assert "Error: submission interrupted" not in line_output
    assert "assistant> first" in line_output
    assert "assistant> second" in line_output
    assert stream_chunks == []


@pytest.mark.asyncio
async def test_run_interactive_session_queue_command_enqueues_turn(
    monkeypatch,
) -> 'None':
    release_first_turn = asyncio.Event()

    class _DelayedModelClient:
        def __init__(self) -> 'None':
            self.call_count = 0

        async def complete(self, prompt, event_handler):
            del prompt
            self.call_count += 1
            text = "first" if self.call_count == 1 else "second"
            if self.call_count == 1:
                await release_first_turn.wait()
            event_handler(ModelStreamEvent(kind="assistant_delta", payload={"delta": text}))
            return ModelResponse(items=[AssistantMessage(text=text)])

    model = _DelayedModelClient()
    runtime = CliSubmissionQueue(Agent(model, ToolRegistry()))
    line_output: 'typing.List[str]' = []
    stream_chunks: 'typing.List[str]' = []
    async def prompt_hook(_view, _prompt: 'str', value: 'str') -> 'None':
        if value == "/exit":
            release_first_turn.set()

    _install_test_cli_view(
        monkeypatch,
        ["hello", "/queue again", "/exit"],
        line_output,
        stream_chunks,
        prompt_hook=prompt_hook,
    )

    code = await run_interactive_session(
        runtime,
        False,
    )

    assert code == 0
    assert model.call_count == 2
    assert "[steer] queued: again" in line_output
    assert "[steer] inserted: again" in line_output
    assert line_output.index("[steer] queued: again") < line_output.index(
        "[steer] inserted: again"
    )
    assert "assistant> first" in line_output
    assert "assistant> second" in line_output
    assert stream_chunks == []


@pytest.mark.asyncio
async def test_run_interactive_session_keeps_status_inactive_while_waiting_for_input(
    monkeypatch,
) -> 'None':
    captured_view = {}
    model = ScriptedModelClient(
        [
            ModelResponse(items=[AssistantMessage(text="first")]),
            ModelResponse(items=[AssistantMessage(text="second")]),
        ]
    )
    runtime = CliSubmissionQueue(Agent(model, ToolRegistry()))

    async def prompt_hook(_view, _prompt: 'str', _value: 'str') -> 'None':
        assert captured_view["view"].prompter._status is None

    _install_test_cli_view(
        monkeypatch,
        ["hello", "again", "/exit"],
        [],
        [],
        prompt_hook=prompt_hook,
        capture_view=captured_view,
    )

    code = await run_interactive_session(
        runtime,
        False,
    )

    assert code == 0
    assert model.call_count == 1


@pytest.mark.asyncio
async def test_run_interactive_session_supports_history_and_title_commands(
    monkeypatch,
) -> 'None':
    model = ScriptedModelClient([ModelResponse(items=[AssistantMessage(text="done")])])
    runtime = CliSubmissionQueue(Agent(model, ToolRegistry()))
    line_output: 'typing.List[str]' = []
    stream_chunks: 'typing.List[str]' = []

    async def prompt_hook(_view, _prompt: 'str', value: 'str') -> 'None':
        if value in {"/title", "/history"}:
            while "assistant> done" not in line_output:
                await asyncio.sleep(0.01)

    _install_test_cli_view(
        monkeypatch,
        ["hello there", "/title", "/history", "/exit"],
        line_output,
        stream_chunks,
        prompt_hook=prompt_hook,
    )

    code = await run_interactive_session(
        runtime,
        False,
    )

    assert code == 0
    assert "Session: hello there" in line_output
    assert "[1]U> hello there" in line_output
    assert "[1]A> done" in line_output
    assert stream_chunks == []


@pytest.mark.asyncio
async def test_run_interactive_session_supports_compact_command(
    monkeypatch,
) -> 'None':
    model = ScriptedModelClient(
        [
            ModelResponse(items=[AssistantMessage(text="done")]),
            ModelResponse(items=[AssistantMessage(text="checkpoint summary")]),
            ModelResponse(items=[AssistantMessage(text="after compact")]),
        ]
    )
    runtime = CliSubmissionQueue(Agent(model, ToolRegistry()))
    line_output: 'typing.List[str]' = []
    stream_chunks: 'typing.List[str]' = []

    async def prompt_hook(_view, _prompt: 'str', value: 'str') -> 'None':
        if value == "/compact":
            while "assistant> done" not in line_output:
                await asyncio.sleep(0.01)
            while runtime._has_active_turn():
                await asyncio.sleep(0.01)
            await asyncio.sleep(0)

    _install_test_cli_view(
        monkeypatch,
        ["hello", "/compact", "continue", "/exit"],
        line_output,
        stream_chunks,
        prompt_hook=prompt_hook,
    )

    code = await run_interactive_session(
        runtime,
        False,
    )

    assert code == 0
    assert "Compacting conversation history..." in line_output
    assert "compact(2 items) -> 1 item + [summary]" in line_output
    assert "checkpoint summary" not in "".join(line_output)
    assert "assistant> after compact" in line_output

    compact_prompt_items = [
        item
        for item in model.prompts[1].input
        if isinstance(
            item,
            (
                UserMessage,
                AssistantMessage,
                ReasoningItem,
                ToolCall,
                ToolResult,
            ),
        )
    ]
    assert [type(item).__name__ for item in compact_prompt_items] == [
        "UserMessage",
        "AssistantMessage",
        "UserMessage",
    ]
    assert compact_prompt_items[0].text == "hello"
    assert compact_prompt_items[1].text == "done"
    assert "CONTEXT CHECKPOINT COMPACTION" in compact_prompt_items[2].text
    assert model.prompts[1].tools == []
    assert model.prompts[1].parallel_tool_calls is False

    follow_up_prompt_items = [
        item
        for item in model.prompts[2].input
        if isinstance(
            item,
            (
                UserMessage,
                AssistantMessage,
                ReasoningItem,
                ToolCall,
                ToolResult,
            ),
        )
    ]
    assert [type(item).__name__ for item in follow_up_prompt_items] == [
        "UserMessage",
        "UserMessage",
        "UserMessage",
    ]
    assert follow_up_prompt_items[0].text == "hello"
    assert "checkpoint summary" in follow_up_prompt_items[1].text
    assert follow_up_prompt_items[2].text == "continue"


@pytest.mark.asyncio
async def test_run_interactive_session_supports_resume_command(
    tmp_path,
    monkeypatch,
) -> 'None':
    thread_id = "11111111-2222-3333-4444-555555555555"
    thread_name = "saved thread"
    codex_home = tmp_path / ".codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    codex_home.mkdir()
    (codex_home / "session_index.jsonl").write_text(
        json.dumps(
            {
                "id": thread_id,
                "thread_name": thread_name,
                "updated_at": "2026-04-07T00:00:00Z",
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    _write_test_rollout(
        codex_home,
        thread_id,
        [
            {
                "timestamp": "2026-04-07T00:00:00Z",
                "type": "session_meta",
                "payload": {
                    "id": thread_id,
                    "timestamp": "2026-04-07T00:00:00Z",
                },
            },
            {
                "timestamp": "2026-04-07T00:00:01Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "ignored"}],
                },
            },
            {
                "timestamp": "2026-04-07T00:00:02Z",
                "type": "turn_context",
                "payload": {
                    "cwd": "/tmp",
                    "approval_policy": "never",
                    "sandbox_policy": {"type": "danger-full-access"},
                    "model": "gpt-5.4",
                    "summary": "detailed",
                },
            },
            {
                "timestamp": "2026-04-07T00:00:03Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "older prompt"}],
                },
            },
            {
                "timestamp": "2026-04-07T00:00:04Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "older prompt",
                },
            },
            {
                "timestamp": "2026-04-07T00:00:05Z",
                "type": "response_item",
                "payload": {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "inspect"}],
                    "content": None,
                    "encrypted_content": "enc",
                },
            },
            {
                "timestamp": "2026-04-07T00:00:06Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "echo",
                    "arguments": '{"text":"hello"}',
                    "call_id": "call_saved",
                },
            },
            {
                "timestamp": "2026-04-07T00:00:07Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_saved",
                    "output": '{"text":"hello"}',
                },
            },
            {
                "timestamp": "2026-04-07T00:00:08Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "older answer"}],
                },
            },
            {
                "timestamp": "2026-04-07T00:00:09Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "<environment_context>ignored</environment_context>",
                        }
                    ],
                },
            },
            {
                "timestamp": "2026-04-07T00:00:10Z",
                "type": "turn_context",
                "payload": {
                    "cwd": "/tmp",
                    "approval_policy": "never",
                    "sandbox_policy": {"type": "danger-full-access"},
                    "model": "gpt-5.4",
                    "summary": "detailed",
                },
            },
            {
                "timestamp": "2026-04-07T00:00:11Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "second prompt"}],
                },
            },
            {
                "timestamp": "2026-04-07T00:00:12Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "second prompt",
                },
            },
            {
                "timestamp": "2026-04-07T00:00:13Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "second answer"}],
                },
            },
        ],
    )

    class _ResumableClient(ScriptedModelClient):
        def __init__(self) -> 'None':
            super().__init__([ModelResponse(items=[AssistantMessage(text="after resume")])])
            self._session_id = None

    model = _ResumableClient()
    runtime = CliSubmissionQueue(Agent(model, ToolRegistry()))
    line_output: 'typing.List[str]' = []
    stream_chunks: 'typing.List[str]' = []

    async def prompt_hook(_view, _prompt: 'str', value: 'str') -> 'None':
        if value == "/history":
            while "assistant> after resume" not in line_output:
                await asyncio.sleep(0.01)

    _install_test_cli_view(
        monkeypatch,
        ["/resume", "/resume 1", "/title", "continue", "/history", "/exit"],
        line_output,
        stream_chunks,
        prompt_hook=prompt_hook,
    )

    code = await run_interactive_session(
        runtime,
        False,
    )

    assert code == 0
    assert model._session_id == thread_id
    assert "Available sessions:" in line_output
    assert "[1] older prompt" in line_output
    assert "Resumed session: saved thread" in line_output
    assert "Session: saved thread" in line_output
    assert "[1]U> older prompt" in line_output
    assert "[1]A> older answer" in line_output
    assert "[2]U> second prompt" in line_output
    assert "[2]A> second answer" in line_output
    assert "[3]U> continue" in line_output
    assert "[3]A> after resume" in line_output

    resumed_prompt_items = [
        item
        for item in model.prompts[0].input
        if isinstance(
            item,
            (
                UserMessage,
                AssistantMessage,
                ReasoningItem,
                ToolCall,
                ToolResult,
            ),
        )
    ]
    assert [type(item).__name__ for item in resumed_prompt_items] == [
        "UserMessage",
        "ReasoningItem",
        "ToolCall",
        "ToolResult",
        "AssistantMessage",
        "UserMessage",
        "AssistantMessage",
        "UserMessage",
    ]
    assert isinstance(resumed_prompt_items[2], ToolCall)
    assert resumed_prompt_items[2].name == "echo"
    assert isinstance(resumed_prompt_items[3], ToolResult)
    assert resumed_prompt_items[3].output == '{"text":"hello"}'
    assert isinstance(resumed_prompt_items[-1], UserMessage)
    assert resumed_prompt_items[-1].text == "continue"


def test_load_resumed_session_applies_compacted_replacement_history(
    tmp_path,
) -> 'None':
    thread_id = "11111111-2222-3333-4444-555555555555"
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "session_index.jsonl").write_text(
        json.dumps(
            {
                "id": thread_id,
                "thread_name": "saved thread",
                "updated_at": "2026-04-07T00:00:00Z",
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    _write_test_rollout(
        codex_home,
        thread_id,
        [
            {
                "timestamp": "2026-04-07T00:00:00Z",
                "type": "session_meta",
                "payload": {
                    "id": thread_id,
                    "timestamp": "2026-04-07T00:00:00Z",
                },
            },
            {
                "timestamp": "2026-04-07T00:00:01Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "old prompt",
                },
            },
            {
                "timestamp": "2026-04-07T00:00:02Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "old answer"}],
                },
            },
            {
                "timestamp": "2026-04-07T00:00:03Z",
                "type": "compacted",
                "payload": {
                    "replacement_history": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "kept user"}],
                        },
                        {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": f"{SUMMARY_PREFIX}\ncheckpoint summary",
                                }
                            ],
                        },
                    ]
                },
            },
            {
                "timestamp": "2026-04-07T00:00:04Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "after compact",
                },
            },
            {
                "timestamp": "2026-04-07T00:00:05Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "after answer"}],
                },
            },
        ],
    )

    resumed = load_resumed_session(codex_home, "1")

    assert [type(item).__name__ for item in resumed["history"]] == [
        "UserMessage",
        "UserMessage",
        "UserMessage",
        "AssistantMessage",
    ]
    assert [item.text for item in resumed["history"][:3]] == [
        "kept user",
        f"{SUMMARY_PREFIX}\ncheckpoint summary",
        "after compact",
    ]
    assert resumed["history"][3].text == "after answer"
    assert resumed["turns"] == (
        ("kept user", ""),
        (f"{SUMMARY_PREFIX}\ncheckpoint summary", ""),
        ("after compact", "after answer"),
    )


@pytest.mark.asyncio
async def test_run_interactive_session_resume_without_args_lists_sessions(
    tmp_path,
    monkeypatch,
) -> 'None':
    named_thread_id = "11111111-2222-3333-4444-555555555555"
    unnamed_thread_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    codex_home = tmp_path / ".codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    codex_home.mkdir()
    (codex_home / "session_index.jsonl").write_text(
        json.dumps(
            {
                "id": named_thread_id,
                "thread_name": "named session",
                "updated_at": "2026-04-07T00:00:00Z",
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    named_rollout = _write_test_rollout(
        codex_home,
        named_thread_id,
        [
            {
                "timestamp": "2026-04-07T00:00:00Z",
                "type": "session_meta",
                "payload": {"id": named_thread_id},
            },
            {
                "timestamp": "2026-04-07T00:00:01Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "named prompt",
                },
            },
            {
                "timestamp": "2026-04-07T00:00:02Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "named answer"}],
                },
            },
        ],
    )
    unnamed_rollout = _write_test_rollout(
        codex_home,
        unnamed_thread_id,
        [
            {
                "timestamp": "2026-04-07T00:00:00Z",
                "type": "session_meta",
                "payload": {"id": unnamed_thread_id},
            },
            {
                "timestamp": "2026-04-07T00:00:01Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "unnamed prompt",
                },
            },
        ],
    )
    os.utime(named_rollout, (100, 100))
    os.utime(unnamed_rollout, (200, 200))

    runtime = CliSubmissionQueue(
        Agent(
            ScriptedModelClient([ModelResponse(items=[AssistantMessage(text="done")])]),
            ToolRegistry(),
        )
    )
    line_output: 'typing.List[str]' = []
    stream_chunks: 'typing.List[str]' = []
    _install_test_cli_view(
        monkeypatch,
        ["/resume", "/exit"],
        line_output,
        stream_chunks,
    )

    code = await run_interactive_session(
        runtime,
        False,
    )

    assert code == 0
    assert "Available sessions:" in line_output
    assert "[1] unnamed prompt" in line_output
    assert "[2] named prompt" in line_output


@pytest.mark.asyncio
async def test_run_interactive_session_resume_rejects_non_numeric_target(
    tmp_path,
    monkeypatch,
) -> 'None':
    codex_home = tmp_path / ".codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    codex_home.mkdir()

    runtime = CliSubmissionQueue(
        Agent(
            ScriptedModelClient([ModelResponse(items=[AssistantMessage(text="done")])]),
            ToolRegistry(),
        )
    )
    line_output: 'typing.List[str]' = []
    stream_chunks: 'typing.List[str]' = []
    _install_test_cli_view(
        monkeypatch,
        ["/resume foo", "/exit"],
        line_output,
        stream_chunks,
    )

    code = await run_interactive_session(
        runtime,
        False,
    )

    assert code == 0
    assert "Error: Usage: /resume <number>" in line_output


@pytest.mark.asyncio
async def test_interactive_session_resume_restores_prior_history_after_restart(
    tmp_path,
    monkeypatch,
) -> 'None':
    config_path = tmp_path / ".codex" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                'model = "demo-model"',
                'model_provider = "demo"',
                '[model_providers.demo]',
                'base_url = "https://example.com/v1"',
                'env_key = "DUMMY_KEY"',
            ]
        )
    )
    first_runtime, first_client = _build_scripted_tui_runtime(
        config_path,
        [
            ModelResponse(items=[AssistantMessage(text="first answer")]),
            ModelResponse(items=[AssistantMessage(text="second answer")]),
        ],
    )
    first_line_output: 'typing.List[str]' = []
    first_stream_chunks: 'typing.List[str]' = []

    async def first_prompt_hook(_view, _prompt: 'str', value: 'str') -> 'None':
        if value == "what next":
            while "assistant> first answer" not in first_line_output:
                await asyncio.sleep(0.01)
        if value == "/exit":
            while "assistant> second answer" not in first_line_output:
                await asyncio.sleep(0.01)

    _install_test_cli_view(
        monkeypatch,
        ["hello", "what next", "/exit"],
        first_line_output,
        first_stream_chunks,
        prompt_hook=first_prompt_hook,
    )

    first_code = await run_interactive_session(
        first_runtime,
        False,
        str(config_path),
    )
    assert first_code == 0
    assert first_client._session_id is not None
    assert list((config_path.parent / "sessions").rglob("rollout-*.jsonl"))

    second_runtime, second_client = _build_scripted_tui_runtime(
        config_path,
        [ModelResponse(items=[AssistantMessage(text="unused")])],
    )
    line_output: 'typing.List[str]' = []
    stream_chunks: 'typing.List[str]' = []
    _install_test_cli_view(
        monkeypatch,
        ["/resume", "/resume 1", "/exit"],
        line_output,
        stream_chunks,
    )

    code = await run_interactive_session(
        second_runtime,
        False,
        str(config_path),
    )

    assert code == 0
    assert second_client._session_id == first_client._session_id
    assert "[1] hello" in line_output
    assert "Resumed session: hello" in line_output
    assert "Session: hello" in line_output
    assert "[1]U> hello" in line_output
    assert "[1]A> first answer" in line_output
    assert "[2]U> what next" in line_output
    assert "[2]A> second answer" in line_output


@pytest.mark.asyncio
async def test_interactive_session_resume_then_continue_updates_history(
    tmp_path,
    monkeypatch,
) -> 'None':
    config_path = tmp_path / ".codex" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                'model = "demo-model"',
                'model_provider = "demo"',
                '[model_providers.demo]',
                'base_url = "https://example.com/v1"',
                'env_key = "DUMMY_KEY"',
            ]
        )
    )

    first_runtime, _first_client = _build_scripted_tui_runtime(
        config_path,
        [ModelResponse(items=[AssistantMessage(text="first answer")])],
    )
    first_line_output: 'typing.List[str]' = []
    first_stream_chunks: 'typing.List[str]' = []
    _install_test_cli_view(
        monkeypatch,
        ["hello", "/exit"],
        first_line_output,
        first_stream_chunks,
    )

    first_code = await run_interactive_session(
        first_runtime,
        False,
        str(config_path),
    )
    assert first_code == 0

    second_runtime, _second_client = _build_scripted_tui_runtime(
        config_path,
        [ModelResponse(items=[AssistantMessage(text="after resume")])],
    )
    line_output: 'typing.List[str]' = []
    stream_chunks: 'typing.List[str]' = []

    async def prompt_hook(_view, _prompt: 'str', value: 'str') -> 'None':
        if value == "/history":
            while "assistant> after resume" not in line_output:
                await asyncio.sleep(0.01)

    _install_test_cli_view(
        monkeypatch,
        ["/resume", "/resume 1", "continue", "/history", "/exit"],
        line_output,
        stream_chunks,
        prompt_hook=prompt_hook,
    )

    code = await run_interactive_session(
        second_runtime,
        False,
        str(config_path),
    )

    assert code == 0
    assert "Resumed session: hello" in line_output
    assert "[1]U> hello" in line_output
    assert "[1]A> first answer" in line_output
    assert "[2]U> continue" in line_output
    assert "[2]A> after resume" in line_output


@pytest.mark.asyncio
async def test_resume_ignores_empty_saved_session(
    tmp_path,
    monkeypatch,
) -> 'None':
    config_path = tmp_path / ".codex" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                'model = "demo-model"',
                'model_provider = "demo"',
                '[model_providers.demo]',
                'base_url = "https://example.com/v1"',
                'env_key = "DUMMY_KEY"',
            ]
        )
    )

    first_runtime, _first_client = _build_scripted_tui_runtime(
        config_path,
        [ModelResponse(items=[AssistantMessage(text="unused")])],
    )
    first_line_output: 'typing.List[str]' = []
    first_stream_chunks: 'typing.List[str]' = []
    _install_test_cli_view(
        monkeypatch,
        ["/exit"],
        first_line_output,
        first_stream_chunks,
    )

    first_code = await run_interactive_session(
        first_runtime,
        False,
        str(config_path),
    )
    assert first_code == 0

    second_runtime, _second_client = _build_scripted_tui_runtime(
        config_path,
        [ModelResponse(items=[AssistantMessage(text="unused")])],
    )
    line_output: 'typing.List[str]' = []
    stream_chunks: 'typing.List[str]' = []
    _install_test_cli_view(
        monkeypatch,
        ["/resume", "/exit"],
        line_output,
        stream_chunks,
    )

    code = await run_interactive_session(
        second_runtime,
        False,
        str(config_path),
    )

    assert code == 0
    assert "No resumable sessions found." in line_output


@pytest.mark.asyncio
async def test_resume_restores_saved_tool_call_history(
    tmp_path,
    monkeypatch,
) -> 'None':
    config_path = tmp_path / ".codex" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                'model = "demo-model"',
                'model_provider = "demo"',
                '[model_providers.demo]',
                'base_url = "https://example.com/v1"',
                'env_key = "DUMMY_KEY"',
            ]
        )
    )

    first_runtime, _first_client = _build_scripted_tui_runtime(
        config_path,
        [
            ModelResponse(
                items=[
                    ToolCall(
                        call_id="call_1",
                        name="update_plan",
                        arguments={
                            "explanation": "track work",
                            "plan": [
                                {"step": "saved step", "status": "completed"},
                            ],
                        },
                    )
                ]
            ),
            ModelResponse(items=[AssistantMessage(text="tool-backed answer")]),
        ],
    )
    first_line_output: 'typing.List[str]' = []
    first_stream_chunks: 'typing.List[str]' = []
    _install_test_cli_view(
        monkeypatch,
        ["use tool", "/exit"],
        first_line_output,
        first_stream_chunks,
    )

    first_code = await run_interactive_session(
        first_runtime,
        False,
        str(config_path),
    )
    assert first_code == 0

    second_runtime, second_client = _build_scripted_tui_runtime(
        config_path,
        [ModelResponse(items=[AssistantMessage(text="after resume")])],
    )
    line_output: 'typing.List[str]' = []
    stream_chunks: 'typing.List[str]' = []
    _install_test_cli_view(
        monkeypatch,
        ["/resume", "/resume 1", "continue", "/exit"],
        line_output,
        stream_chunks,
    )

    code = await run_interactive_session(
        second_runtime,
        False,
        str(config_path),
    )

    assert code == 0
    assert "Resumed session: use tool" in line_output
    resumed_prompt_items = [
        item
        for item in second_client.prompts[0].input
        if isinstance(
            item,
            (
                UserMessage,
                AssistantMessage,
                ReasoningItem,
                ToolCall,
                ToolResult,
            ),
        )
    ]
    assert [type(item).__name__ for item in resumed_prompt_items] == [
        "UserMessage",
        "ToolCall",
        "ToolResult",
        "AssistantMessage",
        "UserMessage",
    ]
    assert isinstance(resumed_prompt_items[1], ToolCall)
    assert resumed_prompt_items[1].name == "update_plan"
    assert isinstance(resumed_prompt_items[2], ToolResult)
    assert resumed_prompt_items[2].name == "update_plan"
    assert isinstance(resumed_prompt_items[-1], UserMessage)
    assert resumed_prompt_items[-1].text == "continue"


@pytest.mark.asyncio
async def test_run_interactive_session_supports_model_command(
    tmp_path,
    monkeypatch,
) -> 'None':
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'model = "demo-model"',
                'model_provider = "demo"',
                '[model_providers.demo]',
                'base_url = "https://example.com/v1"',
                'env_key = "DUMMY_KEY"',
            ]
        )
    )

    class _FakeResponsesModelClient:
        def __init__(
            self,
            config,
            timeout_seconds,
            session_id=None,
            originator=None,
            user_agent=None,
            openai_subagent=None,
        ):
            self._config = config
            self.model = config.model
            self._timeout_seconds = timeout_seconds
            self._session_id = session_id
            self._originator = originator
            self._user_agent = user_agent
            self._openai_subagent = openai_subagent

        def with_overrides(
            self,
            model=None,
            reasoning_effort=None,
            session_id=None,
            openai_subagent=None,
        ):
            del reasoning_effort
            config = self._config
            if model is not None:
                config = replace(config, model=model)
            return _FakeResponsesModelClient(
                config,
                self._timeout_seconds,
                session_id=self._session_id if session_id is None else session_id,
                originator=self._originator,
                user_agent=self._user_agent,
                openai_subagent=(
                    self._openai_subagent
                    if openai_subagent is None
                    else openai_subagent
                ),
            )

        async def complete(self, prompt, event_handler):
            del prompt
            text = self.model
            event_handler(ModelStreamEvent(kind="assistant_delta", payload={"delta": text}))
            return ModelResponse(items=[AssistantMessage(text=text)])

        async def list_models(self):
            return ["demo-model", "alt-model", "third-model"]

    monkeypatch.setattr("pycodex.cli.ResponsesModelClient", _FakeResponsesModelClient)

    args = build_parser().parse_args(["--config", str(config_path)])
    client = build_model(
        args.config,
        args.profile,
        args.timeout_seconds,
    )
    agent = build_agent(
        client,
        config_path=args.config,
        profile=args.profile,
        system_prompt=args.system_prompt,
        session_mode="tui",
    )
    runtime = build_cli_queue(agent)
    line_output: 'typing.List[str]' = []
    stream_chunks: 'typing.List[str]' = []
    async def prompt_hook(_view, _prompt: 'str', value: 'str') -> 'None':
        if value == "/model alt-model":
            while "assistant> demo-model" not in line_output:
                await asyncio.sleep(0.01)
        elif value == "/model":
            while "assistant> alt-model" not in line_output:
                await asyncio.sleep(0.01)

    _install_test_cli_view(
        monkeypatch,
        ["hello", "/model alt-model", "again", "/model", "/exit"],
        line_output,
        stream_chunks,
        prompt_hook=prompt_hook,
    )

    code = await run_interactive_session(
        runtime,
        False,
    )

    assert code == 0
    assert line_output == [
        "pycodex interactive mode. Type /exit to quit.",
        EXTRA_COMMANDS_LINE,
        "Session: hello",
        "user> hello",
        "assistant> demo-model",
        "Switched model to alt-model.",
        "user> again",
        "assistant> alt-model",
        "Current model: alt-model",
        "Available models: demo-model, alt-model, third-model",
    ]
    assert stream_chunks == []


@pytest.mark.asyncio
async def test_run_interactive_session_rejects_model_switch_while_steer_work_pending(
    monkeypatch,
) -> 'None':
    release_turn = asyncio.Event()

    class _DelayedResponsesModelClient:
        def __init__(self) -> 'None':
            self.model = "demo-model"

        async def complete(self, prompt, event_handler):
            del prompt
            await release_turn.wait()
            event_handler(
                ModelStreamEvent(
                    kind="assistant_delta",
                    payload={"delta": self.model},
                )
            )
            return ModelResponse(items=[AssistantMessage(text=self.model)])

        async def list_models(self):
            return ["demo-model", "alt-model"]

    model = _DelayedResponsesModelClient()
    runtime = CliSubmissionQueue(Agent(model, ToolRegistry()))
    line_output: 'typing.List[str]' = []
    stream_chunks: 'typing.List[str]' = []

    async def prompt_hook(_view, _prompt: 'str', value: 'str') -> 'None':
        if value == "/exit":
            release_turn.set()

    _install_test_cli_view(
        monkeypatch,
        ["hello", "/model alt-model", "/exit"],
        line_output,
        stream_chunks,
        prompt_hook=prompt_hook,
    )

    code = await run_interactive_session(
        runtime,
        False,
    )

    assert code == 0
    assert model.model == "demo-model"
    assert line_output[:2] == [
        "pycodex interactive mode. Type /exit to quit.",
        EXTRA_COMMANDS_LINE,
    ]
    assert "Session: hello" in line_output
    assert "Cannot change model while work is running or queued in steer mode." in line_output
    assert "assistant> demo-model" in line_output
    assert stream_chunks == []


@pytest.mark.asyncio
async def test_run_interactive_session_continues_after_model_error(
    monkeypatch,
) -> 'None':
    class _FailOnceModelClient:
        def __init__(self) -> 'None':
            self.call_count = 0

        async def complete(self, prompt, event_handler):
            del prompt
            self.call_count += 1
            if self.call_count == 1:
                raise RuntimeError("synthetic client error")
            event_handler(
                ModelStreamEvent(kind="assistant_delta", payload={"delta": "done"})
            )
            return ModelResponse(items=[AssistantMessage(text="done")])

    runtime = CliSubmissionQueue(Agent(_FailOnceModelClient(), ToolRegistry()))
    line_output: 'typing.List[str]' = []
    stream_chunks: 'typing.List[str]' = []
    saw_error = asyncio.Event()

    async def prompt_hook(_view, _prompt: 'str', value: 'str') -> 'None':
        if value == "again":
            await saw_error.wait()

    _install_test_cli_view(
        monkeypatch,
        ["hello", "again", "/exit"],
        line_output,
        stream_chunks,
        prompt_hook=prompt_hook,
        line_callback=(
            lambda text: saw_error.set()
            if text == "Error: synthetic client error"
            else None
        ),
    )

    code = await run_interactive_session(
        runtime,
        False,
    )

    assert code == 0
    assert line_output == [
        "pycodex interactive mode. Type /exit to quit.",
        EXTRA_COMMANDS_LINE,
        "Session: hello",
        "user> hello",
        "Error: synthetic client error",
        "user> again",
        "assistant> done",
    ]
    assert stream_chunks == []


class _EchoTool(BaseTool):
    name = "echo"
    description = "Echo text."
    input_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
        },
        "required": ["text"],
    }

    async def run(self, context, args):
        del context
        return {"echo": args["text"]}


@pytest.mark.asyncio
async def test_run_interactive_session_shows_tool_progress_without_iteration_noise(
    monkeypatch,
) -> 'None':
    model = ScriptedModelClient(
        [
            ModelResponse(
                items=[
                    ToolCall(
                        call_id="call_1",
                        name="echo",
                        arguments={"text": "hello"},
                    )
                ]
            ),
            ModelResponse(items=[AssistantMessage(text="done")]),
        ]
    )
    tools = ToolRegistry()
    tools.register(_EchoTool())
    runtime = CliSubmissionQueue(Agent(model, tools))
    line_output: 'typing.List[str]' = []
    stream_chunks: 'typing.List[str]' = []
    _install_test_cli_view(
        monkeypatch,
        ["use a tool", "/exit"],
        line_output,
        stream_chunks,
    )

    code = await run_interactive_session(
        runtime,
        False,
    )

    assert code == 0
    assert line_output == [
        "pycodex interactive mode. Type /exit to quit.",
        EXTRA_COMMANDS_LINE,
        "Session: use a tool",
        "user> use a tool",
        '[echo] {"echo":"hello"}',
        "assistant> done",
    ]
    assert not any("iteration" in line for line in line_output)


def test_cli_session_view_formats_plan_and_exec_messages() -> 'None':
    output: 'typing.List[str]' = []
    view = _build_cli_view(output)

    view.handle_event(
        AgentEvent(
            kind="tool_completed",
            turn_id="turn_1",
            payload={
                "tool_name": "update_plan",
                "summary": "Working on 2/2",
                "plan_items": [
                    {"step": "inspect repo", "status": "completed"},
                    {"step": "collect docs", "status": "in_progress"},
                ],
                "is_error": False,
            },
        )
    )
    view.handle_event(
        AgentEvent(
            kind="tool_completed",
            turn_id="turn_1",
            payload={
                "tool_name": "exec_command",
                "summary": "pwd && ls -la -> /data/codex-python",
                "is_error": False,
            },
        )
    )
    view.handle_event(
        AgentEvent(
            kind="tool_completed",
            turn_id="turn_1",
            payload={
                "tool_name": "spawn_agent",
                "summary": "Curie (019d25a7...a8c6)",
                "is_error": False,
            },
        )
    )
    view.handle_event(
        AgentEvent(
            kind="tool_completed",
            turn_id="turn_1",
            payload={
                "tool_name": "wait_agent",
                "summary": "019d25a7...a8c6=completed: listed tools",
                "is_error": False,
            },
        )
    )

    assert output == [
        "[update_plan] Working on 2/2",
        "  [x] inspect repo",
        "  [>] collect docs",
        "[exec_command] pwd && ls -la -> /data/codex-python",
        "[spawn_agent] spawned Curie (019d25a7...a8c6)",
        "[wait_agent] Curie=completed: listed tools",
    ]


def test_colorize_cli_message_wraps_ansi_when_enabled() -> 'None':
    colored = colorize_cli_message("[status] working", "status", True)
    assert colored.startswith("\x1b[1m\x1b[36m")
    assert colored.endswith("\x1b[0m")


def test_colorize_tool_message_uses_tool_name_palette() -> 'None':
    colored = colorize_tool_message("[spawn_agent] spawned Curie", True)
    assert colored.startswith("\x1b[1m\x1b[34m")
    assert colored.endswith("\x1b[0m")


def test_colorize_tool_message_uses_default_highlight_for_exec_code() -> 'None':
    colored = colorize_tool_message("import json", True, "exec_command")
    assert colored == "\x1b[1mimport json\x1b[0m"


def test_tool_summary_extracts_python_heredoc_and_session_id() -> 'None':
    command = "python3 - <<'PY'\nimport json\nprint(json.dumps({'ok': True}))\nPY"
    call = ToolCall(
        call_id="call_1",
        name="exec_command",
        arguments={"cmd": command},
    )
    result = ToolResult(
        call_id="call_1",
        name="exec_command",
        output='Exit code: 0\nOutput:\n{"ok": true}\n',
    )

    assert tool_summary(
        {
            "tool_name": "exec_command",
            "call": call,
            "result": result,
            "is_error": False,
        }
    ) == (
        "[exec_command] python3 - <<'PY'\n"
        "import json\n"
        "print(json.dumps({'ok': True}))\n"
        'PY -> {"ok": true}'
    )

    running = ToolResult(
        call_id="call_1",
        name="exec_command",
        output="Process running with session ID 7\n",
    )
    assert tool_summary(
        {
            "tool_name": "exec_command",
            "call": call,
            "result": running,
            "is_error": False,
        }
    ).endswith("PY -> session_id=7")


def test_cli_session_view_shows_web_search_tool_called_message() -> 'None':
    output: 'typing.List[str]' = []
    view = _build_cli_view(output)

    view.handle_event(
        AgentEvent(
            kind="tool_called",
            turn_id="turn_1",
            payload={
                "tool_name": "web_search",
                "call_id": "ws_1",
                "action_type": "search",
                "query": "github codex",
            },
        )
    )
    view.handle_event(
        AgentEvent(
            kind="tool_called",
            turn_id="turn_1",
            payload={
                "tool_name": "web_search",
                "call_id": "ws_2",
                "action_type": "open_page",
                "url": "http://example.com",
            },
        )
    )

    assert output == [
        "[web_search] searched: github codex",
        "[web_search] opened: http://example.com",
    ]


def test_cli_session_view_turn_failed_clears_pending_prompt() -> 'None':
    output: 'typing.List[str]' = []
    view = _build_cli_view(output)
    view.handle_event(
        AgentEvent(
            kind="turn_started",
            turn_id="turn_1",
            payload={"user_text": "hello"},
        )
    )

    view.handle_event(
        AgentEvent(
            kind="turn_failed",
            turn_id="turn_1",
            payload={"error": "synthetic client error"},
        )
    )

    assert view._pending_user_prompts == {}
    assert view.prompter._status is None
    assert output == ["Session: hello", "user> hello"]


def test_cli_session_view_shows_stream_error_message() -> 'None':
    output: 'typing.List[str]' = []
    view = _build_cli_view(output)

    view.handle_event(
        AgentEvent(
            kind="turn_started",
            turn_id="turn_1",
            payload={"user_text": "hello"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="stream_error",
            turn_id="turn_1",
            payload={"message": "Reconnecting... 1/5"},
        )
    )

    assert output == [
        "Session: hello",
        "user> hello",
        "[status] Reconnecting... 1/5",
    ]
    assert view.prompter._status == "reconnecting"


def test_cli_session_view_discards_stream_buffer_before_retry_status() -> 'None':
    output: 'typing.List[str]' = []
    view = _build_cli_view(output)

    view.handle_event(
        AgentEvent(
            kind="turn_started",
            turn_id="turn_1",
            payload={"user_text": "hello"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="assistant_delta",
            turn_id="turn_1",
            payload={"delta": "same final"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="stream_error",
            turn_id="turn_1",
            payload={"message": "Reconnecting... 1/5"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="assistant_delta",
            turn_id="turn_1",
            payload={"delta": "same final"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="turn_completed",
            turn_id="turn_1",
            payload={"output_text": "same final"},
        )
    )

    assert output == [
        "Session: hello",
        "user> hello",
        "[status] Reconnecting... 1/5",
        "assistant> same final",
    ]


def test_cli_session_view_flushes_stream_buffer_on_turn_failure() -> 'None':
    output: 'typing.List[str]' = []
    view = _build_cli_view(output)

    view.handle_event(
        AgentEvent(
            kind="turn_started",
            turn_id="turn_1",
            payload={"user_text": "hello"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="assistant_delta",
            turn_id="turn_1",
            payload={"delta": "partial"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="turn_failed",
            turn_id="turn_1",
            payload={"error": "synthetic"},
        )
    )

    assert output == [
        "Session: hello",
        "user> hello",
        "assistant> partial",
    ]


def test_cli_session_view_renders_tool_completion_with_current_status() -> 'None':
    output: 'typing.List[str]' = []
    view = _build_cli_view(output)

    view.handle_event(
        AgentEvent(
            kind="turn_started",
            turn_id="turn_1",
            payload={"user_text": "hello"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="model_called",
            turn_id="turn_1",
            payload={},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="tool_started",
            turn_id="turn_1",
            payload={"tool_name": "exec_command"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="tool_completed",
            turn_id="turn_1",
            payload={
                "tool_name": "exec_command",
                "summary": "pwd",
                "is_error": False,
            },
        )
    )

    assert view.prompter._status == "called exec_command"
    assert output == ["Session: hello", "user> hello", "[exec_command] pwd"]


def test_cli_session_view_tool_started_uses_call_arguments_in_status_label() -> 'None':
    output: 'typing.List[str]' = []
    view = _build_cli_view(output)

    view.handle_event(
        AgentEvent(
            kind="tool_started",
            turn_id="turn_1",
            payload={
                "tool_name": "exec_command",
                "call_id": "call_1",
                "call": ToolCall(
                    call_id="call_1",
                    name="exec_command",
                    arguments={"cmd": "pwd"},
                ),
            },
        )
    )

    assert view.prompter._status == "calling exec_command({'cmd': 'pwd'})"


def test_cli_session_view_keeps_prompt_text_unchanged_without_context_usage() -> 'None':
    output: 'typing.List[str]' = []
    view = _build_cli_view(output)

    view.handle_event(
        AgentEvent(
            kind="turn_started",
            turn_id="turn_1",
            payload={"user_text": "hello"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="model_called",
            turn_id="turn_1",
            payload={},
        )
    )
    assert view.prompter.prompt == "pycodex> "


def test_cli_session_view_formats_prompt_with_context_remaining_percent() -> 'None':
    output: 'typing.List[str]' = []
    view = _configure_cli_view_output(
        CliSessionView(context_window_tokens=100_000),
        output,
    )
    view.handle_event(
        AgentEvent(
            kind="token_count",
            turn_id="turn_1",
            payload={"usage": {"total_tokens": 20_800}},
        )
    )
    assert view.prompter.prompt == "pyco(90%)> "


def test_cli_session_view_keeps_failed_context_usage() -> 'None':
    output: 'typing.List[str]' = []
    view = _configure_cli_view_output(
        CliSessionView(context_window_tokens=262_144),
        output,
    )
    view.handle_event(
        AgentEvent(
            kind="token_count",
            turn_id="turn_1",
            payload={
                "usage": {
                    "total_tokens": 264_568,
                    "input_tokens": 264_568,
                    "output_tokens": 0,
                }
            },
        )
    )
    view.handle_event(
        AgentEvent(
            kind="turn_failed",
            turn_id="turn_1",
            payload={"error": "context_length_exceeded"},
        )
    )
    assert view.prompter.prompt == "pyco(0%)> "


def test_cli_session_view_shows_auto_compact_events() -> 'None':
    output: 'typing.List[str]' = []
    view = _configure_cli_view_output(CliSessionView(), output)

    view.handle_event(
        AgentEvent(
            kind="auto_compact_started",
            turn_id="turn_1",
            payload={"total_tokens": 90, "token_limit": 100},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="auto_compact_completed",
            turn_id="turn_1",
            payload={"summary": "compact(4 items) -> 2 items + [summary]"},
        )
    )

    assert output == [
        "[status] auto-compact: 90/100 tokens",
        "[status] compact(4 items) -> 2 items + [summary]",
    ]


def test_cli_session_view_shows_full_context_on_initial_prompt() -> 'None':
    output: 'typing.List[str]' = []
    view = _configure_cli_view_output(
        CliSessionView(context_window_tokens=100_000),
        output,
    )
    assert view._format_main_prompt("pycodex> ") == "pyco(100%)> "


def test_cli_session_view_buffers_streaming_text() -> 'None':
    output: 'typing.List[str]' = []
    view = _build_cli_view(output)

    view.handle_event(
        AgentEvent(
            kind="turn_started",
            turn_id="turn_1",
            payload={"user_text": "hello"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="assistant_delta",
            turn_id="turn_1",
            payload={"delta": "hi"},
        )
    )

    assert view._stream_buffer == "hi"
    assert output == ["Session: hello", "user> hello"]


def test_cli_session_view_buffers_streaming_text_without_prompt_state() -> 'None':
    output: 'typing.List[str]' = []
    view = _build_cli_view(output)

    view.handle_event(
        AgentEvent(
            kind="assistant_delta",
            turn_id="turn_1",
            payload={"delta": "hi"},
        )
    )

    assert view._stream_buffer == "hi"


def test_cli_session_view_preserves_prompt_managed_stream_output_on_completion() -> 'None':
    output: 'typing.List[str]' = []
    view = _build_cli_view(output)

    view.handle_event(
        AgentEvent(
            kind="assistant_delta",
            turn_id="turn_1",
            payload={"delta": "hi"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="turn_completed",
            turn_id="turn_1",
            payload={"output_text": "hi"},
        )
    )

    assert output == ["assistant> hi"]


def test_cli_session_view_finish_stream_flushes_buffered_output() -> 'None':
    output: 'typing.List[str]' = []
    stream_chunks: 'typing.List[str]' = []
    view = _build_cli_view(output, stream_chunks)

    view.handle_event(
        AgentEvent(
            kind="assistant_delta",
            turn_id="turn_1",
            payload={"delta": "hi"},
        )
    )
    view.finish_stream()

    assert output == ["assistant> hi"]
    assert stream_chunks == []


def test_cli_session_view_turn_completed_clears_status() -> 'None':
    output: 'typing.List[str]' = []
    view = _build_cli_view(output)

    view.handle_event(
        AgentEvent(
            kind="turn_started",
            turn_id="turn_1",
            payload={"user_text": "hello"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="model_called",
            turn_id="turn_1",
            payload={},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="turn_completed",
            turn_id="turn_1",
            payload={"output_text": ""},
        )
    )

    assert view.prompter._status is None


def test_cli_session_view_turn_completed_listens_with_background_work() -> 'None':
    output: 'typing.List[str]' = []
    view = _build_cli_view(output)

    view.handle_event(
        AgentEvent(
            kind="turn_started",
            turn_id="turn_1",
            payload={"user_text": "hello"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="turn_completed",
            turn_id="turn_1",
            payload={"output_text": "", "background_exec_count": 1},
        )
    )

    assert view.prompter._status == "idle: listening"


def test_cli_session_view_turn_completed_clears_status_without_background_work() -> 'None':
    output: 'typing.List[str]' = []
    view = _build_cli_view(output)

    view.handle_event(
        AgentEvent(
            kind="turn_started",
            turn_id="turn_1",
            payload={"user_text": "hello"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="turn_completed",
            turn_id="turn_1",
            payload={"output_text": "", "background_exec_count": 0},
        )
    )

    assert view.prompter._status is None


def test_cli_session_view_shows_steer_queue_and_insert_messages() -> 'None':
    output: 'typing.List[str]' = []
    view = _build_cli_view(output)

    view.show_steer_queued("turn_2", "follow up prompt")
    view.handle_event(
        AgentEvent(
            kind="turn_started",
            turn_id="turn_2",
            payload={"user_text": "follow up prompt"},
        )
    )

    assert output == [
        "[steer] queued: follow up prompt",
        "Session: follow up prompt",
        "[steer] inserted: follow up prompt",
        "user> follow up prompt",
    ]


def test_cli_session_view_preserves_prompt_managed_stream_output_on_interrupt() -> 'None':
    output: 'typing.List[str]' = []
    view = _build_cli_view(output)

    view.handle_event(
        AgentEvent(
            kind="turn_started",
            turn_id="turn_1",
            payload={"user_text": "say hi", "submission_id": "sub_1"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="assistant_delta",
            turn_id="turn_1",
            payload={"delta": "hi", "submission_id": "sub_1"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="turn_interrupted",
            turn_id="turn_1",
            payload={"output_text": "hi", "submission_id": "sub_1"},
        )
    )

    assert output == [
        "Session: say hi",
        "user> say hi",
        "assistant> hi",
    ]
    assert view._history == [("say hi", "hi")]


def test_cli_session_view_keeps_history_for_reused_turn_id_across_submissions() -> 'None':
    output: 'typing.List[str]' = []
    stream_chunks: 'typing.List[str]' = []
    view = _build_cli_view(output, stream_chunks)

    view.handle_event(
        AgentEvent(
            kind="turn_started",
            turn_id="turn_1",
            payload={"user_text": "say hi", "submission_id": "sub_1"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="assistant_delta",
            turn_id="turn_1",
            payload={"delta": "hi", "submission_id": "sub_1"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="turn_interrupted",
            turn_id="turn_1",
            payload={"output_text": "hi", "submission_id": "sub_1"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="turn_started",
            turn_id="turn_1",
            payload={"user_text": "say bye instead", "submission_id": "sub_2"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="assistant_delta",
            turn_id="turn_1",
            payload={"delta": "bye", "submission_id": "sub_2"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="turn_completed",
            turn_id="turn_1",
            payload={"output_text": "bye", "submission_id": "sub_2"},
        )
    )

    assert view._history == [
        ("say hi", "hi"),
        ("say bye instead", "bye"),
    ]


def test_cli_session_view_assistant_stream_updates_status_until_stream_finishes() -> 'None':
    output: 'typing.List[str]' = []
    stream_chunks: 'typing.List[str]' = []
    view = _build_cli_view(output, stream_chunks)

    view.handle_event(
        AgentEvent(
            kind="turn_started",
            turn_id="turn_1",
            payload={"user_text": "hello"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="model_called",
            turn_id="turn_1",
            payload={},
        )
    )

    view.handle_event(
        AgentEvent(
            kind="assistant_delta",
            turn_id="turn_1",
            payload={"delta": "Hel"},
        )
    )
    view.handle_event(
        AgentEvent(
            kind="assistant_delta",
            turn_id="turn_1",
            payload={"delta": "lo"},
        )
    )

    assert output == ["Session: hello", "user> hello"]
    assert stream_chunks == []
    assert view._stream_buffer == "Hello"
    assert view.prompter._status == "talking"

    view.handle_event(
        AgentEvent(
            kind="tool_started",
            turn_id="turn_1",
            payload={"tool_name": "exec_command"},
        )
    )

    assert view.prompter._status == "calling exec_command"


@pytest.mark.asyncio
async def test_run_cli_non_interactive_uses_tui_context_for_default_cli(
    tmp_path,
    monkeypatch,
) -> 'None':
    capture_root = tmp_path / "capture"
    capture_store = CaptureStore(capture_root)
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        build_handler(capture_store, "gpt-5.4", "OK"),
    )
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'model = "gpt-5.4"',
                'model_provider = "neo"',
                'model_reasoning_summary = "auto"',
                'model_reasoning_effort = "medium"',
                'model_verbosity = "medium"',
                'approval_policy = "never"',
                'sandbox_mode = "danger-full-access"',
                'personality = "pragmatic"',
                '',
                '[features]',
                'guardian_approval = true',
                '',
                '[model_providers.neo]',
                f'base_url = "http://127.0.0.1:{httpd.server_port}/v1"',
                'env_key = "NEO_KEY"',
                'wire_api = "responses"',
            ]
        )
    )
    monkeypatch.setenv("NEO_KEY", "dummy-key")
    monkeypatch.setattr("pycodex.cli.configure_loguru", lambda: None)
    monkeypatch.setattr("sys.stdin.read", lambda: "")

    try:
        args = build_parser().parse_args(
            ["Reply with OK only.", "--config", str(config_path)]
        )
        exit_code = await run_cli(args)
    finally:
        httpd.shutdown()
        server_thread.join(timeout=5)
        httpd.server_close()

    assert exit_code == 0

    request_files = sorted(capture_root.glob("*_POST_*.json"))
    assert len(request_files) == 1
    request = json.loads(request_files[0].read_text())
    body = request["body"]

    headers = _normalized_headers(request["headers"])
    assert headers["originator"] == "codex-tui"
    assert headers["user-agent"].startswith("codex-tui/")
    assert "(codex-tui;" in headers["user-agent"]
    turn_metadata = json.loads(headers["x-codex-turn-metadata"])
    assert turn_metadata["sandbox"] == "none"
    assert turn_metadata["turn_id"]
    assert str(Path.cwd()) in turn_metadata["workspaces"]
    assert [tool.get("name", tool.get("type")) for tool in body["tools"]] == [
        "exec_command",
        "write_stdin",
        "update_plan",
        "request_user_input",
        "apply_patch",
        "web_search",
        "view_image",
        "spawn_agent",
        "send_input",
        "resume_agent",
        "wait_agent",
        "close_agent",
    ]
    assert body["input"][0]["role"] == "developer"
    assert body["input"][0]["content"][1]["text"].startswith(
        "<collaboration_mode># Collaboration Mode: Default"
    )


@pytest.mark.asyncio
async def test_run_interactive_session_preserves_tui_context_across_turns(
    tmp_path,
    monkeypatch,
) -> 'None':
    capture_root = tmp_path / "capture_repl"
    capture_store = CaptureStore(capture_root)
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        build_handler(capture_store, "gpt-5.4", "OK"),
    )
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'model = "gpt-5.4"',
                'model_provider = "neo"',
                'model_reasoning_summary = "auto"',
                'model_reasoning_effort = "medium"',
                'model_verbosity = "medium"',
                'approval_policy = "never"',
                'sandbox_mode = "danger-full-access"',
                'personality = "pragmatic"',
                '',
                '[features]',
                'guardian_approval = true',
                '',
                '[model_providers.neo]',
                f'base_url = "http://127.0.0.1:{httpd.server_port}/v1"',
                'env_key = "NEO_KEY"',
                'wire_api = "responses"',
            ]
        )
    )
    monkeypatch.setenv("NEO_KEY", "dummy-key")

    outputs: 'typing.List[str]' = []
    stream_chunks: 'typing.List[str]' = []

    async def prompt_hook(_view, _prompt: 'str', value: 'str') -> 'None':
        if value == "Reply with TWO only.":
            while len(list(capture_root.glob("*_POST_*.json"))) < 1:
                await asyncio.sleep(0.01)
            while runtime._has_active_turn():
                await asyncio.sleep(0.01)

    try:
        _install_test_cli_view(
            monkeypatch,
            ["Reply with ONE only.", "Reply with TWO only.", "/exit"],
            outputs,
            stream_chunks,
            prompt_hook=prompt_hook,
        )
        args = build_parser().parse_args(["--config", str(config_path)])
        client = build_model(
            args.config,
            args.profile,
            args.timeout_seconds,
        )
        agent = build_agent(
            client,
            config_path=args.config,
            profile=args.profile,
            system_prompt=args.system_prompt,
            session_mode="tui",
        )
        runtime = build_cli_queue(agent)
        code = await run_interactive_session(
            runtime,
            False,
        )
    finally:
        httpd.shutdown()
        server_thread.join(timeout=5)
        httpd.server_close()

    assert code == 0
    request_files = sorted(capture_root.glob("*_POST_*.json"))
    assert len(request_files) == 2

    first_request = json.loads(request_files[0].read_text())
    second_request = json.loads(request_files[1].read_text())
    first_body = first_request["body"]
    second_body = second_request["body"]

    first_headers = _normalized_headers(first_request["headers"])
    second_headers = _normalized_headers(second_request["headers"])
    assert first_headers["originator"] == "codex-tui"
    assert second_headers["originator"] == "codex-tui"
    assert list(json.loads(first_headers["x-codex-turn-metadata"])) == [
        "turn_id",
        "workspaces",
        "sandbox",
    ]
    assert list(json.loads(second_headers["x-codex-turn-metadata"])) == [
        "turn_id",
        "sandbox",
    ]
    assert (
        json.loads(first_headers["x-codex-turn-metadata"])["turn_id"]
        != json.loads(second_headers["x-codex-turn-metadata"])["turn_id"]
    )
    assert first_body["prompt_cache_key"] == second_body["prompt_cache_key"]
    assert len(first_body["input"]) == 3
    assert len(second_body["input"]) == 5
    assert second_body["input"][3] == {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "OK"}],
    }
    assert second_body["input"][4] == {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "Reply with TWO only."}],
    }


@pytest.mark.asyncio
async def test_run_interactive_session_can_resume_after_network_drop_with_go_on(
    tmp_path,
    monkeypatch,
) -> 'None':
    captured_bodies: 'typing.List[typing.Dict[str, object]]' = []
    request_count = {"value": 0}

    class _DropThenRecoverHandler(BaseHTTPRequestHandler):
        server_version = "DropThenRecover/0.1"

        def log_message(self, format: 'str', *args) -> 'None':
            del format, args
            return

        def do_POST(self) -> 'None':
            if self.path != "/v1/responses":
                self.send_error(404)
                return

            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            captured_bodies.append(json.loads(body.decode("utf-8")))
            request_count["value"] += 1

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            if request_count["value"] == 1:
                partial_sse = "".join(
                    [
                        'event: response.created\n',
                        'data: {"type":"response.created","response":{"id":"resp_drop","object":"response","status":"in_progress"}}\n\n',
                        'event: response.output_text.delta\n',
                        'data: {"type":"response.output_text.delta","delta":"partial"}\n\n',
                    ]
                )
                self.wfile.write(partial_sse.encode("utf-8"))
                self.wfile.flush()
                self.connection.shutdown(2)
                self.connection.close()
                return

            ok_sse = "".join(
                [
                    'event: response.created\n',
                    'data: {"type":"response.created","response":{"id":"resp_ok","object":"response","status":"in_progress"}}\n\n',
                    'event: response.output_item.done\n',
                    'data: {"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"RESUMED"}]}}\n\n',
                    'event: response.completed\n',
                    'data: {"type":"response.completed","response":{"id":"resp_ok","output":[]}}\n\n',
                ]
            )
            self.wfile.write(ok_sse.encode("utf-8"))
            self.wfile.flush()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _DropThenRecoverHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'model = "gpt-5.4"',
                'model_provider = "neo"',
                'model_reasoning_summary = "auto"',
                'model_reasoning_effort = "medium"',
                'model_verbosity = "medium"',
                'approval_policy = "never"',
                'sandbox_mode = "danger-full-access"',
                'personality = "pragmatic"',
                '',
                '[features]',
                'guardian_approval = true',
                '',
                '[model_providers.neo]',
                f'base_url = "http://127.0.0.1:{httpd.server_port}/v1"',
                'env_key = "NEO_KEY"',
                'wire_api = "responses"',
                'stream_max_retries = 0',
            ]
        )
    )
    monkeypatch.setenv("NEO_KEY", "dummy-key")

    outputs: 'typing.List[str]' = []
    stream_chunks: 'typing.List[str]' = []
    saw_error = asyncio.Event()

    async def prompt_hook(_view, _prompt: 'str', value: 'str') -> 'None':
        if value == "go on":
            await saw_error.wait()

    try:
        _install_test_cli_view(
            monkeypatch,
            ["Analyze current directory", "go on", "/exit"],
            outputs,
            stream_chunks,
            prompt_hook=prompt_hook,
            line_callback=(
                lambda text: saw_error.set()
                if text.startswith("Error: responses ")
                else None
            ),
        )
        args = build_parser().parse_args(["--config", str(config_path)])
        client = build_model(
            args.config,
            args.profile,
            args.timeout_seconds,
        )
        agent = build_agent(
            client,
            config_path=args.config,
            profile=args.profile,
            system_prompt=args.system_prompt,
            session_mode="tui",
        )
        runtime = build_cli_queue(agent)
        code = await run_interactive_session(
            runtime,
            False,
        )
    finally:
        httpd.shutdown()
        server_thread.join(timeout=5)
        httpd.server_close()

    assert code == 0
    assert request_count["value"] == 2
    assert outputs[:4] == [
        "pycodex interactive mode. Type /exit to quit.",
        EXTRA_COMMANDS_LINE,
        "Session: Analyze current directory",
        "user> Analyze current directory",
    ]
    assert outputs[4] == "assistant> partial"
    error_line = next(line for line in outputs if line.startswith("Error: responses "))
    assert "\n  - provider: neo" in error_line
    assert "\n  - model: gpt-5.4" in error_line
    assert (
        "\n  - last_event: response.output_text.delta" in error_line
        or "\n  - last_event_type: response.output_text.delta" in error_line
    )
    error_index = outputs.index(error_line)
    assert outputs[error_index + 1 :] == [
        "user> go on",
        "assistant> RESUMED",
    ]
    assert stream_chunks == []

    second_body = captured_bodies[1]
    second_user_messages = [
        item["content"][0]["text"]
        for item in second_body["input"]
        if item.get("type") == "message" and item.get("role") == "user"
    ]
    assert second_user_messages[-2:] == ["Analyze current directory", "go on"]


@pytest.mark.asyncio
async def test_go_on_after_network_drop_does_not_replay_partial_reasoning_or_toolcall(
    tmp_path,
    monkeypatch,
) -> 'None':
    captured_bodies: 'typing.List[typing.Dict[str, object]]' = []
    request_count = {"value": 0}

    class _DropAfterReasoningAndToolHandler(BaseHTTPRequestHandler):
        server_version = "DropAfterReasoningAndTool/0.1"

        def log_message(self, format: 'str', *args) -> 'None':
            del format, args
            return

        def do_POST(self) -> 'None':
            if self.path != "/v1/responses":
                self.send_error(404)
                return

            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            captured_bodies.append(json.loads(body.decode("utf-8")))
            request_count["value"] += 1

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            if request_count["value"] == 1:
                partial_sse = "".join(
                    [
                        'event: response.created\n',
                        'data: {"type":"response.created","response":{"id":"resp_drop","object":"response","status":"in_progress"}}\n\n',
                        'event: response.output_item.done\n',
                        'data: {"type":"response.output_item.done","item":{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"inspecting files"}]}}\n\n',
                        'event: response.output_item.done\n',
                        'data: {"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_1","name":"exec_command","arguments":"{\\"cmd\\":\\"pwd\\"}"}}\n\n',
                    ]
                )
                self.wfile.write(partial_sse.encode("utf-8"))
                self.wfile.flush()
                self.connection.shutdown(2)
                self.connection.close()
                return

            ok_sse = "".join(
                [
                    'event: response.created\n',
                    'data: {"type":"response.created","response":{"id":"resp_ok","object":"response","status":"in_progress"}}\n\n',
                    'event: response.output_item.done\n',
                    'data: {"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"RESUMED"}]}}\n\n',
                    'event: response.completed\n',
                    'data: {"type":"response.completed","response":{"id":"resp_ok","output":[]}}\n\n',
                ]
            )
            self.wfile.write(ok_sse.encode("utf-8"))
            self.wfile.flush()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _DropAfterReasoningAndToolHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'model = "gpt-5.4"',
                'model_provider = "neo"',
                'model_reasoning_summary = "auto"',
                'model_reasoning_effort = "medium"',
                'model_verbosity = "medium"',
                'approval_policy = "never"',
                'sandbox_mode = "danger-full-access"',
                'personality = "pragmatic"',
                '',
                '[features]',
                'guardian_approval = true',
                '',
                '[model_providers.neo]',
                f'base_url = "http://127.0.0.1:{httpd.server_port}/v1"',
                'env_key = "NEO_KEY"',
                'wire_api = "responses"',
                'stream_max_retries = 0',
            ]
        )
    )
    monkeypatch.setenv("NEO_KEY", "dummy-key")

    outputs: 'typing.List[str]' = []
    saw_error = asyncio.Event()

    async def prompt_hook(_view, _prompt: 'str', value: 'str') -> 'None':
        if value == "go on":
            await saw_error.wait()

    try:
        _install_test_cli_view(
            monkeypatch,
            ["Analyze current directory", "go on", "/exit"],
            outputs,
            [],
            prompt_hook=prompt_hook,
            line_callback=(
                lambda text: saw_error.set()
                if text.startswith("Error: responses ")
                else None
            ),
        )
        args = build_parser().parse_args(["--config", str(config_path)])
        client = build_model(
            args.config,
            args.profile,
            args.timeout_seconds,
        )
        agent = build_agent(
            client,
            config_path=args.config,
            profile=args.profile,
            system_prompt=args.system_prompt,
            session_mode="tui",
        )
        runtime = build_cli_queue(agent)
        code = await run_interactive_session(
            runtime,
            False,
        )
    finally:
        httpd.shutdown()
        server_thread.join(timeout=5)
        httpd.server_close()

    assert code == 0
    assert request_count["value"] == 2

    second_body = captured_bodies[1]
    second_input = second_body["input"]

    second_user_messages = [
        item["content"][0]["text"]
        for item in second_input
        if item.get("type") == "message" and item.get("role") == "user"
    ]
    assert second_user_messages[-2:] == ["Analyze current directory", "go on"]

    assert not any(item.get("type") == "reasoning" for item in second_input)
    assert not any(item.get("type") == "function_call" for item in second_input)
    assert not any(item.get("type") == "function_call_output" for item in second_input)
    assert not any(
        item.get("type") == "message"
        and item.get("role") == "assistant"
        and any(
            content.get("text") in {"inspecting files", "pwd"}
            for content in item.get("content", [])
            if isinstance(content, dict)
        )
        for item in second_input
    )


@pytest.mark.asyncio
async def test_go_on_after_later_failure_keeps_committed_reasoning_and_tool_results(
    monkeypatch,
) -> 'None':
    class _EchoTool(BaseTool):
        name = "echo"
        description = "Echo text."
        input_schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        }

        async def run(self, context, args):
            del context
            return {"text": args["text"]}

    prompts = []

    class _FailOnSecondCallModelClient:
        def __init__(self) -> 'None':
            self.call_count = 0

        async def complete(self, prompt, event_handler):
            del event_handler
            prompts.append(prompt)
            self.call_count += 1
            if self.call_count == 1:
                return ModelResponse(
                    items=[
                        ReasoningItem(
                            payload={
                                "type": "reasoning",
                                "id": "rs_1",
                                "summary": [
                                    {
                                        "type": "summary_text",
                                        "text": "inspect repo",
                                    }
                                ],
                            }
                        ),
                        ToolCall(
                            call_id="call_1",
                            name="echo",
                            arguments={"text": "hello"},
                        ),
                    ]
                )
            if self.call_count == 2:
                raise RuntimeError("synthetic client error")
            return ModelResponse(items=[AssistantMessage(text="resumed")])

    tools = ToolRegistry()
    tools.register(_EchoTool())
    runtime = CliSubmissionQueue(Agent(_FailOnSecondCallModelClient(), tools))
    first_failure_reported = asyncio.Event()

    async def prompt_hook(_view, _prompt: 'str', value: 'str') -> 'None':
        if value == "go on":
            while len(prompts) < 2:
                await asyncio.sleep(0)
            await first_failure_reported.wait()
    _install_test_cli_view(
        monkeypatch,
        ["Analyze current directory", "go on", "/exit"],
        [],
        [],
        prompt_hook=prompt_hook,
        line_callback=(
            lambda text: first_failure_reported.set()
            if text == "Error: synthetic client error"
            else None
        ),
    )

    code = await run_interactive_session(
        runtime,
        False,
    )

    assert code == 0
    assert len(prompts) == 3

    resumed_prompt = prompts[2]
    resumed_history_items = [
        item
        for item in resumed_prompt.input
        if isinstance(item, (UserMessage, AssistantMessage, ReasoningItem, ToolCall, ToolResult))
    ]
    assert [type(item).__name__ for item in resumed_history_items] == [
        "UserMessage",
        "ReasoningItem",
        "ToolCall",
        "ToolResult",
        "UserMessage",
    ]

    reasoning_item = resumed_history_items[1]
    assert isinstance(reasoning_item, ReasoningItem)
    assert reasoning_item.payload["summary"][0]["text"] == "inspect repo"

    tool_call = resumed_history_items[2]
    tool_result = resumed_history_items[3]
    assert isinstance(tool_call, ToolCall)
    assert tool_call.name == "echo"
    assert isinstance(tool_result, ToolResult)
    assert tool_result.output == {"text": "hello"}


def _prompt_tool_outputs(prompt) -> 'typing.Dict[str, object]':
    outputs: 'typing.Dict[str, object]' = {}
    for item in prompt.input:
        if isinstance(item, ToolResult):
            outputs[item.call_id] = item.output
    return outputs


def _last_user_message_text(prompt) -> 'str':
    for item in reversed(prompt.input):
        if isinstance(item, UserMessage):
            return item.text
    raise AssertionError("prompt does not contain a user message")


@pytest.mark.asyncio
async def test_build_cli_queue_subagents_match_upstream_subset_and_context(
    tmp_path,
    monkeypatch,
) -> 'None':
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'model = "gpt-5.4"',
                'model_provider = "neo"',
                'model_reasoning_summary = "auto"',
                'model_reasoning_effort = "medium"',
                'model_verbosity = "medium"',
                'approval_policy = "never"',
                'sandbox_mode = "danger-full-access"',
                'personality = "pragmatic"',
                '',
                '[features]',
                'guardian_approval = true',
                '',
                '[model_providers.neo]',
                'base_url = "http://127.0.0.1:9/v1"',
                'env_key = "NEO_KEY"',
                'wire_api = "responses"',
            ]
        )
    )
    monkeypatch.setenv("NEO_KEY", "dummy-key")

    sub_clients: 'typing.List[_ScriptedResponsesClient]' = []

    def build_subclient() -> '_ScriptedResponsesClient':
        client = _ScriptedResponsesClient(
            response_factory=lambda prompt, _count: ModelResponse(
                items=[
                    AssistantMessage(
                        text=(
                            "initial done"
                            if _last_user_message_text(prompt) == "initial"
                            else "after resume"
                        )
                    )
                ]
            )
        )
        sub_clients.append(client)
        return client

    def main_response_factory(prompt, _count) -> 'ModelResponse':
        outputs = _prompt_tool_outputs(prompt)
        spawn_output = outputs.get("call_spawn")
        agent_id = None
        if isinstance(spawn_output, dict):
            agent_id = spawn_output.get("agent_id")

        if "call_wait_final" in outputs:
            return ModelResponse(items=[AssistantMessage(text="RESUME_AGENT_OK")])
        if "call_send_after" in outputs:
            assert isinstance(agent_id, str)
            return ModelResponse(
                items=[
                    ToolCall(
                        call_id="call_wait_final",
                        name="wait_agent",
                        arguments={"ids": [agent_id], "timeout_ms": 1000},
                    )
                ]
            )
        if "call_resume" in outputs:
            assert isinstance(agent_id, str)
            return ModelResponse(
                items=[
                    ToolCall(
                        call_id="call_send_after",
                        name="send_input",
                        arguments={"id": agent_id, "message": "after"},
                    )
                ]
            )
        if "call_close" in outputs:
            assert isinstance(agent_id, str)
            return ModelResponse(
                items=[
                    ToolCall(
                        call_id="call_resume",
                        name="resume_agent",
                        arguments={"id": agent_id},
                    )
                ]
            )
        if "call_wait_initial" in outputs:
            assert isinstance(agent_id, str)
            return ModelResponse(
                items=[
                    ToolCall(
                        call_id="call_close",
                        name="close_agent",
                        arguments={"id": agent_id},
                    )
                ]
            )
        if "call_spawn" in outputs:
            assert isinstance(agent_id, str)
            return ModelResponse(
                items=[
                    ToolCall(
                        call_id="call_wait_initial",
                        name="wait_agent",
                        arguments={"ids": [agent_id], "timeout_ms": 1000},
                    )
                ]
            )
        return ModelResponse(
            items=[
                ToolCall(
                    call_id="call_spawn",
                    name="spawn_agent",
                    arguments={"message": "initial"},
                )
            ]
        )

    main_client = _ScriptedResponsesClient(
        response_factory=main_response_factory,
        override_factory=(
            lambda _model, _reasoning, _session_id=None, _openai_subagent=None:
            build_subclient()
        ),
    )

    class _FakeResponsesModelClient:
        def __new__(cls, *args, **kwargs):
            del args, kwargs
            return main_client

    monkeypatch.setattr("pycodex.cli.ResponsesModelClient", _FakeResponsesModelClient)

    args = build_parser().parse_args(["--config", str(config_path)])
    client = build_model(
        args.config,
        args.profile,
        args.timeout_seconds,
    )
    agent = build_agent(
        client,
        config_path=args.config,
        profile=args.profile,
        system_prompt=args.system_prompt,
        session_mode="tui",
    )
    runtime = build_cli_queue(agent)
    worker = asyncio.create_task(runtime.run_forever())
    try:
        result = await runtime.submit_user_turn("drive the subagent flow")
        assert result.output_text == "RESUME_AGENT_OK"
    finally:
        await runtime.shutdown()
        await worker

    assert len(sub_clients) == 1
    assert len(sub_clients[0].prompts) == 2

    first_sub_prompt = sub_clients[0].prompts[0]
    second_sub_prompt = sub_clients[0].prompts[1]

    assert [tool.name for tool in first_sub_prompt.tools] == [
        "exec_command",
        "write_stdin",
        "update_plan",
        "apply_patch",
        "web_search",
        "view_image",
    ]
    assert [tool.name for tool in second_sub_prompt.tools] == [
        "exec_command",
        "write_stdin",
        "update_plan",
        "apply_patch",
        "web_search",
        "view_image",
    ]

    developer_messages = [
        item
        for item in first_sub_prompt.input
        if isinstance(item, ContextMessage) and item.role == "developer"
    ]
    assert len(developer_messages) == 1
    developer_texts = [
        content["text"] for content in developer_messages[0].serialize()["content"]
    ]
    assert any(text.startswith("<permissions instructions>") for text in developer_texts)
    assert not any(
        text.startswith("<collaboration_mode>") for text in developer_texts
    )

    assert list(first_sub_prompt.turn_metadata) == ["turn_id", "workspaces", "sandbox"]
    assert list(second_sub_prompt.turn_metadata) == ["turn_id", "sandbox"]

    assert [
        item.text for item in second_sub_prompt.input if isinstance(item, UserMessage)
    ] == ["initial", "after"]


@pytest.mark.asyncio
async def test_prompt_request_user_input_collects_choice_labels() -> 'None':
    outputs: 'typing.List[str]' = []
    inputs = iter(["1"])
    view = type(
        "DummyView",
        (),
        {
            "finish_stream": lambda self: None,
            "write_line": lambda self, text: outputs.append(text),
            "get_prompt": lambda self, _prompt: asyncio.sleep(
                0,
                result=next(inputs),
            ),
        },
    )()

    result = await prompt_request_user_input(
        view,
        {
            "questions": [
                {
                    "id": "choice",
                    "header": "Select",
                    "question": "Pick one",
                    "options": [
                        {
                            "label": "Use tool A (Recommended)",
                            "description": "Fast path",
                        },
                        {
                            "label": "Use tool B",
                            "description": "Slow path",
                        },
                    ],
                }
            ]
        },
    )

    assert result == {
        "answers": {
            "choice": {
                "answers": ["Use tool A (Recommended)"],
            }
        }
    }
    assert any("[request_user_input]" in line for line in outputs)


@pytest.mark.asyncio
async def test_prompt_request_permissions_supports_session_scope() -> 'None':
    outputs: 'typing.List[str]' = []
    inputs = iter(["s"])
    view = type(
        "DummyView",
        (),
        {
            "finish_stream": lambda self: None,
            "write_line": lambda self, text: outputs.append(text),
            "get_prompt": lambda self, _prompt: asyncio.sleep(
                0,
                result=next(inputs),
            ),
        },
    )()

    result = await prompt_request_permissions(
        view,
        {
            "reason": "Need write access",
            "permissions": {
                "file_system": {
                    "write": ["/tmp/demo"],
                }
            },
        },
    )

    assert result == {
        "permissions": {
            "file_system": {
                "write": ["/tmp/demo"],
            }
        },
        "scope": "session",
    }
    assert any("[request_permissions]" in line for line in outputs)


@pytest.mark.asyncio
async def test_run_cli_returns_non_zero_on_single_turn_error(
    monkeypatch: 'pytest.MonkeyPatch',
    capsys: 'pytest.CaptureFixture[str]',
    tmp_path: 'Path',
) -> 'None':
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'model = "demo-model"',
                'model_provider = "demo"',
                '[model_providers.demo]',
                'base_url = "https://example.com/v1"',
                'env_key = "DUMMY_KEY"',
            ]
        )
    )
    monkeypatch.setenv("DUMMY_KEY", "test-key")

    class _FakeRuntime:
        def __init__(self):
            self._stopped = asyncio.Event()

        def set_event_handler(self, _handler=None):
            return None

        async def run_forever(self):
            await self._stopped.wait()

        async def submit_user_turn(self, _prompt_text):
            raise RuntimeError("synthetic client error")

        async def shutdown(self):
            self._stopped.set()

    monkeypatch.setattr("pycodex.cli.should_run_interactive", lambda *_args: False)
    monkeypatch.setattr("pycodex.cli.build_cli_queue", lambda *args, **kwargs: _FakeRuntime())
    monkeypatch.setattr("sys.stdin.read", lambda: "")

    args = build_parser().parse_args(["--config", str(config_path), "hello"])
    code = await run_cli(args)

    assert code == 1
    assert capsys.readouterr().err.strip() == "Error: synthetic client error"
