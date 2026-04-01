from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pycodex import (
    AgentEvent,
    AgentLoop,
    AgentRuntime,
    AssistantMessage,
    BaseTool,
    ContextMessage,
    ModelResponse,
    ModelStreamEvent,
    ReasoningItem,
    ToolCall,
    ToolResult,
    ToolRegistry,
    UserMessage,
)
from pycodex.cli import (
    CliSessionView,
    LOCAL_RESPONSES_SERVER_API_KEY_ENV,
    _build_model_client,
    build_parser,
    build_runtime,
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
from pycodex.utils.visualize import colorize_cli_message
from tests.fake_responses_server import CaptureStore, build_handler
from tests.fakes import ScriptedModelClient


def _normalized_headers(headers: dict[str, str]) -> dict[str, str]:
    return {key.lower(): value for key, value in headers.items()}


def _configure_cli_view_output(
    view: CliSessionView,
    line_output: list[str],
    stream_chunks: list[str] | None = None,
    line_callback=None,
    raw_callback=None,
) -> CliSessionView:
    if stream_chunks is None:
        stream_chunks = []

    def write_line(text: str) -> None:
        line_output.append(text)
        if line_callback is not None:
            line_callback(text)

    def raw_write(text: str) -> None:
        stream_chunks.append(text)
        if raw_callback is not None:
            raw_callback(text)

    view._line_output = write_line
    view._raw_write = raw_write
    view._raw_flush = lambda: None
    view._color_enabled = False
    view._spinner._raw_write = view._raw_write
    view._spinner._raw_flush = view._raw_flush
    view._spinner._color_enabled = False
    return view


def _build_cli_view(
    line_output: list[str],
    stream_chunks: list[str] | None = None,
) -> CliSessionView:
    return _configure_cli_view_output(CliSessionView(), line_output, stream_chunks)


def _install_test_cli_view(
    monkeypatch: pytest.MonkeyPatch,
    inputs,
    line_output: list[str],
    stream_chunks: list[str],
    prompt_hook=None,
    line_callback=None,
    raw_callback=None,
    capture_view: dict[str, CliSessionView] | None = None,
) -> None:
    input_iter = iter(inputs)
    real_view_class = CliSessionView

    class _TestView(real_view_class):
        def __init__(self) -> None:
            super().__init__()
            _configure_cli_view_output(
                self,
                line_output,
                stream_chunks,
                line_callback=line_callback,
                raw_callback=raw_callback,
            )
            if capture_view is not None:
                capture_view["view"] = self

        async def prompt_async(self, prompt: str) -> str:
            self.set_input_active(True)
            try:
                value = next(input_iter)
                if prompt_hook is not None:
                    await prompt_hook(self, prompt, value)
                return value
            finally:
                self.set_input_active(False, resume_spinner=False)

    monkeypatch.setattr("pycodex.cli.CliSessionView", _TestView)


class _ScriptedResponsesClient(ScriptedModelClient):
    def __init__(
        self,
        responses=None,
        response_factory=None,
        override_factory=None,
    ) -> None:
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


def test_resolve_prompt_text_prefers_argv() -> None:
    assert resolve_prompt_text(["hello", "world"]) == "hello world"


def test_resolve_prompt_text_falls_back_to_stdin(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdin.read", lambda: "  hello from stdin  ")
    assert resolve_prompt_text([]) == "hello from stdin"


def test_resolve_prompt_text_rejects_missing_prompt(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    with pytest.raises(ValueError, match="prompt is required"):
        resolve_prompt_text([])


def test_build_parser_recognizes_json_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["--json", "hello"])
    assert args.json is True
    assert args.prompt == ["hello"]


def test_build_parser_recognizes_vllm_endpoint() -> None:
    parser = build_parser()
    args = parser.parse_args(["--vllm-endpoint", "http://127.0.0.1:18000", "hello"])
    assert args.vllm_endpoint == "http://127.0.0.1:18000"
    assert args.prompt == ["hello"]

def test_vllm_base_url_normalizes_empty_path_to_v1() -> None:
    from responses_server import CompatServerConfig

    config = CompatServerConfig.from_base_url("http://127.0.0.1:18000")

    assert config.outcomming_base_url == "http://127.0.0.1:18000/v1"


def test_vllm_base_url_preserves_existing_v1_path() -> None:
    from responses_server import CompatServerConfig

    config = CompatServerConfig.from_base_url("http://127.0.0.1:18000/v1/")

    assert config.outcomming_base_url == "http://127.0.0.1:18000/v1"


def test_launch_chat_completion_compat_server_normalizes_vllm_base_url(
    monkeypatch,
) -> None:
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


def test_build_runtime_overrides_provider_for_managed_vllm_mode(
    tmp_path,
    monkeypatch,
) -> None:
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

    args = build_parser().parse_args(
        [
            "--config",
            str(config_path),
        ]
    )
    client = _build_model_client(
        args.config,
        args.profile,
        args.timeout_seconds,
        managed_responses_base_url="http://127.0.0.1:18001/v1",
    )
    build_runtime(
        args.config,
        args.profile,
        args.system_prompt,
        client,
        session_mode="tui",
    )

    assert seen["originator"] == "codex-tui"
    assert seen["config"].base_url == "http://127.0.0.1:18001/v1"
    assert seen["config"].api_key_env == LOCAL_RESPONSES_SERVER_API_KEY_ENV
    assert os.environ[LOCAL_RESPONSES_SERVER_API_KEY_ENV] == "dummy"


@pytest.mark.asyncio
async def test_run_cli_launches_managed_responses_server_for_vllm_endpoint(
    monkeypatch,
) -> None:
    started = {}
    registered = {}

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
    ):
        started["endpoint"] = base_url
        started["api_key_env"] = api_key_env
        started["model_provider"] = model_provider
        return _FakeManagedServer()

    def fake_build_runtime(
        config_path,
        profile,
        system_prompt,
        client,
        session_mode="exec",
        collaboration_mode="default",
    ):
        del config_path, profile, system_prompt, collaboration_mode
        started["session_mode"] = session_mode
        started["base_url_override"] = client._config.base_url
        return _FakeRuntime()

    monkeypatch.setattr("pycodex.cli.launch_chat_completion_compat_server", fake_launch)
    monkeypatch.setattr("pycodex.cli.build_runtime", fake_build_runtime)
    monkeypatch.setattr(
        "pycodex.cli.atexit.register",
        lambda callback: registered.setdefault("callback", callback),
    )
    monkeypatch.setattr("pycodex.cli.configure_loguru", lambda: None)
    monkeypatch.setattr("sys.stdin.read", lambda: "")

    args = build_parser().parse_args(
        [
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


def test_get_tools_registers_expected_builtin_tools() -> None:
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


def test_get_tools_exec_mode_matches_codex_exec_subset() -> None:
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


def test_get_tools_exec_mode_serialization_matches_upstream_snapshot() -> None:
    registry = get_tools(exec_mode=True)
    expected = json.loads(
        (Path(__file__).resolve().parents[1] / "pycodex" / "prompts" / "exec_tools.json").read_text()
    )

    assert [spec.serialize() for spec in registry.model_visible_specs()] == expected


def test_get_subagent_tools_matches_upstream_subset() -> None:
    registry = get_subagent_tools()
    assert registry.names() == (
        "exec_command",
        "write_stdin",
        "update_plan",
        "apply_patch",
        "web_search",
        "view_image",
    )


def test_get_subagent_tools_serialization_matches_upstream_snapshot() -> None:
    registry = get_subagent_tools()
    expected = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "pycodex"
            / "prompts"
            / "subagent_tools.json"
        ).read_text()
    )

    assert [spec.serialize() for spec in registry.model_visible_specs()] == expected


def test_load_codex_dotenv_reads_env_file_and_filters_codex_prefix(tmp_path, monkeypatch) -> None:
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


def test_should_run_interactive_only_without_prompt_on_tty() -> None:
    assert should_run_interactive([], True) is True
    assert should_run_interactive(["hello"], True) is False
    assert should_run_interactive([], False) is False


@pytest.mark.asyncio
async def test_run_interactive_session_steer_mode_restarts_at_request_boundary(
    monkeypatch,
) -> None:
    first_turn_started = asyncio.Event()
    release_first_turn = asyncio.Event()

    class _DelayedModelClient:
        def __init__(self) -> None:
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
    runtime = AgentRuntime(AgentLoop(model, ToolRegistry()))
    line_output: list[str] = []
    stream_chunks: list[str] = []
    async def prompt_hook(_view, _prompt: str, value: str) -> None:
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
        "Extra commands: /history, /title, /model",
    ]
    assert "Session: hello" in line_output
    assert "[steer] inserted: again" in line_output
    assert "Error: submission interrupted" not in line_output
    assert stream_chunks == ["assistant> ", "first", "\n", "assistant> ", "second", "\n"]


@pytest.mark.asyncio
async def test_run_interactive_session_queue_command_enqueues_turn(
    monkeypatch,
) -> None:
    release_first_turn = asyncio.Event()

    class _DelayedModelClient:
        def __init__(self) -> None:
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
    runtime = AgentRuntime(AgentLoop(model, ToolRegistry()))
    line_output: list[str] = []
    stream_chunks: list[str] = []
    async def prompt_hook(_view, _prompt: str, value: str) -> None:
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
    assert stream_chunks == ["assistant> ", "first", "\n", "assistant> ", "second", "\n"]


@pytest.mark.asyncio
async def test_run_interactive_session_pauses_spinner_while_waiting_for_input(
    monkeypatch,
) -> None:
    captured_view = {}
    model = ScriptedModelClient(
        [
            ModelResponse(items=[AssistantMessage(text="first")]),
            ModelResponse(items=[AssistantMessage(text="second")]),
        ]
    )
    runtime = AgentRuntime(AgentLoop(model, ToolRegistry()))

    async def prompt_hook(_view, _prompt: str, _value: str) -> None:
        assert captured_view["view"]._spinner._paused is True

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
    assert model.call_count == 2


@pytest.mark.asyncio
async def test_run_interactive_session_disables_raw_spinner_thread(
    monkeypatch,
) -> None:
    captured_view = {}
    _install_test_cli_view(
        monkeypatch,
        ["hello", "/exit"],
        [],
        [],
        capture_view=captured_view,
    )

    model = ScriptedModelClient([ModelResponse(items=[AssistantMessage(text="done")])])
    runtime = AgentRuntime(AgentLoop(model, ToolRegistry()))

    code = await run_interactive_session(
        runtime,
        False,
    )

    assert code == 0
    assert captured_view["view"]._spinner._enabled is False


@pytest.mark.asyncio
async def test_run_interactive_session_supports_history_and_title_commands(
    monkeypatch,
) -> None:
    model = ScriptedModelClient([ModelResponse(items=[AssistantMessage(text="done")])])
    runtime = AgentRuntime(AgentLoop(model, ToolRegistry()))
    line_output: list[str] = []
    stream_chunks: list[str] = []
    _install_test_cli_view(
        monkeypatch,
        ["hello there", "/title", "/history", "/exit"],
        line_output,
        stream_chunks,
    )

    code = await run_interactive_session(
        runtime,
        False,
    )

    assert code == 0
    assert "Session: hello there" in line_output
    assert "[1] user> hello there" in line_output
    assert "    assistant> done" in line_output
    assert stream_chunks == ["assistant> ", "done", "\n"]


@pytest.mark.asyncio
async def test_run_interactive_session_supports_model_command(
    tmp_path,
    monkeypatch,
) -> None:
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
            if text == "demo-model":
                first_turn_completed.set()
            elif text == "alt-model":
                second_turn_completed.set()
            return ModelResponse(items=[AssistantMessage(text=text)])

        async def list_models(self):
            return ["demo-model", "alt-model", "third-model"]

    monkeypatch.setattr("pycodex.cli.ResponsesModelClient", _FakeResponsesModelClient)

    args = build_parser().parse_args(["--config", str(config_path)])
    client = _build_model_client(
        args.config,
        args.profile,
        args.timeout_seconds,
    )
    runtime = build_runtime(
        args.config,
        args.profile,
        args.system_prompt,
        client,
        session_mode="tui",
    )
    line_output: list[str] = []
    stream_chunks: list[str] = []
    first_turn_completed = asyncio.Event()
    second_turn_completed = asyncio.Event()

    async def prompt_hook(_view, _prompt: str, value: str) -> None:
        if value == "/model alt-model":
            await first_turn_completed.wait()
        elif value == "/model":
            await second_turn_completed.wait()

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
        "Extra commands: /history, /title, /model",
        "Session: hello",
        "assistant> demo-model",
        "Switched model to alt-model.",
        "assistant> alt-model",
        "Current model: alt-model",
        "Available models: demo-model, alt-model, third-model",
    ]
    assert stream_chunks == []


@pytest.mark.asyncio
async def test_run_interactive_session_rejects_model_switch_while_steer_work_pending(
    monkeypatch,
) -> None:
    release_turn = asyncio.Event()

    class _DelayedResponsesModelClient:
        def __init__(self) -> None:
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
    runtime = AgentRuntime(AgentLoop(model, ToolRegistry()))
    line_output: list[str] = []
    stream_chunks: list[str] = []

    async def prompt_hook(_view, _prompt: str, value: str) -> None:
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
        "Extra commands: /history, /title, /model",
    ]
    assert "Session: hello" in line_output
    assert "Cannot change model while work is running or queued in steer mode." in line_output
    assert stream_chunks == ["assistant> ", "demo-model", "\n"]


@pytest.mark.asyncio
async def test_run_interactive_session_continues_after_model_error(
    monkeypatch,
) -> None:
    class _FailOnceModelClient:
        def __init__(self) -> None:
            self.call_count = 0

        async def complete(self, prompt, event_handler):
            del prompt, event_handler
            self.call_count += 1
            if self.call_count == 1:
                raise RuntimeError("synthetic client error")
            return ModelResponse(items=[AssistantMessage(text="done")])

    runtime = AgentRuntime(AgentLoop(_FailOnceModelClient(), ToolRegistry()))
    line_output: list[str] = []
    stream_chunks: list[str] = []
    saw_error = asyncio.Event()

    async def prompt_hook(_view, _prompt: str, value: str) -> None:
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
        "Extra commands: /history, /title, /model",
        "Session: hello",
        "Error: synthetic client error",
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
) -> None:
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
    runtime = AgentRuntime(AgentLoop(model, tools))
    line_output: list[str] = []
    stream_chunks: list[str] = []
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
        "Extra commands: /history, /title, /model",
        "Session: use a tool",
        '[tool] echo: {"echo":"hello"}',
    ]
    assert stream_chunks == ["assistant> ", "done", "\n"]


def test_cli_session_view_formats_plan_and_exec_messages() -> None:
    output: list[str] = []
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
        "[plan] Working on 2/2",
        "  [x] inspect repo",
        "  [>] collect docs",
        "[exec] pwd && ls -la -> /data/codex-python",
        "[agent] spawned Curie (019d25a7...a8c6)",
        "[agent] wait: Curie=completed: listed tools",
    ]


def test_colorize_cli_message_wraps_ansi_when_enabled() -> None:
    colored = colorize_cli_message("[agent] spawned Curie", "agent", True)
    assert colored.startswith("\x1b[1m\x1b[34m")
    assert colored.endswith("\x1b[0m")


def test_cli_session_view_shows_web_search_tool_called_message() -> None:
    output: list[str] = []
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
        "[web] searched: github codex",
        "[web] opened: http://example.com",
    ]


def test_cli_session_view_turn_failed_clears_pending_prompt() -> None:
    output: list[str] = []
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
    assert view._spinner._turn_active is False
    assert output == ["Session: hello"]


def test_cli_session_view_keeps_spinner_paused_while_input_active() -> None:
    output: list[str] = []
    view = _build_cli_view(output)

    view.set_input_active(True)
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

    assert view._spinner._turn_active is True
    assert view._spinner._paused is True
    assert output == ["Session: hello", "[exec] pwd"]


def test_cli_session_view_builds_second_line_input_spinner_prompt() -> None:
    output: list[str] = []
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
    view.set_input_active(True)

    rendered = view.build_input_prompt("pycodex> ")

    assert rendered.endswith("\npycodex> ")
    assert "waiting model" in rendered


def test_cli_session_view_shows_prompt_managed_streaming_text() -> None:
    output: list[str] = []
    view = _build_cli_view(output)

    view.set_input_active(True)
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

    assert view.build_input_prompt("pycodex> ") == "assistant> hi\n"


def test_cli_session_view_renders_streaming_text_inside_prompt_when_enabled() -> None:
    output: list[str] = []
    view = _build_cli_view(output)

    view.set_input_active(True)
    view.handle_event(
        AgentEvent(
            kind="assistant_delta",
            turn_id="turn_1",
            payload={"delta": "hi"},
        )
    )

    assert view.build_input_prompt("pycodex> ") == "assistant> hi\n"


def test_cli_session_view_preserves_prompt_managed_stream_output_on_completion() -> None:
    output: list[str] = []
    view = _build_cli_view(output)

    view.set_input_active(True)
    view.handle_event(
        AgentEvent(
            kind="assistant_delta",
            turn_id="turn_1",
            payload={"delta": "hi"},
        )
    )
    view.set_input_active(False, resume_spinner=False)
    view.handle_event(
        AgentEvent(
            kind="turn_completed",
            turn_id="turn_1",
            payload={"output_text": "hi"},
        )
    )

    assert output == ["assistant> hi"]


def test_cli_session_view_handoffs_prompt_stream_to_regular_output() -> None:
    output: list[str] = []
    stream_chunks: list[str] = []
    view = _build_cli_view(output, stream_chunks)

    view.set_input_active(True)
    view.handle_event(
        AgentEvent(
            kind="assistant_delta",
            turn_id="turn_1",
            payload={"delta": "hi"},
        )
    )
    view.set_input_active(False, resume_spinner=False)
    view.handoff_prompt_stream_to_output()

    assert stream_chunks == ["assistant> ", "hi"]
    assert view._streaming is True
    assert view._streaming_in_prompt is False


def test_cli_session_view_can_leave_spinner_paused_after_input_submit() -> None:
    output: list[str] = []
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
    view.set_input_active(True)
    view.set_input_active(False, resume_spinner=False)

    assert view._spinner._turn_active is True
    assert view._spinner._paused is True


def test_cli_session_view_shows_steer_queue_and_insert_messages() -> None:
    output: list[str] = []
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
    ]


def test_cli_session_view_preserves_prompt_managed_stream_output_on_interrupt() -> None:
    output: list[str] = []
    view = _build_cli_view(output)

    view.set_input_active(True)
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
        "assistant> hi",
    ]
    assert view._history == [("say hi", "hi")]


def test_cli_session_view_keeps_history_for_reused_turn_id_across_submissions() -> None:
    output: list[str] = []
    stream_chunks: list[str] = []
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


def test_cli_session_view_assistant_stream_pauses_spinner_until_stream_finishes() -> None:
    output: list[str] = []
    stream_chunks: list[str] = []
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

    assert output == ["Session: hello"]
    assert stream_chunks == ["assistant> ", "Hel", "lo"]
    assert view._spinner._paused is True
    assert view._streaming is True

    view.handle_event(
        AgentEvent(
            kind="tool_started",
            turn_id="turn_1",
            payload={"tool_name": "exec_command"},
        )
    )

    assert stream_chunks == ["assistant> ", "Hel", "lo", "\n"]
    assert view._spinner._paused is False
    assert view._streaming is False


@pytest.mark.asyncio
async def test_run_cli_non_interactive_uses_tui_context_for_default_cli(
    tmp_path,
    monkeypatch,
) -> None:
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
) -> None:
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

    outputs: list[str] = []
    stream_chunks: list[str] = []
    first_turn_completed = asyncio.Event()

    async def prompt_hook(_view, _prompt: str, value: str) -> None:
        if value == "Reply with TWO only.":
            await first_turn_completed.wait()

    try:
        _install_test_cli_view(
            monkeypatch,
            ["Reply with ONE only.", "Reply with TWO only.", "/exit"],
            outputs,
            stream_chunks,
            prompt_hook=prompt_hook,
            line_callback=(
                lambda text: first_turn_completed.set()
                if text == "assistant> OK"
                else None
            ),
        )
        args = build_parser().parse_args(["--config", str(config_path)])
        client = _build_model_client(
            args.config,
            args.profile,
            args.timeout_seconds,
        )
        runtime = build_runtime(
            args.config,
            args.profile,
            args.system_prompt,
            client,
            session_mode="tui",
        )
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
) -> None:
    captured_bodies: list[dict[str, object]] = []
    request_count = {"value": 0}

    class _DropThenRecoverHandler(BaseHTTPRequestHandler):
        server_version = "DropThenRecover/0.1"

        def log_message(self, format: str, *args) -> None:
            del format, args
            return

        def do_POST(self) -> None:
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
            ]
        )
    )
    monkeypatch.setenv("NEO_KEY", "dummy-key")

    outputs: list[str] = []
    stream_chunks: list[str] = []
    saw_error = asyncio.Event()

    async def prompt_hook(_view, _prompt: str, value: str) -> None:
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
                if text == "Error: responses stream ended before response.completed"
                else None
            ),
        )
        args = build_parser().parse_args(["--config", str(config_path)])
        client = _build_model_client(
            args.config,
            args.profile,
            args.timeout_seconds,
        )
        runtime = build_runtime(
            args.config,
            args.profile,
            args.system_prompt,
            client,
            session_mode="tui",
        )
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
    assert outputs == [
        "pycodex interactive mode. Type /exit to quit.",
        "Extra commands: /history, /title, /model",
        "Session: Analyze current directory",
        "Error: responses stream ended before response.completed",
        "assistant> RESUMED",
    ]
    assert stream_chunks == ["\n"]

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
) -> None:
    captured_bodies: list[dict[str, object]] = []
    request_count = {"value": 0}

    class _DropAfterReasoningAndToolHandler(BaseHTTPRequestHandler):
        server_version = "DropAfterReasoningAndTool/0.1"

        def log_message(self, format: str, *args) -> None:
            del format, args
            return

        def do_POST(self) -> None:
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
            ]
        )
    )
    monkeypatch.setenv("NEO_KEY", "dummy-key")

    try:
        _install_test_cli_view(
            monkeypatch,
            ["Analyze current directory", "go on", "/exit"],
            [],
            [],
        )
        args = build_parser().parse_args(["--config", str(config_path)])
        client = _build_model_client(
            args.config,
            args.profile,
            args.timeout_seconds,
        )
        runtime = build_runtime(
            args.config,
            args.profile,
            args.system_prompt,
            client,
            session_mode="tui",
        )
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
) -> None:
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
        def __init__(self) -> None:
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
    runtime = AgentRuntime(AgentLoop(_FailOnSecondCallModelClient(), tools))
    first_failure_reported = asyncio.Event()

    async def prompt_hook(_view, _prompt: str, value: str) -> None:
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


def _prompt_tool_outputs(prompt) -> dict[str, object]:
    outputs: dict[str, object] = {}
    for item in prompt.input:
        if isinstance(item, ToolResult):
            outputs[item.call_id] = item.output
    return outputs


def _last_user_message_text(prompt) -> str:
    for item in reversed(prompt.input):
        if isinstance(item, UserMessage):
            return item.text
    raise AssertionError("prompt does not contain a user message")


@pytest.mark.asyncio
async def test_build_runtime_subagents_match_upstream_subset_and_context(
    tmp_path,
    monkeypatch,
) -> None:
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

    sub_clients: list[_ScriptedResponsesClient] = []

    def build_subclient() -> _ScriptedResponsesClient:
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

    def main_response_factory(prompt, _count) -> ModelResponse:
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
    client = _build_model_client(
        args.config,
        args.profile,
        args.timeout_seconds,
    )
    runtime = build_runtime(
        args.config,
        args.profile,
        args.system_prompt,
        client,
        session_mode="tui",
    )
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
async def test_prompt_request_user_input_collects_choice_labels() -> None:
    outputs: list[str] = []
    inputs = iter(["1"])
    view = type(
        "DummyView",
        (),
        {
            "finish_stream": lambda self: None,
            "pause_spinner": lambda self: None,
            "resume_spinner": lambda self: None,
            "write_line": lambda self, text: outputs.append(text),
            "prompt_async": lambda self, _prompt: asyncio.sleep(
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
async def test_prompt_request_permissions_supports_session_scope() -> None:
    outputs: list[str] = []
    inputs = iter(["s"])
    view = type(
        "DummyView",
        (),
        {
            "finish_stream": lambda self: None,
            "pause_spinner": lambda self: None,
            "resume_spinner": lambda self: None,
            "write_line": lambda self, text: outputs.append(text),
            "prompt_async": lambda self, _prompt: asyncio.sleep(
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
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
    monkeypatch.setattr("pycodex.cli.build_runtime", lambda *args, **kwargs: _FakeRuntime())
    monkeypatch.setattr("sys.stdin.read", lambda: "")

    args = build_parser().parse_args(["hello"])
    code = await run_cli(args)

    assert code == 1
    assert capsys.readouterr().err.strip() == "Error: synthetic client error"
