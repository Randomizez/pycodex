import asyncio
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from pycodex import (
    Agent,
    AgentRuntime,
    AssistantMessage,
    ContextConfig,
    ContextMessage,
    ModelResponse,
    ToolCall,
    ToolRegistry,
    ToolResult,
)
from pycodex.bootstrap import build_agent, build_model, get_subagent_tools, get_tools
from pycodex.cli import (
    CliSessionView,
    build_parser,
    resolve_prompt_text,
    run_cli,
    run_interactive_session,
    should_run_interactive,
)
from pycodex.events import (
    TokenCountEvent,
    ToolCompletedEvent,
    ToolStartedEvent,
    TurnCompletedEvent,
)
from pycodex.portable_server import CodexStorageServer
from tests.fakes import ScriptedModelClient


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        'model = "gpt-5.4"\nmodel_provider = "demo"\n'
        '[model_providers.demo]\nbase_url = "https://example.invalid/v1"\n',
        encoding="utf-8",
    )
    return path


class InputView(CliSessionView):
    def __init__(self, inputs, before_input=None):
        super().__init__()
        self.inputs = iter(inputs)
        self.before_input = before_input
        self.lines = []
        self._line_output = self.lines.append
        self.display.color_enabled = False

    async def poll_prompt(self, prompt=None):
        try:
            text = next(self.inputs)
        except StopIteration:
            raise EOFError()
        if self.before_input is not None:
            await self.before_input(text)
        return text


def test_cli_arguments_and_input_selection(monkeypatch):
    args = build_parser().parse_args(
        [
            "--config",
            "custom.toml",
            "--profile",
            "demo",
            "--json",
            "--use-messages",
            "--timeout-seconds",
            "17",
            "hello",
            "world",
        ]
    )
    assert (
        args.config,
        args.profile,
        args.json,
        args.use_messages,
        args.timeout_seconds,
    ) == (
        "custom.toml",
        "demo",
        True,
        True,
        17,
    )
    assert resolve_prompt_text(args.prompt) == "hello world"
    assert should_run_interactive([], True)
    assert not should_run_interactive(["hello"], True)
    assert not should_run_interactive([], False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdin.read", lambda: " piped\n")
    assert resolve_prompt_text([]) == "piped"
    monkeypatch.setattr("sys.stdin.read", lambda: "")
    with pytest.raises(ValueError, match="prompt is required"):
        resolve_prompt_text([])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--put"])


@pytest.mark.parametrize("mode", ["chat", "messages", "vllm", "managed"])
def test_model_bootstrap_selects_transport(config_path, monkeypatch, mode):
    launches = []
    probes = []
    server = SimpleNamespace(base_url="http://127.0.0.1:18083/v1", stop=lambda: None)

    def launch(*args, **kwargs):
        launches.append((args, kwargs))
        return server

    def list_models(client):
        probes.append(client._config.base_url)
        return ["first", "last"]

    monkeypatch.setattr(
        "pycodex.bootstrap.launch_chat_completion_compat_server", launch
    )
    monkeypatch.setattr(
        "pycodex.bootstrap.ResponsesModelClient.list_models_sync", list_models
    )
    options = {}
    if mode == "chat":
        with config_path.open("a", encoding="utf-8") as handle:
            handle.write("use_chat_completion = true\n")
    elif mode == "messages":
        options["use_messages"] = True
    elif mode == "vllm":
        options["vllm_endpoint"] = "http://127.0.0.1:18000"
    else:
        options["managed_responses_base_url"] = server.base_url
    client = build_model(str(config_path), timeout_seconds=17, **options)
    assert client._config.base_url == server.base_url
    assert client._config.responses_lite_override is False
    assert client._originator == "codex-tui"
    assert client._config.api_key_env == "PYCODEX_LOCAL_RESPONSES_SERVER_KEY"
    if mode == "vllm":
        assert probes == ["http://127.0.0.1:18000/v1"]
        assert client.model == "last"
        assert launches[0][1] == {"model_provider": "vllm"}
    elif mode == "managed":
        assert not launches
    else:
        assert launches[0][1]["outcomming_api"] == (
            "messages" if mode == "messages" else "chat_completions"
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        build_model(str(config_path), use_chat_completion=True, use_messages=True)


async def test_bootstrap_preserves_context_and_tool_boundaries(config_path, tmp_path):
    client = ScriptedModelClient([ModelResponse([AssistantMessage("done")])])
    agent = build_agent(
        client,
        config_path,
        system_prompt="session rules",
        extra_contextual_user_messages=["workspace context"],
        cwd=tmp_path,
        toolset=["apply_patch", "view_image"],
    )
    await agent.run_turn(["hello"])
    prompt = client.prompts[0]
    assert prompt.base_instructions == "session rules"
    assert [tool.name for tool in prompt.tools] == ["apply_patch", "view_image"]
    assert any(
        "workspace context" in str(item.content_items)
        for item in prompt.input
        if isinstance(item, ContextMessage)
    )
    assert str(tmp_path) in repr(prompt.input)
    assert get_tools(toolset=[]).names() == ()
    with pytest.raises(ValueError, match="unknown toolset"):
        get_tools(toolset=["missing"])
    assert get_subagent_tools().names() == (
        "exec_command",
        "write_stdin",
        "update_plan",
        "apply_patch",
        "web_search",
        "view_image",
    )
    defaults = get_tools(exec_mode=True)
    assert "clock" in defaults.names()
    assert "shell" not in defaults.names()
    for spec in defaults.model_visible_specs():
        assert "output_schema" not in spec.serialize()


@pytest.mark.parametrize("json_mode", [False, True])
async def test_cli_one_shot_uses_backend_and_tui_context(
    config_path, monkeypatch, capsys, json_mode
):
    client = ScriptedModelClient([ModelResponse([AssistantMessage("done")])])
    monkeypatch.setattr("pycodex.cli.build_model", lambda **kwargs: client)
    args = build_parser().parse_args(
        ["--config", str(config_path)] + (["--json"] if json_mode else []) + ["hello"],
    )
    assert await run_cli(args) == 0
    output = capsys.readouterr().out
    assert (
        json.loads(output)["output_text"] if json_mode else output.strip()
    ) == "done"
    assert client.prompts[0].input[-1].text == "hello"
    assert "<environment_context>" in repr(client.prompts[0].input)


async def test_cli_exit_code_and_command_dispatch(config_path, monkeypatch, capsys):
    client = ScriptedModelClient([])
    monkeypatch.setattr("pycodex.cli.build_model", lambda **kwargs: client)
    args = build_parser().parse_args(["--config", str(config_path), "/help"])
    assert await run_cli(args) == 0
    assert "/resume" in capsys.readouterr().out
    assert client.call_count == 0
    args.prompt = ["hello"]
    assert await run_cli(args) == 1
    assert "scripted model ran out of responses" in capsys.readouterr().err


async def test_cli_portable_roundtrip_and_utf8_locale(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").write_text(
        'model = "demo"\nmodel_provider = "demo"\n'
        '[model_providers.demo]\nbase_url = "https://example.invalid/v1"\n',
        encoding="utf-8",
    )
    (home / "AGENTS.md").write_text("规则 ≤ 5 min — 中文", encoding="utf-8")
    (home / ".env").write_text(
        'PYCODEX_TEST_DOTENV="loaded-from-dotenv"\nCODEX_TEST_DOTENV=blocked\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("PYCODEX_TEST_DOTENV", raising=False)
    monkeypatch.delenv("CODEX_TEST_DOTENV", raising=False)
    original_read = Path.read_text

    def read_as_gbk(path, encoding=None, errors=None):
        return original_read(path, encoding=encoding or "gbk", errors=errors)

    clients = []

    def client_factory(config, timeout, **kwargs):
        client = ScriptedModelClient(
            [ModelResponse([AssistantMessage("OK")])], model=config.model
        )
        clients.append(client)
        return client

    monkeypatch.setattr(Path, "read_text", read_as_gbk)
    monkeypatch.setattr("pycodex.bootstrap.ResponsesModelClient", client_factory)
    server = CodexStorageServer(tmp_path / "storage", port=0)
    server.start()
    try:
        args = build_parser().parse_args(
            ["--put", "{}@{}".format(home, server.server_address)]
        )
        assert await run_cli(args) == 0
        lines = capsys.readouterr().out.splitlines()
        assert "[put] file: AGENTS.md" in lines
        assert "[put] call test ok: config.toml" in lines
        assert lines[-1].startswith("pycodex --call ")
        call_spec = lines[-1].split()[-1]
        assert (
            await run_cli(build_parser().parse_args(["--call", call_spec, "hello"]))
            == 0
        )
        assert "规则 ≤ 5 min — 中文" in repr(clients[0].prompts[0].input)
        assert Path(os.environ["CODEX_HOME"]).is_dir()
        assert os.environ["PYCODEX_TEST_DOTENV"] == "loaded-from-dotenv"
        assert "CODEX_TEST_DOTENV" not in os.environ
    finally:
        server.stop()


@pytest.mark.parametrize("queued", [False, True])
async def test_cli_steer_and_queue_feedback(queued):
    started = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, call_count):
        if call_count == 1:
            started.set()
            await release.wait()
        return ModelResponse([AssistantMessage(str(call_count))])

    async def before_input(text):
        if text.endswith("second"):
            await asyncio.wait_for(started.wait(), 1)
        if text == "/exit":
            release.set()

    client = ScriptedModelClient(response_factory=respond)
    queue = AgentRuntime(Agent(client, ToolRegistry(), ContextConfig()))
    view = InputView(
        ["first", "/queue second" if queued else "second", "/exit"], before_input
    )
    try:
        assert (
            await asyncio.wait_for(run_interactive_session(queue, False, view=view), 2)
            == 0
        )
    finally:
        release.set()
    assert "[steer] inserted: second" in view.lines
    assert ("[steer] queued: second" in view.lines) == queued
    assert client.call_count == 2
    assert queue.agent.is_shutdown


@pytest.mark.parametrize("json_mode", [False, True])
async def test_cli_receipt_and_empty_input_without_result_tasks(monkeypatch, json_mode):
    client = ScriptedModelClient([ModelResponse([AssistantMessage("done")])])
    queue = AgentRuntime(Agent(client, ToolRegistry(), ContextConfig()))
    view = InputView(["", "/queue hello", "/exit"])
    await queue.start()
    tasks = []
    create_task = asyncio.create_task

    def record_task(coroutine):
        tasks.append(coroutine.cr_code.co_name)
        return create_task(coroutine)

    monkeypatch.setattr(asyncio, "create_task", record_task)
    assert await run_interactive_session(queue, json_mode, view=view) == 0
    results = [json.loads(line) for line in view.lines if line.startswith("{")]
    assert [result["output_text"] for result in results] == (
        ["done"] if json_mode else []
    )
    assert client.call_count == 1
    assert not tasks


async def test_cli_detaches_view_when_close_fails():
    closed = []

    class ClosingView(InputView):
        def close(self):
            closed.append(True)
            super().close()

    async def fail_cleanup():
        raise RuntimeError("cleanup failed")

    queue = AgentRuntime(
        Agent(ScriptedModelClient([]), ToolRegistry(), ContextConfig())
    )
    queue.add_close_handler(fail_cleanup)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await run_interactive_session(queue, False, view=ClosingView([]))
    assert closed == [True]
    assert not queue._frontends
    assert queue._worker.done()


@pytest.mark.parametrize("busy", [False, True])
@pytest.mark.parametrize(
    "exit_input",
    [
        "ctrl_c",
        "eof",
        pytest.param(
            "sigint", marks=pytest.mark.skipif(os.name == "nt", reason="POSIX SIGINT")
        ),
    ],
)
def test_cli_input_exit_closes_without_cancelling_work(tmp_path, busy, exit_input):
    source = textwrap.dedent("""
        import asyncio
        import os
        import signal
        import sys
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        from pycodex import (
            Agent, AgentRuntime, AssistantMessage, BaseTool, ContextConfig,
            ModelResponse, ToolRegistry,
        )
        from pycodex.cli import CliSessionView, run_interactive_session
        from tests.fakes import ScriptedModelClient

        async def run(pipe):
            busy = sys.argv[1] == "True"
            exit_input = sys.argv[2]
            started = asyncio.Event()
            release = asyncio.Event()
            cleaned = []

            class CleanupTool(BaseTool):
                name = "cleanup"
                description = "Records final cleanup."

                async def run(self, context, args):
                    return None

                def shutdown(self):
                    cleaned.append(True)

            async def respond(prompt, call_count):
                started.set()
                await release.wait()
                assert not cleaned
                return ModelResponse([AssistantMessage("finished")])

            tools = ToolRegistry()
            tools.register(CleanupTool())
            client = ScriptedModelClient(response_factory=respond)
            runtime = AgentRuntime(Agent(client, tools, ContextConfig()))
            view = CliSessionView()
            receipts = []

            async def send_exit():
                if busy:
                    for text in ("first", "second"):
                        receipts.append(await runtime.submit_input("/queue " + text))
                    await started.wait()
                while not view.prompter._prompt_session.app.is_running:
                    await asyncio.sleep(0.01)
                if exit_input == "sigint":
                    os.kill(os.getpid(), signal.SIGINT)
                else:
                    pipe.send_text("\\x03" if exit_input == "ctrl_c" else "\\x04")
                while runtime.accepts_input:
                    await asyncio.sleep(0.01)
                if busy:
                    assert not runtime.agent.is_shutdown
                    assert not cleaned
                release.set()

            sender = asyncio.create_task(send_exit())
            assert await run_interactive_session(runtime, False, view=view) == 0
            await sender
            assert runtime.agent.is_shutdown
            assert runtime._worker.done() and not runtime._worker.cancelled()
            assert not runtime._frontends
            assert cleaned == [True]
            assert client.call_count == (2 if busy else 0)
            for receipt in receipts:
                assert receipt.future.result().output_text == "finished"
            print("CLI_EXIT_OK")

        with create_pipe_input() as pipe:
            with create_app_session(input=pipe, output=DummyOutput()):
                asyncio.run(run(pipe))
    """)
    result = subprocess.run(
        [sys.executable, "-c", source, str(busy), exit_input],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=dict(
            os.environ, HOME=str(tmp_path), CODEX_HOME=str(tmp_path / "codex-home")
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CLI_EXIT_OK" in result.stdout
    assert not any(
        text in result.stderr
        for text in (
            "Traceback",
            "KeyboardInterrupt",
            "CancelledError",
            "Task exception",
        )
    )


def test_cli_executes_event_presentation_without_type_dispatch():
    view = InputView([])

    class Presentation:
        def render(self, display):
            assert display is view.display
            display.log("rendered log")
            display.set_status("rendered status")
            display.set_prompt("rendered> ")

    try:
        view.handle_event(Presentation())
        assert view.lines == ["rendered log"]
        assert view.prompter._status == "rendered status"
        assert view.prompter.prompt == "rendered> "
    finally:
        view.close()


def test_cli_context_and_tool_progress():
    view = InputView([])
    try:
        view.display.set_context_window_tokens(100000)
        view.handle_event(TokenCountEvent({"total_tokens": 56000}, "turn"))
        assert view.prompter.prompt == "pyco(50%)> "
        call = ToolCall("call", "exec_command", {"cmd": "pwd"})
        view.handle_event(ToolStartedEvent("turn", call))
        assert view.prompter._status == "calling exec_command({'cmd': 'pwd'})"
        view.handle_event(
            ToolCompletedEvent("turn", call, ToolResult("call", "exec_command", ""))
        )
        assert "[exec_command] pwd" in view.lines
        view.handle_event(TurnCompletedEvent("turn", 1, None, 1))
        assert view.prompter._status == "idle: sleeping"
        assert not any("iteration" in line for line in view.lines)
    finally:
        view.close()


@pytest.mark.parametrize("with_frame", [False, True])
def test_prompt_options_match_installed_terminal_api(monkeypatch, with_frame):
    from pycodex.cli import prompt_session_kwargs

    def old_init(
        self,
        erase_when_done=False,
        enable_system_prompt=False,
        interrupt_exception=KeyboardInterrupt,
    ):
        pass

    def new_init(
        self,
        erase_when_done=False,
        enable_system_prompt=False,
        interrupt_exception=KeyboardInterrupt,
        show_frame=False,
    ):
        pass

    session_type = type(
        "Session", (), {"__init__": new_init if with_frame else old_init}
    )
    monkeypatch.setattr("pycodex.cli.PromptSession", session_type)
    expected = {
        "erase_when_done": True,
        "enable_system_prompt": True,
        "interrupt_exception": EOFError,
    }
    if with_frame:
        expected["show_frame"] = True
    assert prompt_session_kwargs() == expected
