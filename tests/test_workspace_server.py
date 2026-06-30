import asyncio
import json
import threading
import time as time_module

from fastapi.testclient import TestClient
import pytest

from pycodex import (
    Agent,
    AgentEvent,
    AssistantMessage,
    BaseTool,
    CliSubmissionQueue,
    ContextManager,
    ContextMessage,
    ModelStreamEvent,
    ModelResponse,
    ToolCall,
    ToolRegistry,
)
from pycodex.utils.session_persist import SessionRolloutRecorder
from workspace_server import (
    WebSessionView,
    ThreadedWorkspaceInteractiveSession,
    WorkspaceInteractiveSession,
    WorkspaceSessionManager,
    build_parser,
    create_app,
    parse_target,
)
from workspace_server.app import (
    SPINNER_STATUS_PREVIEW_LIMIT,
    WorkspaceStateStore,
    _board_context_text,
    _default_board_path,
    _format_board_path_for_prompt,
)
from tests.fakes import ScriptedModelClient
import time


class _FailingWorkspaceTool(BaseTool):
    name = "fail"
    description = "Always fails."
    input_schema = {"type": "object"}

    async def run(self, context, args):
        del context, args
        raise RuntimeError("synthetic tool failure")


def test_parse_target_accepts_plus_board_suffix(tmp_path) -> None:
    board = tmp_path / "board.html"
    board.write_text("<html>board</html>", encoding="utf-8")

    host, port, board_path = parse_target("0.0.0.0:6008+{0}".format(board))

    assert host == "0.0.0.0"
    assert port == 6008
    assert board_path == board.resolve()


def test_parse_target_board_argument_overrides_suffix(tmp_path) -> None:
    suffix_board = tmp_path / "suffix.html"
    board = tmp_path / "board.html"
    suffix_board.write_text("<html>suffix</html>", encoding="utf-8")
    board.write_text("<html>board</html>", encoding="utf-8")

    host, port, board_path = parse_target(
        "127.0.0.1:6007+{0}".format(suffix_board),
        str(board),
    )

    assert host == "127.0.0.1"
    assert port == 6007
    assert board_path == board.resolve()


def test_workspace_parser_uses_console_script_entry() -> None:
    help_text = build_parser().format_help()

    assert "pycodex-ws" in help_text
    assert "--listen" in help_text
    assert "--board" in help_text
    assert "pycodex serve" not in help_text


def test_workspace_parser_accepts_listen_and_board_flags(tmp_path) -> None:
    board = tmp_path / "board.html"
    board.write_text("<html>board</html>", encoding="utf-8")

    args = build_parser().parse_args(
        ["--listen", "0.0.0.0:6007", "--board", str(board)]
    )
    host, port, board_path = parse_target(args.listen, args.board)

    assert host == "0.0.0.0"
    assert port == 6007
    assert board_path == board.resolve()


def test_workspace_default_board_path_is_writable_tmp_html() -> None:
    board_path = _default_board_path()

    assert board_path.parent.is_dir()
    assert board_path.name.startswith("pcws-")
    assert board_path.suffix == ".html"
    assert len(board_path.stem) == len("pcws-") + 8


def test_workspace_formats_board_prompt_path_relative_to_cwd(tmp_path, monkeypatch) -> None:
    board = tmp_path / "board.html"
    board.write_text("<html>board</html>", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert _format_board_path_for_prompt(board) == "./board.html"


@pytest.mark.asyncio
async def test_workspace_start_does_not_submit_board_context(tmp_path) -> None:
    board = tmp_path / "board.html"
    board.write_text("<html>board</html>", encoding="utf-8")
    model = ScriptedModelClient([ModelResponse([AssistantMessage("noted")])])
    context_text = _board_context_text(board)
    assert "Current workspace board file:" in context_text
    assert "shown to the user in real time" in context_text
    session = WorkspaceInteractiveSession(
        CliSubmissionQueue(
            Agent(
                model,
                ToolRegistry(),
                context_manager=ContextManager(
                    extra_contextual_user_messages=[context_text],
                    include_permissions_instructions=False,
                    include_skills_instructions=False,
                ),
            )
        ),
    )

    await session.start()
    try:
        await asyncio.sleep(0.05)
        assert session.snapshot()["turns"] == []
        assert model.call_count == 0
        await session.submit("hello")
        snapshot = await _wait_for_async_snapshot(
            session,
            lambda item: item["turns"]
            and item["turns"][0]["prompt"] == "hello"
            and item["turns"][0]["response"] == "noted",
        )
    finally:
        await session.close()

    assert snapshot["turns"][0]["sender"] == "web"
    assert model.prompts[0].input[-1].text == "hello"
    contextual_message = model.prompts[0].input[-2]
    assert isinstance(contextual_message, ContextMessage)
    assert contextual_message.content_items is not None
    assert context_text in [item["text"] for item in contextual_message.content_items]


def test_workspace_app_serves_board(tmp_path) -> None:
    board = tmp_path / "board.html"
    board.write_text("<!doctype html><title>Board</title>", encoding="utf-8")
    link = _DormantLink()

    app = create_app(lambda: link, board)

    with TestClient(app) as client:
        response = client.get("/board")

    assert response.status_code == 200
    assert "Board" in response.text


def test_workspace_app_empty_board_mentions_standalone_entry() -> None:
    link = _DormantLink()
    app = create_app(lambda: link, None)

    with TestClient(app) as client:
        response = client.get("/board")

    assert response.status_code == 200
    assert "pycodex-ws --listen" in response.text
    assert "pycodex serve" not in response.text


def test_workspace_app_missing_board_returns_pending_page(tmp_path) -> None:
    board = tmp_path / "agent-created-board.html"
    link = _DormantLink()
    app = create_app(lambda: link, board)

    with TestClient(app) as client:
        response = client.get("/board")
        status = client.get("/api/board")

    assert response.status_code == 200
    assert "Board pending" in response.text
    assert str(board) in response.text
    assert status.json() == {"exists": False}


def test_workspace_app_session_snapshot_uses_turns_not_raw_messages(tmp_path) -> None:
    board = tmp_path / "board.html"
    board.write_text("<!doctype html><title>Board</title>", encoding="utf-8")
    link = _DormantLink()

    app = create_app(lambda: link, board)

    with TestClient(app) as client:
        response = client.get("/api/session")

    assert response.status_code == 200
    snapshot = response.json()["snapshot"]
    assert "messages" not in snapshot
    assert snapshot["turns"][0]["response"] == "pong"
    assert snapshot["turns"][0]["thinking"] == ""

    sessions = response.json()["sessions"]
    assert sessions[0]["last_assistant"] == "pong"


def test_workspace_app_session_endpoint_returns_spinner_and_turns(tmp_path) -> None:
    board = tmp_path / "board.html"
    board.write_text("<!doctype html><title>Board</title>", encoding="utf-8")
    link = _ScrollAwareLink()

    app = create_app(lambda: link, board)

    with TestClient(app) as client:
        response = client.get("/api/session")

    assert response.status_code == 200
    snapshot = response.json()["snapshot"]
    assert snapshot["spinner"] == "thinking"
    assert snapshot["turns"][0]["prompt"] == "prompt"
    assert snapshot["turns"][0]["response"] == "response"


def test_workspace_view_tracks_context_remaining_percent() -> None:
    view = WebSessionView()
    assert view.snapshot()["context_remaining_percent"] is None

    view.set_context_window_tokens(100_000)
    assert view.snapshot()["context_remaining_percent"] == 100
    assert view.summary()["context_remaining_percent"] == 100

    view.handle_event(
        AgentEvent(
            kind="token_count",
            turn_id="turn_1",
            payload={"usage": {"total_tokens": 20_800}},
        )
    )

    assert view.snapshot()["context_remaining_percent"] == 90
    assert view.summary()["context_remaining_percent"] == 90


def test_workspace_app_shell_uses_spinner_without_send_button(tmp_path) -> None:
    board = tmp_path / "board.html"
    board.write_text("<!doctype html><title>Board</title>", encoding="utf-8")
    link = _DormantLink()

    app = create_app(lambda: link, board)

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert 'class="spinner-wrap"' in response.text
    assert 'id="spinner" class="spinner" type="button" role="checkbox"' in response.text
    assert 'id="toast" class="toast" role="status"' in response.text
    assert response.text.index('class="spinner-wrap"') < response.text.index('id="contextMeter"')
    assert response.text.index('id="contextMeter"') < response.text.index('id="spinner"')
    assert 'id="contextMeter" class="context-meter hidden" aria-hidden="true"' in response.text
    assert 'id="contextMeterValue" class="context-meter-value"' in response.text
    assert 'contextMeter.style.setProperty("--context-remaining", `${value}%`)' in response.text
    assert "contextMeter.setAttribute(\"aria-valuenow\"" not in response.text
    assert "contextMeterValue.textContent = String(value)" in response.text
    assert "sessionId === activeSessionId && spinnerText" in response.text
    assert "spinner.addEventListener(\"click\", toggleSpinnerNotification)" in response.text
    assert "new Notification(`session: ${title}`" in response.text
    assert "body: lastAssistant" in response.text
    assert "notifySessionDone(sessionId, session)" in response.text
    assert "notify-on-done enabled" in response.text
    assert "notify-on-done canceled" in response.text
    assert "browser notifications blocked" in response.text
    assert "requestNotificationPermission" in response.text
    assert 'document.addEventListener("visibilitychange"' in response.text
    assert 'document.visibilityState !== "visible"' in response.text
    assert ">Send<" not in response.text
    assert "Enter sends. Shift+Enter adds a newline." not in response.text
    assert "compositionstart" in response.text
    assert "event.isComposing" in response.text
    assert "__BOARD_LABEL__" not in response.text
    assert "boardbar" in response.text
    assert "pycodex</div>" not in response.text
    assert 'id="splitter"' in response.text
    assert "centerSplit" in response.text
    assert "mobile-switch" not in response.text


def test_workspace_spinner_tool_call_preview_is_longer_than_title() -> None:
    view = WebSessionView()
    call = ToolCall(
        call_id="call_1",
        name="exec_command",
        arguments={"cmd": "x" * 140},
    )

    view.handle_event(
        ModelStreamEvent(
            kind="tool_started",
            payload={"tool_name": "exec_command", "call": call},
        )
    )

    spinner = str(view.snapshot()["spinner"])
    assert len(spinner) > 72
    assert len(spinner) <= SPINNER_STATUS_PREVIEW_LIMIT
    assert spinner.startswith("calling exec_command(")


def test_workspace_app_message_uses_shared_interactive_commands(tmp_path) -> None:
    board = tmp_path / "board.html"
    board.write_text("<!doctype html><title>Board</title>", encoding="utf-8")
    model = ScriptedModelClient(
        responses=[ModelResponse([AssistantMessage("should not be used")])]
    )

    def build_session():
        runtime = CliSubmissionQueue(Agent(model, ToolRegistry()))
        return WorkspaceInteractiveSession(runtime)

    app = create_app(build_session, board)

    with TestClient(app) as client:
        response = client.post("/api/session/message", json={"prompt": "/history"})
        snapshot = _wait_for_snapshot(
            client,
            lambda item: item["turns"] and item["turns"][-1]["response"] == "No history yet.",
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["type"] == "submitted"
    assert snapshot["turns"][-1]["kind"] == "control"
    assert snapshot["turns"][-1]["prompt"] == ""
    assert model.call_count == 0


def test_workspace_app_resume_list_control_output_is_not_duplicated(tmp_path) -> None:
    board = tmp_path / "board.html"
    board.write_text("<!doctype html><title>Board</title>", encoding="utf-8")
    codex_home = tmp_path / "codex-home"
    config_path = codex_home / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("", encoding="utf-8")
    thread_id = "22222222-2222-4222-8222-222222222222"
    rollout_path = (
        codex_home
        / "sessions"
        / "2026"
        / "04"
        / "07"
        / "rollout-2026-04-07T00-00-00-{0}.jsonl".format(thread_id)
    )
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    rollout_items = [
        {
            "type": "session_meta",
            "payload": {"id": thread_id},
        },
        {
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "old prompt"},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "old answer"}],
            },
        },
    ]
    rollout_path.write_text(
        "\n".join(json.dumps(item) for item in rollout_items) + "\n",
        encoding="utf-8",
    )

    def build_session():
        runtime = CliSubmissionQueue(Agent(ScriptedModelClient([]), ToolRegistry()))
        return WorkspaceInteractiveSession(runtime, config_path=str(config_path))

    app = create_app(build_session, board)

    with TestClient(app) as client:
        response = client.post("/api/session/message", json={"prompt": "/resume"})
        snapshot = _wait_for_snapshot(
            client,
            lambda item: item["turns"]
            and any(
                "Available sessions:" in turn["response"]
                for turn in item["turns"]
            ),
        )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    controls = [turn for turn in snapshot["turns"] if turn["kind"] == "control"]
    assert [turn["prompt"] for turn in controls] == ["", ""]
    rendered_text = "\n".join(turn["response"] for turn in controls)
    assert rendered_text.count("Available sessions:") == 1
    assert rendered_text.count("old prompt") == 1


def test_workspace_app_title_command_updates_tab_name(tmp_path) -> None:
    board = tmp_path / "board.html"
    board.write_text("<!doctype html><title>Board</title>", encoding="utf-8")
    model = ScriptedModelClient(
        responses=[ModelResponse([AssistantMessage("should not be used")])]
    )

    def build_session():
        runtime = CliSubmissionQueue(Agent(model, ToolRegistry()))
        return WorkspaceInteractiveSession(runtime)

    app = create_app(build_session, board)

    with TestClient(app) as client:
        response = client.post("/api/session/message", json={"prompt": "/title tab1"})
        payload = response.json()
        snapshot = _wait_for_snapshot(
            client,
            lambda item: item["title"] == "tab1",
        )
        sessions = client.get("/api/sessions").json()["sessions"]

    assert response.status_code == 200
    assert payload["ok"] is True
    assert snapshot["title"] == "tab1"
    assert snapshot["turns"] == []
    assert sessions[0]["title"] == "tab1"
    assert model.call_count == 0


def test_workspace_app_title_command_persists_tab_state_to_sidecar(tmp_path) -> None:
    board = tmp_path / "board.html"
    board_html = "<!doctype html><body><h1>Board</h1></body>"
    board.write_text(board_html, encoding="utf-8")
    codex_home = tmp_path / "codex-home"
    session_id = "33333333-3333-4333-8333-333333333333"
    recorder = SessionRolloutRecorder.create(
        codex_home,
        session_id,
        tmp_path,
        "test",
        None,
        "base",
    )

    def build_session():
        runtime = CliSubmissionQueue(
            Agent(
                ScriptedModelClient([ModelResponse([AssistantMessage("unused")])]),
                ToolRegistry(),
                rollout_recorder=recorder,
            )
        )
        return WorkspaceInteractiveSession(runtime)

    app = create_app(WorkspaceSessionManager(build_session, board), board)

    with TestClient(app) as client:
        response = client.post("/api/session/message", json={"prompt": "/title tab1"})
        snapshot = _wait_for_snapshot(client, lambda item: item["title"] == "tab1")

    assert response.status_code == 200
    assert snapshot["title"] == "tab1"
    assert board.read_text(encoding="utf-8") == board_html
    assert WorkspaceStateStore(board).path == tmp_path / "board.pycodex-ws.json"
    assert WorkspaceStateStore(board).load_tabs() == [
        {"title": "tab1", "rollout_path": str(recorder.rollout_path)}
    ]


def test_workspace_app_auto_title_does_not_persist_tab_state(tmp_path) -> None:
    board = tmp_path / "board.html"
    board.write_text("<!doctype html><body><h1>Board</h1></body>", encoding="utf-8")
    recorder = SessionRolloutRecorder.create(
        tmp_path / "codex-home",
        "55555555-5555-4555-8555-555555555555",
        tmp_path,
        "test",
        None,
        "base",
    )

    def build_session():
        runtime = CliSubmissionQueue(
            Agent(
                ScriptedModelClient([ModelResponse([AssistantMessage("auto title done")])]),
                ToolRegistry(),
                rollout_recorder=recorder,
            )
        )
        return WorkspaceInteractiveSession(runtime)

    app = create_app(WorkspaceSessionManager(build_session, board), board)
    state_store = WorkspaceStateStore(board)

    with TestClient(app) as client:
        response = client.post(
            "/api/session/message",
            json={"prompt": "please create an automatic title"},
        )
        snapshot = _wait_for_snapshot(
            client,
            lambda item: item["title"]
            and item["turns"]
            and item["turns"][-1]["response"] == "auto title done",
        )

    assert response.status_code == 200
    assert snapshot["title"]
    assert state_store.load_tabs() == []


def test_workspace_app_restores_tabs_from_board_state(tmp_path) -> None:
    board = tmp_path / "board.html"
    codex_home = tmp_path / "codex-home"
    thread_id = "44444444-4444-4444-8444-444444444444"
    rollout_path = (
        codex_home
        / "sessions"
        / "2026"
        / "04"
        / "07"
        / "rollout-2026-04-07T00-00-00-{0}.jsonl".format(thread_id)
    )
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    rollout_path.write_text(
        "\n".join(
            json.dumps(item)
            for item in [
                {"type": "session_meta", "payload": {"id": thread_id}},
                {
                    "type": "event_msg",
                    "payload": {"type": "user_message", "message": "old prompt"},
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "old answer"}],
                    },
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    board.write_text("<!doctype html><body><h1>Board</h1></body>", encoding="utf-8")
    WorkspaceStateStore(board).save_tabs(
        [{"title": "restored tab", "rollout_path": str(rollout_path)}]
    )

    def build_session():
        runtime = CliSubmissionQueue(
            Agent(ScriptedModelClient([]), ToolRegistry())
        )
        return WorkspaceInteractiveSession(runtime)

    app = create_app(WorkspaceSessionManager(build_session, board), board)

    with TestClient(app) as client:
        snapshot = client.get("/api/session").json()["snapshot"]
        sessions = client.get("/api/sessions").json()["sessions"]

    assert snapshot["title"] == "restored tab"
    assert snapshot["turns"][0]["prompt"] == "old prompt"
    assert snapshot["turns"][0]["response"] == "old answer"
    assert sessions[0]["title"] == "restored tab"


def test_workspace_app_restore_tabs_starts_session_then_restores(tmp_path) -> None:
    board = tmp_path / "board.html"
    board.write_text("<!doctype html><body><h1>Board</h1></body>", encoding="utf-8")
    rollout_path = tmp_path / "rollout.jsonl"
    rollout_path.write_text("", encoding="utf-8")
    WorkspaceStateStore(board).save_tabs(
        [{"title": "restored tab", "rollout_path": str(rollout_path)}]
    )
    started = []

    class _RestoreLink:
        async def start(self):
            started.append(True)
            return self

        async def close(self):
            return None

        async def restore_from_rollout(self, _rollout_path, title=""):
            del title
            return None

        def summary(self):
            return {
                "title": "restored tab",
                "running": False,
                "spinner": "",
                "turn_count": 0,
            }

        def snapshot(self):
            return {
                "running": False,
                "status": "",
                "status_kind": "idle",
                "spinner": "",
                "model": "test",
                "title": "restored tab",
                "turns": [],
            }

        def rollout_path(self):
            return str(rollout_path)

        def subscribe(self):
            queue = asyncio.Queue()
            queue.put_nowait({"type": "hello", "events": [], "snapshot": self.snapshot()})
            return queue

        def unsubscribe(self, _queue):
            return None

        async def submit(self, prompt, sender="web"):
            del prompt, sender
            return {"ok": True, "type": "submitted", "snapshot": self.snapshot()}

    app = create_app(WorkspaceSessionManager(_RestoreLink, board), board)

    with TestClient(app) as client:
        assert client.get("/api/sessions").status_code == 200

    assert started == [True]


def test_workspace_app_close_persists_remaining_tab_state_to_sidecar(tmp_path) -> None:
    board = tmp_path / "board.html"
    board.write_text("<!doctype html><body><h1>Board</h1></body>", encoding="utf-8")
    created = []

    class _PersistentLink:
        def __init__(self, label: str, rollout_path) -> None:
            self.label = label
            self._rollout_path = rollout_path
            self.closed = False

        async def start(self):
            return self

        async def close(self):
            self.closed = True
            return None

        def snapshot(self):
            return {
                "running": False,
                "status": "",
                "status_kind": "idle",
                "spinner": "",
                "model": "test",
                "title": self.label,
                "turns": [],
            }

        def rollout_path(self) -> str:
            return str(self._rollout_path)

        async def restore_from_rollout(self, rollout_path: str, title: str = "") -> None:
            self._rollout_path = rollout_path
            if title:
                self.label = str(title or "")

        def subscribe(self):
            queue = asyncio.Queue()
            queue.put_nowait({"type": "hello", "events": [], "snapshot": self.snapshot()})
            return queue

        def unsubscribe(self, _queue):
            return None

    def build_link():
        index = len(created) + 1
        rollout_path = tmp_path / "rollout-{0}.jsonl".format(index)
        rollout_path.write_text("", encoding="utf-8")
        link = _PersistentLink("tab-{0}".format(index), rollout_path)
        created.append(link)
        return link

    app = create_app(build_link, board)
    state_store = WorkspaceStateStore(board)

    with TestClient(app) as client:
        initial = client.get("/api/sessions").json()["sessions"]
        second_id = client.post("/api/sessions").json()["session_id"]

        assert state_store.load_tabs() == []

        response = client.delete("/api/sessions/{0}".format(second_id))

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert created[1].closed is True
    assert state_store.load_tabs() == [
        {"title": "tab-1", "rollout_path": str(created[0]._rollout_path)}
    ]
    assert initial[0]["title"] == "tab-1"


def test_workspace_app_resume_renders_only_restored_turns(tmp_path) -> None:
    board = tmp_path / "board.html"
    board.write_text("<!doctype html><title>Board</title>", encoding="utf-8")
    codex_home = tmp_path / "codex-home"
    config_path = codex_home / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("", encoding="utf-8")
    thread_id = "11111111-1111-4111-8111-111111111111"
    rollout_path = (
        codex_home
        / "sessions"
        / "2026"
        / "04"
        / "07"
        / "rollout-2026-04-07T00-00-00-{0}.jsonl".format(thread_id)
    )
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    rollout_items = [
        {
            "type": "session_meta",
            "payload": {"id": thread_id},
        },
        {
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "old prompt"},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "old answer"}],
            },
        },
    ]
    rollout_path.write_text(
        "\n".join(json.dumps(item) for item in rollout_items) + "\n",
        encoding="utf-8",
    )

    def build_session():
        runtime = CliSubmissionQueue(Agent(ScriptedModelClient([]), ToolRegistry()))
        return WorkspaceInteractiveSession(runtime, config_path=str(config_path))

    app = create_app(build_session, board)

    with TestClient(app) as client:
        response = client.post("/api/session/message", json={"prompt": "/resume 1"})
        snapshot = _wait_for_snapshot(
            client,
            lambda item: item["turns"] and item["turns"][0]["prompt"] == "old prompt",
        )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert snapshot["title"] == "old prompt"
    assert snapshot["turns"] == [
        {
            "submission_id": snapshot["turns"][0]["submission_id"],
            "turn_id": snapshot["turns"][0]["turn_id"],
            "prompt": "old prompt",
            "response": "old answer",
            "thinking": "",
            "status": "completed",
            "error": "",
            "queue": "history",
            "sender": "resume",
            "kind": "assistant",
        }
    ]


def test_workspace_view_resume_clears_stale_event_backlog() -> None:
    view = WebSessionView()
    view.write_line("[steer] inserted: stale")

    view.load_session_history("resumed", [("old prompt", "old answer")])
    subscriber = view.subscribe()
    hello = subscriber.get_nowait()

    assert hello["type"] == "hello"
    assert len(hello["events"]) == 1
    assert "[steer] inserted: stale" not in json.dumps(hello["events"])
    assert hello["snapshot"]["turns"][0]["prompt"] == "old prompt"


def test_workspace_app_manages_multiple_sessions(tmp_path) -> None:
    board = tmp_path / "board.html"
    board.write_text("<!doctype html><title>Board</title>", encoding="utf-8")
    created = []

    def build_link():
        link = _DormantLink(label="session-{0}".format(len(created) + 1))
        created.append(link)
        return link

    app = create_app(WorkspaceSessionManager(build_link), board)

    with TestClient(app) as client:
        initial = client.get("/api/sessions").json()["sessions"]
        assert len(initial) == 1
        first_id = initial[0]["id"]

        created_response = client.post("/api/sessions").json()
        assert created_response["ok"] is True
        second_id = created_response["session_id"]
        assert second_id != first_id

        first_message = client.post(
            "/api/session/message",
            json={"session_id": first_id, "prompt": "first tab"},
        ).json()
        second_message = client.post(
            "/api/session/message",
            json={"session_id": second_id, "prompt": "second tab"},
        ).json()

        assert first_message["snapshot"]["turns"][-1]["prompt"] == "first tab"
        assert second_message["snapshot"]["turns"][-1]["prompt"] == "second tab"

        close = client.delete("/api/sessions/{0}".format(second_id)).json()
        assert close["ok"] is True
        assert [session["id"] for session in close["sessions"]] == [first_id]
        assert created[1].closed is True


def test_workspace_app_accepts_session_factory(tmp_path) -> None:
    board = tmp_path / "board.html"
    board.write_text("<!doctype html><title>Board</title>", encoding="utf-8")
    created = []

    def build_link():
        link = _DormantLink(label="factory-{0}".format(len(created) + 1))
        created.append(link)
        return link

    app = create_app(build_link, board)

    with TestClient(app) as client:
        first_id = client.get("/api/sessions").json()["sessions"][0]["id"]
        second = client.post("/api/sessions").json()["session_id"]

        first_message = client.post(
            "/api/session/message",
            json={"session_id": first_id, "prompt": "first"},
        ).json()
        second_message = client.post(
            "/api/session/message",
            json={"session_id": second, "prompt": "second"},
        ).json()

    assert [link.prompts for link in created] == [["first"], ["second"]]
    assert first_message["snapshot"]["turns"][-1]["prompt"] == "first"
    assert second_message["snapshot"]["turns"][-1]["prompt"] == "second"


def test_workspace_app_session_list_uses_lightweight_summary(tmp_path) -> None:
    board = tmp_path / "board.html"
    board.write_text("<!doctype html><title>Board</title>", encoding="utf-8")

    class _SummaryOnlyLink:
        async def start(self):
            return self

        async def close(self):
            return None

        def subscribe(self):
            queue = asyncio.Queue()
            queue.put_nowait({"type": "hello", "events": [], "snapshot": {}})
            return queue

        def unsubscribe(self, _queue):
            return None

        async def restore_from_rollout(self, _rollout_path, title=""):
            del title
            return None

        def rollout_path(self):
            return ""

        def summary(self):
            return {
                "title": "summary title",
                "running": True,
                "spinner": "thinking",
                "turn_count": 123,
                "last_assistant": "latest answer",
                "context_remaining_percent": 48,
            }

        def snapshot(self):
            raise AssertionError("session list should not build a full snapshot")

    app = create_app(lambda: _SummaryOnlyLink(), board)

    with TestClient(app) as client:
        response = client.get("/api/sessions")

    assert response.status_code == 200
    assert response.json()["sessions"][0]["title"] == "summary title"
    assert response.json()["sessions"][0]["running"] is True
    assert response.json()["sessions"][0]["spinner"] == "thinking"
    assert response.json()["sessions"][0]["turn_count"] == 123
    assert response.json()["sessions"][0]["last_assistant"] == "latest answer"
    assert response.json()["sessions"][0]["context_remaining_percent"] == 48


def test_workspace_app_websocket_ping(tmp_path) -> None:
    board = tmp_path / "board.html"
    board.write_text("<!doctype html><title>Board</title>", encoding="utf-8")
    link = _DormantLink()

    app = create_app(lambda: link, board)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/session") as websocket:
            hello = websocket.receive_json()
            websocket.send_json({"type": "ping"})
            pong = websocket.receive_json()

    assert hello["type"] == "hello"
    assert pong["type"] == "pong"


def _wait_for_snapshot(client, predicate):
    deadline = time.time() + 5.0
    last_snapshot = None
    while time.time() < deadline:
        response = client.get("/api/session")
        assert response.status_code == 200
        last_snapshot = response.json()["snapshot"]
        if predicate(last_snapshot):
            return last_snapshot
        time.sleep(0.05)
    raise AssertionError("snapshot predicate did not pass: {0}".format(last_snapshot))


async def _wait_for_async_snapshot(session, predicate):
    deadline = time.time() + 5.0
    last_snapshot = None
    while time.time() < deadline:
        last_snapshot = session.snapshot()
        if predicate(last_snapshot):
            return last_snapshot
        await asyncio.sleep(0.05)
    raise AssertionError("snapshot predicate did not pass: {0}".format(last_snapshot))


@pytest.mark.asyncio
async def test_workspace_session_submits_and_streams_events() -> None:
    model = ScriptedModelClient(
        responses=[ModelResponse([AssistantMessage("hello from model")])]
    )
    runtime = CliSubmissionQueue(Agent(model, ToolRegistry()))
    link = WorkspaceInteractiveSession(runtime)
    await link.start()
    subscriber = link.subscribe()
    try:
        hello = await asyncio.wait_for(subscriber.get(), timeout=5.0)
        assert hello["type"] == "hello"

        result = await link.submit("say hello")
        assert result["ok"] is True

        snapshot = await _wait_for_async_snapshot(
            link,
            lambda item: item["turns"] and item["turns"][-1]["response"] == "hello from model",
        )
        assert "messages" not in snapshot
        assert snapshot["status"] == ""
        assert snapshot["spinner"] == ""
        assert snapshot["turns"][-1]["prompt"] == "say hello"
        assert snapshot["turns"][-1]["response"] == "hello from model"
        assert snapshot["turns"][-1]["thinking"] == ""
    finally:
        link.unsubscribe(subscriber)
        await link.close()


@pytest.mark.asyncio
async def test_workspace_session_tool_error_is_not_rendered_as_turn_error() -> None:
    model = ScriptedModelClient(
        responses=[
            ModelResponse([ToolCall(call_id="call_1", name="fail", arguments={})]),
            ModelResponse([AssistantMessage("handled after tool failure")]),
        ]
    )
    tools = ToolRegistry()
    tools.register(_FailingWorkspaceTool())
    runtime = CliSubmissionQueue(Agent(model, tools))
    link = WorkspaceInteractiveSession(runtime)
    await link.start()
    try:
        result = await link.submit("use failing tool")
        assert result["ok"] is True

        snapshot = await _wait_for_async_snapshot(
            link,
            lambda item: item["turns"]
            and item["turns"][-1]["response"] == "handled after tool failure",
        )
        turn = snapshot["turns"][-1]
        assert turn["prompt"] == "use failing tool"
        assert turn["response"] == "handled after tool failure"
        assert turn["status"] == "completed"
        assert turn["error"] == ""
    finally:
        await link.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/title renamed", "/title", "/history"])
async def test_workspace_control_command_does_not_finish_active_thinking(
    command: str,
) -> None:
    request_started = asyncio.Event()
    release_request = asyncio.Event()

    class _StreamingModelClient:
        model = "streaming-test"

        async def complete(self, _prompt, event_handler):
            event_handler(ModelStreamEvent(kind="assistant_delta", payload={"delta": "thinking"}))
            request_started.set()
            await release_request.wait()
            return ModelResponse([AssistantMessage("final")])

    runtime = CliSubmissionQueue(Agent(_StreamingModelClient(), ToolRegistry()))
    link = WorkspaceInteractiveSession(runtime)
    await link.start()
    try:
        result = await link.submit("work")
        assert result["ok"] is True
        await request_started.wait()

        command_result = await link.submit(command)
        assert command_result["ok"] is True

        snapshot = await _wait_for_async_snapshot(
            link,
            lambda item: (
                command != "/title renamed" or item["title"] == "renamed"
            )
            and any(
                turn["kind"] == "assistant"
                and turn["prompt"] == "work"
                and turn["thinking"] == "thinking"
                for turn in item["turns"]
            ),
        )
        turn = next(
            turn
            for turn in snapshot["turns"]
            if turn["kind"] == "assistant" and turn["prompt"] == "work"
        )
        assert turn["response"] == ""
        assert turn["thinking"] == "thinking"
    finally:
        release_request.set()
        await link.close()


@pytest.mark.asyncio
async def test_workspace_session_steer_interruption_is_not_error() -> None:
    first_request_started = asyncio.Event()
    release_first_request = asyncio.Event()

    class _DelayedModelClient:
        model = "delayed-test"

        async def complete(self, _prompt, event_handler):
            if not first_request_started.is_set():
                first_request_started.set()
                await release_first_request.wait()
                event_handler(ModelStreamEvent(kind="assistant_delta", payload={"delta": "partial"}))
                return ModelResponse([AssistantMessage("partial")])

            event_handler(ModelStreamEvent(kind="assistant_delta", payload={"delta": "second"}))
            return ModelResponse([AssistantMessage("second")])

    runtime = CliSubmissionQueue(Agent(_DelayedModelClient(), ToolRegistry()))
    link = WorkspaceInteractiveSession(runtime)
    await link.start()
    try:
        first_result = await link.submit("first")
        assert first_result["ok"] is True
        await first_request_started.wait()

        second_result = await link.submit("second")
        assert second_result["ok"] is True
        release_first_request.set()

        snapshot = await _wait_for_async_snapshot(
            link,
            lambda item: any(
                turn["prompt"] == "second" and turn["response"] == "second"
                for turn in item["turns"]
            ),
        )
        assistant_turns = [
            turn for turn in snapshot["turns"] if turn["kind"] == "assistant"
        ]
        assert assistant_turns[0]["prompt"] == "first"
        assert assistant_turns[0]["error"] == ""
        assert assistant_turns[-1]["prompt"] == "second"
        assert assistant_turns[-1]["response"] == "second"
        assert all(not turn["error"] for turn in assistant_turns)
    finally:
        release_first_request.set()
        await link.close()


@pytest.mark.asyncio
async def test_workspace_session_steer_does_not_overwrite_active_tool_turn() -> None:
    first_request_started = asyncio.Event()
    release_first_request = asyncio.Event()

    class _DelayedToolModelClient:
        model = "delayed-tool-test"

        def __init__(self) -> None:
            self.call_count = 0

        async def complete(self, _prompt, _event_handler):
            self.call_count += 1
            if self.call_count == 1:
                return ModelResponse(
                    [ToolCall(call_id="call_1", name="wait", arguments={})]
                )
            if self.call_count == 2:
                first_request_started.set()
                await release_first_request.wait()
                return ModelResponse([AssistantMessage("first done")])
            return ModelResponse([AssistantMessage("second done")])

    class _WaitTool(BaseTool):
        name = "wait"
        description = "Wait briefly."
        input_schema = {"type": "object"}

        async def run(self, context, args):
            del context, args
            return "ready"

    tools = ToolRegistry()
    tools.register(_WaitTool())
    runtime = CliSubmissionQueue(Agent(_DelayedToolModelClient(), tools))
    link = WorkspaceInteractiveSession(runtime)
    await link.start()
    try:
        first_result = await link.submit("first")
        assert first_result["ok"] is True
        await first_request_started.wait()

        second_result = await link.submit("second")
        assert second_result["ok"] is True
        release_first_request.set()

        snapshot = await _wait_for_async_snapshot(
            link,
            lambda item: len(
                [turn for turn in item["turns"] if turn["kind"] == "assistant"]
            )
            == 2
            and item["turns"][-1]["response"] == "second done",
        )
        assistant_turns = [
            turn for turn in snapshot["turns"] if turn["kind"] == "assistant"
        ]
        assert [turn["prompt"] for turn in assistant_turns] == ["first", "second"]
        assert [turn["response"] for turn in assistant_turns] == [
            "first done",
            "second done",
        ]
        assert all(not turn["error"] for turn in assistant_turns)
        assert "[steer]" not in json.dumps(snapshot)
    finally:
        release_first_request.set()
        await link.close()


@pytest.mark.asyncio
async def test_workspace_session_close_cancels_stuck_turn(monkeypatch) -> None:
    started = asyncio.Event()

    class _StuckModelClient:
        model = "stuck-test"

        async def complete(self, _prompt, _event_handler):
            started.set()
            await asyncio.Event().wait()

    monkeypatch.setattr("workspace_server.app.SESSION_CLOSE_TIMEOUT_SECONDS", 0.01)
    runtime = CliSubmissionQueue(Agent(_StuckModelClient(), ToolRegistry()))
    link = WorkspaceInteractiveSession(runtime)
    await link.start()
    result = await link.submit("hang")
    assert result["ok"] is True
    await asyncio.wait_for(started.wait(), timeout=5.0)

    await asyncio.wait_for(link.close(), timeout=5.0)

    assert link._task is None
    assert runtime._current_task is None or runtime._current_task.done()


@pytest.mark.asyncio
async def test_workspace_session_handles_shell_command_before_model() -> None:
    model = ScriptedModelClient(
        responses=[ModelResponse([AssistantMessage("slash response")])]
    )
    runtime = CliSubmissionQueue(Agent(model, ToolRegistry()))
    link = WorkspaceInteractiveSession(runtime)
    await link.start()
    try:
        result = await link.submit("/history")

        assert result["ok"] is True

        snapshot = await _wait_for_async_snapshot(
            link,
            lambda item: item["turns"] and item["turns"][-1]["response"] == "No history yet.",
        )
        assert snapshot["turns"][-1]["kind"] == "control"
        assert model.call_count == 0
    finally:
        await link.close()


@pytest.mark.asyncio
async def test_threaded_workspace_session_runs_turn_in_worker_thread() -> None:
    server_thread = threading.get_ident()
    model_thread = None

    class _ThreadRecordingModel:
        model = "thread-test"

        async def complete(self, _prompt, _event_handler):
            nonlocal model_thread
            model_thread = threading.get_ident()
            return ModelResponse([AssistantMessage("threaded response")])

    def build_session():
        return WorkspaceInteractiveSession(
            CliSubmissionQueue(Agent(_ThreadRecordingModel(), ToolRegistry()))
        )

    link = ThreadedWorkspaceInteractiveSession(build_session, asyncio.get_running_loop())
    await link.start()
    try:
        result = await link.submit("hello")
        assert result["ok"] is True

        snapshot = await _wait_for_async_snapshot(
            link,
            lambda item: item["turns"]
            and item["turns"][-1]["response"] == "threaded response",
        )
        assert snapshot["turns"][-1]["prompt"] == "hello"
        assert model_thread is not None
        assert model_thread != server_thread
    finally:
        await link.close()


def test_workspace_app_threaded_session_blocking_turn_does_not_block_other_session(tmp_path) -> None:
    board = tmp_path / "board.html"
    board.write_text("<!doctype html><title>Board</title>", encoding="utf-8")

    class _BlockingModel:
        model = "blocking-test"

        async def complete(self, _prompt, _event_handler):
            time_module.sleep(0.4)
            return ModelResponse([AssistantMessage("blocked done")])

    class _FastModel:
        model = "fast-test"

        async def complete(self, _prompt, _event_handler):
            return ModelResponse([AssistantMessage("fast done")])

    created = 0

    def build_link():
        nonlocal created
        created += 1
        model = _BlockingModel() if created == 1 else _FastModel()

        def build_session():
            return WorkspaceInteractiveSession(
                CliSubmissionQueue(Agent(model, ToolRegistry()))
            )

        return ThreadedWorkspaceInteractiveSession(
            build_session,
            asyncio.get_running_loop(),
        )

    app = create_app(WorkspaceSessionManager(build_link), board)

    with TestClient(app) as client:
        first_id = client.get("/api/sessions").json()["sessions"][0]["id"]
        second_id = client.post("/api/sessions").json()["session_id"]
        first_started = time_module.perf_counter()
        client.post(
            "/api/session/message",
            json={"session_id": first_id, "prompt": "block"},
        )
        elapsed = time_module.perf_counter() - first_started
        assert elapsed < 0.25

        response_started = time_module.perf_counter()
        response = client.get(
            "/api/session?session_id={0}".format(second_id)
        )
        elapsed = time_module.perf_counter() - response_started

    assert response.status_code == 200
    assert elapsed < 0.25


class _DormantLink:
    def __init__(self, label="test"):
        self.label = label
        self.closed = False
        self.prompts = []

    async def start(self):
        return self

    async def close(self):
        self.closed = True
        return None

    def _snapshot(self):
        return {
            "running": False,
            "status": "",
            "status_kind": "idle",
            "spinner": "",
            "model": self.label,
            "title": self.label,
            "turns": [
                {
                    "submission_id": "sub_1",
                    "turn_id": "turn_1",
                    "prompt": "ping",
                    "response": "pong",
                    "thinking": "",
                    "status": "completed",
                    "error": "",
                    "queue": "history",
                    "sender": "test",
                    "kind": "assistant",
                }
            ] + [
                {
                    "submission_id": "sub_{0}".format(index + 2),
                    "turn_id": "turn_{0}".format(index + 2),
                    "prompt": item,
                    "response": "submitted",
                    "thinking": "",
                    "status": "completed",
                    "error": "",
                    "queue": "steer",
                    "sender": "test",
                    "kind": "assistant",
                }
                for index, item in enumerate(self.prompts)
            ],
        }

    def snapshot(self):
        return self._snapshot()

    def subscribe(self):
        queue = asyncio.Queue()
        queue.put_nowait({"type": "hello", "events": [], "snapshot": self._snapshot()})
        return queue

    def unsubscribe(self, _queue):
        return None

    async def restore_from_rollout(self, _rollout_path, title=""):
        if title:
            self.label = str(title or "")

    def rollout_path(self):
        return ""

    async def submit(self, prompt, sender="web"):
        self.prompts.append(prompt)
        snapshot = self._snapshot()
        return {"ok": True, "type": "submitted", "snapshot": snapshot}


class _ScrollAwareLink:
    async def start(self):
        return self

    async def close(self):
        return None

    def _snapshot(self):
        return {
            "running": True,
            "status": "thinking",
            "status_kind": "spinner",
            "spinner": "thinking",
            "model": "test",
            "title": "",
            "turns": [
                {
                    "submission_id": "sub_1",
                    "turn_id": "turn_1",
                    "prompt": "prompt",
                    "response": "response",
                    "thinking": "",
                    "status": "completed",
                    "error": "",
                    "queue": "steer",
                    "sender": "test",
                    "kind": "assistant",
                }
            ],
        }

    def snapshot(self):
        return self._snapshot()

    def subscribe(self):
        queue = asyncio.Queue()
        queue.put_nowait({"type": "hello", "events": [], "snapshot": self._snapshot()})
        return queue

    def unsubscribe(self, _queue):
        return None

    async def restore_from_rollout(self, _rollout_path, title=""):
        del title
        return None

    def rollout_path(self):
        return ""

    async def submit(self, prompt, sender="web"):
        return {"ok": True, "type": "control", "snapshot": self._snapshot()}
