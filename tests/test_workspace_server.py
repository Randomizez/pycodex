import asyncio
import json
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from pycodex import (
    Agent,
    AgentRuntime,
    AssistantMessage,
    BaseTool,
    ContextConfig,
    ModelResponse,
    ToolCall,
    ToolRegistry,
)
from pycodex.events import (
    AssistantDeltaEvent,
    AutoCompactCompletedEvent,
    CommandCompletedEvent,
    CompactCompletedEvent,
    InputRequestedEvent,
    SessionClosedEvent,
    SessionStateEvent,
    TokenCountEvent,
    ToolCompletedEvent,
    ToolStartedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
    TurnInterruptedEvent,
    TurnStartedEvent,
)
from pycodex.protocol import ToolResult
from pycodex.runtime import SubmissionInterrupted
from pycodex.utils import uuid7_string
from pycodex.utils.session_persist import resolve_codex_home, rollout_path_for_session
from tests.fakes import ScriptedModelClient
from workspace_server import (
    ThreadedWorkspaceInteractiveSession,
    WebSessionView,
    WorkspaceDefinition,
    WorkspaceEntry,
    WorkspaceInteractiveSession,
    WorkspaceRegistry,
    WorkspaceSessionManager,
    WorkspaceStateStore,
    build_parser,
    create_app,
    create_multi_workspace_app,
    load_workspace_definitions,
    parse_listen,
)
from workspace_server.app import _build_workspace_entry, _event_data


def make_session(model=None, tools=None):
    if model is None:
        model = ScriptedModelClient(
            response_factory=lambda prompt, count: ModelResponse(
                [AssistantMessage("done")]
            )
        )
    session_id = uuid7_string()
    return WorkspaceInteractiveSession(
        AgentRuntime(
            Agent(
                model,
                tools or ToolRegistry(),
                ContextConfig(),
                session_file_path=rollout_path_for_session(
                    resolve_codex_home(), session_id
                ),
                session_id=session_id,
            )
        )
    )


def make_entry(definition, persist_callback=None):
    return WorkspaceEntry(
        definition,
        WorkspaceSessionManager(
            make_session,
            definition.board_path,
            persist_callback=persist_callback,
        ),
    )


def wait_snapshot(client, predicate, path="/api/session"):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get(path)
        assert response.status_code == 200
        snapshot = response.json()["snapshot"]
        if predicate(snapshot):
            return snapshot
        time.sleep(0.01)
    raise AssertionError("session did not reach expected state")


@pytest.mark.parametrize("late_poll", ["none", "success", "error"])
def test_closing_active_tab_restores_visited_tab(late_poll):
    source = (
        Path(__file__).parents[1] / "workspace_server" / "workspace.html"
    ).read_text(encoding="utf-8")
    fragments = []
    for start, end in [
        ("    let pollTimer = null;", "    const markdownRenderer ="),
        ("    function stateForSession(", "    function renderTabs("),
        ("    function messageSignature(", "    function startPolling("),
        ("    function renderSnapshot(", "    function renderTurn("),
        ("    async function switchSession(", '    form.addEventListener("submit"'),
    ]:
        fragments.append(source[source.index(start) : source.index(end)])
    script = r"""
const assert = require("assert").strict;
const log = {textContent: "", scrollTop: 0, scrollHeight: 1000, clientHeight: 100};
const prompt = {value: ""};
const relativeUrl = path => path;
const resizePrompt = () => {};
const renderTabs = () => {};
const renderQueueHint = () => {};
const updateContextMeter = () => {};
const updatePromptPlaceholder = () => {};
const updateSpinnerNotifyUi = () => {};
const notifySessionDone = () => {};
const isNearBottom = () => false;
const scrollToBottom = () => {};
const setSpinner = text => { spinnerText = text; };
const addEntry = (_role, text) => { log.textContent += text; };
const renderTurn = turn => { log.textContent += turn.response; };
""" + "\n".join(fragments)
    script += r"""
const tabs = [{id: "a"}, {id: "b"}];
let serverTabs = tabs.slice();
function snapshot(id) {
  return {turns: [{prompt: `prompt ${id}`, response: `answer ${id}`}],
          spinner: "", queued_inputs: []};
}
function response(id) {
  const payload = {session_id: id, sessions: serverTabs.slice(), snapshot: snapshot(id)};
  return {ok: true, json: async () => payload};
}
let delayOldPoll = false;
let finishOldPoll;
const fetch = async (path, options = {}) => {
  if (options.method === "DELETE") {
    serverTabs = [tabs[0]];
    return {ok: true, json: async () => ({ok: true, sessions: serverTabs.slice()})};
  }
  const id = path.endsWith("=b") ? "b" : "a";
  const reply = response(id);
  if (delayOldPoll && id === "b") {
    // Deliver a late response even if the browser has already aborted it.
    return new Promise((resolve, reject) => {
      finishOldPoll = () => {
        if (process.argv[2] === "error") reject(new Error("old tab is gone"));
        else resolve(reply);
      };
    });
  }
  return reply;
};
(async () => {
  sessions = tabs.slice();
  activeSessionId = "a";
  renderSnapshot(snapshot("a"));
  prompt.value = "draft a";
  log.scrollTop = 123;
  await switchSession("b");
  assert.equal(log.textContent, "answer b");
  prompt.value = "draft b";
  log.scrollTop = 9;
  delayOldPoll = process.argv[2] !== "none";
  const oldPoll = delayOldPoll ? pollSession() : null;
  await closeSession("b");
  assert.equal(activeSessionId, "a");
  assert.equal(log.textContent, "answer a");
  assert.equal(prompt.value, "draft a");
  assert.equal(log.scrollTop, 123);
  assert.equal(sessionState.has("b"), false);
  if (oldPoll) {
    finishOldPoll();
    await oldPoll;
  }
  assert.equal(activeSessionId, "a");
  assert.deepEqual(sessions.map(tab => tab.id), ["a"]);
  assert.equal(spinnerText, "");
  assert.equal(log.textContent, "answer a");
  await pollSession();
  assert.equal(log.textContent, "answer a");
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        ["node", "-", late_poll],
        input=script,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_typed_event_wire_preserves_existing_fields():
    call = ToolCall("call", "echo", {"text": "hello"})
    result = ToolResult("call", "echo", "hello")
    assert _event_data(TurnStartedEvent("turn", ("first", "second"), "submission")) == {
        "kind": "turn_started",
        "turn_id": "turn",
        "payload": {
            "user_text": "first\nsecond",
            "user_texts": ["first", "second"],
            "submission_id": "submission",
            "turn_id": "turn",
        },
    }
    assert _event_data(ToolStartedEvent("turn", call))["payload"] == {
        "tool_name": "echo",
        "call_id": "call",
        "call": asdict(call),
    }
    assert _event_data(ToolCompletedEvent("turn", call, result))["payload"] == {
        "tool_name": "echo",
        "call_id": "call",
        "call": asdict(call),
        "result": asdict(result),
        "is_error": False,
    }
    assert _event_data(TokenCountEvent(None))["payload"] == {"usage": None}
    assert _event_data(
        AutoCompactCompletedEvent(
            "turn",
            "pre_turn",
            None,
            None,
            10,
            1,
            0,
        )
    )["payload"] == {
        "phase": "pre_turn",
        "original_item_count": 10,
        "retained_item_count": 1,
        "summary": "compact(10 items) -> 1 item + [summary]",
    }
    assert _event_data(TurnFailedEvent("turn", 1, "failed", "ValueError", None))[
        "payload"
    ] == {
        "iteration": 1,
        "error": "failed",
        "error_type": "ValueError",
    }
    question = {
        "id": "choice",
        "header": "Choice",
        "question": "Choose a path",
        "options": [{"label": "Alpha", "description": "Path A"}],
    }
    request = InputRequestedEvent("request", "questions", False, question=question)
    request_data = {
        "request_id": "request",
        "kind": "questions",
        "other": False,
        "question": question,
        "text": request.visualize(),
    }
    assert _event_data(request)["payload"] == request_data
    assert _event_data(SessionStateEvent("attach", {"input_request": request}))[
        "payload"
    ] == {
        "reason": "attach",
        "state": {"input_request": request_data},
    }
    assert _event_data(
        CommandCompletedEvent("command", "/help", {"kind": "help"}, "web")
    ) == {
        "kind": "command_completed",
        "turn_id": "command",
        "payload": {"command": "/help", "result": {"kind": "help"}, "sender": "web"},
    }
    assert _event_data(SessionClosedEvent()) == {
        "kind": "session_closed",
        "turn_id": "",
        "payload": {},
    }


@pytest.mark.asyncio
async def test_history_command_publishes_one_control_turn():
    view = WebSessionView()
    history = tuple(
        ("question {}".format(index), "answer {}".format(index)) for index in range(40)
    )
    view.load_session_history("history", history)
    subscriber = view.subscribe()
    subscriber.get_nowait()
    try:
        view.handle_event(
            CommandCompletedEvent(
                "command",
                "/history",
                {"kind": "history", "state": {"title": "history", "history": history}},
                "web",
            )
        )
        turns = view.snapshot()["turns"]
        assert len(turns) == 41
        assert [(turn["prompt"], turn["response"]) for turn in turns[:-1]] == list(
            history
        )
        assert len(turns[-1]["response"].splitlines()) == 81
        assert turns[-1]["response"].endswith("[40]A> answer 39")
        assert subscriber.qsize() == 2
    finally:
        view.unsubscribe(subscriber)
        view.close()


def test_workspace_configuration_and_entrypoint(tmp_path):
    config = tmp_path / "workspaces.json"
    assert load_workspace_definitions(config) == []
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    config.write_text(
        json.dumps(
            {
                "workspaces": [
                    {"id": "first", "work_dir": "first", "board": "first/board.html"},
                    {"work_dir": "second", "board": "second/board.html"},
                ]
            }
        ),
        encoding="utf-8",
    )
    definitions = load_workspace_definitions(config)
    assert [item.workspace_id for item in definitions] == ["first", "workspace-1"]
    assert definitions[0].work_dir == tmp_path / "first"
    assert definitions[1].board_path == tmp_path / "second/board.html"
    args = build_parser().parse_args(
        [
            "--listen",
            "0.0.0.0:6007",
            "--workspace-config",
            str(config),
            "--password",
            "test",
        ]
    )
    assert parse_listen(args.listen) == ("0.0.0.0", 6007)
    assert (args.workspace_config, args.password) == (str(config), "test")
    assert "--board" not in build_parser().format_help()
    with pytest.raises(ValueError):
        parse_listen("not-a-port")


@pytest.mark.parametrize("multi", [False, True])
def test_board_routes_enforce_image_boundaries(tmp_path, multi):
    board_dir = tmp_path / "board"
    board_dir.mkdir()
    board = board_dir / "board.html"
    board.write_text('<!doctype html><img src="plot.png">', encoding="utf-8")
    image = board_dir / "plot.png"
    image.write_bytes(b"board image")
    (board_dir / "notes.txt").write_text("private notes", encoding="utf-8")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    (board_dir / "linked.png").symlink_to(outside)
    if multi:
        definition = WorkspaceDefinition("first", board, board_dir)
        app = create_multi_workspace_app(WorkspaceRegistry([make_entry(definition)]))
        prefix = "/w/first"
    else:
        app = create_app(make_session, board)
        prefix = ""
    with TestClient(app) as browser:
        assert browser.get(prefix + "/board").text == board.read_text(encoding="utf-8")
        response = browser.get(prefix + "/plot.png")
        assert response.content == b"board image"
        assert response.headers["content-type"] == "image/png"
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert browser.head(prefix + "/plot.png").content == b""
        for path in ("notes.txt", "linked.png", "%2e%2e/outside.png"):
            assert browser.get(prefix + "/" + path).status_code == 404
        assert browser.get(prefix + "/api/session").status_code == 200
        if multi:
            assert browser.get("/w/missing/api/session").status_code == 404


@pytest.mark.parametrize("multi", [False, True])
def test_password_protects_http_and_websocket(tmp_path, multi):
    board = tmp_path / "board.html"
    board.write_text("board", encoding="utf-8")
    if multi:
        app = create_multi_workspace_app(
            WorkspaceRegistry(
                [
                    make_entry(WorkspaceDefinition("first", board, tmp_path)),
                ]
            ),
            password="test-password",
        )
        prefix = "/w/first"
    else:
        app = create_app(make_session, board, password="test-password")
        prefix = ""
    with TestClient(app) as browser:
        target = prefix + "/?session=saved&view=chat"
        response = browser.get(target)
        assert response.status_code == 200
        assert len(response.history) == 1
        assert response.history[0].status_code == 303
        login_url = response.history[0].headers["location"]
        assert browser.get(prefix + "/api/session").status_code == 401
        assert browser.post(login_url, json={"password": "wrong"}).status_code == 401
        with pytest.raises(WebSocketDisconnect):
            with browser.websocket_connect(prefix + "/ws/session"):
                pass
        response = browser.post(login_url, json={"password": "test-password"})
        assert response.status_code == 200
        assert response.json() == {"ok": True, "redirect": target}
        assert browser.get(response.json()["redirect"]).status_code == 200
        assert browser.get(prefix + "/board").status_code == 200
        assert browser.get(prefix + "/api/session").status_code == 200
        with browser.websocket_connect(prefix + "/ws/session") as websocket:
            assert websocket.receive_json()["type"] == "hello"
            websocket.send_json({"type": "ping"})
            assert websocket.receive_json()["type"] == "pong"


@pytest.mark.parametrize(
    "target",
    [None, "https://example.com", "//example.com", "/\\example.com", "/\t/example.com"],
)
def test_login_redirect_stays_on_server(target):
    with TestClient(
        create_app(make_session, None, password="test-password")
    ) as browser:
        response = browser.post(
            "/login",
            params={} if target is None else {"next": target},
            json={"password": "test-password"},
        )
        assert response.status_code == 200
        assert response.json()["redirect"] == "/"


def test_workspace_registry_crud_and_duplicate_board(tmp_path):
    config = tmp_path / "workspaces.json"
    registry = WorkspaceRegistry([], config_path=config, entry_factory=make_entry)
    with TestClient(create_multi_workspace_app(registry)) as browser:
        added = browser.post(
            "/api/workspaces",
            json={
                "name": "alpha",
                "dir": "repos/alpha",
                "board": "boards/alpha.html",
            },
        )
        assert added.status_code == 200
        assert (tmp_path / "repos/alpha").is_dir()
        duplicate = browser.post(
            "/api/workspaces",
            json={
                "name": "beta",
                "dir": "repos/beta",
                "board": "boards/alpha.html",
            },
        )
        assert duplicate.status_code == 400
        assert "board already exists" in duplicate.json()["error"]
        generated = browser.post("/api/workspaces", json={"dir": "repos/generated"})
        assert generated.status_code == 200
        workspace = generated.json()["workspace"]
        assert workspace["id"]
        assert workspace["board_path"].startswith("/tmp/pcws-")
        assert (
            "Board pending" in browser.get("/w/{}/board".format(workspace["id"])).text
        )
        assert browser.get("/w/alpha/").status_code == 200
        assert browser.delete("/api/workspaces/alpha").status_code == 200
        assert browser.get("/w/alpha/api/session").status_code == 404
    definitions = load_workspace_definitions(config)
    assert len(definitions) == 1
    assert definitions[0].workspace_id == workspace["id"]


@pytest.mark.parametrize("toolset", [["exec_command", "apply_patch"], []])
def test_boardless_workspace_preserves_context_and_tools(
    tmp_path, monkeypatch, toolset
):
    config = tmp_path / "workspaces.json"
    model_config = tmp_path / "model.toml"
    model_config.write_text('model = "test"\n', encoding="utf-8")
    model = ScriptedModelClient([ModelResponse([AssistantMessage("done")])])
    monkeypatch.setattr("workspace_server.app.build_model", lambda **kwargs: model)
    args = build_parser().parse_args(
        [
            "--config",
            str(model_config),
            "--system-prompt",
            "Fix the task.",
            "--toolset",
        ]
        + toolset
    )
    registry = WorkspaceRegistry(
        [],
        config_path=config,
        entry_factory=lambda definition, persist_callback: _build_workspace_entry(
            definition,
            args,
            persist_callback,
        ),
    )
    with TestClient(create_multi_workspace_app(registry)) as browser:
        response = browser.post(
            "/api/workspaces",
            json={
                "name": "alpha",
                "dir": "repo",
                "board": False,
            },
        )
        assert response.status_code == 200
        assert response.json()["workspace"]["board_path"] == ""
        browser.post("/w/alpha/api/session/message", json={"prompt": "fix"})
        wait_snapshot(
            browser,
            lambda state: state["turns"] and state["turns"][-1]["response"] == "done",
            "/w/alpha/api/session",
        )
    prompt = model.prompts[0]
    assert prompt.base_instructions == "Fix the task."
    assert str(tmp_path / "repo") in repr(prompt.input)
    assert "Current workspace board file:" not in repr(prompt.input)
    assert [tool.name for tool in prompt.tools] == toolset
    assert load_workspace_definitions(config)[0].board_path is None


def test_sessions_persist_explicit_titles_and_restore_history(tmp_path):
    board = tmp_path / "board.html"
    board.write_text("unchanged board", encoding="utf-8")
    store = WorkspaceStateStore(board)
    with TestClient(create_app(make_session, board)) as browser:
        first_id = browser.get("/api/sessions").json()["sessions"][0]["id"]
        browser.post(
            "/api/session/message",
            json={"session_id": first_id, "prompt": "saved prompt"},
        )
        state = wait_snapshot(
            browser,
            lambda item: item["turns"] and item["turns"][-1]["response"] == "done",
        )
        assert store.load_tabs() == []
        response = browser.post(
            "/api/session/message",
            json={"session_id": first_id, "prompt": "/title saved"},
        )
        assert response.status_code == 200
        second_id = browser.post("/api/sessions").json()["session_id"]
        assert len(browser.get("/api/sessions").json()["sessions"]) == 2
        assert browser.delete("/api/sessions/" + second_id).status_code == 200
        assert browser.delete("/api/sessions/" + first_id).status_code == 400
    assert board.read_text(encoding="utf-8") == "unchanged board"
    assert store.load_tabs() == [
        {"title": "saved", "rollout_path": state["rollout_path"]}
    ]
    with TestClient(create_app(make_session, board)) as browser:
        restored = browser.get("/api/session").json()["snapshot"]
        assert restored["title"] == "saved"
        assert [(turn["prompt"], turn["response"]) for turn in restored["turns"]] == [
            ("saved prompt", "done")
        ]
        assert (
            browser.post(
                "/api/session/message", json={"prompt": "/resume 1"}
            ).status_code
            == 200
        )
        assert len(browser.get("/api/session").json()["snapshot"]["turns"]) == 1


@pytest.mark.parametrize("threaded", [False, True])
@pytest.mark.parametrize("continue_after_fork", [False, True])
def test_workspace_fork_survives_restart(tmp_path, threaded, continue_after_fork):
    board = tmp_path / "board.html"
    store = WorkspaceStateStore(board)

    def session_factory():
        if threaded:
            return ThreadedWorkspaceInteractiveSession(
                make_session, asyncio.get_running_loop()
            )
        return make_session()

    def send(browser, prompt):
        response = browser.post("/api/session/message", json={"prompt": prompt})
        assert response.status_code == 200
        return response.json()["snapshot"]

    def wait_reply(browser, prompt):
        return wait_snapshot(
            browser,
            lambda state: any(
                turn["prompt"] == prompt and turn["response"] == "done"
                for turn in state["turns"]
            ),
        )

    def conversation(state):
        return [
            (turn["prompt"], turn["response"])
            for turn in state["turns"]
            if turn["kind"] != "control"
        ]

    with TestClient(create_app(session_factory, board)) as browser:
        send(browser, "before fork")
        state = wait_reply(browser, "before fork")
        original_path = Path(state["rollout_path"])
        original_bytes = original_path.read_bytes()
        send(browser, "/title saved")
        state = send(browser, "/fork")
        assert not Path(state["rollout_path"]).exists()
        send(browser, "/title forked")
        # A second pending fork must still restore from the recorded ancestor.
        state = send(browser, "/fork")
        fork_path = Path(state["rollout_path"])
        assert fork_path != original_path
        assert not fork_path.exists()
        if continue_after_fork:
            send(browser, "after fork")
            wait_reply(browser, "after fork")

    expected = [("before fork", "done")]
    if continue_after_fork:
        expected.append(("after fork", "done"))
        assert store.load_tabs() == [
            {"title": "forked", "rollout_path": str(fork_path)}
        ]
    else:
        assert not fork_path.exists()
        assert store.load_tabs() == [
            {"title": "forked", "rollout_path": str(original_path), "fork": True}
        ]

    with TestClient(create_app(session_factory, board)) as browser:
        state = browser.get("/api/session").json()["snapshot"]
        assert state["title"] == "forked"
        assert conversation(state) == expected
        restored_path = Path(state["rollout_path"])
        assert restored_path != original_path
        if continue_after_fork:
            assert restored_path == fork_path
        else:
            assert not restored_path.exists()
        send(browser, "after restart")
        assert conversation(wait_reply(browser, "after restart")) == expected + [
            ("after restart", "done")
        ]

    assert original_path.read_bytes() == original_bytes
    assert store.load_tabs() == [
        {"title": "forked", "rollout_path": str(restored_path)}
    ]


def test_titled_empty_workspace_tab_survives_restart(tmp_path):
    board = tmp_path / "board.html"
    store = WorkspaceStateStore(board)
    with TestClient(create_app(make_session, board)) as browser:
        response = browser.post("/api/session/message", json={"prompt": "/title empty"})
        assert response.status_code == 200
        path = Path(response.json()["snapshot"]["rollout_path"])
        assert not path.exists()
    assert store.load_tabs() == [{"title": "empty", "rollout_path": ""}]

    with TestClient(create_app(make_session, board)) as browser:
        state = browser.get("/api/session").json()["snapshot"]
        assert state["title"] == "empty"
        assert not state["turns"]
        browser.post("/api/session/message", json={"prompt": "first prompt"})
        state = wait_snapshot(
            browser,
            lambda item: item["turns"] and item["turns"][-1]["response"] == "done",
        )
    assert store.load_tabs() == [
        {"title": "empty", "rollout_path": state["rollout_path"]}
    ]


@pytest.mark.parametrize("command", ["/title saved prompt", "/resume 1"])
def test_workspace_explicit_save_keeps_unchanged_state(tmp_path, command):
    board = tmp_path / "board.html"
    store = WorkspaceStateStore(board)
    with TestClient(create_app(make_session, board)) as browser:
        browser.post("/api/session/message", json={"prompt": "saved prompt"})
        state = wait_snapshot(
            browser, lambda s: s["turns"] and s["turns"][-1]["response"] == "done"
        )
        assert store.load_tabs() == []
        response = browser.post("/api/session/message", json={"prompt": command})
        assert response.status_code == 200
    assert store.load_tabs() == [
        {"title": "saved prompt", "rollout_path": state["rollout_path"]}
    ]


@pytest.mark.parametrize("next_input", ["/compact", "failed turn"])
def test_workspace_fork_saves_new_target_after_compact_or_failure(tmp_path, next_input):
    board = tmp_path / "board.html"
    store = WorkspaceStateStore(board)

    def respond(prompt, count):
        if count == 2 and next_input == "failed turn":
            raise RuntimeError("offline model failure")
        return ModelResponse([AssistantMessage("done")])

    def session_factory():
        return make_session(ScriptedModelClient(response_factory=respond))

    with TestClient(create_app(session_factory, board)) as browser:
        browser.post("/api/session/message", json={"prompt": "before fork"})
        state = wait_snapshot(
            browser, lambda s: s["turns"] and s["turns"][-1]["response"] == "done"
        )
        original_path = Path(state["rollout_path"])
        original_bytes = original_path.read_bytes()
        browser.post("/api/session/message", json={"prompt": "/title saved"})
        response = browser.post("/api/session/message", json={"prompt": "/fork"})
        fork_path = Path(response.json()["snapshot"]["rollout_path"])
        browser.post("/api/session/message", json={"prompt": next_input})
        if next_input == "failed turn":
            wait_snapshot(browser, lambda s: s["turns"][-1]["status"] == "error")
    assert fork_path.is_file()
    assert store.load_tabs() == [{"title": "saved", "rollout_path": str(fork_path)}]
    assert original_path.read_bytes() == original_bytes
    with TestClient(create_app(session_factory, board)) as browser:
        assert browser.get("/api/session").json()["snapshot"]["rollout_path"] == str(
            fork_path
        )


@pytest.mark.asyncio
async def test_workspace_close_persists_fork_after_draining_turn(tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, count):
        if count == 2:
            entered.set()
            await release.wait()
        return ModelResponse([AssistantMessage("done")])

    session = make_session(ScriptedModelClient(response_factory=respond))
    board = tmp_path / "board.html"
    manager = WorkspaceSessionManager(lambda: session, board)
    await manager.start()
    first = await session.runtime.submit_input("before fork")
    await first.future
    await session.submit("/title saved")
    await session.submit("/fork")
    fork_path = session.runtime.agent.session_file_path
    second = await session.runtime.submit_input("after fork")
    await entered.wait()
    closing = asyncio.create_task(manager.close())
    try:
        await asyncio.sleep(0)
        assert not closing.done()
    finally:
        release.set()
        await closing
    await second.future
    assert WorkspaceStateStore(board).load_tabs() == [
        {"title": "saved", "rollout_path": str(fork_path)}
    ]


@pytest.mark.asyncio
async def test_workspace_missing_restore_path_still_fails_and_closes_session(tmp_path):
    session = make_session()
    board = tmp_path / "board.html"
    missing_path = str(tmp_path / "missing-rollout.jsonl")
    WorkspaceStateStore(board).save_tabs(
        [{"title": "saved", "rollout_path": missing_path, "fork": True}]
    )
    manager = WorkspaceSessionManager(lambda: session, board)
    with pytest.raises(FileNotFoundError):
        await manager.start()
    assert session.runtime.agent.is_shutdown
    assert not manager.list_sessions()


def test_workspace_does_not_save_a_fork_destination_owned_by_another_writer(tmp_path):
    board = tmp_path / "board.html"
    with TestClient(create_app(make_session, board)) as browser:
        browser.post("/api/session/message", json={"prompt": "before fork"})
        state = wait_snapshot(
            browser, lambda s: s["turns"] and s["turns"][-1]["response"] == "done"
        )
        original_path = state["rollout_path"]
        browser.post("/api/session/message", json={"prompt": "/title saved"})
        response = browser.post("/api/session/message", json={"prompt": "/fork"})
        fork_path = Path(response.json()["snapshot"]["rollout_path"])
        fork_path.write_text("another writer", encoding="utf-8")
        browser.post("/api/session/message", json={"prompt": "after fork"})
        state = wait_snapshot(browser, lambda s: s["turns"][-1]["status"] == "error")
        assert state["recorded_rollout_path"] is None
        assert "File exists" in state["turns"][-1]["error"]
    assert fork_path.read_text(encoding="utf-8") == "another writer"
    assert WorkspaceStateStore(board).load_tabs() == [
        {"title": "saved", "rollout_path": original_path, "fork": True}
    ]


def test_workspace_resume_hides_compact_handoff_and_keeps_real_reply(tmp_path):
    model = ScriptedModelClient(
        [
            ModelResponse([AssistantMessage("old answer")]),
            ModelResponse([AssistantMessage("handoff summary")]),
            ModelResponse([AssistantMessage("continued answer")]),
        ]
    )
    session_id = uuid7_string()
    source = Agent(
        model,
        ToolRegistry(),
        ContextConfig(),
        session_file_path=rollout_path_for_session(resolve_codex_home(), session_id),
        session_id=session_id,
    )
    source.ask("original prompt")
    asyncio.run(source.compact())
    asyncio.run(source.run_turn([]))
    source.shutdown()

    with TestClient(create_app(make_session, None)) as browser:
        response = browser.post("/api/session/message", json={"prompt": "/resume 1"})
        assert response.status_code == 200
        restored = browser.get("/api/session").json()["snapshot"]
        assert restored["title"] == "original prompt"
        assert [(turn["prompt"], turn["response"]) for turn in restored["turns"]] == [
            ("", "continued answer")
        ]
        assert (
            browser.post(
                "/api/session/message", json={"prompt": "next prompt"}
            ).status_code
            == 200
        )
        state = wait_snapshot(
            browser, lambda item: item["turns"][-1]["response"] == "done"
        )
        assert [(turn["prompt"], turn["response"]) for turn in state["turns"]] == [
            ("", "continued answer"),
            ("next prompt", "done"),
        ]


def test_session_list_uses_lightweight_summary(monkeypatch):
    def build():
        session = make_session()

        def no_snapshot():
            raise AssertionError("list should not build a full transcript")

        monkeypatch.setattr(session, "snapshot", no_snapshot)
        return session

    with TestClient(create_app(build, None)) as browser:
        response = browser.get("/api/sessions")
        assert response.status_code == 200
        assert response.json()["sessions"][0]["turn_count"] == 0


@pytest.mark.asyncio
async def test_web_view_projects_context_tool_and_stream_events():
    view = WebSessionView()
    runtime = AgentRuntime(
        Agent(
            ScriptedModelClient([]),
            ToolRegistry(),
            ContextConfig(
                model_context_window=100000, model_auto_compact_token_limit=80000
            ),
        )
    )
    view.handle_event(SessionStateEvent("attach", runtime.snapshot()))
    view.handle_event(TokenCountEvent({"total_tokens": 56000}, "turn"))
    view.handle_event(TurnStartedEvent("turn", ("hello",)))
    view.handle_event(AssistantDeltaEvent("part", "turn"))
    assert view.snapshot()["context_remaining_percent"] == 30
    assert view.snapshot()["turns"][-1]["thinking"] == "part"
    view.handle_event(TurnInterruptedEvent("turn", 1, "partial", 0))
    assert view.snapshot()["turns"][-1]["response"] == "partial"
    assert not view.snapshot()["turns"][-1]["error"]
    view.handle_event(TurnCompletedEvent("turn", 1, None, 1))
    assert view.snapshot()["spinner"] == "idle: sleeping"
    view.load_session_history("restored", (("old", "answer"),))
    assert len(view.snapshot()["turns"]) == 1
    subscriber = view.subscribe()
    hello = await subscriber.get()
    assert not any(event.get("kind") == "assistant_delta" for event in hello["events"])
    view.close()


@pytest.mark.parametrize(
    "compact_limit,max_len,current,remaining",
    [
        (80000, 100000, None, 100),
        (80000, 100000, 0, 100),
        (80000, 100000, 60000, 25),
        (80000, 100000, 79999, 1),
        (80000, 100000, 80000, 0),
        (80000, 100000, 81000, 0),
        (80000, None, 60000, 25),
        (None, 100000, 60000, 40),
        (None, None, 60000, None),
    ],
)
def test_web_context_meter_uses_compact_threshold(
    compact_limit, max_len, current, remaining
):
    runtime = AgentRuntime(
        Agent(
            ScriptedModelClient([]),
            ToolRegistry(),
            ContextConfig(
                model_context_window=max_len,
                model_auto_compact_token_limit=compact_limit,
            ),
        )
    )
    view = WebSessionView()
    view.handle_event(SessionStateEvent("attach", runtime.snapshot()))
    if current is not None:
        view.handle_event(TokenCountEvent({"total_tokens": current}, "turn"))
    for state in (view.snapshot(), view.summary()):
        assert state["usage_tokens"] == current
        assert state["auto_compact_token_limit"] == compact_limit
        assert state["max_context_window"] == max_len
        assert state["context_remaining_percent"] == remaining


@pytest.mark.parametrize(
    "completed",
    [
        AutoCompactCompletedEvent("turn", "pre_turn", 80000, 80000, 10, 1, 0),
        CompactCompletedEvent("turn", "manual", None, None, 10, 1, 0, 0),
    ],
)
def test_web_context_usage_clears_after_compaction(completed):
    runtime = AgentRuntime(
        Agent(
            ScriptedModelClient([]),
            ToolRegistry(),
            ContextConfig(
                model_context_window=100000, model_auto_compact_token_limit=80000
            ),
        )
    )
    view = WebSessionView()
    view.handle_event(SessionStateEvent("attach", runtime.snapshot()))
    view.handle_event(TokenCountEvent({"total_tokens": 80000}, "turn"))
    view.handle_event(completed)
    assert view.snapshot()["usage_tokens"] is None
    assert view.snapshot()["context_remaining_percent"] == 100
    view.handle_event(TokenCountEvent({"total_tokens": 8000}, "turn"))
    assert view.snapshot()["usage_tokens"] == 8000
    assert view.snapshot()["context_remaining_percent"] == 90
    view.handle_event(SessionStateEvent("history", runtime.snapshot()))
    assert view.snapshot()["usage_tokens"] is None


@pytest.mark.asyncio
async def test_workspace_displays_steer_and_enqueue_in_execution_order():
    started = [asyncio.Event() for _ in range(3)]
    release = [asyncio.Event() for _ in range(3)]

    async def respond(prompt, call_count):
        started[call_count - 1].set()
        await release[call_count - 1].wait()
        return ModelResponse([AssistantMessage("answer " + str(call_count))])

    model = ScriptedModelClient(response_factory=respond)
    session = await make_session(model).start()
    try:
        first = await session.runtime.submit_input("first")
        await asyncio.wait_for(started[0].wait(), 1)
        queued = await session.runtime.submit_input("/queue last", "cli")
        steered = await session.runtime.submit_input("steer one")
        merged = await session.runtime.submit_input("steer two")

        assert steered.submission_id == merged.submission_id
        assert [turn["prompt"] for turn in session.snapshot()["turns"]] == ["first"]
        assert session.summary()["turn_count"] == 1
        assert session.snapshot()["queued_inputs"] == [
            {"queue": "enqueue", "prompt": "last"},
            {"queue": "steer", "prompt": "steer one\nsteer two"},
        ]

        release[0].set()
        await asyncio.wait_for(started[1].wait(), 1)
        with pytest.raises(SubmissionInterrupted):
            await first.future
        turns = session.snapshot()["turns"]
        assert [turn["prompt"] for turn in turns] == ["first", "steer one\nsteer two"]
        assert turns[-1]["turn_id"] == turns[0]["turn_id"]
        assert turns[-1]["submission_id"] == steered.submission_id
        assert session.snapshot()["queued_inputs"] == [
            {"queue": "enqueue", "prompt": "last"}
        ]

        release[1].set()
        await asyncio.wait_for(started[2].wait(), 1)
        turns = session.snapshot()["turns"]
        assert [turn["prompt"] for turn in turns] == [
            "first",
            "steer one\nsteer two",
            "last",
        ]
        assert turns[-1]["queue"] == "enqueue"
        assert turns[-1]["sender"] == "cli"
        assert session.snapshot()["queued_inputs"] == []
        release[2].set()
        await asyncio.gather(queued.future, steered.future, merged.future)
    finally:
        for event in release:
            event.set()
        await session.close()


@pytest.mark.asyncio
async def test_tool_failure_remains_a_tool_result():
    class FailingTool(BaseTool):
        name = "fail"
        description = "Fails visibly to the model."

        async def run(self, context, args):
            raise ValueError("tool failed")

    tools = ToolRegistry()
    tools.register(FailingTool())
    model = ScriptedModelClient(
        [
            ModelResponse([ToolCall("call", "fail", {})]),
            ModelResponse([AssistantMessage("handled")]),
        ]
    )
    session = await make_session(model, tools).start()
    try:
        receipt = await session.runtime.submit_input("run")
        await receipt.future
        turn = session.snapshot()["turns"][-1]
        assert turn["response"] == "handled"
        assert not turn["error"]
    finally:
        await session.close()


def test_threaded_sessions_do_not_block_other_sessions():
    owner_thread = threading.get_ident()
    workers = []
    release = threading.Event()
    started = threading.Event()
    sessions = []

    class BlockingClient:
        model = "blocking"

        async def complete(self, prompt, event_handler):
            workers.append(threading.get_ident())
            started.set()
            assert release.wait(3)
            return ModelResponse([AssistantMessage("done")])

    def build():
        model = BlockingClient() if not sessions else None
        session = ThreadedWorkspaceInteractiveSession(
            lambda: make_session(model),
            asyncio.get_running_loop(),
        )
        sessions.append(session)
        return session

    with TestClient(create_app(build, None)) as browser:
        first_id = browser.get("/api/sessions").json()["sessions"][0]["id"]
        second_id = browser.post("/api/sessions").json()["session_id"]
        try:
            begin = time.monotonic()
            response = browser.post(
                "/api/session/message", json={"session_id": first_id, "prompt": "block"}
            )
            assert response.status_code == 200
            assert time.monotonic() - begin < 0.5
            assert started.wait(1)
            response = browser.get("/api/session", params={"session_id": second_id})
            assert response.status_code == 200
            assert workers and owner_thread not in workers
        finally:
            release.set()
    assert all(session._thread is None for session in sessions)


@pytest.mark.parametrize("failure", ["start", "close"])
@pytest.mark.asyncio
async def test_threaded_session_releases_resources_on_lifecycle_failure(failure):
    cleanup_calls = []

    def build():
        session = make_session()

        async def cleanup():
            cleanup_calls.append(True)
            if failure == "close":
                raise RuntimeError("cleanup failed")

        session.runtime.add_close_handler(cleanup)
        if failure == "start":
            original_start = session.start

            async def fail_start():
                await original_start()
                raise RuntimeError("startup failed")

            session.start = fail_start
        return session

    session = ThreadedWorkspaceInteractiveSession(build, asyncio.get_running_loop())
    if failure == "start":
        with pytest.raises(RuntimeError, match="failed to start"):
            await session.start()
    else:
        await session.start()
    thread = session._thread
    try:
        if failure == "close":
            with pytest.raises(RuntimeError, match="cleanup failed"):
                await session.close()
        else:
            await session.close()
        assert not thread.is_alive()
        assert session._worker_loop.is_closed()
        assert session._thread is None
        assert cleanup_calls == [True]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_threaded_close_drains_active_turn_when_caller_is_cancelled():
    started = threading.Event()
    release = threading.Event()
    cancelled = []

    async def respond(prompt, call_count):
        started.set()
        try:
            await asyncio.to_thread(release.wait)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return ModelResponse([AssistantMessage("done")])

    model = ScriptedModelClient(response_factory=respond)
    session = ThreadedWorkspaceInteractiveSession(
        lambda: make_session(model),
        asyncio.get_running_loop(),
    )
    await session.start()
    thread = session._thread
    closing = None
    try:
        await session.submit("finish")
        assert await asyncio.to_thread(started.wait, 1)
        closing = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(closing, 2)
        assert not cancelled
        assert not thread.is_alive()
        assert session.snapshot()["turns"][-1]["response"] == "done"
    finally:
        release.set()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await session.close()
