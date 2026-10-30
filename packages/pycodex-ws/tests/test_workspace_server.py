import asyncio
import json
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path

import pytest
from starlette.testclient import TestClient
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
    UserMessage,
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
from pycodex.model import ResponsesIncompleteError
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
from workspace_server.workspaces import save_workspace_definitions


def make_session(model=None, tools=None, cwd=None):
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
                ContextConfig(cwd=cwd),
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
            definition.state_path,
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


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("fields", [
    {"title": "Project review", "prompt": "请检查这个项目"},
    {"title": "Notes"},
    {"prompt": "Summarize the project"},
])
def test_new_session_accepts_title_and_initial_prompt(tmp_path, multi, fields):
    model = ScriptedModelClient(
        response_factory=lambda prompt, count: ModelResponse([AssistantMessage("done")])
    )
    factory = lambda: make_session(model, cwd=tmp_path)
    manager = WorkspaceSessionManager(factory, tmp_path / "state")
    if multi:
        app = create_multi_workspace_app(WorkspaceRegistry([WorkspaceEntry(
            WorkspaceDefinition("test", tmp_path / "state", tmp_path, False), manager,
        )]))
    else:
        app = create_app(manager, tmp_path / "state", work_dir=tmp_path, enable_board=False)
    prefix = "/w/test" if multi else ""
    with TestClient(app) as browser:
        original = browser.get(prefix + "/api/session").json()["session_id"]
        response = browser.post(prefix + "/api/sessions", json=fields)
        assert response.status_code == 200
        payload = response.json()
        assert payload["ok"]
        assert payload["session_id"] != original
        assert len(payload["sessions"]) == 2
        path = prefix + "/api/session?session_id=" + payload["session_id"]
        if fields.get("prompt"):
            snapshot = wait_snapshot(
                browser, lambda state: state["turns"] and state["turns"][-1]["status"] == "completed", path
            )
            assert snapshot["turns"][0]["prompt"] == fields["prompt"]
            assert [item.text for item in model.prompts[0].input if isinstance(item, UserMessage)] == [
                fields["prompt"]
            ]
        else:
            snapshot = browser.get(path).json()["snapshot"]
            assert snapshot["turns"] == []
            assert model.call_count == 0
        if "title" in fields:
            assert snapshot["title"] == fields["title"]
        assert browser.get(prefix + "/api/session?session_id=" + original).json()["snapshot"]["turns"] == []


@pytest.mark.parametrize("field", ["title", "prompt"])
@pytest.mark.parametrize("value", [None, 7, ["text"]])
def test_new_session_rejects_non_string_fields(field, value):
    with TestClient(create_app(make_session, None)) as browser:
        before = browser.get("/api/sessions").json()["sessions"]
        assert browser.post("/api/sessions", json={field: value}).status_code == 422
        assert browser.get("/api/sessions").json()["sessions"] == before


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("password", [None, "workspace-test-password"])
def test_runtime_info_uses_current_identity_and_local_frontend_api(tmp_path, multi, password):
    model = ScriptedModelClient(
        response_factory=lambda prompt, count: ModelResponse([AssistantMessage("noted")])
    )
    factory = lambda: make_session(model, cwd=tmp_path)
    if multi:
        registry = WorkspaceRegistry([WorkspaceEntry(
            WorkspaceDefinition("alpha", tmp_path / "board", tmp_path),
            WorkspaceSessionManager(factory, None),
        )])
        app = create_multi_workspace_app(registry, password)
        prefix = "/w/alpha"
    else:
        app = create_app(factory, None, password, tmp_path)
        prefix = ""
    app.state.local_api_url = "http://127.0.0.1:12345"
    (tmp_path / "changed").mkdir()
    with TestClient(app) as browser:
        if password:
            assert browser.get(prefix + "/api/session", params={
                "runtime_info": "true",
            }).status_code == 401
            assert browser.post("/login", json={"password": password}).status_code == 200
        own = browser.get(prefix + "/api/session").json()["session_id"]
        # A different default tab must not receive this tab's information.
        other = browser.post(prefix + "/api/sessions").json()["session_id"]
        for prompt in ["/cd changed", "/title runtime test"]:
            assert browser.post(prefix + "/api/session/message", json={
                "session_id": own, "prompt": prompt,
            }).status_code == 200
        result = browser.get(prefix + "/api/session", params={
            "session_id": own, "runtime_info": "true",
        })
        assert result.status_code == 200
        assert "submission_id" not in result.json()
        assert model.call_count == 0
        text = result.json()["runtime_info"]
        assert "Runtime: pycodex / pycodex-ws" in text
        assert "session: runtime test" in text
        assert "API ID: " + own in text
        assert str(tmp_path / "changed") in text
        assert "L=http://127.0.0.1:12345; current workspace=L" + prefix in text
        assert ("GET L/api/workspaces" in text) == multi
        assert "router" not in text.lower()
        assert "/queue From " + own in text
        assert "B/api/session/message" in text
        assert "B/api/session?session_id=" in text
        assert "login" not in text.lower()
        assert "cookie" not in text.lower()
        assert "workspace-test-password" not in text
        assert not text.startswith("/queue")
        sent = browser.post(prefix + "/api/session/message", json={
            "session_id": own, "prompt": text,
        })
        assert sent.status_code == 200
        submission_id = sent.json()["submission_id"]
        snapshot = wait_snapshot(
            browser, lambda s: any(
                turn["submission_id"] == submission_id and turn["response"] == "noted"
                for turn in s["turns"]
            ), prefix + "/api/session?session_id=" + own,
        )
        assert next(t["prompt"] for t in snapshot["turns"] if t["submission_id"] == submission_id) == text
        assert any(isinstance(item, UserMessage) and item.text == text
                   for item in model.prompts[-1].input)
        assert browser.get(prefix + "/api/session?session_id=" + other).json()["snapshot"]["turns"] == []


@pytest.mark.parametrize("connected,api_url", [
    (False, None), (False, "https://router.example/proxy"),
    (True, None), (True, "https://router.example/proxy"),
])
def test_runtime_info_router_state_contains_no_credentials(connected, api_url):
    from pycodex_router.client import RouterClient

    app = create_app(make_session, None)
    client = RouterClient(app, "123456789ABCDEFGH@router.example:6000", "test", 12345, lambda: [])
    client.connected = connected
    client.api_url = api_url
    app.state.router_client = client
    app.state.local_api_url = "http://127.0.0.1:12345"
    with TestClient(app) as browser:
        response = browser.get("/api/session", params={"runtime_info": "true"})
        assert response.status_code == 200
        assert response.json()["snapshot"]["turns"] == []
    text = response.json()["runtime_info"]
    assert ("router" in text.lower()) == connected
    assert ("R=" in text) == bool(connected and api_url)
    if connected and api_url:
        assert "R=" + api_url in text
        assert "GET R/api/workspaces" in text
        assert "device=/n/" + client.node_id in text
    assert "http://127.0.0.1:12345" in text
    assert "123456789ABCDEFGH" not in text
    assert client._identity_key not in text


def test_runtime_info_read_does_not_queue_during_active_turn():
    release = threading.Event()
    started = threading.Event()

    async def respond(prompt, count):
        if count == 1:
            started.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
        return ModelResponse([AssistantMessage("done")])

    model = ScriptedModelClient(response_factory=respond)
    with TestClient(create_app(lambda: make_session(model), None)) as browser:
        try:
            browser.post("/api/session/message", json={"prompt": "active task"})
            assert started.wait(2)
            response = browser.get("/api/session", params={"runtime_info": "true"})
            assert response.status_code == 200
            assert response.json()["runtime_info"]
            assert response.json()["snapshot"]["queued_inputs"] == []
        finally:
            release.set()
        snapshot = wait_snapshot(
            browser, lambda s: s["turns"][-1]["status"] == "completed"
        )
        assert [t["status"] for t in snapshot["turns"]] == ["completed"]
        assert model.call_count == 1


@pytest.mark.parametrize("late_poll", ["none", "success", "error"])
@pytest.mark.parametrize("initial_pane", ["chat", "board"])
def test_closing_active_tab_restores_visited_tab(late_poll, initial_pane):
    source = (
        Path(__file__).parents[1]
        / "workspace_server" / "workspace.html"
    ).read_text(encoding="utf-8")
    fragments = []
    for start, end in [
        ("    let pollTimer = null;", "    const markdownRenderer ="),
        ("    function isNearBottom(", "    function resizePrompt("),
        ("    function stateForSession(", "    function renderTabs("),
        ("    function sessionById(", "    function isSessionRunning("),
        ("    function updateSessionDirectory(", "    function updateContextMeter("),
        ("    function messageSignature(", "    function startPolling("),
        ("    function renderSnapshot(", "    function renderConversation("),
        ("    async function switchSession(", '    form.addEventListener("submit"'),
    ]:
        fragments.append(source[source.index(start) : source.index(end)])
    script = r"""
const assert = require("assert").strict;
// Browser APIs belong to this harness; Node 12 has no global AbortController.
class AbortController {
  constructor() { this.signal = {aborted: false}; }
  abort() { this.signal.aborted = true; }
}
let logVisible = process.argv[3] === "chat";
let scrollTop = 0;
const log = {
  get textContent() { return this.children.map(node => node.textContent).join(""); },
  set textContent(value) {
    this.children = value ? [{
      textContent: value, parentElement: this,
      getBoundingClientRect: () => ({top: 0, bottom: 1000}),
    }] : [];
  },
  replaceChildren(...nodes) { this.children = nodes; },
  get scrollTop() { return logVisible ? scrollTop : 0; },
  set scrollTop(value) {
    if (logVisible) scrollTop = Math.max(0, Math.min(value, this.scrollHeight - this.clientHeight));
  },
  get scrollHeight() { return logVisible ? (this.textContent ? 1000 : 100) : 0; },
  get clientHeight() { return logVisible ? 100 : 0; },
  children: [], getBoundingClientRect: () => ({top: 0}),
};
const prompt = {addEventListener() {}};
class PromptHistory {
  reset() {}
  update() {}
}
const contextBlockDetail = {hidden: true};
const showToast = () => {};
class Composer {
  read() { return this.value; }
  set(value) { this.value = value; }
}
const cwdName = {};
const cwdLabel = {dataset: {defaultCwd: "/a"}, querySelector: () => cwdName};
const canvasTab = {};
const contentTabs = {
  dataset: {boardTitle: "Old tooltip", boardPath: "/board.html"},
  querySelector: () => canvasTab,
};
const document = {
  title: "Workspace",
  getElementById: id => id === "sessionCwd" ? cwdLabel : contentTabs,
};
const requestAnimationFrame = () => {};
const markVisibleFinalRead = () => {};
const workspaceTitle = document.title;
const activeSessionKey = "test-workspace";
const window = {sessionStorage: {getItem: () => null}};
const relativeUrl = path => path;
const resizePrompt = () => {};
const dismissSessionsDrawer = () => {};
const renderTabs = () => {};
const renderQueueHint = () => {};
const updateContextMeter = () => {};
const updatePromptPlaceholder = () => {};
const updateSpinnerNotifyUi = () => {};
const notifySessionDone = () => {};
const setSpinner = text => { spinnerText = text; };
const addEntry = (_role, text) => { log.textContent += text; };
const renderConversation = turns => { log.textContent = turns.map(turn => turn.response).join(""); };
""" + "\n".join(fragments)
    script += r"""
const tabs = [{id: "a", cwd: "/a"}, {id: "b", cwd: "/b"}];
let serverTabs = tabs.slice();
function snapshot(id) {
  return {cwd: "/" + id, turns: [{prompt: `prompt ${id}`, response: `answer ${id}`}],
          spinner: "", queued_inputs: []};
}
function response(id) {
  const payload = {session_id: id, sessions: serverTabs.slice(), snapshot: snapshot(id)};
  return {ok: true, json: async () => payload};
}
let delayOldPoll = false;
let finishOldPoll;
let delayedSignal;
const fetch = async (path, options = {}) => {
  if (options.method === "DELETE") {
    serverTabs = [tabs[0]];
    return {ok: true, json: async () => ({ok: true, sessions: serverTabs.slice()})};
  }
  assert.equal(options.signal.aborted, false);
  const id = path.endsWith("=b") ? "b" : "a";
  const reply = response(id);
  if (delayOldPoll && id === "b") {
    // Deliver a late response even if the browser has already aborted it.
    delayedSignal = options.signal;
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
  updateSessions([{id: "a", running: false}, {id: "b", running: true}]);
  assert.equal(activeSessionId, "a");
  assert.equal(document.title, "● Workspace");
  updateSessions([{id: "a", running: false}, {id: "b", running: true}]);
  assert.equal(document.title, "● Workspace");
  renderSnapshot(snapshot("a"));
  assert.equal(cwdName.textContent, "a");
  assert.equal(cwdLabel.hidden, true);
  renderSnapshot({...snapshot("a"), cwd: "/a/line\nbreak"});
  renderSnapshot({...snapshot("a"), cwd: "/a/$& changed"});
  assert.equal(cwdName.textContent, "$& changed");
  assert.equal(cwdLabel.hidden, false);
  assert.equal(cwdLabel.title, "Current dir: /a/$& changed\nUse /cd <directory> to change directory");
  assert.equal(canvasTab.title, "Workspace: Workspace\nCurrent dir: /a/$& changed\nCanvas file: /board.html");
  if (!logVisible) {
    assert.equal(log.scrollTop, 0);
    assert.equal(stateForSession("a").scrollTop, null);
    renderSnapshot(snapshot("a"));
    logVisible = true;
    restoreSessionScroll();
  }
  assert.equal(log.scrollTop, 900);
  const draftA = {parts: [
    {type: "text", text: "draft a"},
    {type: "context", id: "a-code", label: "sample.py:1", text: "captured code"},
  ], selection: {start: 7, end: 7}};
  composer.set(draftA);
  log.scrollTop = 123;
  saveActiveSessionUiState();
  logVisible = false;
  renderSnapshot({turns: [{prompt: "prompt a", response: "answer a updated"}]});
  assert.equal(stateForSession("a").scrollTop, 123);
  logVisible = true;
  restoreSessionScroll();
  assert.equal(log.scrollTop, 123);
  await switchSession("b");
  assert.equal(log.textContent, "answer b");
  assert.equal(cwdName.textContent, "b");
  assert.equal(cwdLabel.hidden, false);
  assert.equal(document.title, "Workspace");
  composer.set({parts: [{type: "text", text: "draft b"}]});
  log.scrollTop = 9;
  delayOldPoll = process.argv[2] !== "none";
  const oldPoll = delayOldPoll ? pollSession() : null;
  logVisible = process.argv[3] === "chat";
  await closeSession("b");
  if (!logVisible) {
    assert.equal(stateForSession("a").scrollTop, 123);
    logVisible = true;
    restoreSessionScroll();
  }
  assert.equal(activeSessionId, "a");
  assert.equal(log.textContent, "answer a");
  assert.equal(cwdName.textContent, "a");
  assert.equal(cwdLabel.hidden, true);
  assert.deepEqual(composer.read(), draftA);
  assert.equal(log.scrollTop, 123);
  assert.equal(sessionState.has("b"), false);
  assert.equal(document.title, "Workspace");
  if (oldPoll) {
    assert.equal(delayedSignal.aborted, true);
    finishOldPoll();
    await oldPoll;
  }
  assert.equal(activeSessionId, "a");
  assert.deepEqual(sessions.map(tab => tab.id), ["a"]);
  assert.equal(spinnerText, "");
  assert.equal(log.textContent, "answer a");
  await pollSession();
  assert.equal(log.textContent, "answer a");
  updateSessions([{id: "a", running: true}]);
  assert.equal(document.title, "● Workspace");
  updateSessions([]);
  assert.equal(document.title, "Workspace");
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        ["node", "-", late_poll, initial_pane],
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
                    {"id": "first", "work_dir": "first", "state_path": "first/board"},
                    {"work_dir": "second", "state_path": "second/board"},
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


@pytest.mark.parametrize("board", ["saved.board.html", False])
def test_legacy_workspace_config_restores_original_state(tmp_path, board):
    config = tmp_path / "workspaces.json"
    legacy = {"id": "saved", "work_dir": ".", "board": board}
    config.write_text(json.dumps({"workspaces": [legacy]}), encoding="utf-8")
    state_path = tmp_path / ("saved.board" if board else "workspaces.saved")
    saved_file = tmp_path / "saved.txt"
    saved_file.write_text("saved view", encoding="utf-8")
    with TestClient(create_app(make_session, state_path, enable_board=bool(board))) as browser:
        browser.post("/api/session/message", json={"prompt": "saved conversation"})
        original = wait_snapshot(
            browser,
            lambda state: state["turns"] and state["turns"][-1]["response"] == "done",
        )
        opened = browser.post("/api/views/open", json={"href": str(saved_file)}).json()
    rollout = Path(original["rollout_path"])
    original_bytes = rollout.read_bytes()
    definition = load_workspace_definitions(config)[0]
    assert definition.state_path == state_path
    assert definition.enable_board is bool(board)
    assert definition.board_path == (tmp_path / board if board else None)
    with TestClient(create_multi_workspace_app(WorkspaceRegistry([make_entry(definition)]))) as browser:
        restored = browser.get("/w/saved/api/session").json()["snapshot"]
        assert restored["rollout_path"] == original["rollout_path"]
        assert [(turn["prompt"], turn["response"]) for turn in restored["turns"]] == [
            ("saved conversation", "done")
        ]
        assert browser.get("/w/saved/api/views").json()["views"][0]["id"] == opened["view"]["id"]
    assert rollout.read_bytes() == original_bytes
    assert json.loads(config.read_text(encoding="utf-8"))["workspaces"] == [legacy]
    save_workspace_definitions(config, [definition])
    saved = json.loads(config.read_text(encoding="utf-8"))["workspaces"][0]
    assert saved == {
        "id": "saved",
        "work_dir": ".",
        "state_path": state_path.name,
        "enable_board": bool(board),
    }
    assert load_workspace_definitions(config) == [definition]


@pytest.mark.parametrize(
    "fields",
    [
        {"board": "saved.html", "state_path": "saved"},
        {"board": False, "enable_board": False},
        {"board": "saved.htm"},
        {"board": True},
        {"state_path": "saved", "enable_board": "false"},
        {"state_path": ""},
        {"state_path": False},
    ],
)
def test_workspace_config_rejects_invalid_or_ambiguous_paths(tmp_path, fields):
    config = tmp_path / "workspaces.json"
    config.write_text(
        json.dumps({"workspaces": [{"id": "saved", "work_dir": ".", **fields}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_workspace_definitions(config)


@pytest.mark.parametrize("multi", [False, True])
def test_canvas_toggle_preserves_sessions_and_file_views(tmp_path, multi):
    state_path = tmp_path / "saved.workspace"
    board = tmp_path / "saved.workspace.html"
    board.write_text("saved Canvas", encoding="utf-8")
    source = tmp_path / "saved.txt"
    source.write_text("saved file view", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"image")
    prefix = "/w/saved" if multi else ""
    original = None
    for enable_board in (True, False, True):
        if multi:
            definition = WorkspaceDefinition("saved", state_path, tmp_path, enable_board)
            app = create_multi_workspace_app(WorkspaceRegistry([make_entry(definition)]))
        else:
            app = create_app(make_session, state_path, work_dir=tmp_path, enable_board=enable_board)
        with TestClient(app) as browser:
            shell = browser.get(prefix + "/").text
            assert 'data-enable-board="{}"'.format(str(enable_board).lower()) in shell
            assert 'src="{}" title="Canvas"'.format("board" if enable_board else "about:blank") in shell
            for route in ("/board", "/api/board", "/image.png"):
                assert browser.get(prefix + route).status_code == (200 if enable_board else 404)
            if original is None:
                browser.post(prefix + "/api/session/message", json={"prompt": "keep this conversation"})
                original = wait_snapshot(
                    browser,
                    lambda state: state["turns"] and state["turns"][-1]["response"] == "done",
                    prefix + "/api/session",
                )
                opened = browser.post(
                    prefix + "/api/views/open", json={"href": str(source)}
                ).json()["view"]
            else:
                restored = browser.get(prefix + "/api/session").json()["snapshot"]
                assert restored["rollout_path"] == original["rollout_path"]
                assert [(turn["prompt"], turn["response"]) for turn in restored["turns"]] == [
                    ("keep this conversation", "done")
                ]
                assert browser.get(prefix + "/api/views").json()["views"][0]["id"] == opened["id"]
                view = browser.get(prefix + "/api/views/" + opened["id"]).json()
                assert view["documents"][0]["text"] == "saved file view"
    assert board.read_text(encoding="utf-8") == "saved Canvas"
    assert (tmp_path / "saved.workspace.pycodex-ws.json").is_file()
    assert (tmp_path / "saved.workspace.pycodex-views.json").is_file()


def test_dotted_state_prefixes_keep_independent_histories(tmp_path):
    for name in ("saved.one", "saved.two"):
        with TestClient(create_app(make_session, tmp_path / name, enable_board=False)) as browser:
            browser.post("/api/session/message", json={"prompt": name})
            wait_snapshot(
                browser,
                lambda state: state["turns"] and state["turns"][-1]["response"] == "done",
            )
    for name in ("saved.one", "saved.two"):
        with TestClient(create_app(make_session, tmp_path / name, enable_board=False)) as browser:
            snapshot = browser.get("/api/session").json()["snapshot"]
            assert [(turn["prompt"], turn["response"]) for turn in snapshot["turns"]] == [
                (name, "done")
            ]
    assert not (tmp_path / "saved.pycodex-ws.json").exists()


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
        definition = WorkspaceDefinition("first", board.with_suffix(""), board_dir)
        app = create_multi_workspace_app(WorkspaceRegistry([make_entry(definition)]))
        prefix = "/w/first"
    else:
        app = create_app(make_session, board.with_suffix(""))
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
                    make_entry(WorkspaceDefinition("first", board.with_suffix(""), tmp_path)),
                ]
            ),
            password="test-password",
        )
        prefix = "/w/first"
    else:
        app = create_app(make_session, board.with_suffix(""), password="test-password")
        prefix = ""
    with TestClient(app) as browser:
        target = prefix + "/?session=saved&view=chat"
        response = browser.get(target)
        assert response.status_code == 200
        assert len(response.history) == 1
        assert response.history[0].status_code == 303
        assert "<h1>Log in to pycodex workspace</h1>" in response.text
        assert "<code>http://testserver/</code>" in response.text
        login_url = response.history[0].headers["location"]
        assert browser.get(prefix + "/api/session").status_code == 401
        assert browser.post(
            prefix + "/api/sessions/missing/read", json={"final_id": "final"}
        ).status_code == 401
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


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("password", [None, "test-password"])
def test_workspace_websocket_rejects_cross_origin(tmp_path, multi, password):
    if multi:
        app = create_multi_workspace_app(
            WorkspaceRegistry([
                make_entry(WorkspaceDefinition("first", tmp_path / "state", tmp_path)),
            ]),
            password=password,
        )
        prefix = "/w/first"
    else:
        app = create_app(make_session, None, password=password)
        prefix = ""
    with TestClient(app) as browser:
        if password:
            assert browser.post("/login", json={"password": password}).status_code == 200
        for origin in (
            "https://outside.example",
            "http://testserver:6007",
            "https://testserver",
            "null",
            "",
        ):
            with pytest.raises(WebSocketDisconnect) as rejected:
                with browser.websocket_connect(
                    prefix + "/ws/session", headers={"Origin": origin},
                ):
                    pass
            assert rejected.value.code == 1008
        assert browser.get(prefix + "/api/session").json()["snapshot"]["turns"] == []


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("scheme, host, origin", [
    ("ws", "127.0.0.1:6007", "http://127.0.0.1:6007"),
    ("ws", "LOCALHOST:80", "http://localhost"),
    ("wss", "router.example:443", "https://router.example"),
    ("wss", "[::1]:8443", "https://[::1]:8443"),
    ("ws", "testserver", None),
])
def test_workspace_websocket_accepts_its_origin(tmp_path, multi, scheme, host, origin):
    if multi:
        app = create_multi_workspace_app(WorkspaceRegistry([
            make_entry(WorkspaceDefinition("first", tmp_path / "state", tmp_path)),
        ]))
        prefix = "/w/first"
    else:
        app = create_app(make_session, None)
        prefix = ""
    headers = {"Host": host}
    if origin is not None:
        headers["Origin"] = origin
    with TestClient(app) as browser:
        with browser.websocket_connect(
            scheme + "://testserver" + prefix + "/ws/session?session_id=",
            headers=headers,
        ) as websocket:
            assert websocket.receive_json()["type"] == "hello"
            websocket.send_json({"type": "ping"})
            assert websocket.receive_json()["type"] == "pong"


@pytest.mark.parametrize("multi", [False, True])
def test_workspace_json_body_validation(tmp_path, multi):
    if multi:
        registry = WorkspaceRegistry([
            make_entry(WorkspaceDefinition("first", tmp_path / "state", tmp_path, False)),
        ])
        app = create_multi_workspace_app(registry, password="test-password")
        prefix = "/w/first"
    else:
        app = create_app(make_session, None, password="test-password")
        prefix = ""
    with TestClient(app) as browser:
        invalid_bodies = [
            (b"", "application/json"),
            (b'{"password":"request-body-marker"', "application/json"),
            (b"[]", "application/json"),
            (b"null", "application/json"),
            (b'"text"', "application/json"),
            (b"1", "application/json"),
            (b"{}", "text/plain"),
            (b"{}", None),
        ]
        endpoints = [
            "/login", prefix + "/api/views/open", prefix + "/api/session/message",
            prefix + "/api/sessions/missing/read",
        ]
        if multi:
            endpoints.append("/api/workspaces")
        for endpoint in endpoints:
            for body, media_type in invalid_bodies:
                headers = {} if media_type is None else {"Content-Type": media_type}
                response = browser.post(endpoint, data=body, headers=headers)
                assert response.status_code == 422, (endpoint, body, media_type)
                assert "detail" in response.json()
                assert "request-body-marker" not in response.text
            if endpoint == "/login":
                response = browser.post(
                    "/login",
                    data=b'{"password":"test-password"}',
                    headers={"Content-Type": "application/vendor+json; charset=utf-8"},
                )
                assert response.status_code == 200
        assert browser.get(prefix + "/api/session").json()["snapshot"]["turns"] == []
        assert browser.get(prefix + "/api/views").json() == {"views": []}
        if multi:
            assert len(registry.list_workspaces()) == 1


@pytest.mark.parametrize("multi", [False, True])
def test_login_limits_body_without_limiting_messages(tmp_path, multi):
    if multi:
        app = create_multi_workspace_app(
            WorkspaceRegistry([
                make_entry(WorkspaceDefinition("first", tmp_path / "state", tmp_path)),
            ]),
            password="test-password",
        )
        prefix = "/w/first"
    else:
        app = create_app(make_session, None, password="test-password")
        prefix = ""
    body = b'{"password":"test-password"}'
    at_limit = body + b" " * (4096 - len(body))
    multibyte_body = json.dumps(
        {"password": "test-password", "padding": "é" * 2048}, ensure_ascii=False,
    ).encode("utf-8")
    with TestClient(app) as browser:
        for oversized in (at_limit + b" ", b"[" * 4097, multibyte_body):
            response = browser.post(
                "/login", content=oversized,
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 413
            assert response.headers["connection"] == "close"
            assert "set-cookie" not in response.headers
        assert browser.post(
            "/login", content=at_limit,
            headers={"Content-Type": "application/json"},
        ).status_code == 200
        prompt = "message body " * 1024 + "end"
        response = browser.post(
            prefix + "/api/session/message", json={"prompt": prompt},
        )
        assert response.status_code == 200
        snapshot = wait_snapshot(
            browser,
            lambda state: state["turns"] and state["turns"][-1]["status"] == "completed",
            path=prefix + "/api/session",
        )
        assert snapshot["turns"][-1]["prompt"] == prompt


@pytest.mark.parametrize("multi", [False, True])
def test_workspace_http_errors_remain_json(tmp_path, multi):
    if multi:
        app = create_multi_workspace_app(WorkspaceRegistry([
            make_entry(WorkspaceDefinition("first", tmp_path / "state", tmp_path, False)),
        ]))
        prefix = "/w/first"
    else:
        app = create_app(make_session, None)
        prefix = ""
    with TestClient(app) as browser:
        response = browser.get(
            prefix + "/api/session", params={"session_id": "missing"}
        )
        assert response.status_code == 404
        assert response.json() == {"detail": "session not found"}
        assert browser.post(prefix + "/api/sessions").status_code == 200
        response = browser.delete(prefix + "/api/sessions/missing")
        assert response.status_code == 404
        assert response.json() == {"detail": "session not found"}
        response = browser.options(prefix + "/api/session/message")
        assert response.status_code == 405
        assert response.json() == {"detail": "Method Not Allowed"}
        assert response.headers["allow"] == "POST"
        if multi:
            response = browser.get("/w/missing/api/session")
            assert response.status_code == 404
            assert response.json() == {"detail": "workspace not found"}


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


def test_workspace_registry_crud_and_duplicate_state_path(tmp_path):
    config = tmp_path / "workspaces.json"
    registry = WorkspaceRegistry([], config_path=config, entry_factory=make_entry)
    with TestClient(create_multi_workspace_app(registry)) as browser:
        added = browser.post(
            "/api/workspaces",
            json={
                "name": "alpha",
                "dir": "repos/alpha",
                "state_path": "boards/alpha",
            },
        )
        assert added.status_code == 200
        assert (tmp_path / "repos/alpha").is_dir()
        duplicate = browser.post(
            "/api/workspaces",
            json={
                "name": "beta",
                "dir": "repos/beta",
                "state_path": "boards/alpha",
                "enable_board": False,
            },
        )
        assert duplicate.status_code == 400
        assert "state_path already exists" in duplicate.json()["error"]
        generated = browser.post("/api/workspaces", json={"dir": "repos/generated"})
        assert generated.status_code == 200
        workspace = generated.json()["workspace"]
        assert workspace["id"]
        assert workspace["state_path"].startswith("/tmp/pcws-")
        assert workspace["board_path"] == workspace["state_path"] + ".html"
        assert workspace["enable_board"] is True
        assert (
            "Canvas pending" in browser.get("/w/{}/board".format(workspace["id"])).text
        )
        assert browser.get("/w/alpha/").status_code == 200
        assert browser.delete("/api/workspaces/alpha").status_code == 200
        assert browser.get("/w/alpha/api/session").status_code == 404
    definitions = load_workspace_definitions(config)
    assert len(definitions) == 1
    assert definitions[0].workspace_id == workspace["id"]


@pytest.mark.parametrize("toolset", [["exec_command", "apply_patch"], []])
@pytest.mark.parametrize("state_path", [None, "workspaces.alpha"])
def test_boardless_workspace_preserves_context_and_tools(
    tmp_path, monkeypatch, toolset, state_path
):
    config = tmp_path / "workspaces.json"
    model_config = tmp_path / "model.toml"
    model_config.write_text('model = "test"\n', encoding="utf-8")
    model = ScriptedModelClient([ModelResponse([AssistantMessage("done")])])
    monkeypatch.setattr("workspace_server.app.build_model", lambda **kwargs: model)
    args = build_parser().parse_args(
        [
            "--workspace-config",
            str(config),
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
                "state_path": state_path,
                "enable_board": False,
            },
        )
        assert response.status_code == 200
        assert response.json()["workspace"]["board_path"] == ""
        saved_prefix = response.json()["workspace"]["state_path"]
        browser.post("/w/alpha/api/session/message", json={"prompt": "fix"})
        state = wait_snapshot(
            browser,
            lambda state: state["turns"] and state["turns"][-1]["response"] == "done",
            "/w/alpha/api/session",
        )
    prompt = model.prompts[0]
    assert prompt.base_instructions == "Fix the task."
    assert str(tmp_path / "repo") in repr(prompt.input)
    assert "Current workspace Canvas file:" not in repr(prompt.input)
    assert [tool.name for tool in prompt.tools] == toolset
    assert load_workspace_definitions(config)[0].board_path is None
    store = WorkspaceStateStore(Path(saved_prefix + ".pycodex-ws.json"))
    assert store.load_tabs() == [
        {"title": "fix", "rollout_path": state["rollout_path"], "cwd": str(tmp_path / "repo"), "profile": None}
    ]
    restored_registry = WorkspaceRegistry(
        [
            _build_workspace_entry(definition, args)
            for definition in load_workspace_definitions(config)
        ],
        config_path=config,
    )
    with TestClient(create_multi_workspace_app(restored_registry)) as browser:
        restored = browser.get("/w/alpha/api/session").json()["snapshot"]
        assert restored["title"] == "fix"
        assert [
            (turn["prompt"], turn["response"]) for turn in restored["turns"]
        ] == [("fix", "done")]
    assert len(model.prompts) == 1


@pytest.mark.parametrize("explicit_title", [False, True])
@pytest.mark.parametrize("threaded", [False, True])
def test_sessions_persist_titles_and_restore_history(
    tmp_path, explicit_title, threaded
):
    board = tmp_path / "board.html"
    board.write_text("unchanged board", encoding="utf-8")
    store = WorkspaceStateStore(board.with_suffix(".pycodex-ws.json"))

    def session_factory():
        if threaded:
            return ThreadedWorkspaceInteractiveSession(
                make_session, asyncio.get_running_loop()
            )
        return make_session()

    with TestClient(create_app(session_factory, board.with_suffix(""))) as browser:
        first_id = browser.get("/api/sessions").json()["sessions"][0]["id"]
        browser.post(
            "/api/session/message",
            json={"session_id": first_id, "prompt": "saved prompt"},
        )
        state = wait_snapshot(
            browser,
            lambda item: item["turns"] and item["turns"][-1]["response"] == "done",
        )
        assert store.load_tabs() == [
            {"title": "saved prompt", "rollout_path": state["rollout_path"], "cwd": str(Path.cwd()), "profile": None}
        ]
        if explicit_title:
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
    title = "saved" if explicit_title else "saved prompt"
    assert store.load_tabs() == [
        {"title": title, "rollout_path": state["rollout_path"], "cwd": str(Path.cwd()), "profile": None}
    ]
    with TestClient(create_app(session_factory, board.with_suffix(""))) as browser:
        restored = browser.get("/api/session").json()["snapshot"]
        assert restored["title"] == title
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
    store = WorkspaceStateStore(board.with_suffix(".pycodex-ws.json"))

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

    with TestClient(create_app(session_factory, board.with_suffix(""))) as browser:
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
            {"title": "forked", "rollout_path": str(fork_path), "cwd": str(Path.cwd()), "profile": None}
        ]
    else:
        assert not fork_path.exists()
        assert store.load_tabs() == [
            {"title": "forked", "rollout_path": str(original_path), "fork": True, "cwd": str(Path.cwd()), "profile": None}
        ]

    with TestClient(create_app(session_factory, board.with_suffix(""))) as browser:
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
        {"title": "forked", "rollout_path": str(restored_path), "cwd": str(Path.cwd()), "profile": None}
    ]


@pytest.mark.parametrize("threaded", [False, True])
def test_empty_workspace_tabs_survive_restart(tmp_path, threaded):
    board = tmp_path / "board.html"
    store = WorkspaceStateStore(board.with_suffix(".pycodex-ws.json"))
    empty_tab = {"title": "", "rollout_path": "", "cwd": str(Path.cwd()), "profile": None}

    def session_factory():
        if threaded:
            return ThreadedWorkspaceInteractiveSession(
                make_session, asyncio.get_running_loop()
            )
        return make_session()

    with TestClient(create_app(session_factory, board.with_suffix(""))) as browser:
        assert store.load_tabs() == [empty_tab]
        first_id = browser.get("/api/sessions").json()["sessions"][0]["id"]
        second_id = browser.post("/api/sessions").json()["session_id"]
        browser.post("/api/sessions")
        assert store.load_tabs() == [empty_tab] * 3
        assert browser.delete("/api/sessions/" + second_id).status_code == 200
        assert store.load_tabs() == [empty_tab] * 2
        snapshot = browser.get(
            "/api/session", params={"session_id": first_id}
        ).json()["snapshot"]
        assert not Path(snapshot["rollout_path"]).exists()

    with TestClient(create_app(session_factory, board.with_suffix(""))) as browser:
        tabs = browser.get("/api/sessions").json()["sessions"]
        assert len(tabs) == 2
        assert store.load_tabs() == [empty_tab] * 2
        for tab in tabs:
            snapshot = browser.get(
                "/api/session", params={"session_id": tab["id"]}
            ).json()["snapshot"]
            assert snapshot["title"] == ""
            assert not snapshot["turns"]
            assert not Path(snapshot["rollout_path"]).exists()


def test_titled_empty_workspace_tab_survives_restart(tmp_path):
    board = tmp_path / "board.html"
    store = WorkspaceStateStore(board.with_suffix(".pycodex-ws.json"))
    with TestClient(create_app(make_session, board.with_suffix(""))) as browser:
        response = browser.post("/api/session/message", json={"prompt": "/title empty"})
        assert response.status_code == 200
        path = Path(response.json()["snapshot"]["rollout_path"])
        assert not path.exists()
    assert store.load_tabs() == [{"title": "empty", "rollout_path": "", "cwd": str(Path.cwd()), "profile": None}]

    with TestClient(create_app(make_session, board.with_suffix(""))) as browser:
        state = browser.get("/api/session").json()["snapshot"]
        assert state["title"] == "empty"
        assert not state["turns"]
        browser.post("/api/session/message", json={"prompt": "first prompt"})
        state = wait_snapshot(
            browser,
            lambda item: item["turns"] and item["turns"][-1]["response"] == "done",
        )
    assert store.load_tabs() == [
        {"title": "empty", "rollout_path": state["rollout_path"], "cwd": str(Path.cwd()), "profile": None}
    ]


@pytest.mark.parametrize("threaded", [False, True])
def test_session_directories_are_independent_and_restored(tmp_path, threaded):
    original = Path.cwd()
    target = tmp_path / "other project"
    target.mkdir()
    (tmp_path / "file.txt").write_text("default", encoding="utf-8")
    (target / "file.txt").write_text("changed", encoding="utf-8")
    state_path = tmp_path / "board"
    store = WorkspaceStateStore(tmp_path / "board.pycodex-ws.json")
    # Older state files have no cwd; these sessions inherit the workspace default.
    store.save_tabs([{"title": "saved", "rollout_path": ""}])

    def factory():
        if threaded:
            return ThreadedWorkspaceInteractiveSession(
                lambda: make_session(cwd=tmp_path), asyncio.get_running_loop()
            )
        return make_session(cwd=tmp_path)

    with TestClient(create_app(factory, state_path, work_dir=tmp_path)) as browser:
        first = browser.get("/api/sessions").json()["sessions"][0]
        assert first["cwd"] == str(tmp_path)
        response = browser.post("/api/session/message", json={
            "session_id": first["id"], "prompt": "/cd other project",
        })
        assert response.status_code == 200
        assert response.json()["snapshot"]["cwd"] == str(target)
        second_id = browser.post("/api/sessions").json()["session_id"]
        assert [s["cwd"] for s in browser.get("/api/sessions").json()["sessions"]] == [
            str(target), str(tmp_path),
        ]
        views = []
        for identity, text in [(first["id"], "changed"), (second_id, "default")]:
            response = browser.post("/api/views/open", json={
                "href": "file.txt", "session_id": identity,
            })
            assert response.status_code == 200
            view = response.json()["view"]
            assert view["documents"][0]["text"] == text
            views.append(view["id"])
        assert views[0] != views[1]
        assert browser.post("/api/views/open", json={
            "href": "file.txt", "session_id": "missing",
        }).status_code == 404
        assert browser.post("/api/views/open", json={
            "href": "file.txt", "session_id": [],
        }).status_code == 400
        # Keep one empty session and one recorded session across restart.
        browser.post("/api/session/message", json={
            "session_id": second_id, "prompt": "hello",
        })
        wait_snapshot(browser, lambda s: s["turns"] and s["turns"][-1]["response"] == "done",
                      "/api/session?session_id=" + second_id)
    assert [tab["cwd"] for tab in store.load_tabs()] == [str(target), str(tmp_path)]
    assert store.load_tabs()[0]["rollout_path"] == ""
    assert store.load_tabs()[1]["rollout_path"]
    with TestClient(create_app(factory, state_path, work_dir=tmp_path)) as browser:
        sessions = browser.get("/api/sessions").json()["sessions"]
        assert [tab["cwd"] for tab in sessions] == [str(target), str(tmp_path)]
        for tab in sessions:
            snapshot = browser.get("/api/session", params={"session_id": tab["id"]}).json()["snapshot"]
            assert snapshot["cwd"] == tab["cwd"]
    assert Path.cwd() == original


@pytest.mark.parametrize("command", ["/title saved prompt", "/resume 1"])
def test_workspace_explicit_save_keeps_unchanged_state(tmp_path, command):
    board = tmp_path / "board.html"
    store = WorkspaceStateStore(board.with_suffix(".pycodex-ws.json"))
    with TestClient(create_app(make_session, board.with_suffix(""))) as browser:
        browser.post("/api/session/message", json={"prompt": "saved prompt"})
        state = wait_snapshot(
            browser, lambda s: s["turns"] and s["turns"][-1]["response"] == "done"
        )
        assert store.load_tabs() == [
            {"title": "saved prompt", "rollout_path": state["rollout_path"], "cwd": str(Path.cwd()), "profile": None}
        ]
        response = browser.post("/api/session/message", json={"prompt": command})
        assert response.status_code == 200
    assert store.load_tabs() == [
        {"title": "saved prompt", "rollout_path": state["rollout_path"], "cwd": str(Path.cwd()), "profile": None}
    ]


@pytest.mark.parametrize("next_input", ["/compact", "failed turn"])
def test_workspace_fork_saves_new_target_after_compact_or_failure(tmp_path, next_input):
    board = tmp_path / "board.html"
    store = WorkspaceStateStore(board.with_suffix(".pycodex-ws.json"))

    def respond(prompt, count):
        if count == 2 and next_input == "failed turn":
            raise RuntimeError("offline model failure")
        return ModelResponse([AssistantMessage("done")])

    def session_factory():
        return make_session(ScriptedModelClient(response_factory=respond))

    with TestClient(create_app(session_factory, board.with_suffix(""))) as browser:
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
    assert store.load_tabs() == [{"title": "saved", "rollout_path": str(fork_path), "cwd": str(Path.cwd()), "profile": None}]
    assert original_path.read_bytes() == original_bytes
    with TestClient(create_app(session_factory, board.with_suffix(""))) as browser:
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
    manager = WorkspaceSessionManager(lambda: session, board.with_suffix(""))
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
    assert WorkspaceStateStore(board.with_suffix(".pycodex-ws.json")).load_tabs() == [
        {"title": "saved", "rollout_path": str(fork_path), "cwd": str(Path.cwd()), "profile": None}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "contents",
    ["", "{}", '{"tabs": {}}', '{"tabs": [{"title": "saved"}]}'],
)
async def test_invalid_workspace_state_fails_without_replacing_history(tmp_path, contents):
    path = tmp_path / "saved.pycodex-ws.json"
    path.write_text(contents, encoding="utf-8")
    manager = WorkspaceSessionManager(make_session, tmp_path / "saved")
    with pytest.raises((ValueError, KeyError)):
        await manager.start()
    assert not manager.list_sessions()
    assert path.read_text(encoding="utf-8") == contents


@pytest.mark.asyncio
async def test_workspace_missing_saved_directory_fails_without_rewriting_state(tmp_path):
    session = make_session(cwd=tmp_path)
    store = WorkspaceStateStore(tmp_path / "saved.pycodex-ws.json")
    tabs = [{"title": "", "rollout_path": "", "cwd": str(tmp_path / "missing")}]
    store.save_tabs(tabs)
    manager = WorkspaceSessionManager(lambda: session, tmp_path / "saved")
    with pytest.raises(FileNotFoundError):
        await manager.start()
    assert session.runtime.agent.is_shutdown
    assert not manager.list_sessions()
    assert store.load_tabs() == tabs


@pytest.mark.asyncio
async def test_workspace_missing_restore_path_still_fails_and_closes_session(tmp_path):
    session = make_session()
    board = tmp_path / "board.html"
    missing_path = str(tmp_path / "missing-rollout.jsonl")
    WorkspaceStateStore(board.with_suffix(".pycodex-ws.json")).save_tabs(
        [{"title": "saved", "rollout_path": missing_path, "fork": True}]
    )
    manager = WorkspaceSessionManager(lambda: session, board.with_suffix(""))
    with pytest.raises(FileNotFoundError):
        await manager.start()
    assert session.runtime.agent.is_shutdown
    assert not manager.list_sessions()


@pytest.mark.asyncio
async def test_workspace_failed_restore_keeps_the_full_saved_tab_list(
    tmp_path, monkeypatch
):
    session = await make_session().start()
    try:
        receipt = await session.runtime.submit_input("saved prompt")
        await receipt.future
        rollout_path = session.runtime.agent.session_file_path
    finally:
        await session.close()
    monkeypatch.chdir(rollout_path.parent)
    board = tmp_path / "board.html"
    store = WorkspaceStateStore(board.with_suffix(".pycodex-ws.json"))
    tabs = [
        {"title": "", "rollout_path": ""},
        {"title": "saved", "rollout_path": rollout_path.name},
        {"title": "missing", "rollout_path": str(tmp_path / "missing.jsonl")},
    ]
    store.save_tabs(tabs)
    manager = WorkspaceSessionManager(make_session, board.with_suffix(""))
    try:
        with pytest.raises(FileNotFoundError):
            await manager.start()
        assert store.load_tabs() == tabs
    finally:
        await manager.close()


def test_workspace_does_not_save_a_fork_destination_owned_by_another_writer(tmp_path):
    board = tmp_path / "board.html"
    with TestClient(create_app(make_session, board.with_suffix(""))) as browser:
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
    assert WorkspaceStateStore(board.with_suffix(".pycodex-ws.json")).load_tabs() == [
        {"title": "saved", "rollout_path": original_path, "fork": True, "cwd": str(Path.cwd()), "profile": None}
    ]


@pytest.mark.parametrize("old_turns,pretty", [(1, False), (85, True)])
def test_workspace_resume_hides_compact_handoff_and_keeps_real_reply(
    tmp_path, old_turns, pretty
):
    prompts = ["original prompt"] + [
        "old " + str(index) for index in range(old_turns - 1)
    ]
    model = ScriptedModelClient(
        [ModelResponse([AssistantMessage("old answer")]) for _ in prompts]
        + [
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
    for prompt in prompts:
        source.ask(prompt)
    asyncio.run(source.compact())
    asyncio.run(source.run_turn([]))
    source.shutdown()
    if pretty:
        path = source.session_file_path
        entries = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        path.write_text(
            "\n".join(json.dumps(entry, indent=2) for entry in entries)
            + '\n{"type":"unfinished"',
            encoding="utf-8",
        )

    expected = [(prompt, "old answer") for prompt in prompts] + [
        ("", ""),
        ("", "continued answer"),
    ]
    next_model = ScriptedModelClient([ModelResponse([AssistantMessage("done")])])
    with TestClient(create_app(lambda: make_session(next_model), None)) as browser:
        response = browser.post("/api/session/message", json={"prompt": "/resume 1"})
        assert response.status_code == 200
        restored = browser.get("/api/session").json()["snapshot"]
        assert restored["title"] == "original prompt"
        assert [
            (turn["prompt"], turn["response"]) for turn in restored["turns"]
        ] == expected[-80:]
        assert restored["turns"][-2]["kind"] == "compaction"
        assert (
            browser.post(
                "/api/session/message", json={"prompt": "next prompt"}
            ).status_code
            == 200
        )
        state = wait_snapshot(
            browser, lambda item: item["turns"][-1]["response"] == "done"
        )
        assert [
            (turn["prompt"], turn["response"]) for turn in state["turns"]
        ] == (expected + [("next prompt", "done")])[-80:]
        assert [
            item.text
            for item in next_model.prompts[-1].input
            if isinstance(item, UserMessage)
        ] == ["next prompt"]


@pytest.mark.asyncio
async def test_manual_compact_keeps_existing_web_conversation():
    session = await make_session(
        ScriptedModelClient(
            [
                ModelResponse([AssistantMessage("answer")]),
                ModelResponse([AssistantMessage("hidden summary")]),
            ]
        )
    ).start()
    try:
        receipt = await session.runtime.submit_input("question")
        await receipt.future
        receipt = await session.runtime.submit_input("/compact")
        await receipt.future
        assert [
            (turn["kind"], turn["prompt"], turn["response"])
            for turn in session.snapshot()["turns"]
        ] == [("assistant", "question", "answer"), ("compaction", "", "")]
    finally:
        await session.close()


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


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_unread_is_shared_and_old_read_cannot_clear_new_final(tmp_path, multi, failure):
    def respond(prompt, count):
        if failure:
            raise RuntimeError("same error")
        return ModelResponse([AssistantMessage("same reply")])

    model = ScriptedModelClient(response_factory=respond)
    manager = WorkspaceSessionManager(lambda: make_session(model), tmp_path / "state")
    if multi:
        app = create_multi_workspace_app(WorkspaceRegistry([
            WorkspaceEntry(WorkspaceDefinition("first", tmp_path / "state", tmp_path), manager),
        ]))
        prefix = "/w/first"
    else:
        app = create_app(manager, tmp_path / "state")
        prefix = ""
    with TestClient(app) as desktop:
        # A second client shares the running app without starting its lifespan again.
        phone = TestClient(app)
        try:
            assert not desktop.get(prefix + "/api/sessions").json()["sessions"][0]["unread"]
            desktop.post(prefix + "/api/session/message", json={"prompt": "first"})
            first = wait_snapshot(desktop, lambda s: s["last_final_id"], prefix + "/api/session")
            session_id = desktop.get(prefix + "/api/session").json()["session_id"]
            read_path = prefix + "/api/sessions/" + session_id + "/read"
            first_id = first["last_final_id"]
            assert first["turns"][-1]["final_id"] == first_id
            assert phone.get(prefix + "/api/sessions").json()["sessions"][0]["unread"]
            assert not phone.post(read_path, json={"final_id": first_id}).json()["sessions"][0]["unread"]
            assert not desktop.get(prefix + "/api/sessions").json()["sessions"][0]["unread"]

            desktop.post(prefix + "/api/session/message", json={"prompt": "second"})
            second = wait_snapshot(
                desktop, lambda s: s["last_final_id"] != first_id, prefix + "/api/session"
            )
            second_id = second["last_final_id"]
            assert second["turns"][-1]["response"] == first["turns"][-1]["response"]
            if failure:
                assert second["turns"][-1]["error"] == first["turns"][-1]["error"]
                assert "same error" in second["turns"][-1]["error"]
            assert second_id != first_id
            state_path = tmp_path / "state.pycodex-ws.json"
            saved = state_path.read_bytes()
            stamp = state_path.stat().st_mtime_ns
            assert phone.post(read_path, json={"final_id": first_id}).json()["sessions"][0]["unread"]
            assert not phone.post(read_path, json={"final_id": second_id}).json()["sessions"][0]["unread"]
            phone.post(read_path, json={"final_id": second_id})
            assert state_path.read_bytes() == saved
            assert state_path.stat().st_mtime_ns == stamp
            assert not desktop.get(prefix + "/api/sessions").json()["sessions"][0]["unread"]
            if multi:
                assert not desktop.get("/api/workspaces").json()["workspaces"][0]["sessions"][0]["unread"]
            for value in (None, "", 1, [], "x" * 129):
                assert phone.post(read_path, json={"final_id": value}).status_code == 400
            assert phone.post(
                prefix + "/api/sessions/missing/read", json={"final_id": second_id}
            ).status_code == 404
            assert model.call_count == 2
        finally:
            phone.close()


@pytest.mark.parametrize("threaded", [False, True])
@pytest.mark.parametrize("read", [False, True])
def test_unread_survives_compact_but_clears_on_restart(tmp_path, threaded, read):
    def factory():
        if threaded:
            return ThreadedWorkspaceInteractiveSession(make_session, asyncio.get_running_loop())
        return make_session()

    state_path = tmp_path / "state"
    store = WorkspaceStateStore(tmp_path / "state.pycodex-ws.json")
    with TestClient(create_app(factory, state_path)) as browser:
        browser.post("/api/session/message", json={"prompt": "question"})
        state = wait_snapshot(browser, lambda s: s["last_final_id"])
        final_id = state["last_final_id"]
        session_id = browser.get("/api/session").json()["session_id"]
        if read:
            browser.post("/api/sessions/" + session_id + "/read", json={"final_id": final_id})
        browser.post("/api/session/message", json={"prompt": "/compact"})
        state = wait_snapshot(browser, lambda s: s["turns"][-1]["kind"] == "compaction")
        assert state["last_final_id"] == final_id
        assert browser.get("/api/sessions").json()["sessions"][0]["unread"] is (not read)
        assert store.load_tabs() == [
            {"title": "question", "rollout_path": state["rollout_path"], "cwd": str(Path.cwd()), "profile": None}
        ]
    with TestClient(create_app(factory, state_path)) as browser:
        payload = browser.get("/api/session").json()
        assert payload["session_id"] != session_id
        assert payload["snapshot"]["last_final_id"] == ""
        assert all(not turn["final_id"] for turn in payload["snapshot"]["turns"])
        assert not payload["sessions"][0]["unread"]
        browser.post("/api/session/message", json={"prompt": "another question"})
        state = wait_snapshot(browser, lambda s: s["last_final_id"])
        assert state["last_final_id"] != final_id
        assert browser.get("/api/sessions").json()["sessions"][0]["unread"]


def test_failed_partial_response_creates_unread(tmp_path):
    def respond(prompt, count):
        if count == 2:
            raise ResponsesIncompleteError(
                "incomplete", [AssistantMessage("partial")], reason="max_output_tokens"
            )
        return ModelResponse([AssistantMessage("final reply")])

    state_path = tmp_path / "state"
    with TestClient(create_app(
        lambda: make_session(ScriptedModelClient(response_factory=respond)), state_path
    )) as browser:
        browser.post("/api/session/message", json={"prompt": "question"})
        first = wait_snapshot(browser, lambda s: s["last_final_id"])
        session_id = browser.get("/api/session").json()["session_id"]
        browser.post(
            "/api/sessions/" + session_id + "/read",
            json={"final_id": first["last_final_id"]},
        )
        browser.post("/api/session/message", json={"prompt": "unfinished"})
        state = wait_snapshot(browser, lambda s: s["turns"][-1]["status"] == "error")
        assert state["last_final_id"] != first["last_final_id"]
        assert state["turns"][-1]["final_id"] == state["last_final_id"]
        assert browser.get("/api/sessions").json()["sessions"][0]["unread"]


@pytest.mark.parametrize("failure", [False, True])
def test_final_or_failure_creates_unread_independently_of_running(failure):
    session = make_session()
    manager = WorkspaceSessionManager(lambda: session)
    view = session.view
    with TestClient(create_app(manager, None)) as browser:
        view.handle_event(TurnStartedEvent("first", ("question",)))
        view.handle_event(AssistantDeltaEvent("commentary", "first"))
        assert not manager.list_sessions()[0]["unread"]
        view.handle_event(TurnCompletedEvent("first", 1, "final", 1))
        assert manager.list_sessions()[0]["running"]
        assert manager.list_sessions()[0]["unread"]
        session_id = manager.list_sessions()[0]["id"]
        browser.post("/api/sessions/" + session_id + "/read", json={"final_id": "first"})
        assert manager.list_sessions()[0]["running"]
        for event in (
            TurnStartedEvent("next", ("next question",)),
            AssistantDeltaEvent("thinking", "next"),
            ToolStartedEvent("next", ToolCall("call", "echo", {})),
            ToolCompletedEvent(
                "next", ToolCall("call", "echo", {}), ToolResult("call", "echo", "tool result")
            ),
            TurnInterruptedEvent("next", 1, "partial", 1),
            TurnCompletedEvent("next", 1, None, 0),
            CompactCompletedEvent("compact", "manual", None, None, 10, 1, 0, 0),
        ):
            view.handle_event(event)
            assert not manager.list_sessions()[0]["unread"]
            assert manager.list_sessions()[0]["last_final_id"] == "first"
        view.handle_event(TurnStartedEvent("last", ("last question",)))
        view.handle_event(
            TurnFailedEvent("last", 1, "failed", "RuntimeError", 0)
            if failure
            else TurnCompletedEvent("last", 1, "final", 0)
        )
        assert manager.list_sessions()[0]["unread"]
        assert manager.list_sessions()[0]["last_final_id"] == "last"
        assert not manager.list_sessions()[0]["running"]


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
    view.load_session_history("old", (("question", "answer"),))
    view.handle_event(TokenCountEvent({"total_tokens": 80000}, "turn"))
    view.handle_event(completed)
    assert view.snapshot()["turns"][0]["response"] == "answer"
    assert view.snapshot()["turns"][-1]["kind"] == "compaction"
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
