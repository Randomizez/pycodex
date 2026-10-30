import asyncio
import threading
import time

import pytest
from starlette.testclient import TestClient

from pycodex import Agent, AgentRuntime, AssistantMessage, ContextConfig, ModelResponse, ToolRegistry, UserMessage
from pycodex.events import AssistantDeltaEvent
from pycodex.tools import ClockManager, ClockTool
from tests.fakes import ScriptedModelClient
from workspace_server import (
    ThreadedWorkspaceInteractiveSession, WebSessionView, WorkspaceDefinition,
    WorkspaceEntry, WorkspaceRegistry, WorkspaceSessionManager,
    create_app, create_multi_workspace_app,
)
from test_workspace_server import make_session


class ModelWithCatalog(ScriptedModelClient):
    async def list_models(self):
        return ["scripted-model", "another-model"]


@pytest.fixture(params=[(False, False), (False, True), (True, False), (True, True)])
def api_setup(request, tmp_path):
    multi, threaded = request.param

    def build(configure=None, password=None):
        def make():
            model = ModelWithCatalog(
                response_factory=lambda prompt, count: ModelResponse([AssistantMessage("done")]),
                model="scripted-model",
            )
            session = make_session(model=model, cwd=tmp_path)
            if configure is not None:
                configure(session)
            return session

        def factory():
            if threaded:
                return ThreadedWorkspaceInteractiveSession(make, asyncio.get_running_loop())
            return make()

        manager = WorkspaceSessionManager(factory, tmp_path / "state")
        if multi:
            definition = WorkspaceDefinition("test", tmp_path / "state", tmp_path, False)
            app = create_multi_workspace_app(
                WorkspaceRegistry([WorkspaceEntry(definition, manager)]), password
            )
        else:
            app = create_app(manager, tmp_path / "state", password, tmp_path, False)
        return app, "/w/test" if multi else ""

    return build


def wait_submission(client, base, submission_id):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get(base + "/submissions/" + submission_id)
        assert response.status_code == 200
        record = response.json()["submission"]
        if record["status"] in {"completed", "failed"}:
            return record
        time.sleep(0.005)
    raise AssertionError("submission did not finish")


def session_base(client, prefix):
    session_id = client.get(prefix + "/api/session").json()["session_id"]
    return prefix + "/api/sessions/" + session_id


def test_state_and_models_are_read_only_and_match_websocket(api_setup):
    app, prefix = api_setup()
    with TestClient(app) as client:
        base = session_base(client, prefix)
        payload = client.get(base + "/state").json()
        state = payload["state"]
        assert payload["api_version"] == 1
        assert state["model"] == {"name": "scripted-model", "reasoning_effort": None}
        assert state["plan"] == {"explanation": None, "plan": []}
        assert state["queued_inputs"] == []
        assert state["busy"] is False
        assert "history" not in state
        assert client.get(base + "/models").json()["models"] == ["scripted-model", "another-model"]
        with client.websocket_connect(prefix + "/ws/session?cursor=" + payload["cursor"]) as ws:
            hello = ws.receive_json()
            assert hello["state"] == state
            assert hello["cursor"] == payload["cursor"]
            assert hello["events"] == []
            assert hello["reset_required"] is False
        assert client.get(prefix + "/api/session").json()["snapshot"]["turns"] == []


def test_command_receipt_remains_queryable_while_execution_is_blocked(api_setup):
    entered = threading.Event()
    release = threading.Event()

    def configure(session):
        async def blocked(argument):
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.001)
            return {"kind": "blocked", "argument": argument, "lines": ["done"]}
        session.runtime.register_command("blocked", blocked)

    app, prefix = api_setup(configure)
    with TestClient(app) as client:
        try:
            base = session_base(client, prefix)
            response = client.post(base + "/commands", json={"command": "blocked", "argument": "value"})
            assert response.status_code == 202
            submission_id = response.json()["submission_id"]
            assert entered.wait(2)
            record = client.get(base + "/submissions/" + submission_id).json()["submission"]
            assert record["status"] == "running"
            second = client.post(base + "/commands", json={"command": "title", "argument": "changed"})
            assert second.status_code == 202
            assert client.get(base + "/state").status_code == 200
            release.set()
            result = wait_submission(client, base, submission_id)
            assert result["result"] == {"kind": "blocked", "argument": "value", "lines": ["done"]}
            assert wait_submission(client, base, second.json()["submission_id"])["status"] == "completed"
            assert client.get(base + "/state").json()["state"]["title"] == "changed"
        finally:
            release.set()


def test_commands_validate_and_expose_failures(api_setup):
    app, prefix = api_setup()
    with TestClient(app) as client:
        base = session_base(client, prefix)
        for payload in (
            {}, {"command": "queue"}, {"command": "missing"},
            {"command": "title", "argument": {}},
            {"command": "title", "extra": True},
        ):
            assert client.post(base + "/commands", json=payload).status_code == 422
        response = client.post(base + "/commands", json={"command": "compact", "argument": "invalid"})
        result = wait_submission(client, base, response.json()["submission_id"])
        assert result["status"] == "failed"
        assert result["error"]["type"] == "ValueError"
        assert client.get(base + "/submissions/missing").status_code == 404
        assert client.get(base + "/state?agent_id=missing").status_code == 404


def test_history_pages_survive_append_and_reject_a_different_session(api_setup):
    app, prefix = api_setup()
    with TestClient(app) as client:
        base = session_base(client, prefix)
        for prompt in ("first", "second", "third"):
            result = client.post(prefix + "/api/session/message", json={"prompt": prompt}).json()
            assert wait_submission(client, base, result["submission_id"])["result"]["output_text"] == "done"
        latest = client.get(base + "/history?limit=2").json()
        assert latest["source"] == "rollout"
        assert len(latest["items"]) == 2
        assert latest["items"][0]["payload"]["content"][0]["text"] == "third"
        result = client.post(prefix + "/api/session/message", json={"prompt": "fourth"}).json()
        wait_submission(client, base, result["submission_id"])
        older = client.get(base + "/history", params={"limit": 2, "before": latest["next_cursor"]}).json()
        assert older["items"][0]["payload"]["content"][0]["text"] == "second"
        earliest = client.get(base + "/history", params={"limit": 2, "before": older["next_cursor"]}).json()
        assert earliest["items"][0]["payload"]["content"][0]["text"] == "first"
        assert earliest["next_cursor"] is None
        for query in ("limit=0", "limit=201", "limit=no", "before=invalid"):
            assert client.get(base + "/history?" + query).status_code == 422
        other = client.post(prefix + "/api/sessions").json()["session_id"]
        assert client.get(
            prefix + "/api/sessions/" + other + "/history",
            params={"before": latest["next_cursor"]},
        ).status_code == 409


@pytest.mark.asyncio
async def test_event_cursor_replay_expiry_and_restart():
    view = WebSessionView()
    cursor = view.api_snapshot()["cursor"]
    for index in range(505):
        view.handle_event(AssistantDeltaEvent(str(index)))
    expired = view.subscribe(cursor).get_nowait()
    assert expired["reset_required"] is True
    assert expired["events"] == []
    recent = view.subscribe().get_nowait()["events"][-3:]
    replay = view.subscribe(recent[0]["cursor"]).get_nowait()
    assert [event["cursor"] for event in replay["events"]] == [
        event["cursor"] for event in recent[1:]
    ]
    assert WebSessionView().subscribe(replay["cursor"]).get_nowait()["reset_required"]
    with pytest.raises(ValueError, match="cursor"):
        view.subscribe("bad")


def test_active_history_cursor_detects_compaction():
    from workspace_server.session_api import history_page
    from starlette.exceptions import HTTPException

    history = tuple(UserMessage(str(index)) for index in range(5))
    page = history_page("memory-session", None, history, None, 2)
    assert page["next_cursor"]
    appended = history + (AssistantMessage("new answer"),)
    older = history_page("memory-session", None, appended, page["next_cursor"], 2)
    assert [item["index"] for item in older["items"]] == [1, 2]
    with pytest.raises(HTTPException) as caught:
        history_page("memory-session", None, (UserMessage("summary"),), page["next_cursor"], 2)
    assert caught.value.status_code == 409


@pytest.mark.asyncio
async def test_child_queries_and_subscriptions_do_not_enable_interactive_requests(tmp_path):
    session = make_session(cwd=tmp_path)
    children = session.runtime.agent.tool_registry.runtime_environment.subagent_manager

    def builder(model, effort, history, agent_id):
        return AgentRuntime(Agent(
            ScriptedModelClient([ModelResponse([AssistantMessage("child done")])]),
            ToolRegistry(), ContextConfig(cwd=tmp_path),
            initial_history=history, session_id=agent_id,
        ))

    children.set_runtime_builder(builder)
    await session.start()
    try:
        child_id = (await children.spawn_agent(None, None, None, False, None, None, ()))["agent_id"]
        listed = await session.api_query("agents", {})
        assert listed["agents"][0]["agent_id"] == child_id
        assert listed["agents"][0]["parent_agent_id"] == session.runtime.agent.session_id
        state = await session.api_query("state", {"agent_id": child_id})
        assert state["state"]["agent_session_id"] == child_id
        child = children.get_runtime(child_id)
        assert await child._request_permissions({"reason": "read-only observer"}) is None
        view = await session.event_view(child_id)
        queue = view.subscribe(state["cursor"])
        queue.get_nowait()
        receipt = await child.submit_input("work")
        await receipt.future
        await asyncio.sleep(0)
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        assert any(event.get("kind") == "assistant_delta" for event in events)
        assert (await session.api_query("history", {"agent_id": child_id, "limit": 10}))["source"] == "active_history"
        assert child.agent.session_file_path is None
        child.fork()
        children_state = await session.api_query("state", {"agent_id": child_id})
        assert children_state["state"]["agent_session_id"] != child_id
        assert (await session.api_query("agents", {}))["agents"][0]["agent_id"] == child_id
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_clock_inventory_exposes_deadline_without_changing_tool_output(tmp_path):
    clock = ClockManager()
    tools = ToolRegistry()
    tools.register(ClockTool(clock))
    session = make_session(tools=tools, cwd=tmp_path)
    await session.start()
    try:
        assert clock.set_period(5) == {"enabled": True, "period_m": 5.0}
        first = await session.api_query("background-work", {})
        assert first["background_work"][0]["status"] == "paused"
        assert session.runtime.state_snapshot()["background_work_count"] == 0
        clock.arm_after_reply()
        armed = (await session.api_query("background-work", {}))["background_work"][0]
        assert armed["status"] == "armed"
        assert armed["next_fire_at"]
        assert session.runtime.state_snapshot()["background_work_count"] == 1
        clock.turn_started()
        assert (await session.api_query("background-work", {}))["background_work"][0]["next_fire_at"] is None
        assert session.runtime.state_snapshot()["background_work_count"] == 0
    finally:
        await session.close()


def test_resource_api_uses_existing_authentication(api_setup):
    app, prefix = api_setup(password="test-password")
    with TestClient(app) as client:
        for suffix in ("state", "models", "history", "agents", "background-work", "submissions/id"):
            assert client.get(prefix + "/api/sessions/id/" + suffix, follow_redirects=False).status_code == 401
        assert client.post(prefix + "/api/sessions/id/commands", json={"command": "title"}).status_code == 401
