import asyncio
import copy
import time

import pytest
from starlette.testclient import TestClient

from pycodex import AgentsManager, AssistantMessage, ContextConfig, ModelResponse, ToolCall
from pycodex.events import AssistantDeltaEvent
from pycodex.tools import ClockTool
from tests.fakes import ScriptedModelClient, wait_for_runtime_idle
from workspace_server import (
    ThreadedWorkspaceInteractiveSession, WorkspaceInteractiveSession,
    WorkspaceSessionManager, create_app,
)


def context(tmp_path):
    return ContextConfig(cwd=tmp_path, codex_home=tmp_path)


async def eventually(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)
    await asyncio.wait_for(wait(), 2)


@pytest.mark.asyncio
async def test_capacity_removes_agent_view_and_closes_subscriptions(tmp_path):
    manager = AgentsManager(
        ScriptedModelClient([ModelResponse([AssistantMessage("retained result")])]),
        ContextConfig(cwd=tmp_path, codex_home=tmp_path, multi_agent_max_concurrency=2),
        toolset=(),
    )
    root = manager.create_root_runtime()
    session = await WorkspaceInteractiveSession(root).start()
    try:
        async def spawn(name):
            path = await manager.spawn_agent(
                root.agent.agent_id, name, "task", fork_turns="none",
            )
            runtime = manager.get_runtime(manager._resolve_agent_id(root.agent.agent_id, path))
            await wait_for_runtime_idle(runtime)
            return runtime

        first = await spawn("first")
        first_view = await session.event_view(first.agent.agent_id)
        subscriber = first_view.subscribe()
        assert (await subscriber.get())["type"] == "hello"
        second = await spawn("second")
        for _ in range(2):
            rows = (await session.api_query("agents", {}))["agents"]
            assert [(row["agent_path"], row["status"]) for row in rows] == [
                ("/root/second", {"completed": "retained result"}),
            ]
            assert rows[0]["last_message"] == "retained result"
        for resource in ("state", "models", "history", "agents", "background-work"):
            with pytest.raises(KeyError):
                await session.api_query(resource, {"agent_id": first.agent.agent_id})
        with pytest.raises(KeyError):
            await session.event_view(first.agent.agent_id)
        assert set(session._child_views) == {second.agent.agent_id}
        assert not first._frontends
        assert not first_view._subscribers
        while await asyncio.wait_for(subscriber.get(), 2) is not None:
            pass
        assert first.agent.model_client.call_count == 1
        assert first.agent.is_shutdown and first._worker.done()
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_agent_tree_tracks_descendants_and_assistant_text_without_interactive_access(tmp_path, monkeypatch):
    clients = []
    tools = {}

    class TreeModel(ScriptedModelClient):
        def __init__(self):
            self.index = len(clients)
            self.finish = asyncio.Event()
            clients.append(self)
            super().__init__(response_factory=self.respond)

        def with_overrides(self, model, effort, session_id, openai_subagent):
            return TreeModel()

        async def respond(self, prompt, count):
            if self.index == 3:
                raise ValueError("fixture failure")
            if count == 1:
                text = [] if self.index == 4 else [AssistantMessage("Preparing " + str(self.index))]
                return ModelResponse(text + [ToolCall("clock", "clock", {"period_m": None})])
            await self.finish.wait()
            return ModelResponse([AssistantMessage("Result " + str(self.index))])

    async def held_clock(self, context, args):
        release = asyncio.Event()
        tools[context.agent_id] = release
        await release.wait()
        return {"enabled": False, "period_m": None}

    monkeypatch.setattr(ClockTool, "run", held_clock)
    manager = AgentsManager(TreeModel(), context(tmp_path), toolset=("clock",))
    root = manager.create_root_runtime()
    session = await WorkspaceInteractiveSession(root).start()
    runtimes = [root]

    async def spawn(parent, name):
        path = await manager.spawn_agent(
            parent.agent.agent_id, name, "task", fork_turns="none",
        )
        child = manager.get_runtime(manager._resolve_agent_id(root.agent.agent_id, path))
        runtimes.append(child)
        return child

    try:
        initial = await session.api_query("agents", {})
        assert initial["root"]["agent_id"] == root.agent.agent_id
        assert initial["agents"] == []
        child = await spawn(root, "worker")
        await eventually(lambda: child.agent.agent_id in tools)
        first = await session.api_query("agents", {})
        assert first["agents"][0]["status"] == "running"
        assert first["agents"][0]["last_message"] == "Preparing 1"
        grandchild = await spawn(child, "nested")
        await eventually(lambda: grandchild.agent.agent_id in tools)
        payload = await session.api_query("agents", {})
        assert [(item["agent_path"], item["parent_agent_id"]) for item in payload["agents"]] == [
            ("/root/worker", root.agent.agent_id),
            ("/root/worker/nested", child.agent.agent_id),
        ]
        subtree = await session.api_query("agents", {"agent_id": child.agent.agent_id})
        assert subtree["root"]["agent_id"] == child.agent.agent_id
        assert [item["agent_id"] for item in subtree["agents"]] == [grandchild.agent.agent_id]
        histories = [copy.deepcopy(runtime.agent.history) for runtime in runtimes]
        calls = [client.call_count for client in clients]
        for _ in range(3):
            await session.api_query("agents", {})
        assert [runtime.agent.history for runtime in runtimes] == histories
        assert [client.call_count for client in clients] == calls
        assert [len(runtime._frontends) for runtime in runtimes] == [1, 1, 1]
        assert not child._interactive_frontends and not grandchild._interactive_frontends

        tools[child.agent.agent_id].set()
        await eventually(lambda: clients[1].call_count == 2)
        assert (await session.api_query("agents", {}))["agents"][0]["last_message"] == "Preparing 1"
        clients[1].finish.set()
        tools[grandchild.agent.agent_id].set()
        clients[2].finish.set()
        await wait_for_runtime_idle(child)
        await wait_for_runtime_idle(grandchild)
        payload = await session.api_query("agents", {})
        assert payload["agents"][0]["status"] == {"completed": "Result 1"}
        assert payload["agents"][0]["last_message"] == "Result 1"
        assert payload["agents"][1]["last_message"] == "Result 2"

        failed = await spawn(root, "failed")
        await wait_for_runtime_idle(failed)
        failed_row = (await session.api_query("agents", {"agent_id": failed.agent.agent_id}))["root"]
        assert failed_row["status"] == {"errored": "ValueError: fixture failure"}
        assert failed_row["last_message"] == ""

        interrupted = await spawn(root, "interrupted")
        await eventually(lambda: interrupted.agent.agent_id in tools)
        row = (await session.api_query("agents", {"agent_id": interrupted.agent.agent_id}))["root"]
        assert row["last_message"] == ""
        await manager.interrupt_agent(root.agent.agent_id, interrupted.agent.agent_id)
        tools[interrupted.agent.agent_id].set()
        await wait_for_runtime_idle(interrupted)
        assert interrupted.agent_status == "interrupted"
        await child.close()
        for removed in (child, grandchild):
            with pytest.raises(KeyError):
                await session.api_query("agents", {"agent_id": removed.agent.agent_id})
            assert removed.agent.agent_id not in session._child_views
            assert not removed._frontends
        assert [
            row["agent_path"] for row in (await session.api_query("agents", {}))["agents"]
        ] == ["/root/failed", "/root/interrupted"]
    finally:
        for release in tools.values():
            release.set()
        for client in clients:
            client.finish.set()
        await session.close()
    assert not session._child_views
    assert all(not runtime._frontends and runtime.agent.is_shutdown for runtime in runtimes)
    assert clients[0].call_count == 0


@pytest.mark.parametrize("threaded", [False, True])
def test_http_agent_tree_reads_and_capacity_removal(tmp_path, threaded):
    sessions = []

    def build():
        manager = AgentsManager(
            ScriptedModelClient(response_factory=lambda prompt, count: ModelResponse([
                AssistantMessage("Offline final"),
            ])),
            ContextConfig(cwd=tmp_path, codex_home=tmp_path, multi_agent_max_concurrency=2),
            toolset=(),
        )
        session = WorkspaceInteractiveSession(manager.create_root_runtime())

        async def spawn(argument):
            await manager.spawn_agent(
                session.runtime.agent.agent_id, argument, "task", fork_turns="none",
            )
            return {"kind": "fixture"}

        session.runtime.register_command("fixture_spawn", spawn)
        sessions.append(session)
        return session

    def factory():
        if threaded:
            return ThreadedWorkspaceInteractiveSession(build, asyncio.get_running_loop())
        return build()

    manager = WorkspaceSessionManager(factory, tmp_path / "state")
    with TestClient(create_app(manager, tmp_path / "state", work_dir=tmp_path, enable_board=False)) as client:
        tab_id = client.get("/api/sessions").json()["sessions"][0]["id"]
        base = "/api/sessions/" + tab_id
        initial = client.get(base + "/agents").json()
        assert initial["root"]["agent_path"] == "/root"
        assert initial["agents"] == []
        for _ in range(3):
            assert client.get(base + "/agents").json() == initial
        assert sessions[0].runtime.agent.history == ()
        assert sessions[0].runtime.agent.model_client.call_count == 0
        child_ids = []
        for name in ("child", "replacement"):
            assert client.post(
                base + "/commands", json={"command": "fixture_spawn", "argument": name},
            ).status_code == 202
            deadline = time.monotonic() + 3
            while True:
                payload = client.get(base + "/agents").json()
                rows = payload["agents"]
                if (
                    len(rows) == 1
                    and rows[0]["agent_path"] == "/root/" + name
                    and isinstance(rows[0]["status"], dict)
                ):
                    break
                assert time.monotonic() < deadline
                time.sleep(0.005)
            assert rows[0]["last_message"] == "Offline final"
            assert rows[0]["parent_agent_id"] == payload["root"]["agent_id"]
            child_ids.append(rows[0]["agent_id"])
        for resource in ("state", "models", "history", "agents", "background-work"):
            assert client.get(
                base + "/" + resource, params={"agent_id": child_ids[0]},
            ).status_code == 404
        assert client.post(
            base + "/commands", params={"agent_id": child_ids[0]},
            json={"command": "help"},
        ).status_code == 404
        assert set(sessions[0]._child_views) == {child_ids[1]}
    assert sessions[0].runtime.agent.is_shutdown


@pytest.mark.asyncio
async def test_attach_observes_existing_descendants(tmp_path):
    manager = AgentsManager(
        ScriptedModelClient([ModelResponse([AssistantMessage("Offline final")])]),
        context(tmp_path), toolset=(),
    )
    root = manager.create_root_runtime()
    await manager.spawn_agent(root.agent.agent_id, "existing", "task", fork_turns="none")
    child = manager.get_runtime(manager.list_agents(root.agent.agent_id)[1]["agent_id"])
    await wait_for_runtime_idle(child)
    session = WorkspaceInteractiveSession(root)
    try:
        await session.start()
        assert (await session.api_query("agents", {}))["agents"][0]["last_message"] == "Offline final"
        assert len(child._frontends) == 1
    finally:
        await session.close()
    assert not child._frontends


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_latest_assistant_stream_survives_tools_commands_and_failure(tmp_path, failure):
    release = [asyncio.Event(), asyncio.Event()]
    started = [asyncio.Event(), asyncio.Event()]

    class StreamingModel(ScriptedModelClient):
        async def complete(self, prompt, event_handler):
            event_handler(AssistantDeltaEvent("正在检查"))
            started[0].set()
            await release[0].wait()
            event_handler(AssistantDeltaEvent("最新消息"))
            started[1].set()
            await release[1].wait()
            if failure:
                raise ValueError("fixture error")
            return ModelResponse([AssistantMessage("检查完成")])

    root = AgentsManager(StreamingModel([]), context(tmp_path), toolset=()).create_root_runtime()
    session = await WorkspaceInteractiveSession(root).start()
    try:
        receipt = await root.submit_input("Check the last message")
        await asyncio.wait_for(started[0].wait(), 2)
        first = (await session.api_query("agents", {}))["root"]
        assert first["status"] == "running"
        assert first["last_message"] == "正在检查"
        session.view.write_line("Command output")
        assert (await session.api_query("agents", {}))["root"]["last_message"] == "正在检查"
        release[0].set()
        await asyncio.wait_for(started[1].wait(), 2)
        assert (await session.api_query("agents", {}))["root"]["last_message"] == "正在检查最新消息"
        release[1].set()
        if failure:
            with pytest.raises(ValueError, match="fixture error"):
                await receipt.future
        else:
            await receipt.future
        final = (await session.api_query("agents", {}))["root"]
        assert final["last_message"] == ("正在检查最新消息" if failure else "检查完成")
        assert ("errored" if failure else "completed") in final["status"]
    finally:
        for event in release:
            event.set()
        await session.close()
