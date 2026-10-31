import asyncio

import pytest

from pycodex import AgentsManager, AssistantMessage, ContextConfig, ModelResponse, ToolCall
from pycodex.protocol import AgentMessage
from pycodex.tools import BaseTool, ListAgentsTool, ToolContext
from tests.fakes import ScriptedModelClient, wait_for_runtime_idle


def make_tree(tmp_path, capacity, response_factory=None, toolset=()):
    model = ScriptedModelClient(
        response_factory=response_factory or (
            lambda prompt, count: ModelResponse([AssistantMessage("done " + str(count))])
        ),
    )
    manager = AgentsManager(
        model, ContextConfig(
            cwd=tmp_path, codex_home=tmp_path,
            multi_agent_max_concurrency=capacity,
        ), toolset=toolset,
    )
    return manager.create_root_runtime()


async def spawn(parent, name):
    manager = parent.agents_manager
    path = await manager.spawn_agent(
        parent.agent.agent_id, name, "task", fork_turns="none",
    )
    return manager.get_runtime(manager._resolve_agent_id(parent.agent.agent_id, path))


@pytest.mark.asyncio
async def test_capacity_includes_root_and_never_reclaims_it(tmp_path):
    root = make_tree(tmp_path, 1)
    await root.start()
    try:
        await root.submit_user_turn("finish")
        with pytest.raises(RuntimeError, match="capacity reached \\(1 including root\\)"):
            await spawn(root, "child")
        assert not root.agent.is_shutdown
        assert len(root.agents_manager.list_agents(root.agent.agent_id)) == 1
    finally:
        await root.close()


@pytest.mark.asyncio
async def test_capacity_removes_least_recently_used_child_and_releases_task_name(tmp_path):
    root = make_tree(tmp_path, 3)
    manager = root.agents_manager
    try:
        first = await spawn(root, "first")
        await wait_for_runtime_idle(first)
        second = await spawn(root, "second")
        await wait_for_runtime_idle(second)
        assert not first.agent.is_shutdown
        events = []
        second.attach(events.append, interactive=False)
        saved_history = second.agent.history
        await manager.send_message(
            root.agent.agent_id, "first", "followup", trigger_turn=True,
        )
        await wait_for_runtime_idle(first)

        third = await spawn(root, "third")
        await wait_for_runtime_idle(third)
        assert second.agent.is_shutdown and second._worker.done()
        assert not first.agent.is_shutdown and not third.agent.is_shutdown
        assert second.agent.history == saved_history
        assert [event.kind for event in events].count("session_closed") == 1
        with pytest.raises(KeyError):
            manager.get_runtime(second.agent.agent_id)
        with pytest.raises(KeyError):
            second.resume()
        assert [agent["agent_path"] for agent in manager.list_agents(root.agent.agent_id)] == [
            "/root", "/root/first", "/root/third",
        ]
        assert any(item.author == "/root/second" for item in root._steer_queue)
        context = ToolContext("list", (), tmp_path, agent_id=root.agent.agent_id)
        tool = ListAgentsTool(manager)
        result = await tool.run(context, {})
        assert [agent["agent_name"] for agent in result["agents"]] == [
            "/root", "/root/first", "/root/third",
        ]
        assert (await tool.run(context, {"path_prefix": "second"}))["agents"] == []
        with pytest.raises(ValueError, match="unknown agent"):
            await manager.send_message(
                root.agent.agent_id, "second", "cannot reload", trigger_turn=True,
            )
        with pytest.raises(ValueError, match="already exists"):
            await spawn(root, "first")
        assert not first.agent.is_shutdown and not third.agent.is_shutdown
        assert all(
            item.agent.recorded_session_file_path is None
            for item in (first, second, third)
        )
        recreated = await spawn(root, "second")
        await wait_for_runtime_idle(recreated)
        assert recreated.agent.agent_path == second.agent.agent_path
        assert recreated.agent.agent_id != second.agent.agent_id
        assert recreated.accepts_input
    finally:
        await root.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", ["running", "enqueue", "trigger", "mailbox", "command"])
async def test_capacity_preserves_accepted_work(tmp_path, pending):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, count):
        if count > 1:
            entered.set()
            await release.wait()
        return ModelResponse([AssistantMessage("done")])

    async def command(argument):
        entered.set()
        await release.wait()

    root = make_tree(tmp_path, 2, respond)
    child = await spawn(root, "child")
    try:
        await wait_for_runtime_idle(child)
        if pending in ("running", "enqueue"):
            child.enqueue("accepted")
        elif pending in ("trigger", "mailbox"):
            child.message("accepted")
            if pending == "trigger":
                child.trigger()
        else:
            child.register_command("held", command)
            await child.submit_input("/held", wait_for_command=False)
        if pending in ("running", "command"):
            await asyncio.wait_for(entered.wait(), 2)

        with pytest.raises(RuntimeError, match="capacity reached"):
            await spawn(root, "blocked")
        assert child.accepts_input
        assert not child.agent.is_shutdown
        assert len(root.agents_manager.list_agents(root.agent.agent_id)) == 2
    finally:
        release.set()
        await root.close()


@pytest.mark.asyncio
async def test_capacity_preserves_an_idle_caller(tmp_path):
    root = make_tree(tmp_path, 2)
    child = await spawn(root, "child")
    try:
        await wait_for_runtime_idle(child)
        with pytest.raises(RuntimeError, match="capacity reached"):
            await spawn(child, "nested")
        assert child.accepts_input
    finally:
        await root.close()


@pytest.mark.asyncio
async def test_capacity_does_not_close_an_ancestor_of_active_work(tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, count):
        task = next(item for item in prompt.input if isinstance(item, AgentMessage))
        if task.recipient.endswith("/leaf"):
            entered.set()
            await release.wait()
        return ModelResponse([AssistantMessage("done")])

    root = make_tree(tmp_path, 3, respond)
    parent = await spawn(root, "parent")
    try:
        await wait_for_runtime_idle(parent)
        leaf = await spawn(parent, "leaf")
        await asyncio.wait_for(entered.wait(), 2)
        with pytest.raises(RuntimeError, match="capacity reached"):
            await asyncio.wait_for(spawn(root, "blocked"), 2)
        assert parent.accepts_input and leaf.accepts_input

        release.set()
        await wait_for_runtime_idle(leaf)
        replacement = await spawn(root, "replacement")
        assert leaf.agent.is_shutdown
        assert parent.accepts_input and replacement.accepts_input
    finally:
        release.set()
        await root.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["errored", "interrupted"])
async def test_capacity_can_reclaim_failed_or_interrupted_work(tmp_path, status):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, count):
        entered.set()
        await release.wait()
        if status == "errored":
            raise ValueError("fixture failure")
        return ModelResponse([AssistantMessage("done")])

    root = make_tree(tmp_path, 2, respond)
    child = await spawn(root, "child")
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if status == "interrupted":
            await root.agents_manager.interrupt_agent(root.agent.agent_id, "child")
        release.set()
        await wait_for_runtime_idle(child)
        replacement = await spawn(root, "replacement")
        assert child.agent.is_shutdown
        assert replacement.accepts_input
    finally:
        release.set()
        await root.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("close_parent", [False, True])
@pytest.mark.parametrize("nested_parent", [False, True])
async def test_capacity_reservation_waits_for_cleanup_and_serializes_spawns(
    tmp_path, close_parent, nested_parent,
):
    can_finish = asyncio.Event()
    can_finish.set()
    closing = asyncio.Event()
    finish_close = asyncio.Event()

    async def respond(prompt, count):
        await can_finish.wait()
        return ModelResponse([AssistantMessage("done")])

    async def hold_cleanup():
        closing.set()
        await finish_close.wait()

    capacity = 3 if nested_parent else 2
    root = make_tree(tmp_path, capacity, respond)
    parent = await spawn(root, "parent") if nested_parent else root
    if nested_parent:
        await wait_for_runtime_idle(parent)
    old = await spawn(parent, "old")
    pending = []
    try:
        await wait_for_runtime_idle(old)
        old.add_close_handler(hold_cleanup)
        can_finish.clear()
        pending.append(asyncio.create_task(spawn(parent, "first")))
        await asyncio.wait_for(closing.wait(), 2)
        assert not old.agent.is_shutdown
        assert len(root.agents_manager.list_agents(root.agent.agent_id)) == capacity
        with pytest.raises(RuntimeError, match="shutting down"):
            await root.agents_manager.send_message(
                parent.agent.agent_id, "old", "too late",
            )
        if close_parent:
            pending.append(asyncio.create_task(parent.close()))
        else:
            pending.append(asyncio.create_task(spawn(parent, "second")))
        await asyncio.sleep(0)
        assert not any(task.done() for task in pending)
        finish_close.set()
        results = await asyncio.wait_for(
            asyncio.gather(*pending, return_exceptions=True), 2,
        )
        if close_parent:
            assert isinstance(results[0], RuntimeError)
            assert "shutting down" in str(results[0])
            assert results[1] is None
            assert len(root.agents_manager.list_agents(root.agent.agent_id)) == 1
        else:
            assert results[0].agent.agent_path == parent.agent.agent_path + "/first"
            assert isinstance(results[1], RuntimeError)
            assert "capacity reached" in str(results[1])
            assert len(root.agents_manager.list_agents(root.agent.agent_id)) == capacity
        assert old.agent.is_shutdown
        with pytest.raises(KeyError):
            root.agents_manager.get_runtime(old.agent.agent_id)
    finally:
        finish_close.set()
        can_finish.set()
        await asyncio.gather(*pending, return_exceptions=True)
        await root.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["close_handler", "tool_shutdown"])
async def test_capacity_cleanup_failure_is_visible_and_does_not_spawn(tmp_path, monkeypatch, failure):
    async def fail_cleanup():
        raise ValueError("cleanup failed")

    def fail_shutdown():
        raise ValueError("cleanup failed")

    root = make_tree(tmp_path, 2, toolset=("clock",))
    child = await spawn(root, "child")
    await wait_for_runtime_idle(child)
    tool = child.agent.tool_registry.get_tool("clock")
    events = []
    child.attach(events.append, interactive=False)
    if failure == "close_handler":
        child.add_close_handler(fail_cleanup)
    else:
        monkeypatch.setattr(tool._manager, "shutdown", fail_shutdown)
    try:
        with pytest.raises(ValueError, match="cleanup failed"):
            await spawn(root, "replacement")
        assert child.agent.is_shutdown
        assert [event.kind for event in events].count("session_closed") == 1
        assert tool._manager._state_listener is None
        assert len(root.agents_manager.list_agents(root.agent.agent_id)) == 1
        with pytest.raises(KeyError):
            root.agents_manager.get_runtime(child.agent.agent_id)
    finally:
        await root.close()


@pytest.mark.asyncio
async def test_capacity_preserves_enabled_clock(tmp_path):
    root = make_tree(tmp_path, 2, toolset=("clock",))
    child = await spawn(root, "child")
    try:
        await wait_for_runtime_idle(child)
        context = ToolContext("clock", (), tmp_path, agent_id=child.agent.agent_id)
        tool = child.agent.tool_registry.get_tool("clock")
        await tool.run(context, {"period_m": 60})
        with pytest.raises(RuntimeError, match="capacity reached"):
            await spawn(root, "blocked")
        assert child.accepts_input
        await tool.run(context, {"period_m": None})
        await spawn(root, "replacement")
        assert child.agent.is_shutdown
    finally:
        await root.close()


@pytest.mark.asyncio
async def test_capacity_preserves_running_code_mode_cell(tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()

    class HeldTool(BaseTool):
        name = "held"
        description = "Wait for the fixture."
        input_schema = {"type": "object", "properties": {}}

        async def run(self, context, args):
            entered.set()
            await release.wait()
            return "finished"

    root = make_tree(tmp_path, 2, toolset=("exec", "wait"))
    child = await spawn(root, "child")
    registry = child.agent.tool_registry
    registry.register(HeldTool())
    cell = None
    try:
        await wait_for_runtime_idle(child)
        context = ToolContext("exec", (), tmp_path, agent_id=child.agent.agent_id)
        result = await registry.execute(
            ToolCall(
                "exec", "exec",
                '# @exec: {"yield_time_ms": 1}\ntext(await tools.held())',
                tool_type="custom",
            ), context,
        )
        assert not result.is_error
        tool = registry.get_tool("exec")
        cell = next(iter(tool._manager._cells.values()))
        await asyncio.wait_for(entered.wait(), 2)
        assert child.agent.tool_registry.runtime_environment.background_work_count(True) == 1
        with pytest.raises(RuntimeError, match="capacity reached"):
            await spawn(root, "blocked")
        assert child.accepts_input
        release.set()
        await asyncio.wait_for(cell.task, 2)
        assert child.agent.tool_registry.runtime_environment.background_work_count(True) == 0
        await spawn(root, "replacement")
        assert child.agent.is_shutdown
    finally:
        release.set()
        try:
            if cell is not None:
                await asyncio.wait_for(cell.task, 2)
        finally:
            await root.close()
