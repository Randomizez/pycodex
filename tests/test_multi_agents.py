import asyncio

import pytest

from pycodex import AgentsManager, AssistantMessage, ContextConfig, ModelResponse, ToolCall
from pycodex.protocol import AgentMessage, ToolResult
from pycodex.tools import ClockTool
from tests.fakes import ScriptedModelClient, wait_for_runtime_idle


def make_tree(tmp_path, model, toolset=()):
    manager = AgentsManager(
        model, ContextConfig(cwd=tmp_path, codex_home=tmp_path), toolset=toolset,
    )
    return manager.create_root_runtime()


async def spawn(root, name="worker"):
    manager = root.agents_manager
    path = await manager.spawn_agent(
        root.agent.agent_id, name, "task", fork_turns="none",
    )
    return manager.get_runtime(manager._resolve_agent_id(root.agent.agent_id, path))


def messages(items):
    return [item for item in items if isinstance(item, AgentMessage)]


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupted", [False, True])
async def test_child_drains_pending_work_after_failure_or_interruption(tmp_path, interrupted):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, count):
        if count == 1:
            entered.set()
            await release.wait()
            if not interrupted:
                raise ValueError("first failed")
            return ModelResponse([AssistantMessage("first answer")])
        return ModelResponse([AssistantMessage("followup answer")])

    root = make_tree(tmp_path, ScriptedModelClient(response_factory=respond))
    child = await spawn(root)
    try:
        await entered.wait()
        _submission_id, pending = child.enqueue("next")
        if interrupted:
            assert await root.agents_manager.interrupt_agent(
                root.agent.agent_id, "worker",
            ) == {"previous_status": "running"}
        release.set()
        assert (await pending).output_text == "followup answer"
        assert child.agent_status == {"completed": "followup answer"}
        notifications = messages(root._steer_queue)
        assert len(notifications) == (1 if interrupted else 2)
        assert notifications[-1].text.endswith("followup answer")
        if not interrupted:
            assert notifications[0].text.endswith("Agent errored: ValueError: first failed")
        assert not root.is_busy
    finally:
        release.set()
        await root.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("wait_first", [False, True])
async def test_completion_is_delivered_once_after_issued_tool_batch(tmp_path, monkeypatch, wait_first):
    release_child = asyncio.Event()

    async def finish_child(prompt, count):
        await release_child.wait()
        return ModelResponse([AssistantMessage("child result")])

    async def respond(prompt, count):
        if count == 1:
            release_child.set()
            calls = [
                ToolCall("wait", "wait_agent", {}),
                ToolCall("clock", "clock", {"period_m": None}),
            ]
            return ModelResponse(calls if wait_first else list(reversed(calls)))
        if count == 2:
            return ModelResponse([ToolCall("again", "clock", {"period_m": None})])
        return ModelResponse([AssistantMessage("parent result")])

    model = ScriptedModelClient(response_factory=respond)
    child_model = ScriptedModelClient(response_factory=finish_child)
    monkeypatch.setattr(model, "with_overrides", lambda *args, **kwargs: child_model)
    root = make_tree(tmp_path, model, ("clock", "wait_agent"))
    child = await spawn(root)

    async def clock(self, context, args):
        if context.turn_id == model.prompts[0].turn_id and len(model.prompts) == 1:
            assert not messages(context.history)
        await wait_for_runtime_idle(child)
        return {"enabled": False, "period_m": None}

    monkeypatch.setattr(ClockTool, "run", clock)
    await root.start()
    try:
        assert (await root.submit_user_turn("work")).output_text == "parent result"
        notifications = messages(root.agent.history)
        assert len(notifications) == 1
        assert notifications[0].author == "/root/worker"
        assert notifications[0].recipient == "/root"
        assert notifications[0].text.endswith("child result")
        assert messages(model.prompts[1].input) == notifications
        assert messages(model.prompts[2].input) == notifications
        first_followup = list(model.prompts[1].input)
        message_index = first_followup.index(notifications[0])
        assert all(
            index < message_index
            for index, item in enumerate(first_followup) if isinstance(item, ToolResult)
        )
        assert not root._steer_queue
    finally:
        release_child.set()
        await root.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("during_reply", [False, True])
async def test_completion_joins_active_parent_or_waits_for_next_turn(tmp_path, monkeypatch, during_reply):
    release_child = asyncio.Event()

    async def finish_child(prompt, count):
        await release_child.wait()
        return ModelResponse([AssistantMessage("late result")])

    async def respond(prompt, count):
        if count == 1 and during_reply:
            release_child.set()
            await wait_for_runtime_idle(child)
        return ModelResponse([AssistantMessage("parent result")])

    model = ScriptedModelClient(response_factory=respond)
    monkeypatch.setattr(
        model, "with_overrides",
        lambda *args, **kwargs: ScriptedModelClient(response_factory=finish_child),
    )
    root = make_tree(tmp_path, model)
    child = await spawn(root)
    await root.start()
    try:
        await root.submit_user_turn("first")
        release_child.set()
        await wait_for_runtime_idle(child)
        assert model.call_count == (2 if during_reply else 1)
        assert len(messages(root.agent.history)) == (1 if during_reply else 0)
        assert not root.is_busy
        await root.submit_user_turn("next")
        assert len(messages(root.agent.history)) == 1
        assert messages(model.prompts[-1].input)[0].text.endswith("late result")
    finally:
        release_child.set()
        await root.close()


@pytest.mark.asyncio
async def test_identical_completed_turns_have_distinct_messages_and_copied_status(tmp_path):
    root = make_tree(tmp_path, ScriptedModelClient([
        ModelResponse([AssistantMessage("same result")]),
        ModelResponse([AssistantMessage("same result")]),
    ]))
    child = await spawn(root)
    manager = root.agents_manager
    try:
        await wait_for_runtime_idle(child)
        status = manager.list_agents(root.agent.agent_id, "worker")[0]["status"]
        status["completed"] = "changed by caller"
        assert child.agent_status == {"completed": "same result"}
        await manager.send_message(
            root.agent.agent_id, "worker", "second", trigger_turn=True,
        )
        await wait_for_runtime_idle(child)
        notifications = messages(root._steer_queue)
        assert len(notifications) == 2
        assert notifications[0].text == notifications[1].text
        assert notifications[0].id != notifications[1].id
        assert await root.wait_for_activity(1) == {
            "message": "Wait completed.", "timed_out": False,
        }
        assert messages(root._steer_queue) == notifications
    finally:
        await root.close()


@pytest.mark.asyncio
async def test_child_background_invocation_updates_status_and_notifies_parent(tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, count):
        if count == 2:
            entered.set()
            await release.wait()
        return ModelResponse([AssistantMessage("answer " + str(count))])

    root = make_tree(tmp_path, ScriptedModelClient(response_factory=respond))
    child = await spawn(root)
    invocation = None
    try:
        await wait_for_runtime_idle(child)
        invocation = asyncio.create_task(child.maybe_invoke({"type": "clock_tick"}))
        await entered.wait()
        assert child.agent_status == "running"
        assert not await child.maybe_invoke({"type": "clock_tick"})
        release.set()
        assert await invocation
        assert child.agent_status == {"completed": "answer 2"}
        assert len(messages(root._steer_queue)) == 2
        await child.close()
        assert not await child.maybe_invoke({"type": "clock_tick"})
        assert child.agent_status == "shutdown"
    finally:
        release.set()
        if invocation is not None:
            await asyncio.gather(invocation, return_exceptions=True)
        await root.close()


@pytest.mark.asyncio
async def test_wait_timeout_can_be_followed_by_child_completion(tmp_path):
    release = asyncio.Event()

    async def respond(prompt, count):
        await release.wait()
        return ModelResponse([AssistantMessage("child result")])

    root = make_tree(tmp_path, ScriptedModelClient(response_factory=respond))
    child = await spawn(root)
    try:
        assert (await root.wait_for_activity(1))["timed_out"]
        waiter = asyncio.create_task(root.wait_for_activity(1000))
        release.set()
        assert await waiter == {"message": "Wait completed.", "timed_out": False}
        await wait_for_runtime_idle(child)
        assert len(messages(root._steer_queue)) == 1
    finally:
        release.set()
        await root.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("start_sampling", [False, True])
async def test_parent_close_drains_active_and_queued_child_work(tmp_path, start_sampling):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, count):
        entered.set()
        await release.wait()
        return ModelResponse([AssistantMessage("done")])

    root = make_tree(tmp_path, ScriptedModelClient(response_factory=respond))
    child = await spawn(root)
    closing = None
    try:
        if start_sampling:
            await entered.wait()
        _submission_id, pending = child.enqueue("also finish")
        closing = asyncio.create_task(root.close())
        await entered.wait()
        await asyncio.sleep(0)
        assert not closing.done()
        assert child.agent.is_running
        release.set()
        await asyncio.wait_for(closing, 2)
        assert (await pending).output_text == "done"
        assert child.agent.model_client.call_count == 2
        assert child.agent_status == "shutdown"
        assert root.agent_status == "shutdown"
        assert len(messages(root._steer_queue)) == 2
    finally:
        release.set()
        if closing is not None:
            await closing
        await root.close()


@pytest.mark.asyncio
async def test_immediate_child_close_finishes_before_deferred_queue_notifications(tmp_path):
    root = make_tree(tmp_path, ScriptedModelClient([ModelResponse([AssistantMessage("done")])]))
    reported = []
    events = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: reported.append(context["exception"]))
    try:
        child = await spawn(root)
        child.attach(events.append, interactive=False)
        await child.close()
        assert child.agent_status == "shutdown"
        assert events[-1].kind == "session_closed"
        assert not reported
        assert len(messages(root._steer_queue)) == 1
        assert len(root.agents_manager.list_agents(root.agent.agent_id)) == 1
    finally:
        try:
            await root.close()
        finally:
            loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_parent_close_starts_child_cleanup_concurrently(tmp_path):
    root = make_tree(tmp_path, ScriptedModelClient([ModelResponse([AssistantMessage("done")])]))
    children = [await spawn(root, "child_" + str(index)) for index in range(2)]
    entered = [asyncio.Event() for child in children]
    release = asyncio.Event()
    for index, child in enumerate(children):
        async def cleanup(index=index):
            entered[index].set()
            await release.wait()
        child.add_close_handler(cleanup)
    closing = asyncio.create_task(root.close())
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered)), 2,
        )
        assert all(not child.accepts_input for child in children)
        assert not closing.done()
    finally:
        release.set()
        await closing


@pytest.mark.asyncio
async def test_parent_close_cleans_every_child_after_cleanup_errors(tmp_path):
    root = make_tree(tmp_path, ScriptedModelClient([ModelResponse([AssistantMessage("done")])]))
    children = [await spawn(root, "child_" + str(index)) for index in range(3)]
    grandchildren = [await spawn(children[0], "nested_" + str(index)) for index in range(2)]
    failures = [RuntimeError("first cleanup failed"), RuntimeError("second cleanup failed")]
    for child, failure in zip(children, failures):
        async def fail_cleanup(error=failure):
            raise error
        child.add_close_handler(fail_cleanup)
    reported = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: reported.append(context["exception"]))
    try:
        with pytest.raises(RuntimeError, match="first cleanup failed") as caught:
            await root.close()
        assert caught.value is failures[0]
        assert reported == failures[1:]
        assert root.agent.is_shutdown
        assert all(
            child.agent.is_shutdown and child._worker.done()
            for child in children + grandchildren
        )
        assert root.agents_manager.list_agents(root.agent.agent_id) == [{
            "agent_id": root.agent.agent_id,
            "parent_agent_id": None,
            "agent_path": "/root",
            "status": "shutdown",
        }]
    finally:
        await asyncio.gather(*(child.close() for child in children), return_exceptions=True)
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_task_paths_and_root_operations_are_validated(tmp_path):
    root = make_tree(tmp_path, ScriptedModelClient([ModelResponse([AssistantMessage("done")])]))
    manager = root.agents_manager
    child = await spawn(root, "worker")
    try:
        nested = await spawn(child, "nested")
        assert nested.agent.agent_path == "/root/worker/nested"
        assert manager.get_parent_agent_id(nested.agent.agent_id) == child.agent.agent_id
        assert [
            item["agent_path"] for item in manager.list_agents(root.agent.agent_id, "worker")
        ] == ["/root/worker", "/root/worker/nested"]
        with pytest.raises(ValueError, match="already exists"):
            await spawn(root, "worker")
        for target in ("worker/", "../worker", "/root/root", "/elsewhere"):
            with pytest.raises(ValueError, match="invalid task path"):
                await manager.send_message(root.agent.agent_id, target, "message")
        with pytest.raises(ValueError, match="unknown agent"):
            await manager.send_message(root.agent.agent_id, "missing", "message")
        with pytest.raises(ValueError, match="cannot target the root"):
            await manager.send_message(
                child.agent.agent_id, "/root", "task", trigger_turn=True,
            )
        with pytest.raises(ValueError, match="cannot interrupt itself"):
            await manager.interrupt_agent(child.agent.agent_id, child.agent.agent_id)
        with pytest.raises(ValueError, match="root is not"):
            await manager.interrupt_agent(child.agent.agent_id, "/root")
    finally:
        await root.close()
