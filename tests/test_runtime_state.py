import asyncio

import pytest

from pycodex import (
    Agent,
    AgentRuntime,
    AssistantMessage,
    BaseTool,
    ContextConfig,
    ContextLengthExceeded,
    ModelResponse,
    ToolCall,
    ToolRegistry,
    TurnInterrupted,
    UserMessage,
)
from pycodex.bootstrap import get_tools
from pycodex.events import TokenCountEvent
from pycodex.runtime import SubmissionInterrupted
from pycodex.runtime_services import AgentRuntimeEnvironment, SubAgentManager
from tests.fakes import ScriptedModelClient


def test_agent_and_tools_share_a_session_local_runtime():
    environment = AgentRuntimeEnvironment()
    tools = get_tools(environment)
    agent = Agent(ScriptedModelClient([]), tools, ContextConfig())
    other = Agent(ScriptedModelClient([]), ToolRegistry(), ContextConfig())

    assert tools.runtime_environment is environment
    assert agent.tool_registry.runtime_environment is environment
    assert other.tool_registry.runtime_environment is not environment
    assert (
        other.tool_registry.runtime_environment.plan_store is not environment.plan_store
    )


@pytest.mark.asyncio
async def test_shutdown_finishes_active_and_pending_submissions():
    started = asyncio.Event()
    finished = asyncio.Event()
    release = asyncio.Event()

    async def response_factory(prompt, call_count):
        started.set()
        await release.wait()
        finished.set()
        return ModelResponse([AssistantMessage("done")])

    agent = Agent(
        ScriptedModelClient(response_factory=response_factory),
        ToolRegistry(),
        ContextConfig(),
    )
    queue = AgentRuntime(agent)
    await queue.start()
    futures = []
    closing = None
    try:
        _submission_id, first = await queue.enqueue_user_turn("first")
        futures.append(first)
        await asyncio.wait_for(started.wait(), 1)
        _submission_id, second = await queue.enqueue_user_turn("second")
        futures.append(second)
        assert queue.is_busy

        closing = asyncio.create_task(queue.close())
        await asyncio.sleep(0)
        assert not closing.done()
        assert agent.is_running
        release.set()
        for future in futures:
            assert (await future).output_text == "done"
        await closing
        assert finished.is_set()
        assert not queue.is_busy
    finally:
        release.set()
        if closing is None:
            await queue.close()
        else:
            await closing
        await asyncio.gather(*futures, return_exceptions=True)


@pytest.mark.asyncio
async def test_queue_busy_includes_direct_agent_work_until_it_finishes():
    started = asyncio.Event()
    release = asyncio.Event()

    async def response_factory(prompt, call_count):
        started.set()
        await release.wait()
        return ModelResponse([AssistantMessage("done")])

    agent = Agent(
        ScriptedModelClient(response_factory=response_factory),
        ToolRegistry(),
        ContextConfig(),
    )
    queue = AgentRuntime(agent)
    turn = asyncio.create_task(agent.run_turn(["direct turn"]))
    try:
        await asyncio.wait_for(started.wait(), 1)
        assert queue.is_busy
        agent.shutdown()
        assert queue.is_busy
        release.set()
        assert (await turn).output_text == "done"
        assert not queue.is_busy
    finally:
        release.set()
        await asyncio.gather(turn, return_exceptions=True)


@pytest.mark.asyncio
async def test_queue_busy_includes_pending_submissions_before_worker_starts():
    queue = AgentRuntime(
        Agent(
            ScriptedModelClient([ModelResponse([AssistantMessage("done")])]),
            ToolRegistry(),
            ContextConfig(),
        )
    )
    _submission_id, future = await queue.enqueue_user_turn("pending")
    assert queue.is_busy
    await queue.close()
    assert (await future).output_text == "done"
    assert not queue.is_busy
    with pytest.raises(ValueError, match="unknown submission queue"):
        await queue.enqueue_user_turn("invalid", queue="typo")


@pytest.mark.asyncio
async def test_interactive_commands_reject_direct_background_work(monkeypatch):
    from workspace_server.app import WorkspaceInteractiveSession

    started = asyncio.Event()
    release = asyncio.Event()

    async def response_factory(prompt, call_count):
        started.set()
        await release.wait()
        return ModelResponse([AssistantMessage("done")])

    model = ScriptedModelClient(response_factory=response_factory)
    model.model = "original"
    agent = Agent(model, ToolRegistry(), ContextConfig())
    session = WorkspaceInteractiveSession(AgentRuntime(agent))
    lines = []
    commands_handled = asyncio.Event()

    def record_line(text):
        lines.append(text)
        if "Cannot resume" in text:
            commands_handled.set()

    monkeypatch.setattr(session.view, "show_error", record_line)
    await session.start()
    await asyncio.sleep(0)
    turn = asyncio.create_task(agent.run_turn(["background"]))
    try:
        await asyncio.wait_for(started.wait(), 1)
        for command in ["/model changed", "/compact", "/fork", "/resume 1"]:
            await asyncio.wait_for(session.submit(command), 1)
        await asyncio.wait_for(commands_handled.wait(), 1)
        output = "\n".join(lines)
        for message in [
            "Cannot change model",
            "Cannot compact",
            "Cannot fork",
            "Cannot resume",
        ]:
            assert message in output
        assert agent.model_name == "original"
        assert model.call_count == 1
    finally:
        release.set()
        await turn
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", [False, True])
async def test_subagent_waits_for_pending_work_after_failed_or_interrupted_turn(
    interrupt,
):
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()
    release_second = asyncio.Event()

    async def response_factory(prompt, call_count):
        if call_count == 1:
            first_started.set()
            await release_first.wait()
            if not interrupt:
                raise RuntimeError("first turn failed")
            return ModelResponse([AssistantMessage("first answer")])
        second_started.set()
        await release_second.wait()
        return ModelResponse([AssistantMessage("follow-up done")])

    model = ScriptedModelClient(response_factory=response_factory)
    manager = SubAgentManager()
    manager.set_runtime_builder(
        lambda model_name, effort, history, session_id: AgentRuntime(
            Agent(model, ToolRegistry(), ContextConfig(), initial_history=history)
        )
    )
    spawned = await manager.spawn_agent("first", None, None, False, None, None, ())
    agent_id = spawned["agent_id"]
    try:
        await asyncio.wait_for(first_started.wait(), 1)
        sent = await manager.send_input(agent_id, "next", interrupt)
        if interrupt:
            batched = await manager.send_input(agent_id, "also next", True)
            assert batched["submission_id"] == sent["submission_id"]
        release_first.set()
        await asyncio.wait_for(second_started.wait(), 1)

        pending = await manager.wait_agents([agent_id], timeout_ms=10)
        assert pending == {"status": {}, "timed_out": True}
        release_second.set()
        result = await manager.wait_agents([agent_id], timeout_ms=1000)
        assert result == {
            "status": {agent_id: {"completed": "follow-up done"}},
            "timed_out": False,
        }
        await manager.close_agent(agent_id)
        await asyncio.sleep(0)
        assert (await manager.wait_agents([agent_id]))["status"] == {
            agent_id: "shutdown"
        }
        assert not manager._agents[agent_id].runtime.is_busy
    finally:
        release_first.set()
        release_second.set()
        await manager.close_agent(agent_id)


@pytest.mark.asyncio
async def test_subagent_status_result_does_not_expose_mutable_state():
    manager = SubAgentManager()
    manager.set_runtime_builder(
        lambda model_name, effort, history, session_id: AgentRuntime(
            Agent(
                ScriptedModelClient([ModelResponse([AssistantMessage("done")])]),
                ToolRegistry(),
                ContextConfig(),
            )
        )
    )
    spawned = await manager.spawn_agent("prompt", None, None, False, None, None, ())
    agent_id = spawned["agent_id"]
    try:
        snapshot = await manager.wait_agents([agent_id], 1000)
        snapshot["status"][agent_id]["completed"] = "changed by caller"
        current = await manager.wait_agents([agent_id], 1000)
        assert current["status"][agent_id] == {"completed": "done"}
    finally:
        await manager.close_agent(agent_id)


@pytest.mark.asyncio
async def test_subagent_status_tracks_background_invocation_and_shutdown():
    background_started = asyncio.Event()
    release_background = asyncio.Event()

    async def response_factory(prompt, call_count):
        if call_count == 1:
            return ModelResponse([AssistantMessage("first answer")])
        background_started.set()
        await release_background.wait()
        return ModelResponse([AssistantMessage("background answer")])

    model = ScriptedModelClient(response_factory=response_factory)
    manager = SubAgentManager()
    manager.set_runtime_builder(
        lambda model_name, effort, history, session_id: AgentRuntime(
            Agent(model, ToolRegistry(), ContextConfig())
        )
    )
    spawned = await manager.spawn_agent("first", None, None, False, None, None, ())
    agent_id = spawned["agent_id"]
    child = manager._agents[agent_id].runtime.agent
    invoked = None
    try:
        assert (await manager.wait_agents([agent_id], 1000))["status"] == {
            agent_id: {"completed": "first answer"}
        }
        invoked = asyncio.create_task(
            child.maybe_invoke({"type": "exec_command_completed"})
        )
        await asyncio.wait_for(background_started.wait(), 1)
        assert not invoked.done()
        assert await manager.wait_agents([agent_id], 10) == {
            "status": {},
            "timed_out": True,
        }
        waiter = asyncio.create_task(manager.wait_agents([agent_id], 1000))
        release_background.set()
        assert await invoked
        assert (await waiter)["status"] == {
            agent_id: {"completed": "background answer"}
        }
        await manager.close_agent(agent_id)
        assert not await child.maybe_invoke({"type": "exec_command_completed"})
        with pytest.raises(RuntimeError, match="shutdown"):
            await child.run_turn(["closed"])
        assert (await manager.resume_agent(agent_id))["status"] == "pending_init"
        await manager.send_input(agent_id, "resume", False)
        assert (await manager.wait_agents([agent_id], 1000))["status"] == {
            agent_id: {"completed": "background answer"}
        }
    finally:
        release_background.set()
        if invoked is not None:
            await asyncio.gather(invoked, return_exceptions=True)
        await manager.close_agent(agent_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupted", [False, True])
async def test_subagent_waiter_is_notified_by_direct_turn_failure_or_stop(interrupted):
    def fail(prompt, call_count):
        if interrupted:
            child.stop_asap()
            return ModelResponse([])
        raise ValueError("direct failure")

    manager = SubAgentManager()
    manager.set_runtime_builder(
        lambda model_name, effort, history, session_id: AgentRuntime(
            Agent(
                ScriptedModelClient(response_factory=fail),
                ToolRegistry(),
                ContextConfig(),
            )
        )
    )
    spawned = await manager.spawn_agent(None, None, None, False, None, None, ())
    agent_id = spawned["agent_id"]
    child = manager._agents[agent_id].runtime.agent
    try:
        waiter = asyncio.create_task(manager.wait_agents([agent_id], 1000))
        await asyncio.sleep(0)
        error_type = TurnInterrupted if interrupted else ValueError
        message = "turn interrupted" if interrupted else "direct failure"
        with pytest.raises(error_type, match=message):
            await child.run_turn(["fail"])
        result = await waiter
        assert result["status"] == {
            agent_id: {"errored": error_type.__name__ + ": " + message}
        }
    finally:
        await manager.close_agent(agent_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_runtime_stops_direct_invoke_then_owns_steer_execution(failed):
    started = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, call_count):
        if call_count == 1:
            started.set()
            await release.wait()
            return ModelResponse([AssistantMessage("first")])
        if failed:
            raise ValueError("follow-up failed")
        return ModelResponse([AssistantMessage("final")])

    agent = Agent(
        ScriptedModelClient(response_factory=respond), ToolRegistry(), ContextConfig()
    )
    queue = AgentRuntime(agent)
    invocation = asyncio.create_task(agent.maybe_invoke({"type": "wake_up"}))
    steered = None
    try:
        await asyncio.wait_for(started.wait(), 1)
        _submission_id, steered = await queue.enqueue_user_turn("steer", queue="steer")
        release.set()
        with pytest.raises(TurnInterrupted):
            await invocation
        assert not steered.done()
        assert agent.model_client.call_count == 1
        await queue.start()
        if failed:
            with pytest.raises(ValueError, match="follow-up failed"):
                await steered
        else:
            assert (await steered).output_text == "final"
        assert not queue.is_busy
    finally:
        release.set()
        await asyncio.gather(invocation, return_exceptions=True)
        await queue.close()
        if steered is not None:
            await asyncio.gather(steered, return_exceptions=True)


@pytest.mark.asyncio
async def test_parent_queue_shutdown_waits_for_child_work():
    started = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, call_count):
        started.set()
        await release.wait()
        return ModelResponse([AssistantMessage("child done")])

    parent = Agent(ScriptedModelClient([]), ToolRegistry(), ContextConfig())
    manager = parent.tool_registry.runtime_environment.subagent_manager
    manager.set_runtime_builder(
        lambda model, effort, history, session_id: AgentRuntime(
            Agent(
                ScriptedModelClient(response_factory=respond),
                ToolRegistry(),
                ContextConfig(),
            )
        )
    )
    queue = AgentRuntime(parent)
    await queue.start()
    spawned = await manager.spawn_agent("finish me", None, None, False, None, None, ())
    await asyncio.wait_for(started.wait(), 1)
    closing = asyncio.create_task(queue.close())
    try:
        await asyncio.sleep(0)
        assert not closing.done()
        child = manager._agents[spawned["agent_id"]].runtime.agent
        assert child.is_running
        release.set()
        await asyncio.wait_for(closing, 1)
        assert child.history[-1] == AssistantMessage("child done")
        assert (await manager.wait_agents([spawned["agent_id"]], 10))["status"] == {
            spawned["agent_id"]: "shutdown",
        }
    finally:
        release.set()
        await closing


@pytest.mark.asyncio
async def test_parent_close_cleans_all_children_after_child_cleanup_errors():
    parent = AgentRuntime(
        Agent(ScriptedModelClient([]), ToolRegistry(), ContextConfig())
    )
    manager = parent.agent.tool_registry.runtime_environment.subagent_manager
    manager.set_runtime_builder(
        lambda model, effort, history, session_id: AgentRuntime(
            Agent(ScriptedModelClient([]), ToolRegistry(), ContextConfig())
        )
    )
    await parent.start()
    for index in range(3):
        await manager.spawn_agent(None, None, None, False, None, None, ())
    children = [managed.runtime for managed in manager._agents.values()]
    failures = [
        RuntimeError("first cleanup failed"),
        RuntimeError("second cleanup failed"),
    ]
    for child, failure in zip(children, failures):

        async def fail_cleanup(error=failure):
            raise error

        child.add_close_handler(fail_cleanup)
    reported = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(
        lambda loop, context: reported.append(context["exception"])
    )
    try:
        with pytest.raises(RuntimeError, match="first cleanup failed") as caught:
            await parent.close()
        assert caught.value is failures[0]
        assert reported == failures[1:]
        assert parent.agent.is_shutdown
        assert all(
            child.agent.is_shutdown and not child.accepts_input for child in children
        )
        assert all(child._worker.done() for child in children)
    finally:
        await asyncio.gather(
            *(child.close() for child in children), return_exceptions=True
        )
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
@pytest.mark.parametrize("overflow", [False, True])
async def test_steer_during_compaction_restarts_execution_with_shared_turn_id(overflow):
    request_started = asyncio.Event()
    release_request = asyncio.Event()
    compact_started = asyncio.Event()
    release_compact = asyncio.Event()

    class CompactingClient:
        model = "test"

        def __init__(self):
            self.prompts = []

        async def complete(self, prompt, event_handler):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                request_started.set()
                await release_request.wait()
                if overflow:
                    raise ContextLengthExceeded("input exceeds the context window")
                event_handler(TokenCountEvent({"total_tokens": 100}))
                return ModelResponse([AssistantMessage("first answer")])
            if len(self.prompts) == 2:
                compact_started.set()
                await release_compact.wait()
                return ModelResponse([AssistantMessage("summary")])
            return ModelResponse([AssistantMessage("final answer")])

    model = CompactingClient()
    agent = Agent(
        model,
        ToolRegistry(),
        ContextConfig(model_auto_compact_token_limit=50),
    )
    queue = AgentRuntime(agent)
    events = []
    queue.event_handler = events.append
    await queue.start()
    futures = []
    try:
        _submission_id, original = await queue.enqueue_user_turn("original")
        futures.append(original)
        await asyncio.wait_for(request_started.wait(), 1)
        first_id, first_steer = await queue.enqueue_user_turn(
            "steer one", queue="steer"
        )
        futures.append(first_steer)
        release_request.set()
        await asyncio.wait_for(compact_started.wait(), 1)
        second_id, second_steer = await queue.enqueue_user_turn(
            "steer two", queue="steer"
        )
        futures.append(second_steer)
        release_compact.set()
        result = await asyncio.wait_for(second_steer, 1)
        with pytest.raises(SubmissionInterrupted):
            await original
        if overflow:
            assert first_id == second_id
            assert await first_steer is result
        else:
            assert first_id != second_id
            with pytest.raises(SubmissionInterrupted):
                await first_steer
        assert [event.kind for event in events].count("turn_started") == (
            2 if overflow else 3
        )
        assert result.output_text == "final answer"
        assert len(model.prompts) == 3
        assert all(prompt.turn_id == result.turn_id for prompt in model.prompts)
        assert [
            item.text
            for item in model.prompts[-1].input
            if isinstance(item, UserMessage)
        ][-2:] == [
            "steer one",
            "steer two",
        ]
    finally:
        release_request.set()
        release_compact.set()
        await queue.close()
        await asyncio.gather(*futures, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("overflow", [False, True])
async def test_steer_arriving_during_tool_follow_up_compact_enters_next_request(
    overflow,
):
    compact_started = asyncio.Event()
    release_compact = asyncio.Event()
    executions = []

    class CountingTool(BaseTool):
        name = "count"
        description = "Records each execution."

        async def run(self, context, args):
            executions.append(context.turn_id)
            return "completed once"

    class CompactingClient:
        model = "test"

        def __init__(self):
            self.prompts = []

        async def complete(self, prompt, event_handler):
            self.prompts.append(prompt)
            request_number = len(self.prompts)
            if request_number == 1:
                if not overflow:
                    event_handler(TokenCountEvent({"total_tokens": 100}))
                return ModelResponse(
                    [
                        AssistantMessage("old commentary"),
                        ToolCall("count_call", "count", {}),
                    ]
                )
            if overflow and request_number == 2:
                raise ContextLengthExceeded("input exceeds the context window")
            if request_number == (3 if overflow else 2):
                compact_started.set()
                await release_compact.wait()
                return ModelResponse([AssistantMessage("summary")])
            return ModelResponse([])

    tools = ToolRegistry()
    tools.register(CountingTool())
    client = CompactingClient()
    agent = Agent(
        client,
        tools,
        ContextConfig(model_auto_compact_token_limit=50),
    )
    queue = AgentRuntime(agent)
    await queue.start()
    futures = []
    events = []
    queue.event_handler = events.append
    try:
        _submission_id, original = await queue.enqueue_user_turn("original")
        futures.append(original)
        await asyncio.wait_for(compact_started.wait(), 1)
        first_id, first = await queue.enqueue_user_turn("steer one", queue="steer")
        second_id, second = await queue.enqueue_user_turn("steer two", queue="steer")
        futures.extend([first, second])
        assert first_id == second_id
        assert not original.done()
        assert not first.done()
        release_compact.set()
        result = await asyncio.wait_for(second, 1)
        with pytest.raises(SubmissionInterrupted):
            await original
        assert await first is result
        assert result.output_text is None
        assert result.iterations == 1
        assert [event.kind for event in events].count("turn_started") == 2
        assert executions == [result.turn_id]
        assert len(client.prompts) == (4 if overflow else 3)
        assert all(prompt.turn_id == result.turn_id for prompt in client.prompts)
        assert [
            item.text
            for item in client.prompts[-1].input
            if isinstance(item, UserMessage)
        ][-2:] == [
            "steer one",
            "steer two",
        ]
        assert not queue.is_busy
        interrupted = [event for event in events if event.kind == "turn_interrupted"]
        assert len(interrupted) == 1
        assert interrupted[0].output_text == "old commentary"
    finally:
        release_compact.set()
        await queue.close()
        await asyncio.gather(*futures, return_exceptions=True)


@pytest.mark.asyncio
async def test_shutdown_drains_existing_inputs_but_rejects_new_invocations():
    started = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, call_count):
        started.set()
        await release.wait()
        return ModelResponse([AssistantMessage("done")])

    agent = Agent(
        ScriptedModelClient(response_factory=respond), ToolRegistry(), ContextConfig()
    )
    queue = AgentRuntime(agent)
    await queue.start()
    _submission_id, first = await queue.enqueue_user_turn("first")
    _submission_id, second = await queue.enqueue_user_turn("second")
    await asyncio.wait_for(started.wait(), 1)
    closing = asyncio.create_task(queue.close())
    try:
        await asyncio.sleep(0)
        assert not queue.accepts_input
        assert not agent.accepts_input
        assert not agent.is_shutdown
        assert len(queue._enqueue_queue) == 1
        assert not await agent.maybe_invoke({"type": "exec_command_completed"})
        assert not await agent.maybe_invoke({"type": "clock_tick"})
        with pytest.raises(RuntimeError, match="shutting down"):
            await queue.enqueue_user_turn("too late")
        release.set()
        assert (await first).output_text == "done"
        assert (await second).output_text == "done"
        await closing
        assert agent.is_shutdown
    finally:
        release.set()
        await asyncio.gather(first, second, closing, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("start_sampling", [False, True])
async def test_close_agent_drains_active_and_queued_work(start_sampling):
    started = asyncio.Event()
    cleaned_up = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, call_count):
        started.set()
        await release.wait()
        cleaned_up.set()
        return ModelResponse([AssistantMessage("done")])

    client = ScriptedModelClient(response_factory=respond)
    manager = SubAgentManager()
    manager.set_runtime_builder(
        lambda model_name, effort, history, session_id: AgentRuntime(
            Agent(client, ToolRegistry(), ContextConfig())
        )
    )
    spawned = await manager.spawn_agent("first", None, None, False, None, None, ())
    agent_id = spawned["agent_id"]
    closing = None
    try:
        if start_sampling:
            await asyncio.wait_for(started.wait(), 1)
        await manager.send_input(agent_id, "also finish", False)
        closing = asyncio.create_task(manager.close_agent(agent_id))
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        await asyncio.wait_for(closing, 1)
        assert client.call_count == 2
        assert cleaned_up.is_set()
        assert (await manager.wait_agents([agent_id], 10))["status"] == {
            agent_id: "shutdown",
        }
        assert not manager._agents[agent_id].runtime.is_busy
    finally:
        release.set()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await manager.close_agent(agent_id)
