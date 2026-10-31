import asyncio

import pytest

from pycodex import (
    AssistantMessage, BaseTool, ContextConfig, ContextLengthExceeded,
    ModelResponse, ToolCall, ToolRegistry, ToolResult, UserMessage,
)
from pycodex.events import TokenCountEvent
from pycodex.runtime import SubmissionInterrupted
from pycodex.tools import ClockManager, ClockTool, ExecCommandTool, UnifiedExecManager
from pycodex.tools.runtime_services import AgentRuntimeEnvironment
from pycodex.tools.unified_exec_manager import UnifiedExecSession
from tests.fakes import ScriptedModelClient, make_agent, make_runtime, make_tools


@pytest.mark.asyncio
async def test_nonblocking_commands_share_serial_execution_and_drain_on_close():
    runtime = make_runtime(make_agent(ScriptedModelClient([]), ToolRegistry(), ContextConfig()))
    entered = asyncio.Event()
    release = asyncio.Event()
    completed = []

    async def command(argument):
        entered.set()
        await release.wait()
        completed.append(argument)
        return {"kind": "test_command", "value": argument}

    runtime.register_command("test_command", command)
    await runtime.start()
    first = await runtime.submit_input("/test_command first", wait_for_command=False)
    await entered.wait()
    second = await runtime.submit_input("/test_command second", wait_for_command=False)
    closing = asyncio.create_task(runtime.close())
    try:
        await asyncio.sleep(0)
        assert not first.future.done()
        assert not second.future.done()
        assert not closing.done()
        release.set()
        await asyncio.wait_for(closing, 2)
        assert (await first.future)["value"] == "first"
        assert (await second.future)["value"] == "second"
        assert completed == ["first", "second"]
    finally:
        release.set()
        await closing


@pytest.mark.asyncio
async def test_lightweight_state_excludes_history_and_preserves_queue_identity():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def response(prompt, count):
        entered.set()
        await release.wait()
        return ModelResponse([AssistantMessage("done")])

    runtime = make_runtime(make_agent(
        ScriptedModelClient(response_factory=response), ToolRegistry(), ContextConfig(),
        initial_history=(UserMessage("old"), AssistantMessage("previous")),
    ))
    await runtime.start()
    try:
        first = await runtime.submit_input("first")
        await entered.wait()
        second = await runtime.submit_input("/queue second")
        state = runtime.state_snapshot()
        assert "history" not in state
        assert "completed_history" not in state["active_turn"]
        assert state["queued_inputs"] == [{
            "submission_id": second.submission_id, "turn_id": second.submission_id,
            "queue": "enqueue", "prompts": ["second"],
        }]
        assert state["active_turn"]["submission_id"] == first.submission_id
    finally:
        release.set()
        await runtime.close()


def test_agent_and_tools_share_a_session_local_runtime():
    environment = AgentRuntimeEnvironment()
    tools = make_tools(environment)
    agent = make_agent(ScriptedModelClient([]), tools, ContextConfig())
    other = make_agent(ScriptedModelClient([]), ToolRegistry(), ContextConfig())
    assert tools.runtime_environment is environment
    assert agent.tool_registry.runtime_environment is environment
    assert other.tool_registry.runtime_environment is not environment
    assert other.tool_registry.runtime_environment.plan_store is not environment.plan_store


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_during", ["model", "tool"])
@pytest.mark.parametrize("with_steer", [False, True])
async def test_close_interrupts_pending_inputs_and_finishes_issued_tools(stop_during, with_steer):
    started = asyncio.Event()
    release = asyncio.Event()
    tool_finished = []

    async def respond(prompt, count):
        if stop_during == "model":
            started.set()
            await release.wait()
        return ModelResponse([ToolCall("work", "work", {})])

    class WorkTool(BaseTool):
        name = "work"
        description = "Finish an issued operation."
        exec_only = False

        async def run(self, context, args):
            if stop_during == "tool":
                started.set()
                await release.wait()
            tool_finished.append(True)
            return "done"

    model = ScriptedModelClient(response_factory=respond)
    tools = ToolRegistry()
    tools.register(WorkTool())
    runtime = make_runtime(make_agent(model, tools, ContextConfig()))
    await runtime.start()
    first_id, first = runtime.enqueue("first")
    second_id, second = runtime.enqueue("second")
    await asyncio.wait_for(started.wait(), 1)
    pending = [second]
    if with_steer:
        steer = await runtime.submit_input("replacement")
        _submission_id, triggered = runtime.trigger()
        pending.extend([steer.future, triggered])
    else:
        runtime.message("discard this mailbox input")
    closing = asyncio.create_task(runtime.close())
    try:
        await asyncio.sleep(0)
        assert not runtime.accepts_input
        assert not runtime.agent.is_shutdown
        assert runtime.is_busy
        assert not closing.done()
        assert not first.done()
        assert not runtime._enqueue_queue
        assert not runtime._steer_queue
        assert runtime._triggered_submission is None
        for future in pending:
            assert future.done()
            with pytest.raises(SubmissionInterrupted):
                await future
        assert not await runtime.maybe_invoke({"type": "clock_tick"})
        with pytest.raises(RuntimeError, match="shutting down"):
            runtime.enqueue("too late")
        with pytest.raises(RuntimeError, match="shutting down"):
            runtime.trigger()
        with pytest.raises(RuntimeError, match="shutting down"):
            await runtime.submit_input("too late")
        with pytest.raises(RuntimeError, match="shutting down"):
            runtime.message("late message")
        with pytest.raises(RuntimeError, match="resume"):
            await runtime.start()
        release.set()
        with pytest.raises(SubmissionInterrupted):
            await first
        assert first_id != second_id
        await closing
        assert tool_finished == [True]
        assert model.call_count == 1
        assert any(
            isinstance(item, ToolResult) and item.call_id == "work"
            for item in runtime.agent.history
        )
        assert runtime.agent.is_shutdown
        assert not runtime.is_busy
    finally:
        release.set()
        await asyncio.gather(first, closing, *pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_close_before_start_discards_work_and_resume_accepts_new_work():
    model = ScriptedModelClient([ModelResponse([AssistantMessage("resumed")])])
    runtime = make_runtime(make_agent(model, ToolRegistry(), ContextConfig()))
    _submission_id, queued = runtime.enqueue("discard")
    steer = await runtime.submit_input("discard too")
    await runtime.close()
    for future in (queued, steer.future):
        with pytest.raises(SubmissionInterrupted):
            await future
    assert model.call_count == 0
    assert not runtime.agent.history
    runtime.resume()
    await runtime.start()
    try:
        assert (await runtime.submit_user_turn("new work")).output_text == "resumed"
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_queue_busy_includes_direct_agent_work_until_it_finishes():
    started = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, count):
        started.set()
        await release.wait()
        return ModelResponse([AssistantMessage("done")])

    runtime = make_runtime(make_agent(
        ScriptedModelClient(response_factory=respond), ToolRegistry(), ContextConfig(),
    ))
    turn = asyncio.create_task(runtime.agent.run_turn(["direct turn"]))
    try:
        await asyncio.wait_for(started.wait(), 1)
        assert runtime.is_busy
        runtime.agent.shutdown()
        assert runtime.is_busy
        release.set()
        assert (await turn).output_text == "done"
        assert not runtime.is_busy
    finally:
        release.set()
        await asyncio.gather(turn, return_exceptions=True)
        await runtime.close()


@pytest.mark.asyncio
async def test_queue_busy_includes_pending_submissions_before_worker_starts():
    runtime = make_runtime(make_agent(
        ScriptedModelClient([ModelResponse([AssistantMessage("done")])]),
        ToolRegistry(), ContextConfig(),
    ))
    _submission_id, future = runtime.enqueue("pending")
    assert runtime.is_busy
    await runtime.close()
    with pytest.raises(SubmissionInterrupted):
        await future
    assert not runtime.is_busy


@pytest.mark.asyncio
async def test_idle_messages_wait_for_execution_and_pending_triggers_merge():
    model = ScriptedModelClient([
        ModelResponse([AssistantMessage("triggered")]),
        ModelResponse([AssistantMessage("queued")]),
    ])
    runtime = make_runtime(make_agent(model, ToolRegistry(), ContextConfig()))
    runtime.message("first message")
    assert not runtime.is_busy
    assert model.call_count == 0
    queued_id, queued = runtime.enqueue("queued prompt")
    first_id, first = runtime.trigger()
    runtime.message("second message")
    second_id, second = runtime.trigger()
    assert first_id == second_id
    assert queued_id != first_id
    await runtime.start()
    try:
        first_result, second_result, queued_result = await asyncio.gather(first, second, queued)
        assert first_result is second_result
        assert first_result.output_text == "triggered"
        assert queued_result.output_text == "queued"
        assert [
            item.text for item in model.prompts[0].input if isinstance(item, UserMessage)
        ] == ["first message", "second message"]
        assert [
            item.text for item in model.prompts[1].input if isinstance(item, UserMessage)
        ] == ["first message", "second message", "queued prompt"]
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_active_message_reuses_turn_and_trigger_does_not_queue_an_extra_turn():
    started = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, count):
        if count == 1:
            started.set()
            await release.wait()
        return ModelResponse([AssistantMessage("answer " + str(count))])

    model = ScriptedModelClient(response_factory=respond)
    runtime = make_runtime(make_agent(model, ToolRegistry(), ContextConfig()))
    events = []
    runtime.event_handler = events.append
    await runtime.start()
    _submission_id, result = runtime.enqueue("start")
    try:
        await started.wait()
        runtime.message("agent update")
        assert runtime.trigger() is None
        release.set()
        assert (await result).output_text == "answer 2"
        assert model.call_count == 2
        assert model.prompts[0].turn_id == model.prompts[1].turn_id
        assert [event.kind for event in events].count("turn_started") == 1
        assert not runtime.state_snapshot()["queued_inputs"]
        assert runtime.agent.history == (
            UserMessage("start"), AssistantMessage("answer 1"),
            UserMessage("agent update"), AssistantMessage("answer 2"),
        )
    finally:
        release.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_wait_activity_timeout_cleans_waiter_and_does_not_consume_mail():
    runtime = make_runtime(make_agent(ScriptedModelClient([]), ToolRegistry(), ContextConfig()))
    tasks = set(asyncio.all_tasks())
    assert await runtime.wait_for_activity(1) == {
        "message": "Wait timed out.", "timed_out": True,
    }
    assert set(asyncio.all_tasks()) == tasks
    waiter = asyncio.create_task(runtime.wait_for_activity(1000))
    await asyncio.sleep(0)
    runtime.message("mail")
    assert await waiter == {"message": "Wait completed.", "timed_out": False}
    assert await runtime.wait_for_activity(1) == {
        "message": "Wait completed.", "timed_out": False,
    }
    assert list(runtime._steer_queue) == [UserMessage("mail")]
    await runtime.close()


@pytest.mark.asyncio
async def test_stop_releases_waiting_tool_without_a_timeout():
    entered = asyncio.Event()

    class WaitingTool(BaseTool):
        name = "wait_for_mail"
        description = "Wait for runtime activity."

        async def run(self, context, args):
            entered.set()
            return await runtime.wait_for_activity(10000)

    tools = ToolRegistry()
    tools.register(WaitingTool())
    model = ScriptedModelClient([ModelResponse([ToolCall("wait", "wait_for_mail", {})])])
    runtime = make_runtime(make_agent(model, tools, ContextConfig()))
    await runtime.start()
    _submission_id, result = runtime.enqueue("wait")
    try:
        await entered.wait()
        runtime.stop_asap()
        with pytest.raises(SubmissionInterrupted):
            await asyncio.wait_for(result, 1)
        assert runtime.agent.history[-1].output == {
            "message": "Wait interrupted by new input.", "timed_out": False,
        }
        assert model.call_count == 1
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_runtime_owns_notification_and_subsequent_steer_execution(failed):
    started = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, count):
        if count == 1:
            started.set()
            await release.wait()
            return ModelResponse([AssistantMessage("first")])
        if failed:
            raise ValueError("follow-up failed")
        return ModelResponse([AssistantMessage("final")])

    runtime = make_runtime(make_agent(
        ScriptedModelClient(response_factory=respond), ToolRegistry(), ContextConfig(),
    ))
    invocation = asyncio.create_task(runtime.maybe_invoke({"type": "wake_up"}))
    try:
        await asyncio.wait_for(started.wait(), 1)
        assert not await runtime.maybe_invoke({"type": "ignored"})
        steered = await runtime.submit_input("steer")
        release.set()
        with pytest.raises(SubmissionInterrupted):
            await invocation
        if failed:
            with pytest.raises(ValueError, match="follow-up failed"):
                await steered.future
        else:
            assert (await steered.future).output_text == "final"
        assert not runtime.is_busy
    finally:
        release.set()
        await asyncio.gather(invocation, return_exceptions=True)
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["exec", "clock"])
@pytest.mark.parametrize("failed", [False, True])
async def test_background_notification_steer_reports_only_failures(source, failed):
    started = asyncio.Event()
    release = asyncio.Event()
    error = ValueError("background model failed")
    errors = []
    events = []
    tools = ToolRegistry()
    if source == "exec":
        class CompletedProcess:
            returncode = 0

            async def wait(self):
                return self.returncode

        manager = UnifiedExecManager()
        manager._sessions[1000] = UnifiedExecSession(
            1000, CompletedProcess(), 0.0, "completed command", False,
        )
        tools.register(ExecCommandTool(manager))
        tools.runtime_environment.register_resource("exec_command", manager)
        notification = manager._notify_when_session_completes(1000)
        tag = "exec_command_completed"
    else:
        manager = ClockManager(seconds_per_minute=0.001)
        manager.set_period(1)
        tools.register(ClockTool(manager))
        tools.runtime_environment.register_resource("clock", manager, observer=manager.handle_agent_event)
        notification = manager._wait_and_notify(manager._generation, 1)
        tag = "clock_tick"

    async def respond(prompt, count):
        if count == 1:
            started.set()
            await release.wait()
            if failed:
                raise error
            return ModelResponse([AssistantMessage("background answer")])
        return ModelResponse([AssistantMessage("steer answer")])

    model = ScriptedModelClient(response_factory=respond)
    runtime = make_runtime(make_agent(model, tools, ContextConfig()))
    runtime.event_handler = events.append
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    notification_task = asyncio.create_task(notification)
    try:
        await asyncio.wait_for(started.wait(), 1)
        steered = await runtime.submit_input("user steer")
        release.set()
        await asyncio.wait_for(notification_task, 1)
        assert (await asyncio.wait_for(steered.future, 1)).output_text == "steer answer"
        await runtime.close()
        assert model.call_count == 2
        assert runtime.agent.history[0].text.startswith("<" + tag + ">\n")
        terminal = [
            event.kind for event in events
            if event.kind in ("turn_completed", "turn_interrupted", "turn_failed")
        ]
        assert terminal == [
            "turn_failed" if failed else "turn_interrupted", "turn_completed",
        ]
        if failed:
            assert len(errors) == 1
            assert errors[0]["exception"] is error
        else:
            assert errors == []
        assert runtime.agent.history[-2:] == (
            UserMessage("user steer"), AssistantMessage("steer answer"),
        )
    finally:
        release.set()
        await asyncio.gather(notification_task, return_exceptions=True)
        await runtime.close()
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
@pytest.mark.parametrize("overflow", [False, True])
async def test_steer_during_compaction_restarts_execution_with_shared_turn_id(overflow):
    request_started = asyncio.Event()
    release_request = asyncio.Event()
    compact_started = asyncio.Event()
    release_compact = asyncio.Event()

    async def respond(prompt, count):
        if count == 1:
            request_started.set()
            await release_request.wait()
            if overflow:
                raise ContextLengthExceeded("input exceeds the context window")
            runtime.agent._last_total_usage_tokens = 100
            return ModelResponse([AssistantMessage("first answer")])
        if count == 2:
            compact_started.set()
            await release_compact.wait()
            return ModelResponse([AssistantMessage("summary")])
        return ModelResponse([AssistantMessage("final answer")])

    model = ScriptedModelClient(response_factory=respond)
    runtime = make_runtime(make_agent(
        model, ToolRegistry(), ContextConfig(model_auto_compact_token_limit=50),
    ))
    events = []
    runtime.event_handler = events.append
    await runtime.start()
    _submission_id, original = runtime.enqueue("original")
    futures = [original]
    try:
        await request_started.wait()
        first = await runtime.submit_input("steer one")
        futures.append(first.future)
        release_request.set()
        await asyncio.wait_for(compact_started.wait(), 1)
        second = await runtime.submit_input("steer two")
        futures.append(second.future)
        release_compact.set()
        result = await asyncio.wait_for(second.future, 1)
        with pytest.raises(SubmissionInterrupted):
            await original
        if overflow:
            assert first.submission_id == second.submission_id
            assert await first.future is result
        else:
            assert first.submission_id != second.submission_id
            with pytest.raises(SubmissionInterrupted):
                await first.future
        assert [event.kind for event in events].count("turn_started") == (2 if overflow else 3)
        assert result.output_text == "final answer"
        assert len(model.prompts) == 3
        assert all(prompt.turn_id == result.turn_id for prompt in model.prompts)
        assert [
            item.text for item in model.prompts[-1].input if isinstance(item, UserMessage)
        ][-2:] == ["steer one", "steer two"]
    finally:
        release_request.set()
        release_compact.set()
        await runtime.close()
        await asyncio.gather(*futures, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("overflow", [False, True])
async def test_steer_during_tool_follow_up_compact_preserves_completed_tools(overflow):
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
        reasoning_effort = None

        def __init__(self):
            self.prompts = []

        async def complete(self, prompt, event_handler):
            self.prompts.append(prompt)
            number = len(self.prompts)
            if number == 1:
                if not overflow:
                    event_handler(TokenCountEvent({"total_tokens": 100}))
                return ModelResponse([
                    AssistantMessage("old commentary"), ToolCall("count_call", "count", {}),
                ])
            if overflow and number == 2:
                raise ContextLengthExceeded("input exceeds the context window")
            if number == (3 if overflow else 2):
                compact_started.set()
                await release_compact.wait()
                return ModelResponse([AssistantMessage("summary")])
            return ModelResponse([])

    tools = ToolRegistry()
    tools.register(CountingTool())
    client = CompactingClient()
    runtime = make_runtime(make_agent(
        client, tools, ContextConfig(model_auto_compact_token_limit=50),
    ))
    events = []
    runtime.event_handler = events.append
    await runtime.start()
    _submission_id, original = runtime.enqueue("original")
    futures = [original]
    try:
        await asyncio.wait_for(compact_started.wait(), 1)
        first = await runtime.submit_input("steer one")
        second = await runtime.submit_input("steer two")
        futures.extend([first.future, second.future])
        assert first.submission_id == second.submission_id
        assert not original.done()
        release_compact.set()
        result = await asyncio.wait_for(second.future, 1)
        with pytest.raises(SubmissionInterrupted):
            await original
        assert await first.future is result
        assert result.output_text is None
        assert result.iterations == 1
        assert executions == [result.turn_id]
        assert len(client.prompts) == (4 if overflow else 3)
        assert all(prompt.turn_id == result.turn_id for prompt in client.prompts)
        assert [
            item.text for item in client.prompts[-1].input if isinstance(item, UserMessage)
        ][-2:] == ["steer one", "steer two"]
        interrupted = [event for event in events if event.kind == "turn_interrupted"]
        assert len(interrupted) == 1
        assert interrupted[0].output_text == "old commentary"
    finally:
        release_compact.set()
        await runtime.close()
        await asyncio.gather(*futures, return_exceptions=True)


@pytest.mark.asyncio
async def test_terminal_observer_can_trigger_a_followup_turn():
    model = ScriptedModelClient([
        ModelResponse([AssistantMessage("first")]),
        ModelResponse([AssistantMessage("second")]),
    ])
    runtime = make_runtime(make_agent(model, ToolRegistry(), ContextConfig()))
    receipts = []

    def observe(event):
        if event.kind == "turn_completed" and event.output_text == "first":
            runtime.message("after terminal event")
            receipts.append(runtime.trigger())

    runtime.event_handler = observe
    await runtime.start()
    try:
        runtime.stop_asap()
        _submission_id, first = runtime.trigger()
        assert (await first).output_text == "first"
        assert (await receipts[0][1]).output_text == "second"
        assert model.call_count == 2
    finally:
        await runtime.close()
