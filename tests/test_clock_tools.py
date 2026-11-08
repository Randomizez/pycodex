import asyncio
import re
from pathlib import Path

import pytest

from pycodex import AgentsManager, AssistantMessage, ContextConfig, ModelResponse
from pycodex.model import ResponsesModelClient
from pycodex.protocol import AgentMessage, ToolCall, ToolResult, UserMessage
from pycodex.runtime import SubmissionInterrupted
from pycodex.tools import ClockManager, SetPeriodTool, ToolContext, ToolRegistry
from tests.fakes import ScriptedModelClient, make_agent, make_runtime, make_tools


@pytest.mark.parametrize("code_mode_only", [False, True])
@pytest.mark.parametrize("lite", [False, True])
def test_clock_tools_share_a_direct_only_namespace(code_mode_only, lite):
    registry = make_tools()
    specs = registry.model_visible_specs(code_mode_only)
    namespaces = ResponsesModelClient._serialize_tools(specs, lite)
    clock = next(spec for spec in namespaces if spec.get("name") == "clock")
    assert [tool["name"] for tool in clock["tools"]] == ["set_period", "sleep"]
    assert not any(tool["tool_name"].startswith("clock.") for tool in registry.exec_tools())
    assert "clock" not in registry.names()
    assert "clock.curr_time" not in registry.names()


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [
    {}, {"duration_ms": 1, "extra": True},
    {"duration_ms": None}, {"duration_ms": True}, {"duration_ms": "1"},
    {"duration_ms": 1.5}, {"duration_ms": 0}, {"duration_ms": -1},
    {"duration_ms": 43200001},
])
async def test_sleep_rejects_invalid_arguments(tmp_path, arguments):
    registry = make_tools(toolset=("clock.sleep",))
    result = await registry.execute(
        ToolCall("invalid", "clock.sleep", arguments),
        ToolContext("test", (), tmp_path),
    )
    assert result.is_error
    assert "duration_ms" in result.output_text()


@pytest.mark.asyncio
async def test_sleep_timeout_returns_elapsed_time_without_pending_waiters(tmp_path):
    model = ScriptedModelClient([
        ModelResponse([ToolCall("sleep", "clock.sleep", {"duration_ms": 10})]),
        ModelResponse([AssistantMessage("done")]),
    ])
    runtime = AgentsManager(
        model, ContextConfig(cwd=tmp_path, codex_home=tmp_path),
        toolset=("clock.sleep",),
    ).create_root_runtime()
    await runtime.start()
    tasks = set(asyncio.all_tasks())
    try:
        assert (await runtime.submit_user_turn("wait")).output_text == "done"
        result = next(item for item in runtime.agent.history if isinstance(item, ToolResult))
        assert not result.is_error
        assert re.fullmatch(r"Wall time: \d+\.\d{4} seconds\nSleep completed\.", result.output)
        assert float(result.output.split()[2]) >= 0.005
        assert set(asyncio.all_tasks()) == tasks
        assert runtime.snapshot()["background_work_count"] == 0
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("wake", ["pending", "message", "agent_message", "steer", "stop", "close"])
async def test_sleep_wakes_and_records_result_before_processing_input(tmp_path, wake):
    entered = asyncio.Event()
    model = ScriptedModelClient([
        ModelResponse([ToolCall("sleep", "clock.sleep", {"duration_ms": 60000})]),
        ModelResponse([AssistantMessage("awake")]),
    ])
    runtime = AgentsManager(
        model, ContextConfig(cwd=tmp_path, codex_home=tmp_path),
        toolset=("clock.set_period", "clock.sleep"),
    ).create_root_runtime()

    def observe(event):
        if event.kind == "tool_started" and event.call.name == "clock.sleep":
            if wake == "pending":
                runtime.message("new input")
            entered.set()

    runtime.attach(observe, interactive=False)
    await runtime.start()
    _submission_id, future = runtime.enqueue(UserMessage("wait"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if wake != "pending":
            assert not future.done()
        if wake == "message":
            runtime.message("new input")
        elif wake == "agent_message":
            await runtime.agents_manager.send_message(
                runtime.agent.agent_id, runtime.agent.agent_id, "new input",
            )
        elif wake == "steer":
            steered = await runtime.submit_input("new input")
        elif wake == "stop":
            runtime.stop_asap()
        elif wake == "close":
            await asyncio.wait_for(runtime.close(), 1)

        if wake in ("steer", "stop", "close"):
            with pytest.raises(SubmissionInterrupted):
                await asyncio.wait_for(future, 1)
        else:
            assert (await asyncio.wait_for(future, 1)).output_text == "awake"
        if wake == "steer":
            assert (await asyncio.wait_for(steered.future, 1)).output_text == "awake"

        history = runtime.agent.history
        result_index = next(
            index for index, item in enumerate(history) if isinstance(item, ToolResult)
        )
        result = history[result_index]
        assert not result.is_error
        assert result.output.endswith("\nSleep interrupted by new input.")
        if wake in ("stop", "close"):
            assert model.call_count == 1
        else:
            message_index = next(
                index for index, item in enumerate(history)
                if isinstance(item, (UserMessage, AgentMessage)) and "new input" in item.text
            )
            assert result_index < message_index
            assert model.call_count == 2
        assert not runtime.agent.tool_registry.runtime_environment.clock_manager.enabled
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_clock_tool_sets_cancels_and_validates_period() -> "None":
    manager = ClockManager()
    registry = ToolRegistry()
    registry.register(SetPeriodTool(manager))
    context = ToolContext(turn_id="turn_clock", history=(), cwd=Path.cwd())

    enabled = await registry.execute(
        ToolCall(
            call_id="clock_set",
            name="clock.set_period",
            arguments={"period_m": 2},
        ),
        context,
    )
    assert enabled.output == {"enabled": True, "period_m": 2.0}
    assert enabled.is_error is False

    invalid = await registry.execute(
        ToolCall(
            call_id="clock_invalid",
            name="clock.set_period",
            arguments={"period_m": 0},
        ),
        context,
    )
    assert invalid.is_error is True
    assert "positive finite number or null" in str(invalid.output)

    disabled = await registry.execute(
        ToolCall(
            call_id="clock_cancel",
            name="clock.set_period",
            arguments={"period_m": None},
        ),
        context,
    )
    assert disabled.output == {"enabled": False, "period_m": None}
    assert disabled.is_error is False


@pytest.mark.asyncio
async def test_clock_wakes_agent_periodically_until_cancelled(monkeypatch) -> "None":
    monkeypatch.setattr(
        "pycodex.tools.clock_manager._current_time",
        lambda: "2026-08-07T12:34:56+08:00",
    )
    manager = ClockManager(seconds_per_minute=0.01)
    tools = ToolRegistry()
    tools.register(SetPeriodTool(manager))
    tools.runtime_environment.register_resource("clock", manager, observer=manager.handle_agent_event)
    model = ScriptedModelClient(
        [
            ModelResponse(
                items=[
                    ToolCall(
                        call_id="clock_set",
                        name="clock.set_period",
                        arguments={"period_m": 1},
                    )
                ]
            ),
            ModelResponse(items=[AssistantMessage(text="clock armed")]),
            ModelResponse(items=[AssistantMessage(text="first tick handled")]),
            ModelResponse(
                items=[
                    ToolCall(
                        call_id="clock_cancel",
                        name="clock.set_period",
                        arguments={"period_m": None},
                    )
                ]
            ),
            ModelResponse(items=[AssistantMessage(text="clock stopped")]),
        ]
    )
    agent = make_agent(model, tools, ContextConfig())
    runtime = make_runtime(agent)

    result = await agent.run_turn([UserMessage("start periodic work")])

    assert result.output_text == "clock armed"

    async def wait_for_second_tick() -> "None":
        while model.call_count < 5 or agent.is_running:
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait_for_second_tick(), timeout=1.0)
    clock_ticks = [
        item
        for item in agent.history
        if isinstance(item, UserMessage) and item.text.startswith("<clock_tick>\n")
    ]
    assert [item.text for item in clock_ticks] == [
        (
            "<clock_tick>\n"
            '{"period_m":1.0,"current_time":"2026-08-07T12:34:56+08:00"}\n'
            "</clock_tick>"
        ),
        (
            "<clock_tick>\n"
            '{"period_m":1.0,"current_time":"2026-08-07T12:34:56+08:00"}\n'
            "</clock_tick>"
        ),
    ]
    assert manager.snapshot() == {"enabled": False, "period_m": None}
    assert manager._timer_task is None
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_shutdown_cancels_clock_and_discards_queued_tools() -> "None":
    manager = ClockManager()
    manager.set_period(1)
    tools = ToolRegistry()
    clock = SetPeriodTool(manager)
    clock.exec_only = False
    tools.register(clock)
    tools.runtime_environment.register_resource("clock", manager, observer=manager.handle_agent_event)
    model = ScriptedModelClient(
        [
            ModelResponse(items=[AssistantMessage(text="done")]),
            ModelResponse(items=[ToolCall("clock", "clock.set_period", {"period_m": 1})]),
            ModelResponse(items=[AssistantMessage(text="done")]),
        ]
    )
    agent = make_agent(model, tools, ContextConfig())
    events = []
    runtime = make_runtime(agent)
    runtime.event_handler = events.append
    await runtime.start()
    try:
        result = await runtime.submit_user_turn("hello")
        assert result.output_text == "done"
        assert manager._timer_task is not None
        pending_timer = manager._timer_task
        completed_event = next(
            event for event in events if event.kind == "turn_completed"
        )
        assert completed_event.background_work_count == 1

        _submission_id, pending_turn = runtime.enqueue(UserMessage("set clock again"))
        await runtime.close()
        with pytest.raises(SubmissionInterrupted):
            await pending_turn
        await asyncio.gather(pending_timer, return_exceptions=True)

        assert manager.snapshot() == {"enabled": True, "period_m": 1.0}
        assert manager._timer_task is None
        assert model.call_count == 1
    finally:
        await runtime.close()
