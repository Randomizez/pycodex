import asyncio

import pytest

from pycodex import (
    Agent, AgentRuntime, AssistantMessage, ContextConfig, ModelResponse,
    ToolCall, ToolRegistry,
)
from pycodex.tools import ClockManager, ClockTool
from tests.fakes import ScriptedModelClient
from workspace_server.app import WorkspaceInteractiveSession


def runtime(path=None, responses=(), minute=60):
    tools = ToolRegistry()
    clock = ClockManager(minute)
    tools.register(ClockTool(clock))
    agent = Agent(
        ScriptedModelClient(responses), tools, ContextConfig(),
        session_file_path=path,
    )
    return AgentRuntime(agent), clock


@pytest.mark.asyncio
async def test_web_restore_clock_shows_sleeping_and_wakes(tmp_path):
    path = tmp_path / "clock.jsonl"
    source, _ = runtime(path, [
        ModelResponse([ToolCall("clock", "clock", {"period_m": 5})]),
        ModelResponse([AssistantMessage("monitoring")]),
    ])
    await source.start()
    await source.submit_user_turn("monitor")
    await source.close()

    restored, clock = runtime(responses=[
        ModelResponse([ToolCall("cancel", "clock", {"period_m": None})]),
        ModelResponse([AssistantMessage("woke after restart")]),
    ], minute=0.005)
    session = await WorkspaceInteractiveSession(restored).start()
    try:
        await session.restore_from_rollout(str(path), title="overnight")
        snapshot = session.snapshot()
        assert snapshot["tool_states"]["clock"] == {"enabled": True, "period_m": 5}
        assert snapshot["spinner"] == "idle: sleeping"
        assert snapshot["title"] == "overnight"
        assert restored.snapshot()["background_work_count"] == 1
        await asyncio.wait_for(clock._timer_task, 1)
        snapshot = session.snapshot()
        assert snapshot["turns"][-1]["response"] == "woke after restart"
        assert snapshot["tool_states"]["clock"]["enabled"] is False
        assert not snapshot["running"]
    finally:
        await session.close()
