import asyncio
import json

import pytest

from pycodex import (
    Agent, AssistantMessage, ContextConfig, ModelResponse, ToolCall, ToolRegistry,
)
from pycodex.runtime import AgentRuntime
from pycodex.tools import ClockManager, ClockTool, ToolContext
from pycodex.utils.session_persist import load_resumed_session_path
from tests.fakes import ScriptedModelClient


def build_agent(path=None, responses=(), seconds_per_minute=60):
    manager = ClockManager(seconds_per_minute)
    registry = ToolRegistry()
    registry.register(ClockTool(manager))
    agent = Agent(
        ScriptedModelClient(responses), registry, ContextConfig(),
        session_file_path=path,
    )
    return agent, manager


async def saved_clock(path, period=5):
    agent, manager = build_agent(path, [
        ModelResponse([ToolCall("set", "clock", {"period_m": period})]),
        ModelResponse([AssistantMessage("ready")]),
    ])
    await agent.run_turn(["monitor overnight"])
    agent.shutdown()
    return agent, manager


@pytest.mark.asyncio
async def test_resume_clock_wakes_without_manual_input(tmp_path):
    path = tmp_path / "rollout.jsonl"
    await saved_clock(path)
    agent, clock = build_agent(
        responses=[
            ModelResponse([ToolCall("stop", "clock", {"period_m": None})]),
            ModelResponse([AssistantMessage("automatically resumed")]),
        ],
        seconds_per_minute=0.005,
    )
    events = []
    agent.event_handler = events.append
    agent.resume(path)
    timer = clock._timer_task
    assert clock.snapshot() == {"enabled": True, "period_m": 5.0}
    assert timer is not None
    runtime = AgentRuntime(agent)
    runtime.event_handler = events.append
    await runtime.start()
    assert clock._timer_task is timer
    try:
        await asyncio.wait_for(timer, 1)
        assert agent.model_client.call_count == 2
        assert agent.history[-1] == AssistantMessage("automatically resumed")
        assert any("<clock_tick>" in prompt.input[-1].text
                   for prompt in agent.model_client.prompts[:1])
        assert events[-1].kind == "turn_completed"
        assert events[-1].background_work_count == 0
        assert clock.snapshot() == {"enabled": False, "period_m": None}
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_cancel_is_restored_and_new_session_does_not_inherit_clock(tmp_path):
    enabled = tmp_path / "enabled.jsonl"
    disabled = tmp_path / "disabled.jsonl"
    await saved_clock(enabled)
    await saved_clock(disabled, None)
    agent, clock = build_agent()
    agent.resume(enabled)
    timer = clock._timer_task
    assert clock.enabled
    agent.resume(disabled)
    await asyncio.gather(timer, return_exceptions=True)
    assert not clock.enabled
    assert clock._timer_task is None
    assert load_resumed_session_path(disabled)["tool_states"]["clock"]["period_m"] is None
    agent.shutdown()


@pytest.mark.asyncio
async def test_clock_state_survives_compaction_and_fork(tmp_path):
    path = tmp_path / "source.jsonl"
    await saved_clock(path)
    agent, clock = build_agent(responses=[ModelResponse([AssistantMessage("summary")])])
    agent.resume(path)
    await agent.compact()
    assert load_resumed_session_path(path)["tool_states"]["clock"]["period_m"] == 5
    agent.fork()
    fork_path = agent.session_file_path
    agent.model_client = ScriptedModelClient([ModelResponse([AssistantMessage("fork ready")])])
    await agent.run_turn(["continue here"])
    assert load_resumed_session_path(fork_path)["tool_states"]["clock"]["period_m"] == 5
    assert fork_path != path
    agent.shutdown()


@pytest.mark.asyncio
async def test_clock_setting_is_durable_before_tool_result(tmp_path):
    path = tmp_path / "state-only.jsonl"
    agent, clock = build_agent(path, [ModelResponse([AssistantMessage("ready")])])
    await agent.run_turn(["start"])
    tool = agent.tool_registry.tools()[0]
    await tool.run(ToolContext("turn", agent.history, tmp_path), {"period_m": 5})
    # Simulates a process stop before the tool result is written.
    agent.shutdown()
    assert load_resumed_session_path(path)["tool_states"]["clock"]["period_m"] == 5


@pytest.mark.asyncio
async def test_failed_state_write_keeps_active_clock(tmp_path, monkeypatch):
    path = tmp_path / "write-failure.jsonl"
    agent, clock = await saved_clock(path)
    agent.resume()
    clock.set_period(5)
    clock.arm_after_reply()
    timer = clock._timer_task

    def fail(*args):
        raise OSError("cannot persist clock")

    monkeypatch.setattr(agent._rollout_recorder, "append_tool_state", fail)
    with pytest.raises(OSError, match="cannot persist clock"):
        await agent.tool_registry.tools()[0].run(
            ToolContext("turn", agent.history, tmp_path), {"period_m": None}
        )
    assert clock.enabled
    assert clock._timer_task is timer
    agent.shutdown()


@pytest.mark.asyncio
async def test_resume_pauses_clock_during_active_turn(tmp_path):
    path = tmp_path / "active.jsonl"
    await saved_clock(path)
    started = asyncio.Event()
    finish = asyncio.Event()

    async def response(prompt, count):
        started.set()
        await finish.wait()
        return ModelResponse([AssistantMessage("done")])

    agent, clock = build_agent(seconds_per_minute=0.005)
    agent.model_client = ScriptedModelClient(response_factory=response)
    agent.resume(path)
    turn = asyncio.create_task(agent.run_turn(["manual work"]))
    await started.wait()
    assert clock._timer_task is None
    await asyncio.sleep(0.04)
    assert agent.model_client.call_count == 1
    finish.set()
    await turn
    assert clock._timer_task is not None
    agent.shutdown()


@pytest.mark.parametrize("period", [5, None])
def test_legacy_clock_restores_before_compaction_and_ignores_failed_calls(tmp_path, period):
    path = tmp_path / "legacy.jsonl"
    records = [
        ("session_meta", {"id": "legacy-session"}),
        ("event_msg", {"type": "user_message", "message": "monitor"}),
        ("response_item", {"type": "function_call", "name": "clock", "call_id": "ok",
                           "arguments": json.dumps({"period_m": period})}),
        ("response_item", {"type": "function_call_output", "call_id": "ok",
                           "output": json.dumps({"enabled": period is not None, "period_m": period})}),
        ("compacted", {"message": "summary", "replacement_history": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "summary"}]},
        ]}),
        ("response_item", {"type": "function_call", "name": "clock", "call_id": "bad",
                           "arguments": '{"period_m":-5}'}),
        ("response_item", {"type": "function_call_output", "call_id": "bad",
                           "output": '{"error":"invalid period"}'}),
        ("response_item", {"type": "function_call", "name": "clock", "call_id": "unmatched",
                           "arguments": '{"period_m":1}'}),
    ]
    # Concatenated pretty JSON and an unfinished tail use the existing reader.
    path.write_text(
        "\n".join(json.dumps({"type": kind, "payload": payload}, indent=2)
                  for kind, payload in records) + '\n{"type":',
        encoding="utf-8",
    )
    restored = load_resumed_session_path(path)
    assert restored["tool_states"]["clock"] == {
        "enabled": period is not None, "period_m": period,
    }


def test_sync_resume_defers_timer_until_runtime_start(tmp_path):
    path = tmp_path / "sync.jsonl"
    asyncio.run(saved_clock(path))
    agent, clock = build_agent()
    agent.resume(path)
    assert clock.enabled
    assert clock._timer_task is None

    async def start():
        runtime = AgentRuntime(agent)
        await runtime.start()
        assert clock._timer_task is not None
        await runtime.close()

    asyncio.run(start())
