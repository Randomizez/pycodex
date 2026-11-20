import asyncio
import json
from pathlib import Path

import pytest

from pycodex import (
    AgentsManager,
    AssistantMessage,
    ContextConfig,
    ModelResponse,
    ToolCall,
    UserMessage,
)
from pycodex.tools import get_tools
from pycodex.events import EventScope, TokenCountEvent
from pycodex.utils import session_persist
from pycodex.utils.compactor import DEFAULT_COMPACT_PROMPT
from tests.fakes import ScriptedModelClient, make_agent, wait_for_runtime_idle


def make_runtime(tmp_path, responses):
    return AgentsManager(
        ScriptedModelClient(responses),
        ContextConfig(cwd=tmp_path, codex_home=tmp_path),
        toolset=("clock.set_period",),
    ).create_root_runtime()


def clock(runtime):
    return runtime.agent.tool_registry.runtime_environment.clock_manager


def state_path(agent):
    return Path(str(agent.session_file_path) + ".state.json")


def state(period):
    return {"version": 1, "managers": {"clock": {"period_m": period}}}


def set_clock_responses(period):
    return [
        ModelResponse([ToolCall("set_clock", "clock.set_period", {"period_m": period})]),
        ModelResponse([AssistantMessage("done")]),
    ]


@pytest.mark.asyncio
async def test_clock_state_survives_compaction_close_and_resume(tmp_path):
    runtime = make_runtime(
        tmp_path,
        set_clock_responses(3) + set_clock_responses(5),
    )
    saved_at_tool_completion = []

    def observe(event):
        if event.kind == "tool_completed":
            saved_at_tool_completion.append(json.loads(state_path(runtime.agent).read_text()))

    runtime.event_handler = observe
    await runtime.start()
    try:
        await runtime.submit_user_turn("set clock")
        assert clock(runtime).armed
        assert json.loads(state_path(runtime.agent).read_text()) == state(3)
        await runtime.agent.compact()
        assert saved_at_tool_completion == [state(3), state(3)]
        assert json.loads(state_path(runtime.agent).read_text()) == state(5)
        assert not any(isinstance(item, ToolCall) for item in runtime.agent.history)
        rollout = runtime.agent.session_file_path
    finally:
        await runtime.close()

    assert clock(runtime).dump_state() == {"period_m": 5}
    assert not clock(runtime).armed
    assert json.loads(state_path(runtime.agent).read_text()) == state(5)

    restored = make_runtime(tmp_path, set_clock_responses(None))
    restored.resume(rollout)
    assert clock(restored).enabled
    assert not clock(restored).armed
    assert restored.snapshot()["background_work_count"] == 0
    await restored.start()
    try:
        assert clock(restored).armed
        assert restored.snapshot()["background_work_count"] == 1
        await restored.submit_user_turn("cancel clock")
        assert not clock(restored).enabled
        assert restored.snapshot()["background_work_count"] == 0
        assert json.loads(state_path(restored.agent).read_text()) == state(None)
    finally:
        await restored.close()

    restored.resume()
    await restored.start()
    try:
        assert not clock(restored).armed
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_goal_clock_and_compaction_use_request_results_with_notifications_disabled(tmp_path):
    usage = {
        "input_tokens": 40, "input_tokens_details": {"cached_tokens": 30},
        "output_tokens": 20, "total_tokens": 60,
    }
    model = ScriptedModelClient([
        ModelResponse([ToolCall("clock", "clock.set_period", {"period_m": 5})], usage),
        ModelResponse([AssistantMessage("automatic summary")], usage),
        ModelResponse([AssistantMessage("answer")], {"output_tokens": 3, "total_tokens": 9}),
        ModelResponse([AssistantMessage("manual summary")], usage),
    ])
    runtime = AgentsManager(
        model, ContextConfig(cwd=tmp_path, codex_home=tmp_path, model_auto_compact_token_limit=50),
        toolset=("clock.set_period",),
    ).create_root_runtime()
    agent = runtime.agent
    goal = agent.tool_registry.runtime_environment.goal_manager
    goal.create(agent.session_id, "finish the task")
    agent.event_handler = runtime._publish = lambda event: None
    runtime.message("pending mail")
    try:
        result = await agent.run_turn([UserMessage("start")])
        assert result.output_text == "answer"
        assert runtime.title == "start"
        assert runtime.agent_status == {"completed": "answer"}
        assert model.prompts[0].input[-2:] == [UserMessage("start"), UserMessage("pending mail")]
        assert model.prompts[1].input[-1] == UserMessage(DEFAULT_COMPACT_PROMPT)
        assert agent.execution_snapshot().usage_tokens == 9
        assert clock(runtime).armed
        assert goal.take_continuation() is not None
        saved = json.loads(state_path(agent).read_text())["managers"]
        assert saved["clock"] == {"period_m": 5}
        assert saved["goal"]["tokensUsed"] == 33

        await agent.compact()
        assert agent.execution_snapshot().usage_tokens is None
        assert goal.take_continuation() is None
        assert clock(runtime).armed
        assert json.loads(state_path(agent).read_text())["managers"]["goal"]["tokensUsed"] == 63
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_runtime_continues_a_goal_from_returned_usage_when_stream_reports_other_counts(tmp_path):
    finished = asyncio.Event()

    class Client(ScriptedModelClient):
        async def complete(self, prompt):
            EventScope.emit(TokenCountEvent({"total_tokens": 999, "output_tokens": 999}))
            return await super().complete(prompt)

    def respond(prompt, count):
        if count == 1:
            return ModelResponse([AssistantMessage("first reply")], {"total_tokens": 8, "output_tokens": 8})
        if count == 2:
            return ModelResponse([ToolCall("done", "update_goal", {"status": "complete"})])
        finished.set()
        return ModelResponse([AssistantMessage("finished")])

    model = Client(response_factory=respond)
    runtime = AgentsManager(
        model, ContextConfig(cwd=tmp_path, codex_home=tmp_path, model_auto_compact_token_limit=10),
        toolset=("update_goal",),
    ).create_root_runtime()
    goal = runtime.agent.tool_registry.runtime_environment.goal_manager
    goal.create(runtime.agent.session_id, "check the fixture", token_budget=10)
    runtime.agent.event_handler = runtime._publish = lambda event: None
    await runtime.start()
    try:
        assert (await runtime.submit_user_turn("begin")).output_text == "first reply"
        await asyncio.wait_for(finished.wait(), 2)
        await wait_for_runtime_idle(runtime)
        assert model.call_count == 3
        assert "check the fixture" in model.prompts[1].input[-1].text
        assert goal.dump_state()["status"] == "complete"
        assert goal.dump_state()["tokensUsed"] == 8
        assert runtime.agent.execution_snapshot().usage_tokens == 8
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_resume_without_snapshot_clears_old_clock_without_replaying_history(tmp_path):
    runtime = make_runtime(tmp_path, set_clock_responses(5))
    await runtime.start()
    try:
        await runtime.submit_user_turn("set clock")
        assert runtime.snapshot()["background_work_count"] == 1
        original_state = state_path(runtime.agent)
        previous_timer = clock(runtime)._timer_task
        other_rollout = tmp_path / "other.jsonl"
        other_rollout.write_bytes(runtime.agent.session_file_path.read_bytes())

        runtime.resume(other_rollout)

        assert any(isinstance(item, ToolCall) for item in runtime.agent.history)
        assert not clock(runtime).enabled
        assert not clock(runtime).armed
        assert runtime.snapshot()["background_work_count"] == 0
        await asyncio.gather(previous_timer, return_exceptions=True)
        assert previous_timer.cancelled() or previous_timer.done()
        assert json.loads(original_state.read_text()) == state(5)
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [
    "{",
    json.dumps({"version": 2, "managers": {"clock": {"period_m": 1}}}),
    json.dumps(state(-1)),
])
async def test_invalid_snapshot_preserves_existing_session_and_timer(tmp_path, invalid):
    runtime = make_runtime(tmp_path, set_clock_responses(5))
    await runtime.start()
    try:
        await runtime.submit_user_turn("set clock")
        previous_path = runtime.agent.session_file_path
        previous_session = runtime.agent.session_id
        previous_history = runtime.agent.history
        previous_timer = clock(runtime)._timer_task
        other_rollout = tmp_path / "invalid.jsonl"
        other_rollout.write_bytes(previous_path.read_bytes())
        Path(str(other_rollout) + ".state.json").write_text(invalid, encoding="utf-8")

        with pytest.raises(ValueError):
            runtime.resume(other_rollout)

        assert runtime.agent.session_file_path == previous_path
        assert runtime.agent.session_id == previous_session
        assert runtime.agent.history == previous_history
        assert clock(runtime).dump_state() == {"period_m": 5}
        assert clock(runtime)._timer_task is previous_timer
        assert not previous_timer.done()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_shutdown_saves_manager_state_before_releasing_clock(tmp_path):
    runtime = make_runtime(tmp_path, [])
    runtime.agent._append_history([UserMessage("saved")])
    clock(runtime).set_period(3)
    await runtime.start()
    assert clock(runtime).armed

    await runtime.close()

    assert json.loads(state_path(runtime.agent).read_text()) == state(3)
    assert not clock(runtime).armed
    assert clock(runtime)._notify_hook is None
    runtime.resume()
    await runtime.start()
    try:
        assert clock(runtime).armed
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_failed_snapshot_replace_keeps_previous_file_and_closes_resources(
    tmp_path, monkeypatch,
):
    runtime = make_runtime(tmp_path, set_clock_responses(5))
    await runtime.start()
    await runtime.submit_user_turn("set clock")
    path = state_path(runtime.agent)
    previous = path.read_bytes()
    clock(runtime).set_period(7)
    clock(runtime).arm_after_reply()
    timer = clock(runtime)._timer_task

    def fail_replace(source, destination):
        raise OSError("synthetic snapshot write failure")

    monkeypatch.setattr(session_persist.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic snapshot write failure"):
        await runtime.close()
    await asyncio.gather(timer, return_exceptions=True)

    assert path.read_bytes() == previous
    assert list(path.parent.glob(path.name + ".*")) == []
    assert runtime.agent.is_shutdown
    assert not clock(runtime).armed
    assert clock(runtime)._notify_hook is None


@pytest.mark.asyncio
async def test_unchanged_environment_does_not_rewrite_snapshot(tmp_path, monkeypatch):
    runtime = make_runtime(tmp_path, set_clock_responses(5) + set_clock_responses(5))
    replaced = []
    replace = session_persist.os.replace

    def track_replace(source, destination):
        replaced.append(destination)
        replace(source, destination)

    monkeypatch.setattr(session_persist.os, "replace", track_replace)
    await runtime.start()
    try:
        await runtime.submit_user_turn("set clock")
        await runtime.submit_user_turn("keep clock")
    finally:
        await runtime.close()

    assert replaced == [str(state_path(runtime.agent))]


@pytest.mark.asyncio
async def test_child_clock_state_remains_in_memory(tmp_path):
    root = make_runtime(tmp_path, set_clock_responses(5))
    manager = root.agents_manager
    try:
        path = await manager.spawn_agent(
            root.agent.agent_id, "worker", "set clock", fork_turns="none",
        )
        child = manager.get_runtime(manager._resolve_agent_id(root.agent.agent_id, path))
        await wait_for_runtime_idle(child)
        assert clock(child).enabled
        assert child.agent.session_file_path is None
        assert not clock(root).enabled
    finally:
        await root.close()

    assert list(tmp_path.rglob("*.state.json")) == []
    assert list(tmp_path.rglob("rollout-*.jsonl")) == []


def test_unused_or_unrecorded_agent_does_not_create_snapshot(tmp_path):
    manager = AgentsManager(ScriptedModelClient([]), ContextConfig(codex_home=tmp_path))
    for path in (None, tmp_path / "unused" / "rollout.jsonl"):
        tools = get_tools(manager, toolset=("clock.set_period",))
        agent = make_agent(
            ScriptedModelClient([]), tools, ContextConfig(codex_home=tmp_path),
            session_file_path=path,
        )
        tools.runtime_environment.clock_manager.set_period(5)
        agent.shutdown()

    assert list(tmp_path.rglob("*")) == []
