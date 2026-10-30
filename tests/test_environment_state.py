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
from pycodex.utils import session_persist
from tests.fakes import ScriptedModelClient, make_agent, wait_for_runtime_idle


def make_runtime(tmp_path, responses):
    return AgentsManager(
        ScriptedModelClient(responses),
        ContextConfig(cwd=tmp_path, codex_home=tmp_path),
        toolset=("clock",),
    ).create_root_runtime()


def clock(runtime):
    return runtime.agent.tool_registry.runtime_environment.clock_manager


def state_path(agent):
    return Path(str(agent.session_file_path) + ".state.json")


def state(period):
    return {"version": 1, "managers": {"clock": {"period_m": period}}}


def set_clock_responses(period):
    return [
        ModelResponse([ToolCall("set_clock", "clock", {"period_m": period})]),
        ModelResponse([AssistantMessage("done")]),
    ]


@pytest.mark.asyncio
async def test_clock_state_survives_compaction_close_and_resume(tmp_path):
    runtime = make_runtime(
        tmp_path,
        set_clock_responses(5) + [ModelResponse([AssistantMessage("summary")])],
    )
    await runtime.start()
    try:
        await runtime.submit_user_turn("set clock")
        assert clock(runtime).armed
        assert json.loads(state_path(runtime.agent).read_text()) == state(5)
        await runtime.agent.compact()
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
        tools = get_tools(manager, toolset=("clock",))
        agent = make_agent(
            ScriptedModelClient([]), tools, ContextConfig(codex_home=tmp_path),
            session_file_path=path,
        )
        tools.runtime_environment.clock_manager.set_period(5)
        agent.shutdown()

    assert list(tmp_path.rglob("*")) == []
