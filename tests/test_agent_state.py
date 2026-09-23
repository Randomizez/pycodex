import asyncio
import json
from pathlib import Path
from uuid import UUID

import pytest

from pycodex import (
    Agent,
    AssistantMessage,
    BaseTool,
    ContextConfig,
    ContextLengthExceeded,
    ContextManager,
    ModelResponse,
    ToolCall,
    ToolRegistry,
    ToolResult,
    TurnInterrupted,
    UserMessage,
)
from pycodex.events import TokenCountEvent, TurnStartedEvent
from pycodex.utils.session_persist import list_resumable_sessions
from tests.fakes import ScriptedModelClient


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_write", [1, 2])
async def test_history_write_failure_is_visible_before_mutation(
    tmp_path, monkeypatch, failed_write
):
    model = ScriptedModelClient([ModelResponse([AssistantMessage("answer")])])
    events = []
    agent = Agent(
        model,
        ToolRegistry(),
        ContextConfig(),
        session_file_path=tmp_path / "rollout.jsonl",
        event_handler=events.append,
    )
    recorder = agent._rollout_recorder
    append = recorder.append_history_items
    writes = []

    def append_items(items, initial_history=()):
        writes.append(tuple(items))
        if len(writes) == failed_write:
            raise OSError("synthetic write failure")
        append(items, initial_history)

    monkeypatch.setattr(recorder, "append_history_items", append_items)
    with pytest.raises(OSError, match="synthetic write failure"):
        await agent.run_turn(["prompt"])

    assert not agent.is_running
    assert events[-1].kind == "turn_failed"
    assert agent.history == (() if failed_write == 1 else (UserMessage("prompt"),))
    assert model.call_count == failed_write - 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,persisted", [("model", False), ("checkpoint", True)])
async def test_failed_compact_preserves_history_and_rollout(
    tmp_path,
    monkeypatch,
    failure,
    persisted,
):
    history = (
        UserMessage("old prompt"),
        ToolCall("call_old", "echo", {}),
        ToolResult("call_old", "echo", "large output"),
    )

    def fail_checkpoint(items, initial_history=()):
        raise OSError("synthetic failure")

    def response_factory(prompt, call_count):
        if call_count == 1:
            raise ContextLengthExceeded("maximum context length exceeded")
        assert not any(
            isinstance(item, (ToolCall, ToolResult)) for item in prompt.input
        )
        if failure == "model":
            raise RuntimeError("synthetic failure")
        return ModelResponse([AssistantMessage("summary")])

    agent = Agent(
        ScriptedModelClient(response_factory=response_factory),
        ToolRegistry(),
        ContextConfig(),
        initial_history=history,
        session_file_path=tmp_path / "rollout.jsonl",
    )
    recorder = agent._rollout_recorder
    if persisted:
        recorder.append_history_items(history)
    original_rollout = agent.session_file_path.read_bytes() if persisted else None
    if failure == "checkpoint":
        monkeypatch.setattr(recorder, "append_compacted_history", fail_checkpoint)
    agent._last_total_usage_tokens = 100

    with pytest.raises((RuntimeError, OSError), match="synthetic failure"):
        await agent.compact()

    assert agent.history == history
    if persisted:
        assert recorder.rollout_path.read_bytes() == original_rollout
    else:
        assert not recorder.rollout_path.exists()
    assert agent._last_total_usage_tokens == 100
    assert not agent.is_running


@pytest.mark.asyncio
async def test_manual_compact_resets_usage_before_next_turn():
    model = ScriptedModelClient(
        [
            ModelResponse([AssistantMessage("summary")]),
            ModelResponse([AssistantMessage("next answer")]),
        ]
    )
    agent = Agent(
        model,
        ToolRegistry(),
        ContextConfig(model_auto_compact_token_limit=10),
        initial_history=(UserMessage("old prompt"),),
    )
    agent._last_total_usage_tokens = 100

    result = await agent.compact()

    assert agent.history == result.history
    assert agent._last_total_usage_tokens is None
    assert (await agent.run_turn(["continue"])).output_text == "next answer"
    assert model.call_count == 2


@pytest.mark.asyncio
async def test_manual_compact_occupies_agent_until_finished():
    started = asyncio.Event()
    release = asyncio.Event()

    async def response_factory(prompt, call_count):
        started.set()
        await release.wait()
        return ModelResponse([AssistantMessage("summary")])

    agent = Agent(
        ScriptedModelClient(response_factory=response_factory),
        ToolRegistry(),
        ContextConfig(),
        initial_history=(UserMessage("prompt"),),
    )
    task = asyncio.create_task(agent.compact())
    try:
        await asyncio.wait_for(started.wait(), 1)
        assert agent.is_running
        assert not await agent.maybe_invoke({"type": "exec_command_completed"})
        with pytest.raises(RuntimeError, match="while agent is running"):
            await agent.compact()
        release.set()
        await task
        await asyncio.wait_for(agent.wait_until_idle(), 1)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_compact_rejects_missing_summary_without_replacing_history():
    agent = Agent(
        ScriptedModelClient([ModelResponse([])]),
        ToolRegistry(),
        ContextConfig(),
        initial_history=(UserMessage("prompt"),),
    )
    with pytest.raises(ValueError, match="no assistant summary"):
        await agent.compact()
    assert agent.history == (UserMessage("prompt"),)


def test_model_response_rejects_invalid_internal_items():
    with pytest.raises(TypeError, match="invalid model output item"):
        ModelResponse([object()])


@pytest.mark.asyncio
async def test_agent_does_not_classify_arbitrary_exception_text():
    def response_factory(prompt, call_count):
        raise RuntimeError("maximum context length exceeded in custom code")

    model = ScriptedModelClient(response_factory=response_factory)
    agent = Agent(model, ToolRegistry(), ContextConfig())
    with pytest.raises(RuntimeError, match="custom code"):
        await agent.run_turn(["prompt"])
    assert model.call_count == 1


@pytest.mark.asyncio
async def test_tool_follow_up_does_not_depend_on_tool_name():
    class NoticeTool(BaseTool):
        name = "notice"
        description = "Returns a follow-up."

        async def run(self, context, args):
            return "tool output"

        def follow_up_messages(self, output):
            return (UserMessage("follow up: " + output),)

    tools = ToolRegistry()
    tools.register(NoticeTool())
    model = ScriptedModelClient(
        [
            ModelResponse([ToolCall("notice_call", "notice", {})]),
            ModelResponse([AssistantMessage("done")]),
        ]
    )
    result = await Agent(model, tools, ContextConfig()).run_turn(["prompt"])
    assert isinstance(result.history[-3], ToolResult)
    assert result.history[-2] == UserMessage("follow up: tool output")


@pytest.mark.asyncio
async def test_follow_up_failure_keeps_completed_tool_result():
    class NoticeTool(BaseTool):
        name = "notice"
        description = "Fails while building the follow-up."

        async def run(self, context, args):
            return "completed output"

        def follow_up_messages(self, output):
            raise ValueError("synthetic follow-up failure")

    tools = ToolRegistry()
    tools.register(NoticeTool())
    agent = Agent(
        ScriptedModelClient([ModelResponse([ToolCall("notice_call", "notice", {})])]),
        tools,
        ContextConfig(),
    )
    with pytest.raises(ValueError, match="follow-up failure"):
        await agent.run_turn(["prompt"])
    assert agent.history[-1] == ToolResult("notice_call", "notice", "completed output")


@pytest.mark.asyncio
async def test_failed_tool_commit_waits_for_remaining_batch(tmp_path, monkeypatch):
    blocking_started = asyncio.Event()
    blocking_finished = asyncio.Event()
    failed = asyncio.Event()
    release = asyncio.Event()

    class BlockingTool(BaseTool):
        name = "blocking"
        description = "Waits until released."

        async def run(self, context, args):
            blocking_started.set()
            await release.wait()
            blocking_finished.set()
            return "completed"

    class FastTool(BaseTool):
        name = "fast"
        description = "Completes once the other tool starts."

        async def run(self, context, args):
            await blocking_started.wait()
            return "fast output"

    tools = ToolRegistry()
    tools.register(BlockingTool())
    tools.register(FastTool())
    agent = Agent(
        ScriptedModelClient(
            [
                ModelResponse(
                    [
                        ToolCall("call_block", "blocking", {}),
                        ToolCall("call_fast", "fast", {}),
                    ]
                )
            ]
        ),
        tools,
        ContextConfig(),
        session_file_path=tmp_path / "rollout.jsonl",
    )
    recorder = agent._rollout_recorder
    append = recorder.append_history_items

    def append_items(items, initial_history=()):
        if any(isinstance(item, ToolResult) and item.name == "fast" for item in items):
            failed.set()
            raise OSError("tool commit failed")
        append(items, initial_history)

    monkeypatch.setattr(recorder, "append_history_items", append_items)
    turn = asyncio.create_task(agent.run_turn(["prompt"]))
    try:
        await asyncio.wait_for(failed.wait(), 1)
        assert agent.is_running
        assert not blocking_finished.is_set()
        release.set()
        with pytest.raises(OSError, match="tool commit failed"):
            await turn
    finally:
        release.set()
        await asyncio.gather(turn, return_exceptions=True)
    assert blocking_finished.is_set()
    assert not agent.is_running
    results = {
        item.call_id: item for item in agent.history if isinstance(item, ToolResult)
    }
    assert "call_fast" not in results
    assert results["call_block"].output == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [False, True])
async def test_shutdown_allows_tool_batch_to_complete_and_commit(tmp_path, parallel):
    completed = asyncio.Event()
    blocked = asyncio.Event()
    release = asyncio.Event()

    class CompleteTool(BaseTool):
        name = "complete"
        description = "Completes without waiting."
        supports_parallel = parallel

        async def run(self, context, args):
            completed.set()
            return "completed output"

    class BlockTool(BaseTool):
        name = "block"
        description = "Waits until released."
        supports_parallel = parallel

        async def run(self, context, args):
            await completed.wait()
            blocked.set()
            await release.wait()
            return "released output"

    tools = ToolRegistry()
    tools.register(CompleteTool())
    tools.register(BlockTool())
    client = ScriptedModelClient(
        [
            ModelResponse(
                [
                    ToolCall("done", "complete", {}),
                    ToolCall("blocked", "block", {}),
                    ToolCall("later", "block", {}),
                ]
            ),
            ModelResponse([AssistantMessage("continued")]),
        ]
    )
    agent = Agent(
        client, tools, ContextConfig(), session_file_path=tmp_path / "rollout.jsonl"
    )
    task = asyncio.create_task(agent.run_turn(["start"]))
    try:
        await asyncio.wait_for(blocked.wait(), 1)
        assert any(
            isinstance(item, ToolResult) and item.call_id == "done"
            for item in agent.history
        )
        agent.shutdown()
        assert agent.is_running
        release.set()
        assert (await task).output_text == "continued"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    results = {
        item.call_id: item for item in agent.history if isinstance(item, ToolResult)
    }
    assert results["done"].output == "completed output"
    assert not results["done"].is_error
    assert results["blocked"].output == "released output"
    assert results["later"].output == "released output"
    assert not any(result.is_error for result in results.values())
    assert "completed output" in agent.session_file_path.read_text(encoding="utf-8")
    assert not agent.is_running


def test_agent_requires_context_config_instead_of_manager():
    client = ScriptedModelClient([])
    with pytest.raises(TypeError, match="context_config"):
        Agent(client, ToolRegistry())
    with pytest.raises(TypeError):
        Agent(client, ToolRegistry(), None)
    with pytest.raises(TypeError):
        Agent(client, ToolRegistry(), ContextManager(ContextConfig()))


def test_agent_builds_independent_contexts_from_explicit_config(tmp_path):
    config = ContextConfig(
        model="config-model",
        base_instructions_override="session rules",
        include_permissions_instructions=False,
        include_skills_instructions=False,
        extra_contextual_user_messages=("workspace context",),
        cwd=tmp_path,
    )
    first = Agent(ScriptedModelClient([], model="first"), ToolRegistry(), config)
    second = Agent(ScriptedModelClient([], model="second"), ToolRegistry(), config)

    assert first.context_manager is not second.context_manager
    assert first.context_manager.cwd == tmp_path
    assert first.context_manager.resolve_base_instructions() == "session rules"
    assert first.context_manager._config.model == "first"
    assert second.context_manager._config.model == "second"
    assert config.model == "config-model"
    prompt = first.context_manager.build_prompt([], [], True)
    assert any("workspace context" in str(item.content_items) for item in prompt.input)
    first.context_manager.get_turn_metadata("first-turn")
    assert second.context_manager._workspace_metadata_turn_id is None


def test_runtime_subscribes_to_agent_without_injecting_a_back_reference():
    from pycodex import AgentRuntime

    client = ScriptedModelClient([])
    tools = ToolRegistry()
    events = []
    agent = Agent(client, tools, ContextConfig(), event_handler=events.append)
    assert vars(agent)["model_client"] is client
    assert vars(agent)["tool_registry"] is tools
    assert vars(agent)["context_manager"] is agent.context_manager
    assert not agent.session_file_path.exists()
    assert not hasattr(agent, "submission_queue")
    for name in (
        "set_event_handler",
        "set_rollout_recorder",
        "bind_submission_queue",
        "runtime_environment",
        "rollout_recorder",
        "restore_session",
        "reopen",
        "replace_history",
    ):
        assert not hasattr(agent, name)

    queue = AgentRuntime(agent)
    queue.event_handler = events.append
    assert vars(queue)["agent"] is agent
    assert not hasattr(queue, "set_event_handler")
    assert not hasattr(agent, "submission_queue")
    assert not hasattr(agent, "runtime")
    agent._emit(TurnStartedEvent("direct", ()))
    assert events[-1].kind == "turn_started"


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [False, True])
async def test_bare_agent_stop_asap_finishes_issued_tools_and_allows_next_turn(
    parallel,
):
    started = asyncio.Event()
    release = asyncio.Event()
    completed = []

    class BlockingTool(BaseTool):
        name = "block"
        description = "Waits without cancellation."
        supports_parallel = True

        async def run(self, context, args):
            label = args["label"]
            if label == "first":
                started.set()
                await release.wait()
            completed.append(label)
            return label

    tools = ToolRegistry()
    tools.register(BlockingTool())
    client = ScriptedModelClient(
        [
            ModelResponse(
                [
                    ToolCall("first_call", "block", {"label": "first"}),
                    ToolCall("second_call", "block", {"label": "second"}),
                ]
            ),
            ModelResponse([AssistantMessage("continued")]),
        ]
    )
    events = []
    agent = Agent(
        client,
        tools,
        ContextConfig(),
        parallel_tool_calls=parallel,
        event_handler=events.append,
    )
    turn = asyncio.create_task(agent.run_turn(["work"]))
    try:
        await asyncio.wait_for(started.wait(), 1)
        agent.stop_asap()
        await asyncio.sleep(0)
        assert not turn.done()
        assert agent.is_running
        release.set()
        with pytest.raises(TurnInterrupted):
            await asyncio.wait_for(turn, 1)
        assert not turn.cancelled()
        assert not agent.is_running
        assert sorted(completed) == ["first", "second"]
        assert client.call_count == 1
        assert (
            {item.call_id for item in agent.history if isinstance(item, ToolCall)}
            == {item.call_id for item in agent.history if isinstance(item, ToolResult)}
            == {"first_call", "second_call"}
        )
        assert events[-1].kind == "turn_interrupted"
        assert not any(event.kind == "turn_failed" for event in events)
        agent.stop_asap()
        result = await agent.run_turn(["continue"])
        assert result.output_text == "continued"
        assert client.call_count == 2
    finally:
        release.set()
        await asyncio.gather(turn, return_exceptions=True)
        agent.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["pre_turn", "mid_turn", "context_length_exceeded"])
async def test_stop_during_compaction_prevents_next_sample_without_advancing_iteration(
    phase,
):
    events = []

    class CountingTool(BaseTool):
        name = "count"
        description = "Produces a tool follow-up."

        async def run(self, context, args):
            return "completed"

    class CompactingClient:
        model = "test"

        def __init__(self):
            self.call_count = 0

        async def complete(self, prompt, event_handler):
            self.call_count += 1
            if phase != "pre_turn" and self.call_count == 1:
                if phase == "context_length_exceeded":
                    raise ContextLengthExceeded("input exceeds the context window")
                event_handler(TokenCountEvent({"total_tokens": 100}))
                return ModelResponse([ToolCall("count_call", "count", {})])
            assert self.call_count == (1 if phase == "pre_turn" else 2)
            agent.stop_asap()
            return ModelResponse([AssistantMessage("summary")])

    tools = ToolRegistry()
    tools.register(CountingTool())
    client = CompactingClient()
    agent = Agent(
        client,
        tools,
        ContextConfig(model_auto_compact_token_limit=50),
        initial_history=(UserMessage("old prompt"),),
        event_handler=events.append,
    )
    if phase == "pre_turn":
        agent._last_total_usage_tokens = 100
    try:
        with pytest.raises(TurnInterrupted):
            await agent.run_turn(["new prompt"])
        assert not agent.is_running
        assert client.call_count == (1 if phase == "pre_turn" else 2)
        assert [
            event.iteration for event in events if event.kind == "model_called"
        ] == ([] if phase == "pre_turn" else [1])
        assert [
            event.phase for event in events if event.kind == "auto_compact_completed"
        ] == [phase]
        assert events[-1].kind == "turn_interrupted"
        assert events[-1].iteration == (0 if phase == "pre_turn" else 1)
        assert not any(event.kind == "turn_failed" for event in events)
        if phase == "pre_turn":
            assert agent.history[-1] == UserMessage("new prompt")
    finally:
        agent.shutdown()


@pytest.mark.asyncio
async def test_agent_lazily_creates_and_owns_new_rollout(tmp_path):
    home = tmp_path / "configured-home"
    client = ScriptedModelClient(
        [
            ModelResponse([AssistantMessage("first answer")]),
            ModelResponse([AssistantMessage("second answer")]),
        ]
    )
    client._session_id = "provider-created-id"
    config = ContextConfig(codex_home=home, base_instructions_override="session rules")
    initial_history = (
        UserMessage("old prompt"),
        ToolCall("old_call", "echo", {}),
        ToolResult("old_call", "echo", "old result"),
        AssistantMessage("old answer"),
    )
    agent = Agent(
        client,
        ToolRegistry(),
        config,
        session_file_path=None,
        initial_history=initial_history,
    )

    path = agent.session_file_path
    session_id = agent.session_id
    assert UUID(agent.session_id).version == 7
    assert client._session_id == agent.session_id
    assert home / "sessions" in path.parents
    assert not path.exists()
    assert not path.parent.exists()
    assert list_resumable_sessions(home) == ()
    agent._append_history(())
    assert not path.parent.exists()

    await agent.run_turn(["first prompt"])
    first_bytes = path.read_bytes()
    await agent.run_turn(["second prompt"])

    assert agent.session_id == session_id
    assert agent.session_file_path == path
    assert path.read_bytes().startswith(first_bytes)
    entries = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert sum(entry["type"] == "session_meta" for entry in entries) == 1
    assert entries[0]["type"] == "session_meta"
    assert entries[0]["payload"]["id"] == agent.session_id
    assert entries[0]["payload"]["base_instructions"] == {"text": "session rules"}
    assert [
        entry["payload"] for entry in entries if entry["type"] == "response_item"
    ] == [item.serialize() for item in agent.history]
    assert [
        entry["payload"]["message"] for entry in entries if entry["type"] == "event_msg"
    ] == ["old prompt", "first prompt", "second prompt"]
    assert len(list_resumable_sessions(home)) == 1
    other = Agent(ScriptedModelClient([]), ToolRegistry(), config)
    assert other.session_id != agent.session_id
    assert other.session_file_path != agent.session_file_path
    assert not other.session_file_path.exists()


@pytest.mark.asyncio
async def test_empty_compaction_does_not_create_rollout():
    agent = Agent(ScriptedModelClient([]), ToolRegistry(), ContextConfig())
    assert await agent.compact() is None
    agent.shutdown()
    assert not agent.session_file_path.exists()


def test_new_session_does_not_overwrite_existing_file(tmp_path):
    path = tmp_path / "existing.jsonl"
    path.write_text("existing session", encoding="utf-8")
    client = ScriptedModelClient([])
    client._session_id = "unchanged"

    with pytest.raises(FileExistsError):
        Agent(client, ToolRegistry(), ContextConfig(), session_file_path=path)

    assert path.read_text(encoding="utf-8") == "existing session"
    assert client._session_id == "unchanged"


@pytest.mark.asyncio
async def test_new_session_creation_race_does_not_overwrite_file(tmp_path):
    path = tmp_path / "rollout.jsonl"
    client = ScriptedModelClient([])
    agent = Agent(
        client,
        ToolRegistry(),
        ContextConfig(),
        initial_history=(UserMessage("seed"),),
        session_file_path=path,
    )
    path.write_text("created by another writer", encoding="utf-8")

    with pytest.raises(FileExistsError):
        await agent.run_turn(["prompt"])

    assert path.read_text(encoding="utf-8") == "created by another writer"
    assert agent.history == (UserMessage("seed"),)
    assert client.call_count == 0
    assert not agent.is_running


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_write", [1, 3])
async def test_first_write_failure_does_not_mark_rollout_initialized(
    tmp_path,
    monkeypatch,
    failed_write,
):
    agent = Agent(
        ScriptedModelClient([]),
        ToolRegistry(),
        ContextConfig(),
        initial_history=(UserMessage("seed"),),
        session_file_path=tmp_path / "rollout.jsonl",
    )
    open_file = Path.open

    class FailingWriter:
        def __init__(self, handle):
            self.handle = handle
            self.writes = 0

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def write(self, text):
            self.writes += 1
            if self.writes == failed_write:
                raise OSError("synthetic first write failure")
            return self.handle.write(text)

        def flush(self):
            self.handle.flush()

    def failing_open(path, mode="r", **kwargs):
        handle = open_file(path, mode, **kwargs)
        return (
            FailingWriter(handle)
            if path == agent.session_file_path and mode == "x"
            else handle
        )

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError, match="synthetic first write failure"):
        await agent.run_turn(["prompt"])

    assert agent.history == (UserMessage("seed"),)
    assert agent._rollout_recorder._session_meta is not None
    assert agent.model_client.call_count == 0
    contents = agent.session_file_path.read_bytes()
    with pytest.raises(FileExistsError):
        await agent.run_turn(["retry"])
    assert agent.session_file_path.read_bytes() == contents
    assert agent.history == (UserMessage("seed"),)


def test_resume_expands_home_and_reopens_closed_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    source = Agent(
        ScriptedModelClient([ModelResponse([AssistantMessage("saved answer")])]),
        ToolRegistry(),
        ContextConfig(),
        session_file_path="~/sessions/会话.jsonl",
    )
    source.ask("恢复 — ≤ 测试")
    agent = Agent(ScriptedModelClient([]), ToolRegistry(), ContextConfig())
    agent.shutdown()

    resumed = agent.resume("~/sessions/会话.jsonl")

    assert agent.session_file_path == tmp_path / "sessions" / "会话.jsonl"
    assert agent.session_id == source.session_id
    assert agent.history == source.history
    assert resumed["turns"] == (("恢复 — ≤ 测试", "saved answer"),)
    assert agent.accepts_input
    assert not agent.is_shutdown


@pytest.mark.asyncio
async def test_resume_compacted_custom_filename_keeps_metadata_and_checkpoint(tmp_path):
    source = Agent(
        ScriptedModelClient([ModelResponse([AssistantMessage("checkpoint summary")])]),
        ToolRegistry(),
        ContextConfig(),
        initial_history=(UserMessage("old prompt"), AssistantMessage("old answer")),
        session_file_path=tmp_path / "custom.jsonl",
    )
    initial_history = source.history
    assert not source.session_file_path.exists()
    await source.compact()
    original_bytes = source.session_file_path.read_bytes()
    entries = [json.loads(line) for line in original_bytes.decode("utf-8").splitlines()]
    assert entries[0]["type"] == "session_meta"
    assert entries[-1]["type"] == "compacted"
    assert [
        entry["payload"] for entry in entries if entry["type"] == "response_item"
    ] == [item.serialize() for item in initial_history]
    agent = Agent(ScriptedModelClient([]), ToolRegistry(), ContextConfig())

    resumed = agent.resume(source.session_file_path)

    assert agent.session_id == source.session_id
    assert agent.history == source.history
    assert len(agent.history) == 1
    assert resumed["turns"] == ()
    assert "checkpoint summary" in agent.history[0].text
    assert str(source.session_file_path) in agent.history[0].text
    assert source.session_file_path.read_bytes() == original_bytes


@pytest.mark.asyncio
async def test_fork_after_compact_keeps_summary_context_on_resume():
    model = ScriptedModelClient(
        [
            ModelResponse([AssistantMessage("summary")]),
            ModelResponse([AssistantMessage("continued answer")]),
            ModelResponse([AssistantMessage("forked answer")]),
        ]
    )
    source = Agent(
        model,
        ToolRegistry(),
        ContextConfig(),
        initial_history=(UserMessage("old prompt"), AssistantMessage("old answer")),
    )
    await source.compact()
    await source.run_turn([])
    source.fork()
    await source.run_turn(["new prompt"])
    agent = Agent(ScriptedModelClient([]), ToolRegistry(), ContextConfig())

    resumed = agent.resume(source.session_file_path)

    assert agent.history == source.history
    assert resumed["turns"] == (
        ("", "continued answer"),
        ("new prompt", "forked answer"),
    )
    assert source.history[0].serialize() in [
        item.serialize() for item in model.prompts[-1].input
    ]


def test_resume_accepts_multiline_records_and_incomplete_tail(tmp_path):
    path = tmp_path / "multiline.jsonl"
    entries = (
        {"type": "session_meta", "payload": {"id": "saved-session"}},
        {"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}},
        {
            "type": "response_item",
            "payload": AssistantMessage("saved answer").serialize(),
        },
    )
    path.write_text(
        "\n".join(json.dumps(entry, indent=2) for entry in entries)
        + '\n{"unfinished":',
        encoding="utf-8",
    )
    original_bytes = path.read_bytes()
    agent = Agent(ScriptedModelClient([]), ToolRegistry(), ContextConfig())

    agent.resume(path)

    assert agent.session_id == "saved-session"
    assert agent.history == (UserMessage("hello"), AssistantMessage("saved answer"))
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize("persisted", [False, True])
def test_resume_without_path_keeps_existing_session_and_history(persisted):
    client = ScriptedModelClient([])
    client._session_id = "provider-created-id"
    agent = Agent(
        client,
        ToolRegistry(),
        ContextConfig(),
        initial_history=(UserMessage("keep this"),),
    )
    path = agent.session_file_path
    if persisted:
        agent._append_history([AssistantMessage("existing answer")])
    contents = path.read_bytes() if persisted else None
    history = agent.history
    recorder = agent._rollout_recorder
    session_id = agent.session_id
    agent._last_total_usage_tokens = 123
    agent.shutdown()

    assert agent.resume() is None

    assert agent.accepts_input
    assert not agent.is_shutdown
    assert agent.session_id == session_id
    assert client._session_id == session_id
    assert agent.session_file_path == path
    assert agent.history == history
    assert agent._rollout_recorder is recorder
    assert agent._last_total_usage_tokens == 123
    if persisted:
        assert path.read_bytes() == contents
    else:
        assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("session_aware", [False, True])
async def test_resume_reads_history_and_continues_same_file(tmp_path, session_aware):
    client = ScriptedModelClient([ModelResponse([AssistantMessage("continued")])])
    if session_aware:
        client._session_id = "original-session"
    source = Agent(
        ScriptedModelClient([ModelResponse([AssistantMessage("previous answer")])]),
        ToolRegistry(),
        ContextConfig(),
        session_file_path=tmp_path / "restored.jsonl",
        session_id="restored-session",
    )
    await source.run_turn(["restored"])
    original_path = tmp_path / "original.jsonl"
    agent = Agent(
        client,
        ToolRegistry(),
        ContextConfig(),
        initial_history=(UserMessage("original"),),
        session_file_path=original_path,
        session_id="original-session",
    )
    agent._last_total_usage_tokens = 123
    assert not original_path.exists()
    restored_bytes = source.session_file_path.read_bytes()

    resumed = agent.resume(str(source.session_file_path))

    assert agent.history == (
        UserMessage("restored"),
        AssistantMessage("previous answer"),
    )
    assert resumed["history"] == agent.history
    assert agent.session_id == "restored-session"
    assert agent.session_file_path == source.session_file_path
    assert agent._last_total_usage_tokens is None
    assert source.session_file_path.read_bytes() == restored_bytes
    if session_aware:
        assert client._session_id == "restored-session"
    else:
        assert not hasattr(client, "_session_id")
    result = await agent.run_turn(["continue"])
    assert result.output_text == "continued"
    assert not original_path.exists()
    assert source.session_file_path.read_bytes().startswith(restored_bytes)
    assert "continued" in source.session_file_path.read_text(encoding="utf-8")
    assert (
        sum(
            json.loads(line)["type"] == "session_meta"
            for line in source.session_file_path.read_text(
                encoding="utf-8"
            ).splitlines()
        )
        == 1
    )
    restored_again = Agent(ScriptedModelClient([]), ToolRegistry(), ContextConfig())
    restored_again.resume(source.session_file_path)
    assert restored_again.history == agent.history


@pytest.mark.parametrize(
    "invalid_file,persisted,closed",
    [
        ("missing", False, False),
        ("empty", True, False),
        ("malformed", False, True),
        ("no_session_id", True, True),
    ],
)
def test_resume_failure_preserves_existing_session(
    tmp_path, invalid_file, persisted, closed
):
    client = ScriptedModelClient([])
    client._session_id = "original-session"
    agent = Agent(
        client,
        ToolRegistry(),
        ContextConfig(),
        initial_history=(UserMessage("original"),),
        session_file_path=tmp_path / "original.jsonl",
        session_id="original-session",
    )
    if persisted:
        agent._append_history([AssistantMessage("existing answer")])
    if closed:
        agent.shutdown()
    original_history = agent.history
    original_recorder = agent._rollout_recorder
    agent._last_total_usage_tokens = 123
    original_path = agent.session_file_path
    original_bytes = original_path.read_bytes() if persisted else None
    invalid_path = tmp_path / "invalid.jsonl"
    if invalid_file == "empty":
        invalid_path.write_text("", encoding="utf-8")
    elif invalid_file == "malformed":
        invalid_path.write_text("{", encoding="utf-8")
    elif invalid_file == "no_session_id":
        invalid_path.write_text(
            json.dumps(
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "user_message",
                        "message": "no session metadata",
                    },
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises((OSError, ValueError)):
        agent.resume(invalid_path)

    assert agent.history == original_history
    assert agent._rollout_recorder is original_recorder
    assert client._session_id == "original-session"
    assert agent.session_id == "original-session"
    assert agent.session_file_path == original_path
    if persisted:
        assert original_path.read_bytes() == original_bytes
    else:
        assert not original_path.exists()
    assert agent._last_total_usage_tokens == 123
    assert agent.accepts_input == (not closed)
    assert agent.is_shutdown == closed


@pytest.mark.parametrize("operation", ["fork", "resume"])
def test_recorder_factory_failure_preserves_existing_session(monkeypatch, operation):
    source = Agent(ScriptedModelClient([]), ToolRegistry(), ContextConfig())
    source._append_history([UserMessage("saved prompt")])
    source_bytes = source.session_file_path.read_bytes()
    client = ScriptedModelClient([])
    client._session_id = "original-session"
    history = (UserMessage("original"),)
    agent = Agent(
        client,
        ToolRegistry(),
        ContextConfig(),
        initial_history=history,
        session_id="original-session",
    )
    recorder = agent._rollout_recorder
    agent._last_total_usage_tokens = 123
    agent.shutdown()

    def fail_recording(*args, **kwargs):
        raise OSError("synthetic recorder failure")

    factory = "create" if operation == "fork" else "resume"
    monkeypatch.setattr(type(recorder), factory, fail_recording)

    with pytest.raises(OSError, match="synthetic recorder failure"):
        if operation == "fork":
            agent.fork()
        else:
            agent.resume(source.session_file_path)

    assert agent.history == history
    assert agent._rollout_recorder is recorder
    assert agent.session_id == client._session_id == "original-session"
    assert agent._last_total_usage_tokens == 123
    assert agent.is_shutdown
    assert not agent.accepts_input
    assert not agent.session_file_path.exists()
    assert source.session_file_path.read_bytes() == source_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["turn", "compact"])
async def test_resume_rejects_active_execution(operation):
    started = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, call_count):
        started.set()
        await release.wait()
        return ModelResponse([AssistantMessage("done")])

    client = ScriptedModelClient(response_factory=respond)
    client._session_id = "original-session"
    agent = Agent(
        client,
        ToolRegistry(),
        ContextConfig(),
        initial_history=(UserMessage("original"),),
        session_id="original-session",
    )
    coroutine = (
        agent.compact() if operation == "compact" else agent.run_turn(["continue"])
    )
    task = asyncio.create_task(coroutine)
    try:
        await asyncio.wait_for(started.wait(), 1)
        original_history = agent.history
        with pytest.raises(RuntimeError, match="agent is running"):
            agent.resume("missing-session.jsonl")
        with pytest.raises(RuntimeError, match="agent is running"):
            agent.resume()
        assert agent.history == original_history
        assert client._session_id == "original-session"
    finally:
        release.set()
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_kind", ["enqueue", "steer"])
async def test_runtime_resume_rejects_pending_submission(queue_kind):
    from pycodex import AgentRuntime

    client = ScriptedModelClient([ModelResponse([AssistantMessage("done")])])
    agent = Agent(
        client,
        ToolRegistry(),
        ContextConfig(),
        initial_history=(UserMessage("original"),),
    )
    queue = AgentRuntime(agent)
    _submission_id, future = await queue.enqueue_user_turn("pending", queue=queue_kind)
    try:
        assert not agent.is_running
        with pytest.raises(RuntimeError, match="running or queued"):
            queue.resume("missing-session.jsonl")
        with pytest.raises(RuntimeError, match="running or queued"):
            queue.resume()
        assert agent.history == (UserMessage("original"),)
    finally:
        await queue.close()
        await future


def test_model_switch_updates_provider_context_and_usage_together():
    from pycodex.model import ResponsesModelClient, ResponsesProviderConfig

    client = ResponsesModelClient(
        ResponsesProviderConfig(
            model="gpt-5.6",
            provider_name="demo",
            base_url="https://example.invalid/v1",
            api_key_env=None,
        )
    )
    agent = Agent(client, ToolRegistry(), ContextConfig(model="gpt-5.6"))
    context = agent.context_manager
    agent._last_total_usage_tokens = 123
    agent.set_model("step-3.6")
    expected = ContextManager(config=ContextConfig(model="step-3.6"))

    assert client.model == client._config.model == "step-3.6"
    assert agent.model_name == "step-3.6"
    assert (
        context.resolve_model_context_window()
        == expected.resolve_model_context_window()
    )
    assert context.resolve_base_instructions() == expected.resolve_base_instructions()
    assert agent._last_total_usage_tokens is None


def test_agent_model_name_uses_required_client_model():
    client = ScriptedModelClient([], model="first")
    agent = Agent(client, ToolRegistry(), ContextConfig())
    assert agent.model_name == "first"
    client.model = "second"
    assert agent.model_name == "second"
    del client.model
    with pytest.raises(AttributeError, match="model"):
        _model_name = agent.model_name


@pytest.mark.asyncio
async def test_invoke_awaits_serial_turn_and_skips_busy_notifications():
    started = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, call_count):
        started.set()
        await release.wait()
        return ModelResponse([AssistantMessage("done")])

    client = ScriptedModelClient(response_factory=respond)
    agent = Agent(client, ToolRegistry(), ContextConfig())
    invocation = asyncio.create_task(agent.maybe_invoke({"type": "wake_up"}))
    try:
        await asyncio.wait_for(started.wait(), 1)
        assert agent.is_running
        assert not invocation.done()
        assert not await agent.maybe_invoke({"type": "ignored"})
        release.set()
        assert await invocation
        assert not agent.is_running
        assert client.call_count == 1
        assert agent.history[-1] == AssistantMessage("done")
    finally:
        release.set()
        await asyncio.gather(invocation, return_exceptions=True)


def test_subagent_context_is_per_agent_and_uses_model_override(tmp_path):
    from pycodex.bootstrap import build_agent
    from pycodex.model import ResponsesModelClient

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'model = "gpt-5.6"\n'
        'model_provider = "demo"\n'
        "[model_providers.demo]\n"
        'base_url = "https://example.invalid/v1"\n',
        encoding="utf-8",
    )
    client = ResponsesModelClient.from_codex_config(config_path)
    agent = build_agent(
        client,
        config_path,
        extra_contextual_user_messages=iter(["shared board"]),
    )
    builder = agent.tool_registry.runtime_environment.subagent_manager._runtime_builder
    first = builder("step-3.6", None, (), "first-session").agent
    second = builder(None, None, (), "second-session").agent

    assert first.context_manager is not second.context_manager
    assert first.context_manager._extra_contextual_user_messages == ("shared board",)
    assert second.context_manager._extra_contextual_user_messages == ("shared board",)
    assert first.context_manager._config.model == first.model_name == "step-3.6"
    assert second.context_manager._config.model == second.model_name == "gpt-5.6"
    assert (
        first.tool_registry.runtime_environment
        is not second.tool_registry.runtime_environment
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_manual_compact_shares_agent_events_and_clock_lifecycle(failed):
    from pycodex.tools import ClockManager, ClockTool

    started = asyncio.Event()
    release = asyncio.Event()
    events = []

    async def respond(prompt, call_count):
        started.set()
        await release.wait()
        if failed:
            raise ValueError("compact failed")
        return ModelResponse([AssistantMessage("summary")])

    clock = ClockManager()
    clock.set_period(1)
    tools = ToolRegistry()
    tools.register(ClockTool(clock))
    initial_history = (UserMessage("old prompt"),)
    agent = Agent(
        ScriptedModelClient(response_factory=respond),
        tools,
        ContextConfig(),
        initial_history=initial_history,
        event_handler=events.append,
    )
    clock.arm_after_reply()
    old_timer = clock._timer_task
    task = asyncio.create_task(agent.compact())
    try:
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.gather(old_timer, return_exceptions=True)
        assert clock._timer_task is None
        assert not await agent.maybe_invoke({"type": "clock_tick"})
        release.set()
        if failed:
            with pytest.raises(ValueError, match="compact failed"):
                await task
            assert agent.history == initial_history
            assert clock._timer_task is None
            assert [event.kind for event in events] == [
                "compact_started",
                "compact_failed",
            ]
        else:
            release.set()
            result = await task
            assert agent.history == result.history
            assert clock._timer_task is not None
            assert [event.kind for event in events] == [
                "compact_started",
                "compact_completed",
            ]
            assert events[-1].background_work_count == 1
    finally:
        release.set()
        agent.shutdown()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_tool_observer_failure_does_not_abort_execution_or_commit():
    class EchoTool(BaseTool):
        name = "echo"
        description = "Returns its arguments."

        async def run(self, context, args):
            return args

    events = []
    observer_errors = []

    def observe(event):
        events.append(event.kind)
        if event.kind in {"tool_started", "tool_completed", "turn_completed"}:
            raise ValueError("observer unavailable")

    tools = ToolRegistry()
    tools.register(EchoTool())
    agent = Agent(
        ScriptedModelClient(
            [
                ModelResponse([ToolCall("call", "echo", {"value": "done"})]),
                ModelResponse([AssistantMessage("answer")]),
            ]
        ),
        tools,
        ContextConfig(),
        event_handler=observe,
    )
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: observer_errors.append(context))
    try:
        result = await agent.run_turn(["test"])
    finally:
        loop.set_exception_handler(previous_handler)
    assert result.output_text == "answer"
    assert "turn_failed" not in events
    assert len(observer_errors) == 3
    assert next(
        item for item in result.history if isinstance(item, ToolResult)
    ).output == {
        "value": "done",
    }


@pytest.mark.asyncio
async def test_follow_up_messages_do_not_split_tool_call_result_groups():
    class NoticeTool(BaseTool):
        name = "notice"
        description = "Produces a notification after tool results."
        supports_parallel = False

        async def run(self, context, args):
            assert not any(
                isinstance(item, UserMessage) and item.text.startswith("notice:")
                for item in context.history
            )
            return args["value"]

        def follow_up_messages(self, output):
            return (UserMessage("notice:" + output),)

    tools = ToolRegistry()
    tools.register(NoticeTool())
    model = ScriptedModelClient(
        [
            ModelResponse(
                [
                    ToolCall("first", "notice", {"value": "first"}),
                    ToolCall("second", "notice", {"value": "second"}),
                ]
            ),
            ModelResponse([AssistantMessage("done")]),
        ]
    )
    await Agent(model, tools, ContextConfig()).run_turn(["start"])
    assert model.prompts[-1].input[-4:] == [
        ToolResult("first", "notice", "first"),
        ToolResult("second", "notice", "second"),
        UserMessage("notice:first"),
        UserMessage("notice:second"),
    ]


@pytest.mark.asyncio
async def test_tool_observer_errors_do_not_change_result_and_invoke_errors_propagate():
    failures = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: failures.append(context))

    class BrokenObserverTool(BaseTool):
        name = "observer"
        description = "Observes Agent lifecycle."

        async def run(self, context, args):
            return None

        def handle_agent_event(self, event):
            raise ValueError("tool observer unavailable")

    async def fail(prompt, call_count):
        raise RuntimeError("background model failed")

    try:
        tools = ToolRegistry()
        tools.register(BrokenObserverTool())
        agent = Agent(
            ScriptedModelClient([ModelResponse([AssistantMessage("done")])]),
            tools,
            ContextConfig(),
        )
        assert (await agent.run_turn(["test"])).output_text == "done"
        assert len(failures) == 2
        assert all(
            item["message"].startswith("Agent event observer failed:")
            for item in failures
        )
        background = Agent(
            ScriptedModelClient(response_factory=fail), ToolRegistry(), ContextConfig()
        )
        with pytest.raises(RuntimeError, match="background model failed"):
            await background.maybe_invoke({"type": "test_notification", "value": 1})
        assert not background.is_running
        assert len(failures) == 2
    finally:
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_background_count_failure_does_not_hide_terminal_event():
    class BrokenCounterTool(BaseTool):
        name = "counter"
        description = "Reports background activity."

        async def run(self, context, args):
            return None

        def background_work_count(self, after_reply):
            raise ValueError("counter failed")

    tools = ToolRegistry()
    tools.register(BrokenCounterTool())
    events = []
    errors = []
    agent = Agent(
        ScriptedModelClient([ModelResponse([AssistantMessage("done")])]),
        tools,
        ContextConfig(),
        event_handler=events.append,
    )
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    try:
        assert (await agent.run_turn(["start"])).output_text == "done"
    finally:
        loop.set_exception_handler(previous_handler)
    assert events[-1].kind == "turn_completed"
    assert len(errors) == 1
    assert str(errors[0]["exception"]) == "counter failed"
    assert not agent.is_running


@pytest.mark.asyncio
async def test_context_recovery_is_local_to_each_sample_without_replaying_tools():
    executions = []
    sampling_states = []
    events = []

    class CountingTool(BaseTool):
        name = "count"
        description = "Records each execution."

        async def run(self, context, args):
            executions.append(context.turn_id)
            return "completed once"

    def respond(prompt, call_count):
        sampling_states.append(agent.is_running)
        if call_count in (1, 4):
            raise ContextLengthExceeded("input exceeds the context window")
        if call_count in (2, 5):
            return ModelResponse([AssistantMessage("summary")])
        if call_count == 3:
            return ModelResponse([ToolCall("count_call", "count", {})])
        assert call_count == 6
        return ModelResponse([AssistantMessage("done")])

    tools = ToolRegistry()
    tools.register(CountingTool())
    client = ScriptedModelClient(response_factory=respond)
    agent = Agent(client, tools, ContextConfig(), event_handler=events.append)

    result = await agent.run_turn(["start"], turn_id="one-turn")

    assert result.output_text == "done"
    assert result.iterations == 2
    assert executions == ["one-turn"]
    assert client.call_count == 6
    assert all(sampling_states)
    assert all(prompt.turn_id == "one-turn" for prompt in client.prompts)
    assert [event.iteration for event in events if event.kind == "model_called"] == [
        1,
        1,
        2,
        2,
    ]
    assert [event.kind for event in events].count("turn_started") == 1
    assert [event.kind for event in events].count("auto_compact_completed") == 2


@pytest.mark.asyncio
async def test_context_recovery_retries_only_once_and_reports_latest_usage():
    events = []

    def respond(prompt, call_count):
        if call_count == 2:
            return ModelResponse([AssistantMessage("summary")])
        assert call_count in (1, 3)
        raise ContextLengthExceeded("requested {0} tokens".format(call_count * 100))

    client = ScriptedModelClient(response_factory=respond)
    agent = Agent(client, ToolRegistry(), ContextConfig(), event_handler=events.append)

    with pytest.raises(ContextLengthExceeded, match="requested 300 tokens"):
        await agent.run_turn(["start"])

    assert client.call_count == 3
    assert not agent.is_running
    assert agent._last_total_usage_tokens == 300
    assert [event.iteration for event in events if event.kind == "model_called"] == [
        1,
        1,
    ]
    assert [event.kind for event in events].count("auto_compact_started") == 1
    assert events[-1].kind == "turn_failed"
    assert events[-1].iteration == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError, ValueError])
async def test_context_recovery_does_not_retry_failed_compaction(failure):
    events = []

    def respond(prompt, call_count):
        if call_count == 1:
            raise ContextLengthExceeded("input exceeds the context window")
        assert call_count == 2
        raise failure("compact stopped")

    client = ScriptedModelClient(response_factory=respond)
    agent = Agent(client, ToolRegistry(), ContextConfig(), event_handler=events.append)

    with pytest.raises(failure):
        await agent.run_turn(["start"])

    assert client.call_count == 2
    assert not agent.is_running
    assert agent.history == (UserMessage("start"),)
    assert [event.kind for event in events][-2:] == [
        "auto_compact_failed",
        "turn_failed",
    ]
