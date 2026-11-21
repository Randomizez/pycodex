import asyncio
import json
from dataclasses import FrozenInstanceError
import typing

import pytest

from pycodex.agent import TurnContext
from pycodex.context import ContextConfig
from pycodex.events import (
    AssistantDeltaEvent,
    EventDisplay,
    EventScope,
    ReasoningEvent,
    SessionStateEvent,
    StreamErrorEvent,
)
from pycodex.model import ResponsesModelClient, ResponsesProviderConfig
from pycodex.protocol import (
    AssistantMessage,
    ModelResponse,
    ReasoningItem,
    ToolCall,
    UserMessage,
)
from pycodex.tools import BaseTool, ToolRegistry
from pycodex.utils.event_helpers import (
    last_reasoning_text,
    reasoning_item_text,
)
from tests.fakes import ScriptedModelClient, make_agent, make_runtime


def test_reasoning_event_display_contract() -> "None":
    operations: "typing.List[typing.Tuple[str, str]]" = []
    display = EventDisplay(
        lambda text: operations.append(("log", text)),
        lambda text: operations.append(("status", text)),
        lambda text: operations.append(("prompt", text)),
        False,
    )
    event = ReasoningEvent("**Thinking**\n\nChecking files.")
    assert event.kind == "reasoning"
    assert event.visualize() == "**Thinking**\n\nChecking files."
    with pytest.raises(FrozenInstanceError):
        event.text = "mutated"  # type: ignore[misc]

    display.set_status("called grep_files")
    operations.clear()
    event.render(display)
    assert display.stream_buffer == ""
    assert operations == []

    # CLI attach must not copy active_turn.reasoning into display.stream_buffer.
    state_event = SessionStateEvent(
        "attach",
        {
            "title": "",
            "accepts_input": True,
            "context_window": 128000,
            "usage_tokens": None,
            "busy": True,
            "background_work_count": 0,
            "input_request": None,
            "active_turn": {
                "turn_id": "t1",
                "submission_id": "s1",
                "messages": (),
                "assistant_text": "",
                "reasoning": "**Thinking**\n\nStill working.",
                "model_errors": [],
            },
        },
    )
    state_event.render(display)
    assert display.stream_buffer == ""


def test_reasoning_stream_boundaries_and_throttling(monkeypatch):
    client = ResponsesModelClient(
        ResponsesProviderConfig(model="demo", provider_name="demo", base_url="https://example.com/v1", api_key_env=None)
    )
    monkeypatch.setattr("pycodex.model.monotonic", lambda: 1.0)

    def parse(*payloads):
        lines = []
        for payload in (*payloads, {"type": "response.completed", "response": {}}):
            lines.extend([("data: " + json.dumps(payload)).encode(), b""])
        events = []
        with EventScope.observe(events.append):
            response = client._parse_stream(lines)
        return [event.text for event in events if isinstance(event, ReasoningEvent)], response.items

    def raw(text, item_id="r1"):
        return {"type": "response.reasoning_text.delta", "delta": text, "item_id": item_id}

    def summary(text, index):
        return {"type": "response.reasoning_summary_text.delta", "delta": text, "summary_index": index, "item_id": "r1"}

    output = {"type": "response.output_text.delta", "delta": "Done."}
    texts, _ = parse(raw("First "), raw("thought."), raw("New item.", "r2"), output)
    assert texts == ["First ", "First thought.", "New item."]

    part = {"type": "response.reasoning_summary_part.added", "item_id": "r1", "summary_index": 0}
    texts, _ = parse(part, summary("Checking ", 0), summary("the repo.", 0), summary("Testing.", 1))
    assert texts == ["Checking ", "Checking the repo.", "Testing."]
    assert parse(raw("Full thinking."), summary("Short summary.", 0))[0] == ["Full thinking."]
    summary_done = {"type": "response.output_item.done", "item": {
        "type": "reasoning", "id": "r1", "summary": [{"type": "summary_text", "text": "Short summary."}],
    }}
    assert parse(raw("Full thinking."), summary_done)[0] == ["Full thinking."]

    deltas = [raw("tok{0} ".format(i)) for i in range(200)]
    full_text = "".join(delta["delta"] for delta in deltas)
    done = {"type": "response.output_item.done", "item": {
        "type": "reasoning", "id": "r1", "content": [{"type": "reasoning_text", "text": full_text}],
    }}
    assert parse(*deltas, done)[0] == ["tok0 ", full_text]
    texts, items = parse(done)
    assert texts == [full_text]
    assert isinstance(items[0], ReasoningItem)


def test_reasoning_item_text_and_last_reasoning_text_rules() -> "None":
    both_item = ReasoningItem(
        payload={
            "type": "reasoning",
            "content": [{"type": "reasoning_text", "text": "  Raw CoT body  "}],
            "summary": [{"type": "summary_text", "text": "Summary fallback"}],
        }
    )
    summary_only_item = ReasoningItem(
        payload={
            "type": "reasoning",
            "content": None,
            "summary": [
                {"type": "summary_text", "text": "**Initial**\n\nFirst summary.\n\n\n"},
                {"type": "summary_text", "text": "**Latest**\n\nSecond summary.\n\n\n"},
            ],
        }
    )
    opaque_signature_item = ReasoningItem(
        payload={
            "type": "reasoning",
            "summary": [],
            "provider_data": [{"provider": "gemini", "part": {"text": "", "thoughtSignature": "sig"}}],
        }
    )

    assert reasoning_item_text(both_item) == "  Raw CoT body  "
    assert reasoning_item_text(summary_only_item) == "**Latest**\n\nSecond summary.\n\n\n"
    assert reasoning_item_text(opaque_signature_item) == ""

    # Invisible wakeup prompt + reasoning item returns latest readable reasoning.
    assert (
        last_reasoning_text(
            [UserMessage("wakeup", visible=False), summary_only_item, opaque_signature_item]
        )
        == "**Latest**\n\nSecond summary.\n\n\n"
    )
    # Visible AssistantMessage clears fallback (even if followed by signature-only ReasoningItem).
    assert (
        last_reasoning_text(
            [summary_only_item, AssistantMessage("Visible final answer"), opaque_signature_item]
        )
        == ""
    )
    # Newer visible UserMessage clears fallback.
    assert (
        last_reasoning_text(
            [both_item, UserMessage("New user prompt", visible=True)]
        )
        == ""
    )
    # Compaction marker clears fallback.
    assert last_reasoning_text([both_item, {"type": "compacted", "id": "c1"}]) == ""


@pytest.mark.asyncio
async def test_turn_context_compaction_filter_and_runtime_stream_error_rollback(
    tmp_path,
) -> "None":
    seen = []
    turn = TurnContext(seen.append, turn_id="t1")
    turn.dispatch(ReasoningEvent("**Step 1**\n\nVisible."))
    assert len(seen) == 1
    assert isinstance(seen[0], ReasoningEvent)
    assert seen[0].turn_id == "t1"

    turn.compact_input = [UserMessage("compact", visible=False)]
    turn.dispatch(ReasoningEvent("**Compacting**\n\nHidden."))
    assert len(seen) == 1

    step2_uncommitted = asyncio.Event()
    step2_error_sent = asyncio.Event()
    finish_retry = asyncio.Event()
    release_step2 = asyncio.Event()

    class EchoTool(BaseTool):
        name = "echo_once"
        description = "Immediate echo"
        input_schema = {"type": "object", "properties": {}}

        async def run(self, context, args):
            return "ok"

    class RollbackClient(ScriptedModelClient):
        def __init__(self):
            super().__init__([])
            self.calls = 0

        async def complete(self, prompt):
            self.calls += 1
            if self.calls == 1:
                EventScope.emit(ReasoningEvent("**Committed**\n\nStep 1."))
                return ModelResponse(
                    [
                        ReasoningItem(
                            payload={
                                "type": "reasoning",
                                "summary": [
                                    {
                                        "type": "summary_text",
                                        "text": "**Committed**\n\nStep 1.",
                                    }
                                ],
                            }
                        ),
                        ToolCall(call_id="c1", name="echo_once", arguments={}),
                    ]
                )
            EventScope.emit(ReasoningEvent("**Uncommitted**\n\nStep 2."))
            step2_uncommitted.set()
            await release_step2.wait()
            EventScope.emit(StreamErrorEvent("retrying", 1, 3, 0.2, "timeout"))
            step2_error_sent.set()
            await finish_retry.wait()
            return ModelResponse([AssistantMessage("Done.")])

    registry = ToolRegistry()
    registry.register(EchoTool())
    agent = make_agent(RollbackClient(), registry, ContextConfig(cwd=tmp_path))
    runtime = make_runtime(agent)
    await runtime.start()
    try:
        receipt = await runtime.submit_input("go")
        await asyncio.wait_for(step2_uncommitted.wait(), 2)
        assert (
            runtime.state_snapshot()["active_turn"]["reasoning"]
            == "**Uncommitted**\n\nStep 2."
        )

        release_step2.set()
        await asyncio.wait_for(step2_error_sent.wait(), 2)
        assert (
            runtime.state_snapshot()["active_turn"]["reasoning"]
            == "**Committed**\n\nStep 1."
        )

        finish_retry.set()
        await receipt.future
    finally:
        release_step2.set()
        finish_retry.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_runtime_preserves_reasoning_across_tool_and_clears_on_assistant(
    tmp_path,
) -> "None":
    tool_entered = asyncio.Event()
    tool_release = asyncio.Event()
    step2_entered = asyncio.Event()
    step2_release = asyncio.Event()

    class SlowEchoTool(BaseTool):
        name = "slow_echo"
        description = "Wait then echo"
        input_schema = {"type": "object", "properties": {}}

        async def run(self, context, args):
            tool_entered.set()
            await tool_release.wait()
            return "ok"

    class TwoStepReasoningClient(ScriptedModelClient):
        def __init__(self):
            super().__init__([])
            self.calls = 0

        async def complete(self, prompt):
            self.calls += 1
            if self.calls == 1:
                EventScope.emit(
                    ReasoningEvent("**Calling tool**\n\nRunning slow_echo.")
                )
                return ModelResponse(
                    [
                        ReasoningItem(
                            payload={
                                "type": "reasoning",
                                "summary": [
                                    {
                                        "type": "summary_text",
                                        "text": "**Calling tool**\n\nRunning slow_echo.\n\n\n",
                                    }
                                ],
                            }
                        ),
                        ToolCall(call_id="c1", name="slow_echo", arguments={}),
                    ]
                )
            EventScope.emit(AssistantDeltaEvent("Final "))
            step2_entered.set()
            await step2_release.wait()
            EventScope.emit(AssistantDeltaEvent("answer."))
            return ModelResponse([AssistantMessage("Final answer.")])

    registry = ToolRegistry()
    registry.register(SlowEchoTool())
    agent = make_agent(TwoStepReasoningClient(), registry, ContextConfig(cwd=tmp_path))
    runtime = make_runtime(agent)
    await runtime.start()
    try:
        receipt = await runtime.submit_input("go")
        await asyncio.wait_for(tool_entered.wait(), 2)
        mid_tool = runtime.state_snapshot()
        assert mid_tool["active_turn"] is not None
        assert (
            mid_tool["active_turn"]["reasoning"]
            == "**Calling tool**\n\nRunning slow_echo.\n\n\n"
        )
        assert mid_tool["active_turn"]["assistant_text"] == ""

        tool_release.set()
        await asyncio.wait_for(step2_entered.wait(), 2)
        mid_reply = runtime.state_snapshot()
        assert mid_reply["active_turn"]["assistant_text"] == "Final "
        assert mid_reply["active_turn"]["reasoning"] == ""

        step2_release.set()
        await receipt.future
        done = runtime.snapshot()
        assert done["last_reasoning"] == ""
        assert done["history"][-1]["response"] == "Final answer."
    finally:
        tool_release.set()
        step2_release.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_runtime_suppresses_last_reasoning_when_errored(tmp_path) -> "None":
    class EchoTool(BaseTool):
        name = "echo_once"
        description = "Immediate echo"
        input_schema = {"type": "object", "properties": {}}

        async def run(self, context, args):
            return "ok"

    class ErroredAfterReasoningClient(ScriptedModelClient):
        def __init__(self):
            super().__init__([])
            self.calls = 0

        async def complete(self, prompt):
            self.calls += 1
            if self.calls == 1:
                EventScope.emit(ReasoningEvent("**Step 1**\n\nCommitted before error."))
                return ModelResponse(
                    [
                        ReasoningItem(
                            payload={
                                "type": "reasoning",
                                "summary": [
                                    {
                                        "type": "summary_text",
                                        "text": "**Step 1**\n\nCommitted before error.",
                                    }
                                ],
                            }
                        ),
                        ToolCall(call_id="c1", name="echo_once", arguments={}),
                    ]
                )
            EventScope.emit(ReasoningEvent("**Step 2**\n\nUncommitted before crash."))
            raise ValueError("boom")

    registry = ToolRegistry()
    registry.register(EchoTool())
    agent = make_agent(
        ErroredAfterReasoningClient(), registry, ContextConfig(cwd=tmp_path)
    )
    runtime = make_runtime(agent)
    await runtime.start()
    try:
        receipt = await runtime.submit_input("go")
        with pytest.raises(ValueError, match="boom"):
            await receipt.future
        assert runtime.snapshot()["last_reasoning"] == ""
    finally:
        await runtime.close()
