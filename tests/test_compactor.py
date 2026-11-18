import json

import pytest

from pycodex import ContextConfig, ContextLengthExceeded, ResponsesModelClient, ResponsesProviderConfig
from pycodex.events import TokenCountEvent
from pycodex.protocol import (
    AssistantMessage,
    ContextMessage,
    ModelResponse,
    ToolCall,
    ToolResult,
    UserMessage,
)
from pycodex.utils.compactor import (
    DEFAULT_COMPACT_PROMPT,
    SUMMARY_PREFIX,
    compact,
    prune_oldest_tool_response,
)
from pycodex.utils.history import conversation_history_to_turns
from pycodex.utils.session_persist import (
    load_resumed_session_path,
)
from tests.fakes import ScriptedModelClient, make_agent, make_runtime, make_tools


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["manual", "auto", "overflow"])
@pytest.mark.parametrize("code_mode_only", [False, True])
@pytest.mark.parametrize("lite", [False, True])
async def test_compact_preserves_request_prefix_and_tools(tmp_path, phase, code_mode_only, lite):
    serializer = ResponsesModelClient(ResponsesProviderConfig(
        "fixture", "fixture", "https://example.invalid/v1", None,
        responses_lite_override=lite,
    ))

    def respond(prompt, call_count):
        if phase == "overflow" and call_count <= 2:
            raise ContextLengthExceeded("Your input exceeds the context window")
        is_compact = prompt.input[-1].text == DEFAULT_COMPACT_PROMPT
        return ModelResponse([AssistantMessage("summary" if is_compact else "answer")])

    class PayloadModelClient(ScriptedModelClient):
        def __init__(self):
            super().__init__(response_factory=respond, model="fixture")
            self.payloads = []

        async def complete(self, prompt, event_handler):
            self.payloads.append(serializer._build_payload(prompt))
            response = await super().complete(prompt, event_handler)
            if phase == "auto" and self.call_count == 1:
                event_handler(TokenCountEvent({"total_tokens": 12}))
            return response

    model = PayloadModelClient()
    agent = make_agent(
        model, make_tools(), ContextConfig(
            cwd=tmp_path, codex_home=tmp_path, code_mode_only=code_mode_only,
            model_auto_compact_token_limit=10 if phase == "auto" else None,
            include_skills_instructions=False, include_permissions_instructions=False,
        ),
        parallel_tool_calls=not code_mode_only,
        initial_history=(
            UserMessage("old request"),
            ToolCall("old", "exec_command", {"cmd": "inspect"}),
            ToolResult("old", "exec_command", "old output"),
            AssistantMessage("old answer"),
        ),
    )
    runtime = make_runtime(agent)
    try:
        await agent.run_turn([UserMessage("current request")])
        if phase == "manual":
            await agent.compact()
            await agent.run_turn([UserMessage("continue")])
        elif phase == "auto":
            await agent.run_turn([UserMessage("continue")])
    finally:
        await runtime.close()

    ordinary = model.payloads[0]
    prefix_length = ordinary["input"].index(UserMessage("old request").serialize())
    compact_payloads = [
        payload for prompt, payload in zip(model.prompts, model.payloads)
        if prompt.input[-1].text == DEFAULT_COMPACT_PROMPT
    ]
    assert len(compact_payloads) == (2 if phase == "overflow" else 1)
    assert compact_payloads[0]["input"][:len(ordinary["input"])] == ordinary["input"]
    for payload in model.payloads:
        assert payload["input"][:prefix_length] == ordinary["input"][:prefix_length]
        assert payload.get("tools") == ordinary.get("tools")
        assert payload["parallel_tool_calls"] == ordinary["parallel_tool_calls"]
        assert payload["prompt_cache_key"] == agent.session_id
        assert payload["tool_choice"] == ordinary["tool_choice"]
    if phase == "overflow":
        assert any(item["type"] == "function_call" for item in compact_payloads[0]["input"])
        assert not any(item["type"] == "function_call" for item in compact_payloads[1]["input"])


def test_compact_replaces_all_prior_context_with_language_preserving_summary():
    history = (
        UserMessage("real user"),
        AssistantMessage("old answer"),
        UserMessage(SUMMARY_PREFIX + "\nold summary"),
        UserMessage('<subagent_notification>{"status":"done"}</subagent_notification>'),
        UserMessage('<exec_command_completed>{"exit_code":0}</exec_command_completed>'),
        UserMessage(DEFAULT_COMPACT_PROMPT),
        AssistantMessage("new summary"),
    )
    assert compact(history) == (ContextMessage(SUMMARY_PREFIX + "\nnew summary"),)
    assert (
        compact(history)[0].serialize()
        == UserMessage(SUMMARY_PREFIX + "\nnew summary").serialize()
    )
    assert "Continue the current task directly" in SUMMARY_PREFIX
    for instruction in (
        "concise verbatim excerpts",
        "in their original language",
        "user's primary language",
    ):
        assert instruction in DEFAULT_COMPACT_PROMPT


def test_prune_oldest_tool_response_keeps_other_pairs_intact():
    history = (
        UserMessage("first"),
        ToolCall("old", "echo", {}),
        ToolResult("old", "echo", "large"),
        AssistantMessage("after first"),
        ToolCall("new", "echo", {}),
        ToolResult("new", "echo", "recent"),
    )
    assert prune_oldest_tool_response(history) == (history[0],) + history[3:]


def test_compact_summary_is_context_but_identical_user_input_is_visible():
    summary = compact([AssistantMessage("summary")])
    assert conversation_history_to_turns(summary) == ()
    turns = conversation_history_to_turns(
        summary + (AssistantMessage("continued answer"), UserMessage(summary[0].text))
    )
    assert [turn["messages"] for turn in turns] == [(), (UserMessage(summary[0].text),)]
    assert [turn["response"] for turn in turns] == ["continued answer", ""]


@pytest.mark.parametrize("has_summary_metadata", [False, True])
def test_resume_summary_checkpoint_keeps_follow_up_items(
    tmp_path, has_summary_metadata
):
    text = (
        "Upstream handoff with its own prefix"
        if has_summary_metadata
        else SUMMARY_PREFIX + "\nold local summary"
    )
    summary = UserMessage(text, id="summary-id").serialize()
    payload = {"replacement_history": [summary]}
    if has_summary_metadata:
        payload["message"] = text
    follow_up = (
        ToolCall("call", "echo", {}, raw_arguments="{}"),
        ToolResult("call", "echo", "output"),
        AssistantMessage("continued answer"),
    )
    entries = (
        [
            {"type": "session_meta", "payload": {"id": "saved-session"}},
            {"type": "compacted", "payload": payload},
        ]
        + [{"type": "response_item", "payload": item.serialize()} for item in follow_up]
        + [
            {"type": "event_msg", "payload": {"type": "user_message", "message": text}},
            {
                "type": "response_item",
                "payload": AssistantMessage("next answer").serialize(),
            },
        ]
    )
    path = tmp_path / "checkpoint.jsonl"
    path.write_text("\n".join(json.dumps(entry) for entry in entries), encoding="utf-8")

    resumed = load_resumed_session_path(path)

    assert resumed["history"] == (
        (ContextMessage(text, id="summary-id"),)
        + follow_up
        + (UserMessage(text), AssistantMessage("next answer"))
    )
    assert resumed["history"][0].serialize() == summary
    assert [turn["messages"] for turn in resumed["turns"]] == [(), (UserMessage(text),)]
    assert [turn["response"] for turn in resumed["turns"]] == ["continued answer", "next answer"]
