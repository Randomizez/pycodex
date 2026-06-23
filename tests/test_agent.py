
import asyncio

import pytest

from pycodex import (
    Agent,
    CliSubmissionQueue,
    AssistantMessage,
    BaseTool,
    ContextConfig,
    ContextManager,
    ModelResponse,
    ModelStreamEvent,
    ResponsesIncompleteError,
    ToolCall,
    ToolRegistry,
    ToolResult,
    UserMessage,
)
from pycodex.tools import ExecCommandTool, UnifiedExecManager
from pycodex.utils.compactor import DEFAULT_COMPACT_PROMPT, SUMMARY_PREFIX
from tests.fakes import ScriptedModelClient
import typing


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo text back."
    input_schema = {"type": "object"}

    async def run(self, context, args):
        del context
        return args["text"]


class SlowTool(BaseTool):
    description = "Slow tool."
    input_schema = {"type": "object"}
    supports_parallel = True

    def __init__(self, name: 'str') -> 'None':
        self.name = name

    async def run(self, context, args):
        del context, args
        await asyncio.sleep(0.05)
        return "done"


class CoordinatedParallelTool(BaseTool):
    description = "Tool that proves both calls entered before either finished."
    input_schema = {"type": "object"}
    supports_parallel = True

    def __init__(self, name: 'str', entered: 'typing.List[str]', both_started: 'asyncio.Event') -> 'None':
        self.name = name
        self._entered = entered
        self._both_started = both_started

    async def run(self, context, args):
        del context, args
        self._entered.append(self.name)
        if len(self._entered) == 2:
            self._both_started.set()
        await asyncio.wait_for(self._both_started.wait(), timeout=0.2)
        return "done"


class WaitAgentNotificationTool(BaseTool):
    name = "wait_agent"
    description = "Returns a completed sub-agent status."
    input_schema = {"type": "object"}

    async def run(self, context, args):
        del context, args
        return {
            "status": {
                "019d0000-0000-7000-8000-000000000000": {
                    "completed": "subagent done",
                }
            },
            "timed_out": False,
        }


class UsageModelClient:
    def __init__(
        self,
        responses: 'typing.Iterable[ModelResponse]',
        usage_by_call: 'typing.Union[typing.Dict[int, int], None]' = None,
    ) -> 'None':
        self._responses = iter(responses)
        self._usage_by_call = usage_by_call or {}
        self.prompts: 'typing.List[object]' = []
        self.call_count = 0

    async def complete(self, prompt, event_handler):
        self.prompts.append(prompt)
        self.call_count += 1
        total_tokens = self._usage_by_call.get(self.call_count)
        if total_tokens is not None:
            event_handler(
                ModelStreamEvent(
                    kind="token_count",
                    payload={"usage": {"total_tokens": total_tokens}},
                )
            )
        try:
            return next(self._responses)
        except StopIteration as exc:
            raise RuntimeError("usage model ran out of responses") from exc


def _auto_compact_context(limit: 'typing.Union[int, None]') -> 'ContextManager':
    return ContextManager(
        config=ContextConfig(model_auto_compact_token_limit=limit),
        include_permissions_instructions=False,
        include_skills_instructions=False,
    )


def _conversation_items(prompt) -> 'typing.List[typing.Union[UserMessage, AssistantMessage, ToolCall, ToolResult]]':
    return [
        item
        for item in prompt.input
        if isinstance(item, (UserMessage, AssistantMessage, ToolCall, ToolResult))
    ]


def _context_length_error_message(
    requested_tokens: 'int' = 264568,
    max_tokens: 'int' = 262144,
) -> 'str':
    return (
        "responses_server.stream_router.OutcommingChatError: outcomming chat "
        "request failed with status 400: {\"error\":{\"message\":\"This model's "
        f"maximum context length is {max_tokens} tokens. However, you requested "
        f"{requested_tokens} tokens ({requested_tokens} in the messages, 0 in "
        "the completion). Please reduce the length of the messages or "
        "completion.\",\"type\":\"context_length_exceeded\"}}"
    )


def test_agent_ask_runs_turn_from_sync_context() -> 'None':
    model = ScriptedModelClient(
        [ModelResponse(items=[AssistantMessage(text="sync answer")])]
    )
    agent = Agent(model, ToolRegistry())

    result = agent.ask("sync prompt")

    assert result.output_text == "sync answer"
    assert result.iterations == 1
    assert model.call_count == 1


@pytest.mark.asyncio
async def test_agent_runs_tool_then_returns_final_message() -> 'None':
    model = ScriptedModelClient(
        [
            ModelResponse(
                items=[
                    AssistantMessage(text="我先看一下。"),
                    ToolCall(call_id="call_1", name="echo", arguments={"text": "hello"}),
                ]
            ),
            ModelResponse(items=[AssistantMessage(text="工具返回了 hello")]),
        ]
    )

    tools = ToolRegistry()
    tools.register(EchoTool())

    agent = Agent(model, tools)
    result = await agent.run_turn(["请回声 hello"])

    assert result.output_text == "工具返回了 hello"
    assert result.iterations == 2
    assert [type(item).__name__ for item in result.history] == [
        "UserMessage",
        "AssistantMessage",
        "ToolCall",
        "ToolResult",
        "AssistantMessage",
    ]
    tool_result = next(item for item in result.history if isinstance(item, ToolResult))
    assert tool_result.output == "hello"


@pytest.mark.asyncio
async def test_parallel_tools_share_one_model_round() -> 'None':
    model = ScriptedModelClient(
        [
            ModelResponse(
                items=[
                    ToolCall(call_id="call_1", name="slow_a", arguments={}),
                    ToolCall(call_id="call_2", name="slow_b", arguments={}),
                ]
            ),
            ModelResponse(items=[AssistantMessage(text="两个工具都执行完了")]),
        ]
    )

    tools = ToolRegistry()
    entered: 'typing.List[str]' = []
    both_started = asyncio.Event()
    tools.register(CoordinatedParallelTool("slow_a", entered, both_started))
    tools.register(CoordinatedParallelTool("slow_b", entered, both_started))

    agent = Agent(model, tools)
    result = await agent.run_turn(["并行跑两个工具"])

    assert result.output_text == "两个工具都执行完了"
    assert entered == ["slow_a", "slow_b"] or entered == ["slow_b", "slow_a"]


@pytest.mark.asyncio
async def test_agent_default_has_no_fixed_iteration_cap() -> 'None':
    model = ScriptedModelClient(
        [
            *(
                ModelResponse(
                    items=[
                        ToolCall(
                            call_id=f"call_{index}",
                            name="echo",
                            arguments={"text": f"step-{index}"},
                        )
                    ]
                )
                for index in range(12)
            ),
            ModelResponse(items=[AssistantMessage(text="超过 12 轮后也收敛了")]),
        ]
    )

    tools = ToolRegistry()
    tools.register(EchoTool())

    agent = Agent(model, tools)
    result = await agent.run_turn(["连续调用工具直到结束"])

    assert result.output_text == "超过 12 轮后也收敛了"
    assert result.iterations == 13


@pytest.mark.asyncio
async def test_agent_auto_compacts_before_next_turn_when_usage_reaches_limit() -> 'None':
    model = UsageModelClient(
        [
            ModelResponse(items=[AssistantMessage(text="first answer")]),
            ModelResponse(items=[AssistantMessage(text="checkpoint summary")]),
            ModelResponse(items=[AssistantMessage(text="second answer")]),
        ],
        usage_by_call={1: 12},
    )
    events = []
    agent = Agent(
        model,
        ToolRegistry(),
        _auto_compact_context(10),
        event_handler=events.append,
    )

    first = await agent.run_turn(["first prompt"])
    second = await agent.run_turn(["second prompt"])

    assert first.output_text == "first answer"
    assert second.output_text == "second answer"
    assert model.call_count == 3

    compact_prompt_items = _conversation_items(model.prompts[1])
    assert [type(item).__name__ for item in compact_prompt_items] == [
        "UserMessage",
        "AssistantMessage",
        "UserMessage",
    ]
    assert compact_prompt_items[0].text == "first prompt"
    assert compact_prompt_items[1].text == "first answer"
    assert compact_prompt_items[2].text == DEFAULT_COMPACT_PROMPT

    second_prompt_items = _conversation_items(model.prompts[2])
    assert [type(item).__name__ for item in second_prompt_items] == [
        "UserMessage",
        "UserMessage",
        "UserMessage",
    ]
    assert second_prompt_items[0].text == "first prompt"
    assert second_prompt_items[1].text == f"{SUMMARY_PREFIX}\ncheckpoint summary"
    assert second_prompt_items[2].text == "second prompt"

    auto_events = [event for event in events if event.kind.startswith("auto_compact_")]
    assert [event.kind for event in auto_events] == [
        "auto_compact_started",
        "auto_compact_completed",
    ]
    assert auto_events[0].payload["phase"] == "pre_turn"
    assert auto_events[0].payload["total_tokens"] == 12
    assert auto_events[0].payload["token_limit"] == 10


@pytest.mark.asyncio
async def test_agent_auto_compacts_before_tool_follow_up_when_usage_reaches_limit() -> 'None':
    model = UsageModelClient(
        [
            ModelResponse(
                items=[
                    ToolCall(
                        call_id="call_1",
                        name="echo",
                        arguments={"text": "tool output"},
                    )
                ]
            ),
            ModelResponse(items=[AssistantMessage(text="summary after tool")]),
            ModelResponse(items=[AssistantMessage(text="final answer")]),
        ],
        usage_by_call={1: 12},
    )
    tools = ToolRegistry()
    tools.register(EchoTool())
    events = []
    agent = Agent(
        model,
        tools,
        _auto_compact_context(10),
        event_handler=events.append,
    )

    result = await agent.run_turn(["use the tool"])

    assert result.output_text == "final answer"
    assert model.call_count == 3
    compact_prompt_items = _conversation_items(model.prompts[1])
    assert [type(item).__name__ for item in compact_prompt_items] == [
        "UserMessage",
        "ToolCall",
        "ToolResult",
        "UserMessage",
    ]
    assert compact_prompt_items[0].text == "use the tool"
    assert compact_prompt_items[1].name == "echo"
    assert compact_prompt_items[2].output == "tool output"
    assert compact_prompt_items[3].text == DEFAULT_COMPACT_PROMPT

    follow_up_items = _conversation_items(model.prompts[2])
    assert [type(item).__name__ for item in follow_up_items] == [
        "UserMessage",
        "UserMessage",
    ]
    assert follow_up_items[0].text == "use the tool"
    assert follow_up_items[1].text == f"{SUMMARY_PREFIX}\nsummary after tool"

    auto_events = [event for event in events if event.kind.startswith("auto_compact_")]
    assert [event.kind for event in auto_events] == [
        "auto_compact_started",
        "auto_compact_completed",
    ]
    assert auto_events[0].payload["phase"] == "mid_turn"


@pytest.mark.asyncio
async def test_agent_midturn_auto_compact_accepts_partial_incomplete_summary() -> 'None':
    class PartialCompactModelClient:
        def __init__(self) -> 'None':
            self.prompts = []
            self.call_count = 0

        async def complete(self, prompt, event_handler):
            self.prompts.append(prompt)
            self.call_count += 1
            if self.call_count == 1:
                event_handler(
                    ModelStreamEvent(
                        kind="token_count",
                        payload={"usage": {"total_tokens": 12}},
                    )
                )
                return ModelResponse(
                    items=[
                        ToolCall(
                            call_id="call_1",
                            name="echo",
                            arguments={"text": "tool output"},
                        )
                    ]
                )
            if self.call_count == 2:
                event_handler(
                    ModelStreamEvent(
                        kind="assistant_delta",
                        payload={"delta": "partial compact summary"},
                    )
                )
                raise ResponsesIncompleteError(
                    "responses stream ended with `response.incomplete`",
                    [AssistantMessage(text="partial compact summary")],
                    reason="max_output_tokens",
                )
            if self.call_count == 3:
                return ModelResponse(items=[AssistantMessage(text="final answer")])
            raise AssertionError(f"unexpected call_count={self.call_count}")

    model = PartialCompactModelClient()
    tools = ToolRegistry()
    tools.register(EchoTool())
    events = []
    agent = Agent(
        model,
        tools,
        _auto_compact_context(10),
        event_handler=events.append,
    )

    result = await agent.run_turn(["use the tool"])

    assert result.output_text == "final answer"
    assert model.call_count == 3
    assert "turn_failed" not in [event.kind for event in events]
    follow_up_items = _conversation_items(model.prompts[2])
    assert [type(item).__name__ for item in follow_up_items] == [
        "UserMessage",
        "UserMessage",
    ]
    assert follow_up_items[1].text == f"{SUMMARY_PREFIX}\npartial compact summary"
    auto_events = [event for event in events if event.kind.startswith("auto_compact_")]
    assert [event.kind for event in auto_events] == [
        "auto_compact_started",
        "auto_compact_completed",
    ]


@pytest.mark.asyncio
async def test_agent_midturn_auto_compact_rejects_non_token_incomplete_summary() -> 'None':
    class PartialCompactModelClient:
        def __init__(self) -> 'None':
            self.call_count = 0

        async def complete(self, prompt, event_handler):
            del prompt
            self.call_count += 1
            if self.call_count == 1:
                event_handler(
                    ModelStreamEvent(
                        kind="token_count",
                        payload={"usage": {"total_tokens": 2}},
                    )
                )
                return ModelResponse(
                    items=[
                        ToolCall(
                            call_id="call_1",
                            name="echo",
                            arguments={"text": "tool output"},
                        )
                    ]
                )
            if self.call_count == 2:
                raise ResponsesIncompleteError(
                    "responses stream ended with `response.incomplete`",
                    [AssistantMessage(text="partial compact summary")],
                    reason="content_filter",
                )
            raise AssertionError(f"unexpected call_count={self.call_count}")

    events = []
    tools = ToolRegistry()
    tools.register(EchoTool())
    agent = Agent(
        PartialCompactModelClient(),
        tools,
        _auto_compact_context(1),
        event_handler=events.append,
    )

    with pytest.raises(ResponsesIncompleteError):
        await agent.run_turn(["new prompt"])

    auto_events = [event for event in events if event.kind.startswith("auto_compact_")]
    assert [event.kind for event in auto_events] == [
        "auto_compact_started",
        "auto_compact_failed",
    ]
    assert auto_events[1].payload["error_type"] == "ResponsesIncompleteError"


@pytest.mark.asyncio
async def test_agent_does_not_auto_compact_without_token_limit() -> 'None':
    model = UsageModelClient(
        [
            ModelResponse(items=[AssistantMessage(text="first answer")]),
            ModelResponse(items=[AssistantMessage(text="second answer")]),
        ],
        usage_by_call={1: 1_000_000},
    )
    events = []
    agent = Agent(
        model,
        ToolRegistry(),
        _auto_compact_context(None),
        event_handler=events.append,
    )

    await agent.run_turn(["first prompt"])
    await agent.run_turn(["second prompt"])

    assert model.call_count == 2
    assert not [event for event in events if event.kind.startswith("auto_compact_")]


@pytest.mark.asyncio
async def test_wait_agent_injects_subagent_notification_into_history() -> 'None':
    model = ScriptedModelClient(
        [
            ModelResponse(
                items=[
                    ToolCall(
                        call_id="call_wait",
                        name="wait_agent",
                        arguments={"ids": ["agent_x"]},
                    )
                ]
            ),
            ModelResponse(items=[AssistantMessage(text="done")]),
        ]
    )

    tools = ToolRegistry()
    tools.register(WaitAgentNotificationTool())

    agent = Agent(model, tools)
    result = await agent.run_turn(["check subagent"])

    assert result.output_text == "done"
    notification = next(
        item
        for item in result.history
        if isinstance(item, UserMessage)
        and item.text.startswith("<subagent_notification>\n")
    )
    assert (
        notification.text
        == "<subagent_notification>\n"
        '{"agent_id":"019d0000-0000-7000-8000-000000000000","status":{"completed":"subagent done"}}\n'
        "</subagent_notification>"
    )


@pytest.mark.asyncio
async def test_runtime_submission_loop_processes_turn_and_shutdown() -> 'None':
    model = ScriptedModelClient([ModelResponse(items=[AssistantMessage(text="done")])])
    tools = ToolRegistry()
    agent = Agent(model, tools)
    runtime = CliSubmissionQueue(agent)

    worker = asyncio.create_task(runtime.run_forever())
    try:
        result = await runtime.submit_user_turn("hello")
        assert result.output_text == "done"
        await runtime.shutdown()
    finally:
        await worker


@pytest.mark.asyncio
async def test_agent_maybe_invoke_formats_exec_completion_when_idle() -> 'None':
    request_seen = asyncio.Event()

    async def response_factory(prompt, call_count):
        del prompt, call_count
        request_seen.set()
        return ModelResponse(items=[AssistantMessage(text="auto done")])

    model = ScriptedModelClient(
        response_factory=response_factory,
    )
    agent = Agent(model, ToolRegistry())

    started = await agent.maybe_invoke(
        {
            "type": "exec_command_completed",
            "session_id": 1000,
            "exit_code": 0,
            "command": "python watch.py",
        }
    )

    assert started is True
    await asyncio.wait_for(request_seen.wait(), timeout=1.0)
    while agent._turn_running:
        await asyncio.sleep(0.01)
    assert model.call_count == 1
    prompt_items = _conversation_items(model.prompts[0])
    assert isinstance(prompt_items[0], UserMessage)
    assert (
        prompt_items[0].text
        == "<exec_command_completed>\n"
        '{"session_id":1000,"exit_code":0,"command":"python watch.py"}\n'
        "</exec_command_completed>"
    )


def test_agent_connects_exec_completion_hook_from_tool_registry() -> 'None':
    tools = ToolRegistry()
    manager = UnifiedExecManager()
    tools.register(ExecCommandTool(manager))
    agent = Agent(
        ScriptedModelClient([ModelResponse(items=[AssistantMessage(text="done")])]),
        tools,
    )

    assert manager._notify_hook == agent.maybe_invoke


@pytest.mark.asyncio
async def test_agent_turn_completed_emits_background_exec_count() -> 'None':
    tools = ToolRegistry()
    manager = UnifiedExecManager()
    tools.register(ExecCommandTool(manager))
    agent = Agent(
        ScriptedModelClient([ModelResponse(items=[AssistantMessage(text="done")])]),
        tools,
    )
    manager.running_session_count = lambda: 2
    events = []
    agent.set_event_handler(events.append)

    await agent.run_turn(["hello"])

    completed_events = [event for event in events if event.kind == "turn_completed"]
    assert completed_events
    assert completed_events[-1].payload["background_exec_count"] == 2


@pytest.mark.asyncio
async def test_agent_maybe_invoke_noops_while_active() -> 'None':
    request_started = asyncio.Event()
    release_request = asyncio.Event()

    async def response_factory(prompt, call_count):
        del prompt, call_count
        request_started.set()
        await release_request.wait()
        return ModelResponse(items=[AssistantMessage(text="done")])

    model = ScriptedModelClient(response_factory=response_factory)
    agent = Agent(model, ToolRegistry())
    first_turn = asyncio.create_task(agent.run_turn(["hello"]))
    try:
        await request_started.wait()
        started = await agent.maybe_invoke(
            {
                "type": "exec_command_completed",
                "session_id": 1000,
                "exit_code": 0,
                "command": "python watch.py",
            }
        )

        assert started is False
        release_request.set()
        result = await first_turn
        assert result.output_text == "done"
        assert model.call_count == 1
    finally:
        release_request.set()


@pytest.mark.asyncio
async def test_runtime_steer_batches_messages_into_next_request() -> 'None':
    first_request_started = asyncio.Event()
    release_first_request = asyncio.Event()

    class _DelayedModelClient:
        def __init__(self) -> 'None':
            self.prompts = []
            self.call_count = 0

        async def complete(self, prompt, event_handler):
            self.prompts.append(prompt)
            self.call_count += 1
            if self.call_count == 1:
                first_request_started.set()
                await release_first_request.wait()
                event_handler(
                    ModelStreamEvent(kind="assistant_delta", payload={"delta": "first"})
                )
                return ModelResponse(items=[AssistantMessage(text="first")])

            event_handler(
                ModelStreamEvent(kind="assistant_delta", payload={"delta": "second"})
            )
            return ModelResponse(items=[AssistantMessage(text="second")])

    model = _DelayedModelClient()
    runtime = CliSubmissionQueue(Agent(model, ToolRegistry()))

    worker = asyncio.create_task(runtime.run_forever())
    first_turn = asyncio.create_task(runtime.submit_user_turn("hello"))
    try:
        await first_request_started.wait()
        steer_submission_id, steer_future_a = await runtime.enqueue_user_turn(
            "again",
            queue="steer",
        )
        steer_submission_id_b, steer_future_b = await runtime.enqueue_user_turn(
            "one more",
            queue="steer",
        )

        assert steer_submission_id == steer_submission_id_b

        release_first_request.set()

        with pytest.raises(RuntimeError, match="submission interrupted"):
            await first_turn

        result_a = await steer_future_a
        result_b = await steer_future_b
        assert result_a is result_b
        assert result_a is not None
        assert result_a.output_text == "second"
        assert model.call_count == 2

        second_prompt = model.prompts[1]
        assert model.prompts[0].turn_id == second_prompt.turn_id
        user_texts = [
            item.text for item in second_prompt.input if isinstance(item, UserMessage)
        ]
        assert user_texts[-3:] == ["hello", "again", "one more"]
        assert any(
            isinstance(item, AssistantMessage) and item.text == "first"
            for item in second_prompt.input
        )

        await runtime.shutdown()
    finally:
        if not release_first_request.is_set():
            release_first_request.set()
        await worker


@pytest.mark.asyncio
async def test_agent_emits_turn_failed_event_on_model_error() -> 'None':
    events = []

    class FailingModelClient:
        async def complete(self, prompt, event_handler):
            del prompt, event_handler
            raise RuntimeError("synthetic client error")

    agent = Agent(FailingModelClient(), ToolRegistry(), event_handler=events.append)

    with pytest.raises(RuntimeError, match="synthetic client error"):
        await agent.run_turn(["hello"])

    assert [event.kind for event in events] == [
        "turn_started",
        "model_called",
        "turn_failed",
    ]
    assert events[-1].payload["error"] == "synthetic client error"


@pytest.mark.asyncio
async def test_agent_emits_token_count_for_context_length_error() -> 'None':
    events = []
    error_message = _context_length_error_message()

    class FailingModelClient:
        async def complete(self, prompt, event_handler):
            del prompt, event_handler
            raise RuntimeError(error_message)

    agent = Agent(FailingModelClient(), ToolRegistry(), event_handler=events.append)

    with pytest.raises(RuntimeError, match="context_length_exceeded"):
        await agent.run_turn(["hello"])

    assert [event.kind for event in events] == [
        "turn_started",
        "model_called",
        "token_count",
        "auto_compact_started",
        "auto_compact_failed",
        "token_count",
        "turn_failed",
    ]
    assert events[2].payload["usage"] == {
        "total_tokens": 264568,
        "input_tokens": 264568,
        "output_tokens": 0,
    }
    assert events[3].payload["phase"] == "context_length_exceeded"
    assert events[3].payload["total_tokens"] == 264568
    assert events[3].payload["token_limit"] == 262144


@pytest.mark.asyncio
async def test_agent_auto_compacts_and_retries_on_context_length_error() -> 'None':
    events = []

    def response_factory(prompt, call_count):
        if call_count == 1:
            raise RuntimeError(_context_length_error_message())
        if call_count == 2:
            return ModelResponse(items=[AssistantMessage(text="checkpoint summary")])
        if call_count == 3:
            return ModelResponse(items=[AssistantMessage(text="final answer")])
        raise AssertionError(f"unexpected call_count={call_count}")

    model = ScriptedModelClient(response_factory=response_factory)
    agent = Agent(model, ToolRegistry(), event_handler=events.append)

    result = await agent.run_turn(["hello"])

    assert result.output_text == "final answer"
    assert model.call_count == 3

    compact_prompt_items = _conversation_items(model.prompts[1])
    assert [type(item).__name__ for item in compact_prompt_items] == [
        "UserMessage",
        "UserMessage",
    ]
    assert compact_prompt_items[0].text == "hello"
    assert compact_prompt_items[1].text == DEFAULT_COMPACT_PROMPT

    retry_prompt_items = _conversation_items(model.prompts[2])
    assert [type(item).__name__ for item in retry_prompt_items] == [
        "UserMessage",
        "UserMessage",
    ]
    assert retry_prompt_items[0].text == "hello"
    assert retry_prompt_items[1].text == f"{SUMMARY_PREFIX}\ncheckpoint summary"

    assert "turn_failed" not in [event.kind for event in events]
    auto_events = [event for event in events if event.kind.startswith("auto_compact_")]
    assert [event.kind for event in auto_events] == [
        "auto_compact_started",
        "auto_compact_completed",
    ]
    assert auto_events[0].payload["phase"] == "context_length_exceeded"
    assert auto_events[1].payload["summary"] == (
        "compact(1 item) -> 1 item + [summary]"
    )


@pytest.mark.asyncio
async def test_agent_auto_compacts_on_context_window_error_without_token_counts() -> 'None':
    events = []
    error_message = (
        "ResponsesApiError: responses stream failed on the server side\n"
        "- detail: Your input exceeds the context window of this model. "
        "Please adjust your input and try again."
    )

    def response_factory(prompt, call_count):
        if call_count == 1:
            raise RuntimeError(error_message)
        if call_count == 2:
            return ModelResponse(items=[AssistantMessage(text="checkpoint summary")])
        if call_count == 3:
            return ModelResponse(items=[AssistantMessage(text="final answer")])
        raise AssertionError(f"unexpected call_count={call_count}")

    model = ScriptedModelClient(response_factory=response_factory)
    agent = Agent(model, ToolRegistry(), event_handler=events.append)

    result = await agent.run_turn(["hello"])

    assert result.output_text == "final answer"
    assert [event.kind for event in events if event.kind == "token_count"] == []
    auto_events = [event for event in events if event.kind.startswith("auto_compact_")]
    assert [event.kind for event in auto_events] == [
        "auto_compact_started",
        "auto_compact_completed",
    ]
    assert auto_events[0].payload == {"phase": "context_length_exceeded"}


@pytest.mark.asyncio
async def test_agent_prunes_old_tool_responses_when_context_compact_overflows() -> 'None':
    events = []
    initial_history = (
        UserMessage(text="old prompt"),
        ToolCall(call_id="call_old", name="echo", arguments={"text": "large"}),
        ToolResult(call_id="call_old", name="echo", output="large output"),
        AssistantMessage(text="old answer"),
    )

    def response_factory(prompt, call_count):
        if call_count == 1:
            raise RuntimeError(_context_length_error_message())
        if call_count == 2:
            compact_items = _conversation_items(prompt)
            assert any(isinstance(item, ToolResult) for item in compact_items)
            raise RuntimeError(_context_length_error_message())
        if call_count == 3:
            compact_items = _conversation_items(prompt)
            assert not any(isinstance(item, ToolCall) for item in compact_items)
            assert not any(isinstance(item, ToolResult) for item in compact_items)
            return ModelResponse(items=[AssistantMessage(text="summary without tools")])
        if call_count == 4:
            retry_items = _conversation_items(prompt)
            assert not any(isinstance(item, ToolCall) for item in retry_items)
            assert not any(isinstance(item, ToolResult) for item in retry_items)
            return ModelResponse(items=[AssistantMessage(text="final after prune")])
        raise AssertionError(f"unexpected call_count={call_count}")

    model = ScriptedModelClient(response_factory=response_factory)
    agent = Agent(
        model,
        ToolRegistry(),
        event_handler=events.append,
        initial_history=initial_history,
    )

    result = await agent.run_turn(["new prompt"])

    assert result.output_text == "final after prune"
    assert model.call_count == 4
    assert not any(isinstance(item, ToolCall) for item in result.history)
    assert not any(isinstance(item, ToolResult) for item in result.history)
    auto_completed = [
        event for event in events if event.kind == "auto_compact_completed"
    ][0]
    assert auto_completed.payload["pruned_tool_results"] == 1
    assert auto_completed.payload["summary"] == (
        "compact(5 items) -> 2 items + [summary] "
        "(dropped 1 old tool response)"
    )


@pytest.mark.asyncio
async def test_agent_relays_stream_error_events() -> 'None':
    events = []

    class RetryingModelClient:
        async def complete(self, prompt, event_handler):
            del prompt
            event_handler(
                ModelStreamEvent(
                    kind="stream_error",
                    payload={"message": "Reconnecting... 1/5"},
                )
            )
            return ModelResponse(items=[AssistantMessage(text="done")])

    agent = Agent(RetryingModelClient(), ToolRegistry(), event_handler=events.append)

    result = await agent.run_turn(["hello"])

    assert result.output_text == "done"
    assert [event.kind for event in events] == [
        "turn_started",
        "model_called",
        "stream_error",
        "model_completed",
        "turn_completed",
    ]
    assert events[2].payload["message"] == "Reconnecting... 1/5"
