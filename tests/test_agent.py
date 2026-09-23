import asyncio
import inspect
import sys
import typing

import pytest

from pycodex import (
    Agent,
    AgentRuntime,
    AssistantMessage,
    BaseTool,
    ContextConfig,
    ContextLengthExceeded,
    ModelResponse,
    ReasoningItem,
    ResponsesIncompleteError,
    ToolCall,
    ToolRegistry,
    ToolResult,
    TurnInterrupted,
    UserMessage,
)
from pycodex.events import AssistantDeltaEvent, StreamErrorEvent, TokenCountEvent
from pycodex.tools import (
    ClockManager,
    ClockTool,
    ExecCommandTool,
    UnifiedExecManager,
    WaitAgentTool,
)
from pycodex.tools.base_tool import StructuredToolOutput
from pycodex.utils.compactor import DEFAULT_COMPACT_PROMPT, compact
from tests.fakes import ScriptedModelClient


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

    def __init__(self, name: "str") -> "None":
        self.name = name

    async def run(self, context, args):
        del context, args
        await asyncio.sleep(0.05)
        return "done"


class CoordinatedParallelTool(BaseTool):
    description = "Tool that proves both calls entered before either finished."
    input_schema = {"type": "object"}
    supports_parallel = True

    def __init__(
        self, name: "str", entered: "typing.List[str]", both_started: "asyncio.Event"
    ) -> "None":
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


class CompletedSubAgentManager:
    async def wait_agents(self, agent_ids, timeout_ms):
        del agent_ids, timeout_ms
        return {
            "status": {
                "019d0000-0000-7000-8000-000000000000": {
                    "completed": "subagent done",
                }
            },
            "timed_out": False,
        }


class LongOutputTool(BaseTool):
    name = "long_output"
    description = "Returns a long string."
    input_schema = {"type": "object"}

    async def run(self, context, args):
        del context
        return str(args["text"])


class ContentItemsTool(BaseTool):
    name = "content_items"
    description = "Returns structured content items."
    input_schema = {"type": "object"}

    async def run(self, context, args):
        del context
        return StructuredToolOutput(
            output="structured output",
            content_items=tuple(args["content_items"]),
        )


class UsageModelClient:
    model = "test"

    def __init__(
        self,
        responses: "typing.Iterable[ModelResponse]",
        usage_by_call: "typing.Union[typing.Dict[int, int], None]" = None,
    ) -> "None":
        self._responses = iter(responses)
        self._usage_by_call = usage_by_call or {}
        self.prompts: "typing.List[object]" = []
        self.call_count = 0

    async def complete(self, prompt, event_handler):
        self.prompts.append(prompt)
        self.call_count += 1
        total_tokens = self._usage_by_call.get(self.call_count)
        if total_tokens is not None:
            event_handler(TokenCountEvent({"total_tokens": total_tokens}))
        try:
            return next(self._responses)
        except StopIteration as exc:
            raise RuntimeError("usage model ran out of responses") from exc


def _auto_compact_context(limit: "typing.Union[int, None]") -> "ContextConfig":
    return ContextConfig(
        model_auto_compact_token_limit=limit,
        include_permissions_instructions=False,
        include_skills_instructions=False,
    )


def _model_context(model: "str") -> "ContextConfig":
    return ContextConfig(
        model=model,
        include_permissions_instructions=False,
        include_skills_instructions=False,
    )


def _conversation_items(
    prompt,
) -> "typing.List[typing.Union[UserMessage, AssistantMessage, ReasoningItem, ToolCall, ToolResult]]":
    return [
        item
        for item in prompt.input
        if isinstance(
            item,
            (UserMessage, AssistantMessage, ReasoningItem, ToolCall, ToolResult),
        )
    ]


def _context_length_error_message(
    requested_tokens: "int" = 264568,
    max_tokens: "int" = 262144,
) -> "str":
    return (
        "responses_server.stream_router.OutcommingChatError: outcomming chat "
        'request failed with status 400: {"error":{"message":"This model\'s '
        f"maximum context length is {max_tokens} tokens. However, you requested "
        f"{requested_tokens} tokens ({requested_tokens} in the messages, 0 in "
        "the completion). Please reduce the length of the messages or "
        'completion.","type":"context_length_exceeded"}}'
    )


def test_agent_ask_runs_turn_from_sync_context() -> "None":
    model = ScriptedModelClient(
        [ModelResponse(items=[AssistantMessage(text="sync answer")])]
    )
    agent = Agent(model, ToolRegistry(), ContextConfig())

    result = agent.ask("sync prompt")

    assert result.output_text == "sync answer"
    assert result.iterations == 1
    assert model.call_count == 1


@pytest.mark.parametrize("fail_model", [False, True])
def test_agent_ask_uses_run_turn_and_propagates_its_result(
    monkeypatch, fail_model
) -> "None":
    def respond(prompt, call_count):
        if fail_model:
            raise ValueError("sync model failed")
        return ModelResponse([AssistantMessage("sync answer")])

    client = ScriptedModelClient(response_factory=respond)
    agent = Agent(client, ToolRegistry(), ContextConfig())
    received_inputs = []
    run_turn = agent.run_turn

    async def record_turn(texts, turn_id=None):
        received_inputs.append(texts)
        return await run_turn(texts, turn_id)

    monkeypatch.setattr(agent, "run_turn", record_turn)
    if fail_model:
        with pytest.raises(ValueError, match="sync model failed"):
            agent.ask("sync prompt")
    else:
        result = agent.ask("sync prompt")
        assert result.output_text == "sync answer"
    assert received_inputs == [["sync prompt"]]
    assert not agent.is_running
    assert client.call_count == 1


def test_agent_run_turn_is_a_regular_coroutine() -> "None":
    client = ScriptedModelClient([ModelResponse([AssistantMessage("done")])])
    agent = Agent(client, ToolRegistry(), ContextConfig())
    coroutine = agent.run_turn(["prompt"])
    assert inspect.iscoroutinefunction(agent.run_turn)
    assert inspect.iscoroutine(coroutine)
    assert not agent.is_running
    assert agent.history == ()
    assert client.call_count == 0
    assert asyncio.run(coroutine).output_text == "done"
    assert not agent.is_running
    assert not hasattr(agent, "task")
    assert not hasattr(agent, "cancel")
    assert not hasattr(agent, "add_task_listener")


@pytest.mark.asyncio
async def test_agent_ask_does_not_schedule_work_if_sync_bridge_cannot_run(
    monkeypatch,
) -> "None":
    monkeypatch.setitem(sys.modules, "nest_asyncio", None)
    client = ScriptedModelClient([ModelResponse([AssistantMessage("unused")])])
    agent = Agent(client, ToolRegistry(), ContextConfig())

    with pytest.raises(RuntimeError, match="cannot block on a running event loop"):
        agent.ask("prompt")

    assert not agent.is_running
    assert agent.history == ()
    assert client.call_count == 0


@pytest.mark.asyncio
async def test_agent_runs_tool_then_returns_final_message() -> "None":
    model = ScriptedModelClient(
        [
            ModelResponse(
                items=[
                    AssistantMessage(text="我先看一下。"),
                    ToolCall(
                        call_id="call_1", name="echo", arguments={"text": "hello"}
                    ),
                ]
            ),
            ModelResponse(items=[AssistantMessage(text="工具返回了 hello")]),
        ]
    )

    tools = ToolRegistry()
    tools.register(EchoTool())

    agent = Agent(model, tools, ContextConfig())
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
async def test_parallel_tools_share_one_model_round() -> "None":
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
    entered: "typing.List[str]" = []
    both_started = asyncio.Event()
    tools.register(CoordinatedParallelTool("slow_a", entered, both_started))
    tools.register(CoordinatedParallelTool("slow_b", entered, both_started))

    agent = Agent(model, tools, ContextConfig())
    result = await agent.run_turn(["并行跑两个工具"])

    assert result.output_text == "两个工具都执行完了"
    assert entered == ["slow_a", "slow_b"] or entered == ["slow_b", "slow_a"]


@pytest.mark.asyncio
async def test_agent_default_has_no_fixed_iteration_cap() -> "None":
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

    agent = Agent(model, tools, ContextConfig())
    result = await agent.run_turn(["连续调用工具直到结束"])

    assert result.output_text == "超过 12 轮后也收敛了"
    assert result.iterations == 13


@pytest.mark.asyncio
async def test_agent_auto_compacts_before_next_turn_when_usage_reaches_limit() -> (
    "None"
):
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
    ]
    assert (
        second_prompt_items[0]
        == compact(
            [AssistantMessage("checkpoint summary")], str(agent.session_file_path)
        )[0]
    )
    assert second_prompt_items[1].text == "second prompt"

    auto_events = [event for event in events if event.kind.startswith("auto_compact_")]
    assert [event.kind for event in auto_events] == [
        "auto_compact_started",
        "auto_compact_completed",
    ]
    assert auto_events[0].phase == "pre_turn"
    assert auto_events[0].total_tokens == 12
    assert auto_events[0].token_limit == 10


@pytest.mark.asyncio
async def test_agent_auto_compacts_before_tool_follow_up_when_usage_reaches_limit() -> (
    "None"
):
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
    ]
    assert (
        follow_up_items[0]
        == compact(
            [AssistantMessage("summary after tool")], str(agent.session_file_path)
        )[0]
    )

    auto_events = [event for event in events if event.kind.startswith("auto_compact_")]
    assert [event.kind for event in auto_events] == [
        "auto_compact_started",
        "auto_compact_completed",
    ]
    assert auto_events[0].phase == "mid_turn"


@pytest.mark.asyncio
async def test_agent_midturn_auto_compact_accepts_partial_incomplete_summary() -> (
    "None"
):
    class PartialCompactModelClient:
        model = "test"

        def __init__(self) -> "None":
            self.prompts = []
            self.call_count = 0

        async def complete(self, prompt, event_handler):
            self.prompts.append(prompt)
            self.call_count += 1
            if self.call_count == 1:
                event_handler(TokenCountEvent({"total_tokens": 12}))
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
                event_handler(AssistantDeltaEvent("partial compact summary"))
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
    ]
    assert (
        follow_up_items[0]
        == compact(
            [AssistantMessage("partial compact summary")], str(agent.session_file_path)
        )[0]
    )
    auto_events = [event for event in events if event.kind.startswith("auto_compact_")]
    assert [event.kind for event in auto_events] == [
        "auto_compact_started",
        "auto_compact_completed",
    ]


@pytest.mark.asyncio
async def test_agent_midturn_auto_compact_rejects_non_token_incomplete_summary() -> (
    "None"
):
    class PartialCompactModelClient:
        model = "test"

        def __init__(self) -> "None":
            self.call_count = 0

        async def complete(self, prompt, event_handler):
            del prompt
            self.call_count += 1
            if self.call_count == 1:
                event_handler(TokenCountEvent({"total_tokens": 2}))
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
    assert auto_events[1].error_type == "ResponsesIncompleteError"


@pytest.mark.asyncio
async def test_agent_does_not_auto_compact_without_token_limit() -> "None":
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
async def test_wait_agent_injects_subagent_notification_into_history() -> "None":
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
    tools.register(WaitAgentTool(CompletedSubAgentManager()))

    agent = Agent(model, tools, ContextConfig())
    result = await agent.run_turn(["check subagent"])

    assert result.output_text == "done"
    notification = next(
        item
        for item in result.history
        if isinstance(item, UserMessage)
        and item.text.startswith("<subagent_notification>\n")
    )
    assert (
        notification.text == "<subagent_notification>\n"
        '{"agent_id":"019d0000-0000-7000-8000-000000000000","status":{"completed":"subagent done"}}\n'
        "</subagent_notification>"
    )


@pytest.mark.asyncio
async def test_agent_truncates_tool_output_before_history_follow_up() -> "None":
    long_output = "0123456789" * 7000
    model = ScriptedModelClient(
        [
            ModelResponse(
                items=[
                    ToolCall(
                        call_id="call_long",
                        name="long_output",
                        arguments={"text": long_output},
                    )
                ]
            ),
            ModelResponse(items=[AssistantMessage(text="done")]),
        ]
    )
    tools = ToolRegistry()
    tools.register(LongOutputTool())

    agent = Agent(model, tools, _model_context("gpt-5.5"))

    result = await agent.run_turn(["run long tool"])

    assert result.output_text == "done"
    tool_result = next(item for item in result.history if isinstance(item, ToolResult))
    assert isinstance(tool_result.output, str)
    assert tool_result.output != long_output
    assert "tokens truncated" in tool_result.output

    follow_up_items = _conversation_items(model.prompts[1])
    prompt_tool_result = next(
        item for item in follow_up_items if isinstance(item, ToolResult)
    )
    assert prompt_tool_result.output == tool_result.output


@pytest.mark.asyncio
async def test_agent_truncates_tool_content_items_for_history() -> "None":
    model = ScriptedModelClient(
        [
            ModelResponse(
                items=[
                    ToolCall(
                        call_id="call_content",
                        name="content_items",
                        arguments={
                            "content_items": [
                                {"type": "input_text", "text": "a" * 60000},
                                {
                                    "type": "input_image",
                                    "image_url": "file:///tmp/x.png",
                                },
                                {"type": "input_text", "text": "b" * 5000},
                            ]
                        },
                    )
                ]
            ),
            ModelResponse(items=[AssistantMessage(text="done")]),
        ]
    )
    tools = ToolRegistry()
    tools.register(ContentItemsTool())

    agent = Agent(model, tools, _model_context("gpt-5.5"))

    await agent.run_turn(["run content tool"])

    tool_result = next(item for item in agent.history if isinstance(item, ToolResult))
    assert tool_result.content_items is not None
    assert tool_result.content_items[0]["type"] == "input_text"
    assert "tokens truncated" in tool_result.content_items[0]["text"]
    assert tool_result.content_items[1] == {
        "type": "input_image",
        "image_url": "file:///tmp/x.png",
    }
    assert tool_result.content_items[2] == {
        "type": "input_text",
        "text": "[omitted 1 text items ...]",
    }


@pytest.mark.asyncio
async def test_agent_keeps_small_structured_tool_output_for_follow_up() -> "None":
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
    tools.register(WaitAgentTool(CompletedSubAgentManager()))

    agent = Agent(model, tools, _model_context("gpt-5.5"))

    await agent.run_turn(["check subagent"])

    tool_result = next(item for item in agent.history if isinstance(item, ToolResult))
    assert isinstance(tool_result.output, dict)
    notification = next(
        item
        for item in agent.history
        if isinstance(item, UserMessage)
        and item.text.startswith("<subagent_notification>\n")
    )
    assert "019d0000-0000-7000-8000-000000000000" in notification.text


@pytest.mark.asyncio
async def test_runtime_submission_loop_processes_turn_and_shutdown() -> "None":
    model = ScriptedModelClient([ModelResponse(items=[AssistantMessage(text="done")])])
    tools = ToolRegistry()
    agent = Agent(model, tools, ContextConfig())
    runtime = AgentRuntime(agent)

    await runtime.start()
    try:
        result = await runtime.submit_user_turn("hello")
        assert result.output_text == "done"
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_agent_maybe_invoke_formats_exec_completion_when_idle() -> "None":
    request_seen = asyncio.Event()

    async def response_factory(prompt, call_count):
        del prompt, call_count
        request_seen.set()
        return ModelResponse(items=[AssistantMessage(text="auto done")])

    model = ScriptedModelClient(
        response_factory=response_factory,
    )
    agent = Agent(model, ToolRegistry(), ContextConfig())

    started = await agent.maybe_invoke(
        {
            "type": "exec_command_completed",
            "session_id": 1000,
            "exit_code": 0,
            "command": "python watch.py",
        }
    )

    assert started is True
    assert not agent.is_running
    await asyncio.wait_for(request_seen.wait(), timeout=1.0)
    await asyncio.wait_for(agent.wait_until_idle(), timeout=1.0)
    assert model.call_count == 1
    prompt_items = _conversation_items(model.prompts[0])
    assert isinstance(prompt_items[0], UserMessage)
    assert (
        prompt_items[0].text == "<exec_command_completed>\n"
        '{"session_id":1000,"exit_code":0,"command":"python watch.py"}\n'
        "</exec_command_completed>"
    )


@pytest.mark.parametrize("tool_name", ["exec_command", "renamed_exec"])
def test_agent_connects_exec_completion_hook_from_tool_registry(tool_name) -> "None":
    tools = ToolRegistry()
    manager = UnifiedExecManager()
    tool = ExecCommandTool(manager)
    tool.name = tool_name
    tools.register(tool)
    agent = Agent(
        ScriptedModelClient([ModelResponse(items=[AssistantMessage(text="done")])]),
        tools,
        ContextConfig(),
    )

    assert manager._notify_hook == agent.maybe_invoke


@pytest.mark.asyncio
async def test_clock_wakes_agent_periodically_until_cancelled(monkeypatch) -> "None":
    monkeypatch.setattr(
        "pycodex.tools.clock_tool._current_time",
        lambda: "2026-08-07T12:34:56+08:00",
    )
    manager = ClockManager(seconds_per_minute=0.01)
    tools = ToolRegistry()
    tools.register(ClockTool(manager))
    model = ScriptedModelClient(
        [
            ModelResponse(
                items=[
                    ToolCall(
                        call_id="clock_set",
                        name="clock",
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
                        name="clock",
                        arguments={"period_m": None},
                    )
                ]
            ),
            ModelResponse(items=[AssistantMessage(text="clock stopped")]),
        ]
    )
    agent = Agent(model, tools, ContextConfig())

    result = await agent.run_turn(["start periodic work"])

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


@pytest.mark.asyncio
async def test_runtime_shutdown_cancels_clock_after_draining_queued_tools() -> "None":
    manager = ClockManager()
    manager.set_period(1)
    tools = ToolRegistry()
    tools.register(ClockTool(manager))
    model = ScriptedModelClient(
        [
            ModelResponse(items=[AssistantMessage(text="done")]),
            ModelResponse(items=[ToolCall("clock", "clock", {"period_m": 1})]),
            ModelResponse(items=[AssistantMessage(text="done")]),
        ]
    )
    agent = Agent(model, tools, ContextConfig())
    events = []
    runtime = AgentRuntime(agent)
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

        _submission_id, pending_turn = await runtime.enqueue_user_turn(
            "set clock again"
        )
        await runtime.close()
        assert (await pending_turn).output_text == "done"
        await asyncio.gather(pending_timer, return_exceptions=True)

        assert manager.snapshot() == {"enabled": False, "period_m": None}
        assert manager._timer_task is None
        assert model.call_count == 3
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_agent_turn_completed_counts_background_exec_work() -> "None":
    tools = ToolRegistry()
    manager = UnifiedExecManager()
    tools.register(ExecCommandTool(manager))
    agent = Agent(
        ScriptedModelClient([ModelResponse(items=[AssistantMessage(text="done")])]),
        tools,
        ContextConfig(),
    )
    manager.running_session_count = lambda: 2
    events = []
    agent.event_handler = events.append

    await agent.run_turn(["hello"])

    completed_events = [event for event in events if event.kind == "turn_completed"]
    assert completed_events
    assert completed_events[-1].background_work_count == 2


@pytest.mark.asyncio
async def test_agent_maybe_invoke_noops_while_active() -> "None":
    request_started = asyncio.Event()
    release_request = asyncio.Event()

    async def response_factory(prompt, call_count):
        del prompt, call_count
        request_started.set()
        await release_request.wait()
        return ModelResponse(items=[AssistantMessage(text="done")])

    model = ScriptedModelClient(response_factory=response_factory)
    agent = Agent(model, ToolRegistry(), ContextConfig())
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
async def test_runtime_waits_for_agent_background_turn_before_next_submission() -> (
    "None"
):
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()
    second_request_started = asyncio.Event()

    class _DelayedModelClient:
        model = "test"

        def __init__(self) -> "None":
            self.prompts = []
            self.call_count = 0

        async def complete(self, prompt, event_handler):
            del event_handler
            self.prompts.append(prompt)
            self.call_count += 1
            if self.call_count == 1:
                return ModelResponse(
                    items=[ToolCall(call_id="call_1", name="block", arguments={})]
                )
            second_request_started.set()
            return ModelResponse(items=[AssistantMessage(text="queued done")])

    class _BlockingTool(BaseTool):
        name = "block"
        description = "Block until released."
        input_schema = {"type": "object"}

        async def run(self, context, args):
            del context, args
            tool_started.set()
            await release_tool.wait()
            return "tool done"

    model = _DelayedModelClient()
    tools = ToolRegistry()
    tools.register(_BlockingTool())
    agent = Agent(model, tools, ContextConfig())
    runtime = AgentRuntime(agent)

    background_turn = asyncio.create_task(agent.run_turn(["background"]))
    await tool_started.wait()
    await runtime.start()
    try:
        _submission_id, queued_future = await runtime.enqueue_user_turn(
            "queued",
            queue="steer",
        )
        await asyncio.sleep(0.05)

        assert model.call_count == 1
        assert not second_request_started.is_set()

        release_tool.set()
        with pytest.raises(TurnInterrupted):
            await background_turn
        queued_result = await queued_future

        assert queued_result is not None
        assert queued_result.output_text == "queued done"
        assert model.call_count == 2
        prompt_items = _conversation_items(model.prompts[1])
        assert [type(item).__name__ for item in prompt_items[-4:]] == [
            "UserMessage",
            "ToolCall",
            "ToolResult",
            "UserMessage",
        ]
        assert prompt_items[-4].text == "background"
        assert prompt_items[-3].call_id == "call_1"
        assert prompt_items[-2].call_id == "call_1"
        assert prompt_items[-1].text == "queued"

    finally:
        release_tool.set()
        await asyncio.gather(background_turn, return_exceptions=True)
        await runtime.close()


@pytest.mark.asyncio
async def test_runtime_steer_batches_messages_into_next_request() -> "None":
    first_request_started = asyncio.Event()
    release_first_request = asyncio.Event()

    class _DelayedModelClient:
        model = "test"

        def __init__(self) -> "None":
            self.prompts = []
            self.call_count = 0

        async def complete(self, prompt, event_handler):
            self.prompts.append(prompt)
            self.call_count += 1
            if self.call_count == 1:
                first_request_started.set()
                await release_first_request.wait()
                event_handler(AssistantDeltaEvent("first"))
                return ModelResponse(items=[AssistantMessage(text="first")])

            event_handler(AssistantDeltaEvent("second"))
            return ModelResponse(items=[AssistantMessage(text="second")])

    model = _DelayedModelClient()
    runtime = AgentRuntime(Agent(model, ToolRegistry(), ContextConfig()))
    events = []
    runtime.event_handler = events.append

    await runtime.start()
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
        assert [event.kind for event in events].count("turn_started") == 2
        assert [event.kind for event in events].count("turn_interrupted") == 1
        assert "turn_failed" not in [event.kind for event in events]

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

    finally:
        if not release_first_request.is_set():
            release_first_request.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_agent_emits_turn_failed_event_on_model_error() -> "None":
    events = []

    class FailingModelClient:
        model = "test"

        async def complete(self, prompt, event_handler):
            del prompt, event_handler
            raise RuntimeError("synthetic client error")

    agent = Agent(
        FailingModelClient(),
        ToolRegistry(),
        ContextConfig(),
        event_handler=events.append,
    )

    with pytest.raises(RuntimeError, match="synthetic client error"):
        await agent.run_turn(["hello"])

    assert [event.kind for event in events] == [
        "turn_started",
        "model_called",
        "turn_failed",
    ]
    assert events[-1].error == "synthetic client error"


@pytest.mark.asyncio
async def test_runtime_recovers_after_pre_turn_compact_failure() -> "None":
    def response_factory(prompt, call_count):
        del prompt
        if call_count == 1:
            raise RuntimeError("synthetic compact failure")
        return ModelResponse(items=[AssistantMessage(text="recovered")])

    model = ScriptedModelClient(response_factory=response_factory)
    agent = Agent(
        model,
        ToolRegistry(),
        _auto_compact_context(1),
        initial_history=(UserMessage(text="older prompt"),),
    )
    agent._last_total_usage_tokens = 1
    runtime = AgentRuntime(agent)
    events = []
    runtime.event_handler = events.append
    await runtime.start()
    try:
        with pytest.raises(RuntimeError, match="synthetic compact failure"):
            await asyncio.wait_for(runtime.submit_user_turn("new prompt"), timeout=1.0)

        assert [event.kind for event in events] == [
            "session_state",
            "turn_started",
            "auto_compact_started",
            "auto_compact_failed",
            "turn_failed",
        ]
        assert not agent.is_running
        assert agent.history == (UserMessage(text="older prompt"),)
        await asyncio.wait_for(agent.wait_until_idle(), timeout=1.0)

        result = await asyncio.wait_for(
            runtime.submit_user_turn("retry prompt"), timeout=1.0
        )
        assert result.output_text == "recovered"
        assert model.call_count == 3
    finally:
        await asyncio.wait_for(runtime.close(), timeout=1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failed_event", ["turn_started", "turn_completed", "turn_failed"]
)
async def test_agent_recovers_after_event_handler_failure(failed_event) -> "None":
    def response_factory(prompt, call_count):
        del prompt
        if call_count == 1 and failed_event == "turn_failed":
            raise ValueError("synthetic model failure")
        return ModelResponse(items=[AssistantMessage(text="done")])

    def handle_event(event):
        if event.kind == failed_event:
            raise RuntimeError("synthetic event failure")

    model = ScriptedModelClient(response_factory=response_factory)
    agent = Agent(model, ToolRegistry(), ContextConfig(), event_handler=handle_event)
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    observer_errors = []
    loop.set_exception_handler(lambda _loop, context: observer_errors.append(context))
    try:
        if failed_event == "turn_failed":
            with pytest.raises(ValueError, match="synthetic model failure"):
                await agent.run_turn(["first prompt"])
        else:
            assert (await agent.run_turn(["first prompt"])).output_text == "done"
    finally:
        loop.set_exception_handler(previous_handler)
    assert len(observer_errors) == 1
    assert str(observer_errors[0]["exception"]) == "synthetic event failure"

    assert not agent.is_running
    await asyncio.wait_for(agent.wait_until_idle(), timeout=1.0)
    agent.event_handler = lambda _event: None
    result = await agent.run_turn(["retry prompt"])
    assert result.output_text == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("compact_limit", [None, 1])
async def test_agent_failed_turn_releases_idle_waiters(compact_limit) -> "None":
    request_started = asyncio.Event()
    release_request = asyncio.Event()

    async def response_factory(prompt, call_count):
        del prompt, call_count
        request_started.set()
        await release_request.wait()
        raise ValueError("synthetic failure")

    events = []
    agent = Agent(
        ScriptedModelClient(response_factory=response_factory),
        ToolRegistry(),
        _auto_compact_context(compact_limit),
        event_handler=events.append,
        initial_history=(UserMessage(text="older prompt"),),
    )
    agent._last_total_usage_tokens = 1
    turn = asyncio.create_task(agent.run_turn(["prompt"]))
    waiters = []
    try:
        await asyncio.wait_for(request_started.wait(), timeout=1.0)
        assert agent.is_running
        waiters.append(asyncio.create_task(agent.wait_until_idle()))
        await asyncio.sleep(0)
        assert not waiters[0].done()

        release_request.set()
        with pytest.raises(ValueError, match="synthetic failure"):
            await turn
        await asyncio.wait_for(waiters[0], timeout=1.0)
        assert not agent.is_running
        assert "turn_failed" in [event.kind for event in events]
    finally:
        release_request.set()
        await asyncio.gather(turn, *waiters, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_model", [False, True])
async def test_agent_idle_waiters_are_independent_of_turn_result(fail_model) -> "None":
    request_started = asyncio.Event()
    release_request = asyncio.Event()

    async def response_factory(prompt, call_count):
        del prompt, call_count
        request_started.set()
        await release_request.wait()
        if fail_model:
            raise RuntimeError("synthetic model failure")
        return ModelResponse(items=[AssistantMessage(text="done")])

    agent = Agent(
        ScriptedModelClient(response_factory=response_factory),
        ToolRegistry(),
        ContextConfig(),
    )
    turn = asyncio.create_task(agent.run_turn(["prompt"]))
    waiters = []
    try:
        await asyncio.wait_for(request_started.wait(), timeout=1.0)
        waiters = [asyncio.create_task(agent.wait_until_idle()) for _index in range(3)]
        await asyncio.sleep(0)
        assert not any(waiter.done() for waiter in waiters)

        waiters[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiters[0]
        assert not turn.done()
        assert agent.is_running
        release_request.set()

        if fail_model:
            with pytest.raises(RuntimeError, match="synthetic model failure"):
                await turn
        else:
            assert (await turn).output_text == "done"
        await asyncio.wait_for(asyncio.gather(*waiters[1:]), timeout=1.0)
        assert not agent.is_running
    finally:
        release_request.set()
        await asyncio.gather(turn, *waiters, return_exceptions=True)


@pytest.mark.asyncio
async def test_agent_run_turn_rejects_overlapping_execution() -> "None":
    started = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, call_count):
        started.set()
        await release.wait()
        return ModelResponse([AssistantMessage("done")])

    model = ScriptedModelClient(response_factory=respond)
    agent = Agent(model, ToolRegistry(), ContextConfig())
    turn = asyncio.create_task(agent.run_turn(["first prompt"]))
    try:
        await asyncio.wait_for(started.wait(), 1)
        assert agent.is_running
        with pytest.raises(RuntimeError, match="agent already has an active turn"):
            await agent.run_turn(["overlapping prompt"])
        assert not await agent.maybe_invoke({"type": "exec_command_completed"})
        release.set()
        assert (await turn).output_text == "done"
        assert not agent.is_running
        assert model.call_count == 1
        assert agent.history[0].text == "first prompt"
    finally:
        release.set()
        await asyncio.gather(turn, return_exceptions=True)


@pytest.mark.asyncio
async def test_agent_shutdown_does_not_cancel_current_turn() -> "None":
    started = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, call_count):
        started.set()
        await release.wait()
        return ModelResponse([AssistantMessage("done")])

    model = ScriptedModelClient(response_factory=respond)
    agent = Agent(model, ToolRegistry(), ContextConfig())
    turn = asyncio.create_task(agent.run_turn(["finish this prompt"]))
    try:
        await asyncio.wait_for(started.wait(), 1)
        agent.shutdown()
        assert agent.is_running
        assert agent.is_shutdown
        assert not await agent.maybe_invoke({"type": "clock_tick"})
        with pytest.raises(RuntimeError, match="shutdown"):
            await agent.run_turn(["new prompt"])
        release.set()
        assert (await turn).output_text == "done"
        assert not agent.is_running
        assert model.call_count == 1
    finally:
        release.set()
        await asyncio.gather(turn, return_exceptions=True)


@pytest.mark.asyncio
async def test_agent_does_not_consume_runtime_inputs_without_a_worker() -> "None":
    model = ScriptedModelClient(
        [
            ModelResponse([AssistantMessage("first")]),
            ModelResponse([AssistantMessage("next")]),
        ]
    )
    agent = Agent(model, ToolRegistry(), ContextConfig())
    queue = AgentRuntime(agent)
    turn = asyncio.create_task(agent.run_turn(["first prompt"]))
    _submission_id, steered = await queue.enqueue_user_turn(
        "next prompt", queue="steer"
    )
    result = await turn

    assert not agent.is_running
    assert model.call_count == 1
    assert result.output_text == "first"
    assert not steered.done()
    assert [
        item.text for item in model.prompts[0].input if isinstance(item, UserMessage)
    ] == [
        "first prompt",
    ]
    await queue.start()
    try:
        assert (await asyncio.wait_for(steered, 1)).output_text == "next"
        assert [
            item.text
            for item in model.prompts[1].input
            if isinstance(item, UserMessage)
        ] == [
            "first prompt",
            "next prompt",
        ]
    finally:
        await queue.close()


@pytest.mark.asyncio
async def test_agent_keeps_done_incomplete_items_without_unresolved_tool_calls() -> (
    "None"
):
    class PartialThenContinueModelClient:
        model = "test"

        def __init__(self) -> "None":
            self.prompts = []
            self.call_count = 0

        async def complete(self, prompt, event_handler):
            self.prompts.append(prompt)
            self.call_count += 1
            if self.call_count == 1:
                event_handler(AssistantDeltaEvent("partial answer"))
                raise ResponsesIncompleteError(
                    "responses stream ended with `response.incomplete`",
                    [
                        AssistantMessage(text="partial answer"),
                        ToolCall(call_id="unfinished", name="echo", arguments={}),
                    ],
                    reason="max_output_tokens",
                )
            if self.call_count == 2:
                return ModelResponse(items=[AssistantMessage(text="continued answer")])
            raise AssertionError(f"unexpected call_count={self.call_count}")

    model = PartialThenContinueModelClient()
    events = []
    agent = Agent(model, ToolRegistry(), ContextConfig(), event_handler=events.append)

    with pytest.raises(ResponsesIncompleteError):
        await agent.run_turn(["write a long answer"])

    assert [type(item).__name__ for item in agent.history] == [
        "UserMessage",
        "AssistantMessage",
    ]
    assert agent.history[0].text == "write a long answer"
    assert "turn_failed" in [event.kind for event in events]

    result = await agent.run_turn(["continue"])

    assert result.output_text == "continued answer"
    follow_up_items = _conversation_items(model.prompts[1])
    assert [type(item).__name__ for item in follow_up_items] == [
        "UserMessage",
        "AssistantMessage",
        "UserMessage",
    ]
    assert follow_up_items[0].text == "write a long answer"
    assert follow_up_items[1].text == "partial answer"
    assert follow_up_items[2].text == "continue"


@pytest.mark.asyncio
async def test_agent_keeps_reasoning_done_item_from_incomplete_response() -> "None":
    class ReasoningThenContinueModelClient:
        model = "test"

        def __init__(self) -> "None":
            self.prompts = []
            self.call_count = 0

        async def complete(self, prompt, event_handler):
            self.prompts.append(prompt)
            self.call_count += 1
            if self.call_count == 1:
                raise ResponsesIncompleteError(
                    "responses stream ended with `response.incomplete`",
                    [
                        ReasoningItem(
                            payload={
                                "type": "reasoning",
                                "id": "rs_1",
                                "summary": [],
                                "encrypted_content": "encrypted",
                            }
                        )
                    ],
                    reason="max_output_tokens",
                )
            if self.call_count == 2:
                return ModelResponse(items=[AssistantMessage(text="continued answer")])
            raise AssertionError(f"unexpected call_count={self.call_count}")

    model = ReasoningThenContinueModelClient()
    agent = Agent(model, ToolRegistry(), ContextConfig())

    with pytest.raises(ResponsesIncompleteError):
        await agent.run_turn(["think for a while"])

    assert [type(item).__name__ for item in agent.history] == [
        "UserMessage",
        "ReasoningItem",
    ]

    await agent.run_turn(["continue"])

    follow_up_items = _conversation_items(model.prompts[1])
    assert [type(item).__name__ for item in follow_up_items] == [
        "UserMessage",
        "ReasoningItem",
        "UserMessage",
    ]


@pytest.mark.asyncio
async def test_agent_emits_token_count_for_context_length_error() -> "None":
    events = []
    error_message = _context_length_error_message()

    class FailingModelClient:
        model = "test"

        async def complete(self, prompt, event_handler):
            del prompt, event_handler
            raise ContextLengthExceeded(error_message)

    agent = Agent(
        FailingModelClient(),
        ToolRegistry(),
        ContextConfig(),
        event_handler=events.append,
    )

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
    assert events[2].usage == {
        "total_tokens": 264568,
        "input_tokens": 264568,
        "output_tokens": 0,
    }
    assert events[3].phase == "context_length_exceeded"
    assert events[3].total_tokens == 264568
    assert events[3].token_limit == 262144


@pytest.mark.asyncio
async def test_agent_auto_compacts_and_retries_on_context_length_error() -> "None":
    events = []

    def response_factory(prompt, call_count):
        if call_count == 1:
            raise ContextLengthExceeded(_context_length_error_message())
        if call_count == 2:
            return ModelResponse(items=[AssistantMessage(text="checkpoint summary")])
        if call_count == 3:
            return ModelResponse(items=[AssistantMessage(text="final answer")])
        raise AssertionError(f"unexpected call_count={call_count}")

    model = ScriptedModelClient(response_factory=response_factory)
    agent = Agent(model, ToolRegistry(), ContextConfig(), event_handler=events.append)

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
    ]
    assert (
        retry_prompt_items[0]
        == compact(
            [AssistantMessage("checkpoint summary")], str(agent.session_file_path)
        )[0]
    )

    assert "turn_failed" not in [event.kind for event in events]
    auto_events = [event for event in events if event.kind.startswith("auto_compact_")]
    assert [event.kind for event in auto_events] == [
        "auto_compact_started",
        "auto_compact_completed",
    ]
    assert auto_events[0].phase == "context_length_exceeded"
    assert auto_events[1].summary == ("compact(1 item) -> 0 items + [summary]")


@pytest.mark.asyncio
async def test_agent_auto_compacts_on_context_window_error_without_token_counts() -> (
    "None"
):
    events = []
    error_message = (
        "ResponsesApiError: responses stream failed on the server side\n"
        "- detail: Your input exceeds the context window of this model. "
        "Please adjust your input and try again."
    )

    def response_factory(prompt, call_count):
        if call_count == 1:
            raise ContextLengthExceeded(error_message)
        if call_count == 2:
            return ModelResponse(items=[AssistantMessage(text="checkpoint summary")])
        if call_count == 3:
            return ModelResponse(items=[AssistantMessage(text="final answer")])
        raise AssertionError(f"unexpected call_count={call_count}")

    model = ScriptedModelClient(response_factory=response_factory)
    agent = Agent(model, ToolRegistry(), ContextConfig(), event_handler=events.append)

    result = await agent.run_turn(["hello"])

    assert result.output_text == "final answer"
    assert [event.kind for event in events if event.kind == "token_count"] == []
    auto_events = [event for event in events if event.kind.startswith("auto_compact_")]
    assert [event.kind for event in auto_events] == [
        "auto_compact_started",
        "auto_compact_completed",
    ]
    assert auto_events[0].phase == "context_length_exceeded"
    assert auto_events[0].total_tokens is None
    assert auto_events[0].token_limit is None


@pytest.mark.asyncio
async def test_agent_prunes_old_tool_responses_when_context_compact_overflows() -> (
    "None"
):
    events = []
    initial_history = (
        UserMessage(text="old prompt"),
        ToolCall(call_id="call_old", name="echo", arguments={"text": "large"}),
        ToolResult(call_id="call_old", name="echo", output="large output"),
        AssistantMessage(text="old answer"),
    )

    def response_factory(prompt, call_count):
        if call_count == 1:
            raise ContextLengthExceeded(_context_length_error_message())
        if call_count == 2:
            compact_items = _conversation_items(prompt)
            assert any(isinstance(item, ToolResult) for item in compact_items)
            raise ContextLengthExceeded(_context_length_error_message())
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
        ContextConfig(),
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
    assert auto_completed.pruned_tool_results == 1
    assert auto_completed.summary == (
        "compact(5 items) -> 0 items + [summary] " "(dropped 1 old tool response)"
    )


@pytest.mark.asyncio
async def test_agent_relays_stream_error_events() -> "None":
    events = []

    class RetryingModelClient:
        model = "test"

        async def complete(self, prompt, event_handler):
            del prompt
            event_handler(
                StreamErrorEvent("Reconnecting... 1/5", 1, 5, 0, "disconnected")
            )
            return ModelResponse(items=[AssistantMessage(text="done")])

    agent = Agent(
        RetryingModelClient(),
        ToolRegistry(),
        ContextConfig(),
        event_handler=events.append,
    )

    result = await agent.run_turn(["hello"])

    assert result.output_text == "done"
    assert [event.kind for event in events] == [
        "turn_started",
        "model_called",
        "stream_error",
        "model_completed",
        "turn_completed",
    ]
    assert events[2].message == "Reconnecting... 1/5"
