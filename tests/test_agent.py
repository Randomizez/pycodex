from __future__ import annotations

import asyncio
import time

import pytest

from pycodex import (
    AgentLoop,
    AgentRuntime,
    AssistantMessage,
    BaseTool,
    ModelResponse,
    ModelStreamEvent,
    ToolCall,
    ToolRegistry,
    ToolResult,
    UserMessage,
)
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

    def __init__(self, name: str) -> None:
        self.name = name

    async def run(self, context, args):
        del context, args
        await asyncio.sleep(0.05)
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


@pytest.mark.asyncio
async def test_agent_loop_runs_tool_then_returns_final_message() -> None:
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

    agent = AgentLoop(model, tools)
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
async def test_parallel_tools_share_one_model_round() -> None:
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
    tools.register(SlowTool("slow_a"))
    tools.register(SlowTool("slow_b"))

    agent = AgentLoop(model, tools)

    started = time.perf_counter()
    result = await agent.run_turn(["并行跑两个工具"])
    elapsed = time.perf_counter() - started

    assert result.output_text == "两个工具都执行完了"
    assert elapsed < 0.11


@pytest.mark.asyncio
async def test_agent_loop_default_has_no_fixed_iteration_cap() -> None:
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

    agent = AgentLoop(model, tools)
    result = await agent.run_turn(["连续调用工具直到结束"])

    assert result.output_text == "超过 12 轮后也收敛了"
    assert result.iterations == 13


@pytest.mark.asyncio
async def test_wait_agent_injects_subagent_notification_into_history() -> None:
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

    agent = AgentLoop(model, tools)
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
async def test_runtime_submission_loop_processes_turn_and_shutdown() -> None:
    model = ScriptedModelClient([ModelResponse(items=[AssistantMessage(text="done")])])
    tools = ToolRegistry()
    agent = AgentLoop(model, tools)
    runtime = AgentRuntime(agent)

    worker = asyncio.create_task(runtime.run_forever())
    try:
        result = await runtime.submit_user_turn("hello")
        assert result.output_text == "done"
        await runtime.shutdown()
    finally:
        await worker


@pytest.mark.asyncio
async def test_runtime_steer_batches_messages_into_next_request() -> None:
    first_request_started = asyncio.Event()
    release_first_request = asyncio.Event()

    class _DelayedModelClient:
        def __init__(self) -> None:
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
    runtime = AgentRuntime(AgentLoop(model, ToolRegistry()))

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
async def test_agent_loop_emits_turn_failed_event_on_model_error() -> None:
    events = []

    class FailingModelClient:
        async def complete(self, prompt, event_handler):
            del prompt, event_handler
            raise RuntimeError("synthetic client error")

    agent = AgentLoop(FailingModelClient(), ToolRegistry(), event_handler=events.append)

    with pytest.raises(RuntimeError, match="synthetic client error"):
        await agent.run_turn(["hello"])

    assert [event.kind for event in events] == [
        "turn_started",
        "model_called",
        "turn_failed",
    ]
    assert events[-1].payload["error"] == "synthetic client error"
