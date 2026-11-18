import asyncio
import json
import threading

import pytest
import requests

from pycodex import BaseTool, ContextConfig, ResponsesModelClient, ResponsesProviderConfig, ToolRegistry
from pycodex.model import ResponsesApiError, ResponsesRetryableError
from pycodex.protocol import Prompt, ToolCall, ToolResult, UserMessage
from pycodex.runtime import SubmissionInterrupted
from tests.fakes import make_agent, make_runtime, wait_for_runtime_idle
from tests.test_model_fallback import Response


def make_model(monkeypatch, retries=0, fallback_models=(), delay=0):
    monkeypatch.setattr(ResponsesModelClient, "_retry_delay_seconds", lambda *args: delay)
    return ResponsesModelClient(ResponsesProviderConfig(
        model="test", provider_name="test", base_url="https://example.invalid/v1",
        api_key_env=None, stream_max_retries=retries, fallback_models=fallback_models,
    ))


def answer():
    return Response(events=[
        {"type": "response.output_item.done", "item": {
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "done"}],
        }},
        {"type": "response.completed"},
    ])


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [400, 403, 404, 429, 503, "sse-int", "sse-str"])
async def test_selected_code_retries_beyond_budget_on_the_same_model(monkeypatch, failure):
    model = make_model(monkeypatch, fallback_models=("backup",))
    code = failure if isinstance(failure, int) else 429
    model.set_retry_error_code(code)
    bodies = []

    def send(self, request, **kwargs):
        bodies.append(json.loads(request.body))
        if len(bodies) > 7:
            return answer()
        if isinstance(failure, int):
            return Response(failure)
        return Response(events=[{
            "type": "response.failed",
            "response": {"error": {
                "code": 429 if failure == "sse-int" else "429",
                "message": "unavailable",
            }},
        }])

    monkeypatch.setattr(requests.Session, "send", send)
    events = []
    result = await model.complete(Prompt(input=[UserMessage("hello")], tools=[]), events.append)

    assert result.items[-1].text == "done"
    assert len(bodies) == 8
    assert all(body == bodies[0] for body in bodies)
    notices = [event for event in events if event.kind == "stream_error"]
    assert [event.attempt for event in notices] == list(range(1, 8))
    assert [event.message for event in notices] == [
        "Reconnecting... {0}/∞".format(attempt) for attempt in range(1, 8)
    ]
    assert model.last_error is None
    assert model.retry_error_code == code
    assert model.with_overrides().retry_error_code is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [400, 503, "transport", "symbolic"])
async def test_other_errors_raise_without_retry_or_fallback(monkeypatch, failure):
    model = make_model(monkeypatch, retries=5, fallback_models=("backup",))
    model.set_retry_error_code(429)
    attempts = []

    def send(self, request, **kwargs):
        attempts.append(request)
        if failure == "transport":
            raise requests.ConnectionError("offline")
        if failure == "symbolic":
            return Response(events=[{
                "type": "response.failed",
                "response": {"error": {"code": "rate_limit_exceeded", "message": "unavailable"}},
            }])
        return Response(failure)

    monkeypatch.setattr(requests.Session, "send", send)
    with pytest.raises(ResponsesApiError) as failed:
        await model.complete(Prompt(input=[], tools=[]))
    assert len(attempts) == 1
    assert model.last_error is failed.value
    assert not failed.value.matches_code(429)


@pytest.mark.asyncio
@pytest.mark.parametrize("command,status", [("/retry", 400), ("/retry 429", 429)])
async def test_retry_command_continues_failed_sample_without_replaying_tools(monkeypatch, command, status):
    model = make_model(monkeypatch)
    executions = []

    class CountTool(BaseTool):
        name = "count"
        description = "Count one execution."
        input_schema = {"type": "object"}

        async def run(self, context, args):
            executions.append("count")
            return "counted"

    tools = ToolRegistry()
    tools.register(CountTool())
    runtime = make_runtime(make_agent(model, tools, ContextConfig()))
    bodies = []

    def send(self, request, **kwargs):
        bodies.append(json.loads(request.body))
        if len(bodies) == 1:
            return Response(events=[
                {"type": "response.output_item.done", "item": {
                    "type": "function_call", "name": "count", "call_id": "count-once", "arguments": "{}",
                }},
                {"type": "response.completed"},
            ])
        return Response(status) if len(bodies) == 2 else answer()

    monkeypatch.setattr(requests.Session, "send", send)
    await runtime.start()
    try:
        with pytest.raises(ResponsesApiError):
            await runtime.submit_user_turn("hello")
        failed_history = runtime.agent.history
        receipt = await runtime.submit_input(command)
        result = await receipt.future
        assert result["kind"] == "retry"
        assert result["submission_id"]
        await wait_for_runtime_idle(runtime)

        assert executions == ["count"]
        assert len(bodies) == 3
        assert bodies[1] == bodies[2]
        assert runtime.agent.history[:-1] == failed_history
        assert [item.text for item in runtime.agent.history if isinstance(item, UserMessage)] == ["hello"]
        assert len([item for item in runtime.agent.history if isinstance(item, ToolCall)]) == 1
        assert len([item for item in runtime.agent.history if isinstance(item, ToolResult)]) == 1
        assert runtime.agent.history[-1].text == "done"
        assert model.last_error is None
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["None", "off", "anything", "429x", "²"])
async def test_nonnumeric_command_wakes_infinite_backoff_and_returns_last_error(monkeypatch, value):
    model = make_model(monkeypatch, retries=5, delay=3600)
    runtime = make_runtime(make_agent(model, ToolRegistry(), ContextConfig()))
    retrying = asyncio.Event()
    attempts = []

    def send(self, request, **kwargs):
        attempts.append(request)
        return Response(429)

    def observe(event):
        if event.kind == "stream_error":
            retrying.set()

    monkeypatch.setattr(requests.Session, "send", send)
    runtime.event_handler = observe
    await runtime.start()
    await runtime.submit_input("/retry 429")
    first = await runtime.submit_input("hello")
    try:
        await asyncio.wait_for(retrying.wait(), 2)
        receipt = await runtime.submit_input("/retry " + value)
        assert (await receipt.future)["lines"] == ["Infinite retry disabled."]
        with pytest.raises(ResponsesRetryableError) as failed:
            await asyncio.wait_for(first.future, 2)
        assert model.last_error is failed.value
        assert model.retry_error_code is None
        assert len(attempts) == 1
        assert [item.text for item in runtime.agent.history if isinstance(item, UserMessage)] == ["hello"]
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["stop", "close", "steer"])
async def test_stop_close_and_steer_end_backoff_cooperatively(monkeypatch, action):
    model = make_model(monkeypatch, delay=3600)
    runtime = make_runtime(make_agent(model, ToolRegistry(), ContextConfig()))
    retrying = asyncio.Event()
    attempts = []
    events = []

    def send(self, request, **kwargs):
        attempts.append(request)
        return Response(429) if len(attempts) == 1 else answer()

    def observe(event):
        events.append(event.kind)
        if event.kind == "stream_error":
            retrying.set()

    monkeypatch.setattr(requests.Session, "send", send)
    runtime.event_handler = observe
    await runtime.start()
    await runtime.submit_input("/retry 429")
    first = await runtime.submit_input("hello")
    try:
        await asyncio.wait_for(retrying.wait(), 2)
        if action == "close":
            await asyncio.wait_for(runtime.close(), 2)
        elif action == "steer":
            second = await runtime.submit_input("next")
        else:
            runtime.stop_asap()
        with pytest.raises(SubmissionInterrupted):
            await asyncio.wait_for(first.future, 2)
        assert "turn_interrupted" in events
        assert "turn_failed" not in events
        if action == "steer":
            assert (await asyncio.wait_for(second.future, 2)).output_text == "done"
            assert len(attempts) == 2
        else:
            assert len(attempts) == 1
        assert model.retry_error_code == 429
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["off", "stop", "close"])
async def test_retry_control_waits_for_an_inflight_http_request(monkeypatch, action):
    model = make_model(monkeypatch, retries=5)
    runtime = make_runtime(make_agent(model, ToolRegistry(), ContextConfig()))
    started = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    loop = asyncio.get_running_loop()
    attempts = []

    def send(self, request, **kwargs):
        attempts.append(request)
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        finished.set()
        return Response(429)

    monkeypatch.setattr(requests.Session, "send", send)
    await runtime.start()
    await runtime.submit_input("/retry 429")
    first = await runtime.submit_input("hello")
    closing = None
    try:
        await asyncio.wait_for(started.wait(), 2)
        if action == "off":
            await runtime.submit_input("/retry None")
        elif action == "stop":
            runtime.stop_asap()
        else:
            closing = asyncio.create_task(runtime.close())
            await asyncio.sleep(0)
            assert not closing.done()
        assert not first.future.done()
        assert not finished.is_set()
        release.set()
        expected = ResponsesRetryableError if action == "off" else SubmissionInterrupted
        with pytest.raises(expected):
            await asyncio.wait_for(first.future, 2)
        if closing is not None:
            await asyncio.wait_for(closing, 2)
        assert finished.is_set()
        assert len(attempts) == 1
    finally:
        release.set()
        if closing is not None:
            await closing
        await runtime.close()


@pytest.mark.asyncio
async def test_stop_at_fallback_notice_prevents_the_next_http_request(monkeypatch):
    model = make_model(monkeypatch, fallback_models=("backup",))
    runtime = make_runtime(make_agent(model, ToolRegistry(), ContextConfig()))
    attempts = []

    def send(self, request, **kwargs):
        attempts.append(request)
        return Response(403)

    def observe(event):
        if event.kind == "stream_error":
            runtime.stop_asap()

    monkeypatch.setattr(requests.Session, "send", send)
    runtime.event_handler = observe
    await runtime.start()
    try:
        with pytest.raises(SubmissionInterrupted):
            await asyncio.wait_for(runtime.submit_user_turn("hello"), 2)
        assert len(attempts) == 1
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_retry_can_be_disabled_during_a_command_owned_compaction(monkeypatch):
    model = make_model(monkeypatch, delay=3600)
    runtime = make_runtime(make_agent(
        model, ToolRegistry(), ContextConfig(), initial_history=(UserMessage("old prompt"),),
    ))
    retrying = asyncio.Event()

    def observe(event):
        if event.kind == "stream_error":
            retrying.set()

    monkeypatch.setattr(requests.Session, "send", lambda *args, **kwargs: Response(429))
    runtime.event_handler = observe
    await runtime.start()
    await runtime.submit_input("/retry 429")
    compact = await runtime.submit_input("/compact", wait_for_command=False)
    try:
        await asyncio.wait_for(retrying.wait(), 2)
        disabled = await asyncio.wait_for(runtime.submit_input("/retry None"), 2)
        assert (await disabled.future)["lines"] == ["Infinite retry disabled."]
        with pytest.raises(ResponsesRetryableError):
            await asyncio.wait_for(compact.future, 2)
        assert runtime.agent.history == (UserMessage("old prompt"),)
    finally:
        model.set_retry_error_code(None)
        await runtime.close()


@pytest.mark.asyncio
async def test_retry_without_failure_reports_an_error_and_help_lists_command(monkeypatch):
    model = make_model(monkeypatch)
    runtime = make_runtime(make_agent(model, ToolRegistry(), ContextConfig()))
    try:
        help_receipt = await runtime.submit_input("/help")
        assert "retry" in (await help_receipt.future)["commands"]
        receipt = await runtime.submit_input("/retry")
        with pytest.raises(ValueError, match="No failed model request"):
            await receipt.future
        assert runtime.agent.history == ()
    finally:
        await runtime.close()
