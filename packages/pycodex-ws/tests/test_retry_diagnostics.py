import asyncio

from pycodex import AssistantMessage, ContextConfig, ModelResponse, ToolRegistry
from pycodex.events import EventScope, StreamErrorEvent
from tests.fakes import ScriptedModelClient, make_agent, make_runtime
from workspace_server import WebSessionView


async def test_retry_details_survive_active_turn_reattachment():
    reported = asyncio.Event()
    release = asyncio.Event()

    class RetryingClient(ScriptedModelClient):
        async def complete(self, prompt):
            EventScope.emit(StreamErrorEvent(
                "Reconnecting... 1/1", 1, 1, 0.0, "upstream connection timed out",
            ))
            reported.set()
            await release.wait()
            return await super().complete(prompt)

    runtime = make_runtime(make_agent(
        RetryingClient([ModelResponse([AssistantMessage("answer")])]), ToolRegistry(), ContextConfig(),
    ))
    view = WebSessionView()
    runtime.attach(view.handle_event)
    await runtime.start()
    try:
        receipt = await runtime.submit_input("hello")
        await asyncio.wait_for(reported.wait(), 2)
        expected = ["Reconnecting... 1/1\nupstream connection timed out"]
        assert view.snapshot()["turns"][-1]["model_errors"] == expected
        restored = WebSessionView()
        runtime.attach(restored.handle_event)
        assert restored.snapshot()["turns"][-1]["model_errors"] == expected
        assert restored.snapshot()["turns"][-1]["response"] == ""
        release.set()
        assert (await receipt.future).output_text == "answer"
    finally:
        release.set()
        await runtime.close()
