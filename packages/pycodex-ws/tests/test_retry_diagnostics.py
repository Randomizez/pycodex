from pycodex import Agent, AgentRuntime, ContextConfig, ToolRegistry
from pycodex.events import StreamErrorEvent, TurnStartedEvent
from tests.fakes import ScriptedModelClient
from workspace_server import WebSessionView


async def test_retry_details_survive_active_turn_reattachment():
    runtime = AgentRuntime(Agent(
        ScriptedModelClient([], model="primary"), ToolRegistry(), ContextConfig(),
    ))
    view = WebSessionView()
    frontend = runtime.attach(view.handle_event)
    runtime._handle_agent_event(TurnStartedEvent("turn", ("hello",)))
    runtime.agent._handle_model_stream_event("turn", StreamErrorEvent(
        "Reconnecting... 1/1", 1, 1, 0.0, "upstream connection timed out",
    ))
    expected = ["Reconnecting... 1/1\nupstream connection timed out"]
    assert view.snapshot()["turns"][-1]["model_errors"] == expected
    restored = WebSessionView()
    attached = runtime.attach(restored.handle_event)
    assert restored.snapshot()["turns"][-1]["model_errors"] == expected
    assert restored.snapshot()["turns"][-1]["response"] == ""
    runtime.detach(frontend)
    runtime.detach(attached)
    await runtime.close()
