import asyncio
import typing
from typing import Awaitable, Callable, Iterable

from pycodex import Agent, AgentRuntime, AgentsManager, ContextConfig
from pycodex.events import AssistantDeltaEvent, ModelEvent, ToolCalledEvent
from pycodex.model import NOOP_MODEL_STREAM_EVENT_HANDLER, ModelRecordingMetadata
from pycodex.protocol import AssistantMessage, ModelResponse, Prompt, ToolCall
from pycodex.tools import get_tools
from pycodex.utils import uuid7_string

ResponseFactory = Callable[
    [Prompt, int], typing.Union[ModelResponse, Awaitable[ModelResponse]]
]


class ScriptedModelClient:
    """Test double that returns pre-recorded responses in order."""

    def __init__(
        self,
        responses: "typing.Union[Iterable[ModelResponse], None]" = None,
        response_factory: "typing.Union[ResponseFactory, None]" = None,
        model: "str" = "scripted",
    ) -> "None":
        if responses is None and response_factory is None:
            raise ValueError("either responses or response_factory must be provided")
        self._response_items = tuple(responses or ())
        self._responses = iter(self._response_items)
        self._response_factory = response_factory
        self.model = model
        self.reasoning_effort = None
        self.endpoint = "in-process"
        self.recording_metadata = ModelRecordingMetadata("pycodex", None)
        self.prompts: "typing.List[Prompt]" = []
        self.call_count = 0

    def with_overrides(
        self, model, reasoning_effort, session_id=None, openai_subagent=None,
    ):
        client = ScriptedModelClient(
            self._response_items, self._response_factory, model or self.model,
        )
        client.reasoning_effort = (
            self.reasoning_effort if reasoning_effort is None else reasoning_effort
        )
        client._session_id = session_id
        client._openai_subagent = openai_subagent
        return client

    async def complete(
        self,
        prompt: "Prompt",
        event_handler: "Callable[[ModelEvent], None]" = NOOP_MODEL_STREAM_EVENT_HANDLER,
    ) -> "ModelResponse":
        self.prompts.append(prompt)
        self.call_count += 1

        if self._response_factory is not None:
            response = self._response_factory(prompt, self.call_count)
            if isinstance(response, ModelResponse):
                final_response = response
            else:
                final_response = await response
        else:
            try:
                final_response = next(self._responses)
            except StopIteration as exc:
                raise RuntimeError("scripted model ran out of responses") from exc

        for item in final_response.items:
            if isinstance(item, AssistantMessage):
                event_handler(AssistantDeltaEvent(item.text))
            elif isinstance(item, ToolCall):
                event_handler(ToolCalledEvent(item.call_id, item.name))

        return final_response


def make_agent(model_client, tool_registry, context_config, session_id=None, **options):
    """Construct a standalone test Agent with an explicit, independent identity."""
    return Agent(
        model_client, tool_registry, context_config,
        session_id=uuid7_string() if session_id is None else session_id, **options,
    )


def make_runtime(agent):
    """Register a test Agent with custom tools in its own coordination tree."""
    manager = AgentsManager(agent.model_client, agent.context_manager.config)
    runtime = AgentRuntime(agent, manager)
    manager._runtimes[agent.agent_id] = runtime
    agent.context_manager.agent_path = agent.agent_path
    return runtime


def make_tools(runtime_environment=None, toolset=None):
    """Build a real registry for isolated tool and schema tests."""
    manager = AgentsManager(ScriptedModelClient([]), ContextConfig())
    return get_tools(manager, runtime_environment, toolset)


async def wait_for_runtime_idle(runtime):
    """Wait for accepted work, including a trigger not yet picked up by the worker."""
    async def wait():
        while runtime.is_busy:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), 2)
