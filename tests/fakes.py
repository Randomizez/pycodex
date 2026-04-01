from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable

from pycodex.model import NOOP_MODEL_STREAM_EVENT_HANDLER
from pycodex.protocol import AssistantMessage, ModelResponse, ModelStreamEvent, Prompt, ToolCall

ResponseFactory = Callable[[Prompt, int], ModelResponse | Awaitable[ModelResponse]]


class ScriptedModelClient:
    """Test double that returns pre-recorded responses in order."""

    def __init__(
        self,
        responses: Iterable[ModelResponse] | None = None,
        response_factory: ResponseFactory | None = None,
    ) -> None:
        if responses is None and response_factory is None:
            raise ValueError("either responses or response_factory must be provided")
        self._responses = iter(responses or [])
        self._response_factory = response_factory
        self.prompts: list[Prompt] = []
        self.call_count = 0

    async def complete(
        self,
        prompt: Prompt,
        event_handler: Callable[[ModelStreamEvent], None] = NOOP_MODEL_STREAM_EVENT_HANDLER,
    ) -> ModelResponse:
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
                event_handler(
                    ModelStreamEvent(
                        kind="assistant_delta",
                        payload={"delta": item.text},
                    )
                )
            elif isinstance(item, ToolCall):
                event_handler(
                    ModelStreamEvent(
                        kind="tool_call",
                        payload={
                            "call_id": item.call_id,
                            "tool_name": item.name,
                        },
                    )
                )

        return final_response
