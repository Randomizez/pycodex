
"""Provider-specific post-process hooks for canonical outgoing chat requests.

Each downstream chat-completions provider may have its own payload quirks:
extra fields, removed fields, role normalization, tool-shape tweaks, etc.
Keep all of those provider-specific rewrites here so `StreamRouter` can keep
building one canonical `outcomming_request`, while `server.py` selects the
appropriate hook from `CompatServerConfig.model_provider`.
"""

from copy import deepcopy
from typing import Callable, Optional
import typing
from typing_extensions import TypedDict

ChatMessage = typing.Dict[str, object]


class OutgoingRequest(TypedDict):
    """Canonical downstream `/v1/chat/completions` request shape.

    `model`, `messages`, and `stream` are always populated by
    `StreamRouter.build_outcomming_request(...)`. Provider-specific fields that
    may be omitted use `Optional[...]` here so the schema stays simple and does
    not rely on TypedDict inheritance.
    """

    model: 'str'
    messages: 'typing.List[ChatMessage]'
    stream: 'bool'
    max_tokens: 'Optional[int]'
    tools: 'Optional[typing.List[typing.Dict[str, object]]]'
    tool_choice: 'Optional[object]'
    parallel_tool_calls: 'Optional[bool]'
    return_token_ids: 'Optional[bool]'


PayloadPostProcessor = Callable[[OutgoingRequest], OutgoingRequest]


def _identity(outcomming_request: 'OutgoingRequest') -> 'OutgoingRequest':
    """Keep the canonical request unchanged."""

    return outcomming_request


def _drop_developer_messages(outcomming_request: 'OutgoingRequest') -> 'OutgoingRequest':
    """Remove all developer-role messages for providers that reject them."""

    outcomming_request["messages"] = [
        message
        for message in outcomming_request["messages"]
        if message.get("role") != "developer"
    ]
    return outcomming_request

def _replace_developer_messages(outcomming_request: 'OutgoingRequest') -> 'OutgoingRequest':
    """Replace all developer-role messages to system-role messages"""

    for message in outcomming_request['messages']:
        if message.get("role") == "developer":
            message['role'] = "system"

    return outcomming_request


PAYLOAD_POST_PROCESSORS: 'typing.Dict[str, PayloadPostProcessor]' = {
    "stepfun": _replace_developer_messages,
    "vllm": _identity,
}
"""Mapping from normalized `model_provider` name to payload rewrite hook."""


def post_process_outcomming_request(
    outcomming_request: 'OutgoingRequest',
    model_provider: 'typing.Union[str, None]',
) -> 'OutgoingRequest':
    """Apply the provider-specific payload hook to one outgoing request.

    This is the single wrapper around `PAYLOAD_POST_PROCESSORS`: it normalizes
    the provider name, falls back to the default `vllm` behavior when the
    provider is missing or unknown, deep-copies the canonical request, applies
    the selected hook, and validates that the hook returns another request dict.
    """

    processed_request = deepcopy(outcomming_request)
    provider_name = str(model_provider or "").strip().lower()
    provider_processor = PAYLOAD_POST_PROCESSORS.get(
        provider_name,
        PAYLOAD_POST_PROCESSORS.get("vllm"),
    )
    if provider_processor is None:
        return processed_request
    processed_request = provider_processor(processed_request)
    if not isinstance(processed_request, dict):
        raise TypeError("payload processor must return a dict outcomming_request")
    return processed_request
