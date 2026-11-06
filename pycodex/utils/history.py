"""Pure conversation-history projections, independent of rollout storage."""

import typing
from dataclasses import replace

from ..protocol import AgentMessage, AssistantMessage, ConversationItem, UserMessage


def serialize_history_item(item: "ConversationItem") -> "typing.Dict[str, object]":
    """Include local message metadata without changing model serialization."""
    payload = item.serialize()
    if isinstance(item, UserMessage):
        payload["name"] = item.name
    if isinstance(item, (UserMessage, AgentMessage, AssistantMessage)):
        payload["visible"] = item.visible
    return payload


def restore_visibility(item, payload):
    """Older rollouts use the message type's visibility default."""
    visible = payload.get("visible", item.visible)
    if not isinstance(visible, bool):
        raise ValueError("history item visible must be a boolean")
    return replace(item, visible=visible)


def iter_display_turns(history):
    """Project message visibility and grouping; None marks a compaction."""
    for entry in iter_timed_display_turns((item, None) for item in history):
        yield None if entry is None else entry[:2]


def iter_timed_display_turns(history):
    """Like iter_display_turns for (item, timestamp) pairs.

    Yields (prompt, response, prompt_at, response_at); the times come from the
    items that supplied the visible prompt and response.
    """
    prompt = None
    response = ""
    phase = None
    prompt_at = response_at = None
    for item, timestamp in history:
        if item is None:
            if prompt is not None or response:
                yield (prompt or "", response, prompt_at, response_at)
            yield None
            prompt, response, phase = None, "", None
            prompt_at = response_at = None
        elif isinstance(item, (UserMessage, AgentMessage)):
            # Hidden mail must not split commentary from its final answer.
            # Keep an existing unphased reply when the next input arrives.
            if item.visible or (response and phase != "commentary"):
                if prompt is not None or response:
                    yield (prompt or "", response, prompt_at, response_at)
                prompt, response, phase = None, "", None
                prompt_at = response_at = None
            if item.visible:
                prompt, prompt_at = item.text, timestamp
        elif isinstance(item, AssistantMessage) and item.visible:
            response, phase, response_at = item.text, item.phase, timestamp
            if phase == "final":
                yield (prompt or "", response, prompt_at, response_at)
                prompt, response, phase = None, "", None
                prompt_at = response_at = None
    if prompt is not None or response:
        yield (prompt or "", response, prompt_at, response_at)


def conversation_history_to_turns(
    history: "typing.Iterable[ConversationItem]",
) -> "typing.Tuple[typing.Tuple[str, str], ...]":
    return tuple(entry for entry in iter_display_turns(history) if entry is not None)
