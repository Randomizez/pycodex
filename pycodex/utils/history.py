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
    prompt = None
    response = ""
    phase = None
    for item in history:
        if item is None:
            if prompt is not None or response:
                yield (prompt or "", response)
            yield None
            prompt, response, phase = None, "", None
        elif isinstance(item, (UserMessage, AgentMessage)):
            # Hidden mail must not split commentary from its final answer.
            # Keep an existing unphased reply when the next input arrives.
            if item.visible or (response and phase != "commentary"):
                if prompt is not None or response:
                    yield (prompt or "", response)
                prompt, response, phase = None, "", None
            if item.visible:
                prompt = item.text
        elif isinstance(item, AssistantMessage) and item.visible:
            response, phase = item.text, item.phase
            if phase == "final":
                yield (prompt or "", response)
                prompt, response, phase = None, "", None
    if prompt is not None or response:
        yield (prompt or "", response)


def conversation_history_to_turns(
    history: "typing.Iterable[ConversationItem]",
) -> "typing.Tuple[typing.Tuple[str, str], ...]":
    return tuple(entry for entry in iter_display_turns(history) if entry is not None)
