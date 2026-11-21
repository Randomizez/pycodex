"""Message storage metadata and pure display conversion."""

import json
from dataclasses import replace
from hashlib import sha256

from ..protocol import AgentMessage, AssistantMessage, UserMessage


def serialize_history_item(item):
    payload = item.serialize()
    if isinstance(item, UserMessage):
        payload["name"] = item.name
    if isinstance(item, (UserMessage, AgentMessage, AssistantMessage)):
        payload.update(message_id=item.message_id, created_at=item.created_at,
                       turn_id=item.turn_id, visible=item.visible)
    return payload


def restore_message_metadata(item, payload, record_id, timestamp=None):
    visible = payload.get("visible", item.visible)
    if not isinstance(visible, bool) and (
        not isinstance(visible, str) or not visible.strip()
    ):
        raise ValueError("history item visible must be a boolean or non-empty string")
    legacy_id = sha256(json.dumps(
        [record_id, timestamp, payload], ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")).hexdigest()[:32]
    return replace(
        item, visible=visible, message_id=payload.get("message_id", "legacy-" + legacy_id),
        created_at=payload.get("created_at", timestamp), turn_id=payload.get("turn_id"),
    )


def compaction_prefix_id(turn):
    first = turn["messages"][0].message_id if turn["messages"] else turn["response_id"]
    return "part-" + first


def conversation_history_to_turns(history, previous=()):
    """Convert a transcript or append committed items to existing display values.

    The caller owns the result; neither the previous values nor messages change.
    Explicit turn IDs group messages between compactions. Old records retain phase boundaries.
    """
    turns = [
        {key: turn[key] for key in (
            "turn_id", "messages", "response", "response_id", "response_at",
            "response_phase", "kind",
        )}
        for turn in previous
    ]
    by_id = {turn["turn_id"]: index for index, turn in enumerate(turns)}
    boundary = next((i + 1 for i in reversed(range(len(turns)))
                     if turns[i]["kind"] == "compaction"), 0)
    current = turns[-1] if turns else None
    if current is not None and (
        current["kind"] == "compaction" or current["response_phase"] == "final"
    ):
        current = None
    for item in history:
        is_compaction = item is None or (
            isinstance(item, dict) and item.get("type") == "compacted"
        )
        if is_compaction:
            turn_id = item["id"] if item is not None else "compaction-{0}".format(len(turns))
            boundary = len(turns)
            current = None
        elif isinstance(item, (UserMessage, AgentMessage, AssistantMessage)):
            is_input = isinstance(item, (UserMessage, AgentMessage))
            if item.turn_id is None and is_input and (
                item.visible or (current is not None and current["response"]
                                 and current["response_phase"] != "commentary")
            ):
                current = None
            if not item.visible:
                continue
            turn_id = item.turn_id or (current["turn_id"] if current is not None else item.message_id)
        else:
            continue
        index = by_id.get(turn_id)
        if index is None or index < boundary:
            if index is not None:
                earlier = turns[index]
                earlier["turn_id"] = compaction_prefix_id(earlier)
            turn = {
                "turn_id": turn_id, "messages": (), "response": "",
                "response_id": None, "response_at": None, "response_phase": None,
                "kind": "compaction" if is_compaction else "assistant",
            }
            by_id[turn_id] = len(turns)
            turns.append(turn)
        else:
            turn = turns[index]
        if is_compaction:
            continue
        if is_input:
            messages = list(turn["messages"])
            position = next((index for index, message in enumerate(messages)
                             if message.message_id == item.message_id), len(messages))
            messages[position:position + 1] = [item]
            turn["messages"] = tuple(messages)
        else:
            turn.update(response=item.text, response_id=item.message_id,
                        response_at=item.created_at, response_phase=item.phase)
        current = None if turn["response_phase"] == "final" else turn
    return tuple(turns)
