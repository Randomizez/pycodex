import typing
import uuid
from copy import deepcopy


class ResponseOutputStream:
    """Keep output identities and lifecycle events local to one response."""

    def __init__(self) -> "None":
        self.items: "typing.List[typing.Dict[str, object]]" = []
        self._next_index = 0
        self._active_item: "typing.Optional[typing.Dict[str, object]]" = None
        self._active_index = 0
        self._text_parts: "typing.List[str]" = []

    def _event(self, event_type: "str", fields: "typing.Dict[str, object]"):
        return event_type, {"type": event_type, **fields}

    def _start_item(self, item: "typing.Dict[str, object]"):
        index = self._next_index
        self._next_index += 1
        started = dict(item, id=item.get("id") or "item_" + uuid.uuid4().hex)
        return index, started

    def text_delta(self, item_type: "str", delta: "str"):
        if self._active_item is not None and self._active_item["type"] != item_type:
            yield from self.finish_text()
        if self._active_item is None:
            item: "typing.Dict[str, object]" = {
                "type": item_type,
                "status": "in_progress",
                "content": [],
            }
            if item_type == "message":
                item["role"] = "assistant"
            else:
                item["summary"] = []
            self._active_index, self._active_item = self._start_item(item)
            yield self._event("response.output_item.added", {
                "output_index": self._active_index,
                "item": deepcopy(self._active_item),
            })
            if item_type == "message":
                yield self._event("response.content_part.added", {
                    "item_id": self._active_item["id"],
                    "output_index": self._active_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                })
        self._text_parts.append(delta)
        event_type = (
            "response.output_text.delta" if item_type == "message"
            else "response.reasoning_text.delta"
        )
        yield self._event(event_type, {
            "delta": delta,
            "item_id": self._active_item["id"],
            "output_index": self._active_index,
            "content_index": 0,
        })

    def finish_text(self):
        if self._active_item is None:
            return
        text = "".join(self._text_parts)
        item = self._active_item
        index = self._active_index
        self._active_item = None
        self._text_parts = []
        is_message = item["type"] == "message"
        part = {
            "type": "output_text" if is_message else "reasoning_text",
            "text": text,
        }
        if is_message:
            part["annotations"] = []
        item = dict(item, status="completed", content=[part])
        fields = {"item_id": item["id"], "output_index": index, "content_index": 0}
        yield self._event(
            "response.output_text.done" if is_message else "response.reasoning_text.done",
            dict(fields, text=text),
        )
        if is_message:
            yield self._event("response.content_part.done", dict(fields, part=deepcopy(part)))
        self.items.append(item)
        yield self._event("response.output_item.done", {
            "output_index": index,
            "item": deepcopy(item),
        })

    def discard_text(self) -> "None":
        # A rejected reasoning-only attempt must never be committed with `done`.
        # Its replacement gets a fresh identity, like an abandoned partial item.
        self._active_item = None
        self._text_parts = []

    def complete_tool(self, item: "typing.Dict[str, object]"):
        yield from self.finish_text()
        index, completed = self._start_item(dict(item, status="completed"))
        started = dict(completed, status="in_progress")
        field = None
        if item["type"] == "function_call":
            field = "arguments"
            event_prefix = "response.function_call_arguments"
        elif item["type"] == "custom_tool_call":
            field = "input"
            event_prefix = "response.custom_tool_call_input"
        if field is not None:
            started[field] = ""
        yield self._event("response.output_item.added", {
            "output_index": index, "item": started,
        })
        if field is not None:
            fields = {
                "item_id": completed["id"],
                "output_index": index,
                "call_id": completed["call_id"],
            }
            yield self._event(event_prefix + ".delta", dict(fields, delta=completed[field]))
            yield self._event(event_prefix + ".done", dict(fields, **{field: completed[field]}))
        self.items.append(completed)
        yield self._event("response.output_item.done", {
            "output_index": index, "item": deepcopy(completed),
        })
