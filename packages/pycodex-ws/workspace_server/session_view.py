"""Runtime event projection and ordered workspace subscriptions."""

import asyncio
import json
import threading
import typing
from dataclasses import asdict, fields, is_dataclass

from pycodex.events import (
    IDLE_SLEEPING_STATUS,
    AssistantDeltaEvent,
    AutoCompactCompletedEvent,
    AutoCompactFailedEvent,
    AutoCompactStartedEvent,
    CommandCompletedEvent,
    CommandFailedEvent,
    CommandStartedEvent,
    CompactCompletedEvent,
    CompactFailedEvent,
    CompactStartedEvent,
    Event,
    InputQueuedEvent,
    InputRequestedEvent,
    InputResolvedEvent,
    SessionClosedEvent,
    SessionStateEvent,
    StreamErrorEvent,
    TerminalEvent,
    TokenCountEvent,
    ToolCalledEvent,
    ToolCompletedEvent,
    ToolStartedEvent,
    TurnCompletedEvent,
    TurnEvent,
    TurnFailedEvent,
    TurnInterruptedEvent,
    TurnStartedEvent,
)
from pycodex.utils import uuid7_string
from pycodex.utils.event_helpers import completed_history, shorten_title
from pycodex.utils.session_persist import load_display_history


JSONValue = typing.Union[
    None,
    bool,
    int,
    float,
    str,
    typing.List["JSONValue"],
    typing.Dict[str, "JSONValue"],
]


SPINNER_STATUS_PREVIEW_LIMIT = 180


class WebSessionView:
    def __init__(self) -> None:
        self._subscribers: "typing.Set[asyncio.Queue]" = set()
        self._events: "typing.List[typing.Dict[str, object]]" = []
        self._stream_id = uuid7_string()
        self._sequence = 0
        self._runtime_state = None
        self._turns: "typing.List[typing.Dict[str, object]]" = []
        self._turns_by_submission_id: "typing.Dict[str, typing.Dict[str, object]]" = {}
        self._turns_by_turn_id: "typing.Dict[str, typing.Dict[str, object]]" = {}
        self._title = ""
        self._last_final_id = ""
        self._model = "pycodex"
        self._cwd = ""
        self._rollout_path = ""
        self._recorded_rollout_path = ""
        self._input_request = None
        self._accepts_input = True
        self._spinner_status = ""
        self._stream_buffer = ""
        self._max_context_window: "typing.Union[int, None]" = None
        self._auto_compact_token_limit: "typing.Union[int, None]" = None
        self._usage_tokens: "typing.Union[int, None]" = None
        self.server_loop: "typing.Union[asyncio.AbstractEventLoop, None]" = None
        self._lock = threading.RLock()

    def attach_server_loop(self, loop: "asyncio.AbstractEventLoop") -> None:
        self.server_loop = loop

    def handle_event(self, event: "Event", runtime_state=None) -> None:
        with self._lock:
            if runtime_state is not None:
                self._runtime_state = _json_safe(runtime_state)
            self._apply_runtime_event(event)
            payload = _event_data(event)
            payload.update({"type": "event", "snapshot": self.snapshot()})
            if isinstance(event, ToolCompletedEvent):
                payload["summary"] = event.visualize()
            self.publish(payload)

    def api_snapshot(self):
        with self._lock:
            return {
                "api_version": 1,
                "state": self._runtime_state,
                "cursor": "{0}:{1}".format(self._stream_id, self._sequence),
            }

    def synchronize_state(self, state):
        with self._lock:
            state = _json_safe(state)
            if state != self._runtime_state:
                self._runtime_state = state
                self.publish({"type": "state"})

    def finish_stream(self) -> None:
        with self._lock:
            if not self._stream_buffer:
                return
            active_turn = self._last_active_turn()
            if active_turn is not None and not active_turn.get("response"):
                active_turn["response"] = self._stream_buffer
                active_turn["thinking"] = ""
                active_turn["_thinking_active"] = False
            self._stream_buffer = ""
            event = {"type": "snapshot", "snapshot": self.snapshot()}
        self.publish(event)

    def write_line(self, text: str) -> None:
        with self._lock:
            text = str(text or "")
            turn = self._new_control_turn(text)
            turn["response"] = text
            turn["status"] = "completed"
            event = {"type": "snapshot", "snapshot": self.snapshot()}
        self.publish(event)

    def show_error(self, text: str) -> None:
        self.finish_stream()
        with self._lock:
            turn = self._new_control_turn("")
            turn["error"] = str(text or "")
            turn["status"] = "error"
            event = {"type": "snapshot", "snapshot": self.snapshot()}
        self.publish(event)

    def set_session_title(self, title: str) -> None:
        with self._lock:
            self._set_title(title)
            event = {
                "type": "title_changed",
                "title": self._title,
                "snapshot": self.snapshot(),
            }
        self.publish(event)

    def load_session_history(
        self,
        title: "typing.Union[str, None]",
        history: "typing.Iterable[typing.Union[typing.Tuple[str, str], None]]",
    ) -> None:
        self.finish_stream()
        with self._lock:
            self._set_title(title)
            self._turns = []
            self._turns_by_submission_id = {}
            self._turns_by_turn_id = {}
            self._events = []
            self._last_final_id = ""
            for entry in history:
                if entry is None:
                    self._add_compaction_marker()
                    continue
                prompt, response = entry
                submission_id = uuid7_string()
                turn = self._ensure_turn(
                    submission_id, submission_id, str(prompt or "")
                )
                turn["response"] = str(response or "")
                turn["status"] = "completed"
                turn["queue"] = "history"
                turn["sender"] = "resume"
            event = {"type": "snapshot", "snapshot": self.snapshot()}
        self.publish(event)

    def subscribe(self, cursor=None) -> "asyncio.Queue":
        queue: "asyncio.Queue" = asyncio.Queue()
        with self._lock:
            events = list(self._events[-200:])
            reset_required = False
            if cursor is not None:
                stream_id, separator, sequence = cursor.rpartition(":")
                if not separator or not sequence.isdigit():
                    raise ValueError("invalid event cursor")
                sequence = int(sequence)
                oldest = self._events[0]["sequence"] if self._events else self._sequence + 1
                reset_required = (
                    stream_id != self._stream_id
                    or sequence < oldest - 1
                    or sequence > self._sequence
                )
                events = [] if reset_required else [
                    event for event in self._events if event["sequence"] > sequence
                ]
            self._subscribers.add(queue)
            event = {
                "type": "hello",
                "events": events,
                "snapshot": self.snapshot(),
                **self.api_snapshot(),
                "reset_required": reset_required,
            }
            queue.put_nowait(event)
        return queue

    def unsubscribe(self, queue: "asyncio.Queue") -> None:
        with self._lock:
            self._subscribers.discard(queue)

    def close(self) -> None:
        with self._lock:
            subscribers = tuple(self._subscribers)
            self._subscribers.clear()
        self._publish_to_queues(subscribers, None)

    def snapshot(self) -> "typing.Dict[str, object]":
        with self._lock:
            return {
                "running": bool(self._spinner_status),
                "last_final_id": self._last_final_id,
                "status": self._spinner_status,
                "status_kind": "spinner" if self._spinner_status else "idle",
                "spinner": self._spinner_status,
                "model": self._model,
                "cwd": self._cwd,
                "rollout_path": self._rollout_path,
                "recorded_rollout_path": self._recorded_rollout_path,
                "input_request": _json_safe(self._input_request),
                "queued_inputs": [
                    {"queue": turn["queue"], "prompt": turn["prompt"]}
                    for turn in self._turns_by_submission_id.values()
                    if not turn["turn_id"]
                ],
                "accepts_input": self._accepts_input,
                "title": self._title,
                **self._context_usage(),
                "turns": [_public_turn(turn) for turn in self._turns[-80:]],
            }

    def summary(self) -> "typing.Dict[str, object]":
        with self._lock:
            return {
                "model": self._model,
                "cwd": self._cwd,
                "running": bool(self._spinner_status),
                "last_final_id": self._last_final_id,
                "spinner": self._spinner_status,
                "title": self._title,
                "turn_count": len(self._turns),
                "last_assistant": _last_assistant_text(self._turns),
                **self._context_usage(),
            }

    def _apply_runtime_event(self, event: "Event") -> None:
        if isinstance(event, SessionStateEvent):
            state = event.state
            self._model = state["model"]
            self._cwd = state["cwd"]
            self._rollout_path = state["rollout_path"]
            self._recorded_rollout_path = state["recorded_rollout_path"]
            self._input_request = state["input_request"]
            self._accepts_input = state["accepts_input"]
            self._max_context_window = state["max_context_window"]
            self._auto_compact_token_limit = state["auto_compact_token_limit"]
            self._usage_tokens = state["usage_tokens"]
            if event.reason in {"attach", "history"}:
                active = state["active_turn"]
                history = completed_history(state)
                if active is None and state["recorded_rollout_path"] is not None:
                    history = load_display_history(state["recorded_rollout_path"])
                self.load_session_history(state["title"], history)
                if active is not None:
                    self._apply_runtime_event(
                        TurnStartedEvent(
                            active["turn_id"],
                            tuple(active["user_texts"]),
                            active["submission_id"],
                        )
                    )
                    self._apply_runtime_event(
                        AssistantDeltaEvent(
                            active["assistant_text"],
                            active["turn_id"],
                            active["submission_id"],
                        )
                    )
                    self._turns_by_turn_id[active["turn_id"]]["model_errors"] = list(
                        active.get("model_errors", [])
                    )
                elif not state["busy"]:
                    self._set_idle_spinner_status(state["background_work_count"])
            elif event.reason == "title":
                self.set_session_title(state["title"])
            else:
                self._set_title(state["title"])
            return
        if isinstance(event, CommandCompletedEvent):
            if event.result["kind"] in {"title_changed", "resumed", "compacted"}:
                return
            message = event.visualize()
            if message:
                self.write_line(message)
            return
        if isinstance(event, CommandFailedEvent):
            self.show_error(event.visualize())
            return
        if isinstance(event, InputRequestedEvent):
            self._input_request = event
            return
        if isinstance(event, InputResolvedEvent):
            self._input_request = None
            return
        if isinstance(event, SessionClosedEvent):
            self._accepts_input = False
            self._spinner_status = ""
            return
        if isinstance(event, InputQueuedEvent):
            turn = self._ensure_turn(event.submission_id, "", "")
            turn["prompt"] += ("\n" if turn["prompt"] else "") + event.prompt
            turn["queue"] = event.queue
            turn["sender"] = event.sender
            return
        if isinstance(event, TokenCountEvent):
            self._usage_tokens = int(event.usage["total_tokens"])
            return
        if isinstance(event, (AutoCompactCompletedEvent, CompactCompletedEvent)):
            self._usage_tokens = None
            self._add_compaction_marker()
        if not isinstance(event, TurnEvent):
            return
        turn_id = event.turn_id
        submission_id = event.submission_id or turn_id
        turn = self._turns_by_submission_id.get(submission_id)

        if isinstance(event, TurnStartedEvent):
            self._set_spinner_status("understood")
            turn = self._ensure_turn(submission_id, turn_id, event.visualize())
            turn["status"] = "running"
            turn["thinking"] = ""
            turn["_thinking_active"] = False
            turn["error"] = ""
            return

        self._apply_spinner_event(event)
        if turn is None:
            return

        if isinstance(event, AssistantDeltaEvent):
            turn["status"] = "responding"
            delta = event.visualize()
            self._stream_buffer += delta
            if turn.get("_thinking_active"):
                turn["thinking"] = str(turn.get("thinking") or "") + delta
            else:
                turn["thinking"] = delta
                turn["_thinking_active"] = True
            return

        if isinstance(event, StreamErrorEvent):
            turn.setdefault("model_errors", []).append(event.detail())
            return

        if isinstance(event, ToolStartedEvent):
            turn["status"] = "tool"
            turn["tool_name"] = event.call.name
            turn["_thinking_active"] = False
            return

        if isinstance(event, ToolCompletedEvent):
            turn["_thinking_active"] = False
            turn["status"] = "running"
            return

        if isinstance(event, TurnCompletedEvent):
            response = event.visualize()
            if response:
                turn["response"] = response
                turn["final_id"] = event.turn_id
                self._last_final_id = event.turn_id
            elif turn.get("thinking"):
                turn["response"] = str(turn.get("thinking") or "")
            turn["thinking"] = ""
            turn["_thinking_active"] = False
            turn["status"] = "completed"
            self._stream_buffer = ""
            return

        if isinstance(event, TurnFailedEvent):
            turn["status"] = "error"
            turn["error"] = event.visualize()
            turn["final_id"] = event.turn_id
            self._last_final_id = event.turn_id
            self._stream_buffer = ""
            return

        if isinstance(event, TurnInterruptedEvent):
            if event.output_text:
                turn["response"] = event.output_text
            elif turn.get("thinking") and not turn.get("response"):
                turn["response"] = str(turn.get("thinking") or "")
            turn["thinking"] = ""
            turn["_thinking_active"] = False
            turn["status"] = "interrupted"
            self._stream_buffer = ""

    def _context_usage(self) -> "typing.Dict[str, object]":
        limit = self._auto_compact_token_limit
        if limit is None:
            limit = self._max_context_window
        if limit is None:
            remaining_percent = None
        elif self._usage_tokens is None:
            remaining_percent = 100
        elif self._usage_tokens >= limit:
            remaining_percent = 0
        else:
            # Round up so zero means the actual threshold has been reached.
            remaining_percent = min(
                100, ((limit - self._usage_tokens) * 100 + limit - 1) // limit
            )
        return {
            "usage_tokens": self._usage_tokens,
            "auto_compact_token_limit": self._auto_compact_token_limit,
            "max_context_window": self._max_context_window,
            "context_remaining_percent": remaining_percent,
        }

    def _apply_spinner_event(self, event: "TurnEvent") -> None:
        if isinstance(event, AssistantDeltaEvent):
            self._set_spinner_status("talking")
            return
        if isinstance(event, StreamErrorEvent):
            self._set_spinner_status("reconnecting")
            return
        if isinstance(event, (AutoCompactStartedEvent, CompactStartedEvent)):
            self._set_spinner_status("compacting")
            return
        if isinstance(event, AutoCompactCompletedEvent):
            self._set_spinner_status("compacted")
            return
        if isinstance(event, ToolStartedEvent):
            self._set_spinner_status(
                shorten_title(
                    event.visualize(),
                    limit=SPINNER_STATUS_PREVIEW_LIMIT,
                )
            )
            return
        if isinstance(event, ToolCompletedEvent):
            self._set_spinner_status("called {0}".format(event.call.name))
            return
        if isinstance(event, TerminalEvent):
            self._set_idle_spinner_status(event.background_work_count)

    def _set_spinner_status(self, text: "typing.Union[str, None]") -> None:
        self._spinner_status = str(text or "").strip()

    def _set_idle_spinner_status(
        self, background_work_count: "typing.Union[int, None]"
    ) -> None:
        if (background_work_count or 0) > 0:
            self._set_spinner_status(IDLE_SLEEPING_STATUS)
        else:
            self._set_spinner_status("")

    def _ensure_turn(
        self,
        submission_id: str,
        turn_id: str,
        prompt: str,
    ) -> "typing.Dict[str, object]":
        submission_id = str(submission_id or "").strip()
        turn_id = str(turn_id or "").strip()
        turn = self._turns_by_submission_id.get(submission_id)
        if turn is None and turn_id and not submission_id:
            turn = self._turns_by_turn_id.get(turn_id)
        if turn is None:
            turn = {
                "submission_id": submission_id,
                "turn_id": "",
                "prompt": prompt,
                "response": "",
                "thinking": "",
                "_thinking_active": False,
                "status": "queued",
                "error": "",
                "kind": "assistant",
                "queue": "steer",
                "sender": "web",
            }
        if submission_id:
            turn["submission_id"] = submission_id
            self._turns_by_submission_id[submission_id] = turn
        if turn_id:
            if not turn["turn_id"]:
                # Queue admission stays hidden until the turn actually starts.
                self._turns.append(turn)
            turn["turn_id"] = turn_id
            self._turns_by_turn_id[turn_id] = turn
        if prompt:
            turn["prompt"] = prompt
        return turn

    def _new_control_turn(self, text: str) -> "typing.Dict[str, object]":
        submission_id = uuid7_string()
        turn = self._ensure_turn(submission_id, submission_id, "")
        turn["kind"] = "control"
        turn["queue"] = "control"
        turn["sender"] = "web"
        turn["status"] = "running"
        turn["error"] = ""
        turn["response"] = ""
        turn["thinking"] = ""
        turn["prompt"] = ""
        return turn

    def _add_compaction_marker(self) -> None:
        turn = self._new_control_turn("")
        turn["kind"] = "compaction"
        turn["status"] = "completed"

    def _set_title(self, title: "typing.Union[str, None]") -> None:
        self._title = str(title or "").strip()

    def _last_active_turn(self) -> "typing.Union[typing.Dict[str, object], None]":
        for turn in reversed(self._turns):
            if turn.get("kind") != "control" and turn.get("status") not in {
                "completed",
                "error",
                "interrupted",
            }:
                return turn
        return None

    def publish(self, event: "typing.Dict[str, object]") -> None:
        with self._lock:
            self._sequence += 1
            event.update(self.api_snapshot(), sequence=self._sequence)
            self._events.append(event)
            if len(self._events) > 500:
                del self._events[:-500]
            subscribers = tuple(self._subscribers)
            self._publish_to_queues(subscribers, event)

    def _publish_to_queues(
        self,
        queues: "typing.Iterable[asyncio.Queue]",
        event: "typing.Union[typing.Dict[str, object], None]",
    ) -> None:
        loop = self.server_loop
        if loop is None:
            for queue in queues:
                queue.put_nowait(event)
            return

        def publish() -> None:
            for queue in queues:
                queue.put_nowait(event)

        loop.call_soon_threadsafe(publish)


def _last_assistant_text(turns: "typing.Iterable[typing.Dict[str, object]]") -> str:
    for turn in reversed(list(turns)):
        if str(turn.get("kind") or "assistant") == "control":
            continue
        text = str(turn.get("thinking") or turn.get("response") or "").strip()
        if text:
            return text
    return ""


def _public_turn(turn: "typing.Dict[str, object]") -> "typing.Dict[str, object]":
    return typing.cast(
        "typing.Dict[str, object]",
        _json_safe(
            {
                "submission_id": turn.get("submission_id", ""),
                "turn_id": turn.get("turn_id", ""),
                "final_id": turn.get("final_id", ""),
                "prompt": turn.get("prompt", ""),
                "response": turn.get("response", ""),
                "thinking": turn.get("thinking", ""),
                "status": turn.get("status", ""),
                "error": turn.get("error", ""),
                "queue": turn.get("queue", ""),
                "sender": turn.get("sender", ""),
                "kind": turn.get("kind", "assistant"),
                "model_errors": turn.get("model_errors", []),
            }
        ),
    )


def _event_data(event: "Event") -> "typing.Dict[str, object]":
    payload = {
        item.name: getattr(event, item.name)
        for item in fields(event)
        if item.name not in {"turn_id", "submission_id"}
    }
    turn_id = ""
    if isinstance(event, TurnEvent):
        turn_id = event.turn_id
        if event.submission_id is not None:
            payload.update(submission_id=event.submission_id, turn_id=event.turn_id)
    elif isinstance(
        event, (CommandStartedEvent, CommandCompletedEvent, CommandFailedEvent, InputQueuedEvent)
    ):
        turn_id = event.submission_id
        if isinstance(event, InputQueuedEvent):
            payload["submission_id"] = event.submission_id
    if isinstance(event, TurnStartedEvent):
        payload["user_text"] = "\n".join(event.user_texts)
    if isinstance(event, (ToolStartedEvent, ToolCompletedEvent)):
        payload.update(tool_name=event.call.name, call_id=event.call.call_id)
        if isinstance(event, ToolCompletedEvent):
            payload["is_error"] = event.result.is_error
    if isinstance(event, ToolCalledEvent):
        payload = {name: value for name, value in payload.items() if value is not None}
    if isinstance(
        event,
        (
            CompactStartedEvent,
            CompactCompletedEvent,
            CompactFailedEvent,
            AutoCompactStartedEvent,
            AutoCompactCompletedEvent,
            AutoCompactFailedEvent,
        ),
    ):
        for name in ("total_tokens", "token_limit"):
            if payload[name] is None:
                del payload[name]
        if payload.get("pruned_tool_results") == 0:
            del payload["pruned_tool_results"]
    if isinstance(event, (CompactCompletedEvent, AutoCompactCompletedEvent)):
        payload["summary"] = event.summary
    if isinstance(event, TerminalEvent) and event.background_work_count is None:
        del payload["background_work_count"]
    if isinstance(event, InputRequestedEvent):
        payload["kind"] = payload.pop("request_kind")
        payload["text"] = event.visualize()
        payload = {name: value for name, value in payload.items() if value is not None}
    return {"kind": event.kind, "turn_id": turn_id, "payload": _json_safe(payload)}


def _json_safe(value: object) -> "JSONValue":
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, InputRequestedEvent):
        return _event_data(value)["payload"]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return typing.cast(JSONValue, value)
