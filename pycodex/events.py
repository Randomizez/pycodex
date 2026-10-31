"""Typed events, plain-text views and stateful presentation without frontend I/O."""

import json
import typing
from dataclasses import dataclass
from typing import ClassVar

from .protocol import JSONDict, ToolCall, ToolResult
from .utils.event_helpers import (
    colorize_cli_message,
    colorize_tool_message,
    compact_summary,
    format_command_result,
    format_error,
    percent_of_context_window_remaining,
    shorten_title,
    summarize_tool_result,
)

DEFAULT_MAIN_PROMPT = "pycodex> "
IDLE_SLEEPING_STATUS = "idle: sleeping"


class EventDisplay:
    """Per-subscriber presentation state with log, status and prompt executors."""

    def __init__(
        self,
        log,
        set_status,
        set_prompt,
        color_enabled=False,
        context_window_tokens=None,
    ):
        self.log = log
        self.set_status = set_status
        self.set_prompt = set_prompt
        self.color_enabled = color_enabled
        self.title = None
        self.stream_buffer = ""
        self.queued_steer_prompts = {}
        self.inserted_steer_prompts = {}
        self.closed = False
        self.set_context_window_tokens(context_window_tokens)

    def start(self, model, reasoning_effort, endpoint, commands):
        self.log("Welcome to PyCodex")
        self.log(
            "model: {0} | effort: {1}".format(
                model, reasoning_effort or "provider default"
            )
        )
        self.log("endpoint: " + endpoint)
        self.log(format_command_result({"kind": "help", "commands": commands}))

    def write(self, text, kind=""):
        self.log(colorize_cli_message(text, kind, self.color_enabled))

    def finish_stream(self):
        if self.stream_buffer:
            self.log("assistant> " + self.stream_buffer)
            self.stream_buffer = ""

    def show_error(self, text):
        self.finish_stream()
        self.write(format_error(text), "error")

    def show_title(self):
        self.finish_stream()
        self.log("Session: " + (self.title or "untitled"))

    def begin_turn(self, event):
        self.finish_stream()
        self.set_status("understood")
        submission_id = event.submission_id or event.turn_id
        for prompts in (self.inserted_steer_prompts, self.queued_steer_prompts):
            for prompt in prompts.pop(submission_id, []):
                self.write("[steer] inserted: " + prompt, "status")
        text = event.visualize().strip()
        if text:
            self.write("user> " + text, "assistant")

    def set_idle_status(self, background_count):
        self.set_status(IDLE_SLEEPING_STATUS if (background_count or 0) > 0 else None)

    def set_context_window_tokens(self, context_window_tokens):
        self.context_window_tokens = context_window_tokens
        self.context_remaining_percent = (
            100 if context_window_tokens is not None else None
        )

    def update_context_window(self, usage):
        if self.context_window_tokens is None:
            return
        if not isinstance(usage, dict):
            self.context_remaining_percent = None
            return
        try:
            total_tokens = int(usage["total_tokens"])
        except (KeyError, TypeError, ValueError):
            self.context_remaining_percent = None
            return
        self.context_remaining_percent = percent_of_context_window_remaining(
            total_tokens,
            self.context_window_tokens,
        )

    def main_prompt(self):
        if self.context_remaining_percent is None:
            return DEFAULT_MAIN_PROMPT
        return f"pyco({self.context_remaining_percent}%)> "

    @staticmethod
    def status_frame(text, frame_index):
        if text is None:
            return None
        frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
        return f"{frames[frame_index % len(frames)]} {text}"


class Event:
    kind: ClassVar[str]

    def visualize(self) -> "str":
        return ""

    def render(self, display: "EventDisplay") -> "None":
        pass


class TurnEvent(Event):
    turn_id: "str"
    submission_id: "typing.Union[str, None]"


class ModelEvent(TurnEvent):
    """Model stream notification; Agent supplies turn identity before dispatch."""


class TerminalEvent(TurnEvent):
    background_work_count: "typing.Union[int, None]"

    def render(self, display: "EventDisplay") -> "None":
        display.set_idle_status(self.background_work_count)


@dataclass(frozen=True)
class TurnStartedEvent(TurnEvent):
    turn_id: "str"
    user_texts: "typing.Tuple[str, ...]"
    submission_id: "typing.Union[str, None]" = None
    kind: ClassVar[str] = "turn_started"

    def visualize(self) -> "str":
        return "\n".join(self.user_texts)

    def render(self, display: "EventDisplay") -> "None":
        display.begin_turn(self)


@dataclass(frozen=True)
class TurnInputReadyEvent(TurnEvent):
    """Synchronous input coordination; delivery failures must fail the turn."""

    turn_id: "str"
    submission_id: "typing.Union[str, None]" = None
    kind: ClassVar[str] = "turn_input_ready"


@dataclass(frozen=True)
class TurnCompletedEvent(TerminalEvent):
    turn_id: "str"
    iteration: "int"
    output_text: "typing.Union[str, None]"
    background_work_count: "typing.Union[int, None]"
    submission_id: "typing.Union[str, None]" = None
    kind: ClassVar[str] = "turn_completed"

    def visualize(self) -> "str":
        return self.output_text or ""

    def render(self, display: "EventDisplay") -> "None":
        text = self.visualize()
        if text:
            if text.startswith(display.stream_buffer):
                display.stream_buffer += text[len(display.stream_buffer) :]
            else:
                display.stream_buffer += text
        display.finish_stream()
        super().render(display)


@dataclass(frozen=True)
class TurnFailedEvent(TerminalEvent):
    turn_id: "str"
    iteration: "int"
    error: "str"
    error_type: "str"
    background_work_count: "typing.Union[int, None]"
    submission_id: "typing.Union[str, None]" = None
    kind: ClassVar[str] = "turn_failed"

    def visualize(self) -> "str":
        return self.error

    def render(self, display: "EventDisplay") -> "None":
        display.show_error(self.visualize())
        super().render(display)


@dataclass(frozen=True)
class ModelCalledEvent(TurnEvent):
    turn_id: "str"
    iteration: "int"
    history_size: "int"
    tool_count: "int"
    submission_id: "typing.Union[str, None]" = None
    kind: ClassVar[str] = "model_called"

    def render(self, display: "EventDisplay") -> "None":
        display.finish_stream()


@dataclass(frozen=True)
class ModelCompletedEvent(TurnEvent):
    turn_id: "str"
    iteration: "int"
    item_count: "int"
    submission_id: "typing.Union[str, None]" = None
    kind: ClassVar[str] = "model_completed"


@dataclass(frozen=True)
class AssistantDeltaEvent(ModelEvent):
    delta: "str"
    turn_id: "str" = ""
    submission_id: "typing.Union[str, None]" = None
    kind: ClassVar[str] = "assistant_delta"

    def visualize(self) -> "str":
        return self.delta

    def render(self, display: "EventDisplay") -> "None":
        display.stream_buffer += self.visualize()
        display.set_status("talking")


@dataclass(frozen=True)
class TokenCountEvent(ModelEvent):
    usage: "typing.Union[JSONDict, None]"
    turn_id: "str" = ""
    submission_id: "typing.Union[str, None]" = None
    kind: ClassVar[str] = "token_count"

    def render(self, display: "EventDisplay") -> "None":
        display.update_context_window(self.usage)
        display.set_prompt(display.main_prompt())


@dataclass(frozen=True)
class ModelRequestTimingEvent(ModelEvent):
    model: "str"
    elapsed_seconds: "float"
    first_data_seconds: "typing.Union[float, None]"
    outcome: "str"
    turn_id: "str" = ""
    submission_id: "typing.Union[str, None]" = None
    kind: ClassVar[str] = "model_request_timing"


@dataclass(frozen=True)
class StreamErrorEvent(ModelEvent):
    message: "str"
    attempt: "int"
    max_retries: "int"
    delay_seconds: "float"
    error: "str"
    turn_id: "str" = ""
    submission_id: "typing.Union[str, None]" = None
    kind: ClassVar[str] = "stream_error"

    def visualize(self) -> "str":
        return self.message

    def detail(self) -> "str":
        if not self.error:
            return self.message
        return "{0}\n{1}".format(self.message, self.error)

    def render(self, display: "EventDisplay") -> "None":
        display.stream_buffer = ""
        display.write("[status] " + self.detail(), "status")
        display.set_status("reconnecting")


@dataclass(frozen=True)
class ToolCalledEvent(ModelEvent):
    call_id: "str"
    tool_name: "str"
    action_type: "typing.Union[str, None]" = None
    query: "typing.Union[str, None]" = None
    queries: "typing.Union[typing.Tuple[str, ...], None]" = None
    url: "typing.Union[str, None]" = None
    pattern: "typing.Union[str, None]" = None
    turn_id: "str" = ""
    submission_id: "typing.Union[str, None]" = None
    kind: ClassVar[str] = "tool_called"

    def visualize(self) -> "str":
        if self.tool_name != "web_search":
            return ""
        if self.action_type == "search":
            query = (self.query or "").strip()
            if not query and self.queries:
                query = self.queries[0].strip()
            return (
                f"[web_search] searched: {query}" if query else "[web_search] searched"
            )
        if self.action_type == "open_page":
            url = (self.url or "").strip()
            return f"[web_search] opened: {url}" if url else "[web_search] opened"
        if self.action_type == "find_in_page":
            pattern = (self.pattern or "").strip()
            url = (self.url or "").strip()
            if pattern and url:
                return f"[web_search] found: {pattern} @ {url}"
            if pattern:
                return f"[web_search] found: {pattern}"
            return "[web_search] found in page"
        return "[web_search] browsing"

    def render(self, display: "EventDisplay") -> "None":
        display.finish_stream()
        message = self.visualize()
        if message:
            display.log(colorize_tool_message(message, display.color_enabled))


@dataclass(frozen=True)
class ToolStartedEvent(TurnEvent):
    turn_id: "str"
    call: "ToolCall"
    submission_id: "typing.Union[str, None]" = None
    kind: ClassVar[str] = "tool_started"

    def visualize(self) -> "str":
        return f"calling {self.call.name}({self.call.arguments})"

    def render(self, display: "EventDisplay") -> "None":
        display.finish_stream()
        display.set_status(shorten_title(self.visualize(), 72))


@dataclass(frozen=True)
class ToolCompletedEvent(TurnEvent):
    turn_id: "str"
    call: "ToolCall"
    result: "ToolResult"
    submission_id: "typing.Union[str, None]" = None
    kind: ClassVar[str] = "tool_completed"

    def render(self, display: "EventDisplay") -> "None":
        display.finish_stream()
        name = self.call.name
        message = self.visualize()
        if name:
            display.set_status("called " + name)
        for line in message.splitlines() or [""]:
            display.log(colorize_tool_message(line, display.color_enabled, name))

    def visualize(self) -> "str":
        name = self.call.name
        summary = summarize_tool_result(self.call, self.result)
        if self.result.is_error:
            return f"[error] {name} failed" + (f": {summary}" if summary else "")
        if name == "spawn_agent":
            return "[spawn_agent] spawned" + (f" {summary}" if summary else "")
        if name == "update_plan":
            lines = ["[update_plan] " + (summary or "Plan updated")]
            plan = (
                self.call.arguments.get("plan")
                if isinstance(self.call.arguments, dict)
                else None
            )
            if isinstance(plan, list):
                for item in plan:
                    if not isinstance(item, dict):
                        continue
                    step = str(item.get("step", "")).strip()
                    status = str(item.get("status", "")).strip()
                    if step:
                        marker = (
                            "[x]"
                            if status == "completed"
                            else ("[>]" if status == "in_progress" else "[ ]")
                        )
                        lines.append(f"  {marker} {step}")
            return "\n".join(lines)
        return f"[{name}]" + (f" {summary}" if summary else "")


@dataclass(frozen=True)
class CompactStartedEvent(TurnEvent):
    turn_id: "str"
    phase: "str"
    total_tokens: "typing.Union[int, None]"
    token_limit: "typing.Union[int, None]"
    submission_id: "typing.Union[str, None]" = None
    kind: ClassVar[str] = "compact_started"

    def visualize(self) -> "str":
        return "Compacting conversation history..."

    def render(self, display: "EventDisplay") -> "None":
        display.log(self.visualize())
        display.set_status("compacting")


@dataclass(frozen=True)
class CompactCompletedEvent(TerminalEvent):
    turn_id: "str"
    phase: "str"
    total_tokens: "typing.Union[int, None]"
    token_limit: "typing.Union[int, None]"
    original_item_count: "int"
    retained_item_count: "int"
    pruned_tool_results: "int"
    background_work_count: "typing.Union[int, None]"
    submission_id: "typing.Union[str, None]" = None
    kind: ClassVar[str] = "compact_completed"

    @property
    def summary(self) -> "str":
        return compact_summary(
            self.original_item_count, self.retained_item_count, self.pruned_tool_results
        )

    def visualize(self) -> "str":
        return self.summary


@dataclass(frozen=True)
class CompactFailedEvent(TerminalEvent):
    turn_id: "str"
    phase: "str"
    total_tokens: "typing.Union[int, None]"
    token_limit: "typing.Union[int, None]"
    error: "str"
    error_type: "str"
    background_work_count: "typing.Union[int, None]"
    submission_id: "typing.Union[str, None]" = None
    kind: ClassVar[str] = "compact_failed"

    def visualize(self) -> "str":
        return self.error


@dataclass(frozen=True)
class AutoCompactStartedEvent(TurnEvent):
    turn_id: "str"
    phase: "str"
    total_tokens: "typing.Union[int, None]"
    token_limit: "typing.Union[int, None]"
    submission_id: "typing.Union[str, None]" = None
    kind: ClassVar[str] = "auto_compact_started"

    def visualize(self) -> "str":
        if self.total_tokens is not None and self.token_limit is not None:
            return (
                f"[status] auto-compact: {self.total_tokens}/{self.token_limit} tokens"
            )
        return "[status] auto-compact"

    def render(self, display: "EventDisplay") -> "None":
        display.finish_stream()
        display.write(self.visualize(), "status")
        display.set_status("compacting")


@dataclass(frozen=True)
class AutoCompactCompletedEvent(TurnEvent):
    turn_id: "str"
    phase: "str"
    total_tokens: "typing.Union[int, None]"
    token_limit: "typing.Union[int, None]"
    original_item_count: "int"
    retained_item_count: "int"
    pruned_tool_results: "int"
    submission_id: "typing.Union[str, None]" = None
    kind: ClassVar[str] = "auto_compact_completed"

    @property
    def summary(self) -> "str":
        return compact_summary(
            self.original_item_count, self.retained_item_count, self.pruned_tool_results
        )

    def visualize(self) -> "str":
        return "[status] " + self.summary

    def render(self, display: "EventDisplay") -> "None":
        display.finish_stream()
        display.write(self.visualize(), "status")
        display.set_status("compacted")


@dataclass(frozen=True)
class AutoCompactFailedEvent(TurnEvent):
    turn_id: "str"
    phase: "str"
    total_tokens: "typing.Union[int, None]"
    token_limit: "typing.Union[int, None]"
    error: "str"
    error_type: "str"
    submission_id: "typing.Union[str, None]" = None
    kind: ClassVar[str] = "auto_compact_failed"

    def visualize(self) -> "str":
        error = self.error.strip()
        return "[error] auto-compact failed" + (f": {error}" if error else "")

    def render(self, display: "EventDisplay") -> "None":
        display.finish_stream()
        display.write(self.visualize(), "error")


@dataclass(frozen=True)
class TurnInterruptedEvent(TerminalEvent):
    turn_id: "str"
    iteration: "int"
    output_text: "typing.Union[str, None]"
    background_work_count: "typing.Union[int, None]"
    submission_id: "typing.Union[str, None]" = None
    kind: ClassVar[str] = "turn_interrupted"

    def visualize(self) -> "str":
        return self.output_text or ""

    def render(self, display: "EventDisplay") -> "None":
        display.finish_stream()
        super().render(display)


@dataclass(frozen=True)
class SessionStateEvent(Event):
    reason: "str"
    state: "typing.Dict[str, object]"
    kind: ClassVar[str] = "session_state"

    def render(self, display: "EventDisplay") -> "None":
        state = self.state
        display.title = state["title"] or None
        if self.reason == "admission" and not state["accepts_input"]:
            display.write(
                "[closing] Waiting for accepted work and cleanup to finish.", "status"
            )
        if self.reason in {"attach", "history", "model"}:
            display.set_context_window_tokens(state["context_window"])
            if state["usage_tokens"] is not None:
                display.update_context_window({"total_tokens": state["usage_tokens"]})
        if self.reason in {"attach", "history"}:
            display.finish_stream()
            display.queued_steer_prompts.clear()
            display.inserted_steer_prompts.clear()
            active = state["active_turn"]
            if active is not None:
                TurnStartedEvent(
                    active["turn_id"],
                    tuple(active["user_texts"]),
                    active["submission_id"],
                ).render(display)
                display.stream_buffer = active["assistant_text"]
            elif not state["busy"]:
                display.set_idle_status(state["background_work_count"])
        if self.reason == "auto_title":
            display.show_title()
        if self.reason == "attach" and state["input_request"] is not None:
            state["input_request"].render(display)


@dataclass(frozen=True)
class RuntimeUpdatedEvent(Event):
    reason: "str"
    kind: ClassVar[str] = "runtime_updated"


@dataclass(frozen=True)
class SessionClosedEvent(Event):
    kind: ClassVar[str] = "session_closed"

    def render(self, display: "EventDisplay") -> "None":
        display.closed = True
        display.finish_stream()


@dataclass(frozen=True)
class CommandStartedEvent(Event):
    submission_id: "str"
    command: "str"
    sender: "str" = "user"
    kind: ClassVar[str] = "command_started"


@dataclass(frozen=True)
class CommandCompletedEvent(Event):
    submission_id: "str"
    command: "str"
    result: "typing.Dict[str, object]"
    sender: "str"
    kind: ClassVar[str] = "command_completed"

    def visualize(self) -> "str":
        return format_command_result(self.result)

    def render(self, display: "EventDisplay") -> "None":
        for line in self.visualize().splitlines():
            display.log(line)


@dataclass(frozen=True)
class CommandFailedEvent(Event):
    submission_id: "str"
    command: "str"
    error: "str"
    sender: "str"
    error_type: "str" = "CommandError"
    kind: ClassVar[str] = "command_failed"

    def visualize(self) -> "str":
        return self.error

    def render(self, display: "EventDisplay") -> "None":
        display.show_error(self.visualize())


@dataclass(frozen=True)
class InputQueuedEvent(Event):
    submission_id: "str"
    prompt: "str"
    queue: "str"
    sender: "str"
    explicit_queue: "bool"
    was_busy: "bool"
    kind: ClassVar[str] = "input_queued"

    def render(self, display: "EventDisplay") -> "None":
        preview = shorten_title(self.prompt, 72)
        if self.explicit_queue:
            display.queued_steer_prompts.setdefault(self.submission_id, []).append(
                preview
            )
            display.write("[steer] queued: " + preview, "status")
        elif self.was_busy:
            display.inserted_steer_prompts.setdefault(self.submission_id, []).append(
                preview
            )


@dataclass(frozen=True)
class InputRequestedEvent(Event):
    request_id: "str"
    request_kind: "str"
    other: "bool"
    question: "typing.Union[JSONDict, None]" = None
    permissions: "typing.Union[JSONDict, None]" = None
    kind: ClassVar[str] = "input_requested"

    def render(self, display: "EventDisplay") -> "None":
        display.finish_stream()
        for line in self.visualize().splitlines():
            display.log(line)
        display.set_status(None)
        display.set_prompt(
            "permissions> "
            if self.request_kind == "permissions"
            else ("other> " if self.other else "answer> ")
        )

    def visualize(self) -> "str":
        if self.request_kind == "permissions":
            lines = ["[request_permissions] user approval required"]
            if self.permissions.get("reason"):
                lines.append("Reason: " + self.permissions["reason"])
            return "\n".join(
                lines
                + [
                    "Requested permissions:",
                    json.dumps(
                        self.permissions.get("permissions", {}),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "Choose: [n] deny / [t] grant for turn / [s] grant for session",
                ]
            )
        if self.other:
            return "Enter your answer (blank to cancel):"
        question = self.question
        return "\n".join(
            [
                "[request_user_input] waiting for user response",
                "[{0}] {1}".format(question["header"], question["question"]),
            ]
            + [
                "  {0}. {1} - {2}".format(index, option["label"], option["description"])
                for index, option in enumerate(question["options"], 1)
            ]
            + ["  0. Other"]
        )


@dataclass(frozen=True)
class InputResolvedEvent(Event):
    request_id: "str"
    kind: ClassVar[str] = "input_resolved"

    def render(self, display: "EventDisplay") -> "None":
        display.set_prompt(display.main_prompt())
