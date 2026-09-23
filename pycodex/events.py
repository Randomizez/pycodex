"""Typed events, plain-text views and stateful presentation without frontend I/O."""

import json
import re
import shlex
import typing
from dataclasses import dataclass
from typing import ClassVar

from .protocol import JSONDict, ToolCall, ToolResult
from .utils.event_helpers import (
    agent_status_summary,
    colorize_cli_message,
    colorize_tool_message,
    compact_summary,
    format_command_result,
    format_error,
    percent_of_context_window_remaining,
    short_id,
    shorten_title,
    truncate_text,
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
        self.agent_names = {}
        self.closed = False
        self.set_context_window_tokens(context_window_tokens)

    def start(self, commands):
        self.log("pycodex interactive mode. Type /exit or press Ctrl+C to quit.")
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
        self.set_status(event.kind)
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

    def remember_agent_name(self, summary):
        if " (" not in summary or not summary.endswith(")"):
            return
        nickname, rest = summary.rsplit(" (", 1)
        agent_short_id = rest[:-1].strip()
        nickname = nickname.strip()
        if nickname and agent_short_id:
            self.agent_names[agent_short_id] = nickname

    def replace_agent_ids_with_names(self, message):
        for agent_short_id, nickname in sorted(
            self.agent_names.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            message = message.replace(agent_short_id, nickname)
        return message

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

    def render(self, display: "EventDisplay") -> "None":
        display.stream_buffer = ""
        display.write("[status] " + self.visualize(), "status")
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
        if name in {"wait_agent", "send_input", "resume_agent", "close_agent"}:
            message = display.replace_agent_ids_with_names(message)
        if name == "spawn_agent":
            summary = message
            prefix = "[spawn_agent] spawned "
            if summary.startswith(prefix):
                summary = summary[len(prefix) :]
            display.remember_agent_name(summary)
        for line in message.splitlines() or [""]:
            display.log(colorize_tool_message(line, display.color_enabled, name))

    def visualize(self) -> "str":
        name = self.call.name
        summary = self._summary()
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

    def _summary(self) -> "str":
        name = self.call.name
        arguments = self.call.arguments if isinstance(self.call.arguments, dict) else {}
        output = self.result.output
        call_summary = ""
        result_summary = None
        if name == "exec_command":
            command = arguments.get("cmd")
            if command not in (None, ""):
                command = str(command)
                call_summary = truncate_text(command, 200)
                lines = command.splitlines()
                match = re.search(
                    r"\bpython(?:\d+(?:\.\d+)?)?\s+-\s+<<[-]?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1",
                    lines[0] if lines else "",
                )
                if match is not None:
                    for index, line in enumerate(lines[1:], 1):
                        if line.strip() == match.group(2):
                            call_summary = "\n".join(lines[: index + 1])
                            break
            marker = "Process running with session ID "
            for line in self.result.output_text().splitlines():
                line = line.strip()
                if line.startswith(marker) and line[len(marker) :].strip():
                    result_summary = "session_id=" + line[len(marker) :].strip()
                    break
        elif name == "shell_command":
            command = arguments.get("command")
            if command not in (None, ""):
                call_summary = truncate_text(str(command), 200)
        elif name == "shell":
            command = arguments.get("command")
            if isinstance(command, list) and command:
                call_summary = truncate_text(
                    " ".join(shlex.quote(str(part)) for part in command), 200
                )
        elif name == "write_stdin":
            session_id = arguments.get("session_id")
            if session_id not in (None, ""):
                session_id = str(session_id)
                chars = arguments.get("chars") or ""
                call_summary = (
                    f"session {session_id} <- {truncate_text(str(chars), 32)}"
                    if chars
                    else f"poll session {session_id}"
                )
        elif name in {"read_file", "list_dir", "view_image"}:
            key = (
                "file_path"
                if name == "read_file"
                else ("dir_path" if name == "list_dir" else "path")
            )
            path = arguments.get(key)
            if path not in (None, ""):
                call_summary = truncate_text(str(path), 200)
            if name == "view_image" and isinstance(output, list):
                result_summary = f"{len(output)} image item(s)"
        elif name == "grep_files":
            pattern, path = arguments.get("pattern"), arguments.get("path")
            if pattern not in (None, ""):
                call_summary = truncate_text(
                    f"{pattern} @ {path}" if path not in (None, "") else str(pattern),
                    200,
                )
        elif name == "update_plan":
            plan = arguments.get("plan")
            if isinstance(plan, list):
                total = len(plan)
                completed = 0
                in_progress = 0
                for item in plan:
                    if isinstance(item, dict):
                        status = str(item.get("status", "")).strip()
                        completed += status == "completed"
                        in_progress += status == "in_progress"
                if not total:
                    return "0 steps"
                if completed >= total:
                    return f"Done {completed}/{total}"
                if in_progress:
                    return f"Working on {completed + in_progress}/{total}"
                return f"Planned {completed}/{total}"
            if isinstance(output, dict) and isinstance(output.get("plan"), list):
                return f"{len(output['plan'])} steps"
        elif name == "spawn_agent":
            if isinstance(output, dict):
                agent_id = str(output.get("agent_id", "")).strip()
                nickname = str(output.get("nickname", "")).strip()
                if nickname and agent_id:
                    return f"{nickname} ({short_id(agent_id)})"
        elif name == "send_input":
            agent_id, message = arguments.get("id"), arguments.get("message")
            prefix = f"{short_id(str(agent_id))} <- " if agent_id else ""
            call_summary = (
                prefix + truncate_text(str(message), 40)
                if message not in (None, "")
                else prefix.rstrip()
            )
            if isinstance(output, dict):
                submission_id = str(output.get("submission_id", "")).strip()
                if submission_id:
                    result_summary = "queued " + short_id(submission_id)
        elif name == "wait_agent":
            if isinstance(output, dict):
                if output.get("timed_out") is True:
                    return "timed out"
                status = output.get("status")
                if isinstance(status, dict):
                    parts = [
                        f"{short_id(agent_id)}={agent_status_summary(agent_status)}"
                        for agent_id, agent_status in status.items()
                        if isinstance(agent_id, str)
                    ]
                    if parts:
                        return truncate_text(", ".join(parts))
        elif name in {"resume_agent", "close_agent"}:
            agent_id = arguments.get("id")
            if agent_id not in (None, ""):
                call_summary = short_id(str(agent_id))
            if isinstance(output, dict):
                result_summary = agent_status_summary(output.get("status"))
        if result_summary is None:
            if isinstance(output, (dict, list)):
                result_summary = truncate_text(
                    json.dumps(output, ensure_ascii=False, separators=(",", ":"))
                )
            else:
                lines = [
                    line.strip() for line in self.result.output_text().splitlines()
                ]
                if "Output:" in lines:
                    result_summary = next(
                        (line for line in lines[lines.index("Output:") + 1 :] if line),
                        None,
                    )
                if result_summary is None:
                    result_summary = next(
                        (
                            line
                            for line in lines
                            if line
                            and not line.startswith(
                                ("Exit code:", "Wall time:", "Command:")
                            )
                        ),
                        "",
                    )
                result_summary = truncate_text(result_summary)
        if call_summary and result_summary:
            return f"{call_summary} -> {result_summary}"
        return call_summary or result_summary


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
        if self.reason == "auto_title":
            display.show_title()
        if self.reason == "attach" and state["input_request"] is not None:
            state["input_request"].render(display)


@dataclass(frozen=True)
class SessionClosedEvent(Event):
    kind: ClassVar[str] = "session_closed"

    def render(self, display: "EventDisplay") -> "None":
        display.closed = True
        display.finish_stream()


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
