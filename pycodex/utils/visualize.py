import asyncio
import json
import os
import shlex
import sys
import threading
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import StdoutProxy
from prompt_toolkit.patch_stdout import patch_stdout

from ..protocol import JSONDict, ToolCall, ToolResult
import typing

ANSI_RESET = "\x1b[0m"
ANSI_BOLD = "\x1b[1m"
ANSI_DIM = "\x1b[2m"
ANSI_GREEN = "\x1b[32m"
ANSI_BLUE = "\x1b[34m"
ANSI_CYAN = "\x1b[36m"
ANSI_YELLOW = "\x1b[33m"
ANSI_MAGENTA = "\x1b[35m"
ANSI_RED = "\x1b[31m"
SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
PROMPT_CONTEXT_BASELINE_TOKENS = 12_000
DEFAULT_MAIN_PROMPT = "pycodex> "


def shorten_title(text: "str", limit: "int" = 48) -> "str":
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def cli_color_enabled() -> "bool":
    return os.environ.get("PYCODEX_NO_COLOR", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def colorize_cli_message(text: "str", kind: "str", enabled: "bool") -> "str":
    if not enabled:
        return text
    palette = {
        "assistant": ANSI_GREEN,
        "plan": ANSI_CYAN,
        "exec": ANSI_YELLOW,
        "agent": ANSI_BLUE,
        "web": ANSI_MAGENTA,
        "status": ANSI_CYAN,
        "tool": ANSI_DIM,
        "error": ANSI_RED,
    }
    color = palette.get(kind)
    if color is None:
        return text
    return f"{ANSI_BOLD}{color}{text}{ANSI_RESET}"


def format_cli_plan_messages(
    summary: "str",
    plan_items: "typing.List[JSONDict]",
) -> "typing.List[str]":
    lines = [f"[plan] {summary}" if summary else "[plan] Plan updated"]
    for item in plan_items:
        step = str(item.get("step", "")).strip()
        status = str(item.get("status", "")).strip()
        if not step:
            continue
        marker = {
            "completed": "[x]",
            "in_progress": "[>]",
            "pending": "[ ]",
        }.get(status, "[ ]")
        lines.append(f"  {marker} {step}")
    return lines


def percent_of_context_window_remaining(
    total_tokens: "int",
    context_window_tokens: "int",
) -> "int":
    if context_window_tokens <= PROMPT_CONTEXT_BASELINE_TOKENS:
        return 0

    effective_window = context_window_tokens - PROMPT_CONTEXT_BASELINE_TOKENS
    used = max(total_tokens - PROMPT_CONTEXT_BASELINE_TOKENS, 0)
    remaining = max(effective_window - used, 0)
    return int(round(max(0.0, min(100.0, (remaining / effective_window) * 100.0))))


class Spinner:
    def __init__(
        self,
        raw_write=None,
        raw_flush=None,
        terminal_lock: "threading.RLock" = None,
    ) -> "None":
        self._raw_write = raw_write or sys.stdout.write
        self._raw_flush = raw_flush or sys.stdout.flush
        self._terminal_lock = terminal_lock or threading.Lock()
        self._color_enabled = False

        self._visible = False
        self._paused = True
        self._index = 0
        self._label = "thinking"
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="pycodex-cli-spinner",
            daemon=True,
        )
        self._thread.start()

    @property
    def spinner_frame(self) -> "str":
        self._index += 1
        suffix = f" {self._label}"
        frame = f"{SPINNER_FRAMES[self._index % len(SPINNER_FRAMES)]}{suffix}"
        frame = colorize_cli_message(frame, kind="status", enabled=self._color_enabled)
        return frame

    def set_label(self, label: "str") -> "None":
        with self._terminal_lock:
            self._label = label

    def pause(self) -> "None":
        self._paused = True
        self.clear()

    def resume(self) -> "None":
        self._paused = False

    def clear(self) -> "None":
        with self._terminal_lock:
            if not self._visible:
                return
            self._raw_write("\r")
            self._raw_flush()
            self._visible = False

    def render_one(self) -> "None":
        if self._paused:
            return
        with self._terminal_lock:
            if self._paused:
                return
            self._raw_write(f"\r{self.spinner_frame}")
            self._raw_flush()
            self._visible = True

    def close(self) -> "None":
        self._stop.set()
        self._thread.join(timeout=0.5)

    def _run(self) -> "None":
        while not self._stop.wait(0.12):
            self.render_one()


class Prompter:
    def __init__(self, prompt: str = DEFAULT_MAIN_PROMPT, lock=None):
        self.lock = lock or threading.Lock()
        self._prompt_session = PromptSession(
            erase_when_done=True,
            enable_system_prompt=True,
            show_frame=True,
        )
        
        self.stdout_proxy = StdoutProxy(raw=False)

        self.spinner = Spinner(
            raw_write=self.stdout_proxy.write,
            raw_flush=self.stdout_proxy.flush,
            terminal_lock=self.lock,
        )

        self.prompt = prompt

        self._prompt_task = None

    def set_prompt(self, prompt):
        self.prompt = prompt

    def set_status(self, text, active=True):
        self.spinner.set_label(text)
        if active:
            self.spinner.resume()
        else:
            self.spinner.pause()
            self.spinner.clear()

    async def poll_input(self) -> "typing.Union[str, None]":
        if self._prompt_task is None:
            self._prompt_task = asyncio.create_task(self._block_prompt())

        done, _pending = await asyncio.wait(
            {self._prompt_task},
            timeout=0.05,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            return None

        prompt_task, self._prompt_task = self._prompt_task, None
        try:
            return prompt_task.result()
        except asyncio.CancelledError:
            return None

    async def _block_prompt(self):
        with patch_stdout(raw=True):
            return await self._prompt_session.prompt_async(
                lambda: '\n' + self.prompt,
                refresh_interval=0.12
            )
    
    def close(self) -> "None":
        if self._prompt_task is not None and not self._prompt_task.done():
            self._prompt_task.cancel()
            self._prompt_task = None
        self.stdout_proxy.close()
        self.spinner.close()


def format_cli_tool_call_message(
    tool_name: "str", payload: "JSONDict"
) -> "typing.Union[str, None]":
    if tool_name != "web_search":
        return None

    action_type = str(payload.get("action_type", "")).strip()
    if action_type == "search":
        query = str(payload.get("query", "")).strip()
        if not query:
            queries = payload.get("queries")
            if isinstance(queries, list) and queries:
                query = str(queries[0]).strip()
        return f"[web] searched: {query}" if query else "[web] searched"

    if action_type == "open_page":
        url = str(payload.get("url", "")).strip()
        return f"[web] opened: {url}" if url else "[web] opened"

    if action_type == "find_in_page":
        pattern = str(payload.get("pattern", "")).strip()
        url = str(payload.get("url", "")).strip()
        if pattern and url:
            return f"[web] found: {pattern} @ {url}"
        if pattern:
            return f"[web] found: {pattern}"
        return "[web] found in page"

    return "[web] browsing"


def short_id(value: "str", limit: "int" = 8) -> "str":
    compact = value.strip()
    if len(compact) <= limit + 4:
        return compact
    return f"{compact[:limit]}...{compact[-4:]}"


def format_cli_tool_message(
    tool_name: "str",
    summary: "str",
    is_error: "bool",
) -> "str":
    if tool_name == "update_plan":
        if is_error:
            return (
                f"[error] plan failed: {summary}" if summary else "[error] plan failed"
            )
        return f"[plan] {summary}" if summary else "[plan] Plan updated"

    if tool_name in {
        "exec_command",
        "write_stdin",
        "shell",
        "shell_command",
        "exec",
        "wait",
    }:
        if is_error:
            return (
                f"[error] exec failed: {summary}" if summary else "[error] exec failed"
            )
        return f"[exec] {summary}" if summary else f"[exec] {tool_name}"

    if tool_name == "spawn_agent":
        if is_error:
            return (
                f"[error] agent spawn failed: {summary}"
                if summary
                else "[error] agent spawn failed"
            )
        return f"[agent] spawned {summary}" if summary else "[agent] spawned"

    if tool_name == "send_input":
        if is_error:
            return (
                f"[error] agent send failed: {summary}"
                if summary
                else "[error] agent send failed"
            )
        return f"[agent] send: {summary}" if summary else "[agent] send"

    if tool_name == "wait_agent":
        if is_error:
            return (
                f"[error] agent wait failed: {summary}"
                if summary
                else "[error] agent wait failed"
            )
        return f"[agent] wait: {summary}" if summary else "[agent] wait"

    if tool_name == "resume_agent":
        if is_error:
            return (
                f"[error] agent resume failed: {summary}"
                if summary
                else "[error] agent resume failed"
            )
        return f"[agent] resume: {summary}" if summary else "[agent] resume"

    if tool_name == "close_agent":
        if is_error:
            return (
                f"[error] agent close failed: {summary}"
                if summary
                else "[error] agent close failed"
            )
        return f"[agent] close: {summary}" if summary else "[agent] close"

    if is_error:
        return (
            f"[error] {tool_name} failed: {summary}"
            if summary
            else f"[error] {tool_name} failed"
        )
    return f"[tool] {tool_name}: {summary}" if summary else f"[tool] {tool_name}"


def extract_plan_items(arguments: "object") -> "typing.List[JSONDict]":
    if not isinstance(arguments, dict):
        return []
    raw_plan = arguments.get("plan")
    if not isinstance(raw_plan, list):
        return []
    plan_items: "typing.List[JSONDict]" = []
    for item in raw_plan:
        if not isinstance(item, dict):
            continue
        plan_items.append(
            {
                "step": str(item.get("step", "")).strip(),
                "status": str(item.get("status", "")).strip(),
            }
        )
    return plan_items


def summarize_tool_event(
    call: "ToolCall", result: "ToolResult"
) -> "typing.Union[str, None]":
    command_preview = _command_preview(call)
    result_summary = _summarize_tool_result(result)
    if call.name == "update_plan":
        return command_preview or result_summary
    if command_preview and result_summary:
        return f"{command_preview} -> {result_summary}"
    if command_preview:
        return command_preview
    return result_summary


def extract_tool_event_display(
    payload: "typing.Dict[str, object]",
) -> "typing.Tuple[str, str, bool]":
    tool_name = str(payload.get("tool_name", "")).strip()
    is_error = bool(payload.get("is_error"))
    call = payload.get("call")
    result = payload.get("result")
    if isinstance(call, ToolCall) and isinstance(result, ToolResult):
        return tool_name, summarize_tool_event(call, result) or "", is_error
    summary = str(payload.get("summary", "") or "").strip()
    return tool_name, summary, is_error


def extract_plan_event_items(
    payload: "typing.Dict[str, object]",
) -> "typing.List[JSONDict]":
    call = payload.get("call")
    if isinstance(call, ToolCall):
        return extract_plan_items(call.arguments)
    raw_plan_items = payload.get("plan_items")
    if isinstance(raw_plan_items, list):
        return [item for item in raw_plan_items if isinstance(item, dict)]
    return []


def _truncate_text(text: "str", limit: "int" = 96) -> "str":
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _extract_output_preview(text: "str") -> "typing.Union[str, None]":
    lines = [line.strip() for line in text.splitlines()]
    if "Output:" in lines:
        output_index = lines.index("Output:")
        for line in lines[output_index + 1 :]:
            if line:
                return _truncate_text(line)

    for line in lines:
        if not line:
            continue
        if line.startswith(("Exit code:", "Wall time:", "Command:")):
            continue
        return _truncate_text(line)
    return None


def _summarize_agent_status(status: "object") -> "str":
    if isinstance(status, str):
        return status
    if isinstance(status, dict):
        if "completed" in status:
            completed = status.get("completed")
            if completed is None:
                return "completed"
            return f"completed: {_truncate_text(str(completed), limit=48)}"
        if "errored" in status:
            return (
                f"errored: {_truncate_text(str(status.get('errored', '')), limit=48)}"
            )
    return _truncate_text(json.dumps(status, ensure_ascii=False, separators=(",", ":")))


def _summarize_tool_result(result: "ToolResult") -> "typing.Union[str, None]":
    if result.name == "spawn_agent" and isinstance(result.output, dict):
        agent_id = str(result.output.get("agent_id", "")).strip()
        nickname = str(result.output.get("nickname", "")).strip()
        if nickname and agent_id:
            return f"{nickname} ({short_id(agent_id)})"
    if result.name == "send_input" and isinstance(result.output, dict):
        submission_id = str(result.output.get("submission_id", "")).strip()
        if submission_id:
            return f"queued {short_id(submission_id)}"
    if result.name in {"resume_agent", "close_agent"} and isinstance(
        result.output, dict
    ):
        return _summarize_agent_status(result.output.get("status"))
    if result.name == "wait_agent" and isinstance(result.output, dict):
        if result.output.get("timed_out") is True:
            return "timed out"
        status = result.output.get("status")
        if isinstance(status, dict):
            parts: "typing.List[str]" = []
            for agent_id, agent_status in status.items():
                if not isinstance(agent_id, str):
                    continue
                parts.append(
                    f"{short_id(agent_id)}={_summarize_agent_status(agent_status)}"
                )
            if parts:
                return _truncate_text(", ".join(parts), limit=96)
    if result.name == "update_plan" and isinstance(result.output, dict):
        plan = result.output.get("plan")
        if isinstance(plan, list):
            return f"{len(plan)} steps"
    if result.name == "view_image" and isinstance(result.output, list):
        return f"{len(result.output)} image item(s)"
    if isinstance(result.output, (dict, list)):
        return _truncate_text(
            json.dumps(result.output, ensure_ascii=False, separators=(",", ":"))
        )

    preview = _extract_output_preview(result.output_text())
    if preview:
        return preview
    return None


def _string_arg(arguments: "object", key: "str") -> "typing.Union[str, None]":
    if not isinstance(arguments, dict):
        return None
    value = arguments.get(key)
    if value in (None, ""):
        return None
    return str(value)


def _int_arg(arguments: "object", key: "str") -> "typing.Union[int, None]":
    if not isinstance(arguments, dict):
        return None
    value = arguments.get(key)
    if value in (None, ""):
        return None
    return int(value)


def _command_preview(call: "ToolCall") -> "typing.Union[str, None]":
    if call.name == "exec_command":
        cmd = _string_arg(call.arguments, "cmd")
        if cmd:
            return _truncate_text(cmd, limit=72)
    if call.name == "shell_command":
        command = _string_arg(call.arguments, "command")
        if command:
            return _truncate_text(command, limit=72)
    if call.name == "shell" and isinstance(call.arguments, dict):
        command = call.arguments.get("command")
        if isinstance(command, list) and command:
            rendered = " ".join(shlex.quote(str(part)) for part in command)
            return _truncate_text(rendered, limit=72)
    if call.name == "write_stdin":
        session_id = _int_arg(call.arguments, "session_id")
        chars = _string_arg(call.arguments, "chars") or ""
        if session_id is None:
            return None
        if not chars:
            return f"poll session {session_id}"
        return f"session {session_id} <- {_truncate_text(chars, limit=32)}"
    if call.name == "read_file":
        path = _string_arg(call.arguments, "file_path")
        if path:
            return _truncate_text(path, limit=72)
    if call.name == "list_dir":
        path = _string_arg(call.arguments, "dir_path")
        if path:
            return _truncate_text(path, limit=72)
    if call.name == "grep_files":
        pattern = _string_arg(call.arguments, "pattern")
        path = _string_arg(call.arguments, "path")
        if pattern and path:
            return _truncate_text(f"{pattern} @ {path}", limit=72)
        if pattern:
            return _truncate_text(pattern, limit=72)
    if call.name == "view_image":
        path = _string_arg(call.arguments, "path")
        if path:
            return _truncate_text(path, limit=72)
    if call.name == "update_plan" and isinstance(call.arguments, dict):
        plan = call.arguments.get("plan")
        if isinstance(plan, list):
            return _plan_progress_summary(plan)
    if call.name == "send_input":
        agent_id = _string_arg(call.arguments, "id")
        message = _string_arg(call.arguments, "message")
        prefix = f"{short_id(agent_id)} <- " if agent_id else ""
        if message:
            return f"{prefix}{_truncate_text(message, limit=40)}"
        if prefix:
            return prefix.rstrip()
    if call.name in {"resume_agent", "close_agent"}:
        agent_id = _string_arg(call.arguments, "id")
        if agent_id:
            return short_id(agent_id)
    return None


def _plan_progress_summary(plan: "typing.List[object]") -> "str":
    total = len(plan)
    completed = 0
    in_progress = 0

    for item in plan:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "")).strip()
        if status == "completed":
            completed += 1
        elif status == "in_progress":
            in_progress += 1

    if total == 0:
        return "0 steps"
    if completed >= total:
        return f"Done {completed}/{total}"
    if in_progress:
        return f"Working on {completed + in_progress}/{total}"
    return f"Planned {completed}/{total}"


class CliSessionView:
    """Own the interactive CLI terminal surface for one session.

    This class is the single place that knows how to:
    - render `AgentEvent`s into human-facing terminal output;
    - multiplex prompt input, streamed assistant output, and spinner state;
    - keep lightweight session UI state such as title, history, and steer markers.

    Public interface:
    - `handle_event(event)`: feed runtime/agent events into the view.
    - `write_line(text)`, `finish_stream()`, `show_error(text)`: imperative output
      helpers for CLI-side messages that do not come from `AgentEvent`.
    - `show_history()`, `show_title()`, `load_session_history(...)`, `show_steer_queued(...)`,
      `schedule_steer_inserted(...)`: small session UI helpers used by the
      interactive command loop.
    - `close()`: release prompt/spinner resources at shutdown.

    """

    def __init__(
        self,
        context_window_tokens: "typing.Union[int, None]" = None,
    ) -> "None":
        import sys

        self._line_output = print
        self._terminal_lock = threading.RLock()
        self._title: "typing.Union[str, None]" = None
        self._pending_user_prompts: "typing.Dict[str, str]" = {}
        self._queued_steer_prompts: "typing.Dict[str, typing.List[str]]" = {}
        self._inserted_steer_prompts: "typing.Dict[str, typing.List[str]]" = {}
        self._history: "typing.List[typing.Tuple[str, str]]" = []
        self._context_window_tokens = context_window_tokens
        self._context_remaining_percent: "typing.Union[int, None]" = (
            100 if context_window_tokens is not None else None
        )
        self._color_enabled = cli_color_enabled() and sys.stdout.isatty()
        self._agent_names: "typing.Dict[str, str]" = {}

        self.prompter = Prompter(lock=self._terminal_lock)

        self._stream_buffer = "assistant> "

    def handle_event(self, event: "AgentEvent") -> "None":
        if event.kind == "assistant_delta":
            self._stream_buffer += str(event.payload.get("delta", ""))
            self.prompter.set_status(event.kind)
            return

        if event.kind == "turn_started":
            self.prompter.set_status(event.kind, active=True)
            submission_id = str(
                event.payload.get("submission_id", event.turn_id)
            ).strip()
            user_texts = event.payload.get("user_texts")
            if isinstance(user_texts, list):
                normalized_user_texts = [
                    str(text).strip() for text in user_texts if str(text).strip()
                ]
            else:
                normalized_user_texts = []
            user_text = str(event.payload.get("user_text", "")).strip()
            if not user_text and normalized_user_texts:
                user_text = "\n".join(normalized_user_texts)
            if self._title is None and user_text:
                self._title = shorten_title(user_text)
                self._print_line(f"Session: {self._title}")
            if user_text:
                self._pending_user_prompts[submission_id] = user_text
            inserted_steer_prompts = self._inserted_steer_prompts.pop(submission_id, [])
            for inserted_steer_prompt in inserted_steer_prompts:
                self._print_line(
                    colorize_cli_message(
                        f"[steer] inserted: {inserted_steer_prompt}",
                        "status",
                        self._color_enabled,
                    )
                )
            queued_steer_prompts = self._queued_steer_prompts.pop(submission_id, [])
            for queued_steer_prompt in queued_steer_prompts:
                self._print_line(
                    colorize_cli_message(
                        f"[steer] inserted: {queued_steer_prompt}",
                        "status",
                        self._color_enabled,
                    )
                )
            if user_text:
                self._print_user_turn(user_text)
            return

        if event.kind == "model_called":
            self.prompter.set_status(event.kind, active=True)
            return

        if event.kind == "token_count":
            self._update_context_window(event.payload.get("usage"))
            self.prompter.set_prompt(self._format_main_prompt(DEFAULT_MAIN_PROMPT))
            return

        if event.kind == "stream_error":
            message = str(event.payload.get("message", "")).strip() or "Reconnecting..."
            self._print_line(
                colorize_cli_message(
                    f"[status] {message}",
                    "status",
                    self._color_enabled,
                )
            )
            self.prompter.set_status(event.kind, active=True)
            return

        if event.kind == "auto_compact_started":
            total_tokens = event.payload.get("total_tokens")
            token_limit = event.payload.get("token_limit")
            if total_tokens is not None and token_limit is not None:
                message = f"[status] auto-compact: {total_tokens}/{token_limit} tokens"
            else:
                message = "[status] auto-compact"
            self._print_line(
                colorize_cli_message(message, "status", self._color_enabled)
            )
            self.prompter.set_status(event.kind, active=True)
            return

        if event.kind == "auto_compact_completed":
            summary = str(event.payload.get("summary", "")).strip()
            message = f"[status] {summary}" if summary else "[status] context compacted"
            self._print_line(
                colorize_cli_message(message, "status", self._color_enabled)
            )
            self.prompter.set_status(event.kind, active=True)
            return

        if event.kind == "auto_compact_failed":
            self.finish_stream()
            error = str(event.payload.get("error", "")).strip()
            message = (
                f"[error] auto-compact failed: {error}"
                if error
                else "[error] auto-compact failed"
            )
            self._print_line(
                colorize_cli_message(message, "error", self._color_enabled)
            )
            self.prompter.set_status(event.kind, active=True)
            return

        if event.kind == "tool_called":
            tool_name = str(event.payload.get("tool_name", "")).strip()
            message = format_cli_tool_call_message(tool_name, event.payload)
            if message is not None:
                self._print_line(
                    colorize_cli_message(message, "web", self._color_enabled)
                )
                self.prompter.set_status(event.kind, active=True)
            return

        if event.kind == "tool_started":
            tool_name = str(event.payload.get("tool_name", "")).strip()
            call = event.payload.get("call")
            args = None
            if isinstance(call, ToolCall):
                args = call.arguments
            if tool_name and args is not None:
                self.prompter.set_status(
                    shorten_title(f"calling {tool_name}({args})", limit=72), active=True
                )
            elif tool_name:
                self.prompter.set_status(f"calling {tool_name}")
            else:
                self.prompter.set_status("calling provider tools")
            return

        if event.kind == "tool_completed":
            tool_name, summary, is_error = extract_tool_event_display(event.payload)
            summary = self._rewrite_agent_summary(tool_name, summary)
            if tool_name == "update_plan" and not is_error:
                plan_items = extract_plan_event_items(event.payload)
                for line in format_cli_plan_messages(summary, plan_items):
                    self._print_line(
                        colorize_cli_message(line, "plan", self._color_enabled)
                    )
            if tool_name:
                self.prompter.set_status(f"called {tool_name}")
            message = format_cli_tool_message(
                tool_name,
                summary,
                is_error,
            )
            self._remember_agent_name(tool_name, summary)
            self._print_line(self._colorize_formatted_tool_message(message))
            return

        if event.kind == "turn_completed":
            submission_id = str(
                event.payload.get("submission_id", event.turn_id)
            ).strip()
            final_text = str(event.payload.get("output_text", "") or "")
            self._finalize_turn_output(final_text, allow_standalone_output=True)
            pending_prompt = self._pending_user_prompts.pop(submission_id, None)
            if pending_prompt is not None:
                self._history.append((pending_prompt, final_text))
            self.prompter.set_status(event.kind, active=False)
            return

        if event.kind == "turn_failed":
            submission_id = str(
                event.payload.get("submission_id", event.turn_id)
            ).strip()
            self.finish_stream()
            self._pending_user_prompts.pop(submission_id, None)
            self.prompter.set_status(event.kind, active=False)
            return

        if event.kind == "turn_interrupted":
            submission_id = str(
                event.payload.get("submission_id", event.turn_id)
            ).strip()
            final_text = str(event.payload.get("output_text", "") or "")
            self._finalize_turn_output(final_text, allow_standalone_output=False)
            pending_prompt = self._pending_user_prompts.pop(submission_id, None)
            if pending_prompt is not None and final_text:
                self._history.append((pending_prompt, final_text))
            self.prompter.set_status(event.kind, active=False)
            return

    def show_history(self) -> "None":
        self.finish_stream()
        if not self._history:
            self._print_line("No history yet.")
            return

        self._print_line(f"Session: {self._title or 'untitled'}")
        for index, (user_text, assistant_text) in enumerate(self._history, start=1):
            self._print_line(f"[{index}] user> {user_text}")
            self._print_line(f"    assistant> {assistant_text}")

    def show_title(self) -> "None":
        self.finish_stream()
        self._print_line(f"Session: {self._title or 'untitled'}")

    def load_session_history(
        self,
        title: "typing.Union[str, None]",
        history: "typing.Tuple[typing.Tuple[str, str], ...]",
    ) -> "None":
        self.finish_stream()
        self._title = title or None
        self._history = list(history)
        self._pending_user_prompts.clear()
        self._queued_steer_prompts.clear()
        self._inserted_steer_prompts.clear()

    def close(self):
        self.prompter.close()

    async def prompt_async(self, prompt: "str" = None) -> "typing.Union[str, None]":
        if prompt:
            self.prompter.set_prompt(prompt)
        return await self.prompter.poll_input()

    def _update_context_window(self, usage: "object") -> "None":
        if self._context_window_tokens is None:
            return
        if not isinstance(usage, dict):
            self._context_remaining_percent = None
            return
        try:
            total_tokens = int(usage["total_tokens"])
        except (KeyError, TypeError, ValueError):
            self._context_remaining_percent = None
            return
        self._context_remaining_percent = percent_of_context_window_remaining(
            total_tokens,
            self._context_window_tokens,
        )

    def _format_main_prompt(self, prompt: "str") -> "str":
        if prompt != DEFAULT_MAIN_PROMPT:
            return prompt
        if self._context_remaining_percent is None:
            return prompt
        return f"pyco({self._context_remaining_percent}%)> "

    def show_steer_queued(self, turn_id: "str", prompt: "str") -> "None":
        preview = shorten_title(prompt, limit=72)
        self._queued_steer_prompts.setdefault(turn_id, []).append(preview)
        self._print_line(
            colorize_cli_message(
                f"[steer] queued: {preview}",
                "status",
                self._color_enabled,
            )
        )

    def schedule_steer_inserted(self, turn_id: "str", prompt: "str") -> "None":
        self._inserted_steer_prompts.setdefault(turn_id, []).append(
            shorten_title(prompt, limit=72)
        )

    def set_context_window_tokens(
        self,
        context_window_tokens: "typing.Union[int, None]",
    ) -> "None":
        self._context_window_tokens = context_window_tokens
        self._context_remaining_percent = (
            100 if context_window_tokens is not None else None
        )

    def finish_stream(self) -> "None":
        with self._terminal_lock:
            if self._stream_buffer:
                self._print_line(self._stream_buffer)
                self._stream_buffer = "assistant> "
        self.prompter.spinner.resume()

    def write_line(self, text: "str") -> "None":
        self._print_line(text)

    def show_error(self, text: "str") -> "None":
        self.finish_stream()
        lines = str(text).splitlines() or [""]
        formatted = [f"Error: {lines[0]}"]
        formatted.extend(f"  {line}" if line else "" for line in lines[1:])
        self._print_line(
            colorize_cli_message(
                "\n".join(formatted),
                "error",
                self._color_enabled,
            )
        )

    def _finalize_turn_output(
        self,
        final_text: "str",
        allow_standalone_output: "bool",
    ) -> "None":
        if self._stream_buffer:
            self.finish_stream()
            return
        if allow_standalone_output and final_text:
            self._print_line(
                colorize_cli_message(
                    f"assistant> {final_text}",
                    "assistant",
                    self._color_enabled,
                )
            )

    def _colorize_formatted_tool_message(self, message: "str") -> "str":
        if message.startswith("[plan]"):
            return colorize_cli_message(message, "plan", self._color_enabled)
        if message.startswith("[exec]"):
            return colorize_cli_message(message, "exec", self._color_enabled)
        if message.startswith("[agent]"):
            return colorize_cli_message(message, "agent", self._color_enabled)
        if message.startswith("[web]"):
            return colorize_cli_message(message, "web", self._color_enabled)
        if message.startswith("[error]"):
            return colorize_cli_message(message, "error", self._color_enabled)
        return colorize_cli_message(message, "tool", self._color_enabled)

    def _print_line(self, text: "str") -> "None":
        with self._terminal_lock:
            self._line_output(text)

    def _print_user_turn(self, text: "str") -> "None":
        self._print_line(f"user> {text}")

    def _remember_agent_name(self, tool_name: "str", summary: "str") -> "None":
        if tool_name != "spawn_agent":
            return
        if " (" not in summary or not summary.endswith(")"):
            return
        nickname, rest = summary.rsplit(" (", 1)
        agent_short_id = rest[:-1].strip()
        nickname = nickname.strip()
        if not nickname or not agent_short_id:
            return
        self._agent_names[agent_short_id] = nickname

    def _rewrite_agent_summary(self, tool_name: "str", summary: "str") -> "str":
        if tool_name not in {"wait_agent", "send_input", "resume_agent", "close_agent"}:
            return summary
        rewritten = summary
        for agent_short_id, nickname in sorted(
            self._agent_names.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            rewritten = rewritten.replace(agent_short_id, nickname)
        return rewritten
