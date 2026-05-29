import asyncio
import os
import threading
from contextlib import suppress
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from ..protocol import AgentEvent, JSONDict, ToolCall
from .toolcall_visualize import (
    colorize_cli_message,
    colorize_tool_message,
    tool_summary,
)
import typing

STATUS_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
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


class Prompter:
    def __init__(self, prompt: str = DEFAULT_MAIN_PROMPT, lock=None):
        self.lock = lock or threading.Lock()
        self._prompt_session = PromptSession(
            erase_when_done=True,
            enable_system_prompt=True,
            show_frame=True,
        )

        self.prompt = prompt
        self._status = None
        self._status_frame_index = 0

        self._prompt_task = None

    def set_prompt(self, prompt):
        self.prompt = prompt

    def set_status(self, text=None, active=True):
        self._status = text if active else None

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

    async def require_input(self):
        if self._prompt_task and not self._prompt_task.done():
            self._prompt_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._prompt_task

        return await self._block_prompt()

    async def _block_prompt(self):
        with patch_stdout(raw=True):
            return await self._prompt_session.prompt_async(
                lambda: self.prompt,
                refresh_interval=0.12,
                bottom_toolbar=self._get_status,
            )

    def _get_status(self):
        self._status_frame_index += 1
        if self._status is None:
            return None
        else:
            frame = STATUS_FRAMES[self._status_frame_index % len(STATUS_FRAMES)]
            status = f"{frame} {self._status}"
            return status

    def close(self) -> "None":
        if self._prompt_task is not None and not self._prompt_task.done():
            self._prompt_task.cancel()
            self._prompt_task = None


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
        return f"[web_search] searched: {query}" if query else "[web_search] searched"

    if action_type == "open_page":
        url = str(payload.get("url", "")).strip()
        return f"[web_search] opened: {url}" if url else "[web_search] opened"

    if action_type == "find_in_page":
        pattern = str(payload.get("pattern", "")).strip()
        url = str(payload.get("url", "")).strip()
        if pattern and url:
            return f"[web_search] found: {pattern} @ {url}"
        if pattern:
            return f"[web_search] found: {pattern}"
        return "[web_search] found in page"

    return "[web_search] browsing"


def short_id(value: "str", limit: "int" = 8) -> "str":
    compact = value.strip()
    if len(compact) <= limit + 4:
        return compact
    return f"{compact[:limit]}...{compact[-4:]}"


class CliSessionView:
    """Own the interactive CLI terminal surface for one session.

    This class is the single place that knows how to:
    - render `AgentEvent`s into human-facing terminal output;
    - multiplex prompt input, streamed assistant output, and status state;
    - keep lightweight session UI state such as title, history, and steer markers.

    Public interface:
    - `handle_event(event)`: feed runtime/agent events into the view.
    - `write_line(text)`, `finish_stream()`, `show_error(text)`: imperative output
      helpers for CLI-side messages that do not come from `AgentEvent`.
    - `show_history()`, `show_title()`, `load_session_history(...)`, `show_steer_queued(...)`,
      `schedule_steer_inserted(...)`: small session UI helpers used by the
      interactive command loop.
    - `close()`: release prompt resources at shutdown.

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

        self._stream_buffer = ""

    def handle_event(self, event: "AgentEvent") -> "None":
        if event.kind == "assistant_delta":
            self._stream_buffer += str(event.payload.get("delta", ""))
            self.prompter.set_status("talking")
            return
        if event.kind == "token_count":
            self._update_context_window(event.payload.get("usage"))
            self.prompter.set_prompt(self._format_main_prompt(DEFAULT_MAIN_PROMPT))
            return
        if event.kind == "model_completed":
            return
        if event.kind == "turn_completed":
            submission_id = str(
                event.payload.get("submission_id", event.turn_id)
            ).strip()
            final_text = str(event.payload.get("output_text", "") or "")
            if final_text:
                if final_text.startswith(self._stream_buffer):
                    self._stream_buffer += final_text[len(self._stream_buffer) :]
                else:
                    self._stream_buffer += final_text
            self.finish_stream()
            pending_prompt = self._pending_user_prompts.pop(submission_id, None)
            if pending_prompt is not None:
                self._history.append((pending_prompt, final_text))
            self.prompter.set_status(active=False)
            return
        self.finish_stream()

        if event.kind == "turn_started":
            self.prompter.set_status(event.kind)
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
            # self.prompter.set_status("thinking")
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
            self.prompter.set_status("reconnecting")
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
            self.prompter.set_status("compacting")
            return

        if event.kind == "auto_compact_completed":
            summary = str(event.payload.get("summary", "")).strip()
            message = f"[status] {summary}" if summary else "[status] context compacted"
            self._print_line(
                colorize_cli_message(message, "status", self._color_enabled)
            )
            self.prompter.set_status("compacted")
            return

        if event.kind == "auto_compact_failed":
            error = str(event.payload.get("error", "")).strip()
            message = (
                f"[error] auto-compact failed: {error}"
                if error
                else "[error] auto-compact failed"
            )
            self._print_line(
                colorize_cli_message(message, "error", self._color_enabled)
            )
            return

        if event.kind == "tool_called":
            tool_name = str(event.payload.get("tool_name", "")).strip()
            message = format_cli_tool_call_message(tool_name, event.payload)
            if message is not None:
                self._print_line(colorize_tool_message(message, self._color_enabled))
            return

        if event.kind == "tool_started":
            tool_name = str(event.payload.get("tool_name", "")).strip()
            call = event.payload.get("call")
            args = None
            if isinstance(call, ToolCall):
                args = call.arguments
            if tool_name and args is not None:
                self.prompter.set_status(
                    shorten_title(f"calling {tool_name}({args})", limit=72)
                )
            elif tool_name:
                self.prompter.set_status(f"calling {tool_name}")
            else:
                self.prompter.set_status("calling provider tools")
            return

        if event.kind == "tool_completed":
            tool_name = str(event.payload.get("tool_name", "")).strip()
            message = tool_summary(event.payload)
            if tool_name:
                self.prompter.set_status(f"called {tool_name}")
            message = self._replace_agent_ids_with_names(tool_name, message)
            if tool_name == "spawn_agent":
                agent_summary = message
                prefix = "[spawn_agent] spawned "
                if agent_summary.startswith(prefix):
                    agent_summary = agent_summary[len(prefix) :]
                self._remember_agent_name(tool_name, agent_summary)
            for line in message.splitlines() or [""]:
                self._print_line(
                    colorize_tool_message(line, self._color_enabled, tool_name)
                )
            return

        if event.kind == "turn_failed":
            submission_id = str(
                event.payload.get("submission_id", event.turn_id)
            ).strip()
            self._pending_user_prompts.pop(submission_id, None)
            self.prompter.set_status(active=False)
            return

        if event.kind == "turn_interrupted":
            submission_id = str(
                event.payload.get("submission_id", event.turn_id)
            ).strip()
            final_text = str(event.payload.get("output_text", "") or "")
            pending_prompt = self._pending_user_prompts.pop(submission_id, None)
            if pending_prompt is not None and final_text:
                self._history.append((pending_prompt, final_text))
            self.prompter.set_status(active=False)
            return

    def show_history(self) -> "None":
        self.finish_stream()
        if not self._history:
            self._print_line("No history yet.")
            return

        self._print_line(f"Session: {self._title or 'untitled'}")
        for index, (user_text, assistant_text) in enumerate(self._history, start=1):
            self._print_line(f"[{index}]U> {user_text}")
            self._print_line(f"[{index}]A> {assistant_text}")

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

    async def poll_prompt(self, prompt: "str" = None) -> "typing.Union[str, None]":
        if prompt:
            self.prompter.set_prompt(prompt)
        return await self.prompter.poll_input()

    async def get_prompt(self, prompt: "str" = None) -> "str":
        if prompt:
            self.prompter.set_prompt(prompt)
        return await self.prompter.require_input()

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
                self._print_line("assistant> " + self._stream_buffer)
                self._stream_buffer = ""

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

    def _print_line(self, text: "str") -> "None":
        with self._terminal_lock:
            self._line_output(text)

    def _print_user_turn(self, text: "str") -> "None":
        self._print_line(
            colorize_cli_message(f"user> {text}", "assistant", self._color_enabled)
        )

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

    def _replace_agent_ids_with_names(self, tool_name: "str", message: "str") -> "str":
        if tool_name not in {"wait_agent", "send_input", "resume_agent", "close_agent"}:
            return message
        for agent_short_id, nickname in sorted(
            self._agent_names.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            message = message.replace(agent_short_id, nickname)
        return message
