import argparse
import asyncio
import html
import json
import os
import threading
import tempfile
from uuid import uuid4
from dataclasses import asdict, is_dataclass
try:
    from contextlib import asynccontextmanager
except ImportError:  # pragma: no cover - Python 3.6 compatibility
    asynccontextmanager = None
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response

from pycodex.cli import build_agent, build_cli_queue, build_model, configure_loguru
from pycodex.interactive_session import run_interactive_session
from pycodex.model import DEFAULT_CODEX_CONFIG_PATH
from pycodex.protocol import AgentEvent, ToolCall
from pycodex.utils.session_persist import (
    SessionRolloutRecorder,
    load_resumed_session_path,
)
from pycodex.utils import uuid7_string
from pycodex.utils.visualize import (
    IDLE_LISTENING_STATUS,
    percent_of_context_window_remaining,
    shorten_title,
    tool_summary,
)
import typing


JSONValue = typing.Union[
    None,
    bool,
    int,
    float,
    str,
    typing.List["JSONValue"],
    typing.Dict[str, "JSONValue"],
]


class WorkspaceStateStore:
    def __init__(self, board_path: "typing.Union[Path, None]") -> None:
        self.path = None if board_path is None else board_path.with_suffix(".pycodex-ws.json")

    def load_tabs(self) -> "typing.List[typing.Dict[str, str]]":
        if self.path is None or not self.path.is_file():
            return []

        try:
            payload = json.loads(
                self.path.read_text(encoding="utf-8", errors="replace") or "{}"
            )
        except (OSError, ValueError):
            return []

        tabs = payload.get("tabs") if isinstance(payload, dict) else None
        if not isinstance(tabs, list):
            return []

        result = []
        for tab in tabs:
            if not isinstance(tab, dict):
                continue
            title = str(tab.get("title") or "").strip()
            rollout_path = str(tab.get("rollout_path") or "").strip()
            if title or rollout_path:
                result.append({"title": title, "rollout_path": rollout_path})
        return result

    def save_tabs(self, tabs: "typing.Iterable[typing.Dict[str, str]]") -> None:
        if self.path is None:
            return

        state_tabs = [
            {
                "title": str(tab.get("title") or ""),
                "rollout_path": str(tab.get("rollout_path") or ""),
            }
            for tab in tabs
        ]
        payload = json.dumps(
            {"version": 1, "tabs": state_tabs},
            ensure_ascii=False,
            indent=2,
        ) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(payload, encoding="utf-8")


def build_parser() -> "argparse.ArgumentParser":
    parser = argparse.ArgumentParser(
        prog="pycodex-ws",
        description="Run a local pycodex workspace with a board and chat session.",
    )
    parser.add_argument(
        "--listen",
        default="127.0.0.1:6007",
        help="Bind address as host:port, for example 0.0.0.0:6007.",
    )
    parser.add_argument(
        "--board",
        default=None,
        help="Optional board HTML path. Defaults to a writable random path under /tmp.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CODEX_CONFIG_PATH),
        help="Path to Codex config.toml.",
    )
    parser.add_argument("--profile", default=None, help="Optional profile name.")
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional base instructions override passed to the model.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="HTTP timeout for one model call.",
    )
    parser.add_argument(
        "--vllm-endpoint",
        default=None,
        help="Optional base URL for a chat-completions-backed vLLM server.",
    )
    parser.add_argument(
        "--use-chat-completion",
        default=False,
        action="store_true",
        help="Start a local responses compat server for this session.",
    )
    parser.add_argument(
        "--use-messages",
        default=False,
        action="store_true",
        help="Route through a downstream /v1/messages backend.",
    )
    return parser


def parse_target(
    target: str,
    board: "typing.Union[str, None]" = None,
) -> "typing.Tuple[str, int, typing.Union[Path, None]]":
    target_text = str(target or "").strip() or "127.0.0.1:6007"
    board_text = board
    if "+" in target_text:
        target_text, suffix = target_text.split("+", 1)
        if board_text is None:
            board_text = suffix
    if ":" not in target_text:
        raise ValueError("workspace target must look like host:port")
    host, port_text = target_text.rsplit(":", 1)
    host = host.strip() or "127.0.0.1"
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError("workspace port must be an integer") from exc
    board_path = Path(board_text).expanduser().resolve() if board_text else None
    return host, port, board_path


SessionFactory = typing.Callable[[], object]
ThreadedSessionFactory = typing.Callable[[], "WorkspaceInteractiveSession"]
SESSION_CLOSE_TIMEOUT_SECONDS = 2.0
SPINNER_STATUS_PREVIEW_LIMIT = 180


class WebSessionView:
    def __init__(self) -> None:
        self._input_queue: "asyncio.Queue" = asyncio.Queue()
        self._subscribers: "typing.Set[asyncio.Queue]" = set()
        self._events: "typing.List[typing.Dict[str, object]]" = []
        self._turns: "typing.List[typing.Dict[str, object]]" = []
        self._turns_by_submission_id: "typing.Dict[str, typing.Dict[str, object]]" = {}
        self._turns_by_turn_id: "typing.Dict[str, typing.Dict[str, object]]" = {}
        self._title = ""
        self._spinner_status = ""
        self._stream_buffer = ""
        self._context_window_tokens: "typing.Union[int, None]" = None
        self._context_remaining_percent: "typing.Union[int, None]" = None
        self._closed = False
        self._server_loop: "typing.Union[asyncio.AbstractEventLoop, None]" = None
        self._worker_loop: "typing.Union[asyncio.AbstractEventLoop, None]" = None
        self._lock = threading.RLock()

    def attach_server_loop(self, loop: "asyncio.AbstractEventLoop") -> None:
        self._server_loop = loop

    def attach_worker_loop(self, loop: "asyncio.AbstractEventLoop") -> None:
        self._worker_loop = loop

    async def submit(self, prompt: str) -> "typing.Dict[str, object]":
        prompt = str(prompt or "").strip()
        if not prompt:
            return {"ok": False, "error": "prompt is empty"}
        await self._put_input(prompt)
        await self._publish(
            {
                "type": "input",
                "prompt": prompt,
                "snapshot": self.snapshot(),
            }
        )
        return {"ok": True, "type": "submitted", "snapshot": self.snapshot()}

    async def _put_input(self, item: object) -> None:
        worker_loop = self._worker_loop
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if worker_loop is None or worker_loop is running_loop:
            await self._input_queue.put(item)
            return
        future = asyncio.run_coroutine_threadsafe(self._input_queue.put(item), worker_loop)
        await asyncio.wrap_future(future)

    async def poll_prompt(self, prompt: "typing.Union[str, None]" = None) -> "typing.Union[str, None]":
        del prompt
        if self._closed and self._input_queue.empty():
            raise EOFError()
        try:
            item = self._input_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        if item is None:
            raise EOFError()
        return str(item)

    async def get_prompt(self, prompt: "typing.Union[str, None]" = None) -> "str":
        if prompt:
            self.write_line(prompt)
        item = await self._input_queue.get()
        if item is None:
            raise EOFError()
        return str(item)

    def handle_event(self, event: "AgentEvent") -> None:
        with self._lock:
            self._apply_runtime_event(event)
            payload = {
                "type": "event",
                "kind": str(getattr(event, "kind", "")),
                "turn_id": str(getattr(event, "turn_id", "")),
                "payload": _json_safe(getattr(event, "payload", {})),
                "snapshot": self.snapshot(),
            }
            if payload["kind"] == "tool_completed":
                payload["summary"] = tool_summary(getattr(event, "payload", {}))
        self._publish_nowait(payload)

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
        self._publish_nowait(event)

    def write_line(self, text: str) -> None:
        with self._lock:
            text = str(text or "")
            turn = self._new_control_turn(text)
            turn["response"] = text
            turn["status"] = "completed"
            event = {"type": "snapshot", "snapshot": self.snapshot()}
        self._publish_nowait(event)

    def show_error(self, text: str) -> None:
        self.finish_stream()
        with self._lock:
            turn = self._new_control_turn("")
            turn["error"] = str(text or "")
            turn["status"] = "error"
            event = {"type": "snapshot", "snapshot": self.snapshot()}
        self._publish_nowait(event)

    def show_history(self) -> None:
        assistant_turns = [turn for turn in self._turns if turn.get("kind") != "control"]
        if not assistant_turns:
            self.write_line("No history yet.")
            return
        lines = ["Session: {0}".format(self._title or "untitled")]
        for index, turn in enumerate(assistant_turns, start=1):
            prompt = str(turn.get("prompt") or "")
            response = str(turn.get("response") or turn.get("thinking") or "")
            lines.append("[{0}]U> {1}".format(index, prompt))
            if response:
                lines.append("[{0}]A> {1}".format(index, response))
        self.write_line("\n".join(lines))

    def show_title(self) -> None:
        self.write_line("Session: {0}".format(self._title or "untitled"))

    def set_session_title(self, title: str) -> None:
        with self._lock:
            self._set_title(title)
            event = {
                "type": "title_changed",
                "title": self._title,
                "snapshot": self.snapshot(),
            }
        self._publish_nowait(event)

    def show_resumed_session(self, title: str) -> None:
        with self._lock:
            self._set_title(title)
            event = {"type": "snapshot", "snapshot": self.snapshot()}
        self._publish_nowait(event)

    def load_session_history(
        self,
        title: "typing.Union[str, None]",
        history: "typing.Iterable[typing.Tuple[str, str]]",
    ) -> None:
        self.finish_stream()
        with self._lock:
            self._set_title(title)
            self._turns = []
            self._turns_by_submission_id = {}
            self._turns_by_turn_id = {}
            self._events = []
            for prompt, response in history:
                submission_id = uuid7_string()
                turn = self._ensure_turn(submission_id, submission_id, str(prompt or ""))
                turn["response"] = str(response or "")
                turn["status"] = "completed"
                turn["queue"] = "history"
                turn["sender"] = "resume"
            event = {"type": "snapshot", "snapshot": self.snapshot()}
        self._publish_nowait(event)

    def show_steer_queued(self, turn_id: str, prompt: str) -> None:
        del turn_id, prompt

    def schedule_steer_inserted(self, turn_id: str, prompt: str) -> None:
        del turn_id, prompt

    def set_context_window_tokens(
        self,
        context_window_tokens: "typing.Union[int, None]",
    ) -> None:
        with self._lock:
            self._context_window_tokens = context_window_tokens
            self._context_remaining_percent = (
                100 if context_window_tokens is not None else None
            )

    def subscribe(self) -> "asyncio.Queue":
        queue: "asyncio.Queue" = asyncio.Queue()
        with self._lock:
            self._subscribers.add(queue)
            event = {
                "type": "hello",
                "events": list(self._events[-200:]),
                "snapshot": self.snapshot(),
            }
        queue.put_nowait(event)
        return queue

    def unsubscribe(self, queue: "asyncio.Queue") -> None:
        with self._lock:
            self._subscribers.discard(queue)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            subscribers = tuple(self._subscribers)
            self._subscribers.clear()
        worker_loop = self._worker_loop
        if worker_loop is None:
            self._input_queue.put_nowait(None)
        else:
            asyncio.run_coroutine_threadsafe(self._input_queue.put(None), worker_loop)
        self._publish_to_queues(subscribers, None)

    def snapshot(self) -> "typing.Dict[str, object]":
        with self._lock:
            return {
                "running": bool(self._spinner_status),
                "status": self._spinner_status,
                "status_kind": "spinner" if self._spinner_status else "idle",
                "spinner": self._spinner_status,
                "model": "pycodex",
                "title": self._title,
                "context_remaining_percent": self._context_remaining_percent,
                "turns": [_public_turn(turn) for turn in self._turns[-80:]],
            }

    def summary(self) -> "typing.Dict[str, object]":
        with self._lock:
            return {
                "running": bool(self._spinner_status),
                "spinner": self._spinner_status,
                "title": self._title,
                "turn_count": len(self._turns),
                "last_assistant": _last_assistant_text(self._turns),
                "context_remaining_percent": self._context_remaining_percent,
            }

    def _apply_runtime_event(self, event: "AgentEvent") -> None:
        kind = str(getattr(event, "kind", "") or "")
        payload = getattr(event, "payload", {})
        if not isinstance(payload, dict):
            payload = {}
        if kind == "token_count":
            self._update_context_window(payload.get("usage"))
            return
        turn_id = str(payload.get("turn_id") or getattr(event, "turn_id", "") or "")
        submission_id = str(payload.get("submission_id") or turn_id or "")
        turn = self._turns_by_submission_id.get(submission_id)
        if turn is None and turn_id and not submission_id:
            turn = self._turns_by_turn_id.get(turn_id)

        if kind == "turn_started":
            self._set_spinner_status(kind)
            prompt = payload.get("user_text") or "\n".join(
                str(item) for item in payload.get("user_texts", []) or []
            )
            if not self._title and str(prompt or "").strip():
                self._set_title(shorten_title(str(prompt or "")))
            turn = self._ensure_turn(submission_id, turn_id, str(prompt or ""))
            turn["status"] = "running"
            turn["thinking"] = ""
            turn["_thinking_active"] = False
            turn["error"] = ""
            return

        self._apply_spinner_event(kind, payload)
        if turn is None:
            return

        if kind == "assistant_delta":
            turn["status"] = "responding"
            delta = str(payload.get("delta") or "")
            self._stream_buffer += delta
            if turn.get("_thinking_active"):
                turn["thinking"] = str(turn.get("thinking") or "") + delta
            else:
                turn["thinking"] = delta
                turn["_thinking_active"] = True
            return

        if kind == "tool_started":
            turn["status"] = "tool"
            turn["tool_name"] = str(payload.get("tool_name") or "")
            turn["_thinking_active"] = False
            return

        if kind == "tool_completed":
            turn["_thinking_active"] = False
            turn["status"] = "running"
            return

        if kind == "turn_completed":
            response = str(payload.get("output_text") or "")
            if response:
                turn["response"] = response
            elif turn.get("thinking"):
                turn["response"] = str(turn.get("thinking") or "")
            turn["thinking"] = ""
            turn["_thinking_active"] = False
            turn["status"] = "completed"
            self._stream_buffer = ""
            return

        if kind in {"turn_failed", "submission_failed"}:
            turn["status"] = "error"
            turn["error"] = str(payload.get("error") or kind)
            self._stream_buffer = ""
            return

        if kind in {"turn_interrupted", "submission_cancelled"}:
            if turn.get("thinking") and not turn.get("response"):
                turn["response"] = str(turn.get("thinking") or "")
            turn["thinking"] = ""
            turn["_thinking_active"] = False
            turn["status"] = "interrupted"
            self._stream_buffer = ""

    def _update_context_window(self, usage: "object") -> None:
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

    def _apply_spinner_event(
        self,
        kind: str,
        payload: "typing.Dict[str, object]",
    ) -> None:
        if kind == "assistant_delta":
            self._set_spinner_status("talking")
            return
        if kind == "stream_error":
            self._set_spinner_status("reconnecting")
            return
        if kind == "auto_compact_started":
            self._set_spinner_status("compacting")
            return
        if kind == "auto_compact_completed":
            self._set_spinner_status("compacted")
            return
        if kind == "tool_started":
            tool_name = str(payload.get("tool_name") or "").strip()
            call = payload.get("call")
            if tool_name and isinstance(call, ToolCall):
                self._set_spinner_status(
                    shorten_title(
                        "calling {0}({1})".format(tool_name, call.arguments),
                        limit=SPINNER_STATUS_PREVIEW_LIMIT,
                    )
                )
            elif tool_name:
                self._set_spinner_status("calling {0}".format(tool_name))
            else:
                self._set_spinner_status("calling provider tools")
            return
        if kind == "tool_completed":
            tool_name = str(payload.get("tool_name") or "").strip()
            if tool_name:
                self._set_spinner_status("called {0}".format(tool_name))
            return
        if kind in {"turn_completed", "turn_failed", "turn_interrupted"}:
            self._set_idle_spinner_status(payload)
            return
        if kind == "submission_failed":
            self._set_spinner_status("")

    def _set_spinner_status(self, text: "typing.Union[str, None]") -> None:
        self._spinner_status = str(text or "").strip()

    def _set_idle_spinner_status(self, payload: "typing.Dict[str, object]") -> None:
        try:
            background_work_count = int(payload.get("background_exec_count", 0))
        except (TypeError, ValueError):
            background_work_count = 0
        if background_work_count > 0:
            self._set_spinner_status(IDLE_LISTENING_STATUS)
        else:
            self._set_spinner_status("")

    def _ensure_turn(
        self,
        submission_id: str,
        turn_id: str,
        prompt: str,
    ) -> "typing.Dict[str, object]":
        submission_id = str(submission_id or "").strip()
        turn_id = str(turn_id or submission_id).strip()
        turn = self._turns_by_submission_id.get(submission_id)
        if turn is None and turn_id and not submission_id:
            turn = self._turns_by_turn_id.get(turn_id)
        if turn is None:
            turn = {
                "submission_id": submission_id,
                "turn_id": turn_id,
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
            self._turns.append(turn)
        if submission_id:
            turn["submission_id"] = submission_id
            self._turns_by_submission_id[submission_id] = turn
        if turn_id:
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

    def _publish_nowait(self, event: "typing.Dict[str, object]") -> None:
        with self._lock:
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
        loop = self._server_loop
        if loop is None:
            for queue in queues:
                queue.put_nowait(event)
            return

        def publish() -> None:
            for queue in queues:
                queue.put_nowait(event)

        loop.call_soon_threadsafe(publish)

    async def _publish(self, event: "typing.Dict[str, object]") -> None:
        self._publish_nowait(event)


class WorkspaceInteractiveSession:
    def __init__(
        self,
        queue,
        config_path: "typing.Union[str, None]" = None,
    ) -> None:
        self.queue = queue
        self.config_path = config_path
        self.view = WebSessionView()
        self._task: "typing.Union[asyncio.Task[int], None]" = None

    async def start(self) -> "WorkspaceInteractiveSession":
        if self._task is None:
            self._task = asyncio.create_task(
                run_interactive_session(
                    self.queue,
                    False,
                    self.config_path,
                    view=self.view,
                    show_banner=False,
                )
            )
        return self

    async def close(self) -> None:
        self.view.close()
        task = self._task
        if task is None:
            return
        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=SESSION_CLOSE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            cancel_current = getattr(self.queue, "cancel_current", None)
            if callable(cancel_current):
                cancel_current()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        finally:
            self._task = None

    async def submit(self, prompt: str, sender: str = "web") -> "typing.Dict[str, object]":
        del sender
        result = await self.view.submit(prompt)
        result["snapshot"] = self.snapshot()
        return result

    def subscribe(self) -> "asyncio.Queue":
        return self.view.subscribe()

    def unsubscribe(self, queue: "asyncio.Queue") -> None:
        self.view.unsubscribe(queue)

    def snapshot(self) -> "typing.Dict[str, object]":
        snapshot = self.view.snapshot()
        agent = getattr(self.queue, "_agent", None)
        snapshot["model"] = getattr(getattr(agent, "_model_client", None), "model", "pycodex")
        return snapshot

    def summary(self) -> "typing.Dict[str, object]":
        summary = self.view.summary()
        agent = getattr(self.queue, "_agent", None)
        summary["model"] = getattr(getattr(agent, "_model_client", None), "model", "pycodex")
        return summary

    def rollout_path(self) -> str:
        recorder = getattr(getattr(self.queue, "_agent", None), "_rollout_recorder", None)
        path = getattr(recorder, "rollout_path", None)
        return "" if path is None else str(path)

    async def restore_from_rollout(self, rollout_path: str, title: str = "") -> None:
        resumed = load_resumed_session_path(rollout_path, thread_name=title or None)
        agent = self.queue._agent
        agent.replace_history(resumed["history"])
        model_client = getattr(agent, "_model_client", None)
        if hasattr(model_client, "_session_id"):
            model_client._session_id = str(resumed["session_id"])
        agent.set_rollout_recorder(SessionRolloutRecorder.resume(resumed["rollout_path"]))
        self.view.load_session_history(
            str(title or resumed["title"]),
            tuple(resumed["turns"]),
        )


class ThreadedWorkspaceInteractiveSession:
    def __init__(
        self,
        session_factory: "ThreadedSessionFactory",
        server_loop: "asyncio.AbstractEventLoop",
    ) -> None:
        self._session_factory = session_factory
        self._server_loop = server_loop
        self._view = WebSessionView()
        self._view.attach_server_loop(server_loop)
        self._thread: "typing.Union[threading.Thread, None]" = None
        self._worker_loop: "typing.Union[asyncio.AbstractEventLoop, None]" = None
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._startup_error: "typing.Union[BaseException, None]" = None
        self._session: "typing.Union[WorkspaceInteractiveSession, None]" = None

    async def start(self) -> "ThreadedWorkspaceInteractiveSession":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._thread_main,
            name="pycodex-workspace-session",
            daemon=True,
        )
        self._thread.start()
        await asyncio.to_thread(self._ready.wait)
        if self._startup_error is not None:
            raise RuntimeError("workspace session thread failed to start") from self._startup_error
        return self

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._worker_loop = loop
        self._view.attach_worker_loop(loop)
        asyncio.set_event_loop(loop)
        try:
            session = self._session_factory()
            session.view = self._view
            self._session = session
            loop.run_until_complete(session.start())
            self._ready.set()
            loop.run_forever()
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
        finally:
            session = self._session
            if session is not None:
                try:
                    loop.run_until_complete(session.close())
                except BaseException:
                    pass
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            asyncio.set_event_loop(None)
            loop.close()
            self._closed.set()

    async def close(self) -> None:
        session = self._session
        loop = self._worker_loop
        if session is not None and loop is not None and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(session.close(), loop)
            try:
                await asyncio.wait_for(
                    asyncio.wrap_future(future),
                    timeout=SESSION_CLOSE_TIMEOUT_SECONDS + 1.0,
                )
            except (asyncio.TimeoutError, RuntimeError):
                cancel_current = getattr(getattr(session, "queue", None), "cancel_current", None)
                if callable(cancel_current):
                    cancel_current()
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        thread = self._thread
        if thread is not None:
            await asyncio.to_thread(thread.join, SESSION_CLOSE_TIMEOUT_SECONDS + 1.0)
        self._thread = None

    async def submit(self, prompt: str, sender: str = "web") -> "typing.Dict[str, object]":
        del sender
        result = await self._view.submit(prompt)
        result["snapshot"] = self.snapshot()
        return result

    def subscribe(self) -> "asyncio.Queue":
        return self._view.subscribe()

    def unsubscribe(self, queue: "asyncio.Queue") -> None:
        self._view.unsubscribe(queue)

    def snapshot(self) -> "typing.Dict[str, object]":
        snapshot = self._view.snapshot()
        session = self._session
        queue = getattr(session, "queue", None)
        agent = getattr(queue, "_agent", None)
        snapshot["model"] = getattr(getattr(agent, "_model_client", None), "model", "pycodex")
        return snapshot

    def summary(self) -> "typing.Dict[str, object]":
        summary = self._view.summary()
        session = self._session
        queue = getattr(session, "queue", None)
        agent = getattr(queue, "_agent", None)
        summary["model"] = getattr(getattr(agent, "_model_client", None), "model", "pycodex")
        return summary

    def rollout_path(self) -> str:
        if self._session is None:
            return ""
        return self._session.rollout_path()

    async def restore_from_rollout(self, rollout_path: str, title: str = "") -> None:
        session = self._session
        loop = self._worker_loop
        if session is None or loop is None:
            return

        future = asyncio.run_coroutine_threadsafe(
            session.restore_from_rollout(rollout_path, title=title),
            loop,
        )
        await asyncio.wrap_future(future)


class WorkspaceSessionManager:
    def __init__(
        self,
        session_factory: "SessionFactory",
        board_path: "typing.Union[Path, None]" = None,
    ) -> None:
        self._session_factory = session_factory
        self._sessions: "typing.Dict[str, WorkspaceInteractiveSession]" = {}
        self._session_order: "typing.List[str]" = []
        self._state_watchers: "typing.Dict[str, asyncio.Task]" = {}
        self._persisted_titles: "typing.Dict[str, str]" = {}
        self._lock = asyncio.Lock()
        self._state_store = WorkspaceStateStore(board_path)

    async def start(self) -> None:
        state_tabs = self._state_store.load_tabs()
        if not state_tabs:
            await self.create_session()
            return
        for tab in state_tabs:
            await self.create_session(
                title=str(tab.get("title") or ""),
                rollout_path=str(tab.get("rollout_path") or ""),
            )

    async def close(self) -> None:
        sessions = list(self._sessions.values())
        watchers = list(self._state_watchers.values())
        self._sessions.clear()
        self._session_order = []
        self._state_watchers.clear()
        self._persisted_titles.clear()
        for watcher in watchers:
            watcher.cancel()
        if watchers:
            await asyncio.gather(*watchers, return_exceptions=True)
        for session in sessions:
            await session.close()

    async def create_session(
        self,
        title: str = "",
        rollout_path: str = "",
    ) -> str:
        async with self._lock:
            session_id = uuid7_string()
            session = self._session_factory()
            await session.start()

            if rollout_path:
                await session.restore_from_rollout(rollout_path, title=title)

            self._sessions[session_id] = session
            self._session_order.append(session_id)
            self._persisted_titles[session_id] = str(
                _session_summary(session).get("title") or ""
            )
            self._state_watchers[session_id] = asyncio.create_task(
                self._watch_session_title(session_id, session)
            )
            return session_id

    async def close_session(self, session_id: str) -> None:
        async with self._lock:
            if len(self._session_order) <= 1:
                raise ValueError("cannot close the last session")
            session = self._sessions.pop(session_id, None)
            if session is None:
                raise KeyError(session_id)
            watcher = self._state_watchers.pop(session_id, None)
            self._persisted_titles.pop(session_id, None)
            self._session_order = [
                item for item in self._session_order if item != session_id
            ]
        if watcher is not None:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        await session.close()
        self.persist_workspace_state()

    async def _watch_session_title(self, session_id: str, session) -> None:
        subscriber = session.subscribe()
        try:
            while True:
                event = await subscriber.get()
                if event is None:
                    return
                if not isinstance(event, dict) or event.get("type") != "title_changed":
                    continue
                title = str(event.get("title") or "")
                if title == self._persisted_titles.get(session_id, ""):
                    continue
                self._persisted_titles[session_id] = title
                self.persist_workspace_state()
        finally:
            session.unsubscribe(subscriber)

    def persist_workspace_state(self) -> None:
        tabs = []
        for session_id in self._session_order:
            session = self._sessions.get(session_id)
            if session is None:
                continue
            summary = _session_summary(session)
            title = str(summary.get("title") or "").strip()
            rollout_path = str(session.rollout_path() or "")
            if not title and not rollout_path:
                continue
            tabs.append({"title": title, "rollout_path": rollout_path})
        self._state_store.save_tabs(tabs)

    def get(self, session_id: "typing.Union[str, None]" = None) -> "WorkspaceInteractiveSession":
        resolved_id = self.resolve_session_id(session_id)
        try:
            return self._sessions[resolved_id]
        except KeyError:
            raise KeyError(resolved_id)

    def resolve_session_id(self, session_id: "typing.Union[str, None]" = None) -> str:
        if session_id:
            return str(session_id)
        if not self._session_order:
            raise KeyError("no sessions")
        return self._session_order[0]

    def list_sessions(self) -> "typing.List[typing.Dict[str, object]]":
        result = []
        for session_id in self._session_order:
            session = self._sessions[session_id]
            summary = _session_summary(session)
            result.append(
                {
                    "id": session_id,
                    "title": summary.get("title") or "pycodex",
                    "running": bool(summary.get("running")),
                    "spinner": summary.get("spinner") or "",
                    "turn_count": summary.get("turn_count") or 0,
                    "last_assistant": summary.get("last_assistant") or "",
                    "context_remaining_percent": summary.get(
                        "context_remaining_percent"
                    ),
                }
            )
        return result


def create_app(
    session_source: "typing.Union[WorkspaceSessionManager, SessionFactory]",
    board_path: "typing.Union[Path, None]",
) -> FastAPI:
    manager = (
        session_source
        if isinstance(session_source, WorkspaceSessionManager)
        else WorkspaceSessionManager(session_source, board_path)
    )

    if asynccontextmanager is not None:
        @asynccontextmanager
        async def lifespan(_app):
            await manager.start()
            try:
                yield
            finally:
                await manager.close()

        app = FastAPI(lifespan=lifespan)
    else:
        app = FastAPI()

        @app.on_event("startup")
        async def startup() -> None:
            await manager.start()

        @app.on_event("shutdown")
        async def shutdown() -> None:
            await manager.close()

    @app.get("/")
    async def index() -> HTMLResponse:
        return _html_response(_render_workspace_shell(board_path))

    @app.get("/favicon.ico")
    async def favicon() -> Response:
        return Response(status_code=204)

    @app.api_route("/board", methods=["GET", "HEAD"])
    async def board() -> Response:
        return _board_response(board_path)

    @app.api_route("/{path_name}", methods=["GET", "HEAD"])
    async def legacy_board_path(path_name: str) -> Response:
        if board_path is not None and path_name == board_path.name:
            return _board_response(board_path)
        raise HTTPException(status_code=404, detail="not found")

    @app.get("/api/board")
    async def board_status() -> JSONResponse:
        if board_path is None or not board_path.is_file():
            return JSONResponse({"exists": False})
        stat = board_path.stat()
        return JSONResponse(
            {
                "exists": True,
                "path": str(board_path),
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
            }
        )

    @app.get("/ws/session")
    async def websocket_backend_hint() -> JSONResponse:
        return JSONResponse(
            {
                "error": "websocket backend is unavailable; HTTP polling is active",
            },
            status_code=426,
        )

    def _board_response(board_path: "typing.Union[Path, None]") -> Response:
        if board_path is None:
            return _html_response(_render_empty_board())
        if not board_path.is_file():
            return _html_response(_render_missing_board(board_path))
        return _html_response(board_path.read_text(encoding="utf-8", errors="replace"))

    @app.get("/api/sessions")
    async def sessions() -> JSONResponse:
        return JSONResponse({"sessions": manager.list_sessions()})

    @app.post("/api/sessions")
    async def new_session() -> JSONResponse:
        session_id = await manager.create_session()
        return JSONResponse(
            {
                "ok": True,
                "session_id": session_id,
                "sessions": manager.list_sessions(),
                "snapshot": _session_snapshot(manager.get(session_id)),
            }
        )

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str) -> JSONResponse:
        try:
            await manager.close_session(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="session not found")
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, "sessions": manager.list_sessions()})

    @app.get("/api/session")
    async def session(
        session_id: "typing.Union[str, None]" = None,
    ) -> JSONResponse:
        try:
            resolved_id = manager.resolve_session_id(session_id)
            link = manager.get(resolved_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="session not found")
        return JSONResponse(
            {
                "session_id": resolved_id,
                "sessions": manager.list_sessions(),
                "snapshot": _session_snapshot(link),
            }
        )

    @app.post("/api/session/message")
    async def message(
        payload: "typing.Dict[str, object]",
    ) -> JSONResponse:
        session_id = str(payload.get("session_id") or "")
        try:
            link = manager.get(session_id or None)
        except KeyError:
            raise HTTPException(status_code=404, detail="session not found")
        result = await link.submit(str(payload.get("prompt") or ""))
        if isinstance(result, dict):
            result.setdefault("sessions", manager.list_sessions())
        status = 200 if result.get("ok") else 400
        return JSONResponse(result, status_code=status)

    @app.websocket("/ws/session")
    async def websocket_session(websocket: WebSocket) -> None:
        await websocket.accept()
        session_id = str(websocket.query_params.get("session_id") or "")
        try:
            link = manager.get(session_id or None)
        except KeyError:
            await websocket.close(code=1008)
            return
        subscriber = link.subscribe()
        sender = asyncio.create_task(_send_ws_events(websocket, subscriber))
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    payload = json.loads(data)
                except ValueError:
                    await websocket.send_json({"type": "error", "error": "invalid json"})
                    continue
                action = str(payload.get("type") or payload.get("action") or "")
                if action == "send":
                    target_session_id = str(payload.get("session_id") or session_id or "")
                    try:
                        target_link = manager.get(target_session_id or None)
                    except KeyError:
                        await websocket.send_json(
                            {"type": "error", "error": "session not found"}
                        )
                        continue
                    result = await target_link.submit(
                        str(payload.get("prompt") or ""),
                        sender=str(payload.get("sender") or "web"),
                    )
                    await websocket.send_json({"type": "send_result", "result": result})
                elif action == "ping":
                    await websocket.send_json({"type": "pong"})
                else:
                    await websocket.send_json({"type": "error", "error": "unknown action"})
        except WebSocketDisconnect:
            pass
        finally:
            link.unsubscribe(subscriber)
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)

    return app


def _session_snapshot(session) -> "typing.Dict[str, object]":
    return typing.cast("typing.Dict[str, object]", session.snapshot())


def _session_summary(session) -> "typing.Dict[str, object]":
    summary = getattr(session, "summary", None)
    if callable(summary):
        return typing.cast("typing.Dict[str, object]", summary())
    snapshot = _session_snapshot(session)
    return {
        "title": snapshot.get("title") or "",
        "running": bool(snapshot.get("running")),
        "spinner": snapshot.get("spinner") or "",
        "turn_count": len(snapshot.get("turns") or []),
        "last_assistant": _last_assistant_text(snapshot.get("turns") or []),
        "context_remaining_percent": snapshot.get("context_remaining_percent"),
    }


def _last_assistant_text(turns: "typing.Iterable[typing.Dict[str, object]]") -> str:
    for turn in reversed(list(turns)):
        if str(turn.get("kind") or "assistant") == "control":
            continue
        response = str(turn.get("response") or "").strip()
        if response:
            return response
    return ""


def _public_turn(turn: "typing.Dict[str, object]") -> "typing.Dict[str, object]":
    return typing.cast(
        "typing.Dict[str, object]",
        _json_safe(
            {
                "submission_id": turn.get("submission_id", ""),
                "turn_id": turn.get("turn_id", ""),
                "prompt": turn.get("prompt", ""),
                "response": turn.get("response", ""),
                "thinking": turn.get("thinking", ""),
                "status": turn.get("status", ""),
                "error": turn.get("error", ""),
                "queue": turn.get("queue", ""),
                "sender": turn.get("sender", ""),
                "kind": turn.get("kind", "assistant"),
            }
        ),
    )


def _json_safe(value: object) -> "JSONValue":
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if is_dataclass(value):
        return _json_safe(asdict(value))
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return typing.cast(JSONValue, value)


def _html_response(content: str) -> HTMLResponse:
    return HTMLResponse(
        content,
        headers={
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
        },
    )


async def _send_ws_events(websocket: WebSocket, subscriber: "asyncio.Queue") -> None:
    while True:
        event = await subscriber.get()
        if event is None:
            return
        await websocket.send_json(event)


def run_serve_cli(args: "argparse.Namespace") -> int:
    import uvicorn

    board_arg = args.board if args.board is not None else _default_board_path()
    host, port, board_path = parse_target(args.listen, board_arg)
    if board_path is not None and not board_path.parent.is_dir():
        raise ValueError("board parent directory does not exist: {0}".format(board_path.parent))

    configure_loguru()
    def build_session() -> "WorkspaceInteractiveSession":
        model = build_model(
            config_path=args.config,
            profile=args.profile,
            timeout_seconds=args.timeout_seconds,
            vllm_endpoint=args.vllm_endpoint,
            use_chat_completion=args.use_chat_completion or None,
            use_messages=args.use_messages,
        )
        agent = build_agent(
            model,
            config_path=args.config,
            profile=args.profile,
            system_prompt=args.system_prompt,
            session_mode="tui",
            extra_contextual_user_messages=(
                [_board_context_text(board_path)] if board_path is not None else []
            ),
        )
        return WorkspaceInteractiveSession(
            build_cli_queue(agent),
            config_path=args.config,
        )

    def session_factory() -> "ThreadedWorkspaceInteractiveSession":
        return ThreadedWorkspaceInteractiveSession(build_session, asyncio.get_running_loop())

    app = create_app(WorkspaceSessionManager(session_factory, board_path), board_path)
    print(
        "pycodex workspace listening on http://{0}:{1}".format(host, port),
        flush=True,
    )
    if board_path is not None:
        print("board: {0}".format(board_path), flush=True)
    uvicorn.run(app, host=host, port=port, loop="asyncio")
    return 0


def _board_context_text(board_path: Path) -> str:
    return (
        "Current workspace board file: {0}. "
        "Changes you make to this file are shown to the user in real time. "
        "You can create or modify this file anytime."
    ).format(_format_board_path_for_prompt(board_path))


def _default_board_path() -> Path:
    return Path(tempfile.gettempdir()) / "pcws-{0}.html".format(uuid4().hex[:8])


def _format_board_path_for_prompt(board_path: Path) -> str:
    resolved = board_path.resolve()
    try:
        relative = os.path.relpath(str(resolved), str(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)
    if relative == ".":
        return "."
    if relative.startswith(".."):
        return str(resolved)
    return "./{0}".format(relative)


def _render_workspace_shell(board_path: "typing.Union[Path, None]") -> str:
    board_label = str(board_path) if board_path is not None else "No board"
    template = (Path(__file__).with_name("workspace.html")).read_text(
        encoding="utf-8"
    )
    return template.replace("__BOARD_LABEL__", html.escape(board_label))


def _render_empty_board() -> str:
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>No board</title></head>
<body style="font:14px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:24px">
<h1>No board configured</h1>
<p>Start with <code>pycodex-ws --listen 0.0.0.0:6007 --board ./board.html</code>.</p>
</body></html>"""


def _render_missing_board(board_path: Path) -> str:
    escaped_path = html.escape(str(board_path))
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>Board pending</title></head>
<body style="font:14px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:24px">
<h1>Board pending</h1>
<p>The board file does not exist yet.</p>
<p><code>{0}</code></p>
</body></html>""".format(escaped_path)


def main(argv: "typing.Union[typing.Sequence[str], None]" = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_serve_cli(args)
    except ValueError as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        return 130
    return 0
