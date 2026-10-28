import argparse
import asyncio
import gzip
import hashlib
import html
import json
import mimetypes
import os
import secrets
import socket
import subprocess
import sys
import threading
from collections import deque
from dataclasses import asdict, fields, is_dataclass

try:
    from contextlib import asynccontextmanager
except ImportError:  # pragma: no cover - Python 3.6 compatibility
    asynccontextmanager = None
import typing
from pathlib import Path
from urllib.parse import urlencode

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from pycodex.bootstrap import build_agent, build_model, configure_loguru
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
from pycodex.model import DEFAULT_CODEX_CONFIG_PATH
from pycodex.utils import uuid7_string
from pycodex.utils.compat import importlib_metadata
from pycodex.utils.event_helpers import (
    completed_history,
    shorten_title,
)
from pycodex.utils.session_persist import load_display_history
from pycodex_ws import build_runtime
from pycodex_router.protocol import route_path, trusted_agent_request

from .customization import (
    frontend_asset_response,
    frontend_directory,
    load_workspace_template,
)
from .runtime_info import runtime_info_prompt
from .session_api import history_page, session_api_routes
from .sharing import WorkspaceSharing
from .workspace_views import WorkspaceViewOpener
from .workspaces import (
    WorkspaceDefinition,
    WorkspaceEntry,
    WorkspaceRegistry,
    WorkspaceSessionManager,
    load_workspace_definitions,
    session_snapshot,
    workspace_file_path,
)

JSONValue = typing.Union[
    None,
    bool,
    int,
    float,
    str,
    typing.List["JSONValue"],
    typing.Dict[str, "JSONValue"],
]


def build_parser() -> "argparse.ArgumentParser":
    parser = argparse.ArgumentParser(
        prog="pycodex-ws",
        description="Run a local pycodex workspace with a board and chat session.",
        epilog="Stdin EOF gracefully shuts down the server after accepted work finishes.",
    )
    parser.add_argument(
        "--listen",
        default="127.0.0.1:6007",
        help="Bind address as host:port, for example 0.0.0.0:6007.",
    )
    parser.add_argument(
        "--listen-to",
        help=(
            "Connect outward with the router registration key: "
            "KEY@host[:port] (default port 6000) or ws(s)://KEY@host[:port]."
        ),
    )
    parser.add_argument(
        "--workspace-config",
        default="./workspaces.json",
        help=(
            "Optional JSON file listing workspaces. Each entry needs `state_path` "
            "and `work_dir`; `id` is optional and `enable_board` defaults to true. "
            "Routes are served under /w/<id>/."
        ),
    )
    parser.add_argument(
        "--password",
        default=None,
        help="Optional password required to open the workspace server.",
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
        "--toolset",
        nargs="*",
        default=None,
        help="Builtin tool names for all sessions. An empty list disables all tools.",
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


def parse_listen(target: str) -> "typing.Tuple[str, int]":
    target_text = str(target or "").strip() or "127.0.0.1:6007"
    if ":" not in target_text:
        raise ValueError("workspace listen target must look like host:port")
    host, port_text = target_text.rsplit(":", 1)
    host = host.strip() or "127.0.0.1"
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError("workspace port must be an integer") from exc
    return host, port


SessionFactory = typing.Callable[[], object]
ThreadedSessionFactory = typing.Callable[[], "WorkspaceInteractiveSession"]
SPINNER_STATUS_PREVIEW_LIMIT = 180
AUTH_COOKIE_NAME = "pycodex_ws_auth"


class WebSessionView:
    def __init__(self) -> None:
        self._subscribers: "typing.Set[asyncio.Queue]" = set()
        self._events: "typing.List[typing.Dict[str, object]]" = []
        self._stream_id = uuid7_string()
        self._sequence = 0
        self._runtime_state = None
        self._tool_states: "typing.Dict[str, object]" = {}
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
        self._server_loop: "typing.Union[asyncio.AbstractEventLoop, None]" = None
        self._lock = threading.RLock()

    def attach_server_loop(self, loop: "asyncio.AbstractEventLoop") -> None:
        self._server_loop = loop

    def handle_event(self, event: "Event", runtime_state=None) -> None:
        with self._lock:
            if runtime_state is not None:
                self._runtime_state = _json_safe(runtime_state)
            self._apply_runtime_event(event)
            payload = _event_data(event)
            payload.update({"type": "event", "snapshot": self.snapshot()})
            if isinstance(event, ToolCompletedEvent):
                payload["summary"] = event.visualize()
            self._publish_nowait(payload)

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
                self._publish_nowait({"type": "state"})

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

    def set_session_title(self, title: str) -> None:
        with self._lock:
            self._set_title(title)
            event = {
                "type": "title_changed",
                "title": self._title,
                "snapshot": self.snapshot(),
            }
        self._publish_nowait(event)

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
        self._publish_nowait(event)

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
                "tool_states": _json_safe(self._tool_states),
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
            self._tool_states = state.get("tool_states", {})
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
            elif event.reason == "title":
                self.set_session_title(state["title"])
            else:
                self._set_title(state["title"])
            if event.reason in {"attach", "history"} and state["active_turn"] is None:
                self._set_spinner_status(
                    IDLE_SLEEPING_STATUS if state.get("background_work_count", 0) else ""
                )
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
            self._set_idle_spinner_status(event)

    def _set_spinner_status(self, text: "typing.Union[str, None]") -> None:
        self._spinner_status = str(text or "").strip()

    def _set_idle_spinner_status(self, event: "TerminalEvent") -> None:
        if (event.background_work_count or 0) > 0:
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

    def _publish_nowait(self, event: "typing.Dict[str, object]") -> None:
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
        loop = self._server_loop
        if loop is None:
            for queue in queues:
                queue.put_nowait(event)
            return

        def publish() -> None:
            for queue in queues:
                queue.put_nowait(event)

        loop.call_soon_threadsafe(publish)


class WorkspaceInteractiveSession:
    def __init__(
        self,
        runtime,
        config_path: "typing.Union[str, None]" = None,
    ) -> None:
        self.runtime = runtime
        self.config_path = config_path
        self.view = WebSessionView()
        self._frontend_id = None
        self._child_views = {}
        self._submissions = {}
        self._finished_submissions = deque()

    async def start(self) -> "WorkspaceInteractiveSession":
        await self.runtime.start(self.config_path)
        if self._frontend_id is None:
            self._frontend_id = self.runtime.attach(self._handle_event)
        return self

    def _handle_event(self, event):
        self._handle_runtime_event(event, self.runtime, self.view)

    def _handle_runtime_event(self, event, runtime, view):
        view.handle_event(event, runtime.state_snapshot())
        if (
            isinstance(event, (CommandStartedEvent, CommandCompletedEvent, CommandFailedEvent))
            and event.submission_id not in self._submissions
        ):
            self._submissions[event.submission_id] = {
                "submission_id": event.submission_id,
                "agent_id": runtime.agent.session_id,
                "kind": "command", "command": event.command.lstrip("/"),
                "status": "queued", "result": None, "error": None,
            }
        if (
            isinstance(event, (TurnStartedEvent, CommandStartedEvent))
            and event.submission_id in self._submissions
        ):
            record = self._submissions[event.submission_id]
            record["status"] = "running"
            self._publish_submission(record, view)
        elif isinstance(event, CommandCompletedEvent):
            self._finish_submission(
                self._submissions[event.submission_id], view, result=event.result,
            )
        elif isinstance(event, CommandFailedEvent):
            self._finish_submission(self._submissions[event.submission_id], view, error={
                "type": event.error_type, "message": event.error,
            })

    async def close(self) -> None:
        try:
            await self.runtime.close()
        finally:
            self.detach()

    def detach(self):
        if self._frontend_id is not None:
            self.runtime.detach(self._frontend_id)
            self._frontend_id = None
        for runtime, frontend_id, view in self._child_views.values():
            runtime.detach(frontend_id)
            view.close()
        self._child_views.clear()
        self.view.close()

    async def submit(
        self, prompt: str, sender: str = "web"
    ) -> "typing.Dict[str, object]":
        try:
            receipt = await self.runtime.submit_input(prompt, sender)
        except (ValueError, RuntimeError) as exc:
            return {"ok": False, "error": str(exc), "snapshot": self.snapshot()}
        self._track_submission(receipt)
        return {
            "ok": True,
            "type": "submitted",
            "submission_id": receipt.submission_id,
            "snapshot": self.snapshot(),
        }

    def _publish_submission(self, record, view):
        view._publish_nowait({"type": "submission", "submission": _json_safe(record)})

    def _finish_submission(self, record, view, result=None, error=None):
        record.update(
            status="failed" if error is not None else "completed",
            result=_json_safe(result), error=error,
        )
        self._finished_submissions.append(record["submission_id"])
        while len(self._finished_submissions) > 500:
            del self._submissions[self._finished_submissions.popleft()]
        self._publish_submission(record, view)

    def _track_submission(self, receipt, runtime=None, view=None, command=None):
        if receipt.submission_id in self._submissions:
            return
        runtime = self.runtime if runtime is None else runtime
        view = self.view if view is None else view
        record = {
            "submission_id": receipt.submission_id,
            "agent_id": runtime.agent.session_id,
            "kind": receipt.kind, "status": "queued", "result": None, "error": None,
        }
        if command is not None:
            record["command"] = command
        self._submissions[receipt.submission_id] = record
        self._publish_submission(record, view)
        if receipt.kind == "command":
            return

        def completed(future):
            try:
                result = future.result()
                if receipt.kind == "turn":
                    result = {
                        "turn_id": result.turn_id, "output_text": result.output_text,
                        "iterations": result.iterations,
                    }
            except Exception as error:
                self._finish_submission(record, view, error={
                    "type": type(error).__name__, "message": str(error),
                })
            else:
                self._finish_submission(record, view, result=result)

        receipt.future.add_done_callback(completed)

    def _runtime_for_agent(self, agent_id):
        if agent_id is None or agent_id == self.runtime.agent.session_id:
            return self.runtime
        manager = self.runtime.agent.tool_registry.runtime_environment.subagent_manager
        return manager.get_runtime(agent_id)

    async def event_view(self, agent_id=None):
        runtime = self._runtime_for_agent(agent_id)
        if runtime is self.runtime:
            view = self.view
        elif agent_id in self._child_views:
            view = self._child_views[agent_id][2]
        else:
            view = WebSessionView()
            view.attach_server_loop(self.view._server_loop)
            frontend_id = runtime.attach(
                lambda event: self._handle_runtime_event(event, runtime, view),
                interactive=False,
            )
            self._child_views[agent_id] = (runtime, frontend_id, view)
        view.synchronize_state(runtime.state_snapshot())
        return view

    async def api_query(self, resource, options):
        if resource == "submission":
            return {"submission": _json_safe(self._submissions[options["submission_id"]])}
        agent_id = options.get("agent_id")
        runtime = self._runtime_for_agent(agent_id)
        if resource == "state":
            view = await self.event_view(agent_id)
            return view.api_snapshot()
        if resource == "models":
            return {
                "model": runtime.state_snapshot()["model"],
                "models": await runtime.list_models(),
            }
        if resource == "history":
            agent = runtime.agent
            return await run_in_threadpool(
                history_page, agent.session_id, agent.recorded_session_file_path,
                agent.history, options.get("before"), options["limit"],
            )
        if resource == "agents":
            manager = runtime.agent.tool_registry.runtime_environment.subagent_manager
            return {"agents": manager.list_agents(agent_id or runtime.agent.session_id)}
        if resource == "background-work":
            return {"background_work": runtime.background_work_snapshot()}
        raise ValueError("unknown session resource: " + resource)

    async def submit_command(self, command, argument="", agent_id=None):
        runtime = self._runtime_for_agent(agent_id)
        if command not in runtime.commands() or command == "queue":
            raise ValueError("expected an available session command other than queue")
        if not runtime.accepts_input:
            raise RuntimeError("agent is shutting down")
        view = await self.event_view(agent_id)
        receipt = await runtime.submit_input(
            "/" + command + (" " + argument if argument else ""),
            sender="web", wait_for_command=False,
        )
        self._track_submission(receipt, runtime, view, command)
        return {
            "submission": _json_safe(self._submissions[receipt.submission_id]),
            "submission_id": receipt.submission_id,
        }

    async def answer_input(self, request_id, answer):
        try:
            self.runtime.answer_input(request_id, answer)
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "snapshot": self.snapshot()}
        return {"ok": True, "type": "answered", "snapshot": self.snapshot()}

    def subscribe(self) -> "asyncio.Queue":
        return self.view.subscribe()

    def unsubscribe(self, queue: "asyncio.Queue") -> None:
        self.view.unsubscribe(queue)

    def snapshot(self) -> "typing.Dict[str, object]":
        return self.view.snapshot()

    def summary(self) -> "typing.Dict[str, object]":
        return self.view.summary()

    def rollout_path(self) -> str:
        return self.view.snapshot()["rollout_path"]

    async def restore_from_rollout(
        self,
        rollout_path: str,
        title: str = "",
        fork: bool = False,
        cwd: "typing.Optional[str]" = None,
    ) -> None:
        if cwd is not None:
            self.runtime.set_cwd(cwd)
        if rollout_path:
            self.runtime.resume(rollout_path, title)
            if fork:
                self.runtime.fork()
        else:
            self.runtime.set_title(title)


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
            raise RuntimeError(
                "workspace session thread failed to start"
            ) from self._startup_error
        return self

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._worker_loop = loop
        asyncio.set_event_loop(loop)
        try:
            session = self._session_factory()
            session.view = self._view
            self._session = session
            try:
                loop.run_until_complete(session.start())
            except BaseException:
                loop.run_until_complete(session.close())
                raise
            self._ready.set()
            loop.run_forever()
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            asyncio.set_event_loop(None)
            loop.close()

    async def close(self) -> None:
        session = self._session
        loop = self._worker_loop
        try:
            if session is not None and loop is not None and loop.is_running():
                future = asyncio.wrap_future(
                    asyncio.run_coroutine_threadsafe(session.close(), loop)
                )
                try:
                    await asyncio.shield(future)
                except asyncio.CancelledError:
                    await future
                    raise
        finally:
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
            thread = self._thread
            if thread is not None:
                await asyncio.to_thread(thread.join)
            self._thread = None

    async def submit(
        self, prompt: str, sender: str = "web"
    ) -> "typing.Dict[str, object]":
        future = asyncio.run_coroutine_threadsafe(
            self._session.submit(prompt, sender),
            self._worker_loop,
        )
        return await asyncio.wrap_future(future)

    async def answer_input(self, request_id, answer):
        future = asyncio.run_coroutine_threadsafe(
            self._session.answer_input(request_id, answer),
            self._worker_loop,
        )
        return await asyncio.wrap_future(future)

    async def api_query(self, resource, options):
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(
            self._session.api_query(resource, options), self._worker_loop,
        ))

    async def submit_command(self, command, argument="", agent_id=None):
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(
            self._session.submit_command(command, argument, agent_id), self._worker_loop,
        ))

    async def event_view(self, agent_id=None):
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(
            self._session.event_view(agent_id), self._worker_loop,
        ))

    def subscribe(self) -> "asyncio.Queue":
        return self._view.subscribe()

    def unsubscribe(self, queue: "asyncio.Queue") -> None:
        self._view.unsubscribe(queue)

    def snapshot(self) -> "typing.Dict[str, object]":
        return self._view.snapshot()

    def summary(self) -> "typing.Dict[str, object]":
        return self._view.summary()

    def rollout_path(self) -> str:
        if self._session is None:
            return ""
        return self._session.rollout_path()

    async def restore_from_rollout(
        self,
        rollout_path: str,
        title: str = "",
        fork: bool = False,
        cwd: "typing.Optional[str]" = None,
    ) -> None:
        session = self._session
        loop = self._worker_loop
        if session is None or loop is None:
            return

        future = asyncio.run_coroutine_threadsafe(
            session.restore_from_rollout(rollout_path, title=title, fork=fork, cwd=cwd),
            loop,
        )
        await asyncio.wrap_future(future)


def create_app(
    session_source: "typing.Union[WorkspaceSessionManager, SessionFactory]",
    state_path: "typing.Union[Path, None]",
    password: "typing.Union[str, None]" = None,
    work_dir: "typing.Union[Path, None]" = None,
    enable_board: bool = True,
) -> Starlette:
    manager = (
        session_source
        if isinstance(session_source, WorkspaceSessionManager)
        else WorkspaceSessionManager(session_source, state_path)
    )
    board_path = workspace_file_path(state_path, ".html") if enable_board else None
    app = _create_lifespan_app(manager.start, manager.close)
    sharing = WorkspaceSharing(password, multi=False)
    _install_auth(app, password, sharing=sharing)
    _install_workspace_routes(app, manager, board_path, Path(work_dir or Path.cwd()).resolve())
    _install_sharing_routes(app, sharing)
    return app


def create_multi_workspace_app(
    registry: "WorkspaceRegistry",
    password: "typing.Union[str, None]" = None,
) -> Starlette:
    app = _create_lifespan_app(registry.start, registry.close)
    sharing = WorkspaceSharing(password, multi=True)
    _install_auth(app, password, sharing=sharing)
    view_asset_cache = {}

    async def index(request: Request) -> Response:
        return _html_response(_render_workspaces_manager_shell())

    async def favicon(request: Request) -> Response:
        return Response(status_code=204)

    async def workspaces(request: Request) -> JSONResponse:
        return JSONResponse({"workspaces": registry.list_workspaces()})

    async def add_workspace(request: Request) -> JSONResponse:
        payload = await _json_object_body(request)
        try:
            if "board" in payload:
                raise ValueError("use `state_path` and `enable_board` instead of `board`")
            entry = await registry.add_workspace(
                str(payload.get("name") or ""),
                work_dir=str(payload.get("dir") or "./"),
                state_path=payload.get("state_path"),
                enable_board=payload.get("enable_board", True),
            )
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return JSONResponse(
            {
                "ok": True,
                "workspace": entry.to_dict(),
                "workspaces": registry.list_workspaces(),
            }
        )

    async def delete_workspace(request: Request) -> JSONResponse:
        try:
            sharing.revoke(request.path_params["workspace_id"])
            await registry.delete_workspace(request.path_params["workspace_id"])
        except KeyError:
            raise HTTPException(status_code=404, detail="workspace not found")
        return JSONResponse({"ok": True, "workspaces": registry.list_workspaces()})

    async def workspace_index_redirect(request: Request) -> RedirectResponse:
        workspace_id = request.path_params["workspace_id"]
        _workspace_entry_or_404(registry, workspace_id)
        return RedirectResponse(
            url=request.scope.get("root_path", "") + "/w/{0}/".format(workspace_id),
            status_code=307,
        )

    async def workspace_index(request: Request) -> HTMLResponse:
        entry = _workspace_entry_or_404(registry, request.path_params["workspace_id"])
        return _html_response(
            _render_workspace_shell(
                entry.definition.board_path,
                title=entry.definition.workspace_id,
                work_dir=entry.definition.work_dir,
                builtin=request.query_params.get("builtin") == "1",
                access_role=request.scope["pycodex.access_role"],
            )
        )

    async def workspace_board(request: Request) -> Response:
        entry = _workspace_entry_or_404(registry, request.path_params["workspace_id"])
        return _board_response(entry.definition.board_path)

    async def workspace_board_status(request: Request) -> JSONResponse:
        entry = _workspace_entry_or_404(registry, request.path_params["workspace_id"])
        return _board_status_response(entry.definition.board_path)

    def workspace_views(request: Request) -> JSONResponse:
        entry = _workspace_entry_or_404(registry, request.path_params["workspace_id"])
        return _views_response(entry.manager)

    async def workspace_open_view(request: Request) -> JSONResponse:
        payload = await _json_object_body(request)
        entry = _workspace_entry_or_404(registry, request.path_params["workspace_id"])
        return await run_in_threadpool(
            _open_view_response, entry.manager, entry.definition.work_dir, payload
        )

    def workspace_view(request: Request) -> JSONResponse:
        entry = _workspace_entry_or_404(registry, request.path_params["workspace_id"])
        return _view_response(entry.manager, request.path_params["view_id"])

    def workspace_close_view(request: Request) -> JSONResponse:
        entry = _workspace_entry_or_404(registry, request.path_params["workspace_id"])
        return _close_view_response(entry.manager, request.path_params["view_id"])

    def workspace_view_asset(request: Request) -> Response:
        _workspace_entry_or_404(registry, request.path_params["workspace_id"])
        return _view_asset_response(
            request.url.path.rsplit("/", 1)[-1], request, view_asset_cache
        )

    async def workspace_websocket_backend_hint(request: Request) -> JSONResponse:
        _workspace_entry_or_404(registry, request.path_params["workspace_id"])
        return _websocket_backend_hint_response()

    async def workspace_sessions(request: Request) -> JSONResponse:
        entry = _workspace_entry_or_404(registry, request.path_params["workspace_id"])
        return _sessions_response(entry.manager)

    async def workspace_new_session(request: Request) -> JSONResponse:
        entry = _workspace_entry_or_404(registry, request.path_params["workspace_id"])
        return await _new_session_response(entry.manager, request)

    def workspace_frontend_asset(request: Request) -> Response:
        _workspace_entry_or_404(registry, request.path_params["workspace_id"])
        return frontend_asset_response(request.path_params["asset_path"])

    async def workspace_delete_session(request: Request) -> JSONResponse:
        entry = _workspace_entry_or_404(registry, request.path_params["workspace_id"])
        return await _delete_session_response(
            entry.manager, request.path_params["session_id"]
        )

    async def workspace_read_session(request: Request) -> JSONResponse:
        payload = await _json_object_body(request)
        entry = _workspace_entry_or_404(registry, request.path_params["workspace_id"])
        return _read_session_response(
            entry.manager, request.path_params["session_id"], payload
        )

    async def workspace_session(request: Request) -> JSONResponse:
        entry = _workspace_entry_or_404(registry, request.path_params["workspace_id"])
        return _session_response(entry.manager, request, entry.definition.workspace_id)

    async def workspace_message(request: Request) -> JSONResponse:
        payload = await _json_object_body(request)
        entry = _workspace_entry_or_404(registry, request.path_params["workspace_id"])
        return await _message_response(entry.manager, payload)

    async def workspace_websocket_session(websocket: WebSocket) -> None:
        try:
            entry = registry.get(websocket.path_params["workspace_id"])
        except (KeyError, ValueError):
            await websocket.close(code=1008)
            return
        await sharing.serve_websocket(
            websocket, lambda connection: _websocket_session_handler(entry.manager, connection),
        )

    async def workspace_board_asset(request: Request) -> Response:
        entry = _workspace_entry_or_404(registry, request.path_params["workspace_id"])
        return _board_asset_response(
            entry.definition.board_path, request.path_params["asset_path"]
        )

    def api_manager(request):
        return _workspace_entry_or_404(
            registry, request.path_params["workspace_id"]
        ).manager

    app.routes.extend(session_api_routes(
        api_manager, _json_object_body, "/w/{workspace_id}",
    ))
    app.routes.extend([
        Route("/", index, methods=["GET"]),
        Route("/favicon.ico", favicon, methods=["GET"]),
        Route("/api/workspaces", workspaces, methods=["GET"]),
        Route("/api/workspaces", add_workspace, methods=["POST"]),
        Route("/api/workspaces/{workspace_id}", delete_workspace, methods=["DELETE"]),
        Route("/w/{workspace_id}", workspace_index_redirect, methods=["GET", "HEAD"]),
        Route("/w/{workspace_id}/", workspace_index, methods=["GET", "HEAD"]),
        Route("/w/{workspace_id}/board", workspace_board, methods=["GET", "HEAD"]),
        Route("/w/{workspace_id}/api/board", workspace_board_status, methods=["GET"]),
        Route("/w/{workspace_id}/api/views", workspace_views, methods=["GET"]),
        Route("/w/{workspace_id}/api/views/open", workspace_open_view, methods=["POST"]),
        Route("/w/{workspace_id}/api/views/{view_id}", workspace_view, methods=["GET"]),
        Route(
            "/w/{workspace_id}/api/views/{view_id}",
            workspace_close_view, methods=["DELETE"],
        ),
        Route(
            "/w/{workspace_id}/assets/content_views.mjs",
            workspace_view_asset, methods=["GET", "HEAD"],
        ),
        Route(
            "/w/{workspace_id}/assets/content_views.css",
            workspace_view_asset, methods=["GET", "HEAD"],
        ),
        Route(
            "/w/{workspace_id}/assets/content_views.LICENSE.txt",
            workspace_view_asset, methods=["GET", "HEAD"],
        ),
        Route(
            "/w/{workspace_id}/assets/editor.mjs",
            workspace_view_asset, methods=["GET", "HEAD"],
        ),
        Route(
            "/w/{workspace_id}/assets/editor.css",
            workspace_view_asset, methods=["GET", "HEAD"],
        ),
        Route(
            "/w/{workspace_id}/ws/session",
            workspace_websocket_backend_hint, methods=["GET"],
        ),
        Route("/w/{workspace_id}/api/sessions", workspace_sessions, methods=["GET"]),
        Route("/w/{workspace_id}/api/sessions", workspace_new_session, methods=["POST"]),
        Route(
            "/w/{workspace_id}/frontend/{asset_path:path}",
            workspace_frontend_asset, methods=["GET", "HEAD"],
        ),
        Route(
            "/w/{workspace_id}/api/sessions/{session_id}",
            workspace_delete_session, methods=["DELETE"],
        ),
        Route(
            "/w/{workspace_id}/api/sessions/{session_id}/read",
            workspace_read_session, methods=["POST"],
        ),
        Route("/w/{workspace_id}/api/session", workspace_session, methods=["GET"]),
        Route(
            "/w/{workspace_id}/api/session/message",
            workspace_message, methods=["POST"],
        ),
        WebSocketRoute("/w/{workspace_id}/ws/session", workspace_websocket_session),
        Route(
            "/w/{workspace_id}/{asset_path:path}",
            workspace_board_asset, methods=["GET", "HEAD"],
        ),
    ])
    _install_sharing_routes(app, sharing, registry)
    return app


def _install_auth(
    app: Starlette, password: "typing.Union[str, None]",
    cookie_name: str = AUTH_COOKIE_NAME,
    login_title: str = "pycodex workspace",
    blocked_ips: "typing.Union[typing.Set[str], None]" = None,
    sharing=None,
) -> str:
    password_text = str(password or "")
    if not password_text and sharing is None:
        return ""
    token = secrets.token_urlsafe(32) if password_text else ""
    if sharing is not None:
        sharing.admin_token = token
        app.state.workspace_sharing = sharing
    login_failures = {}

    def ip_blocked_response() -> JSONResponse:
        return JSONResponse(
            {
                "ok": False,
                "error": (
                    "Too many incorrect passwords. This IP is blocked "
                    "until the router restarts."
                ),
            },
            status_code=403,
        )

    async def auth_middleware(request: Request, call_next):
        if blocked_ips is not None and request.client.host in blocked_ips:
            return ip_blocked_response()
        path = route_path(request.scope)
        root = request.scope.get("root_path", "")
        if sharing is not None:
            target = sharing.workspace(path)
            cookie = request.cookies.get(cookie_name)
            if path in {"/favicon.ico", "/login"}:
                return await call_next(request)
            if trusted_agent_request(request.scope) or sharing.is_admin(cookie):
                request.scope["pycodex.access_role"] = "admin"
                return await call_next(request)
            grant = None if target is None else sharing.grant(target[0], cookie)
            if grant is not None:
                if target[1].rstrip("/") == "/api/share":
                    return JSONResponse({"error": "administrator access required"}, 403)
                request.scope["pycodex.access_role"] = "visitor"
                return await call_next(request)
            if "/api/" in path or request.method not in {"GET", "HEAD"}:
                return JSONResponse({"error": "authentication required"}, 401)
            login_url = root + "/login"
            destination = root + path + ("?" + request.url.query if request.url.query else "")
            return RedirectResponse(login_url + "?" + urlencode({"next": destination}), 303)
        if path in {"/favicon.ico", "/login"} or trusted_agent_request(request.scope) or _auth_cookie_matches(
            token,
            request.cookies.get(cookie_name),
        ):
            return await call_next(request)
        if path.startswith("/api/") or "/api/" in path:
            return JSONResponse(
                {"ok": False, "error": "authentication required"},
                status_code=401,
            )
        target = root + path + ("?" + request.url.query if request.url.query else "")
        return RedirectResponse(
            url=root + "/login?" + urlencode({"next": target}), status_code=303
        )

    async def login_page(request: Request) -> HTMLResponse:
        address = str(request.url.replace(
            path=request.scope.get("root_path", "") + "/", query="",
        ))
        return _html_response(_render_login_shell(
            login_title, socket.gethostname(), address,
        ))

    async def login(request: Request) -> JSONResponse:
        payload = await _json_object_body(request, max_bytes=4096)
        client_ip = request.client.host if blocked_ips is not None else None
        # Another login can block this IP while the request body is arriving.
        if blocked_ips is not None and client_ip in blocked_ips:
            return ip_blocked_response()
        supplied = str(payload.get("password") or "")
        accepted_token = token if password_text and secrets.compare_digest(
            supplied.encode("utf-8"), password_text.encode("utf-8"),
        ) else None
        root = request.scope.get("root_path", "")
        destination = request.query_params.get("next", root + "/")
        target = None
        if sharing is not None:
            relative = destination[len(root):] if destination.startswith(root + "/") else ""
            target = sharing.workspace(relative)
            if accepted_token is None and target is not None:
                grant = await run_in_threadpool(sharing.authenticate, target[0], supplied)
                if grant is not None:
                    accepted_token = grant.token
                    destination = root + ("/w/{0}/".format(target[0]) if sharing.multi else "/")
        if accepted_token is None:
            if blocked_ips is not None:
                login_failures[client_ip] = login_failures.get(client_ip, 0) + 1
                if login_failures[client_ip] >= 3:
                    blocked_ips.add(client_ip)
                    del login_failures[client_ip]
                    return ip_blocked_response()
            return JSONResponse(
                {"ok": False, "error": "invalid password"},
                status_code=401,
            )
        login_failures.pop(client_ip, None)
        target = destination
        if (
            not target.startswith("/")
            or target.startswith("//")
            or "\\" in target
            or any(ord(char) < 32 for char in target)
        ):
            target = root + "/"
        response = JSONResponse({"ok": True, "redirect": target})
        response.set_cookie(
            cookie_name,
            accepted_token,
            path=root + "/",
            secure=request.url.scheme == "https",
            httponly=True,
            samesite="lax",
        )
        return response

    app.add_middleware(BaseHTTPMiddleware, dispatch=auth_middleware)
    app.routes.extend([
        Route("/login", login_page, methods=["GET"]),
        Route("/login", login, methods=["POST"]),
    ])
    return token


def _install_sharing_routes(app, sharing, registry=None):
    prefix = "/w/{workspace_id}" if sharing.multi else ""

    async def share_settings(request):
        workspace_id = request.path_params["workspace_id"] if sharing.multi else "workspace"
        if registry is not None:
            _workspace_entry_or_404(registry, workspace_id)
        if request.scope["pycodex.access_role"] != "admin":
            return JSONResponse({"error": "administrator access required"}, 403)
        if request.method == "PUT":
            payload = await _json_object_body(request, max_bytes=4096)
            password = payload.get("password")
            if not sharing.password:
                return JSONResponse({
                    "error": "Set a workspace server administrator password before enabling sharing.",
                }, 400)
            if not isinstance(password, str) or not password:
                return JSONResponse({"error": "share password must be a nonempty string"}, 422)
            if secrets.compare_digest(password.encode("utf-8"), sharing.password.encode("utf-8")):
                return JSONResponse({"error": "share password must differ from administrator password"}, 422)
            await sharing.set_password(workspace_id, password)
        elif request.method == "DELETE":
            sharing.revoke(workspace_id)
        return JSONResponse({
            "enabled": workspace_id in sharing.grants,
            "available": bool(sharing.password),
            "url": request.scope.get("root_path", "") + (
                "/w/{0}/".format(workspace_id) if sharing.multi else "/"
            ),
        }, headers={"Cache-Control": "no-store"})

    app.routes.insert(0, Route(
        prefix + "/api/share", share_settings, methods=["GET", "PUT", "DELETE"],
    ))


async def _json_object_body(
    request: Request, max_bytes: "typing.Union[int, None]" = None,
) -> "typing.Dict[str, object]":
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "application/json" and not (
        media_type.startswith("application/") and media_type.endswith("+json")
    ):
        raise HTTPException(status_code=422, detail="request body must be JSON")
    body = bytearray()
    async for chunk in request.stream():
        if max_bytes is not None and len(body) + len(chunk) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail="request body exceeds {0} bytes".format(max_bytes),
                headers={"Connection": "close"},
            )
        body.extend(chunk)
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=422, detail="invalid JSON body")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="request body must be a JSON object")
    return payload


def _auth_cookie_matches(token: str, cookie: "typing.Union[str, None]") -> bool:
    if not token:
        return True
    return bool(cookie) and secrets.compare_digest(str(cookie), token)


def _render_login_shell(title: str, hostname: str, address: str) -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Log in to __LOGIN_TITLE__</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    body {
      min-height: 100vh;
      margin: 0;
      display: grid;
      place-items: center;
      background: Canvas;
      color: CanvasText;
    }
    form {
      width: min(360px, calc(100vw - 32px));
      display: grid;
      gap: 12px;
    }
    h1 {
      margin: 0;
      font-size: 22px;
    }
    .login-target {
      margin: 0;
      font-size: 13px;
      line-height: 1.6;
      overflow-wrap: anywhere;
    }
    input, button {
      min-height: 38px;
      border-radius: 7px;
      border: 1px solid color-mix(in srgb, CanvasText 18%, Canvas 82%);
      padding: 8px 10px;
      font: inherit;
    }
    button {
      cursor: pointer;
    }
    .status {
      min-height: 20px;
      color: #b42318;
      font-size: 13px;
    }
  </style>
</head>
<body>
  <form id="loginForm">
    <h1>Log in to __LOGIN_TITLE__</h1>
    <p class="login-target">Server: <strong>__LOGIN_HOST__</strong><br><code>__LOGIN_ADDRESS__</code></p>
    <label for="passwordInput">Password</label>
    <input id="passwordInput" type="password" autocomplete="current-password" autofocus>
    <button type="submit">Log in</button>
    <div id="status" class="status" role="status"></div>
  </form>
  <script>
    const form = document.getElementById("loginForm");
    const passwordInput = document.getElementById("passwordInput");
    const statusEl = document.getElementById("status");
    form.addEventListener("submit", async function(event) {
      event.preventDefault();
      statusEl.textContent = "";
      const response = await fetch(window.location.pathname + window.location.search, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({password: passwordInput.value}),
      });
      if (response.ok) {
        const result = await response.json();
        window.location.href = result.redirect;
        return;
      }
      statusEl.textContent = response.status === 403
        ? (await response.json()).error
        : "Invalid password";
    });
  </script>
</body>
</html>""".replace("__LOGIN_TITLE__", html.escape(title)).replace(
        "__LOGIN_HOST__", html.escape(hostname),
    ).replace("__LOGIN_ADDRESS__", html.escape(address))


async def _http_exception_response(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        {"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers
    )


def _create_lifespan_app(
    start: "typing.Callable[[], typing.Awaitable[None]]",
    close: "typing.Callable[[], typing.Awaitable[None]]",
) -> Starlette:
    if asynccontextmanager is not None:

        @asynccontextmanager
        async def lifespan(_app):
            await start()
            try:
                yield
            finally:
                await close()

        return Starlette(
            lifespan=lifespan,
            exception_handlers={HTTPException: _http_exception_response},
        )

    return Starlette(
        on_startup=[start],
        on_shutdown=[close],
        exception_handlers={HTTPException: _http_exception_response},
    )


def _install_workspace_routes(
    app: Starlette,
    manager: WorkspaceSessionManager,
    board_path: "typing.Union[Path, None]",
    work_dir: Path,
) -> None:
    view_asset_cache = {}

    async def index(request: Request) -> HTMLResponse:
        return _html_response(_render_workspace_shell(
            board_path, work_dir=work_dir,
            builtin=request.query_params.get("builtin") == "1",
            access_role=request.scope["pycodex.access_role"],
        ))

    async def favicon(request: Request) -> Response:
        return Response(status_code=204)

    async def board(request: Request) -> Response:
        return _board_response(board_path)

    async def board_status(request: Request) -> JSONResponse:
        return _board_status_response(board_path)

    def views(request: Request) -> JSONResponse:
        return _views_response(manager)

    async def open_view(request: Request) -> JSONResponse:
        payload = await _json_object_body(request)
        return await run_in_threadpool(_open_view_response, manager, work_dir, payload)

    def view(request: Request) -> JSONResponse:
        return _view_response(manager, request.path_params["view_id"])

    def close_view(request: Request) -> JSONResponse:
        return _close_view_response(manager, request.path_params["view_id"])

    def view_asset(request: Request) -> Response:
        return _view_asset_response(
            request.url.path.rsplit("/", 1)[-1], request, view_asset_cache
        )

    async def websocket_backend_hint(request: Request) -> JSONResponse:
        return _websocket_backend_hint_response()

    async def sessions(request: Request) -> JSONResponse:
        return _sessions_response(manager)

    async def new_session(request: Request) -> JSONResponse:
        return await _new_session_response(manager, request)

    def frontend_asset(request: Request) -> Response:
        return frontend_asset_response(request.path_params["asset_path"])

    async def delete_session(request: Request) -> JSONResponse:
        return await _delete_session_response(manager, request.path_params["session_id"])

    async def read_session(request: Request) -> JSONResponse:
        payload = await _json_object_body(request)
        return _read_session_response(manager, request.path_params["session_id"], payload)

    async def session(request: Request) -> JSONResponse:
        return _session_response(manager, request)

    async def message(request: Request) -> JSONResponse:
        payload = await _json_object_body(request)
        return await _message_response(manager, payload)

    async def websocket_session(websocket: WebSocket) -> None:
        await app.state.workspace_sharing.serve_websocket(
            websocket, lambda connection: _websocket_session_handler(manager, connection),
        )

    async def board_asset(request: Request) -> Response:
        return _board_asset_response(board_path, request.path_params["asset_path"])

    app.routes.extend(session_api_routes(lambda request: manager, _json_object_body))
    app.routes.extend([
        Route("/", index, methods=["GET"]),
        Route("/favicon.ico", favicon, methods=["GET"]),
        Route("/board", board, methods=["GET", "HEAD"]),
        Route("/api/board", board_status, methods=["GET"]),
        Route("/api/views", views, methods=["GET"]),
        Route("/api/views/open", open_view, methods=["POST"]),
        Route("/api/views/{view_id}", view, methods=["GET"]),
        Route("/api/views/{view_id}", close_view, methods=["DELETE"]),
        Route("/assets/content_views.mjs", view_asset, methods=["GET", "HEAD"]),
        Route("/assets/content_views.css", view_asset, methods=["GET", "HEAD"]),
        Route("/assets/content_views.LICENSE.txt", view_asset, methods=["GET", "HEAD"]),
        Route("/assets/editor.mjs", view_asset, methods=["GET", "HEAD"]),
        Route("/assets/editor.css", view_asset, methods=["GET", "HEAD"]),
        Route("/ws/session", websocket_backend_hint, methods=["GET"]),
        Route("/api/sessions", sessions, methods=["GET"]),
        Route("/api/sessions", new_session, methods=["POST"]),
        Route("/frontend/{asset_path:path}", frontend_asset, methods=["GET", "HEAD"]),
        Route("/api/sessions/{session_id}", delete_session, methods=["DELETE"]),
        Route("/api/sessions/{session_id}/read", read_session, methods=["POST"]),
        Route("/api/session", session, methods=["GET"]),
        Route("/api/session/message", message, methods=["POST"]),
        WebSocketRoute("/ws/session", websocket_session),
        Route("/{asset_path:path}", board_asset, methods=["GET", "HEAD"]),
    ])


def _views_response(manager: WorkspaceSessionManager) -> JSONResponse:
    try:
        views = manager.views.list_views()
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error))
    return JSONResponse({"views": views}, headers={"Cache-Control": "no-store"})


def _open_view_response(
    manager: WorkspaceSessionManager, work_dir: Path, payload: "typing.Dict[str, object]"
) -> JSONResponse:
    try:
        session_id = payload.get("session_id")
        if session_id is not None:
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("session_id must be a nonempty string")
            work_dir = Path(manager.get(session_id).summary()["cwd"])
        view = WorkspaceViewOpener(manager.views, work_dir).open(
            payload.get("href"), payload.get("src_path")
        )
        views = manager.views.list_views()
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        raise HTTPException(status_code=400, detail=str(error))
    return JSONResponse(
        {"view": view, "views": views}, headers={"Cache-Control": "no-store"}
    )


def _view_response(manager: WorkspaceSessionManager, view_id: str) -> JSONResponse:
    try:
        view = manager.views.get(view_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="view not found")
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error))
    return JSONResponse(view, headers={"Cache-Control": "no-store"})


def _close_view_response(
    manager: WorkspaceSessionManager, view_id: str
) -> JSONResponse:
    try:
        manager.views.close(view_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="view not found")
    return JSONResponse({"ok": True})


def _view_asset_encoding(accept_encoding: str) -> str:
    qualities = {}
    for entry in accept_encoding.lower().split(","):
        name, *parameters = entry.strip().split(";")
        quality = 1.0
        for parameter in parameters:
            key, _, value = parameter.strip().partition("=")
            if key != "q":
                continue
            try:
                quality = float(value)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail="invalid encoding quality"
                ) from exc
            if not 0 <= quality <= 1:
                raise HTTPException(status_code=400, detail="invalid encoding quality")
        qualities[name.strip()] = quality
    gzip_quality = qualities.get("gzip", qualities.get("*", 0))
    if gzip_quality > 0 and gzip_quality >= qualities.get("identity", 0):
        return "gzip"
    if qualities.get("identity", 0 if qualities.get("*") == 0 else 1) > 0:
        return "identity"
    raise HTTPException(status_code=406, detail="no acceptable asset encoding")


def _view_asset_response(asset_name: str, request: Request, cache: dict) -> Response:
    media_types = {
        "content_views.mjs": "text/javascript",
        "content_views.css": "text/css",
        "content_views.LICENSE.txt": "text/plain",
        "editor.mjs": "text/javascript",
        "editor.css": "text/css",
    }
    if asset_name not in media_types:
        raise HTTPException(status_code=404, detail="asset not found")
    encoding = _view_asset_encoding(request.headers.get("accept-encoding", ""))
    path = Path(__file__).with_name(asset_name)
    stat = path.stat()
    stamp = (stat.st_mtime_ns, stat.st_size)
    key = (asset_name, encoding)
    cached = cache.get(key)
    if cached is None or cached[0] != stamp:
        body = path.read_bytes()
        if encoding == "gzip":
            body = gzip.compress(body, compresslevel=6)
        etag = '"' + hashlib.sha256(body).hexdigest() + '"'
        cached = cache[key] = (stamp, body, etag)
    _, body, etag = cached
    headers = {
        "Cache-Control": "no-cache",
        "X-Content-Type-Options": "nosniff",
        "Vary": "Accept-Encoding",
        "ETag": etag,
    }
    if encoding == "gzip":
        headers["Content-Encoding"] = "gzip"
    for candidate in request.headers.get("if-none-match", "").split(","):
        candidate = candidate.strip()
        if candidate.startswith("W/"):
            candidate = candidate[2:]
        if candidate in ("*", etag):
            return Response(status_code=304, headers=headers)
    headers["Content-Length"] = str(len(body))
    return Response(
        b"" if request.method == "HEAD" else body,
        media_type=media_types[asset_name], headers=headers,
    )


def _board_status_response(board_path: "typing.Union[Path, None]") -> JSONResponse:
    if board_path is None:
        raise HTTPException(status_code=404, detail="Canvas is disabled")
    if not board_path.is_file():
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


def _websocket_backend_hint_response() -> JSONResponse:
    return JSONResponse(
        {
            "error": "websocket backend is unavailable; HTTP polling is active",
        },
        status_code=426,
    )


def _sessions_response(manager: "WorkspaceSessionManager") -> JSONResponse:
    return JSONResponse({"sessions": manager.list_sessions()})


def _read_session_response(
    manager: "WorkspaceSessionManager", session_id: str, payload: "typing.Dict[str, object]"
) -> JSONResponse:
    try:
        manager.mark_read(session_id, payload.get("final_id"))
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _sessions_response(manager)


async def _new_session_response(
    manager: "WorkspaceSessionManager", request: Request,
) -> JSONResponse:
    payload = await _json_object_body(request) if await request.body() else {}
    title = payload.get("title", "")
    prompt = payload.get("prompt", "")
    if not isinstance(title, str) or not isinstance(prompt, str):
        raise HTTPException(status_code=422, detail="title and prompt must be strings")
    session_id = await manager.create_session(title=title)
    result = (
        await manager.get(session_id).submit(prompt, sender="web")
        if prompt else {"ok": True}
    )
    return JSONResponse(
        {
            **result,
            "session_id": session_id,
            "sessions": manager.list_sessions(),
            "snapshot": session_snapshot(manager.get(session_id)),
        },
        status_code=200 if result["ok"] else 400,
    )


async def _delete_session_response(
    manager: "WorkspaceSessionManager",
    session_id: str,
) -> JSONResponse:
    try:
        await manager.close_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    return JSONResponse({"ok": True, "sessions": manager.list_sessions()})


def _session_response(
    manager: "WorkspaceSessionManager",
    request: Request,
    workspace_id: "typing.Union[str, None]" = None,
) -> JSONResponse:
    try:
        resolved_id = manager.resolve_session_id(request.query_params.get("session_id"))
        link = manager.get(resolved_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    payload = {
        "session_id": resolved_id,
        "sessions": manager.list_sessions(),
        "snapshot": session_snapshot(link),
    }
    if request.query_params.get("runtime_info") == "true":
        state = request.app.state
        payload["runtime_info"] = runtime_info_prompt(
            resolved_id, link.summary(), workspace_id,
            getattr(state, "local_api_url", None),
            getattr(state, "router_client", None),
            getattr(state, "agent_socket_path", None),
        )
    return JSONResponse(payload, headers={"Cache-Control": "no-store"})


async def _message_response(
    manager: "WorkspaceSessionManager",
    payload: "typing.Dict[str, object]",
) -> JSONResponse:
    session_id = str(payload.get("session_id") or "")
    try:
        link = manager.get(session_id or None)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    if "request_id" in payload:
        result = await link.answer_input(payload["request_id"], payload.get("answer"))
    else:
        result = await link.submit(
            str(payload.get("prompt") or ""),
            sender=str(payload.get("sender") or "web"),
        )
    if isinstance(result, dict):
        result.setdefault("sessions", manager.list_sessions())
    status = 200 if result.get("ok") else 400
    return JSONResponse(result, status_code=status)


async def _websocket_session_handler(
    manager: "WorkspaceSessionManager",
    websocket: WebSocket,
) -> None:
    url = websocket.url
    scheme = "https" if url.scheme == "wss" else "http"
    authority = url.netloc.lower()
    if url.port == (443 if scheme == "https" else 80):
        authority = authority.rsplit(":", 1)[0]
    # Browsers send Origin; non-browser clients may omit it.
    if websocket.headers.get("origin") not in (None, scheme + "://" + authority):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    session_id = str(websocket.query_params.get("session_id") or "")
    try:
        link = manager.get(session_id or None)
    except KeyError:
        await websocket.close(code=1008)
        return
    agent_id = websocket.query_params.get("agent_id")
    try:
        view = await link.event_view(agent_id)
        subscriber = view.subscribe(websocket.query_params.get("cursor"))
    except (KeyError, ValueError) as error:
        await websocket.send_json({"type": "error", "error": str(error)})
        await websocket.close(code=1008)
        return
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
            if action in {"send", "answer"}:
                if agent_id is not None:
                    await websocket.send_json({
                        "type": "error", "error": "child-agent subscriptions are read-only",
                    })
                    continue
                target_session_id = str(payload.get("session_id") or session_id or "")
                try:
                    target_link = manager.get(target_session_id or None)
                except KeyError:
                    await websocket.send_json(
                        {"type": "error", "error": "session not found"}
                    )
                    continue
                if action == "answer":
                    result = await target_link.answer_input(
                        payload.get("request_id"),
                        payload.get("answer"),
                    )
                else:
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
        view.unsubscribe(subscriber)
        sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)


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


def _board_response(board_path: "typing.Union[Path, None]") -> Response:
    if board_path is None:
        raise HTTPException(status_code=404, detail="Canvas is disabled")
    if not board_path.is_file():
        return _html_response(_render_missing_board(board_path))
    return _html_response(board_path.read_text(encoding="utf-8", errors="replace"))


def _board_asset_response(
    board_path: "typing.Union[Path, None]",
    asset_path: str,
) -> Response:
    media_type, unused_encoding = mimetypes.guess_type(str(asset_path or ""))
    del unused_encoding
    if board_path is None or not media_type or not media_type.startswith("image/"):
        raise HTTPException(status_code=404, detail="board image not found")

    resolved_asset = None
    try:
        board_directory = board_path.parent.resolve()
        resolved_asset = (board_directory / asset_path).resolve()
        within_board_directory = os.path.commonpath(
            [str(board_directory), str(resolved_asset)]
        ) == str(board_directory)
    except (OSError, RuntimeError, ValueError):
        within_board_directory = False

    if (
        not within_board_directory
        or resolved_asset is None
        or not resolved_asset.is_file()
    ):
        raise HTTPException(status_code=404, detail="board image not found")

    return FileResponse(
        str(resolved_asset),
        media_type=media_type,
        headers={
            "Cache-Control": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _workspace_entry_or_404(
    registry: "WorkspaceRegistry",
    workspace_id: str,
) -> "WorkspaceEntry":
    try:
        return registry.get(workspace_id)
    except (KeyError, ValueError):
        raise HTTPException(status_code=404, detail="workspace not found")


def run_serve_cli(args: "argparse.Namespace") -> int:
    from .owned_process import run_owned_server

    host, port = parse_listen(args.listen)
    configure_loguru()

    definitions = load_workspace_definitions(args.workspace_config)
    entries = [_build_workspace_entry(definition, args) for definition in definitions]
    registry = WorkspaceRegistry(
        entries,
        config_path=args.workspace_config,
        entry_factory=lambda definition, persist_callback: _build_workspace_entry(
            definition,
            args,
            persist_callback,
        ),
    )
    app = create_multi_workspace_app(registry, password=args.password)
    if args.listen_to:
        from pycodex_router.client import attach_router

        attach_router(app, registry, args.listen_to, port)

    print(
        "pycodex workspace listening on http://{0}:{1}".format(host, port),
        flush=True,
    )
    for definition in definitions:
        print(
            "workspace {0}: state_path={1} work_dir={2} url=http://{3}:{4}/w/{0}/".format(
                definition.workspace_id,
                definition.state_path,
                definition.work_dir,
                host,
                port,
            ),
            flush=True,
        )
    run_owned_server(app, host, port, sys.stdin.fileno())
    return 0


def _build_workspace_entry(
    definition: "WorkspaceDefinition",
    args: "argparse.Namespace",
    persist_callback: "typing.Union[typing.Callable[[], None], None]" = None,
) -> "WorkspaceEntry":
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
            extra_contextual_user_messages=(
                [_workspace_context_text(definition.board_path)]
                if definition.board_path is not None
                else []
            ),
            cwd=definition.work_dir,
            toolset=args.toolset,
        )
        return WorkspaceInteractiveSession(
            build_runtime(agent),
            config_path=args.config,
        )

    def session_factory() -> "ThreadedWorkspaceInteractiveSession":
        return ThreadedWorkspaceInteractiveSession(
            build_session, asyncio.get_running_loop()
        )

    return WorkspaceEntry(
        definition=definition,
        manager=WorkspaceSessionManager(
            session_factory,
            definition.state_path,
            persist_callback=persist_callback,
        ),
    )


def _workspace_context_text(board_path: Path) -> str:
    return (
        "Current workspace Canvas file: {0}. "
        "Changes you make to this HTML file are shown to the user in real time. "
        "You can create or modify this file anytime.\n\n"
        "Replies and Canvas HTML can use extended links to open files and diffs "
        "in the board: git:REV:path, diff:REV:path, diff:BASE..HEAD:path, "
        "and diff:left..right. Append #L42 to select a file or left-side line, "
        "or #R42 for the right side of a diff. "
        "Use workspace://canvas to return to the live Canvas."
    ).format(board_path.resolve())


def _render_workspace_shell(
    board_path: "typing.Union[Path, None]",
    title: "typing.Union[str, None]" = None,
    work_dir: "typing.Union[Path, None]" = None,
    builtin: bool = False,
    access_role: str = "admin",
) -> str:
    board_label = str(board_path) if board_path is not None else "No board"
    cwd_label = str(work_dir or Path.cwd())
    page_title = str(title or "pycodex workspace")
    template = load_workspace_template(builtin)
    return (
        template.replace("__WORKSPACE_TITLE__", html.escape(page_title))
        .replace("__ACCESS_ROLE__", access_role)
        .replace("__BOARD_LABEL__", html.escape(board_label))
        .replace("__CWD_LABEL__", html.escape(cwd_label))
        .replace("__ENABLE_BOARD__", "true" if board_path is not None else "false")
        .replace("__BOARD_URL__", "board" if board_path is not None else "about:blank")
        .replace("__FRONTEND_DIRECTORY__", html.escape(str(frontend_directory())))
        .replace("__PACKAGE_DIRECTORY__", html.escape(str(Path(__file__).resolve().parent)))
        .replace("__PACKAGE_VERSION__", html.escape(importlib_metadata.version("pycodex-ws")))
    )


def _render_workspaces_manager_shell() -> str:
    return (Path(__file__).with_name("workspaces.html")).read_text(encoding="utf-8")


def _render_missing_board(board_path: Path) -> str:
    return _render_board_placeholder(
        "Canvas pending",
        "Your work will appear here",
        "Ask pycodex to create a page, a report, or a visual. "
        "This canvas updates as you work.",
        str(board_path),
    )


def _render_board_placeholder(
    title: str, heading: str, description: str, path_label: str = ""
) -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{0}</title><style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; min-height: 100dvh; display: grid; place-items: center;
  padding: 32px; background: #f8fbfd; color: #172630;
  font: 14px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ max-width: 360px; text-align: center; }}
svg {{ width: 40px; height: 40px; color: #aebfc9; margin-bottom: 20px; }}
h1 {{ margin: 0 0 12px; font-size: 23px; font-weight: 500; line-height: 1.35;
  letter-spacing: -0.5px; }}
p {{ margin: 0; color: #60727d; }}
code {{ display: block; margin-top: 28px; font-size: 11px; color: #60727d;
  overflow-wrap: anywhere; }}
</style></head><body><main>
<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.25"
aria-hidden="true"><rect x="3" y="3" width="26" height="26" rx="5"/>
<path d="M3 11h26M11 11v18"/></svg>
<h1>{1}</h1><p>{2}</p><code>{3}</code>
</main></body></html>""".format(
        html.escape(title),
        html.escape(heading),
        html.escape(description),
        html.escape(path_label),
    )


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
