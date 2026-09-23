import argparse
import asyncio
import html
import json
import mimetypes
import os
import secrets
import threading
from dataclasses import asdict, fields, is_dataclass

try:
    from contextlib import asynccontextmanager
except ImportError:  # pragma: no cover - Python 3.6 compatibility
    asynccontextmanager = None
import typing
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)

from pycodex.bootstrap import build_agent, build_model, build_runtime, configure_loguru
from pycodex.events import (
    IDLE_SLEEPING_STATUS,
    AssistantDeltaEvent,
    AutoCompactCompletedEvent,
    AutoCompactFailedEvent,
    AutoCompactStartedEvent,
    CommandCompletedEvent,
    CommandFailedEvent,
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
from pycodex.utils.event_helpers import (
    completed_history,
    shorten_title,
)

from .workspaces import (
    WorkspaceDefinition,
    WorkspaceEntry,
    WorkspaceRegistry,
    WorkspaceSessionManager,
    load_workspace_definitions,
    session_snapshot,
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
    )
    parser.add_argument(
        "--listen",
        default="127.0.0.1:6007",
        help="Bind address as host:port, for example 0.0.0.0:6007.",
    )
    parser.add_argument(
        "--workspace-config",
        default="./workspaces.json",
        help=(
            "Optional JSON file listing workspaces. Each entry needs `id`, "
            "`board`, and `work_dir`; routes are served under /w/<id>/."
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
        help="Builtin tool names for all sessions; an empty list disables tools.",
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
        self._turns: "typing.List[typing.Dict[str, object]]" = []
        self._turns_by_submission_id: "typing.Dict[str, typing.Dict[str, object]]" = {}
        self._turns_by_turn_id: "typing.Dict[str, typing.Dict[str, object]]" = {}
        self._title = ""
        self._model = "pycodex"
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

    def handle_event(self, event: "Event") -> None:
        with self._lock:
            self._apply_runtime_event(event)
            payload = _event_data(event)
            payload.update({"type": "event", "snapshot": self.snapshot()})
            if isinstance(event, ToolCompletedEvent):
                payload["summary"] = event.visualize()
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
                turn = self._ensure_turn(
                    submission_id, submission_id, str(prompt or "")
                )
                turn["response"] = str(response or "")
                turn["status"] = "completed"
                turn["queue"] = "history"
                turn["sender"] = "resume"
            event = {"type": "snapshot", "snapshot": self.snapshot()}
        self._publish_nowait(event)

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
            subscribers = tuple(self._subscribers)
            self._subscribers.clear()
        self._publish_to_queues(subscribers, None)

    def snapshot(self) -> "typing.Dict[str, object]":
        with self._lock:
            return {
                "running": bool(self._spinner_status),
                "status": self._spinner_status,
                "status_kind": "spinner" if self._spinner_status else "idle",
                "spinner": self._spinner_status,
                "model": self._model,
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
                "running": bool(self._spinner_status),
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
            self._rollout_path = state["rollout_path"]
            self._recorded_rollout_path = state["recorded_rollout_path"]
            self._input_request = state["input_request"]
            self._accepts_input = state["accepts_input"]
            self._max_context_window = state["max_context_window"]
            self._auto_compact_token_limit = state["auto_compact_token_limit"]
            self._usage_tokens = state["usage_tokens"]
            if event.reason in {"attach", "history"}:
                self.load_session_history(state["title"], completed_history(state))
                active = state["active_turn"]
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
            elif event.reason == "title":
                self.set_session_title(state["title"])
            else:
                self._set_title(state["title"])
            return
        if isinstance(event, CommandCompletedEvent):
            if event.result["kind"] in {"title_changed", "resumed"}:
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
        if not isinstance(event, TurnEvent):
            return
        turn_id = event.turn_id
        submission_id = event.submission_id or turn_id
        turn = self._turns_by_submission_id.get(submission_id)

        if isinstance(event, TurnStartedEvent):
            self._set_spinner_status(event.kind)
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

    async def start(self) -> "WorkspaceInteractiveSession":
        await self.runtime.start(self.config_path)
        if self._frontend_id is None:
            self._frontend_id = self.runtime.attach(self.view.handle_event)
        return self

    async def close(self) -> None:
        try:
            await self.runtime.close()
        finally:
            self.detach()

    def detach(self):
        if self._frontend_id is not None:
            self.runtime.detach(self._frontend_id)
            self._frontend_id = None
        self.view.close()

    async def submit(
        self, prompt: str, sender: str = "web"
    ) -> "typing.Dict[str, object]":
        try:
            receipt = await self.runtime.submit_input(prompt, sender)
        except (ValueError, RuntimeError) as exc:
            return {"ok": False, "error": str(exc), "snapshot": self.snapshot()}
        return {
            "ok": True,
            "type": "submitted",
            "submission_id": receipt.submission_id,
            "snapshot": self.snapshot(),
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
        self, rollout_path: str, title: str = "", fork: bool = False
    ) -> None:
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
        self, rollout_path: str, title: str = "", fork: bool = False
    ) -> None:
        session = self._session
        loop = self._worker_loop
        if session is None or loop is None:
            return

        future = asyncio.run_coroutine_threadsafe(
            session.restore_from_rollout(rollout_path, title=title, fork=fork),
            loop,
        )
        await asyncio.wrap_future(future)


def create_app(
    session_source: "typing.Union[WorkspaceSessionManager, SessionFactory]",
    board_path: "typing.Union[Path, None]",
    password: "typing.Union[str, None]" = None,
) -> FastAPI:
    manager = (
        session_source
        if isinstance(session_source, WorkspaceSessionManager)
        else WorkspaceSessionManager(session_source, board_path)
    )
    app = _create_lifespan_app(manager.start, manager.close)
    auth_token = _install_auth(app, password)
    _install_workspace_routes(app, manager, board_path)
    app.state.workspace_auth_token = auth_token
    return app


def create_multi_workspace_app(
    registry: "WorkspaceRegistry",
    password: "typing.Union[str, None]" = None,
) -> FastAPI:
    app = _create_lifespan_app(registry.start, registry.close)
    auth_token = _install_auth(app, password)

    @app.get("/")
    async def index() -> Response:
        return _html_response(_render_workspaces_manager_shell())

    @app.get("/favicon.ico")
    async def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/api/workspaces")
    async def workspaces() -> JSONResponse:
        return JSONResponse({"workspaces": registry.list_workspaces()})

    @app.post("/api/workspaces")
    async def add_workspace(payload: "typing.Dict[str, object]") -> JSONResponse:
        try:
            entry = await registry.add_workspace(
                str(payload.get("name") or ""),
                work_dir=str(payload.get("dir") or "./"),
                board=payload.get("board"),
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

    @app.delete("/api/workspaces/{workspace_id}")
    async def delete_workspace(workspace_id: str) -> JSONResponse:
        try:
            await registry.delete_workspace(workspace_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="workspace not found")
        return JSONResponse({"ok": True, "workspaces": registry.list_workspaces()})

    @app.api_route("/w/{workspace_id}", methods=["GET", "HEAD"])
    async def workspace_index_redirect(workspace_id: str) -> RedirectResponse:
        _workspace_entry_or_404(registry, workspace_id)
        return RedirectResponse(url="/w/{0}/".format(workspace_id), status_code=307)

    @app.api_route("/w/{workspace_id}/", methods=["GET", "HEAD"])
    async def workspace_index(workspace_id: str) -> HTMLResponse:
        entry = _workspace_entry_or_404(registry, workspace_id)
        return _html_response(
            _render_workspace_shell(
                entry.definition.board_path,
                title=entry.definition.workspace_id,
                work_dir=entry.definition.work_dir,
            )
        )

    @app.api_route("/w/{workspace_id}/board", methods=["GET", "HEAD"])
    async def workspace_board(workspace_id: str) -> Response:
        entry = _workspace_entry_or_404(registry, workspace_id)
        return _board_response(entry.definition.board_path)

    @app.get("/w/{workspace_id}/api/board")
    async def workspace_board_status(workspace_id: str) -> JSONResponse:
        entry = _workspace_entry_or_404(registry, workspace_id)
        return _board_status_response(entry.definition.board_path)

    @app.get("/w/{workspace_id}/ws/session")
    async def workspace_websocket_backend_hint(workspace_id: str) -> JSONResponse:
        _workspace_entry_or_404(registry, workspace_id)
        return _websocket_backend_hint_response()

    @app.get("/w/{workspace_id}/api/sessions")
    async def workspace_sessions(workspace_id: str) -> JSONResponse:
        entry = _workspace_entry_or_404(registry, workspace_id)
        return _sessions_response(entry.manager)

    @app.post("/w/{workspace_id}/api/sessions")
    async def workspace_new_session(workspace_id: str) -> JSONResponse:
        entry = _workspace_entry_or_404(registry, workspace_id)
        return await _new_session_response(entry.manager)

    @app.delete("/w/{workspace_id}/api/sessions/{session_id}")
    async def workspace_delete_session(
        workspace_id: str, session_id: str
    ) -> JSONResponse:
        entry = _workspace_entry_or_404(registry, workspace_id)
        return await _delete_session_response(entry.manager, session_id)

    @app.get("/w/{workspace_id}/api/session")
    async def workspace_session(
        workspace_id: str,
        session_id: "typing.Union[str, None]" = None,
    ) -> JSONResponse:
        entry = _workspace_entry_or_404(registry, workspace_id)
        return _session_response(entry.manager, session_id)

    @app.post("/w/{workspace_id}/api/session/message")
    async def workspace_message(
        workspace_id: str,
        payload: "typing.Dict[str, object]",
    ) -> JSONResponse:
        entry = _workspace_entry_or_404(registry, workspace_id)
        return await _message_response(entry.manager, payload)

    @app.websocket("/w/{workspace_id}/ws/session")
    async def workspace_websocket_session(
        workspace_id: str, websocket: WebSocket
    ) -> None:
        if not _auth_cookie_matches(
            auth_token, websocket.cookies.get(AUTH_COOKIE_NAME)
        ):
            await websocket.close(code=1008)
            return
        try:
            entry = registry.get(workspace_id)
        except (KeyError, ValueError):
            await websocket.close(code=1008)
            return
        await _websocket_session_handler(entry.manager, websocket)

    @app.api_route(
        "/w/{workspace_id}/{asset_path:path}",
        methods=["GET", "HEAD"],
    )
    async def workspace_board_asset(workspace_id: str, asset_path: str) -> Response:
        entry = _workspace_entry_or_404(registry, workspace_id)
        return _board_asset_response(entry.definition.board_path, asset_path)

    return app


def _install_auth(app: FastAPI, password: "typing.Union[str, None]") -> str:
    password_text = str(password or "")
    if not password_text:
        return ""
    token = secrets.token_urlsafe(32)

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        path = request.url.path
        if path in {"/favicon.ico", "/login"} or _auth_cookie_matches(
            token,
            request.cookies.get(AUTH_COOKIE_NAME),
        ):
            return await call_next(request)
        if path.startswith("/api/") or "/api/" in path:
            return JSONResponse(
                {"ok": False, "error": "authentication required"},
                status_code=401,
            )
        target = path + ("?" + request.url.query if request.url.query else "")
        return RedirectResponse(
            url="/login?" + urlencode({"next": target}), status_code=303
        )

    @app.get("/login")
    async def login_page() -> HTMLResponse:
        return _html_response(_render_login_shell())

    @app.post("/login")
    async def login(
        request: Request, payload: "typing.Dict[str, object]"
    ) -> JSONResponse:
        if not secrets.compare_digest(
            str(payload.get("password") or ""), password_text
        ):
            return JSONResponse(
                {"ok": False, "error": "invalid password"},
                status_code=401,
            )
        target = request.query_params.get("next", "/")
        if (
            not target.startswith("/")
            or target.startswith("//")
            or "\\" in target
            or any(ord(char) < 32 for char in target)
        ):
            target = "/"
        response = JSONResponse({"ok": True, "redirect": target})
        response.set_cookie(
            AUTH_COOKIE_NAME,
            token,
            httponly=True,
            samesite="lax",
        )
        return response

    return token


def _auth_cookie_matches(token: str, cookie: "typing.Union[str, None]") -> bool:
    if not token:
        return True
    return bool(cookie) and secrets.compare_digest(str(cookie), token)


def _render_login_shell() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>pycodex login</title>
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
    <h1>pycodex workspace</h1>
    <input id="passwordInput" type="password" autocomplete="current-password" autofocus>
    <button type="submit">Open</button>
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
      statusEl.textContent = "Invalid password";
    });
  </script>
</body>
</html>"""


def _create_lifespan_app(
    start: "typing.Callable[[], typing.Awaitable[None]]",
    close: "typing.Callable[[], typing.Awaitable[None]]",
) -> FastAPI:
    if asynccontextmanager is not None:

        @asynccontextmanager
        async def lifespan(_app):
            await start()
            try:
                yield
            finally:
                await close()

        return FastAPI(lifespan=lifespan)

    app = FastAPI()

    @app.on_event("startup")
    async def startup() -> None:
        await start()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await close()

    return app


def _install_workspace_routes(
    app: FastAPI,
    manager: WorkspaceSessionManager,
    board_path: "typing.Union[Path, None]",
) -> None:
    @app.get("/")
    async def index() -> HTMLResponse:
        return _html_response(_render_workspace_shell(board_path))

    @app.get("/favicon.ico")
    async def favicon() -> Response:
        return Response(status_code=204)

    @app.api_route("/board", methods=["GET", "HEAD"])
    async def board() -> Response:
        return _board_response(board_path)

    @app.get("/api/board")
    async def board_status() -> JSONResponse:
        return _board_status_response(board_path)

    @app.get("/ws/session")
    async def websocket_backend_hint() -> JSONResponse:
        return _websocket_backend_hint_response()

    @app.get("/api/sessions")
    async def sessions() -> JSONResponse:
        return _sessions_response(manager)

    @app.post("/api/sessions")
    async def new_session() -> JSONResponse:
        return await _new_session_response(manager)

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str) -> JSONResponse:
        return await _delete_session_response(manager, session_id)

    @app.get("/api/session")
    async def session(
        session_id: "typing.Union[str, None]" = None,
    ) -> JSONResponse:
        return _session_response(manager, session_id)

    @app.post("/api/session/message")
    async def message(
        payload: "typing.Dict[str, object]",
    ) -> JSONResponse:
        return await _message_response(manager, payload)

    @app.websocket("/ws/session")
    async def websocket_session(websocket: WebSocket) -> None:
        auth_token = typing.cast(str, app.state.workspace_auth_token)
        if not _auth_cookie_matches(
            auth_token, websocket.cookies.get(AUTH_COOKIE_NAME)
        ):
            await websocket.close(code=1008)
            return
        await _websocket_session_handler(manager, websocket)

    @app.api_route("/{asset_path:path}", methods=["GET", "HEAD"])
    async def board_asset(asset_path: str) -> Response:
        return _board_asset_response(board_path, asset_path)


def _board_status_response(board_path: "typing.Union[Path, None]") -> JSONResponse:
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


def _websocket_backend_hint_response() -> JSONResponse:
    return JSONResponse(
        {
            "error": "websocket backend is unavailable; HTTP polling is active",
        },
        status_code=426,
    )


def _sessions_response(manager: "WorkspaceSessionManager") -> JSONResponse:
    return JSONResponse({"sessions": manager.list_sessions()})


async def _new_session_response(manager: "WorkspaceSessionManager") -> JSONResponse:
    session_id = await manager.create_session()
    return JSONResponse(
        {
            "ok": True,
            "session_id": session_id,
            "sessions": manager.list_sessions(),
            "snapshot": session_snapshot(manager.get(session_id)),
        }
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
            "snapshot": session_snapshot(link),
        }
    )


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
            if action in {"send", "answer"}:
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
        link.unsubscribe(subscriber)
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
        event, (CommandCompletedEvent, CommandFailedEvent, InputQueuedEvent)
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
        return _html_response(_render_empty_board())
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
    import uvicorn

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

    print(
        "pycodex workspace listening on http://{0}:{1}".format(host, port),
        flush=True,
    )
    for definition in definitions:
        print(
            "workspace {0}: board={1} work_dir={2} url=http://{3}:{4}/w/{0}/".format(
                definition.workspace_id,
                definition.board_path or "",
                definition.work_dir,
                host,
                port,
            ),
            flush=True,
        )
    uvicorn.run(app, host=host, port=port, loop="asyncio")
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
                [_board_context_text(definition.board_path, definition.work_dir)]
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
            definition.board_path,
            persist_callback=persist_callback,
        ),
    )


def _board_context_text(
    board_path: Path,
    work_dir: "typing.Union[Path, None]" = None,
) -> str:
    return (
        "Current workspace board file: {0}. "
        "Changes you make to this file are shown to the user in real time. "
        "You can create or modify this file anytime."
    ).format(_format_board_path_for_prompt(board_path, work_dir=work_dir))


def _format_board_path_for_prompt(
    board_path: Path,
    work_dir: "typing.Union[Path, None]" = None,
) -> str:
    resolved = board_path.resolve()
    try:
        relative = os.path.relpath(
            str(resolved),
            str(Path(work_dir or Path.cwd()).resolve()),
        )
    except ValueError:
        return str(resolved)
    if relative == ".":
        return "."
    if relative.startswith(".."):
        return str(resolved)
    return "./{0}".format(relative)


def _render_workspace_shell(
    board_path: "typing.Union[Path, None]",
    title: "typing.Union[str, None]" = None,
    work_dir: "typing.Union[Path, None]" = None,
) -> str:
    board_label = str(board_path) if board_path is not None else "No board"
    cwd_label = str(work_dir or Path.cwd())
    page_title = str(title or "pycodex workspace")
    template = (Path(__file__).with_name("workspace.html")).read_text(encoding="utf-8")
    return (
        template.replace("__WORKSPACE_TITLE__", html.escape(page_title))
        .replace("__BOARD_LABEL__", html.escape(board_label))
        .replace("__CWD_LABEL__", html.escape(cwd_label))
    )


def _render_workspaces_manager_shell() -> str:
    return (Path(__file__).with_name("workspaces.html")).read_text(encoding="utf-8")


def _render_empty_board() -> str:
    return _render_board_placeholder(
        "No board",
        "No board connected",
        "Add a board from the workspaces page to see your work here.",
    )


def _render_missing_board(board_path: Path) -> str:
    return _render_board_placeholder(
        "Board pending",
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
