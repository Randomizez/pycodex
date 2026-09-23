import asyncio
import typing
from collections import deque
from dataclasses import dataclass, field, replace

from .agent import BASE_EVENT_HANDLER, Agent, TurnInterrupted
from .compat import Literal
from .events import (
    AssistantDeltaEvent,
    AutoCompactCompletedEvent,
    CommandCompletedEvent,
    CommandFailedEvent,
    Event,
    InputQueuedEvent,
    InputRequestedEvent,
    InputResolvedEvent,
    SessionClosedEvent,
    SessionStateEvent,
    StreamErrorEvent,
    TerminalEvent,
    ToolCompletedEvent,
    TurnCompletedEvent,
    TurnEvent,
    TurnFailedEvent,
    TurnInterruptedEvent,
    TurnStartedEvent,
)
from .protocol import TurnResult
from .utils import uuid7_string
from .utils.event_helpers import shorten_title
from .utils.session_persist import (
    conversation_history_to_turns,
    list_resumable_sessions,
    resolve_codex_home,
    select_resumable_session,
)


class SubmissionInterrupted(RuntimeError):
    def __init__(self) -> "None":
        super().__init__("submission interrupted")


@dataclass
class _QueuedSubmission:
    submission_id: "str"
    turn_id: "str"
    texts: "typing.List[str]"
    futures: "typing.List[asyncio.Future[TurnResult]]"


@dataclass(frozen=True)
class SubmittedInput:
    submission_id: "str"
    kind: "str"
    future: "asyncio.Future"


@dataclass
class _InputRequest:
    request_id: "str"
    kind: "str"
    payload: "dict"
    future: "asyncio.Future"
    question_index: "int" = 0
    answers: "dict" = field(default_factory=dict)
    other: "bool" = False
    timer: "typing.Union[asyncio.TimerHandle, None]" = None


class AgentRuntime:
    """Frontend-independent session input, scheduling, state and events."""

    def __init__(self, agent: "Agent") -> "None":
        self.agent = agent
        self._enqueue_queue: "deque[_QueuedSubmission]" = deque()
        self._steer_queue: "deque[_QueuedSubmission]" = deque()
        self._queue_event = asyncio.Event()
        self._current_submission: "typing.Union[_QueuedSubmission, None]" = None
        self.event_handler = BASE_EVENT_HANDLER
        self.agent.event_handler = self._handle_agent_event
        self.codex_home = (
            agent.context_manager._config.codex_home or resolve_codex_home()
        )
        self.title = ""
        self.command_handlers = {}
        self._frontends = {}
        self._close_handlers = []
        self._worker = None
        self._command_lock = asyncio.Lock()
        self._input_request = None
        self._background_work_count = 0
        self._active_turn = None
        self._recorded_rollout_path = agent.recorded_session_file_path
        environment = agent.tool_registry.runtime_environment
        environment.request_user_input_manager.set_handler(self._request_user_input)
        environment.request_permissions_manager.set_handler(self._request_permissions)

    def attach(self, event_handler):
        frontend_id = uuid7_string()
        self._frontends[frontend_id] = event_handler
        self._notify_handler(
            event_handler, SessionStateEvent("attach", self.snapshot())
        )
        return frontend_id

    def detach(self, frontend_id):
        del self._frontends[frontend_id]
        if not self._frontends and self._input_request is not None:
            self._finish_input_request(None)

    def register_command(self, name, handler):
        if name in self.commands():
            raise ValueError("command already registered: /{0}".format(name))
        self.command_handlers[name] = handler

    def add_close_handler(self, handler):
        self._close_handlers.append(handler)

    def commands(self):
        return (
            (
                "help",
                "history",
                "title",
                "model",
                "resume",
                "compact",
                "fork",
            )
            + tuple(self.command_handlers)
            + ("queue", "exit", "quit")
        )

    def snapshot(self):
        rollout_path = self.agent.session_file_path
        recorded_path = self.agent.recorded_session_file_path
        return {
            "session_id": self.agent.session_id,
            "rollout_path": str(rollout_path) if rollout_path is not None else None,
            "recorded_rollout_path": (
                str(recorded_path) if recorded_path is not None else None
            ),
            "model": self.agent.model_name,
            "title": self.title,
            "history": conversation_history_to_turns(self.agent.history),
            "busy": self.is_busy,
            "closed": self.agent.is_shutdown,
            "accepts_input": self.agent.accepts_input,
            "context_window": self.agent.context_manager.resolve_model_context_window(),
            "usage_tokens": self.agent._last_total_usage_tokens,
            "background_work_count": self._background_work_count,
            "input_request": self._input_request_event(),
            "active_turn": (
                dict(self._active_turn) if self._active_turn is not None else None
            ),
            "plan": self.agent.tool_registry.runtime_environment.plan_store.snapshot(),
        }

    def publish_state(self, reason):
        self._recorded_rollout_path = self.agent.recorded_session_file_path
        self._publish(SessionStateEvent(reason, self.snapshot()))

    def require_idle(self, operation):
        if self.is_busy:
            raise RuntimeError(
                "Cannot {0} while work is running or queued.".format(operation)
            )

    def set_title(self, title):
        self.title = title
        self.publish_state("title")

    def set_model(self, model):
        self.require_idle("change model")
        self.agent.set_model(model)
        self.publish_state("model")

    def resume(self, path=None, title=""):
        self.require_idle("resume")
        resumed = self.agent.resume(path)
        if resumed is not None:
            self.title = title or str(resumed["title"])
        self.publish_state("history" if path is not None else "admission")
        return resumed

    def fork(self):
        self.require_idle("fork")
        self.agent.fork()
        self.publish_state("identity")

    async def start(self, config_path=None):
        if config_path is not None:
            self.codex_home = resolve_codex_home(config_path)
        if self._worker is None or self._worker.done():
            if self.agent.is_shutdown:
                raise RuntimeError("resume the agent before restarting its runtime")
            self.agent.accepts_input = True
            self._worker = asyncio.create_task(self._run_forever())
        return self

    async def close(self):
        if self._worker is None:
            self._worker = asyncio.create_task(self._run_forever())
        if self.accepts_input:
            self.agent.accepts_input = False
            self.publish_state("admission")
            if self._input_request is not None:
                self._finish_input_request(None)
            self._queue_event.set()
        await asyncio.shield(self._worker)

    async def submit_input(self, text, sender="user"):
        if not self.agent.accepts_input:
            raise RuntimeError("agent is shutting down")
        text = text.strip()
        if self._input_request is not None and not text.startswith("/"):
            self._answer_input(text)
            future = asyncio.get_running_loop().create_future()
            future.set_result(None)
            return SubmittedInput(uuid7_string(), "answer", future)
        if not text:
            future = asyncio.get_running_loop().create_future()
            future.set_result(None)
            return SubmittedInput(uuid7_string(), "empty", future)
        parts = text.split(None, 1)
        command = parts[0]
        argument = parts[1] if len(parts) == 2 else ""
        if command == "/queue":
            if not argument.strip():
                raise ValueError("Usage: /queue <message>")
            return await self._submit_message(argument.strip(), "enqueue", sender, True)
        if not command.startswith("/"):
            return await self._submit_message(text, "steer", sender, False)
        submission_id = uuid7_string()
        future = asyncio.get_running_loop().create_future()
        try:
            if command in {"/exit", "/quit"}:
                result = await self._execute_command(command[1:], argument.strip())
            else:
                async with self._command_lock:
                    if not self.agent.accepts_input:
                        raise RuntimeError("agent is shutting down")
                    result = await self._execute_command(command[1:], argument.strip())
        except Exception as exc:
            future.set_exception(exc)
            self._publish(CommandFailedEvent(submission_id, command, str(exc), sender))
        else:
            future.set_result(result)
            self._publish(CommandCompletedEvent(submission_id, command, result, sender))
        future.add_done_callback(self._observe_completion)
        return SubmittedInput(submission_id, "command", future)

    async def _execute_command(self, command, argument):
        if (
            command in {"help", "history", "compact", "fork", "exit", "quit"}
            and argument
        ):
            raise ValueError("Usage: /{0}".format(command))
        if command == "help":
            return {"kind": "help", "commands": self.commands()}
        if command == "history":
            return {"kind": "history", "state": self.snapshot()}
        if command == "title":
            if argument:
                self.set_title(argument)
            return {
                "kind": "title_changed" if argument else "title",
                "title": self.title,
            }
        if command == "model":
            if argument:
                self.set_model(argument)
                return {"kind": "model_changed", "model": self.agent.model_name}
            return {
                "kind": "models",
                "model": self.agent.model_name,
                "models": await self.agent.model_client.list_models(),
            }
        if command == "resume":
            if not argument:
                return {
                    "kind": "sessions",
                    "sessions": list_resumable_sessions(self.codex_home),
                }
            self.require_idle("resume")
            selected = select_resumable_session(self.codex_home, argument)
            self.resume(selected["rollout_path"], selected["title"])
            return {"kind": "resumed", "state": self.snapshot()}
        if command == "compact":
            self.require_idle("compact")
            if not self.agent.history:
                return {"kind": "compact_empty"}
            result = await self.agent.compact()
            self.publish_state("history")
            return {
                "kind": "compacted",
                "original_item_count": result.original_item_count,
                "retained_item_count": result.retained_item_count,
                "pruned_tool_results": result.pruned_tool_results,
            }
        if command == "fork":
            self.fork()
            return {"kind": "forked", "session_id": self.agent.session_id}
        if command in {"exit", "quit"}:
            await self.close()
            return {"kind": "closed"}
        handler = self.command_handlers.get(command)
        if handler is None:
            raise ValueError("Unknown command: /{0}".format(command))
        return await handler(argument)

    async def _submit_message(self, text, queue, sender, explicit_queue):
        busy = self.is_busy
        submission_id, future = await self.enqueue_user_turn(text, queue)
        future.add_done_callback(self._observe_completion)
        self._publish(
            InputQueuedEvent(submission_id, text, queue, sender, explicit_queue, busy)
        )
        return SubmittedInput(submission_id, "turn", future)

    @staticmethod
    def _observe_completion(future):
        if not future.cancelled():
            future.exception()

    async def _request_user_input(self, payload):
        return await self._request_input("questions", payload)

    async def _request_permissions(self, payload):
        return await self._request_input("permissions", payload)

    async def _request_input(self, kind, payload):
        if not self._frontends or not self.agent.accepts_input:
            return None
        if self._input_request is not None:
            raise RuntimeError("a user input request is already pending")
        future = asyncio.get_running_loop().create_future()
        request = _InputRequest(uuid7_string(), kind, payload, future)
        self._input_request = request
        if kind == "questions" and not payload["questions"]:
            self._finish_input_request({"answers": {}})
        else:
            self._publish_input_request()
        timeout = payload.get("autoResolutionMs")
        request.timer = (
            None
            if timeout is None
            else asyncio.get_running_loop().call_later(
                timeout / 1000.0,
                self._finish_input_request,
                None,
            )
        )
        try:
            return await future
        finally:
            if request.timer is not None:
                request.timer.cancel()
            if self._input_request is request:
                self._input_request = None
                self._publish(InputResolvedEvent(request.request_id))

    def _input_request_event(self):
        request = self._input_request
        if request is None:
            return None
        if request.kind == "questions":
            return InputRequestedEvent(
                request.request_id,
                request.kind,
                request.other,
                question=request.payload["questions"][request.question_index],
            )
        return InputRequestedEvent(
            request.request_id,
            request.kind,
            request.other,
            permissions=request.payload,
        )

    def _publish_input_request(self):
        self._publish(self._input_request_event())

    def _answer_input(self, text):
        request = self._input_request
        if not text:
            self._finish_input_request(None)
            return
        if request.kind == "permissions":
            answer = text.lower()
            granted = answer in {"t", "turn", "y", "yes", "s", "session"}
            self._finish_input_request(
                {
                    "permissions": (
                        request.payload.get("permissions", {}) if granted else {}
                    ),
                    "scope": "session" if answer in {"s", "session"} else "turn",
                }
            )
            return
        question = request.payload["questions"][request.question_index]
        options = question["options"]
        if not request.other and text.isdigit():
            choice = int(text)
            if choice == 0:
                request.other = True
                self._publish_input_request()
                return
            if 1 <= choice <= len(options):
                text = options[choice - 1]["label"]
        request.answers[question["id"]] = {"answers": [text]}
        request.question_index += 1
        request.other = False
        if request.question_index == len(request.payload["questions"]):
            self._finish_input_request({"answers": request.answers})
        else:
            self._publish_input_request()

    def answer_input(self, request_id, answer):
        if self._input_request is None or self._input_request.request_id != request_id:
            raise ValueError("input request is no longer pending")
        self._finish_input_request(answer)

    def _finish_input_request(self, answer):
        request, self._input_request = self._input_request, None
        if request.timer is not None:
            request.timer.cancel()
        request.future.set_result(answer)
        self._publish(InputResolvedEvent(request.request_id))

    @property
    def accepts_input(self) -> "bool":
        return self.agent.accepts_input

    @property
    def is_busy(self) -> "bool":
        return (
            self.agent.is_running
            or self._current_submission is not None
            or bool(self._steer_queue or self._enqueue_queue)
            or (not self.accepts_input and not self.agent.is_shutdown)
        )

    async def submit_user_turn(self, text: "str") -> "TurnResult":
        _submission_id, future = await self.enqueue_user_turn(text, queue="enqueue")
        return await future

    async def enqueue_user_turn(
        self,
        text: "str",
        queue: 'Literal["enqueue", "steer"]' = "enqueue",
    ) -> "typing.Tuple[str, asyncio.Future[TurnResult]]":
        if queue not in {"enqueue", "steer"}:
            raise ValueError(f"unknown submission queue: {queue}")
        if not self.agent.accepts_input:
            raise RuntimeError("agent is shutting down")
        if queue == "steer":
            self.agent.stop_asap()
        future: "asyncio.Future[TurnResult]" = (
            asyncio.get_running_loop().create_future()
        )
        if queue == "steer" and self._steer_queue:
            queued = self._steer_queue[-1]
            queued.texts.append(text)
            queued.futures.append(future)
            return queued.submission_id, future

        submission_id = uuid7_string()
        current = self._current_submission if self.agent.is_running else None
        queued = _QueuedSubmission(
            submission_id=submission_id,
            turn_id=(
                current.turn_id
                if queue == "steer" and current is not None
                else submission_id
            ),
            texts=[text],
            futures=[future],
        )
        target = self._steer_queue if queue == "steer" else self._enqueue_queue
        target.append(queued)
        asyncio.get_running_loop().call_soon(self._queue_event.set)
        return submission_id, future

    async def _run_forever(self) -> "None":
        while True:
            queued = await self._next_submission()
            if queued is None:
                break
            self._current_submission = queued
            try:
                result = await self.agent.run_turn(
                    list(queued.texts), turn_id=queued.turn_id
                )
            except TurnInterrupted:
                self._finish_submission_exception(queued, SubmissionInterrupted())
            except Exception as exc:
                self._finish_submission_exception(queued, exc)
            else:
                self._finish_submission_result(queued, result)
            finally:
                self._current_submission = None
        try:
            async with self._command_lock:
                await self._close_resources()
        finally:
            self.agent.shutdown()
            self._publish(SessionClosedEvent())

    async def _close_resources(self):
        errors = []
        environment = self.agent.tool_registry.runtime_environment
        for handler in [environment.subagent_manager.shutdown] + self._close_handlers:
            try:
                await handler()
            except Exception as exc:
                errors.append(exc)
        if errors:
            for error in errors[1:]:
                asyncio.get_running_loop().call_exception_handler(
                    {
                        "message": "Session close handler failed",
                        "exception": error,
                    }
                )
            raise errors[0]

    async def _next_submission(self) -> "typing.Union[_QueuedSubmission, None]":
        while True:
            await self.agent.wait_until_idle()
            if self._steer_queue:
                return self._steer_queue.popleft()
            if self._enqueue_queue:
                return self._enqueue_queue.popleft()
            if not self.accepts_input:
                return None
            self._queue_event.clear()
            await self._queue_event.wait()

    @staticmethod
    def _finish_submission_result(
        queued: "_QueuedSubmission",
        result: "TurnResult",
    ) -> "None":
        for future in queued.futures:
            if not future.done():
                future.set_result(result)

    @staticmethod
    def _finish_submission_exception(
        queued: "_QueuedSubmission",
        exc: "Exception",
    ) -> "None":
        for future in queued.futures:
            if not future.done():
                future.set_exception(exc)

    def _handle_agent_event(self, event: "TurnEvent") -> "None":
        queued = self._current_submission
        if queued is not None and event.turn_id == queued.turn_id:
            event = replace(
                event, submission_id=event.submission_id or queued.submission_id
            )
        if isinstance(event, TurnStartedEvent):
            self._active_turn = {
                "turn_id": event.turn_id,
                "submission_id": event.submission_id or event.turn_id,
                "user_text": "\n".join(event.user_texts),
                "user_texts": list(event.user_texts),
                "assistant_text": "",
                "completed_history": conversation_history_to_turns(self.agent.history),
            }
        elif self._active_turn is not None:
            if isinstance(event, AssistantDeltaEvent):
                self._active_turn["assistant_text"] += event.delta
            elif isinstance(event, (StreamErrorEvent, ToolCompletedEvent)):
                self._active_turn["assistant_text"] = ""
            elif isinstance(event, AutoCompactCompletedEvent):
                self._active_turn["completed_history"] = conversation_history_to_turns(
                    self.agent.history
                )
            elif isinstance(
                event, (TurnCompletedEvent, TurnFailedEvent, TurnInterruptedEvent)
            ):
                self._active_turn = None
        if isinstance(event, TurnStartedEvent) and not self.title:
            prompt = "\n".join(event.user_texts)
            if prompt:
                self.title = shorten_title(prompt)
                self.publish_state("auto_title")
        if isinstance(event, TerminalEvent) and event.background_work_count is not None:
            self._background_work_count = event.background_work_count
        if self.agent.recorded_session_file_path != self._recorded_rollout_path:
            self.publish_state("recording")
        self._publish(event)

    def _publish(self, event: "Event"):
        for handler in (self.event_handler,) + tuple(self._frontends.values()):
            self._notify_handler(handler, event)

    @staticmethod
    def _notify_handler(handler, event):
        try:
            handler(event)
        except (Exception, asyncio.CancelledError) as exc:
            asyncio.get_running_loop().call_exception_handler(
                {
                    "message": "Submission event observer failed: " + event.kind,
                    "exception": exc,
                }
            )
