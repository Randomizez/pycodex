import asyncio
import typing
from collections import deque
from dataclasses import dataclass, field, replace

from .agent import BASE_EVENT_HANDLER, Agent, TurnInterrupted
from .utils.compat import Literal
from .events import (
    AssistantDeltaEvent,
    AutoCompactCompletedEvent,
    CommandCompletedEvent,
    CommandFailedEvent,
    CommandStartedEvent,
    Event,
    InputQueuedEvent,
    InputRequestedEvent,
    InputResolvedEvent,
    RuntimeUpdatedEvent,
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
from .model import ModelControl
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
        self._queue_event = None
        self._current_submission: "typing.Union[_QueuedSubmission, None]" = None
        self.event_handler = BASE_EVENT_HANDLER
        self.agent.event_handler = self._handle_agent_event
        self.codex_home = (
            agent.context_manager._config.codex_home or resolve_codex_home()
        )
        self.title = ""
        self.command_handlers = {}
        self._frontends = {}
        self._interactive_frontends = set()
        self._command_tasks = set()
        self._active_commands = {}
        self._close_handlers = []
        self._worker = None
        self._command_lock = None
        self._input_request = None
        self._background_work_count = 0
        self._published_tool_states = agent.tool_states()
        self._active_turn = None
        self._recorded_rollout_path = agent.recorded_session_file_path
        environment = agent.tool_registry.runtime_environment
        environment.request_user_input_manager.set_handler(self._request_user_input)
        environment.request_permissions_manager.set_handler(self._request_permissions)
        environment.plan_store.set_listener(lambda _plan: self._publish_update("plan"))
        environment.subagent_manager.set_state_listener(
            lambda: self._publish_update("agents")
        )
        for tool in self.agent.tool_registry.tools():
            tool.set_state_listener(lambda: self._publish_update("background_work"))

    def attach(self, event_handler, interactive=True):
        frontend_id = uuid7_string()
        self._frontends[frontend_id] = event_handler
        if interactive:
            self._interactive_frontends.add(frontend_id)
        self._notify_handler(
            event_handler, SessionStateEvent("attach", self.snapshot())
        )
        return frontend_id

    def detach(self, frontend_id):
        del self._frontends[frontend_id]
        self._interactive_frontends.discard(frontend_id)
        if not self._interactive_frontends and self._input_request is not None:
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
                "cd",
                "model",
                "effort",
                "resume",
                "compact",
                "fork",
            )
            + tuple(self.command_handlers)
            + ("queue",)
        )

    def snapshot(self):
        rollout_path = self.agent.session_file_path
        recorded_path = self.agent.recorded_session_file_path
        return {
            "session_id": self.agent.session_id,
            "cwd": str(self.agent.context_manager.cwd),
            "rollout_path": str(rollout_path) if rollout_path is not None else None,
            "recorded_rollout_path": (
                str(recorded_path) if recorded_path is not None else None
            ),
            "model": "{0}:{1}".format(
                self.agent.model_name,
                getattr(self.agent.model_client, "reasoning_effort", None) or "default",
            ),
            "title": self.title,
            "history": conversation_history_to_turns(self.agent.history),
            "tool_states": self.agent.tool_states(),
            "busy": self.is_busy,
            "closed": self.agent.is_shutdown,
            "accepts_input": self.agent.accepts_input,
            "context_window": self.agent.context_manager.resolve_model_context_window(),
            "max_context_window": (
                self.agent.context_manager.resolve_model_max_context_window()
            ),
            "auto_compact_token_limit": (
                self.agent.context_manager.resolve_auto_compact_token_limit()
            ),
            "usage_tokens": self.agent._last_total_usage_tokens,
            "background_work_count": self._background_work_count,
            "input_request": self._input_request_event(),
            "active_turn": (
                dict(self._active_turn) if self._active_turn is not None else None
            ),
            "plan": self.agent.tool_registry.runtime_environment.plan_store.snapshot(),
        }

    def state_snapshot(self):
        """Current frontend state, without completed conversation history."""
        agent = self.agent
        client = typing.cast(ModelControl, agent.model_client)
        active = self._active_turn
        return {
            "agent_session_id": agent.session_id,
            "title": self.title,
            "cwd": str(agent.context_manager.cwd),
            "model": {
                "name": agent.model_name,
                "reasoning_effort": client.reasoning_effort,
            },
            "busy": self.is_busy,
            "running": agent.is_running,
            "closed": agent.is_shutdown,
            "accepts_input": self.accepts_input,
            "active_turn": (
                {
                    key: active[key]
                    for key in (
                        "turn_id", "submission_id", "user_texts", "assistant_text",
                        "model_errors",
                    )
                } if active is not None else None
            ),
            "queued_inputs": [
                {
                    "submission_id": item.submission_id,
                    "turn_id": item.turn_id,
                    "queue": name,
                    "prompts": list(item.texts),
                }
                for name, queue in (
                    ("steer", self._steer_queue), ("enqueue", self._enqueue_queue)
                )
                for item in queue
            ],
            "input_request": self._input_request_event(),
            "plan": agent.tool_registry.runtime_environment.plan_store.snapshot(),
            "commands": [dict(command) for command in self._active_commands.values()],
            "background_work_count": sum(
                tool.background_work_count(False)
                for tool in agent.tool_registry.tools()
            ),
            "context_window": agent.context_manager.resolve_model_context_window(),
            "max_context_window": agent.context_manager.resolve_model_max_context_window(),
            "auto_compact_token_limit": agent.context_manager.resolve_auto_compact_token_limit(),
            "usage_tokens": agent._last_total_usage_tokens,
        }

    def background_work_snapshot(self):
        return [
            dict(item, tool=tool.name)
            for tool in self.agent.tool_registry.tools()
            for item in tool.background_work_snapshot()
        ]

    async def list_models(self):
        return await self.agent.model_client.list_models()

    def _publish_update(self, reason):
        self._publish(RuntimeUpdatedEvent(reason))

    def publish_state(self, reason):
        self._published_tool_states = self.agent.tool_states()
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

    def set_cwd(self, cwd):
        self.require_idle("change directory")
        self.agent.set_cwd(cwd)
        self.publish_state("cwd")

    def resume(self, path=None, title=""):
        self.require_idle("resume")
        resumed = self.agent.resume(path)
        if resumed is not None:
            self._background_work_count = sum(
                tool.background_work_count(True)
                for tool in self.agent.tool_registry.tools()
            )
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
            for tool in self.agent.tool_registry.tools():
                tool.start_background_work()
        return self

    def _get_command_lock(self):
        if self._command_lock is None:
            self._command_lock = asyncio.Lock()
        return self._command_lock

    def _wake_worker(self):
        if self._queue_event is not None:
            self._queue_event.set()

    async def close(self):
        if self._worker is None:
            self._worker = asyncio.create_task(self._run_forever())
        if self.accepts_input:
            self.agent.accepts_input = False
            self.publish_state("admission")
            if self._input_request is not None:
                self._finish_input_request(None)
            self._wake_worker()
        await asyncio.shield(self._worker)

    async def submit_input(self, text, sender="user", wait_for_command=True):
        if not self.agent.accepts_input:
            raise RuntimeError("agent is shutting down")
        input_starts_with_space = text.startswith(" ")
        text = text.strip()
        command = text.split(None, 1)[0] if text else ""
        # Codex treats another slash in the name as a path and a leading
        # ASCII space as an escape, even for a registered command.
        is_command = (
            command.startswith("/")
            and not input_starts_with_space
            and "/" not in command[1:]
            and len(command) > 1
        )
        if self._input_request is not None and not is_command:
            self._answer_input(text)
            future = asyncio.get_running_loop().create_future()
            future.set_result(None)
            return SubmittedInput(uuid7_string(), "answer", future)
        if not text:
            future = asyncio.get_running_loop().create_future()
            future.set_result(None)
            return SubmittedInput(uuid7_string(), "empty", future)
        parts = text.split(None, 1)
        argument = parts[1] if len(parts) == 2 else ""
        if not is_command:
            return await self._submit_message(text, "steer", sender, False)
        if command == "/queue":
            if not argument.strip():
                raise ValueError("Usage: /queue <message>")
            return await self._submit_message(argument.strip(), "enqueue", sender, True)
        submission_id = uuid7_string()
        future = asyncio.get_running_loop().create_future()
        future.add_done_callback(self._observe_completion)
        receipt = SubmittedInput(submission_id, "command", future)
        self._active_commands[submission_id] = {
            "submission_id": submission_id, "command": command,
            "sender": sender, "status": "queued",
        }
        self._publish_update("commands")
        operation = self._run_command(receipt, command, argument.strip(), sender)
        if wait_for_command:
            await operation
        else:
            task = asyncio.create_task(operation)
            self._command_tasks.add(task)
            task.add_done_callback(self._command_tasks.discard)
        return receipt

    async def _run_command(self, receipt, command, argument, sender):
        try:
            async with self._get_command_lock():
                self._active_commands[receipt.submission_id]["status"] = "running"
                self._publish(
                    CommandStartedEvent(receipt.submission_id, command, sender)
                )
                result = await self._execute_command(command[1:], argument)
        except asyncio.CancelledError:
            del self._active_commands[receipt.submission_id]
            receipt.future.cancel()
            self._publish(CommandFailedEvent(
                receipt.submission_id, command, "command caller cancelled",
                sender, "CancelledError",
            ))
            raise
        except Exception as exc:
            del self._active_commands[receipt.submission_id]
            receipt.future.set_exception(exc)
            self._publish(CommandFailedEvent(
                receipt.submission_id, command, str(exc), sender, type(exc).__name__,
            ))
        else:
            del self._active_commands[receipt.submission_id]
            receipt.future.set_result(result)
            self._publish(
                CommandCompletedEvent(receipt.submission_id, command, result, sender)
            )

    async def _execute_command(self, command, argument):
        if command in {"help", "history", "compact", "fork"} and argument:
            raise ValueError("Usage: /{0}".format(command))
        if command == "help":
            return {"kind": "help", "commands": self.commands()}
        if command == "history":
            return {"kind": "history", "state": self.snapshot()}
        if command == "cd":
            if argument:
                self.set_cwd(argument)
            return {"kind": "cwd", "cwd": str(self.agent.context_manager.cwd)}
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
                "models": await self.list_models(),
            }
        if command == "effort":
            client = typing.cast(ModelControl, self.agent.model_client)
            effort = client.reasoning_effort
            if argument:
                if len(argument.split()) != 1:
                    raise ValueError("Usage: /effort [<value>|default]")
                self.require_idle("change reasoning effort")
                client.reasoning_effort = None if argument == "default" else argument
                effort = client.reasoning_effort
                self.publish_state("effort")
            return {"kind": "effort", "reasoning_effort": effort}
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
            self.publish_state("compacted")
            return {
                "kind": "compacted",
                "original_item_count": result.original_item_count,
                "retained_item_count": result.retained_item_count,
                "pruned_tool_results": result.pruned_tool_results,
            }
        if command == "fork":
            self.fork()
            return {"kind": "forked", "session_id": self.agent.session_id}
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
        if not self._interactive_frontends or not self.agent.accepts_input:
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
            self._publish_update("queue")
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
        self._publish_update("queue")
        asyncio.get_running_loop().call_soon(self._wake_worker)
        return submission_id, future

    async def _run_forever(self) -> "None":
        self._queue_event = asyncio.Event()
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
                self._publish_update("idle")
        try:
            if self._command_tasks:
                await asyncio.gather(*tuple(self._command_tasks))
            async with self._get_command_lock():
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
                "model_errors": [],
            }
        elif self._active_turn is not None:
            if isinstance(event, AssistantDeltaEvent):
                self._active_turn["assistant_text"] += event.delta
            elif isinstance(event, StreamErrorEvent):
                self._active_turn["assistant_text"] = ""
                self._active_turn["model_errors"].append(event.detail())
            elif isinstance(event, ToolCompletedEvent):
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
        if isinstance(event, TerminalEvent):
            states = self.agent.tool_states()
            if states != self._published_tool_states:
                self._published_tool_states = states
                self.publish_state("tools")
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
