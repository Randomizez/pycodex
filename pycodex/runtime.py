import asyncio
import json
import typing
from collections import deque
from dataclasses import dataclass, field, replace
from time import monotonic

from .agent import BASE_EVENT_HANDLER, Agent, TurnInterrupted
from .events import (
    AssistantDeltaEvent,
    AutoCompactCompletedEvent,
    CommandCompletedEvent,
    CommandFailedEvent,
    CommandStartedEvent,
    CompactStartedEvent,
    Event,
    InputQueuedEvent,
    InputRequestedEvent,
    InputResolvedEvent,
    RuntimeUpdatedEvent,
    SessionClosedEvent,
    SessionStateEvent,
    StreamErrorEvent,
    ToolCompletedEvent,
    TurnCompletedEvent,
    TurnEvent,
    TurnFailedEvent,
    TurnInputReadyEvent,
    TurnInterruptedEvent,
    TurnStartedEvent,
)
from .model import ModelControl
from .protocol import AgentMessage, ToolResult, TurnResult, UserMessage
from .utils import uuid7_string
from .utils.event_helpers import shorten_title
from .utils.truncation import truncate_tool_result_for_history
from .utils.session_persist import (
    conversation_history_to_turns,
    list_resumable_sessions,
    resolve_codex_home,
    select_resumable_session,
)

if typing.TYPE_CHECKING:
    from .multi_agents import AgentsManager


class SubmissionInterrupted(RuntimeError):
    def __init__(self) -> "None":
        super().__init__("submission interrupted")


class _QueuedSubmission:
    def __init__(self, texts: "typing.List[str]") -> "None":
        self.submission_id = uuid7_string()
        self.turn_id = self.submission_id
        self.texts = texts
        self.futures: "typing.List[asyncio.Future[TurnResult]]" = []


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

    def event(self):
        payload = (
            {"question": self.payload["questions"][self.question_index]}
            if self.kind == "questions" else {"permissions": self.payload}
        )
        return InputRequestedEvent(self.request_id, self.kind, self.other, **payload)

    def answer(self, text):
        """Advance question progress and return (resolved, answer)."""
        if not text:
            return True, None
        if self.kind == "permissions":
            answer = text.lower()
            granted = answer in {"t", "turn", "y", "yes", "s", "session"}
            return True, {
                "permissions": self.payload.get("permissions", {}) if granted else {},
                "scope": "session" if answer in {"s", "session"} else "turn",
            }
        question = self.payload["questions"][self.question_index]
        options = question["options"]
        if not self.other and text.isdigit():
            choice = int(text)
            if choice == 0:
                self.other = True
                return False, None
            if 1 <= choice <= len(options):
                text = options[choice - 1]["label"]
        self.answers[question["id"]] = {"answers": [text]}
        self.question_index += 1
        self.other = False
        complete = self.question_index == len(self.payload["questions"])
        return complete, {"answers": self.answers} if complete else None


def _parse_input(text):
    """Recognize slash commands, preserving ASCII-space escapes and paths."""
    escaped = text.startswith(" ")
    text = text.strip()
    parts = text.split(None, 1)
    command = parts[0] if parts else ""
    if escaped or not command.startswith("/") or "/" in command[1:] or len(command) < 2:
        command = None
    return text, command, parts[1] if len(parts) == 2 else ""


class AgentRuntime:
    """Frontend-independent session input, scheduling, state and events."""

    def __init__(self, agent: "Agent", agents_manager: "AgentsManager") -> "None":
        self.agent = agent
        self.agents_manager = agents_manager
        self._enqueue_queue: "deque[_QueuedSubmission]" = deque()
        self._steer_queue: "deque[typing.Union[UserMessage, AgentMessage, ToolResult]]" = deque()
        self._triggered_submission: "typing.Union[_QueuedSubmission, None]" = None
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
        self._active_turn = None
        self._turn_started_at = None
        self._turn_finished_at = None
        self._input_activity: "typing.Union[asyncio.Event, None]" = None
        self._wait_interrupted = False
        self._last_agent_status: "typing.Union[str, dict]" = "pending_init"
        self._recorded_rollout_path = agent.recorded_session_file_path
        environment = agent.tool_registry.runtime_environment
        environment.request_user_input_manager.set_handler(self._request_user_input)
        environment.request_permissions_manager.set_handler(self._request_permissions)
        environment.plan_store.set_listener(lambda _plan: self.publish_update("plan"))
        for tool in self.agent.tool_registry.tools():
            tool.bind_runtime(self)
            tool.set_state_listener(lambda: self.publish_update("background_work"))

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

    def receive_tool_output(self, result: "ToolResult"):
        if not self.accepts_input:
            return
        self.message(truncate_tool_result_for_history(result))
        if not self.is_busy and self.accepts_input:
            self.trigger()

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
            )
            + tuple(self.command_handlers)
            + ("queue",)
        )

    def _snapshot_fields(self, execution):
        agent = self.agent
        return {
            "title": self.title,
            "cwd": str(agent.context_manager.cwd),
            "busy": self.is_busy,
            "closed": agent.is_shutdown,
            "accepts_input": agent.accepts_input,
            "context_window": agent.context_manager.resolve_model_context_window(),
            "max_context_window": agent.context_manager.resolve_model_max_context_window(),
            "auto_compact_token_limit": agent.context_manager.resolve_auto_compact_token_limit(),
            "usage_tokens": execution.usage_tokens,
            "background_work_count": sum(
                tool.background_work_count(False)
                for tool in agent.tool_registry.tools()
            ),
            "input_request": self._input_request_event(),
            "plan": agent.tool_registry.runtime_environment.plan_store.snapshot(),
        }

    def snapshot(self):
        execution = self.agent.execution_snapshot()
        rollout_path = self.agent.session_file_path
        recorded_path = self.agent.recorded_session_file_path
        return {
            **self._snapshot_fields(execution),
            "session_id": execution.session_id,
            "rollout_path": str(rollout_path) if rollout_path is not None else None,
            "recorded_rollout_path": str(recorded_path) if recorded_path is not None else None,
            "model": "{0}:{1}".format(
                self.agent.model_name,
                getattr(self.agent.model_client, "reasoning_effort", None) or "default",
            ),
            "history": conversation_history_to_turns(self.agent.history),
            "active_turn": (
                dict(self._active_turn) if self._active_turn is not None else None
            ),
        }

    def state_snapshot(self):
        """Current frontend state, without completed conversation history."""
        agent = self.agent
        execution = agent.execution_snapshot()
        client = typing.cast(ModelControl, agent.model_client)
        active = self._active_turn
        pending = []
        if self._triggered_submission is not None:
            pending.append(("steer", self._triggered_submission))
        pending.extend(("enqueue", item) for item in self._enqueue_queue)
        return {
            **self._snapshot_fields(execution),
            "agent_id": agent.agent_id,
            "agent_path": agent.agent_path,
            "agent_session_id": execution.session_id,
            "model": {
                "name": agent.model_name,
                "reasoning_effort": client.reasoning_effort,
            },
            "running": execution.running,
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
                for name, item in pending
            ],
            "commands": [dict(command) for command in self._active_commands.values()],
        }

    def background_work_snapshot(self):
        return [
            dict(item, tool=tool.name)
            for tool in self.agent.tool_registry.tools()
            for item in tool.background_work_snapshot()
        ]

    async def list_models(self):
        return await self.agent.model_client.list_models()

    def publish_update(self, reason):
        self._publish(RuntimeUpdatedEvent(reason))

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

    def set_cwd(self, cwd):
        self.require_idle("change directory")
        self.agent.set_cwd(cwd)
        self.publish_state("cwd")

    def resume(self, path=None, title=""):
        """Reopen an idle registered agent, optionally restoring its rollout."""
        self.agents_manager.get_runtime(self.agent.agent_id)
        self.require_idle("resume")
        resumed = self.agent.resume(path)
        for tool in self.agent.tool_registry.tools():
            tool.bind_runtime(self)
        if resumed is not None:
            self.title = title or str(resumed["title"])
        if self._worker is not None and not self._worker.done():
            self.agent.tool_registry.runtime_environment.start()
        self.publish_state("history" if path is not None else "admission")
        return resumed

    async def start(self, config_path=None):
        if config_path is not None:
            self.codex_home = resolve_codex_home(config_path)
        if not self.accepts_input:
            raise RuntimeError("resume the agent before restarting its runtime")
        if self._worker is None or self._worker.done():
            if self.agent.is_shutdown:
                raise RuntimeError("resume the agent before restarting its runtime")
            self._worker = asyncio.create_task(self._run_forever())
            self.agent.tool_registry.runtime_environment.start()
        return self

    def _get_command_lock(self):
        if self._command_lock is None:
            self._command_lock = asyncio.Lock()
        return self._command_lock

    def _wake_worker(self):
        if self._queue_event is not None:
            self._queue_event.set()

    async def close(self):
        """Stop admission across the subtree and wait for issued work and cleanup."""
        if self._worker is None:
            self._worker = asyncio.create_task(self._run_forever())
        if self.accepts_input:
            self.agents_manager.request_close(self.agent.agent_id)
        await asyncio.shield(self._worker)

    def request_close(self):
        """Manager-only local signal, after it captures this node's children."""
        if not self.accepts_input:
            return
        self.agent.accepts_input = False
        self.stop_asap()
        pending = list(self._enqueue_queue)
        if self._triggered_submission is not None:
            pending.append(self._triggered_submission)
        self._enqueue_queue.clear()
        self._steer_queue.clear()
        self._triggered_submission = None
        for queued in pending:
            self._finish_submission_exception(queued, SubmissionInterrupted())
        if self._input_request is not None:
            self._finish_input_request(None)
        self.publish_state("admission")
        self._wake_worker()

    async def submit_input(self, text, sender="user", wait_for_command=True):
        """Route frontend text to a pending answer, slash command or steer input."""
        if not self.agent.accepts_input:
            raise RuntimeError("agent is shutting down")
        text, command, argument = _parse_input(text)
        if self._input_request is not None and command is None:
            self._answer_input(text)
            future = asyncio.get_running_loop().create_future()
            future.set_result(None)
            return SubmittedInput(uuid7_string(), "answer", future)
        if not text:
            future = asyncio.get_running_loop().create_future()
            future.set_result(None)
            return SubmittedInput(uuid7_string(), "empty", future)
        if command is None:
            return self._submit_turn(text, "steer", sender)
        if command == "/queue":
            if not argument.strip():
                raise ValueError("Usage: /queue <message>")
            return self._submit_turn(argument.strip(), "enqueue", sender)
        submission_id = uuid7_string()
        future = asyncio.get_running_loop().create_future()
        future.add_done_callback(self._observe_completion)
        receipt = SubmittedInput(submission_id, "command", future)
        self._active_commands[submission_id] = {
            "submission_id": submission_id, "command": command,
            "sender": sender, "status": "queued",
        }
        self.publish_update("commands")
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
        if command in {"help", "history", "compact"} and argument:
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
        handler = self.command_handlers.get(command)
        if handler is None:
            raise ValueError("Unknown command: /{0}".format(command))
        return await handler(argument)

    def _submit_turn(self, text, queue, sender):
        busy = self.is_busy
        if queue == "enqueue":
            submission_id, future = self.enqueue(text)
        else:
            current = self._current_submission if self.agent.is_running else None
            self.stop_asap()
            self.message(text)
            submission_id, future = self.trigger()
            self._triggered_submission.texts.append(text)
            if current is not None:
                self._triggered_submission.turn_id = current.turn_id
        self._publish(
            InputQueuedEvent(submission_id, text, queue, sender, queue == "enqueue", busy)
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
        return self._input_request.event() if self._input_request is not None else None

    def _publish_input_request(self):
        self._publish(self._input_request_event())

    def _answer_input(self, text):
        complete, answer = self._input_request.answer(text)
        if complete:
            self._finish_input_request(answer)
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
            or self._triggered_submission is not None
            or bool(self._enqueue_queue)
            or (not self.accepts_input and not self.agent.is_shutdown)
        )

    @property
    def turn_elapsed_seconds(self) -> "float":
        if self._turn_started_at is None:
            return 0.0
        end = monotonic() if self._turn_finished_at is None else self._turn_finished_at
        return end - self._turn_started_at

    @property
    def agent_status(self):
        if self.agent.is_shutdown:
            return "shutdown"
        if self._last_agent_status == "interrupted":
            return "interrupted"
        if self.is_busy:
            return "running"
        status = self._last_agent_status
        return dict(status) if isinstance(status, dict) else status

    def can_close_for_capacity(self):
        """Only reclaim a finished runtime with no accepted or background work."""
        return (
            self.accepts_input
            and not self.is_busy
            and self._last_agent_status != "pending_init"
            and not self._active_commands
            and self._input_request is None
            and not self._steer_queue
            and not any(
                tool.background_work_count(True)
                for tool in self.agent.tool_registry.tools()
            )
        )

    async def submit_user_turn(self, text: "str") -> "TurnResult":
        """Enqueue one independent turn and wait for its receipt."""
        _submission_id, future = self.enqueue(text)
        return await future

    async def maybe_invoke(self, event: "typing.Dict[str, object]") -> "bool":
        """Submit a tool notification only while this runtime is idle and open."""
        if self.is_busy or not self.accepts_input:
            return False
        tag = event["type"]
        if not isinstance(tag, str) or not tag.isidentifier():
            raise ValueError("invoke event type must be an identifier")
        payload = {key: value for key, value in event.items() if key != "type"}
        text = (
            f"<{tag}>\n"
            f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
            f"</{tag}>"
        )
        await self.start()
        self.message(text)
        _submission_id, future = self.trigger()
        self._triggered_submission.texts.append(text)
        await future
        return True

    def stop_asap(self) -> "None":
        """Request a cooperative stop; issued model and tool operations finish."""
        if self.agent.is_shutdown:
            raise RuntimeError("agent is shutdown")
        if self.agent.is_running:
            self.agent.stop_asap()
            self._wait_interrupted = True
            if self._input_activity is not None:
                self._input_activity.set()

    def interrupt(self) -> "None":
        """Request a stop and publish interrupted status immediately."""
        self.stop_asap()
        self._last_agent_status = "interrupted"
        self.publish_update("interrupt")

    def enqueue(self, text: "str") -> "typing.Tuple[str, asyncio.Future[TurnResult]]":
        """Schedule an independent user turn without affecting active execution."""
        if not self.accepts_input:
            raise RuntimeError("agent is shutting down")
        queued = _QueuedSubmission([text])
        self._enqueue_queue.append(queued)
        return self._submission_receipt(queued)

    def trigger(self) -> "typing.Union[typing.Tuple[str, asyncio.Future[TurnResult]], None]":
        """Start or restart execution; repeated pending triggers share one run."""
        if not self.accepts_input:
            raise RuntimeError("agent is shutting down")
        if self._active_turn is not None and not self.agent.execution_snapshot().stop_requested:
            return None
        if self._triggered_submission is None:
            self._triggered_submission = _QueuedSubmission([])
        return self._submission_receipt(self._triggered_submission)

    def _submission_receipt(self, queued):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        future.add_done_callback(self._observe_completion)
        queued.futures.append(future)
        # Publish after the input layer attaches display text and turn identity.
        loop.call_soon(self._notify_queue_change)
        return queued.submission_id, future

    def _notify_queue_change(self):
        # The worker can drain and unregister before this notification runs.
        if self.agent.is_shutdown:
            return
        self.publish_update("queue")
        self._wake_worker()

    def message(self, text: "typing.Union[str, AgentMessage, ToolResult]") -> "None":
        """Deliver input to an open runtime's mailbox."""
        if not self.accepts_input:
            raise RuntimeError("agent is shutting down")
        self._steer_queue.append(UserMessage(text) if isinstance(text, str) else text)
        if self._active_turn is not None:
            self.agent.request_follow_up()
        if self._input_activity is not None:
            self._input_activity.set()

    def _deliver_messages(self):
        if not self._steer_queue or self.agent.execution_snapshot().stop_requested:
            return
        self.agent.receive_messages(tuple(self._steer_queue))
        self._steer_queue.clear()
        if self._input_activity is not None and not self._wait_interrupted:
            self._input_activity.clear()

    async def wait_for_activity(self, timeout_ms):
        if self._input_activity is None:
            self._input_activity = asyncio.Event()
        if not self._steer_queue and not self._wait_interrupted:
            waiter = asyncio.create_task(self._input_activity.wait())
            try:
                await asyncio.wait({waiter}, timeout=timeout_ms / 1000.0)
            finally:
                if not waiter.done():
                    waiter.cancel()
                    try:
                        await waiter
                    except asyncio.CancelledError:
                        pass
        if self._wait_interrupted:
            return {"message": "Wait interrupted by new input.", "timed_out": False}
        if self._steer_queue:
            return {"message": "Wait completed.", "timed_out": False}
        return {"message": "Wait timed out.", "timed_out": True}

    async def _run_forever(self) -> "None":
        self._queue_event = asyncio.Event()
        while True:
            pending = await self._next_submission()
            if pending is None:
                break
            queued, texts = pending
            self._current_submission = queued
            try:
                result = await self.agent.run_turn(texts, turn_id=queued.turn_id)
            except TurnInterrupted:
                self._finish_submission_exception(queued, SubmissionInterrupted())
            except Exception as exc:
                self._finish_submission_exception(queued, exc)
            else:
                self._finish_submission_result(queued, result)
            finally:
                self._current_submission = None
                self.publish_update("idle")
        try:
            if self._command_tasks:
                await asyncio.gather(*tuple(self._command_tasks))
            async with self._get_command_lock():
                await self._close_resources()
        finally:
            try:
                self.agent.shutdown()
            finally:
                self._publish(SessionClosedEvent())
                self.agents_manager.unregister_runtime(self.agent.agent_id)

    async def _close_resources(self):
        errors = []
        handlers = [
            lambda: self.agents_manager.shutdown_children(self.agent.agent_id),
        ] + self._close_handlers
        for handler in handlers:
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

    async def _next_submission(self) -> "typing.Union[typing.Tuple[_QueuedSubmission, typing.List[str]], None]":
        while True:
            await self.agent.wait_until_idle()
            if not self.accepts_input:
                return None
            if self._triggered_submission is not None:
                queued, self._triggered_submission = self._triggered_submission, None
                return queued, []
            if self._enqueue_queue:
                queued = self._enqueue_queue.popleft()
                return queued, list(queued.texts)
            if not self._steer_queue:
                continuation = self.agent.tool_registry.runtime_environment.goal_manager.take_continuation()
                if continuation is not None:
                    # Let reply consumers close or submit input before autonomous work.
                    await asyncio.sleep(0)
                    if (
                        not self.accepts_input or self._triggered_submission is not None
                        or self._enqueue_queue or self._steer_queue
                    ):
                        continue
                    self.message(continuation)
                    return _QueuedSubmission([]), []
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
        if isinstance(event, TurnStartedEvent):
            self._turn_started_at = monotonic()
            self._turn_finished_at = None
        elif isinstance(event, (TurnCompletedEvent, TurnFailedEvent, TurnInterruptedEvent)):
            self._turn_finished_at = monotonic()
        queued = self._current_submission
        if queued is not None and event.turn_id == queued.turn_id:
            event = replace(
                event, submission_id=event.submission_id or queued.submission_id
            )
            if isinstance(event, TurnStartedEvent):
                # Triggered input is delivered through steer, not run_turn's texts.
                event = replace(event, user_texts=tuple(queued.texts))
        if isinstance(event, (TurnStartedEvent, CompactStartedEvent)):
            self._wait_interrupted = False
            if self._input_activity is not None and not self._steer_queue:
                self._input_activity.clear()
        elif isinstance(event, TurnInputReadyEvent):
            self._deliver_messages()
        elif isinstance(event, TurnCompletedEvent):
            self._last_agent_status = {"completed": event.output_text}
        elif isinstance(event, TurnFailedEvent):
            self._last_agent_status = {
                "errored": "{0}: {1}".format(event.error_type, event.error),
            }
        elif isinstance(event, TurnInterruptedEvent):
            self._last_agent_status = "interrupted"
        if isinstance(event, TurnStartedEvent):
            self._last_agent_status = "running"
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
        if self.agent.recorded_session_file_path != self._recorded_rollout_path:
            self.publish_state("recording")
        self._publish(event)

    def _publish(self, event: "Event"):
        self._notify_handler(
            lambda event: self.agents_manager.handle_event(self.agent.agent_id, event),
            event,
        )
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
