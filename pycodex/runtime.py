
import asyncio
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .agent import AgentLoop, EventHandler, NOOP_EVENT_HANDLER, TurnInterrupted
from .compat import Literal
from .protocol import AgentEvent, Operation, ShutdownOp, Submission, TurnResult, UserTurnOp
from .utils import uuid7_string
import typing

if TYPE_CHECKING:
    from .runtime_services import RuntimeEnvironment


@dataclass
class _QueuedSubmission:
    submission: 'Submission'
    turn_id: 'str'
    futures: 'typing.List[asyncio.Future[typing.Union[TurnResult, None]]]'


class AgentRuntime:
    """Thin outer queue that mirrors the Rust `submission_loop` shape."""

    def __init__(self, agent_loop: 'AgentLoop', runtime_environment: 'typing.Union[RuntimeEnvironment, None]' = None) -> 'None':
        self._agent_loop = agent_loop
        self.runtime_environment = runtime_environment
        self._enqueue_queue: 'deque[_QueuedSubmission]' = deque()
        self._steer_queue: 'deque[_QueuedSubmission]' = deque()
        self._queue_lock = asyncio.Lock()
        self._queue_event = asyncio.Event()
        self._current_submission: 'typing.Union[_QueuedSubmission, None]' = None
        self._current_task: 'typing.Union[asyncio.Task[TurnResult], None]' = None
        self._event_handler = NOOP_EVENT_HANDLER
        self._agent_loop.set_event_handler(self._handle_agent_event)

    def set_event_handler(self, event_handler: 'EventHandler' = NOOP_EVENT_HANDLER) -> 'None':
        self._event_handler = event_handler

    async def submit_user_turn(self, text: 'str') -> 'TurnResult':
        _submission_id, future = await self.enqueue_user_turn(text, queue="enqueue")
        result = await future
        assert result is not None
        return result

    async def enqueue_user_turn(
        self,
        text: 'str',
        queue: 'Literal["enqueue", "steer"]' = "enqueue",
    ) -> 'typing.Tuple[str, asyncio.Future[typing.Union[TurnResult, None]]]':
        future: 'asyncio.Future[typing.Union[TurnResult, None]]' = asyncio.get_running_loop().create_future()
        return await self._enqueue_user_turn_to_queue(
            text,
            future,
            queue=queue,
        )

    async def shutdown(self) -> 'None':
        submission = Submission(id=uuid7_string(), op=ShutdownOp())
        future: 'asyncio.Future[typing.Union[TurnResult, None]]' = asyncio.get_running_loop().create_future()
        self._enqueue_queue.append(
            _QueuedSubmission(
                submission=submission,
                turn_id=submission.id,
                futures=[future],
            )
        )
        self._queue_event.set()
        await future

    async def run_forever(self) -> 'None':
        while True:
            queued = await self._next_submission()
            submission = queued.submission
            self._current_submission = queued
            try:
                if isinstance(submission.op, UserTurnOp):
                    self._current_task = asyncio.create_task(
                        self._agent_loop.run_turn(
                            list(submission.op.texts),
                            turn_id=queued.turn_id,
                        )
                    )
                    try:
                        result = await self._current_task
                    except TurnInterrupted:
                        self._finish_submission_exception(
                            queued,
                            RuntimeError("submission interrupted"),
                        )
                        continue
                    except asyncio.CancelledError:
                        self._finish_submission_exception(
                            queued,
                            RuntimeError("submission interrupted"),
                        )
                        continue
                    self._finish_submission_result(queued, result)
                    continue

                if isinstance(submission.op, ShutdownOp):
                    self._finish_submission_result(queued, None)
                    break

                self._finish_submission_exception(
                    queued,
                    RuntimeError(f"unsupported operation: {type(submission.op).__name__}"),
                )
            except Exception as exc:  # pragma: no cover - defensive wrapper
                self._finish_submission_exception(queued, exc)
            finally:
                self._current_task = None
                self._current_submission = None

    @staticmethod
    def operation_name(op: 'Operation') -> 'str':
        if isinstance(op, UserTurnOp):
            return "user_turn"
        if isinstance(op, ShutdownOp):
            return "shutdown"
        return type(op).__name__

    async def _enqueue_user_turn_to_queue(
        self,
        text: 'str',
        future: 'asyncio.Future[typing.Union[TurnResult, None]]',
        queue: 'Literal["enqueue", "steer"]',
    ) -> 'typing.Tuple[str, asyncio.Future[typing.Union[TurnResult, None]]]':
        if queue == "steer" and self._has_active_turn():
            self._agent_loop.interrupt_asap = True

        async with self._queue_lock:
            if queue == "steer" and self._steer_queue:
                queued = self._steer_queue[-1]
                queued.submission.op.texts.append(text)
                queued.futures.append(future)
                return queued.submission.id, future

            submission = Submission(id=uuid7_string(), op=UserTurnOp(texts=[text]))
            current = self._current_submission if self._has_active_turn() else None
            turn_id = (
                current.turn_id if queue == "steer" and current is not None else submission.id
            )
            queued = _QueuedSubmission(
                submission=submission,
                turn_id=turn_id,
                futures=[future],
            )
            if queue == "steer":
                self._steer_queue.append(queued)
            else:
                self._enqueue_queue.append(queued)
            self._queue_event.set()
            return submission.id, future

    async def _next_submission(self) -> '_QueuedSubmission':
        while True:
            async with self._queue_lock:
                queued: 'typing.Union[_QueuedSubmission, None]' = None
                if self._steer_queue:
                    queued = self._steer_queue.popleft()
                elif self._enqueue_queue:
                    queued = self._enqueue_queue.popleft()
                if queued is not None:
                    if not self._steer_queue and not self._enqueue_queue:
                        self._queue_event.clear()
                    return queued
                self._queue_event.clear()
            await self._queue_event.wait()

    @staticmethod
    def _finish_submission_result(
        queued: '_QueuedSubmission',
        result: 'typing.Union[TurnResult, None]',
    ) -> 'None':
        for future in queued.futures:
            if not future.done():
                future.set_result(result)

    @staticmethod
    def _finish_submission_exception(
        queued: '_QueuedSubmission',
        exc: 'Exception',
    ) -> 'None':
        for future in queued.futures:
            if not future.done():
                future.set_exception(exc)

    def _has_active_turn(self) -> 'bool':
        current_task = self._current_task
        return current_task is not None and not current_task.done()

    def _handle_agent_event(self, event: 'AgentEvent') -> 'None':
        queued = self._current_submission
        if queued is None:
            self._event_handler(event)
            return
        payload = dict(event.payload)
        payload.setdefault("submission_id", queued.submission.id)
        payload.setdefault("turn_id", queued.turn_id)
        self._event_handler(
            AgentEvent(kind=event.kind, turn_id=event.turn_id, payload=payload)
        )
