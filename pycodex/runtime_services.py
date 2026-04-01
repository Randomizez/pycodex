from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from .protocol import ConversationItem, TurnResult
from .utils import uuid7_string

if TYPE_CHECKING:
    from .runtime import AgentRuntime

PlanStatus = Literal["pending", "in_progress", "completed"]
PlanListener = Callable[[dict[str, object]], None]
RuntimeBuilder = Callable[
    [str | None, str | None, tuple[ConversationItem, ...], str],
    "AgentRuntime",
]
AsyncJSONHandler = Callable[[dict[str, object]], Awaitable[dict[str, object] | None]]

DEFAULT_AGENT_NICKNAME_CANDIDATES = (
    "Bacon",
    "Descartes",
    "Pascal",
    "Fermat",
    "Huygens",
    "Leibniz",
    "Newton",
    "Halley",
    "Euler",
    "Lagrange",
    "Laplace",
    "Volta",
    "Gauss",
    "Ampere",
    "Faraday",
    "Darwin",
    "Lovelace",
    "Boole",
    "Pasteur",
    "Maxwell",
    "Mendel",
    "Curie",
    "Planck",
    "Tesla",
    "Poincare",
    "Noether",
    "Hilbert",
    "Einstein",
    "Raman",
    "Bohr",
    "Turing",
    "Hubble",
    "Feynman",
    "Franklin",
    "McClintock",
    "Meitner",
    "Herschel",
    "Linnaeus",
    "Wegener",
    "Chandrasekhar",
    "Sagan",
    "Goodall",
    "Carson",
    "Carver",
    "Socrates",
    "Plato",
    "Aristotle",
    "Epicurus",
    "Cicero",
    "Confucius",
    "Mencius",
    "Zeno",
    "Locke",
    "Hume",
    "Kant",
    "Hegel",
    "Kierkegaard",
    "Mill",
    "Nietzsche",
    "Peirce",
    "James",
    "Dewey",
    "Russell",
    "Popper",
    "Sartre",
    "Beauvoir",
    "Arendt",
    "Rawls",
    "Singer",
    "Anscombe",
    "Parfit",
    "Kuhn",
    "Boyle",
    "Hooke",
    "Harvey",
    "Dalton",
    "Helmholtz",
    "Gibbs",
    "Lorentz",
    "Schrodinger",
    "Heisenberg",
    "Pauli",
    "Dirac",
    "Bernoulli",
    "Godel",
    "Nash",
    "Banach",
    "Ramanujan",
    "Erdos",
)


@dataclass(frozen=True, slots=True)
class PlanItem:
    step: str
    status: PlanStatus


class PlanStore:
    def __init__(self) -> None:
        self._explanation: str | None = None
        self._plan: tuple[PlanItem, ...] = ()
        self._listener: PlanListener = lambda _payload: None

    def set_listener(self, listener: PlanListener | None) -> None:
        self._listener = listener or (lambda _payload: None)

    def update(self, explanation: str | None, plan: tuple[PlanItem, ...]) -> None:
        in_progress = sum(1 for item in plan if item.status == "in_progress")
        if in_progress > 1:
            raise ValueError("at most one plan step can be in_progress")
        self._explanation = explanation
        self._plan = plan
        self._listener(
            {
                "explanation": explanation,
                "plan": [
                    {"step": item.step, "status": item.status}
                    for item in self._plan
                ],
            }
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "explanation": self._explanation,
            "plan": [
                {"step": item.step, "status": item.status}
                for item in self._plan
            ],
        }


class RequestUserInputManager:
    def __init__(self) -> None:
        self._handler: AsyncJSONHandler | None = None

    def set_handler(self, handler: AsyncJSONHandler | None) -> None:
        self._handler = handler

    async def request(self, payload: dict[str, object]) -> dict[str, object] | None:
        handler = self._handler
        if handler is None:
            return None
        return await handler(payload)


class RequestPermissionsManager:
    def __init__(self) -> None:
        self._handler: AsyncJSONHandler | None = None

    def set_handler(self, handler: AsyncJSONHandler | None) -> None:
        self._handler = handler

    async def request(self, payload: dict[str, object]) -> dict[str, object] | None:
        handler = self._handler
        if handler is None:
            return None
        return await handler(payload)


@dataclass(slots=True)
class ManagedAgent:
    agent_id: str
    runtime: "AgentRuntime"
    worker_task: asyncio.Task[None]
    nickname: str | None = None
    state: str = "pending_init"
    completed_message: str | None = None
    error_message: str | None = None
    pending_submission_ids: set[str] = field(default_factory=set)


class SubAgentManager:
    def __init__(self) -> None:
        self._runtime_builder: RuntimeBuilder | None = None
        self._agents: dict[str, ManagedAgent] = {}
        self._condition = asyncio.Condition()
        self._available_nicknames: list[str] = []
        self._nickname_random = random.Random()

    def set_runtime_builder(self, builder: RuntimeBuilder | None) -> None:
        self._runtime_builder = builder

    async def spawn_agent(
        self,
        message: str | None,
        items: list[dict[str, object]] | None,
        agent_type: str | None,
        fork_context: bool,
        model: str | None,
        reasoning_effort: str | None,
        history: tuple[ConversationItem, ...],
    ) -> dict[str, object]:
        builder = self._runtime_builder
        if builder is None:
            raise RuntimeError("spawn_agent is unavailable before runtime initialization")

        initial_history = history if fork_context else ()
        agent_id = uuid7_string()
        runtime = builder(model, reasoning_effort, initial_history, agent_id)
        worker_task = asyncio.create_task(runtime.run_forever())
        nickname = self._next_nickname()
        managed = ManagedAgent(
            agent_id=agent_id,
            runtime=runtime,
            worker_task=worker_task,
            nickname=nickname,
        )
        async with self._condition:
            self._agents[agent_id] = managed
            self._condition.notify_all()

        initial_prompt = self._compose_prompt(message, items)
        if initial_prompt:
            await self.send_input(agent_id, initial_prompt, interrupt=False)

        return {
            "agent_id": agent_id,
            "nickname": nickname,
        }

    async def send_input(
        self,
        agent_id: str,
        prompt_text: str,
        interrupt: bool,
    ) -> dict[str, object]:
        managed = self._agents.get(agent_id)
        if managed is None:
            raise RuntimeError(f"unknown agent: {agent_id}")
        if managed.state == "shutdown":
            raise RuntimeError(f"agent is shutdown: {agent_id}")

        submission_id, future = await managed.runtime.enqueue_user_turn(
            prompt_text,
            queue="steer" if interrupt else "enqueue",
        )
        managed.state = "running"
        managed.completed_message = None
        managed.error_message = None
        managed.pending_submission_ids.add(submission_id)
        asyncio.create_task(self._track_submission(managed, submission_id, future))
        async with self._condition:
            self._condition.notify_all()
        return {"submission_id": submission_id}

    async def resume_agent(self, agent_id: str) -> dict[str, object]:
        managed = self._agents.get(agent_id)
        if managed is None:
            return {"status": "not_found"}
        if managed.worker_task.done():
            managed.worker_task = asyncio.create_task(managed.runtime.run_forever())
            managed.state = "pending_init"
            managed.completed_message = None
            managed.error_message = None
        async with self._condition:
            self._condition.notify_all()
        return {"status": self._status_payload(managed)}

    async def close_agent(self, agent_id: str) -> dict[str, object]:
        managed = self._agents.get(agent_id)
        if managed is None:
            return {"status": "not_found"}
        previous_status = self._status_payload(managed)
        if not managed.worker_task.done():
            managed.runtime._agent_loop.interrupt_asap = True
            await managed.runtime.shutdown()
            await managed.worker_task
        managed.state = "shutdown"
        managed.pending_submission_ids.clear()
        async with self._condition:
            self._condition.notify_all()
        return {"status": previous_status}

    def _next_nickname(self) -> str:
        if not self._available_nicknames:
            self._available_nicknames = list(DEFAULT_AGENT_NICKNAME_CANDIDATES)
            self._nickname_random.shuffle(self._available_nicknames)
        return self._available_nicknames.pop()

    async def wait_agents(
        self,
        agent_ids: list[str],
        timeout_ms: int = 30_000,
    ) -> dict[str, object]:
        timeout_seconds = max(timeout_ms, 1) / 1000.0
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds

        while True:
            snapshot = {
                agent_id: self._status_payload(self._agents.get(agent_id))
                for agent_id in agent_ids
            }
            if any(self._is_final_status(status) for status in snapshot.values()):
                return {"status": snapshot, "timed_out": False}

            remaining = deadline - loop.time()
            if remaining <= 0:
                return {"status": {}, "timed_out": True}

            async with self._condition:
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return {"status": {}, "timed_out": True}

    async def _track_submission(
        self,
        managed: ManagedAgent,
        submission_id: str,
        future: asyncio.Future[TurnResult | None],
    ) -> None:
        try:
            result = await future
        except Exception as exc:  # pragma: no cover - background safety
            managed.error_message = f"{type(exc).__name__}: {exc}"
            managed.state = "errored"
        else:
            managed.completed_message = None if result is None else result.output_text
            managed.state = "completed"
        finally:
            managed.pending_submission_ids.discard(submission_id)
            async with self._condition:
                self._condition.notify_all()

    def _compose_prompt(
        self,
        message: str | None,
        items: list[dict[str, object]] | None,
    ) -> str:
        parts: list[str] = []
        if message:
            parts.append(message.strip())
        for item in items or []:
            item_type = str(item.get("type", ""))
            if item_type == "text":
                text = str(item.get("text", "")).strip()
                if text:
                    parts.append(text)
            elif item_type == "image":
                image_url = str(item.get("image_url", "")).strip()
                if image_url:
                    parts.append(f"[image] {image_url}")
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n\n".join(part for part in parts if part)

    def _status_payload(self, managed: ManagedAgent | None) -> object:
        if managed is None:
            return "not_found"
        if managed.error_message is not None:
            return {"errored": managed.error_message}
        if managed.state == "completed":
            return {"completed": managed.completed_message}
        if managed.state in {"pending_init", "running", "shutdown"}:
            return managed.state
        return managed.state

    def _is_final_status(self, status: object) -> bool:
        if isinstance(status, str):
            return status in {"shutdown", "not_found"}
        if isinstance(status, dict):
            return "completed" in status or "errored" in status
        return False


class RuntimeEnvironment:
    def __init__(self) -> None:
        self.plan_store = PlanStore()
        self.subagent_manager = SubAgentManager()
        self.request_user_input_manager = RequestUserInputManager()
        self.request_permissions_manager = RequestPermissionsManager()

    def configure_runtime_builder(self, builder: RuntimeBuilder | None) -> None:
        self.subagent_manager.set_runtime_builder(builder)


_RUNTIME_ENV = RuntimeEnvironment()


def get_runtime_environment() -> RuntimeEnvironment:
    return _RUNTIME_ENV
