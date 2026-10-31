import asyncio
import typing
from dataclasses import dataclass
from typing import Awaitable, Callable

from ..events import CompactStartedEvent, TerminalEvent, TurnStartedEvent
from ..utils.compat import Literal
from .clock_manager import ClockManager
from .goal_manager import GoalManager

PlanStatus = Literal["pending", "in_progress", "completed"]
PlanListener = Callable[[typing.Dict[str, object]], None]
AsyncJSONHandler = Callable[
    [typing.Dict[str, object]], Awaitable[typing.Union[typing.Dict[str, object], None]]
]


@dataclass(
    frozen=True,
)
class PlanItem:
    step: "str"
    status: "PlanStatus"


class PlanStore:
    def __init__(self) -> "None":
        self._explanation: "typing.Union[str, None]" = None
        self._plan: "typing.Tuple[PlanItem, ...]" = ()
        self._listener: "PlanListener" = lambda _payload: None

    def set_listener(self, listener: "typing.Union[PlanListener, None]") -> "None":
        self._listener = listener or (lambda _payload: None)

    def update(
        self,
        explanation: "typing.Union[str, None]",
        plan: "typing.Tuple[PlanItem, ...]",
    ) -> "None":
        in_progress = sum(1 for item in plan if item.status == "in_progress")
        if in_progress > 1:
            raise ValueError("at most one plan step can be in_progress")
        self._explanation = explanation
        self._plan = plan
        self._listener(
            {
                "explanation": explanation,
                "plan": [
                    {"step": item.step, "status": item.status} for item in self._plan
                ],
            }
        )

    def snapshot(self) -> "typing.Dict[str, object]":
        return {
            "explanation": self._explanation,
            "plan": [{"step": item.step, "status": item.status} for item in self._plan],
        }


class RequestUserInputManager:
    def __init__(self) -> "None":
        self._handler: "typing.Union[AsyncJSONHandler, None]" = None

    def set_handler(self, handler: "typing.Union[AsyncJSONHandler, None]") -> "None":
        self._handler = handler

    async def request(
        self, payload: "typing.Dict[str, object]"
    ) -> "typing.Union[typing.Dict[str, object], None]":
        handler = self._handler
        if handler is None:
            return None
        return await handler(payload)


class RequestPermissionsManager:
    def __init__(self) -> "None":
        self._handler: "typing.Union[AsyncJSONHandler, None]" = None

    def set_handler(self, handler: "typing.Union[AsyncJSONHandler, None]") -> "None":
        self._handler = handler

    async def request(
        self, payload: "typing.Dict[str, object]"
    ) -> "typing.Union[typing.Dict[str, object], None]":
        handler = self._handler
        if handler is None:
            return None
        return await handler(payload)


class AgentRuntimeEnvironment:
    def __init__(self) -> "None":
        self.clock_manager = ClockManager()
        self.goal_manager = GoalManager()
        self.plan_store = PlanStore()
        self.request_user_input_manager = RequestUserInputManager()
        self.request_permissions_manager = RequestPermissionsManager()
        self._resources = {}
        self._restart_handlers = []
        self._close_handlers = []
        self._event_handlers = []
        self._bindings = None

    def register_resource(self, name, manager, restart=None, close=None, observer=None):
        """Register one shared manager, independently of its callable tools."""
        if name in self._resources or manager in self._resources.values():
            raise ValueError("resource already registered: " + name)
        self._resources[name] = manager
        for handler, handlers in (
            (restart, self._restart_handlers), (close, self._close_handlers),
            (observer, self._event_handlers),
        ):
            if handler is not None:
                handlers.append(handler)

    def bind(self, invoke, output, changed, request_input, request_permissions):
        self._bindings = (invoke, output, lambda: changed("background_work"))
        self.request_user_input_manager.set_handler(request_input)
        self.request_permissions_manager.set_handler(request_permissions)
        self.plan_store.set_listener(lambda _plan: changed("plan"))
        for manager in self._resources.values():
            manager.bind(*self._bindings)

    def restart(self):
        for restart in self._restart_handlers:
            restart()
        if self._bindings is not None:
            for manager in self._resources.values():
                manager.bind(*self._bindings)

    def set_state_listener(self, listener):
        for manager in self._resources.values():
            manager.set_state_listener(listener)

    def background_work_count(self, after_reply):
        return sum(manager.background_work_count(after_reply) for manager in self._resources.values())

    def background_work_snapshot(self):
        return [
            dict(item, tool=name)
            for name, manager in self._resources.items()
            for item in manager.background_work_snapshot()
        ]

    def event_observers(self, event):
        return self._event_handlers if isinstance(
            event, (TerminalEvent, TurnStartedEvent, CompactStartedEvent),
        ) else []

    def dump_state(self):
        managers = {"clock": self.clock_manager.dump_state()}
        goal = self.goal_manager.dump_state()
        if goal is not None:
            managers["goal"] = goal
        return {
            "version": 1,
            "managers": managers,
        }

    def validate_state(self, state):
        if state is not None:
            if state["version"] != 1:
                raise ValueError("unsupported environment state version")
            self.clock_manager.validate_state(state["managers"]["clock"])
            self.goal_manager.validate_state(state["managers"].get("goal"))

    def restore_state(self, state):
        self.validate_state(state)
        self.clock_manager.restore_state(
            {"period_m": None} if state is None else state["managers"]["clock"]
        )
        self.goal_manager.restore_state(
            None if state is None else state["managers"].get("goal")
        )

    def start(self):
        self.clock_manager.arm_after_reply()
        self.goal_manager.start()

    def handle_agent_event(self, event):
        self.goal_manager.handle_agent_event(event)

    def tool_completed(self):
        self.goal_manager.account()

    def follow_up_messages(self):
        return self.goal_manager.follow_up_messages()

    def take_continuation(self):
        return self.goal_manager.take_continuation()

    def shutdown(self, save_state):
        try:
            self.goal_manager.shutdown()
            save_state()
        finally:
            for manager in self._resources.values():
                manager.shutdown()

    async def close(self):
        await close_resources(self._close_handlers)


async def close_resources(handlers):
    errors = []
    for handler in handlers:
        try:
            await handler()
        except Exception as exc:
            errors.append(exc)
    if errors:
        for error in errors[1:]:
            asyncio.get_running_loop().call_exception_handler(
                {"message": "Session close handler failed", "exception": error}
            )
        raise errors[0]


def create_agent_runtime_environment() -> "AgentRuntimeEnvironment":
    return AgentRuntimeEnvironment()
