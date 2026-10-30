import typing
from dataclasses import dataclass
from typing import Awaitable, Callable

from ..utils.compat import Literal
from .clock_manager import ClockManager

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
        self.plan_store = PlanStore()
        self.request_user_input_manager = RequestUserInputManager()
        self.request_permissions_manager = RequestPermissionsManager()

    def dump_state(self):
        return {
            "version": 1,
            "managers": {"clock": self.clock_manager.dump_state()},
        }

    def validate_state(self, state):
        if state is not None:
            if state["version"] != 1:
                raise ValueError("unsupported environment state version")
            self.clock_manager.validate_state(state["managers"]["clock"])

    def restore_state(self, state):
        self.validate_state(state)
        self.clock_manager.restore_state(
            {"period_m": None} if state is None else state["managers"]["clock"]
        )

    def start(self):
        self.clock_manager.arm_after_reply()


def create_agent_runtime_environment() -> "AgentRuntimeEnvironment":
    return AgentRuntimeEnvironment()
