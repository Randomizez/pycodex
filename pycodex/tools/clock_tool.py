"""Session-scoped periodic clock for waking an idle Agent.

`clock` is a pycodex extension rather than an upstream Codex tool. A configured
clock starts counting only after the Agent completes a reply. Every new turn
cancels the pending countdown, and every successful reply starts a fresh one.
"""

from ..events import (
    CompactCompletedEvent,
    CompactStartedEvent,
    Event,
    TurnCompletedEvent,
    TurnStartedEvent,
)
from ..protocol import JSONDict, JSONValue
from .base_tool import BaseTool, ToolContext
from .clock_manager import ClockManager

CLOCK_STATE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "enabled": {
            "type": "boolean",
            "description": "Whether the periodic clock is enabled.",
        },
        "period_m": {
            "anyOf": [
                {"type": "number"},
                {"type": "null"},
            ],
            "description": "Configured period in minutes, or null when disabled.",
        },
    },
    "required": ["enabled", "period_m"],
    "additionalProperties": False,
}


class ClockTool(BaseTool):
    name = "clock"
    description = (
        "Set a periodic clock in minutes, or cancel it with null. After each "
        "Agent reply, the clock waits period_m minutes and then sends a "
        "<clock_tick> message to wake the Agent."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "period_m": {
                "anyOf": [
                    {"type": "number"},
                    {"type": "null"},
                ],
                "description": ("Positive period in minutes, or null to cancel."),
            },
        },
        "required": ["period_m"],
        "additionalProperties": False,
    }
    output_schema = CLOCK_STATE_OUTPUT_SCHEMA
    supports_parallel = False

    def __init__(self, manager: "ClockManager") -> "None":
        self._manager = manager

    def bind_runtime(self, runtime) -> "None":
        self._manager.set_notify_hook(runtime.maybe_invoke)

    def handle_agent_event(self, event: "Event") -> "None":
        if isinstance(event, (TurnStartedEvent, CompactStartedEvent)):
            self._manager.turn_started()
        elif isinstance(event, (TurnCompletedEvent, CompactCompletedEvent)):
            self._manager.arm_after_reply()

    def shutdown(self) -> "None":
        self._manager.shutdown()

    def background_work_count(self, after_reply: "bool") -> "int":
        return int(self._manager.enabled if after_reply else self._manager.armed)

    def background_work_snapshot(self):
        return [self._manager.background_snapshot()]

    def set_state_listener(self, listener):
        self._manager.set_state_listener(listener)

    async def run(self, context: "ToolContext", args: "JSONDict") -> "JSONValue":
        del context
        if not isinstance(args, dict) or "period_m" not in args:
            raise ValueError("clock requires period_m")
        return self._manager.set_period(args["period_m"])
