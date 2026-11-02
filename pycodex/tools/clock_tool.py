"""Periodic wake-ups and interruptible waits in the clock namespace.

`set_period` is a local extension; `sleep` follows upstream's active-turn wait.
"""

from time import monotonic

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


class SetPeriodTool(BaseTool):
    exec_only = False
    name = "set_period"
    namespace = "clock"
    namespace_description = "Tools for waiting on time and scheduling periodic wake-ups."
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

    async def run(self, context: "ToolContext", args: "JSONDict") -> "JSONValue":
        del context
        if not isinstance(args, dict) or "period_m" not in args:
            raise ValueError("set_period requires period_m")
        return self._manager.set_period(args["period_m"])


class SleepTool(BaseTool):
    name = "sleep"
    namespace = "clock"
    namespace_description = "Tools for waiting on time and scheduling periodic wake-ups."
    exec_only = False
    description = (
        "Pause execution for a specified duration. The sleep ends early when new "
        "input arrives for the active turn. Returns the elapsed wall-clock time."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "duration_ms": {
                "type": "number",
                "description": (
                    "How long to sleep in milliseconds. Must be between 1 and 43200000."
                ),
            },
        },
        "required": ["duration_ms"],
        "additionalProperties": False,
    }
    supports_parallel = False

    def __init__(self, agents_manager):
        self._agents_manager = agents_manager

    async def run(self, context, args):
        if set(args) != {"duration_ms"}:
            raise ValueError("sleep requires only duration_ms")
        duration_ms = args["duration_ms"]
        if type(duration_ms) is not int:
            raise ValueError("duration_ms must be an integer")
        if not 1 <= duration_ms <= 43200000:
            raise ValueError("duration_ms must be between 1 and 43200000")
        started = monotonic()
        result = await self._agents_manager.wait_for_activity(
            context.agent_id, duration_ms,
        )
        message = (
            "Sleep completed." if result["timed_out"]
            else "Sleep interrupted by new input."
        )
        return "Wall time: {0:.4f} seconds\n{1}".format(
            monotonic() - started, message,
        )
