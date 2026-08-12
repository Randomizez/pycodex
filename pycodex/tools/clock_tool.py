"""Session-scoped periodic clock for waking an idle Agent.

`clock` is a pycodex extension rather than an upstream Codex tool. A configured
clock starts counting only after the Agent completes a reply. Every new turn
cancels the pending countdown, and every successful reply starts a fresh one.
"""

import asyncio
from datetime import datetime
import math

from ..protocol import JSONDict, JSONValue
from .base_tool import BaseTool, ToolContext
import typing


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


def _current_time() -> 'str':
    return datetime.now().astimezone().isoformat(timespec="seconds")


class ClockManager:
    def __init__(self, seconds_per_minute: 'float' = 60.0) -> 'None':
        self._seconds_per_minute = seconds_per_minute
        self._period_m: 'typing.Union[float, None]' = None
        self._timer_task: 'typing.Union[asyncio.Task, None]' = None
        self._generation = 0
        self._notify_hook: 'typing.Union[typing.Callable[[typing.Dict[str, object]], typing.Awaitable[typing.Any]], None]' = None

    def set_notify_hook(
        self,
        callback: 'typing.Union[typing.Callable[[typing.Dict[str, object]], typing.Awaitable[typing.Any]], None]',
    ) -> 'None':
        self._notify_hook = callback

    def set_period(self, value: 'object') -> 'JSONDict':
        self._cancel_pending()
        if value is None:
            self._period_m = None
            return self.snapshot()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("period_m must be a positive finite number or null")
        period_m = float(value)
        if not math.isfinite(period_m) or period_m <= 0:
            raise ValueError("period_m must be a positive finite number or null")
        self._period_m = period_m
        return self.snapshot()

    def snapshot(self) -> 'JSONDict':
        return {
            "enabled": self.enabled,
            "period_m": self._period_m,
        }

    @property
    def enabled(self) -> 'bool':
        return self._period_m is not None

    def turn_started(self) -> 'None':
        self._cancel_pending()

    def arm_after_reply(self) -> 'None':
        self._cancel_pending()
        period_m = self._period_m
        if period_m is None or self._notify_hook is None:
            return
        generation = self._generation
        self._timer_task = asyncio.create_task(
            self._wait_and_notify(generation, period_m)
        )
        self._timer_task.add_done_callback(
            lambda task: None if task.cancelled() else task.exception()
        )

    def cancel(self) -> 'None':
        self.set_period(None)

    def _cancel_pending(self) -> 'None':
        self._generation += 1
        task = self._timer_task
        self._timer_task = None
        if task is not None and not task.done():
            task.cancel()

    async def _wait_and_notify(self, generation: 'int', period_m: 'float') -> 'None':
        try:
            await asyncio.sleep(period_m * self._seconds_per_minute)
        except asyncio.CancelledError:
            return
        if generation != self._generation or period_m != self._period_m:
            return

        self._timer_task = None
        callback = self._notify_hook
        if callback is None:
            return
        try:
            started = await callback(
                {
                    "type": "clock_tick",
                    "period_m": period_m,
                    "current_time": _current_time(),
                }
            )
        except Exception:
            started = False

        if (
            not started
            and generation == self._generation
            and period_m == self._period_m
        ):
            self.arm_after_reply()


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
                "description": (
                    "Positive period in minutes, or null to cancel."
                ),
            },
        },
        "required": ["period_m"],
        "additionalProperties": False,
    }
    output_schema = CLOCK_STATE_OUTPUT_SCHEMA
    supports_parallel = False

    def __init__(self, manager: 'ClockManager') -> 'None':
        self._manager = manager

    async def run(self, context: 'ToolContext', args: 'JSONDict') -> 'JSONValue':
        del context
        if not isinstance(args, dict) or "period_m" not in args:
            raise ValueError("clock requires period_m")
        return self._manager.set_period(args["period_m"])
