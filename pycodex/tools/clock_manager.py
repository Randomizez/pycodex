"""Session-scoped periodic clock and its restorable configuration."""

import asyncio
import math
import typing
from datetime import datetime, timedelta

from ..protocol import JSONDict


def _current_time() -> "str":
    return datetime.now().astimezone().isoformat(timespec="seconds")


class ClockManager:
    def __init__(self, seconds_per_minute: "float" = 60.0) -> "None":
        self._seconds_per_minute = seconds_per_minute
        self._period_m: "typing.Union[float, None]" = None
        self._timer_task: "typing.Union[asyncio.Task, None]" = None
        self._generation = 0
        self._next_fire_at = None
        self._state_listener = None
        self._notify_hook: "typing.Union[typing.Callable[[typing.Dict[str, object]], typing.Awaitable[typing.Any]], None]" = (None)

    def set_notify_hook(
        self,
        callback: "typing.Union[typing.Callable[[typing.Dict[str, object]], typing.Awaitable[typing.Any]], None]",
    ) -> "None":
        self._notify_hook = callback

    def set_state_listener(self, listener):
        self._state_listener = listener

    def _state_changed(self):
        if self._state_listener is not None:
            self._state_listener()

    def background_snapshot(self):
        return dict(
            self.snapshot(),
            kind="clock",
            next_fire_at=self._next_fire_at,
            status=(
                "armed" if self._next_fire_at is not None
                else "paused" if self.enabled else "disabled"
            ),
        )

    @staticmethod
    def _parse_period(value):
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("period_m must be a positive finite number or null")
        period_m = float(value)
        if not math.isfinite(period_m) or period_m <= 0:
            raise ValueError("period_m must be a positive finite number or null")
        return period_m

    def set_period(self, value: "object") -> "JSONDict":
        period_m = self._parse_period(value)
        self._cancel_pending()
        self._period_m = period_m
        self._state_changed()
        return self.snapshot()

    def dump_state(self):
        return {"period_m": self._period_m}

    def validate_state(self, state):
        self._parse_period(state["period_m"])

    def restore_state(self, state):
        period_m = self._parse_period(state["period_m"])
        self._cancel_pending()
        self._period_m = period_m

    def snapshot(self) -> "JSONDict":
        return {
            "enabled": self.enabled,
            "period_m": self._period_m,
        }

    @property
    def enabled(self) -> "bool":
        return self._period_m is not None

    @property
    def armed(self) -> "bool":
        return self._next_fire_at is not None

    def turn_started(self) -> "None":
        self._cancel_pending()
        self._state_changed()

    def arm_after_reply(self) -> "None":
        self._cancel_pending()
        period_m = self._period_m
        if period_m is None or self._notify_hook is None:
            self._state_changed()
            return
        generation = self._generation
        self._next_fire_at = (
            datetime.now().astimezone()
            + timedelta(seconds=period_m * self._seconds_per_minute)
        ).isoformat()
        self._timer_task = asyncio.create_task(
            self._wait_and_notify(generation, period_m)
        )
        self._timer_task.add_done_callback(
            lambda task: None if task.cancelled() else task.exception()
        )
        self._state_changed()

    def cancel(self) -> "None":
        self.set_period(None)

    def shutdown(self) -> "None":
        self._cancel_pending()
        self._notify_hook = None
        self._state_changed()

    def _cancel_pending(self) -> "None":
        self._generation += 1
        task = self._timer_task
        self._timer_task = None
        self._next_fire_at = None
        if task is not None and not task.done():
            task.cancel()

    async def _wait_and_notify(self, generation: "int", period_m: "float") -> "None":
        from ..runtime import SubmissionInterrupted

        try:
            await asyncio.sleep(period_m * self._seconds_per_minute)
        except asyncio.CancelledError:
            return
        if generation != self._generation or period_m != self._period_m:
            return

        self._timer_task = None
        self._next_fire_at = None
        self._state_changed()
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
        except SubmissionInterrupted:
            # The next successful reply will arm the clock after a steer.
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            asyncio.get_running_loop().call_exception_handler(
                {
                    "message": "Clock notification failed",
                    "exception": exc,
                }
            )
            return

        if (
            not started
            and generation == self._generation
            and period_m == self._period_m
        ):
            self.arm_after_reply()
