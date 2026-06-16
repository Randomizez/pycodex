import asyncio

from pycodex import compat
from pycodex.utils.async_bridge import run_async


def test_get_running_loop_compat_rejects_idle_loop() -> "None":
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        try:
            compat._get_running_loop_compat()
        except RuntimeError as exc:
            assert "running" in str(exc)
        else:
            raise AssertionError("idle event loop should not be treated as running")
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_run_async_with_py36_get_running_loop_polyfill(monkeypatch) -> "None":
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        monkeypatch.setattr(
            asyncio,
            "get_running_loop",
            compat._get_running_loop_compat,
        )

        async def answer():
            return "ok"

        assert run_async(answer()) == "ok"
    finally:
        asyncio.set_event_loop(None)
        loop.close()
