import asyncio
import threading
import typing

from pycodex.events import SessionStateEvent
from pycodex.runtime import AgentRuntime

from .feishu_card import PycodexCard
from .feishu_connection import (
    FeishuCardActionListener as _FeishuCardActionListener,
    feishu_listener as _feishu_listener,
    release_feishu_listener as _release_feishu_listener,
)

CARD_UPDATE_FLUSH_INTERVAL_SECONDS = 0.5


class PycodexRuntimeLink:
    def __init__(
        self,
        runtime: "AgentRuntime",
        target: str,
        loop=None,
        card: "typing.Union[PycodexCard, None]" = None,
    ) -> None:
        self.runtime = runtime
        self.target = target
        self.loop = loop
        self.card = card or PycodexCard.from_env()
        self.session_key = None
        self.message_id = None
        self._listener = None
        self._frontend_id = None
        self._detached = False
        self._update_thread = None
        self._update_stop = threading.Event()
        self._update_lock = threading.Lock()
        self._update_pending = False

    async def start_async(self) -> "PycodexRuntimeLink":
        self.loop = asyncio.get_event_loop()
        self.card.apply_event(SessionStateEvent("attach", self.runtime.snapshot()))
        try:
            await self.loop.run_in_executor(None, self.start)
        except Exception:
            await self.loop.run_in_executor(None, self.stop)
            raise
        self._frontend_id = self.runtime.attach(self._handle_runtime_event)
        return self

    def start(self) -> "PycodexRuntimeLink":
        if self.loop is None:
            raise RuntimeError("PycodexRuntimeLink.start_async() is required")
        self.card.send(self.target)
        self.session_key = self.card.session_key
        self.message_id = self.card.message_id
        self._listener = _feishu_listener(self.card)
        self._listener.register(self)
        self._listener.start()
        self._start_update_thread()
        return self

    def stop(self) -> None:
        self._detach_frontend()
        if self._listener is not None:
            _release_feishu_listener(self._listener, self)
            self._listener = None
        self._stop_update_thread()

    def detach(self) -> None:
        self._detached = True
        self._detach_frontend()
        if self._listener is not None:
            _release_feishu_listener(self._listener, self)
            self._listener = None
        self.card.detach()
        self._safe_update_card()
        self._stop_update_thread()

    def _detach_frontend(self) -> None:
        if self._frontend_id is not None:
            self.runtime.detach(self._frontend_id)
            self._frontend_id = None

    def _handle_runtime_event(self, event) -> None:
        if self._detached:
            return
        self.card.apply_event(event)
        self._update_card()

    async def _submit(
        self, action: "typing.Dict[str, object]"
    ) -> "typing.Dict[str, object]":
        if self._detached:
            return {"toast": {"type": "warning", "content": "pycodex is detached."}}
        if action.get("action") == "history":
            try:
                self.card.show_history(action.get("history_index"))
            except ValueError as exc:
                return {"toast": {"type": "warning", "content": str(exc)}}
            self._update_card()
            return {}
        if action.get("action") != "send":
            return {
                "toast": {
                    "type": "info",
                    "content": "Unsupported action: {0}".format(action.get("action")),
                }
            }
        prompt = str(action.get("prompt") or "")
        future = asyncio.run_coroutine_threadsafe(
            self.runtime.submit_input(
                prompt, sender=str(action.get("sender") or "feishu")
            ),
            self.loop,
        )
        try:
            await asyncio.wrap_future(future)
        except (ValueError, RuntimeError) as exc:
            return {"toast": {"type": "warning", "content": str(exc)}}
        return {"toast": {"type": "info", "content": "Input submitted."}}

    def _update_card(self) -> None:
        with self._update_lock:
            self._update_pending = True

    def _safe_update_card(self) -> None:
        try:
            self.card.update()
        except Exception as e:
            print(e)

    def _start_update_thread(self) -> None:
        if self._update_thread is not None and self._update_thread.is_alive():
            return
        self._update_stop.clear()
        self._update_thread = threading.Thread(
            target=self._run_update_loop,
            name="pycodex-feishu-card",
        )
        self._update_thread.daemon = True
        self._update_thread.start()

    def _stop_update_thread(self) -> None:
        self._update_stop.set()
        thread = self._update_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._update_thread = None

    def _run_update_loop(self) -> None:
        while not self._update_stop.wait(CARD_UPDATE_FLUSH_INTERVAL_SECONDS):
            with self._update_lock:
                if not self._update_pending:
                    continue
                self._update_pending = False
            self._safe_update_card()
