"""Feishu SDK listeners and their thread and event-loop lifecycle."""

import asyncio
import threading
import typing

from .feishu_card import PycodexCard

if typing.TYPE_CHECKING:
    from .feishu_link import PycodexRuntimeLink

_LISTENER = None
_LISTENER_LOCK = threading.Lock()


def feishu_listener(card: PycodexCard) -> "FeishuCardActionListener":
    global _LISTENER
    with _LISTENER_LOCK:
        if _LISTENER is None:
            _LISTENER = FeishuCardActionListener(card)
        else:
            _LISTENER.assert_compatible(card)
        return _LISTENER


def release_feishu_listener(
    listener: "FeishuCardActionListener",
    link: "PycodexRuntimeLink",
) -> None:
    global _LISTENER
    with _LISTENER_LOCK:
        listener.unregister(link)
        if listener.empty():
            listener.stop()
            if _LISTENER is listener:
                _LISTENER = None


class FeishuCardActionListener:
    def __init__(self, card: PycodexCard) -> None:
        self.app_id = card.app_id
        self.app_secret = card.app_secret
        self.domain = card.domain
        self.encrypt_key = card.encrypt_key
        self.verification_token = card.verification_token
        self._links = {}
        self._links_lock = threading.Lock()
        self._async_thread = _AsyncLoopThread()
        self._thread = None
        self._client = None
        self._client_loop = None
        self._stop_requested = threading.Event()

    def assert_compatible(self, card: PycodexCard) -> None:
        if self.app_id != card.app_id or self.domain != card.domain:
            raise RuntimeError("Feishu listener is already running for another app")

    def start(self) -> None:
        if not self.app_id or not self.app_secret:
            raise RuntimeError("FEISHU_APP_ID and FEISHU_APP_SECRET are required")
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_requested.clear()
        self._async_thread.start()
        self._thread = threading.Thread(
            target=self._listen,
            name="pycodex-feishu-link",
        )
        self._thread.daemon = True
        self._thread.start()

    def register(self, link: "PycodexRuntimeLink") -> None:
        message_id = str(getattr(link, "message_id", "") or "").strip()
        if not message_id:
            raise RuntimeError("Feishu linked card message_id is required")
        with self._links_lock:
            self._links[message_id] = link

    def unregister(self, link: "PycodexRuntimeLink") -> None:
        message_id = str(getattr(link, "message_id", "") or "").strip()
        with self._links_lock:
            if message_id and self._links.get(message_id) is link:
                del self._links[message_id]
                return
            stale_ids = [
                registered_id
                for registered_id, registered_link in self._links.items()
                if registered_link is link
            ]
            for registered_id in stale_ids:
                del self._links[registered_id]

    def empty(self) -> bool:
        with self._links_lock:
            return not self._links

    def stop(self) -> None:
        self._stop_requested.set()
        client = self._client
        loop = self._client_loop
        if loop is not None:
            _stop_lark_client(client, loop)
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
        self._async_thread.stop()

    def _listen(self) -> None:
        try:
            import lark_oapi as lark
            import lark_oapi.ws as lark_ws
            from lark_oapi.event.callback.model.p2_card_action_trigger import (
                P2CardActionTriggerResponse,
            )
        except ImportError as exc:
            raise RuntimeError(
                "lark-oapi is required for Feishu long-connection mode"
            ) from exc

        def on_card_action(event):
            return P2CardActionTriggerResponse(self._handle_card_action(event))

        handler = (
            lark.EventDispatcherHandler.builder(
                self.encrypt_key or "",
                self.verification_token or "",
                lark.LogLevel.INFO,
            )
            .register_p2_card_action_trigger(on_card_action)
            .build()
        )
        import lark_oapi.ws.client as lark_ws_client

        client = None
        loop = asyncio.new_event_loop()
        self._client_loop = loop
        asyncio.set_event_loop(loop)
        lark_ws_client.loop = loop
        try:
            if self._stop_requested.is_set():
                return
            client = lark_ws.Client(
                self.app_id,
                self.app_secret,
                event_handler=handler,
                domain=self.domain,
            )
            self._client = client
            if self._stop_requested.is_set():
                return
            client.start()
        except RuntimeError as exc:
            stopped = "Event loop stopped before Future completed" in str(exc)
            if not self._stop_requested.is_set() or not stopped:
                raise
        finally:
            if self._client is client:
                self._client = None
            if self._client_loop is loop:
                self._client_loop = None
            _close_loop(loop)
            asyncio.set_event_loop(None)

    def _handle_card_action(self, event) -> "typing.Dict[str, object]":
        try:
            message_id = _event_message_id(event)
            link = self._resolve_link(message_id)
            if link is None or link._detached:
                return {
                    "toast": {
                        "type": "warning",
                        "content": "pycodex is detached.",
                    }
                }
            action = link.card.parse_action(event)
            return self._async_thread.run(link._submit(action))
        except Exception as exc:
            return {
                "toast": {
                    "type": "error",
                    "content": "pycodex error: {0}".format(exc),
                }
            }

    def _resolve_link(self, message_id):
        message_id = str(message_id or "").strip()
        with self._links_lock:
            if message_id:
                return self._links.get(message_id)
            if len(self._links) == 1:
                return next(iter(self._links.values()))
            return None


def _event_message_id(event) -> "typing.Union[str, None]":
    event_data = getattr(event, "event", None)
    context = getattr(event_data, "context", None)
    for name in ("open_message_id", "message_id"):
        value = str(getattr(context, name, None) or "").strip()
        if value:
            return value
    return None


def _stop_lark_client(client, loop) -> None:
    if loop.is_closed():
        return
    disconnect = getattr(client, "_disconnect", None) if client is not None else None
    if loop.is_running():
        future = asyncio.run_coroutine_threadsafe(
            _disconnect_then_stop(disconnect, loop),
            loop,
        )
        try:
            future.result(timeout=2.0)
        except Exception as exc:
            future.cancel()
            print(exc)
            loop.call_soon_threadsafe(loop.stop)
        return
    if callable(disconnect):
        try:
            loop.run_until_complete(disconnect())
        except Exception as exc:
            print(exc)


async def _disconnect_then_stop(disconnect, loop) -> None:
    try:
        if callable(disconnect):
            await disconnect()
    except Exception as exc:
        print(exc)
    finally:
        loop.stop()


def _close_loop(loop) -> None:
    if loop.is_closed():
        return
    tasks = list(asyncio.all_tasks(loop))
    for task in tasks:
        task.cancel()
    if tasks:
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()


class _AsyncLoopThread:
    def __init__(self) -> None:
        self.loop = None
        self.thread = None
        self._started = threading.Event()

    def start(self) -> None:
        if self.loop is not None and self.loop.is_running():
            return
        self.loop = asyncio.new_event_loop()
        self._started.clear()
        self.thread = threading.Thread(
            target=self._run_loop,
            name="pycodex-feishu-asyncio",
        )
        self.thread.daemon = True
        self.thread.start()
        self._started.wait()

    def run(self, coro):
        if self.loop is None or not self.loop.is_running():
            raise RuntimeError("Feishu asyncio loop is not running")
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def stop(self) -> None:
        loop = self.loop
        if loop is None:
            return
        if loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if self.thread is not None:
            self.thread.join()
        loop.close()
        self.loop = None
        self.thread = None

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self._started.set()
        self.loop.run_forever()
