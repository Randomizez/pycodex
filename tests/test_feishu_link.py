import asyncio
import sys
import threading
import types

from pycodex.feishu_card import PycodexCard
import pycodex.feishu_link as feishu_link
from pycodex.feishu_link import _FeishuCardActionListener, _release_feishu_listener


class _Object:
    def __init__(self, **values):
        self.__dict__.update(values)


def _event(message_id):
    return _Object(event=_Object(context=_Object(open_message_id=message_id)))


def _install_fake_lark(monkeypatch, client_class, client_module):
    lark_module = types.ModuleType("lark_oapi")
    ws_module = types.ModuleType("lark_oapi.ws")
    event_module = types.ModuleType("lark_oapi.event")
    callback_module = types.ModuleType("lark_oapi.event.callback")
    model_module = types.ModuleType("lark_oapi.event.callback.model")
    card_action_module = types.ModuleType(
        "lark_oapi.event.callback.model.p2_card_action_trigger"
    )

    class _Builder:
        def register_p2_card_action_trigger(self, callback):
            self.callback = callback
            return self

        def build(self):
            return _Object(callback=getattr(self, "callback", None))

    class _EventDispatcherHandler:
        @staticmethod
        def builder(*args):
            return _Builder()

    class _LogLevel:
        INFO = "info"

    class _P2CardActionTriggerResponse:
        def __init__(self, payload):
            self.payload = payload

    lark_module.EventDispatcherHandler = _EventDispatcherHandler
    lark_module.LogLevel = _LogLevel
    lark_module.ws = ws_module
    ws_module.Client = client_class
    ws_module.client = client_module
    client_module.Client = client_class
    card_action_module.P2CardActionTriggerResponse = _P2CardActionTriggerResponse

    for module in (
        lark_module,
        ws_module,
        client_module,
        event_module,
        callback_module,
        model_module,
        card_action_module,
    ):
        module.__path__ = []

    monkeypatch.setitem(sys.modules, "lark_oapi", lark_module)
    monkeypatch.setitem(sys.modules, "lark_oapi.ws", ws_module)
    monkeypatch.setitem(sys.modules, "lark_oapi.ws.client", client_module)
    monkeypatch.setitem(sys.modules, "lark_oapi.event", event_module)
    monkeypatch.setitem(sys.modules, "lark_oapi.event.callback", callback_module)
    monkeypatch.setitem(sys.modules, "lark_oapi.event.callback.model", model_module)
    monkeypatch.setitem(
        sys.modules,
        "lark_oapi.event.callback.model.p2_card_action_trigger",
        card_action_module,
    )


async def _noop():
    return None


async def _forever():
    while True:
        await asyncio.sleep(3600)


def test_feishu_listener_tracks_active_links_by_message_id() -> None:
    listener = _FeishuCardActionListener(PycodexCard(app_id="app", app_secret="secret"))
    first = _Object(message_id="om_first")
    second = _Object(message_id="om_second")

    listener.register(first)
    assert listener._resolve_link("om_first") is first
    listener.register(second)
    assert listener._resolve_link("om_first") is first
    assert listener._resolve_link("om_second") is second

    listener.unregister(first)
    assert listener._resolve_link("om_first") is None
    assert listener._resolve_link("om_second") is second

    listener.unregister(second)
    assert listener.empty()


def test_feishu_listener_unregisters_link_if_message_id_changes() -> None:
    listener = _FeishuCardActionListener(PycodexCard(app_id="app", app_secret="secret"))
    link = _Object(message_id="om_first")

    listener.register(link)
    link.message_id = "om_second"
    listener.unregister(link)

    assert listener.empty()


def test_feishu_listener_stops_when_last_link_is_released(monkeypatch) -> None:
    listener = _FeishuCardActionListener(PycodexCard(app_id="app", app_secret="secret"))
    link = _Object(message_id="om_first")
    stopped = []
    listener.stop = lambda: stopped.append(True)
    listener.register(link)
    monkeypatch.setattr(feishu_link, "_LISTENER", listener)

    _release_feishu_listener(listener, link)

    assert listener.empty()
    assert stopped == [True]
    assert feishu_link._LISTENER is None


def test_feishu_listener_does_not_stop_until_all_links_are_released(monkeypatch) -> None:
    listener = _FeishuCardActionListener(PycodexCard(app_id="app", app_secret="secret"))
    first = _Object(message_id="om_first")
    second = _Object(message_id="om_second")
    stopped = []
    listener.stop = lambda: stopped.append(True)
    listener.register(first)
    listener.register(second)
    monkeypatch.setattr(feishu_link, "_LISTENER", listener)

    _release_feishu_listener(listener, first)

    assert stopped == []
    assert feishu_link._LISTENER is listener
    assert listener._resolve_link("om_second") is second

    _release_feishu_listener(listener, second)

    assert stopped == [True]
    assert feishu_link._LISTENER is None


def test_feishu_listener_relink_creates_fresh_listener(monkeypatch) -> None:
    first = _FeishuCardActionListener(PycodexCard(app_id="app", app_secret="secret"))
    link = _Object(message_id="om_first")
    first.stop = lambda: None
    first.register(link)
    monkeypatch.setattr(feishu_link, "_LISTENER", first)

    _release_feishu_listener(first, link)
    second = feishu_link._feishu_listener(PycodexCard(app_id="app", app_secret="secret"))

    assert second is not first
    assert feishu_link._LISTENER is second


def test_feishu_listener_routes_card_action_by_message_id(monkeypatch) -> None:
    listener = _FeishuCardActionListener(PycodexCard(app_id="app", app_secret="secret"))
    first = _Object(
        message_id="om_first",
        _detached=False,
        card=_Object(parse_action=lambda event: {"action": "send", "prompt": "first"}),
    )
    second = _Object(
        message_id="om_second",
        _detached=False,
        card=_Object(parse_action=lambda event: {"action": "send", "prompt": "second"}),
    )
    submitted = []

    async def submit(action):
        submitted.append(action["prompt"])
        return {"toast": {"type": "info", "content": action["prompt"]}}

    first._submit = submit
    second._submit = submit

    def run_coro(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    listener._async_thread.run = run_coro

    listener.register(first)
    listener.register(second)

    result = listener._handle_card_action(_event("om_second"))

    assert submitted == ["second"]
    assert result["toast"]["content"] == "second"


def test_feishu_listener_rejects_unknown_card_action_message_id() -> None:
    listener = _FeishuCardActionListener(PycodexCard(app_id="app", app_secret="secret"))
    listener.register(_Object(message_id="om_first", _detached=False))
    listener.register(_Object(message_id="om_second", _detached=False))

    result = listener._handle_card_action(_event("om_stale"))

    assert result == {
        "toast": {
            "type": "warning",
            "content": "pycodex is detached.",
        }
    }


def test_feishu_listener_replaces_running_sdk_module_loop(monkeypatch) -> None:
    client_module = types.ModuleType("lark_oapi.ws.client")
    started_loops = []

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            loop = client_module.loop
            started_loops.append(loop)
            loop.run_until_complete(_noop())

    _install_fake_lark(monkeypatch, _Client, client_module)

    stale_loop = asyncio.new_event_loop()
    ready = threading.Event()
    thread = threading.Thread(target=_run_loop_forever, args=(stale_loop, ready))
    thread.daemon = True
    thread.start()
    ready.wait(timeout=2.0)
    client_module.loop = stale_loop

    try:
        listener = _FeishuCardActionListener(
            PycodexCard(app_id="app", app_secret="secret")
        )
        listener._listen()
    finally:
        stale_loop.call_soon_threadsafe(stale_loop.stop)
        thread.join(timeout=2.0)
        stale_loop.close()

    assert started_loops
    assert started_loops[0] is not stale_loop
    assert started_loops[0].is_closed()


def test_feishu_listener_stop_closes_sdk_loop(monkeypatch) -> None:
    client_module = types.ModuleType("lark_oapi.ws.client")
    started = threading.Event()
    stopped_loops = []
    disconnected = []

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            loop = client_module.loop
            stopped_loops.append(loop)
            started.set()
            loop.run_until_complete(_forever())

        async def _disconnect(self):
            disconnected.append(True)

    _install_fake_lark(monkeypatch, _Client, client_module)

    listener = _FeishuCardActionListener(PycodexCard(app_id="app", app_secret="secret"))
    listener.start()
    assert started.wait(timeout=2.0)

    listener.stop()

    assert disconnected == [True]
    assert stopped_loops
    assert stopped_loops[0].is_closed()
    assert listener._thread is None


def _run_loop_forever(loop, ready):
    asyncio.set_event_loop(loop)
    ready.set()
    loop.run_forever()
