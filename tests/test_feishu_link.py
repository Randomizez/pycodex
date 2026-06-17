from pycodex.feishu_card import PycodexCard
import pycodex.feishu_link as feishu_link
from pycodex.feishu_link import _FeishuCardActionListener, _release_feishu_listener


class _Object:
    def __init__(self, **values):
        self.__dict__.update(values)


def _event(message_id):
    return _Object(event=_Object(context=_Object(open_message_id=message_id)))


def test_feishu_listener_tracks_one_active_link() -> None:
    listener = _FeishuCardActionListener(PycodexCard(app_id="app", app_secret="secret"))
    first = _Object(message_id="om_first")
    second = _Object(message_id="om_second")

    listener.register(first)
    assert listener.link is first
    listener.unregister(first)
    assert listener.link is None

    listener.register(second)
    assert listener.link is second


def test_feishu_listener_stops_when_last_link_is_released(monkeypatch) -> None:
    listener = _FeishuCardActionListener(PycodexCard(app_id="app", app_secret="secret"))
    link = _Object(message_id="om_first")
    stopped = []
    listener.stop = lambda: stopped.append(True)
    listener.register(link)
    monkeypatch.setattr(feishu_link, "_LISTENER", listener)

    _release_feishu_listener(listener, link)

    assert listener.link is None
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
