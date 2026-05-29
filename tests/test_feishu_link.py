from pycodex.feishu_card import PycodexCard
from pycodex.feishu_link import _FeishuCardActionListener


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
