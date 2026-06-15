from pycodex.utils import visualize


def test_prompt_session_kwargs_omits_unsupported_show_frame(monkeypatch):
    class _OldPromptSession:
        def __init__(
            self,
            erase_when_done=False,
            enable_system_prompt=False,
        ):
            pass

    monkeypatch.setattr(visualize, "PromptSession", _OldPromptSession)

    assert visualize.prompt_session_kwargs() == {
        "erase_when_done": True,
        "enable_system_prompt": True,
    }


def test_prompt_session_kwargs_includes_supported_show_frame(monkeypatch):
    class _NewPromptSession:
        def __init__(
            self,
            erase_when_done=False,
            enable_system_prompt=False,
            show_frame=False,
        ):
            pass

    monkeypatch.setattr(visualize, "PromptSession", _NewPromptSession)

    assert visualize.prompt_session_kwargs() == {
        "erase_when_done": True,
        "enable_system_prompt": True,
        "show_frame": True,
    }
