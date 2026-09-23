import json
import os

import pytest

from pycodex.events import (
    AssistantDeltaEvent,
    Event,
    InputQueuedEvent,
    InputRequestedEvent,
    InputResolvedEvent,
    SessionClosedEvent,
    SessionStateEvent,
    StreamErrorEvent,
    ToolCompletedEvent,
    ToolStartedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
    TurnStartedEvent,
)
from pycodex.feishu_card import CARD_OUTPUT_LIMIT, PycodexCard
from pycodex.protocol import ToolCall, ToolResult


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, responses, post_responses=None):
        self._responses = list(responses)
        self._post_responses = list(post_responses or [])
        self.posts = []
        self.requests = []

    def post(self, url, json=None, timeout=None):
        self.posts.append({"url": url, "json": json, "timeout": timeout})
        if self._post_responses:
            return _Response(self._post_responses.pop(0))
        return _Response(
            {"code": 0, "tenant_access_token": "tenant-token", "expire": 3600}
        )

    def request(self, method, url, params=None, json=None, headers=None, timeout=None):
        self.requests.append(
            {
                "method": method,
                "url": url,
                "params": params,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        if self._responses:
            return _Response(self._responses.pop(0))
        return _Response({"code": 0})


def _card_output_from_message_payload(payload):
    content = json.loads(payload["content"])
    return _answer_markdown_content(content)


def _card_output_from_update_payload(payload):
    content = json.loads(payload["content"])
    return _answer_markdown_content(content)


def _answer_markdown_content(card):
    columns = _element(card, "answer_box")["columns"]
    return columns[0]["elements"][0]["content"]


def _working_markdown_content(card):
    element = _element(card, "working_output_box")
    columns = element["columns"]
    return columns[0]["elements"][0]["content"]


def _has_element(card, element_id):
    return any(
        element.get("element_id") == element_id for element in card["body"]["elements"]
    )


def _element(card, element_id):
    for element in card["body"]["elements"]:
        if element.get("element_id") == element_id:
            return element
    raise AssertionError("{0} not found".format(element_id))


def _prompt_input(card):
    return _element(card, "prompt_input")


def test_feishu_card_uses_session_connected_header_and_input_status() -> None:
    card = PycodexCard()

    rendered = card.render()
    assert rendered["header"]["title"]["content"] == "Session Connected"
    assert _prompt_input(rendered)["placeholder"]["content"] == "pycodex> pycodex"

    card.apply_event(TurnStartedEvent("turn", ("hello",)))
    rendered = card.render()

    assert rendered["header"]["title"]["content"] == "Session Connected"
    assert rendered["header"]["template"] == "blue"
    assert _prompt_input(rendered)["placeholder"]["content"] == (
        "pycodex> turn_started"
    )
    assert not _prompt_input(rendered)["disabled"]


def test_feishu_card_delegates_unknown_events_to_its_own_plain_display() -> None:
    class RenderedEvent(Event):
        def render(self, display):
            display.write("new event")
            display.set_status("custom activity")
            display.set_prompt("custom> ")
            display.stream_buffer = "live text"

    card = PycodexCard()
    other = PycodexCard()
    card.apply_event(RenderedEvent())

    assert card.display is not other.display
    assert not card.display.color_enabled
    assert other.output_text == other.display.stream_buffer == ""
    rendered = card.render()
    assert _answer_markdown_content(rendered) == "Ready."
    assert _element(rendered, "activity_md")["content"] == "new event"
    assert _working_markdown_content(rendered) == "live text"
    assert (
        _prompt_input(rendered)["placeholder"]["content"] == "custom> custom activity"
    )
    assert "\\u001b" not in json.dumps(rendered)


def test_feishu_card_send_falls_back_to_code_mode_without_mutating_output() -> None:
    session = _FakeSession(
        [
            {
                "code": 999,
                "msg": "Failed to create card content: invalid markdown image",
            },
            {"code": 0, "data": {"message_id": "om_test"}},
        ]
    )
    card = PycodexCard(app_id="app", app_secret="secret", session=session)
    card.output_text = "![bad](not-a-valid-feishu-image)"

    card.send("ou_user")

    assert card.message_id == "om_test"
    assert card.output_text == "![bad](not-a-valid-feishu-image)"
    assert len(session.requests) == 2
    assert _card_output_from_message_payload(session.requests[0]["json"]) == (
        "![bad](not-a-valid-feishu-image)"
    )
    assert _card_output_from_message_payload(session.requests[1]["json"]) == (
        "```text\n![bad](not-a-valid-feishu-image)\n```"
    )


def test_feishu_card_update_falls_back_to_escaped_code_mode() -> None:
    session = _FakeSession(
        [
            {
                "code": 999,
                "msg": "Failed to create card content: invalid markdown image",
            },
            {"code": 0},
        ]
    )
    card = PycodexCard(app_id="app", app_secret="secret", session=session)
    card.message_id = "om_test"
    card.output_text = "before ``` fence\n![bad](not-a-valid-feishu-image)"

    card.update()

    assert card.output_text == "before ``` fence\n![bad](not-a-valid-feishu-image)"
    assert len(session.requests) == 2
    assert session.requests[0]["method"] == "PATCH"
    assert _card_output_from_update_payload(session.requests[0]["json"]) == (
        "before ``` fence\n![bad](not-a-valid-feishu-image)"
    )
    assert _card_output_from_update_payload(session.requests[1]["json"]) == (
        "```text\nbefore ''' fence\n![bad](not-a-valid-feishu-image)\n```"
    )


def test_feishu_card_preserves_last_reply_until_next_turn_completes() -> None:
    card = PycodexCard()
    card.apply_event(TurnCompletedEvent("previous", 1, "previous answer", 0))

    card.apply_event(
        InputQueuedEvent("next", "next prompt", "steer", "cli", False, False)
    )
    assert card.output_text == "(*last turn)\nprevious answer"
    card.apply_event(TurnStartedEvent("turn", ("next prompt",), "next"))
    assert card.output_text == "(*last turn)\nprevious answer"

    card.apply_event(TurnCompletedEvent("turn", 1, "new answer", 0))
    assert card.output_text == "new answer"
    assert card.status is None
    assert not _has_element(card.render(), "activity_md")

    card.apply_event(
        TurnCompletedEvent("long", 1, "x" * CARD_OUTPUT_LIMIT + "latest", 0)
    )
    rendered = _answer_markdown_content(card.render())
    assert len(rendered) <= CARD_OUTPUT_LIMIT
    assert rendered.startswith("x")
    assert rendered.endswith("...[truncated]")
    assert "previous answer" not in rendered


def test_feishu_card_shows_current_delta_segment_above_input() -> None:
    card = PycodexCard()
    card.output_text = "previous answer"
    card.apply_event(TurnStartedEvent("turn", ("next prompt",)))

    card.apply_event(AssistantDeltaEvent("first ", "turn"))
    card.apply_event(AssistantDeltaEvent("segment", "turn"))
    rendered = card.render()

    assert _answer_markdown_content(rendered) == "(*last turn)\nprevious answer"
    assert _element(rendered, "working_output_box")["background_style"] == "green-50"
    assert _working_markdown_content(rendered) == "first segment"

    call = ToolCall("call", "shell", {"command": ["pwd"]})
    card.apply_event(ToolStartedEvent("turn", call))
    card.apply_event(
        ToolCompletedEvent("turn", call, ToolResult("call", "shell", "done"))
    )
    rendered = card.render()
    assert not _has_element(rendered, "working_output_box")
    assert _answer_markdown_content(rendered) == "(*last turn)\nprevious answer"
    assert _element(rendered, "activity_md")["content"] == "[shell] pwd -> done"

    card.apply_event(AssistantDeltaEvent("final ", "turn"))
    card.apply_event(AssistantDeltaEvent("answer", "turn"))
    assert _working_markdown_content(card.render()) == "final answer"

    card.apply_event(TurnCompletedEvent("turn", 1, "final answer", 0))
    rendered = card.render()
    assert _answer_markdown_content(rendered) == "final answer"
    assert not _has_element(rendered, "activity_md")
    assert not _has_element(rendered, "working_output_box")


def test_feishu_card_shared_buffer_discards_retries_and_flushes_fatal_output() -> None:
    card = PycodexCard()
    card.apply_event(TurnCompletedEvent("previous", 1, "previous answer", 0))
    card.apply_event(TurnStartedEvent("turn", ("next prompt",)))
    card.apply_event(AssistantDeltaEvent("discarded"))
    assert _working_markdown_content(card.render()) == "discarded"

    card.apply_event(StreamErrorEvent("Retrying", 1, 2, 0, "lost"))
    assert "discarded" not in json.dumps(card.render())
    assert not _has_element(card.render(), "working_output_box")

    card.apply_event(AssistantDeltaEvent("retained"))
    card.apply_event(TurnFailedEvent("turn", 1, "failed", "RuntimeError", 0))
    assert card.output_text == "(*last turn)\nprevious answer"
    assert _element(card.render(), "activity_md")["content"] == (
        "assistant> retained\nError: failed"
    )
    assert not _has_element(card.render(), "working_output_box")


@pytest.mark.parametrize(
    "request_kind,other,prompt",
    [
        ("questions", False, "answer> "),
        ("questions", True, "other> "),
        ("permissions", False, "permissions> "),
    ],
)
def test_feishu_card_uses_shared_input_prompts(request_kind, other, prompt) -> None:
    card = PycodexCard()
    request = InputRequestedEvent(
        "request",
        request_kind,
        other,
        question={
            "header": "Choice",
            "question": "Which?",
            "options": [{"label": "First", "description": "Use first"}],
        },
        permissions={"permissions": {"network": {"enabled": True}}},
    )
    card.apply_event(request)
    rendered = card.render()
    assert _answer_markdown_content(rendered) == "Ready."
    assert _element(rendered, "activity_md")["content"] == request.visualize()
    assert _prompt_input(rendered)["placeholder"]["content"].startswith(prompt)
    assert not _prompt_input(rendered)["disabled"]

    card.apply_event(InputResolvedEvent("request"))
    assert _prompt_input(card.render())["placeholder"]["content"] == "pycodex> pycodex"
    assert not _has_element(card.render(), "activity_md")


def test_feishu_card_restores_snapshot_without_replaying_active_turn_twice() -> None:
    card = PycodexCard()
    request = InputRequestedEvent("request", "questions", True)
    state = {
        "model": "restored-model",
        "title": "Restored",
        "closed": False,
        "accepts_input": True,
        "busy": True,
        "background_work_count": 0,
        "context_window": None,
        "usage_tokens": None,
        "input_request": request,
        "history": (("previous", "answer"), ("live", "partial")),
        "active_turn": {
            "turn_id": "turn",
            "submission_id": "submission",
            "user_texts": ["live"],
            "assistant_text": "partial",
            "completed_history": (("previous", "answer"),),
        },
    }
    card.apply_event(SessionStateEvent("attach", state))
    assert card.output_text == "(*last turn)\nanswer"
    assert _element(card.render(), "activity_md")["content"] == (
        "user> live\nassistant> partial\n" + request.visualize()
    )
    assert card.prompt_text == "other> "
    assert card.display.stream_buffer == ""
    assert card.render()["header"]["title"]["content"] == "Session Connected · Restored"

    card.apply_event(AssistantDeltaEvent("stale"))
    card.apply_event(
        SessionStateEvent(
            "history",
            dict(
                state,
                history=(),
                active_turn=None,
                input_request=None,
                busy=False,
            ),
        )
    )
    assert card.output_text == card.display.stream_buffer == ""
    assert _answer_markdown_content(card.render()) == "Ready."
    assert (
        _prompt_input(card.render())["placeholder"]["content"]
        == "pycodex> restored-model"
    )


def test_feishu_card_disables_closed_input_and_removes_detached_input() -> None:
    card = PycodexCard()
    card.apply_event(AssistantDeltaEvent("last output"))
    card.apply_event(SessionClosedEvent())
    rendered = card.render()
    assert rendered["header"]["title"]["content"] == "Session Closed"
    assert _answer_markdown_content(rendered) == "Ready."
    assert _element(rendered, "activity_md")["content"] == "assistant> last output"
    assert _prompt_input(rendered)["disabled"]
    assert not _has_element(rendered, "working_output_box")

    card.detach()
    rendered = card.render()
    assert rendered["header"]["title"]["content"] == "Session Detached"
    assert not _has_element(rendered, "prompt_input")


def test_resolve_name_uses_default_email_domain_from_env(monkeypatch) -> None:
    monkeypatch.setenv("FEISHU_DEFAULT_EMAIL_DOMAIN", "@example.com")
    card = PycodexCard()
    lookups = []
    card._lookup_user = lambda body: lookups.append(body) or "ou_user"

    assert card.resolve_name("alice") == "ou_user"
    assert lookups == [{"emails": ["alice@example.com"]}]


def test_resolve_name_without_default_email_domain_does_not_guess(monkeypatch) -> None:
    for name in (
        "PYCODEX_FEISHU_DEFAULT_EMAIL_DOMAIN",
        "FEISHU_DEFAULT_EMAIL_DOMAIN",
        "LARK_DEFAULT_EMAIL_DOMAIN",
    ):
        monkeypatch.delenv(name, raising=False)
    card = PycodexCard()
    lookups = []
    card._lookup_user = lambda body: lookups.append(body) or "ou_user"

    assert card.resolve_name("alice") is None
    assert lookups == []


def test_feishu_card_ignores_refresh_token_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("FEISHU_REFRESH_TOKEN", "refresh-token")

    card = PycodexCard.from_env()
    session = _FakeSession([])
    card.session = session

    assert card.user_access_token() is None
    assert session.posts == []


def test_feishu_card_from_env_reads_fixed_refresh_token_store(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("FEISHU_APP_ID", "app")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret")
    monkeypatch.setenv("FEISHU_REFRESH_TOKEN", "stale-refresh")
    token_path = tmp_path / ".codex" / ".feishu_refresh_token"
    token_path.parent.mkdir()
    token_path.write_text("file-refresh\n")

    card = PycodexCard.from_env()
    card.session = _FakeSession(
        [],
        post_responses=[{"code": 0, "access_token": "user-token", "expires_in": 7200}],
    )

    assert card.user_access_token() == "user-token"
    assert card.session.posts[0]["json"]["refresh_token"] == "file-refresh"


def test_feishu_card_can_exchange_refresh_token_for_user_token(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    token_path = tmp_path / ".codex" / ".feishu_refresh_token"
    token_path.parent.mkdir()
    token_path.write_text("refresh-token\n", encoding="utf-8")
    session = _FakeSession(
        [{"code": 0, "data": {"message_id": "om_test"}}],
        post_responses=[
            {
                "code": 0,
                "access_token": "user-token",
                "expires_in": 7200,
                "refresh_token": "next-refresh-token",
            }
        ],
    )
    card = PycodexCard(
        app_id="app",
        app_secret="secret",
        session=session,
    )

    card.send("oc_chat")

    assert session.posts[0]["url"].endswith("/authen/v2/oauth/token")
    assert session.posts[0]["json"] == {
        "grant_type": "refresh_token",
        "client_id": "app",
        "client_secret": "secret",
        "refresh_token": "refresh-token",
    }
    assert session.requests[0]["headers"] == {
        "Authorization": "Bearer user-token",
    }
    assert token_path.read_text(encoding="utf-8") == "next-refresh-token\n"


def test_feishu_card_persists_rotated_refresh_token_to_store(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("FEISHU_REFRESH_TOKEN", "stale-refresh")
    token_path = tmp_path / ".codex" / ".feishu_refresh_token"
    token_path.parent.mkdir()
    token_path.write_text("old-refresh\n")
    dotenv_path = token_path.parent / ".env"
    dotenv_content = "FEISHU_REFRESH_TOKEN=stale-refresh\nKEEP=value\n"
    dotenv_path.write_text(dotenv_content, encoding="utf-8")
    session = _FakeSession(
        [],
        post_responses=[
            {
                "code": 0,
                "access_token": "user-token",
                "expires_in": 7200,
                "refresh_token": "next-refresh-token",
            }
        ],
    )
    card = PycodexCard(
        app_id="app",
        app_secret="secret",
        session=session,
    )

    assert card.user_access_token() == "user-token"

    assert token_path.read_text() == "next-refresh-token\n"
    assert token_path.stat().st_mode & 0o777 == 0o600
    assert dotenv_path.read_text(encoding="utf-8") == dotenv_content
    assert os.environ["FEISHU_REFRESH_TOKEN"] == "stale-refresh"
    assert session.posts[0]["json"]["refresh_token"] == "old-refresh"


def test_feishu_card_rereads_refresh_token_store_for_each_exchange(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    token_path = tmp_path / ".codex" / ".feishu_refresh_token"
    token_path.parent.mkdir()
    token_path.write_text("old-refresh\n")
    first_session = _FakeSession(
        [],
        post_responses=[
            {
                "code": 0,
                "access_token": "first-user-token",
                "expires_in": 7200,
                "refresh_token": "first-refresh-token",
            },
            {
                "code": 0,
                "access_token": "third-user-token",
                "expires_in": 7200,
                "refresh_token": "third-refresh-token",
            },
        ],
    )
    second_session = _FakeSession(
        [],
        post_responses=[
            {
                "code": 0,
                "access_token": "second-user-token",
                "expires_in": 7200,
                "refresh_token": "second-refresh-token",
            }
        ],
    )
    first = PycodexCard(
        app_id="app",
        app_secret="secret",
        session=first_session,
    )
    second = PycodexCard(
        app_id="app",
        app_secret="secret",
        session=second_session,
    )

    assert first.user_access_token() == "first-user-token"
    assert second.user_access_token() == "second-user-token"

    assert first_session.posts[0]["json"]["refresh_token"] == "old-refresh"
    assert second_session.posts[0]["json"]["refresh_token"] == "first-refresh-token"
    assert token_path.read_text() == "second-refresh-token\n"

    first._user_token_expires_at = 0
    assert first.user_access_token() == "third-user-token"
    assert first_session.posts[1]["json"]["refresh_token"] == "second-refresh-token"
    assert token_path.read_text() == "third-refresh-token\n"


def test_feishu_card_refresh_store_failure_does_not_cache_access_token(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    token_path = tmp_path / ".codex" / ".feishu_refresh_token"
    token_path.parent.mkdir()
    token_path.write_text("old-refresh\n", encoding="utf-8")
    session = _FakeSession(
        [],
        post_responses=[
            {
                "code": 0,
                "access_token": "user-token",
                "expires_in": 7200,
                "refresh_token": "next-refresh-token",
            }
        ],
    )
    card = PycodexCard(app_id="app", app_secret="secret", session=session)

    def fail_write(value):
        raise OSError("token store unavailable")

    monkeypatch.setattr("pycodex.feishu_card._write_refresh_token", fail_write)
    with pytest.raises(OSError, match="token store unavailable"):
        card.user_access_token()

    assert card._user_access_token is None
    assert card._user_token_expires_at == 0


def test_feishu_card_user_lookup_uses_tenant_token_with_refresh_token(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    token_path = tmp_path / ".codex" / ".feishu_refresh_token"
    token_path.parent.mkdir()
    token_path.write_text("refresh-token\n", encoding="utf-8")
    session = _FakeSession(
        [
            {
                "code": 0,
                "data": {
                    "user_list": [
                        {
                            "user_id": "ou_user",
                        }
                    ]
                },
            }
        ]
    )
    card = PycodexCard(
        app_id="app",
        app_secret="secret",
        session=session,
    )

    assert card._lookup_user({"emails": ["alice@example.com"]}) == "ou_user"

    assert session.posts[0]["url"].endswith("/auth/v3/tenant_access_token/internal")
    assert session.requests[0]["headers"] == {
        "Authorization": "Bearer tenant-token",
    }
