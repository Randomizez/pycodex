import json

from pycodex.feishu_card import PycodexCard
from pycodex.utils.dotenv import parse_dotenv


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
        return _Response({"code": 0, "tenant_access_token": "tenant-token", "expire": 3600})

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
    columns = card["body"]["elements"][1]["columns"]
    return columns[0]["elements"][0]["content"]


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


def test_feishu_card_preserves_previous_output_while_next_turn_runs() -> None:
    card = PycodexCard()
    card.output_text = "previous answer"

    card.set_queued("next prompt", sender="alice")
    assert card.output_text == "(*last turn)\nprevious answer"

    card.apply_event({"kind": "turn_started", "user_text": "next prompt"})
    assert card.output_text == "(*last turn)\nprevious answer"

    card.apply_event({"kind": "turn_completed", "output_text": "new answer"})
    assert card.output_text == "new answer"


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


def test_feishu_card_from_env_reads_refresh_token_for_user_auth(monkeypatch) -> None:
    monkeypatch.setenv("FEISHU_REFRESH_TOKEN", "refresh-token")

    card = PycodexCard.from_env()

    assert card.refresh_token == "refresh-token"


def test_feishu_card_can_exchange_refresh_token_for_user_token() -> None:
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
        refresh_token="refresh-token",
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
    assert card.refresh_token == "next-refresh-token"


def test_feishu_card_persists_rotated_refresh_token_to_dotenv(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("FEISHU_REFRESH_TOKEN", raising=False)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("FEISHU_APP_ID='app'\nFEISHU_REFRESH_TOKEN='old-refresh'\n")
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
        refresh_token="old-refresh",
        refresh_token_store_path=dotenv_path,
        session=session,
    )

    assert card.user_access_token() == "user-token"

    assert parse_dotenv(dotenv_path.read_text())["FEISHU_REFRESH_TOKEN"] == (
        "next-refresh-token"
    )
    assert card.refresh_token == "next-refresh-token"
    assert session.posts[0]["json"]["refresh_token"] == "old-refresh"


def test_feishu_card_user_lookup_uses_tenant_token_with_refresh_token() -> None:
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
        refresh_token="refresh-token",
        session=session,
    )

    assert card._lookup_user({"emails": ["alice@example.com"]}) == "ou_user"

    assert session.posts[0]["url"].endswith("/auth/v3/tenant_access_token/internal")
    assert session.requests[0]["headers"] == {
        "Authorization": "Bearer tenant-token",
    }
