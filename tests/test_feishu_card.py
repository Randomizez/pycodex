import json

from pycodex.feishu_card import PycodexCard


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.posts = []
        self.requests = []

    def post(self, url, json=None, timeout=None):
        self.posts.append({"url": url, "json": json, "timeout": timeout})
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
