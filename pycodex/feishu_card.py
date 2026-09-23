import json
import os
import time
import typing
from pathlib import Path

import requests

from .events import (
    DEFAULT_MAIN_PROMPT,
    Event,
    EventDisplay,
    InputQueuedEvent,
    SessionStateEvent,
    TurnCompletedEvent,
    TurnStartedEvent,
)
from .utils.event_helpers import completed_history

FEISHU_API_BASE = "https://open.feishu.cn/open-apis"
FEISHU_DOMAIN = "https://open.feishu.cn"
FEISHU_REFRESH_TOKEN_FILE = "~/.codex/.feishu_refresh_token"
CARD_OUTPUT_LIMIT = 6500
CARD_OUTPUT_MODE_MARKDOWN = "markdown"
CARD_OUTPUT_MODE_CODE = "code"
CARD_CONTENT_ERROR_MARKER = "Failed to create card content"


class PycodexCard:
    def __init__(
        self,
        app_id: "typing.Union[str, None]" = None,
        app_secret: "typing.Union[str, None]" = None,
        api_base: str = FEISHU_API_BASE,
        domain: str = FEISHU_DOMAIN,
        verification_token: "typing.Union[str, None]" = None,
        encrypt_key: "typing.Union[str, None]" = None,
        session: "typing.Union[requests.Session, None]" = None,
    ) -> None:
        self.app_id = app_id
        self.app_secret = app_secret
        self.api_base = api_base
        self.domain = domain
        self.verification_token = verification_token
        self.encrypt_key = encrypt_key
        self.session = session or requests.Session()
        self.message_id = None
        self.callback_token = None
        self.session_key = None
        self.status = None
        self.prompt_text = DEFAULT_MAIN_PROMPT
        self.model_name = "pycodex"
        self.output_text = ""
        self.activity_text = ""
        self.user_prompt = ""
        self.turn_history = []
        self.history_index = None
        self._active_turn_id = None
        self._event_output = ""
        self.display = EventDisplay(self._log, self._set_status, self._set_prompt)
        self.detached = False
        self.accepts_input = True
        self._user_access_token = None
        self._user_token_expires_at = 0.0
        self._tenant_token = None
        self._tenant_token_expires_at = 0.0

    @classmethod
    def from_env(cls) -> "PycodexCard":
        api_base = os.environ.get("FEISHU_API_BASE", FEISHU_API_BASE)
        return cls(
            app_id=_env("FEISHU_APP_ID", "LARK_APP_ID"),
            app_secret=_env("FEISHU_APP_SECRET", "LARK_APP_SECRET"),
            api_base=api_base,
            domain=os.environ.get("FEISHU_DOMAIN", _api_base_to_domain(api_base)),
            verification_token=_env(
                "FEISHU_VERIFICATION_TOKEN",
                "LARK_VERIFICATION_TOKEN",
            ),
            encrypt_key=_env("FEISHU_ENCRYPT_KEY", "LARK_ENCRYPT_KEY"),
        )

    def configured(self) -> bool:
        return bool(self.app_id and self.app_secret)

    def send(self, target: str) -> None:
        receive_id = str(target or "").strip()
        if not receive_id:
            raise ValueError("recipient target is required")
        receive_id = self.resolve_name(receive_id)
        if not receive_id:
            raise ValueError(
                "cannot resolve Feishu user from target: {0}".format(target)
            )
        if receive_id.startswith("oc_"):
            resolved_type = "chat_id"
        else:
            resolved_type = "open_id"

        self.session_key = "feishu:manual:{0}:{1}".format(resolved_type, receive_id)

        def build_body(output_mode):
            return {
                "receive_id": receive_id,
                "msg_type": "interactive",
                "content": json.dumps(self.render(output_mode), ensure_ascii=False),
            }

        response = self._request_rendered_card(
            "POST",
            "/im/v1/messages",
            {"receive_id_type": resolved_type},
            build_body,
        )
        self.message_id = _extract_message_id(response)

    def update(
        self,
        message_id: "typing.Union[str, None]" = None,
        callback_token: "typing.Union[str, None]" = None,
    ) -> None:
        if message_id:
            self.message_id = message_id
        if callback_token:
            self.callback_token = callback_token
        if not self.configured():
            return
        if self.message_id:

            def build_body(output_mode):
                return {
                    "content": json.dumps(
                        self.render(output_mode),
                        ensure_ascii=False,
                    )
                }

            self._request_rendered_card(
                "PATCH",
                "/im/v1/messages/{0}".format(self.message_id),
                None,
                build_body,
            )
            return
        if self.callback_token:

            def build_body(output_mode):
                return {"token": self.callback_token, "card": self.render(output_mode)}

            self._request_rendered_card(
                "POST",
                "/interactive/v1/card/update",
                None,
                build_body,
                use_user_token=False,
            )

    def detach(self) -> None:
        self.detached = True

    def apply_event(self, event: "Event") -> None:
        self._event_output = ""
        if isinstance(event, SessionStateEvent):
            state = event.state
            self.model_name = state["model"]
            self.accepts_input = state["accepts_input"]
            self.display.closed = state["closed"]
            if event.reason in {"attach", "history"}:
                self.output_text = ""
                self.activity_text = ""
                self.turn_history = list(completed_history(state))
                self.history_index = None
                active = state["active_turn"]
                self._active_turn_id = active["turn_id"] if active is not None else None
                self.user_prompt = (
                    "\n".join(active["user_texts"])
                    if active is not None
                    else self.turn_history[-1][0] if self.turn_history else ""
                )
                self.display.stream_buffer = ""
                self.prompt_text = DEFAULT_MAIN_PROMPT
                if state["busy"]:
                    self._set_status("working")
                else:
                    self.display.set_idle_status(state["background_work_count"])
                for _prompt, response in self.turn_history[-1:]:
                    self.output_text = response
                if state["busy"]:
                    self._mark_last_turn_output()
        elif isinstance(event, (InputQueuedEvent, TurnStartedEvent)):
            self._mark_last_turn_output()
            if isinstance(event, TurnStartedEvent):
                prompt = event.visualize()
                if event.turn_id == self._active_turn_id and self.user_prompt:
                    self.user_prompt += "\n" + prompt
                else:
                    self.user_prompt = prompt
                self._active_turn_id = event.turn_id
        event.render(self.display)
        if isinstance(event, TurnCompletedEvent):
            if event.output_text:
                self.output_text = event.output_text
                self.turn_history.append((self.user_prompt, event.output_text))
            self.activity_text = ""
            self._active_turn_id = None
        elif isinstance(event, TurnStartedEvent):
            self.activity_text = ""
        elif self._event_output:
            self.activity_text = self._event_output

    def _mark_last_turn_output(self) -> None:
        prefix = "(*last turn)\n"
        if self.output_text and not self.output_text.startswith(prefix):
            self.output_text = prefix + self.output_text

    def _log(self, text: str) -> None:
        self._event_output = (
            self._event_output + ("\n" if self._event_output else "") + text
        )[-CARD_OUTPUT_LIMIT:]

    def _set_status(self, text: "typing.Union[str, None]") -> None:
        self.status = text

    def _set_prompt(self, text: str) -> None:
        if text != self.prompt_text:
            self.activity_text = ""
        self.prompt_text = text

    def render(
        self, output_mode: str = CARD_OUTPUT_MODE_MARKDOWN
    ) -> "typing.Dict[str, object]":
        status = "Closed" if self.display.closed else self.status
        input_disabled = not self.accepts_input or self.detached or self.display.closed
        output = _truncate(self.output_text, CARD_OUTPUT_LIMIT) or "Ready."
        working_output = _truncate(self.display.stream_buffer, CARD_OUTPUT_LIMIT)
        color = _status_color("Detached" if self.detached else status or "Idle")
        body_elements = [
            _output_box(
                "answer_box",
                "answer_md",
                _render_output_content(output, output_mode),
                "grey-50",
            ),
        ]
        if working_output:
            body_elements.append(
                _output_box(
                    "working_output_box",
                    "working_output_md",
                    _render_output_content(working_output, output_mode),
                    "green-50",
                )
            )
        if self.activity_text:
            body_elements.append(
                {
                    "tag": "markdown",
                    "element_id": "activity_md",
                    "content": _render_output_content(self.activity_text, output_mode),
                }
            )
        if self.user_prompt:
            body_elements.insert(
                0,
                {
                    "tag": "markdown",
                    "element_id": "user_prompt_md",
                    "content": _render_output_content(
                        "user> " + _truncate(self.user_prompt, CARD_OUTPUT_LIMIT),
                        output_mode,
                    ),
                },
            )
        history = self.earlier_turns()
        if history:
            index = (
                len(history) - 1 if self.history_index is None else self.history_index
            )
            body_elements.insert(
                0,
                _history_panel(
                    history,
                    index,
                    self.history_index is not None,
                    output_mode,
                    not self.detached,
                ),
            )
        card = {
            "schema": "2.0",
            "config": {"update_multi": True},
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": self.display.title or "pycodex",
                },
                "template": color,
            },
            "body": {
                "elements": body_elements,
            },
        }
        if not self.detached:
            card["body"]["elements"].append(
                {
                    "tag": "input",
                    "element_id": "prompt_input",
                    "name": "prompt",
                    "input_type": "text",
                    "width": "fill",
                    "disabled": input_disabled,
                    "placeholder": {
                        "tag": "plain_text",
                        "content": self.prompt_text + (status or self.model_name),
                    },
                    "behaviors": [{"type": "callback", "value": {"action": "send"}}],
                    "value": {"action": "send"},
                },
            )
        return card

    def earlier_turns(self):
        if self._active_turn_id is not None:
            return self.turn_history
        return self.turn_history[:-1]

    def show_history(self, index):
        if type(index) is not int or not 0 <= index < len(self.earlier_turns()):
            raise ValueError("This history turn is no longer available.")
        self.history_index = index

    def parse_action(self, sdk_event) -> "typing.Dict[str, object]":
        event_data = getattr(sdk_event, "event", None)
        action_payload = getattr(event_data, "action", None)
        operator = getattr(event_data, "operator", None)
        context = getattr(event_data, "context", None)
        form_values = _sdk_form_values(action_payload)
        value = getattr(action_payload, "value", None) or {}
        if not isinstance(value, dict):
            value = {}
        prompt = str(
            form_values.get("prompt")
            or form_values.get("prompt_input")
            or getattr(action_payload, "input_value", None)
            or value.get("input_value")
            or ""
        ).strip()
        if not prompt and form_values:
            prompt = str(next(iter(form_values.values()))).strip()
        action = str(
            value.get("action") or getattr(action_payload, "name", None) or "send"
        ).strip()
        if action in {"send_button", "prompt_form"}:
            action = "send"
        message_id = (
            str(getattr(context, "open_message_id", None) or "").strip() or None
        )
        callback_token = str(getattr(event_data, "token", None) or "").strip() or None
        self.message_id = message_id or self.message_id
        self.callback_token = callback_token or self.callback_token
        if not self.session_key:
            tenant_key = str(getattr(operator, "tenant_key", None) or "").strip()
            open_id = str(getattr(operator, "open_id", None) or "").strip()
            self.session_key = _default_session_key(tenant_key, open_id, message_id)
        return {
            "action": action,
            "history_index": value.get("history_index"),
            "prompt": prompt,
            "sender": self.resolve_operator_name(operator),
        }

    def resolve_name(self, name: str) -> "typing.Union[str, None]":
        normalized = str(name or "").strip()
        if not normalized:
            return None
        aliases = _user_id_aliases()
        if normalized in aliases:
            return aliases[normalized]
        if normalized.startswith("ou_"):
            return normalized
        if normalized.startswith("oc_"):
            return normalized
        if normalized.isnumeric():
            return self._lookup_user({"mobiles": [normalized]})
        if "@" not in normalized:
            email_domain = _default_email_domain()
            if not email_domain:
                return None
            normalized = normalized + "@" + email_domain
        return self._lookup_user({"emails": [normalized]})

    def _lookup_user(
        self, body: "typing.Dict[str, object]"
    ) -> "typing.Union[str, None]":
        payload = dict(body)
        payload["include_resigned"] = False
        response = self._request(
            "POST",
            "/contact/v3/users/batch_get_id",
            params={"user_id_type": "open_id"},
            json_body=payload,
            use_user_token=False,
        )
        user_list = _dig(response, "data", "user_list")
        if not isinstance(user_list, list) or not user_list:
            return None
        item = user_list[0] if isinstance(user_list[0], dict) else {}
        value = item.get("user_id") or item.get("open_id") or item.get("union_id")
        return str(value).strip() if value else None

    def resolve_operator_name(self, operator) -> str:
        for user_id_type in ("user_id", "open_id", "union_id"):
            user_id = str(getattr(operator, user_id_type, None) or "").strip()
            if not user_id:
                continue
            try:
                response = self._request(
                    "GET",
                    "/contact/v3/users/{0}".format(user_id),
                    params={"user_id_type": user_id_type},
                    use_user_token=False,
                )
            except Exception:
                continue
            name = _display_user_name(_dig(response, "data", "user") or response)
            if name:
                return name
        return "cli"

    def _request(
        self,
        method: str,
        path: str,
        params: "typing.Union[typing.Dict[str, str], None]" = None,
        json_body: "typing.Union[typing.Dict[str, object], None]" = None,
        use_user_token: bool = True,
    ) -> "typing.Dict[str, object]":
        user_token = self.user_access_token() if use_user_token else None
        token = user_token or self.tenant_access_token()
        response = self.session.request(
            method,
            self.api_base.rstrip("/") + path,
            params=params,
            json=json_body,
            headers={"Authorization": "Bearer {0}".format(token)},
            timeout=20,
        )
        return _checked_json_response(response)

    def _request_rendered_card(
        self,
        method: str,
        path: str,
        params: "typing.Union[typing.Dict[str, str], None]",
        json_body_builder: "typing.Callable[[str], typing.Dict[str, object]]",
        use_user_token: bool = True,
    ) -> "typing.Dict[str, object]":
        try:
            return self._request(
                method,
                path,
                params=params,
                json_body=json_body_builder(CARD_OUTPUT_MODE_MARKDOWN),
                use_user_token=use_user_token,
            )
        except Exception as exc:
            if not _is_card_content_error(exc):
                raise
            return self._request(
                method,
                path,
                params=params,
                json_body=json_body_builder(CARD_OUTPUT_MODE_CODE),
                use_user_token=use_user_token,
            )

    def user_access_token(self) -> "typing.Union[str, None]":
        refresh_token = _read_refresh_token()
        if not refresh_token:
            return None
        now = time.time()
        if self._user_access_token and now < self._user_token_expires_at:
            return self._user_access_token
        if not self.configured():
            raise RuntimeError("FEISHU_APP_ID and FEISHU_APP_SECRET are required")
        response = self.session.post(
            self.api_base.rstrip("/") + "/authen/v2/oauth/token",
            json={
                "grant_type": "refresh_token",
                "client_id": self.app_id,
                "client_secret": self.app_secret,
                "refresh_token": refresh_token,
            },
            timeout=20,
        )
        payload = _checked_json_response(response)
        token = payload.get("access_token")
        if not token:
            raise RuntimeError("user access_token missing from Feishu response")
        refresh_token = payload.get("refresh_token")
        if refresh_token:
            _write_refresh_token(str(refresh_token))
        self._user_access_token = str(token)
        expires_in = payload.get("expires_in") or 0
        self._user_token_expires_at = time.time() + max(60, int(expires_in) - 120)
        return self._user_access_token

    def tenant_access_token(self) -> str:
        now = time.time()
        if self._tenant_token and now < self._tenant_token_expires_at:
            return self._tenant_token
        response = self.session.post(
            self.api_base.rstrip("/") + "/auth/v3/tenant_access_token/internal",
            json={"app_id": self.app_id, "app_secret": self.app_secret},
            timeout=20,
        )
        payload = _checked_json_response(response)
        token = _dig(payload, "tenant_access_token") or _dig(
            payload,
            "data",
            "tenant_access_token",
        )
        if not token:
            raise RuntimeError("tenant_access_token missing from Feishu response")
        expires_in = _dig(payload, "expire") or _dig(payload, "data", "expire") or 3600
        self._tenant_token = str(token)
        self._tenant_token_expires_at = now + max(60, int(expires_in) - 120)
        return self._tenant_token


def _sdk_form_values(action_payload: "typing.Any") -> "typing.Dict[str, object]":
    for name in ("form_value", "form_values", "input_values"):
        value = getattr(action_payload, name, None)
        if isinstance(value, dict):
            return value
    value = getattr(action_payload, "value", None)
    if isinstance(value, dict):
        for name in ("form_value", "form_values", "input_values"):
            nested = value.get(name)
            if isinstance(nested, dict):
                return nested
    return {}


def _checked_json_response(response) -> "typing.Dict[str, object]":
    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(
            "Feishu API returned non-JSON response: status={0} body={1}".format(
                getattr(response, "status_code", "?"),
                getattr(response, "text", ""),
            )
        ) from exc
    if getattr(response, "status_code", 200) >= 400:
        raise RuntimeError(
            "Feishu API error: status={0} body={1}".format(
                response.status_code,
                json.dumps(payload, ensure_ascii=False),
            )
        )
    code = payload.get("code")
    if code not in (None, 0):
        raise RuntimeError(
            "Feishu API error: {0}".format(json.dumps(payload, ensure_ascii=False))
        )
    return payload


def _extract_message_id(
    response: "typing.Dict[str, object]",
) -> "typing.Union[str, None]":
    for path in (
        ("data", "message_id"),
        ("data", "message", "message_id"),
        ("message_id",),
        ("open_message_id",),
    ):
        value = _dig(response, *path)
        if value:
            return str(value)
    return None


def _default_session_key(
    tenant_key: str,
    open_id: str,
    message_id: "typing.Union[str, None]",
) -> str:
    user_key = open_id or "unknown"
    if message_id:
        return "feishu:{0}:{1}:{2}".format(tenant_key or "tenant", user_key, message_id)
    return "feishu:{0}:{1}:default".format(tenant_key or "tenant", user_key)


def _display_user_name(user: "typing.Any") -> "typing.Union[str, None]":
    if not isinstance(user, dict):
        return None
    for key in ("name", "en_name", "nickname", "email", "enterprise_email"):
        value = str(user.get(key) or "").strip()
        if value:
            return value
    return None


def _status_color(status: str) -> str:
    normalized = status.lower()
    if normalized in {"detached", "closed"}:
        return "grey"
    if normalized in {"idle", "idle: sleeping"}:
        return "green"
    return "blue"


def _truncate(text: str, limit: int) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 32)] + "\n...[truncated]"


def _escape_code_block(text: str) -> str:
    return str(text or "").replace("```", "'''")


def _render_output_content(output: str, output_mode: str) -> str:
    if output_mode == CARD_OUTPUT_MODE_CODE:
        return "```text\n{0}\n```".format(_escape_code_block(output))
    return output


def _history_panel(history, index, expanded, output_mode, interactive):
    prompt, response = history[index]
    content = _truncate(
        "user> {0}\n\nassistant> {1}".format(prompt, response), CARD_OUTPUT_LIMIT
    )
    columns = []
    for label, target, disabled in (
        ("Previous", index - 1, index == 0),
        ("Next", index + 1, index == len(history) - 1),
    ):
        columns.append(
            {
                "tag": "column",
                "width": "auto",
                "elements": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": label},
                        "type": "default",
                        "disabled": disabled or not interactive,
                        "behaviors": [
                            {
                                "type": "callback",
                                "value": {
                                    "action": "history",
                                    "history_index": target,
                                },
                            }
                        ],
                    }
                ],
            }
        )
    return {
        "tag": "collapsible_panel",
        "element_id": "history_panel",
        "expanded": expanded,
        "header": {
            "icon": {
                "tag": "standard_icon",
                "token": "down-small-ccm_outlined",
                "size": "16px 16px",
            },
            "title": {
                "tag": "plain_text",
                "content": "History · {0} / {1}".format(index + 1, len(history)),
            },
        },
        "elements": [
            {
                "tag": "markdown",
                "element_id": "history_md",
                "content": _render_output_content(content, output_mode),
            },
            {"tag": "column_set", "columns": columns},
        ],
    }


def _output_box(
    box_id: str,
    markdown_id: str,
    content: str,
    background_style: str,
) -> "typing.Dict[str, object]":
    return {
        "tag": "column_set",
        "element_id": box_id,
        "background_style": background_style,
        "horizontal_spacing": "8px",
        "horizontal_align": "left",
        "columns": [
            {
                "tag": "column",
                "width": "auto",
                "elements": [
                    {
                        "tag": "markdown",
                        "element_id": markdown_id,
                        "content": content,
                    },
                ],
                "vertical_spacing": "8px",
                "horizontal_align": "left",
                "vertical_align": "top",
            }
        ],
    }


def _is_card_content_error(exc: "BaseException") -> bool:
    return CARD_CONTENT_ERROR_MARKER in str(exc)


def _dig(value: "typing.Any", *path: str) -> "typing.Any":
    current = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _env(*names: str) -> "typing.Union[str, None]":
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _refresh_token_path() -> Path:
    return Path(FEISHU_REFRESH_TOKEN_FILE).expanduser()


def _read_refresh_token() -> "typing.Union[str, None]":
    path = _refresh_token_path()
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8", errors="replace").strip()
    return value or None


def _write_refresh_token(value: str) -> None:
    path = _refresh_token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        path.chmod(0o600)
        stream.write("{0}\n".format(value))


def _api_base_to_domain(api_base: str) -> str:
    base = str(api_base or FEISHU_API_BASE).rstrip("/")
    marker = "/open-apis"
    if base.endswith(marker):
        return base[: -len(marker)]
    return base


def _default_email_domain() -> "typing.Union[str, None]":
    value = _env(
        "PYCODEX_FEISHU_DEFAULT_EMAIL_DOMAIN",
        "FEISHU_DEFAULT_EMAIL_DOMAIN",
        "LARK_DEFAULT_EMAIL_DOMAIN",
    )
    if not value:
        return None
    return str(value).strip().lstrip("@") or None


def _user_id_aliases() -> "typing.Dict[str, str]":
    raw = os.environ.get("PYCODEX_FEISHU_USER_IDS")
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items() if key and value}
