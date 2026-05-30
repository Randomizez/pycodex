import json
import os
from pathlib import Path
import time
import typing

import requests

FEISHU_API_BASE = "https://open.feishu.cn/open-apis"
FEISHU_DOMAIN = "https://open.feishu.cn"
FEISHU_REFRESH_TOKEN_FILE = "~/.codex/.feishu_refresh_token"
CARD_OUTPUT_LIMIT = 6500
CARD_OUTPUT_MODE_MARKDOWN = "markdown"
CARD_OUTPUT_MODE_CODE = "code"
CARD_CONTENT_ERROR_MARKER = "Failed to create card content"
LAST_TURN_PREFIX = "(*last turn)\n"


class PycodexCard:
    def __init__(
        self,
        app_id: "typing.Union[str, None]" = None,
        app_secret: "typing.Union[str, None]" = None,
        api_base: str = FEISHU_API_BASE,
        domain: str = FEISHU_DOMAIN,
        verification_token: "typing.Union[str, None]" = None,
        encrypt_key: "typing.Union[str, None]" = None,
        refresh_token: "typing.Union[str, None]" = None,
        session: "typing.Union[requests.Session, None]" = None,
    ) -> None:
        self.app_id = app_id
        self.app_secret = app_secret
        self.api_base = api_base
        self.domain = domain
        self.verification_token = verification_token
        self.encrypt_key = encrypt_key
        self.refresh_token = refresh_token
        self.session = session or requests.Session()
        self.message_id = None
        self.callback_token = None
        self.session_key = None
        self.status = "Idle"
        self.status_detail = "Ready."
        self.last_sender = "cli"
        self.last_prompt = ""
        self.model_name = "pycodex"
        self.output_text = ""
        self.error = ""
        self.running = False
        self.detached = False
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
            refresh_token=_read_refresh_token() or os.environ.get("FEISHU_REFRESH_TOKEN"),
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

    def set_queued(self, prompt: str, sender: str = "cli") -> None:
        self.status = "Queued"
        self.status_detail = "Waiting for pycodex."
        self.last_sender = sender or "cli"
        self.last_prompt = prompt
        self._mark_last_turn_output()
        self.error = ""
        self.running = True

    def set_snapshot(self, prompt: str, output: str) -> None:
        self.last_prompt = prompt or self.last_prompt
        self.output_text = output or self.output_text

    def detach(self) -> None:
        self.status = "Session Detached"
        self.status_detail = ""
        self.error = ""
        self.running = False
        self.detached = True

    def apply_event(self, event) -> bool:
        kind = getattr(event, "kind", None)
        if kind is None and isinstance(event, dict):
            kind = event.get("kind", "")
        kind = str(kind or "")
        payload = getattr(event, "payload", None)
        if payload is None and isinstance(event, dict):
            payload = event
        if not isinstance(payload, dict):
            payload = {}
        if kind == "turn_started":
            self.status = "Running"
            self.status_detail = "Model request started."
            was_running = self.running
            self.running = True
            prompt = payload.get("user_text") or "\n".join(
                str(item) for item in payload.get("user_texts", []) or []
            )
            if prompt:
                prompt_text = str(prompt)
                if not (was_running and prompt_text == self.last_prompt):
                    self.last_sender = "cli"
                self.last_prompt = prompt_text
            self._mark_last_turn_output()
            self.error = ""
            return False
        if kind == "assistant_delta":
            self.status = "Responding"
            self.status_detail = "Receiving assistant output."
            # self.output_text += str(payload.get("delta", ""))
            return False
        if kind == "tool_started":
            self.status = "Tool"
            self.status_detail = str(payload.get("tool_name") or "tool")
            return False
        if kind == "tool_completed":
            self.status_detail = str(
                payload.get("summary") or payload.get("tool_name") or "tool completed"
            )
            return False
        if kind == "stream_error":
            self.status = "Retrying"
            self.status_detail = str(
                payload.get("summary") or payload.get("message") or ""
            )
            return False
        if kind == "turn_completed":
            final_text = str(payload.get("output_text") or "")
            if final_text:
                self.output_text = final_text
            self.status = "Idle"
            self.status_detail = "Turn completed."
            self.running = False
            return True
        if kind in {"turn_failed", "submission_failed"}:
            self.status = "Error"
            self.error = str(payload.get("error") or kind)
            self.status_detail = self.error
            self.running = False
            return True
        if kind in {"turn_interrupted", "submission_cancelled"}:
            self.status = "Idle"
            self.status_detail = kind.replace("_", " ")
            self.running = False
            return True
        return False

    def _mark_last_turn_output(self) -> None:
        if self.output_text and not self.output_text.startswith(LAST_TURN_PREFIX):
            self.output_text = LAST_TURN_PREFIX + self.output_text

    def render(
        self, output_mode: str = CARD_OUTPUT_MODE_MARKDOWN
    ) -> "typing.Dict[str, object]":
        status_line = _escape_markdown(self.status)
        if self.status_detail:
            status_line += " - " + _escape_markdown(self.status_detail)
        if self.error:
            status_line += " - " + _escape_markdown(self.error)
        input_disabled = self.running or self.detached
        sender = _escape_markdown(self.last_sender or "cli")
        prompt = _truncate(self.last_prompt, 1200) or "-"
        output = _truncate(self.output_text, CARD_OUTPUT_LIMIT) or (
            "Waiting for output..." if self.running else "Ready."
        )
        output_content = _render_output_content(output, output_mode)
        card = {
            "schema": "2.0",
            "config": {"update_multi": True},
            "header": {
                "title": {"tag": "plain_text", "content": status_line},
                "template": _status_template(self.status),
            },
            "body": {
                "elements": [
                    {
                        "tag": "markdown",
                        "element_id": "prompt_md",
                        "content": f"> {sender}: **{_escape_code_block(prompt)}**",
                    },
                    {
                        "tag": "column_set",
                        "background_style": "grey-50",
                        "horizontal_spacing": "8px",
                        "horizontal_align": "left",
                        "columns": [
                            {
                                "tag": "column",
                                "width": "auto",
                                "elements": [
                                    {
                                        "tag": "markdown",
                                        "element_id": "answer_md",
                                        "content": output_content,
                                    },
                                ],
                                "vertical_spacing": "8px",
                                "horizontal_align": "left",
                                "vertical_align": "top",
                            }
                        ],
                    },
                ]
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
                        "content": f"Ask {self.model_name}...",
                    },
                    "behaviors": [{"type": "callback", "value": {"action": "send"}}],
                    "value": {"action": "send"},
                },
            )
        return card

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
        refresh_token = _read_refresh_token() or self.refresh_token
        if not refresh_token:
            return None
        self.refresh_token = refresh_token
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
        self._user_access_token = str(token)
        expires_in = payload.get("expires_in") or 0
        self._user_token_expires_at = time.time() + max(60, int(expires_in) - 120)
        refresh_token = payload.get("refresh_token")
        if refresh_token:
            self.refresh_token = str(refresh_token)
            os.environ["FEISHU_REFRESH_TOKEN"] = self.refresh_token
            _write_refresh_token(self.refresh_token)
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


def _status_template(status: str) -> str:
    normalized = status.lower()
    if normalized in {"error", "failed"}:
        return "red"
    if normalized in {"running", "responding", "tool", "queued", "retrying"}:
        return "blue"
    if normalized in {"session detached"}:
        return "grey"
    return "green"


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


def _is_card_content_error(exc: "BaseException") -> bool:
    return CARD_CONTENT_ERROR_MARKER in str(exc)


def _escape_markdown(text: str) -> str:
    return str(text or "").replace("`", "'")


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
    path.write_text("{0}\n".format(value), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


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
