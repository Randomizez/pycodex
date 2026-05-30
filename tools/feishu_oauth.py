#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
import time
import typing
from urllib.parse import parse_qs, urlencode, urlparse

import requests

AUTHORIZE_URL = "https://accounts.feishu.cn/open-apis/authen/v1/authorize"
FEISHU_API_BASE = "https://open.feishu.cn/open-apis"
DEFAULT_REDIRECT_URI = "https://httpbin.org/get"
DEFAULT_SCOPES = ("offline_access", "im:message", "im:message.send_as_user")
DEFAULT_CONFIG_PATH = Path.home() / ".codex" / "config.toml"


def authorization_url(app_id: str, redirect_uri: str, scope: str) -> str:
    return AUTHORIZE_URL + "?" + urlencode(
        {
            "client_id": app_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": scope,
        }
    )


def extract_code(value: str) -> str:
    value = value.strip()
    query = parse_qs(urlparse(value).query)
    code = (query.get("code") or [""])[0]
    return code or value


def checked_payload(response) -> "typing.Dict[str, object]":
    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(
            "Feishu OAuth returned non-JSON response: status={0} body={1}".format(
                getattr(response, "status_code", "?"),
                getattr(response, "text", ""),
            )
        ) from exc
    if getattr(response, "status_code", 200) >= 400 or payload.get("code") not in (None, 0):
        raise RuntimeError("Feishu OAuth error: {0}".format(json.dumps(payload, ensure_ascii=False)))
    data = payload.get("data")
    if isinstance(data, dict):
        merged = dict(payload)
        merged.update(data)
        return merged
    return payload


def exchange_authorization_code(
    api_base: str,
    app_id: str,
    app_secret: str,
    code: str,
    redirect_uri: str,
) -> "typing.Dict[str, object]":
    response = requests.post(
        api_base.rstrip("/") + "/authen/v2/oauth/token",
        json={
            "grant_type": "authorization_code",
            "client_id": app_id,
            "client_secret": app_secret,
            "code": extract_code(code),
            "redirect_uri": redirect_uri,
        },
        timeout=20,
    )
    return checked_payload(response)


def print_token_result(payload: "typing.Dict[str, object]") -> None:
    refresh_token = payload.get("refresh_token")
    if not refresh_token:
        scope = payload.get("scope") or ""
        raise RuntimeError(
            "Feishu did not return refresh_token; check offline_access. scope={0}".format(scope)
        )
    dotenv_path = DEFAULT_CONFIG_PATH.parent / ".env"
    write_dotenv_value(dotenv_path, "FEISHU_REFRESH_TOKEN", str(refresh_token))
    print("")
    print("Wrote FEISHU_REFRESH_TOKEN to: {0}".format(dotenv_path))

    expires_in = payload.get("refresh_token_expires_in")
    if expires_in is None:
        return
    try:
        seconds = int(expires_in)
    except (TypeError, ValueError):
        return
    days = seconds / 86400.0
    expires_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + seconds))
    print("refresh_token_expires_in: {0:.1f} days, expires_at: {1}".format(days, expires_at))
    print("Restart pycodex so it reloads the updated .env.")


def write_dotenv_value(path: Path, key: str, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    replacement = "{0}={1}\n".format(key, quote_dotenv_value(value))
    lines = []
    replaced = False
    if path.exists():
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines(True)
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        prefix = "export " if stripped.startswith("export ") else ""
        candidate = stripped[len(prefix) :]
        if candidate.split("=", 1)[0].strip() == key and "=" in candidate:
            lines[index] = replacement
            replaced = True
            break
    if not replaced:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(replacement)
    path.write_text("".join(lines), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def quote_dotenv_value(value: str) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def main(argv: "typing.Union[typing.Sequence[str], None]" = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    app_id = os.environ.get("FEISHU_APP_ID") or os.environ.get("LARK_APP_ID")
    app_secret = os.environ.get("FEISHU_APP_SECRET") or os.environ.get("LARK_APP_SECRET")
    api_base = os.environ.get("FEISHU_API_BASE", FEISHU_API_BASE)
    redirect_uri = DEFAULT_REDIRECT_URI
    scope = " ".join(DEFAULT_SCOPES)

    if not app_id:
        print("FEISHU_APP_ID is required", file=sys.stderr)
        return 2
    if args:
        print("usage: python3 tools/feishu_oauth.py", file=sys.stderr)
        return 2

    print("Feishu OAuth refresh token setup")
    print("")
    print("Before opening the URL:")
    print("1. Feishu app redirect URL must include: {0}".format(redirect_uri))
    print("2. Feishu app permissions must include: {0}".format(scope))
    print("")
    print("Then open this URL and approve access:")
    print(authorization_url(app_id, redirect_uri, scope))
    print("")
    print("After Feishu redirects to httpbin:")
    print("1. Copy args.code from the JSON page, or copy the final httpbin URL.")
    print("2. Paste it below. This script will exchange it for FEISHU_REFRESH_TOKEN.")
    try:
        code = input("code> ").strip()
    except EOFError:
        code = ""
    if not code:
        print("No code entered; run this script again when ready.")
        return 0

    if not app_secret:
        print("FEISHU_APP_SECRET is required", file=sys.stderr)
        return 2

    try:
        payload = exchange_authorization_code(
            api_base,
            app_id,
            app_secret,
            code,
            redirect_uri,
        )
        print_token_result(payload)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
