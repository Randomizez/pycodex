"""Shared Feishu endpoints and refresh-token storage."""

import os
import typing
from pathlib import Path

FEISHU_API_BASE = "https://open.feishu.cn/open-apis"
FEISHU_DOMAIN = "https://open.feishu.cn"
FEISHU_REFRESH_TOKEN_FILE = "~/.codex/.feishu_refresh_token"


def refresh_token_path() -> Path:
    return Path(FEISHU_REFRESH_TOKEN_FILE).expanduser()


def read_refresh_token() -> "typing.Union[str, None]":
    path = refresh_token_path()
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8", errors="replace").strip()
    return value or None


def write_refresh_token(value: str) -> None:
    path = refresh_token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        path.chmod(0o600)
        stream.write("{0}\n".format(value))
