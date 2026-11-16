"""Resolve provider headers using Codex's static-then-environment precedence."""

import os
import re
import typing


def build_http_headers(
    http_headers: "typing.Mapping[str, str]",
    env_http_headers: "typing.Mapping[str, str]",
) -> "typing.Dict[str, str]":
    headers = {}
    values = list(http_headers.items())
    for name, env_var in env_http_headers.items():
        value = os.environ.get(env_var)
        if value is not None and value.strip():
            values.append((name, value))
    for name, value in values:
        if not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name):
            continue
        if re.search(r"[\x00-\x08\x0a-\x1f\x7f]", value):
            continue
        # Rust HeaderValue stores UTF-8 bytes; Python HTTP clients encode Latin-1.
        headers[name.lower()] = value.encode("utf-8").decode("latin-1")
    return headers
