"""Carry ASGI scopes and events over WebSocket connections."""

import base64
import hashlib
import json
import re
from http.cookies import SimpleCookie
from urllib.parse import urlsplit, urlunsplit


def normalize_api_url(value):
    """A public HTTP base URL, without credentials or request parameters."""
    try:
        if not isinstance(value, str) or any(
            char.isspace() or ord(char) < 32 or char in "\\?#" for char in value
        ):
            raise ValueError()
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError()
        parsed.port
    except ValueError:
        raise ValueError(
            "API URL must be an http(s) URL without credentials, query or fragment"
        ) from None
    return value.rstrip("/")


def device_id(identity_key):
    if not isinstance(identity_key, str) or not re.fullmatch(
        "[0-9a-f]{64}", identity_key
    ):
        raise ValueError("invalid device identity key")
    digest = hashlib.sha256(bytes.fromhex(identity_key)).digest()
    return base64.b32encode(digest[:10]).decode("ascii").lower()


def route_path(scope):
    path = scope["path"]
    root = scope.get("root_path", "")
    return path[len(root) :] if root and path.startswith(root + "/") else path


def agent_api_request(scope):
    """The existing HTTP discovery, message and polling endpoints."""
    if scope["type"] != "http":
        return False
    match = re.fullmatch(
        r"(?:/n/[a-z2-7]{16})?(?:/w/[^/]+)?/api/(workspaces|sessions|session|session/message)",
        route_path(scope),
    )
    return match is not None and scope["method"] == (
        "POST" if match.group(1) == "session/message" else "GET"
    )


def one_line_agent_request(scope):
    """Only the terminal exchange endpoint; its workspace validates the temporary token."""
    return (
        scope["type"] == "http" and scope["method"] == "POST"
        and re.fullmatch(r"(?:/w/[A-Za-z0-9_-]+)?/b", route_path(scope))
        is not None
    )


def trusted_agent_request(scope):
    # This flag comes only from the private socket or an authenticated tunnel
    # envelope, never from HTTP headers or the forwarded ASGI scope.
    return scope.get("pycodex.agent_access") is True and agent_api_request(scope)


def tunnel_url(target, endpoint):
    parsed = urlsplit(target if "://" in target else "ws://" + target)
    schemes = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}
    if (
        parsed.scheme not in schemes
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("router address must be a host:port or a ws(s) URL")
    # Accessing port also validates its syntax and range.
    parsed.port
    return urlunsplit(
        (
            schemes[parsed.scheme],
            parsed.netloc,
            parsed.path.rstrip("/") + "/tunnel/" + endpoint,
            "",
            "",
        )
    )


def parse_target(target):
    """Separate a registration key from the address used on the wire."""
    try:
        has_scheme = "://" in target
        parsed = urlsplit(target if has_scheme else "ws://" + target)
        key = parsed.username
        if (
            not key
            or not re.fullmatch("[1-9A-HJ-NP-Za-km-z]{1,22}", key)
            or parsed.password is not None
        ):
            raise ValueError("invalid registration key")
        authority = parsed.netloc.rsplit("@", 1)[1]
        if not has_scheme and parsed.port is None:
            authority += ":6000"
        address = urlunsplit(
            (parsed.scheme, authority, parsed.path, parsed.query, parsed.fragment)
        )
        tunnel_url(address, "control")
    except ValueError:
        raise ValueError(
            "router address must be KEY@host[:port] or ws(s)://KEY@host[:port], "
            "using the router-generated registration key"
        ) from None
    return address, key


def encode(value):
    def binary(item):
        if isinstance(item, bytes):
            return {"__bytes__": base64.b64encode(item).decode("ascii")}
        raise TypeError("unsupported router message value")

    return json.dumps(value, default=binary, separators=(",", ":"), allow_nan=False)


def decode(payload):
    def binary(item):
        if set(item) == {"__bytes__"}:
            return base64.b64decode(item["__bytes__"], validate=True)
        return item

    def invalid_constant(value):
        raise ValueError("non-finite numbers are not valid router messages")

    value = json.loads(payload, object_hook=binary, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("router messages must be objects")
    return value


def request_scope(scope):
    """Keep only the portable ASGI fields, with a path relative to the mount."""
    root = scope.get("root_path", "")
    path = scope["path"]
    raw_path = scope.get("raw_path", path.encode("utf-8"))
    if root and path.startswith(root + "/"):
        path = path[len(root) :]
        raw_root = root.encode("utf-8")
        if raw_path.startswith(raw_root + b"/"):
            raw_path = raw_path[len(raw_root) :]
    headers = []
    for name, value in scope["headers"]:
        if name.lower() in {b"authorization", b"proxy-authorization"}:
            continue
        if name.lower() == b"cookie":
            value = b"; ".join(
                part.strip()
                for part in value.split(b";")
                if part.strip().split(b"=", 1)[0] == b"pycodex_ws_auth"
            )
        headers.append((name, value))
    result = {
        "type": scope["type"],
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": scope.get("http_version", "1.1"),
        "scheme": scope["scheme"],
        "path": path,
        "raw_path": raw_path,
        "root_path": root,
        "query_string": scope["query_string"],
        "headers": headers,
        "client": scope.get("client"),
        "server": scope.get("server"),
    }
    if scope["type"] == "http":
        result["method"] = scope["method"]
    else:
        result["subprotocols"] = scope.get("subprotocols", [])
    return result


def response_headers(headers, root, secure):
    result = []
    for name, value in headers:
        lower = name.lower()
        if lower.startswith(b"access-control-"):
            continue
        if lower == b"service-worker-allowed":
            result.append((name, (root + "/").encode("ascii")))
        elif lower == b"set-cookie":
            cookies = SimpleCookie()
            cookies.load(value.decode("latin-1"))
            if "pycodex_ws_auth" in cookies:
                cookie = cookies["pycodex_ws_auth"]
                cookie["domain"] = ""
                cookie["path"] = root + "/"
                cookie["secure"] = secure
                cookie["httponly"] = True
                cookie["samesite"] = "Lax"
                result.append((name, cookie.OutputString().encode("latin-1")))
        else:
            result.append((name, value))
    return result


async def send_event(send, message):
    """Chunk HTTP bodies; preserve WebSocket message boundaries."""
    body = message.get("body", b"")
    if message["type"] in {"http.request", "http.response.body"} and len(body) > 65536:
        for offset in range(0, len(body), 65536):
            chunk = dict(message)
            chunk["body"] = body[offset : offset + 65536]
            chunk["more_body"] = offset + 65536 < len(body) or message.get(
                "more_body", False
            )
            await send(encode(chunk))
    else:
        await send(encode(message))
