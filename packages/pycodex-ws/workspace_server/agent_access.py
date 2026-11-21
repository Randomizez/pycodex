"""Same-user HTTP access to existing local and registered-device APIs."""

import ipaddress
import secrets
from urllib.parse import unquote, urlsplit, urlunsplit

import requests
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from pycodex_router.protocol import agent_api_request

AGENT_TOKEN_HEADER = "X-Pycodex-Agent-Token"
_AGENT_TOKEN_HEADER_BYTES = b"x-pycodex-agent-token"
_DISALLOWED_TOKEN_HEADERS = frozenset({
    b"origin",
    b"sec-fetch-site",
    b"forwarded",
    b"x-forwarded-for",
    b"x-real-ip",
})


def _normalize_ip(value):
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return None
    mapped = getattr(address, "ipv4_mapped", None)
    return mapped if mapped is not None else address


def _is_local_agent_client(scope, state):
    client = scope.get("client")
    if not client or not client[0]:
        return False
    address = _normalize_ip(client[0])
    if address is None:
        return False
    if address.is_loopback:
        return True
    allowed = getattr(state, "agent_local_hosts", ()) or ()
    return any(address == _normalize_ip(host) for host in allowed)


def _match_agent_access(scope, state):
    if scope["type"] != "http":
        return False, None
    token = getattr(state, "agent_auth_token", None)
    if not token or not _is_local_agent_client(scope, state):
        return False, None
    supplied = []
    cleaned = []
    for name, value in scope.get("headers") or ():
        lower = name.lower()
        if lower in _DISALLOWED_TOKEN_HEADERS:
            return False, None
        if lower == _AGENT_TOKEN_HEADER_BYTES:
            supplied.append(value)
        else:
            cleaned.append((name, value))
    if len(supplied) != 1:
        return False, None
    try:
        candidate = supplied[0].decode("ascii").strip()
    except UnicodeDecodeError:
        return False, None
    if not secrets.compare_digest(candidate, token):
        return False, None
    return True, cleaned


class AgentSocketAccess:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        matched, cleaned_headers = _match_agent_access(scope, self.app.state)
        if not matched:
            await self.app(scope, receive, send)
            return
        scope = dict(scope, headers=cleaned_headers)
        scope["pycodex.agent_access"] = True
        if not scope["path"].startswith("/router/"):
            await self.app(scope, receive, send)
            return
        client = getattr(self.app.state, "router_client", None)
        if client is None or not client.connected or client.api_url is None:
            await JSONResponse({"error": "router API is unavailable"}, 503)(scope, receive, send)
            return
        remote = urlsplit(client.api_url)
        scope["path"] = scope["path"][len("/router"):]
        scope["root_path"] = unquote(remote.path)
        if not scope["path"].startswith(scope["root_path"] + "/") or not agent_api_request(scope):
            await JSONResponse({"error": "not a session API"}, 403)(scope, receive, send)
            return
        url = urlunsplit((
            remote.scheme, remote.netloc,
            scope["raw_path"][len(b"/router"):].decode("ascii"),
            scope["query_string"].decode("ascii"), "",
        ))
        body = await Request(scope, receive).body()

        def forward():
            with requests.Session() as session:
                session.trust_env = False
                result = session.request(
                    scope["method"], url, data=body,
                    headers={"Authorization": "Bearer " + client._identity_key,
                             "Content-Type": "application/json"},
                    allow_redirects=False, timeout=(10, 30),
                )
                return Response(
                    result.content, result.status_code,
                    media_type=result.headers.get("content-type", "application/json"),
                )

        try:
            response = await run_in_threadpool(forward)
        except requests.RequestException:
            response = JSONResponse({"error": "router API is unavailable"}, 503)
        await response(scope, receive, send)
