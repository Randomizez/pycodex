"""Shared HTTP authentication, responses and listener options."""

import html
import json
import secrets
import socket
import typing
from urllib.parse import urlencode

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.routing import Route

from pycodex_router.protocol import route_path, trusted_agent_request

AUTH_COOKIE_NAME = "pycodex_ws_auth"


def parse_listen(target: str) -> "typing.Tuple[str, int]":
    target_text = str(target or "").strip() or "127.0.0.1:6007"
    if ":" not in target_text:
        raise ValueError("workspace listen target must look like host:port")
    host, port_text = target_text.rsplit(":", 1)
    host = host.strip() or "127.0.0.1"
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError("workspace port must be an integer") from exc
    return host, port


async def json_object_body(
    request: Request, max_bytes: "typing.Union[int, None]" = None,
) -> "typing.Dict[str, object]":
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "application/json" and not (
        media_type.startswith("application/") and media_type.endswith("+json")
    ):
        raise HTTPException(status_code=422, detail="request body must be JSON")
    body = bytearray()
    async for chunk in request.stream():
        if max_bytes is not None and len(body) + len(chunk) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail="request body exceeds {0} bytes".format(max_bytes),
                headers={"Connection": "close"},
            )
        body.extend(chunk)
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=422, detail="invalid JSON body")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="request body must be a JSON object")
    return payload


def html_response(content: str) -> HTMLResponse:
    return HTMLResponse(
        content,
        headers={
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
        },
    )


def auth_cookie_matches(token: str, cookie: "typing.Union[str, None]") -> bool:
    if not token:
        return True
    return bool(cookie) and secrets.compare_digest(str(cookie), token)


def _render_login_shell(title: str, hostname: str, address: str) -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Log in to __LOGIN_TITLE__</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    body {
      min-height: 100vh;
      margin: 0;
      display: grid;
      place-items: center;
      background: Canvas;
      color: CanvasText;
    }
    form {
      width: min(360px, calc(100vw - 32px));
      display: grid;
      gap: 12px;
    }
    h1 {
      margin: 0;
      font-size: 22px;
    }
    .login-target {
      margin: 0;
      font-size: 13px;
      line-height: 1.6;
      overflow-wrap: anywhere;
    }
    input, button {
      min-height: 38px;
      border-radius: 7px;
      border: 1px solid color-mix(in srgb, CanvasText 18%, Canvas 82%);
      padding: 8px 10px;
      font: inherit;
    }
    button {
      cursor: pointer;
    }
    .status {
      min-height: 20px;
      color: #b42318;
      font-size: 13px;
    }
  </style>
</head>
<body>
  <form id="loginForm">
    <h1>Log in to __LOGIN_TITLE__</h1>
    <p class="login-target">Server: <strong>__LOGIN_HOST__</strong><br><code>__LOGIN_ADDRESS__</code></p>
    <label for="passwordInput">Password</label>
    <input id="passwordInput" type="password" autocomplete="current-password" autofocus>
    <button type="submit">Log in</button>
    <div id="status" class="status" role="status"></div>
  </form>
  <script>
    const form = document.getElementById("loginForm");
    const passwordInput = document.getElementById("passwordInput");
    const statusEl = document.getElementById("status");
    form.addEventListener("submit", async function(event) {
      event.preventDefault();
      statusEl.textContent = "";
      const response = await fetch(window.location.pathname + window.location.search, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({password: passwordInput.value}),
      });
      if (response.ok) {
        const result = await response.json();
        window.location.href = result.redirect;
        return;
      }
      statusEl.textContent = response.status === 403
        ? (await response.json()).error
        : "Invalid password";
    });
  </script>
</body>
</html>""".replace("__LOGIN_TITLE__", html.escape(title)).replace(
        "__LOGIN_HOST__", html.escape(hostname),
    ).replace("__LOGIN_ADDRESS__", html.escape(address))


def install_auth(
    app: Starlette, password: "typing.Union[str, None]",
    cookie_name: str = AUTH_COOKIE_NAME,
    login_title: str = "pycodex workspace",
    blocked_ips: "typing.Union[typing.Set[str], None]" = None,
    sharing=None,
) -> str:
    password_text = str(password or "")
    if not password_text and sharing is None:
        return ""
    token = secrets.token_urlsafe(32) if password_text else ""
    if sharing is not None:
        sharing.admin_token = token
        app.state.workspace_sharing = sharing
    login_failures = {}

    def ip_blocked_response() -> JSONResponse:
        return JSONResponse(
            {
                "ok": False,
                "error": (
                    "Too many incorrect passwords. This IP is blocked "
                    "until the router restarts."
                ),
            },
            status_code=403,
        )

    async def auth_middleware(request: Request, call_next):
        if blocked_ips is not None and request.client.host in blocked_ips:
            return ip_blocked_response()
        path = route_path(request.scope)
        root = request.scope.get("root_path", "")
        if sharing is not None:
            target = sharing.workspace(path)
            cookie = request.cookies.get(cookie_name)
            if path in {"/favicon.ico", "/login"}:
                return await call_next(request)
            if trusted_agent_request(request.scope) or sharing.is_admin(cookie):
                request.scope["pycodex.access_role"] = "admin"
                return await call_next(request)
            grant = None if target is None else sharing.grant(target[0], cookie)
            if grant is not None:
                if target[1].rstrip("/") == "/api/share":
                    return JSONResponse({"error": "administrator access required"}, 403)
                request.scope["pycodex.access_role"] = "visitor"
                return await call_next(request)
            if "/api/" in path or request.method not in {"GET", "HEAD"}:
                return JSONResponse({"error": "authentication required"}, 401)
            login_url = root + "/login"
            destination = root + path + ("?" + request.url.query if request.url.query else "")
            return RedirectResponse(login_url + "?" + urlencode({"next": destination}), 303)
        if path in {"/favicon.ico", "/login"} or trusted_agent_request(request.scope) or auth_cookie_matches(
            token,
            request.cookies.get(cookie_name),
        ):
            return await call_next(request)
        if path.startswith("/api/") or "/api/" in path:
            return JSONResponse(
                {"ok": False, "error": "authentication required"},
                status_code=401,
            )
        target = root + path + ("?" + request.url.query if request.url.query else "")
        return RedirectResponse(
            url=root + "/login?" + urlencode({"next": target}), status_code=303
        )

    async def login_page(request: Request) -> HTMLResponse:
        address = str(request.url.replace(
            path=request.scope.get("root_path", "") + "/", query="",
        ))
        return html_response(_render_login_shell(
            login_title, socket.gethostname(), address,
        ))

    async def login(request: Request) -> JSONResponse:
        payload = await json_object_body(request, max_bytes=4096)
        client_ip = request.client.host if blocked_ips is not None else None
        # Another login can block this IP while the request body is arriving.
        if blocked_ips is not None and client_ip in blocked_ips:
            return ip_blocked_response()
        supplied = str(payload.get("password") or "")
        accepted_token = token if password_text and secrets.compare_digest(
            supplied.encode("utf-8"), password_text.encode("utf-8"),
        ) else None
        root = request.scope.get("root_path", "")
        destination = request.query_params.get("next", root + "/")
        target = None
        if sharing is not None:
            relative = destination[len(root):] if destination.startswith(root + "/") else ""
            target = sharing.workspace(relative)
            if accepted_token is None and target is not None:
                grant = await run_in_threadpool(sharing.authenticate, target[0], supplied)
                if grant is not None:
                    accepted_token = grant.token
                    destination = root + ("/w/{0}/".format(target[0]) if sharing.multi else "/")
        if accepted_token is None:
            if blocked_ips is not None:
                login_failures[client_ip] = login_failures.get(client_ip, 0) + 1
                if login_failures[client_ip] >= 3:
                    blocked_ips.add(client_ip)
                    del login_failures[client_ip]
                    return ip_blocked_response()
            return JSONResponse(
                {"ok": False, "error": "invalid password"},
                status_code=401,
            )
        login_failures.pop(client_ip, None)
        target = destination
        if (
            not target.startswith("/")
            or target.startswith("//")
            or "\\" in target
            or any(ord(char) < 32 for char in target)
        ):
            target = root + "/"
        response = JSONResponse({"ok": True, "redirect": target})
        response.set_cookie(
            cookie_name,
            accepted_token,
            path=root + "/",
            secure=request.url.scheme == "https",
            httponly=True,
            samesite="lax",
        )
        return response

    app.add_middleware(BaseHTTPMiddleware, dispatch=auth_middleware)
    app.routes.extend([
        Route("/login", login_page, methods=["GET"]),
        Route("/login", login, methods=["POST"]),
    ])
    return token
