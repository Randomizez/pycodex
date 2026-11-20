"""Process-local workspace passwords and access grants."""

import asyncio
import hashlib
import re
import secrets

import anyio
from starlette.concurrency import run_in_threadpool

from pycodex_router.protocol import route_path


class ShareGrant:
    def __init__(self, salt, digest):
        self.salt = salt
        self.digest = digest
        self.token = secrets.token_urlsafe(32)
        self.revoked = asyncio.Event()


class WorkspaceSharing:
    def __init__(self, password, multi):
        self.password = password or ""
        self.multi = multi
        self.admin_token = ""
        self.grants = {}
        self.vscode = None

    def workspace(self, path):
        if not self.multi:
            return "workspace", path
        match = re.fullmatch(r"/w/([A-Za-z0-9_-]+)(/.*)?", path)
        return (match.group(1), match.group(2) or "") if match else None

    def is_admin(self, cookie):
        if self.password:
            return bool(cookie) and secrets.compare_digest(
                cookie.encode("utf-8"), self.admin_token.encode("ascii"),
            )
        return True

    def grant(self, workspace_id, cookie):
        grant = self.grants.get(workspace_id)
        if grant is not None and cookie and secrets.compare_digest(
            cookie.encode("utf-8"), grant.token.encode("ascii"),
        ):
            return grant
        return None

    def authenticate(self, workspace_id, password):
        grant = self.grants.get(workspace_id)
        if grant is None:
            return None
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), grant.salt, 100000)
        if secrets.compare_digest(digest, grant.digest):
            return grant
        return None

    async def set_password(self, workspace_id, password):
        salt = secrets.token_bytes(16)
        digest = await run_in_threadpool(
            hashlib.pbkdf2_hmac, "sha256", password.encode("utf-8"), salt, 100000,
        )
        self.revoke(workspace_id)
        self.grants[workspace_id] = ShareGrant(salt, digest)

    def revoke(self, workspace_id):
        grant = self.grants.pop(workspace_id, None)
        if grant is not None:
            grant.revoked.set()

    async def serve_websocket(self, websocket, handler):
        target = self.workspace(route_path(websocket.scope))
        cookie = websocket.cookies.get("pycodex_ws_auth")
        if self.is_admin(cookie):
            await handler(websocket)
            return
        grant = None if target is None else self.grant(target[0], cookie)
        if grant is None:
            await websocket.close(code=1008)
            return
        async with anyio.create_task_group() as tasks:
            async def watch_revocation():
                await grant.revoked.wait()
                tasks.cancel_scope.cancel()

            tasks.start_soon(watch_revocation)
            try:
                await handler(websocket)
            finally:
                tasks.cancel_scope.cancel()
        if grant.revoked.is_set():
            await websocket.close(code=1008)
