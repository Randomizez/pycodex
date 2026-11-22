import asyncio
import os
import secrets
import socket
import tempfile
import threading
from contextlib import AsyncExitStack, ExitStack

from .agent_access import AGENT_TOKEN_HEADER, AgentSocketAccess
from .reloader import (
    preflight_check,
    purge_reloadable_modules,
    restore_modules,
    upgrade_packages,
)


class ReloadableApp:
    """Fixed ASGI entry that restarts the workspace app on freshly imported code.

    The shell also owns the optional router tunnel so a reload keeps the node
    connected; router request handling stays in the reloadable app.
    """

    def __init__(self, build):
        self.build = build
        self.app = build()
        self.state = {}
        self.router = None
        self._lifespan = None
        self._lock = None

    async def __call__(self, scope, receive, send):
        if scope["type"] != "lifespan":
            await AgentSocketAccess(self.app)(scope, receive, send)
            return
        self._lock = asyncio.Lock()
        await receive()
        try:
            await self._start(self.app)
            if self.router is not None:
                await self.router.start()
        except BaseException as exc:
            await send({"type": "lifespan.startup.failed", "message": str(exc)})
            raise
        await send({"type": "lifespan.startup.complete"})
        await receive()
        try:
            if self.router is not None:
                await self.router.close()
            async with self._lock:
                await self._lifespan.aclose()
        except BaseException as exc:
            await send({"type": "lifespan.shutdown.failed", "message": str(exc)})
            raise
        await send({"type": "lifespan.shutdown.complete"})

    async def current_app(self, scope, receive, send):
        await self.app(scope, receive, send)

    def router_inventory(self):
        return self.app.state.router_inventory()

    async def _start(self, app):
        for name, value in self.state.items():
            setattr(app.state, name, value)
        app.state.router_client = self.router
        app.state.reload = self.reload
        self.app = app
        lifespan = AsyncExitStack()
        await lifespan.enter_async_context(app.router.lifespan_context(app))
        self._lifespan = lifespan

    async def reload(self, upgrade=None):
        """Restart like a new process after active turns finish naturally."""
        async with self._lock:
            if upgrade is not None:
                await upgrade_packages(upgrade)
            try:
                await preflight_check()
            except RuntimeError as exc:
                if upgrade is None:
                    raise
                raise RuntimeError(
                    "the upgrade is installed but was not loaded: {0}".format(exc)
                ) from exc
            await self.app.state.drain()
            await self._lifespan.aclose()
            purged = purge_reloadable_modules()
            try:
                await self._start(self.build())
            except BaseException:
                restore_modules(purged)
                await self._start(self.build())
                raise


def _configure_agent_access(state, cleanup, listener):
    directory = cleanup.enter_context(tempfile.TemporaryDirectory(prefix="pyco-agent-"))
    token = secrets.token_urlsafe(32)
    header_path = os.path.join(directory, "api.header")
    descriptor = os.open(header_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
        handle.write("{0}: {1}\n".format(AGENT_TOKEN_HEADER, token))
    bound_host = listener.getsockname()[0]
    state["agent_local_hosts"] = (
        () if bound_host in {"0.0.0.0", "::", ""} else (bound_host,)
    )
    state["agent_auth_token"] = token
    state["agent_header_path"] = header_path


def run_owned_server(build, host, port, input_fd, listen_to=None):
    """Keep a reloadable server alive while its launching process holds stdin open."""
    import uvicorn

    owner_closed = threading.Event()

    def watch_owner():
        # A buffered stdin read can hold its lock across interpreter shutdown.
        while os.read(input_fd, 4096):
            pass
        owner_closed.set()

    class OwnedServer(uvicorn.Server):
        async def on_tick(self, counter):
            # Check after startup so even an early EOF goes through shutdown.
            if owner_closed.is_set():
                self.should_exit = True
            return await super().on_tick(counter)

    watcher = threading.Thread(target=watch_owner, name="workspace-owner", daemon=True)
    watcher.start()
    shell = ReloadableApp(build)
    server = OwnedServer(uvicorn.Config(
        shell, host=host, port=port, loop="asyncio",
    ))
    with ExitStack() as cleanup:
        listener = server.config.bind_socket()
        cleanup.callback(listener.close)
        local_host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
        if ":" in local_host:
            local_host = "[" + local_host + "]"
        actual_port = listener.getsockname()[1]
        shell.state["local_api_url"] = "http://{0}:{1}".format(local_host, actual_port)
        if listen_to:
            from pycodex_router.client import RouterClient

            shell.router = RouterClient(
                shell.current_app, listen_to, socket.gethostname(), actual_port,
                shell.router_inventory,
            )
        _configure_agent_access(shell.state, cleanup, listener)
        try:
            server.run(sockets=[listener])
        finally:
            shell.app.state.agent_header_path = None
            shell.app.state.agent_auth_token = None
            shell.app.state.agent_local_hosts = ()
