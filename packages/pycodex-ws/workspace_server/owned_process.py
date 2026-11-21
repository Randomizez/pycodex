import os
import secrets
import tempfile
import threading
from contextlib import ExitStack

from .agent_access import AGENT_TOKEN_HEADER, AgentSocketAccess


def _configure_agent_access(app, cleanup, listener):
    directory = cleanup.enter_context(tempfile.TemporaryDirectory(prefix="pyco-agent-"))
    token = secrets.token_urlsafe(32)
    header_path = os.path.join(directory, "api.header")
    descriptor = os.open(header_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
        handle.write("{0}: {1}\n".format(AGENT_TOKEN_HEADER, token))
    bound_host = listener.getsockname()[0]
    app.state.agent_local_hosts = (
        () if bound_host in {"0.0.0.0", "::", ""} else (bound_host,)
    )
    app.state.agent_auth_token = token
    app.state.agent_header_path = header_path


def run_owned_server(app, host, port, input_fd):
    """Keep a server alive while its launching process holds stdin open."""
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
    server = OwnedServer(uvicorn.Config(
        AgentSocketAccess(app), host=host, port=port, loop="asyncio",
    ))
    with ExitStack() as cleanup:
        listener = server.config.bind_socket()
        cleanup.callback(listener.close)
        local_host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
        if ":" in local_host:
            local_host = "[" + local_host + "]"
        actual_port = listener.getsockname()[1]
        app.state.local_api_url = "http://{0}:{1}".format(local_host, actual_port)
        if hasattr(app.state, "router_client"):
            app.state.router_client.port = actual_port
        app.state.agent_header_path = None
        app.state.agent_auth_token = None
        app.state.agent_local_hosts = ()
        _configure_agent_access(app, cleanup, listener)
        try:
            server.run(sockets=[listener])
        finally:
            app.state.agent_header_path = None
            app.state.agent_auth_token = None
            app.state.agent_local_hosts = ()
