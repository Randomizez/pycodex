import asyncio
import logging
import secrets

from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatusCode
from websockets.legacy.client import connect

from .protocol import (
    decode, device_id, encode, normalize_api_url, parse_target, send_event, tunnel_url,
)


class NodeIdInUse(Exception):
    pass


class RouterClient:
    def __init__(
        self,
        app,
        target,
        name,
        port,
        workspaces,
        refresh_seconds=2.0,
        reconnect_seconds=1.0,
    ):
        self.app = app
        address, self._registration_key = parse_target(target)
        self.control_url = tunnel_url(address, "control")
        self.data_url = tunnel_url(address, "data")
        self._identity_key = secrets.token_hex(32)
        self.node_id = device_id(self._identity_key)
        self.name = name
        self.port = port
        self.workspaces = workspaces
        self.refresh_seconds = refresh_seconds
        self.reconnect_seconds = reconnect_seconds
        self.connected = False
        self.api_url = None
        self._task = None
        self._channels = set()
        self._requests = set()
        self._logger = logging.getLogger(__name__)

    def _connect(self, url):
        return connect(
            url,
            extra_headers={"Authorization": "Bearer " + self._registration_key},
            open_timeout=10,
            close_timeout=2,
            ping_interval=20,
            ping_timeout=20,
            max_size=32 * 1024 * 1024,
            max_queue=4,
        )

    async def start(self):
        if self._task is not None:
            raise RuntimeError("router client is already started")
        self._task = asyncio.create_task(self._run())

    async def close(self):
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        await self._close_channels()
        if self._requests:
            await asyncio.gather(*tuple(self._requests))

    async def _close_channels(self):
        await asyncio.gather(*(channel.close() for channel in tuple(self._channels)))

    async def _run(self):
        delay = self.reconnect_seconds
        while True:
            failure = None
            try:
                async with self._connect(self.control_url) as channel:
                    await self._register(channel)
                    self.connected = True
                    delay = self.reconnect_seconds
                    self._logger.info("Workspace connected to router as %s", self.name)
                    await self._session(channel)
            except asyncio.CancelledError:
                # close() performs cleanup after stopping this loop.
                raise
            except Exception as exc:
                failure = exc
            finally:
                self.connected = False
                self.api_url = None

            try:
                await self._close_channels()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failure = exc

            # Retry every failure, including registration and connection cleanup.
            # Peer-supplied close reasons and URLs must not enter the log.
            reason = (
                type(failure).__name__ if failure is not None else "connection closed"
            )
            if isinstance(failure, ConnectionClosed) and failure.rcvd is not None:
                reason += ", close_code={0}".format(failure.rcvd.code)
            elif isinstance(failure, InvalidStatusCode):
                reason += ", status_code={0}".format(failure.status_code)
            self._logger.warning(
                "Router node %s disconnected (%s); retrying in %g seconds",
                self.name,
                reason,
                delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)

    async def _register(self, channel):
        await channel.send(
            encode(
                {
                    "type": "register",
                    "version": 2,
                    "identity_key": self._identity_key,
                    "name": self.name,
                    "port": self.port,
                    "workspaces": self.workspaces(),
                }
            )
        )
        reply = decode(await asyncio.wait_for(channel.recv(), 10))
        if reply.get("type") == "rejected":
            raise NodeIdInUse("router node id is still registered")
        if reply.get("type") != "registered" or reply.get("id") != self.node_id:
            raise ValueError("unexpected router registration response")
        api_url = reply.get("api_url")
        self.api_url = normalize_api_url(api_url) if api_url is not None else None

    async def _session(self, channel):
        async def receive_requests():
            async for payload in channel:
                message = decode(payload)
                if message.get("type") != "connect":
                    raise ValueError("unknown router control message")
                task = asyncio.create_task(self._serve(message["stream_id"]))
                self._requests.add(task)
                task.add_done_callback(self._requests.discard)

        tasks = [
            asyncio.create_task(self._refresh(channel)),
            asyncio.create_task(receive_requests()),
        ]
        try:
            done, _pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _refresh(self, channel):
        while True:
            await asyncio.sleep(self.refresh_seconds)
            await channel.send(
                encode(
                    {
                        "type": "workspaces",
                        "workspaces": self.workspaces(),
                    }
                )
            )

    async def _serve(self, stream_id):
        try:
            async with self._connect(self.data_url) as channel:
                self._channels.add(channel)
                try:
                    await channel.send(
                        encode(
                            {"stream_id": stream_id, "identity_key": self._identity_key}
                        )
                    )
                    opening = decode(await asyncio.wait_for(channel.recv(), 10))
                    if opening.get("type") != "open":
                        raise ValueError("expected a router request")
                    scope = opening["scope"]
                    if scope["type"] not in {"http", "websocket"}:
                        raise ValueError("unsupported router request type")
                    scope["pycodex.agent_access"] = opening.get("agent_access") is True

                    async def receive():
                        try:
                            message = decode(await channel.recv())
                        except ConnectionClosed:
                            if scope["type"] == "http":
                                return {"type": "http.disconnect"}
                            return {"type": "websocket.disconnect", "code": 1001}
                        allowed = (
                            {"http.request", "http.disconnect"}
                            if scope["type"] == "http"
                            else {
                                "websocket.connect",
                                "websocket.receive",
                                "websocket.disconnect",
                            }
                        )
                        if message.get("type") not in allowed:
                            raise ValueError("invalid router request event")
                        return message

                    async def send(message):
                        await send_event(channel.send, message)

                    await self.app(scope, receive, send)
                finally:
                    self._channels.discard(channel)
        except (
            ConnectionClosed,
            OSError,
            asyncio.TimeoutError,
            InvalidHandshake,
        ) as exc:
            self._logger.warning("Router request disconnected (%s)", type(exc).__name__)
        except Exception as exc:
            self._logger.error("Router request failed (%s)", type(exc).__name__)
