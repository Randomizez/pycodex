import asyncio
from collections import deque
import logging
import os
from pathlib import Path
import re
import secrets

from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatusCode
from websockets.legacy.client import connect

from .protocol import (
    decode, device_id, encode, normalize_api_url, parse_target, send_event, tunnel_url,
)


class NodeIdInUse(Exception):
    pass


def load_node_identity_key(path):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
    except FileExistsError:
        value = path.read_text(encoding="ascii").strip()
    else:
        value = secrets.token_hex(32)
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
            handle.write(value + "\n")
    if not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("invalid router node identity key file")
    if os.name == "posix" and path.stat().st_mode & 0o077:
        os.chmod(str(path), 0o600)
    return value


def save_node_identity_key_if_absent(path, value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        return
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
    except FileExistsError:
        return
    with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
        handle.write(value + "\n")


def router_client_loop(client):
    if client is None:
        return None
    loop = getattr(client, "loop", None)
    if loop is not None:
        return loop
    task = getattr(client, "_task", None)
    if task is None:
        return None
    get_loop = getattr(task, "get_loop", None)
    if callable(get_loop):
        return get_loop()
    return getattr(task, "_loop", None)


def _client_peer_url(client):
    peer_url = getattr(client, "peer_url", None)
    if peer_url is not None:
        return peer_url
    control_url = client.control_url
    if not control_url.endswith("/tunnel/control"):
        raise ValueError("unexpected router control URL")
    peer_url = control_url[: -len("control")] + "peer"
    client.peer_url = peer_url
    return peer_url


def _client_peer_pool(client):
    pool = getattr(client, "_peer_pool", None)
    if pool is None:
        pool = deque()
        client._peer_pool = pool
    return pool


async def _acquire_peer_channel(client, allow_pooled=True):
    pool = _client_peer_pool(client)
    if allow_pooled:
        while pool:
            ch = pool.popleft()
            if ch.open:
                return ch, True
            client._channels.discard(ch)
    ch = await client._connect(_client_peer_url(client))
    client._channels.add(ch)
    return ch, False


async def _release_peer_channel(client, ch, reusable):
    pool = _client_peer_pool(client)
    if reusable and client.connected and ch.open and len(pool) < 4:
        pool.append(ch)
        return
    client._channels.discard(ch)
    try:
        await ch.close()
    except Exception:
        pass


async def peer_list_nodes(client):
    if client is None or not client.connected:
        raise OSError("router tunnel is not connected")
    for allow_pooled in (True, False):
        ch = None
        reused = False
        reusable = False
        try:
            ch, reused = await _acquire_peer_channel(client, allow_pooled=allow_pooled)
            await ch.send(
                encode({"type": "peers", "identity_key": client._identity_key})
            )
            reply = decode(await asyncio.wait_for(ch.recv(), 10))
            if reply.get("type") != "peers" or not isinstance(
                reply.get("nodes"), list
            ):
                raise ValueError("unexpected router peers response")
            reusable = True
            return reply["nodes"]
        except ConnectionClosed as exc:
            if reused:
                continue
            raise OSError(
                "peer tunnel error ({0})".format(type(exc).__name__)
            ) from exc
        except (InvalidHandshake, asyncio.TimeoutError) as exc:
            raise OSError(
                "peer tunnel error ({0})".format(type(exc).__name__)
            ) from exc
        finally:
            if ch is not None:
                await _release_peer_channel(client, ch, reusable)
    raise OSError("peer tunnel error (ConnectionClosed)")


async def peer_forward(client, target_node_id, scope, receive, send):
    if client is None or not client.connected:
        raise OSError("router tunnel is not connected")
    buffered_first = None
    for allow_pooled in (True, False):
        ch = None
        reused = False
        request_done = False
        response_started = False
        response_done = False
        disconnected = False
        try:
            ch, reused = await _acquire_peer_channel(client, allow_pooled=allow_pooled)
            await ch.send(
                encode(
                    {
                        "type": "open",
                        "identity_key": client._identity_key,
                        "target": target_node_id,
                        "scope": scope,
                    }
                )
            )

            async def requests():
                nonlocal buffered_first, request_done, disconnected
                if buffered_first is None:
                    buffered_first = await receive()
                first_msg = buffered_first
                if first_msg["type"] == "http.disconnect":
                    disconnected = True
                    return
                await send_event(ch.send, first_msg)
                if not first_msg.get("more_body", False):
                    request_done = True
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        disconnected = True
                        return
                    if not request_done:
                        await send_event(ch.send, message)
                        if not message.get("more_body", False):
                            request_done = True

            async def responses():
                nonlocal response_started, response_done
                while True:
                    event = decode(await ch.recv())
                    if event.get("type") not in {
                        "http.response.start",
                        "http.response.body",
                    }:
                        raise ValueError("invalid peer response event")
                    response_started = True
                    await send(event)
                    if event["type"] == "http.response.body" and not event.get(
                        "more_body", False
                    ):
                        response_done = True
                        return

            tasks = [
                asyncio.create_task(requests()),
                asyncio.create_task(responses()),
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
            return
        except ConnectionClosed as exc:
            if reused and not response_started and (
                buffered_first is None or not buffered_first.get("more_body", False)
            ):
                continue
            raise OSError(
                "peer tunnel error ({0})".format(type(exc).__name__)
            ) from exc
        except (InvalidHandshake, asyncio.TimeoutError) as exc:
            raise OSError(
                "peer tunnel error ({0})".format(type(exc).__name__)
            ) from exc
        finally:
            if ch is not None:
                reusable = (
                    request_done and response_done and not disconnected
                )
                await _release_peer_channel(client, ch, reusable)


async def peer_http_request(
    client,
    target_node_id,
    method,
    path,
    headers=(),
    body=b"",
    query_string=b"",
    timeout=15.0,
):
    encoded_headers = [
        [
            (
                name.encode("latin-1") if isinstance(name, str) else bytes(name)
            ).lower(),
            value.encode("latin-1") if isinstance(value, str) else bytes(value),
        ]
        for name, value in headers
    ]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "root_path": "",
        "query_string": (
            query_string.encode("ascii")
            if isinstance(query_string, str)
            else bytes(query_string)
        ),
        "headers": encoded_headers,
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", getattr(client, "port", 0)),
    }
    request_sent = False
    status = 500
    resp_headers = []
    resp_body = bytearray()

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {
                "type": "http.request",
                "body": bytes(body),
                "more_body": False,
            }
        await asyncio.Future()

    async def send(event):
        nonlocal status, resp_headers
        if event["type"] == "http.response.start":
            status = int(event["status"])
            resp_headers = [
                (
                    (
                        name.decode("latin-1")
                        if isinstance(name, bytes)
                        else str(name)
                    ).lower(),
                    (
                        value.decode("latin-1")
                        if isinstance(value, bytes)
                        else str(value)
                    ),
                )
                for name, value in event.get("headers", ())
            ]
        elif event["type"] == "http.response.body":
            chunk = event.get("body", b"")
            if chunk:
                resp_body.extend(chunk)

    coro = peer_forward(client, target_node_id, scope, receive, send)
    try:
        if timeout is None:
            await coro
        else:
            await asyncio.wait_for(coro, timeout)
    except asyncio.TimeoutError as exc:
        raise OSError(
            "peer tunnel error ({0})".format(type(exc).__name__)
        ) from exc
    return status, resp_headers, bytes(resp_body)


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
        identity_key=None,
    ):
        self.app = app
        address, self._registration_key = parse_target(target)
        self.control_url = tunnel_url(address, "control")
        self.data_url = tunnel_url(address, "data")
        self.peer_url = tunnel_url(address, "peer")
        self._identity_key = (
            secrets.token_hex(32) if identity_key is None else identity_key
        )
        self.node_id = device_id(self._identity_key)
        self.name = name
        self.port = port
        self.workspaces = workspaces
        self.refresh_seconds = refresh_seconds
        self.reconnect_seconds = reconnect_seconds
        self.connected = False
        self.api_url = None
        self.loop = None
        self._task = None
        self._channels = set()
        self._peer_pool = deque()
        self._requests = set()
        self._standby_count = 2
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
        self.loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._run())

    async def close(self):
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        await self._close_channels()
        if self._requests:
            await asyncio.gather(*tuple(self._requests))
        self.loop = None

    async def _close_channels(self):
        self._peer_pool.clear()
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

        async def maintain_standby():
            if self._standby_count <= 0:
                await asyncio.Future()
                return
            sem = asyncio.Semaphore(self._standby_count)
            state = {"supported": True}
            while state["supported"]:
                await sem.acquire()
                if not state["supported"]:
                    sem.release()
                    break
                task = asyncio.create_task(self._serve_standby(sem, state))
                self._requests.add(task)
                task.add_done_callback(self._requests.discard)
            await asyncio.Future()

        warm_task = asyncio.create_task(self._warm_peer_pool())
        self._requests.add(warm_task)
        warm_task.add_done_callback(self._requests.discard)

        tasks = [
            asyncio.create_task(self._refresh(channel)),
            asyncio.create_task(receive_requests()),
            asyncio.create_task(maintain_standby()),
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

    async def _dispatch_opening(self, channel, opening):
        if opening.get("type") != "open":
            raise ValueError("expected a router request")
        scope = opening["scope"]
        if scope["type"] not in {"http", "websocket"}:
            raise ValueError("unsupported router request type")
        scope["pycodex.agent_access"] = opening.get("agent_access") is True
        scope["pycodex.peer_access"] = opening.get("peer_access") is True

        disconnected = False
        recv_lock = asyncio.Lock()

        def _disconnect_event():
            if scope["type"] == "http":
                return {"type": "http.disconnect"}
            return {"type": "websocket.disconnect", "code": 1001}

        async def receive():
            nonlocal disconnected
            if disconnected:
                return _disconnect_event()
            async with recv_lock:
                if disconnected:
                    return _disconnect_event()
                try:
                    message = decode(await channel.recv())
                except ConnectionClosed:
                    disconnected = True
                    return _disconnect_event()
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
                if message.get("type") in {"http.disconnect", "websocket.disconnect"}:
                    disconnected = True
                return message

        async def send(message):
            await send_event(channel.send, message)

        await self.app(scope, receive, send)

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
                    await self._dispatch_opening(channel, opening)
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

    async def _serve_standby(self, sem, state):
        released = False
        try:
            async with self._connect(self.data_url) as channel:
                self._channels.add(channel)
                try:
                    await channel.send(
                        encode({"type": "standby", "identity_key": self._identity_key})
                    )
                    opening = decode(await channel.recv())
                    released = True
                    sem.release()
                    await self._dispatch_opening(channel, opening)
                finally:
                    self._channels.discard(channel)
        except ConnectionClosed as exc:
            if not released and exc.rcvd is not None and exc.rcvd.code == 1008:
                state["supported"] = False
            elif not released and self.connected:
                await asyncio.sleep(self.reconnect_seconds)
        except (OSError, asyncio.TimeoutError, InvalidHandshake):
            if not released and self.connected:
                await asyncio.sleep(self.reconnect_seconds)
        except Exception as exc:
            self._logger.error("Router standby request failed (%s)", type(exc).__name__)
            if not released and self.connected:
                await asyncio.sleep(self.reconnect_seconds)
        finally:
            if not released:
                sem.release()

    _acquire_peer_channel = _acquire_peer_channel
    _release_peer_channel = _release_peer_channel

    async def _warm_peer_pool(self):
        pool = _client_peer_pool(self)
        if not self.connected or pool:
            return
        try:
            ch = await self._connect(_client_peer_url(self))
        except Exception:
            return
        if not self.connected:
            try:
                await ch.close()
            except Exception:
                pass
            return
        self._channels.add(ch)
        pool.append(ch)

    list_peers = peer_list_nodes
    forward_peer = peer_forward
    peer_request = peer_http_request
