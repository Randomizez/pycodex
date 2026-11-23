"""Minimal ASGI Chat Completions server for standalone responses-server tests."""

import asyncio
import gzip
import json
import socket
import threading
import time
import typing
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

DEFAULT_MODEL_ID = "gpt-5.4"


def _run_uvicorn_server(server):
    asyncio.set_event_loop(asyncio.new_event_loop())
    server.run()


class CaptureStore:
    def __init__(self, root: "Path") -> "None":
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._counter_path = self._root / "counter.txt"

    def next_request_id(self) -> "int":
        if self._counter_path.exists():
            value = int(self._counter_path.read_text()) + 1
        else:
            value = 1
        self._counter_path.write_text(str(value))
        return value

    def write_capture(
        self,
        request_id: "int",
        method: "str",
        path: "str",
        headers: "typing.Dict[str, str]",
        body: "object",
    ) -> "None":
        parsed = urlsplit(path)
        safe_name = parsed.path.strip("/").replace("/", "_") or "root"
        filename = self._root / f"{request_id:03d}_{method}_{safe_name}.json"
        filename.write_text(
            json.dumps(
                {
                    "method": method,
                    "path": path,
                    "headers": headers,
                    "body": body,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


class _GzipRequestASGIMiddleware:
    def __init__(self, app: "typing.Any") -> "None":
        self.app = app

    async def __call__(
        self,
        scope: "typing.Dict[str, typing.Any]",
        receive: "typing.Any",
        send: "typing.Any",
    ) -> "None":
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = scope.get("headers") or []
        encoding = ""
        filtered = []
        for k, v in headers:
            lower_k = k.lower()
            if lower_k == b"content-encoding":
                encoding = v.decode("latin-1").strip().lower()
            elif lower_k == b"content-length":
                continue
            else:
                filtered.append((k, v))
        if encoding != "gzip":
            await self.app(scope, receive, send)
            return
        chunks = []
        while True:
            msg = await receive()
            if msg["type"] != "http.request":
                break
            chunks.append(msg.get("body", b""))
            if not msg.get("more_body", False):
                break
        raw = gzip.decompress(b"".join(chunks))
        new_scope = dict(scope)
        new_scope["headers"] = filtered + [
            (b"content-length", str(len(raw)).encode("ascii")),
            (b"x-test-decompressed-from", b"gzip"),
        ]
        replayed = [False]

        async def replay_receive() -> "typing.Dict[str, typing.Any]":
            if not replayed[0]:
                replayed[0] = True
                return {"type": "http.request", "body": raw, "more_body": False}
            return await receive()

        await self.app(new_scope, replay_receive, send)


class RunningASGITestServer:
    def __init__(
        self,
        app: "Starlette",
        host: "str" = "127.0.0.1",
        port: "typing.Union[int, None]" = None,
        decompress_gzip_requests: "bool" = True,
    ) -> "None":
        self.app = app
        asgi_app = _GzipRequestASGIMiddleware(app) if decompress_gzip_requests else app
        self.host = host
        self.port = port or _reserve_free_port()
        self._config = uvicorn.Config(
            asgi_app,
            host=self.host,
            port=self.port,
            log_level="error",
            access_log=False,
        )
        self._server = uvicorn.Server(self._config)
        self._thread = threading.Thread(
            target=_run_uvicorn_server,
            args=(self._server,),
            daemon=True,
        )

    @property
    def base_url(self) -> "str":
        return f"http://{self.host}:{self.port}"

    @property
    def server_port(self) -> "int":
        return self.port

    def start(self, timeout_seconds: "float" = 5.0) -> "None":
        self._thread.start()
        deadline = time.time() + timeout_seconds
        while not self._server.started:
            if time.time() >= deadline:
                raise RuntimeError("timed out waiting for fake ASGI server to start")
            time.sleep(0.01)

    def stop(self, timeout_seconds: "float" = 5.0) -> "None":
        self._server.should_exit = True
        self._thread.join(timeout=timeout_seconds)
        if self._thread.is_alive():
            raise RuntimeError("timed out waiting for fake ASGI server to stop")


def build_text_chunks(
    text: "str",
    model_id: "str" = DEFAULT_MODEL_ID,
    prompt_token_ids: "typing.Union[typing.List[int], None]" = None,
    decode_token_ids: "typing.Union[typing.List[int], None]" = None,
) -> "typing.List[typing.Dict[str, object]]":
    first_chunk: "typing.Dict[str, object]" = {
        "id": "chatcmpl_mock",
        "object": "chat.completion.chunk",
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": text},
                "finish_reason": None,
            }
        ],
    }
    if prompt_token_ids is not None:
        first_chunk["prompt_token_ids"] = list(prompt_token_ids)
    if decode_token_ids is not None:
        first_chunk["choices"][0]["token_ids"] = list(decode_token_ids)

    return [
        first_chunk,
        {
            "id": "chatcmpl_mock",
            "object": "chat.completion.chunk",
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        },
    ]


def build_tool_call_chunks(
    call_id: "str",
    tool_name: "str",
    arguments_parts: "typing.List[str]",
    model_id: "str" = DEFAULT_MODEL_ID,
) -> "typing.List[typing.Dict[str, object]]":
    chunks: "typing.List[typing.Dict[str, object]]" = []
    for index, part in enumerate(arguments_parts):
        chunk: "typing.Dict[str, object]" = {
            "id": "chatcmpl_mock",
            "object": "chat.completion.chunk",
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": part},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        if index == 0:
            delta = chunk["choices"][0]["delta"]["tool_calls"][0]
            delta["id"] = call_id
            delta["type"] = "function"
            delta["function"]["name"] = tool_name
        chunks.append(chunk)

    chunks.append(
        {
            "id": "chatcmpl_mock",
            "object": "chat.completion.chunk",
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls",
                }
            ],
        }
    )
    return chunks


def build_messages_text_events(
    text: "str",
    model_id: "str" = DEFAULT_MODEL_ID,
) -> "typing.List[typing.Dict[str, object]]":
    return [
        {
            "event": "message_start",
            "data": {
                "type": "message_start",
                "message": {
                    "id": "msg_mock",
                    "type": "message",
                    "model": model_id,
                    "role": "assistant",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 11},
                },
            },
        },
        {
            "event": "content_block_start",
            "data": {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "text",
                    "text": "",
                },
            },
        },
        {
            "event": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": text,
                },
            },
        },
        {
            "event": "content_block_stop",
            "data": {
                "type": "content_block_stop",
                "index": 0,
            },
        },
        {
            "event": "message_delta",
            "data": {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                },
                "usage": {"output_tokens": max(len(text), 1)},
            },
        },
        {
            "event": "message_stop",
            "data": {
                "type": "message_stop",
            },
        },
    ]


def build_messages_tool_use_events(
    call_id: "str",
    tool_name: "str",
    arguments_parts: "typing.List[str]",
    model_id: "str" = DEFAULT_MODEL_ID,
) -> "typing.List[typing.Dict[str, object]]":
    return [
        {
            "event": "message_start",
            "data": {
                "type": "message_start",
                "message": {
                    "id": "msg_mock",
                    "type": "message",
                    "model": model_id,
                    "role": "assistant",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 11},
                },
            },
        },
        {
            "event": "content_block_start",
            "data": {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": call_id,
                    "name": tool_name,
                    "input": {},
                },
            },
        },
        *[
            {
                "event": "content_block_delta",
                "data": {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": part,
                    },
                },
            }
            for part in arguments_parts
        ],
        {
            "event": "content_block_stop",
            "data": {
                "type": "content_block_stop",
                "index": 0,
            },
        },
        {
            "event": "message_delta",
            "data": {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "tool_use",
                    "stop_sequence": None,
                },
                "usage": {"output_tokens": max(len(arguments_parts), 1)},
            },
        },
        {
            "event": "message_stop",
            "data": {
                "type": "message_stop",
            },
        },
    ]


def build_test_app(
    capture_store: "CaptureStore",
    stream_chunks: "typing.Union[typing.List[typing.Dict[str, object]], typing.List[typing.List[typing.Dict[str, object]]]]",
    model_id: "str" = DEFAULT_MODEL_ID,
    messages_events: "typing.Union[typing.List[typing.Dict[str, object]], typing.List[typing.List[typing.Dict[str, object]]], None]" = None,
    chunk_delay_seconds: "float" = 0.0,
) -> "Starlette":
    chat_completion_count = 0
    messages_count = 0

    async def write_capture(request: "Request", body: "object") -> "None":
        request_id = capture_store.next_request_id()
        path = request.url.path
        if request.url.query:
            path = f"{path}?{request.url.query}"
        capture_store.write_capture(
            request_id,
            request.method,
            path,
            dict(request.headers),
            body,
        )

    async def models(request: "Request"):
        await write_capture(request, None)
        return JSONResponse({
            "object": "list",
            "data": [{"id": model_id, "object": "model"}],
        })

    async def chat_completions(request: "Request"):
        nonlocal chat_completion_count
        try:
            decoded_body: "object" = await request.json()
        except Exception:
            decoded_body = (await request.body()).decode("utf-8", errors="replace")
        await write_capture(request, decoded_body)
        chat_completion_count += 1
        selected_chunks = _select_stream_chunks(stream_chunks, chat_completion_count)

        def event_stream():
            for chunk in selected_chunks:
                if chunk_delay_seconds:
                    time.sleep(chunk_delay_seconds)
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode(
                    "utf-8"
                )
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "close",
            },
        )

    async def messages(request: "Request"):
        nonlocal messages_count
        try:
            decoded_body: "object" = await request.json()
        except Exception:
            decoded_body = (await request.body()).decode("utf-8", errors="replace")
        await write_capture(request, decoded_body)
        messages_count += 1
        selected_events = _select_stream_chunks(messages_events or [], messages_count)

        def event_stream():
            for raw_event in selected_events:
                if chunk_delay_seconds:
                    time.sleep(chunk_delay_seconds)
                if not isinstance(raw_event, dict):
                    continue
                payload = raw_event.get("data", raw_event)
                event_name = str(raw_event.get("event", "message")).strip() or "message"
                yield (
                    f"event: {event_name}\n"
                    f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                ).encode("utf-8")

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "close",
            },
        )

    async def fallback(request: "Request"):
        body: "object" = None
        if request.method != "GET":
            try:
                body = await request.json()
            except Exception:
                body = (await request.body()).decode("utf-8", errors="replace")
        await write_capture(request, body)
        return JSONResponse({"ok": True})

    return Starlette(routes=[
        Route("/models", models, methods=["GET"]),
        Route("/v1/models", models, methods=["GET"]),
        Route("/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/messages", messages, methods=["POST"]),
        Route("/v1/messages", messages, methods=["POST"]),
        Route(
            "/{path:path}", fallback,
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        ),
    ])


def build_test_server(
    capture_store: "CaptureStore",
    stream_chunks: "typing.Union[typing.List[typing.Dict[str, object]], typing.List[typing.List[typing.Dict[str, object]]]]",
    model_id: "str" = DEFAULT_MODEL_ID,
    chunk_delay_seconds: "float" = 0.0,
) -> "RunningASGITestServer":
    return RunningASGITestServer(
        build_test_app(
            capture_store,
            stream_chunks,
            model_id=model_id,
            chunk_delay_seconds=chunk_delay_seconds,
        )
    )


def build_messages_server(
    capture_store: "CaptureStore",
    messages_events: "typing.Union[typing.List[typing.Dict[str, object]], typing.List[typing.List[typing.Dict[str, object]]]]",
    model_id: "str" = DEFAULT_MODEL_ID,
    chunk_delay_seconds: "float" = 0.0,
) -> "RunningASGITestServer":
    return RunningASGITestServer(
        build_test_app(
            capture_store,
            [],
            model_id=model_id,
            messages_events=messages_events,
            chunk_delay_seconds=chunk_delay_seconds,
        )
    )


def _select_stream_chunks(
    stream_chunks: "typing.Union[typing.List[typing.Dict[str, object]], typing.List[typing.List[typing.Dict[str, object]]]]",
    request_index: "int",
) -> "typing.List[typing.Dict[str, object]]":
    if stream_chunks and isinstance(stream_chunks[0], list):
        stream_sequence = stream_chunks
        selected_index = min(request_index - 1, len(stream_sequence) - 1)
        selected = stream_sequence[selected_index]
        if isinstance(selected, list):
            return selected
    return stream_chunks


def _reserve_free_port() -> "int":
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])
    finally:
        sock.close()
