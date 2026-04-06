"""Minimal FastAPI Chat Completions server for standalone responses-server tests."""

import asyncio
import json
from pathlib import Path
import socket
import threading
import time
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import typing

DEFAULT_MODEL_ID = "gpt-5.4"


def _run_uvicorn_server(server):
    asyncio.set_event_loop(asyncio.new_event_loop())
    server.run()


class CaptureStore:
    def __init__(self, root: 'Path') -> 'None':
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._counter_path = self._root / "counter.txt"

    def next_request_id(self) -> 'int':
        if self._counter_path.exists():
            value = int(self._counter_path.read_text()) + 1
        else:
            value = 1
        self._counter_path.write_text(str(value))
        return value

    def write_capture(
        self,
        request_id: 'int',
        method: 'str',
        path: 'str',
        headers: 'typing.Dict[str, str]',
        body: 'object',
    ) -> 'None':
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


class RunningFastAPITestServer:
    def __init__(self, app: 'FastAPI', host: 'str' = "127.0.0.1", port: 'typing.Union[int, None]' = None) -> 'None':
        self.app = app
        self.host = host
        self.port = port or _reserve_free_port()
        self._config = uvicorn.Config(
            self.app,
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
    def base_url(self) -> 'str':
        return f"http://{self.host}:{self.port}"

    @property
    def server_port(self) -> 'int':
        return self.port

    def start(self, timeout_seconds: 'float' = 5.0) -> 'None':
        self._thread.start()
        deadline = time.time() + timeout_seconds
        while not self._server.started:
            if time.time() >= deadline:
                raise RuntimeError("timed out waiting for fake FastAPI server to start")
            time.sleep(0.01)

    def stop(self, timeout_seconds: 'float' = 5.0) -> 'None':
        self._server.should_exit = True
        self._thread.join(timeout=timeout_seconds)
        if self._thread.is_alive():
            raise RuntimeError("timed out waiting for fake FastAPI server to stop")


def build_text_chunks(text: 'str', model_id: 'str' = DEFAULT_MODEL_ID) -> 'typing.List[typing.Dict[str, object]]':
    return [
        {
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
        },
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
    call_id: 'str',
    tool_name: 'str',
    arguments_parts: 'typing.List[str]',
    model_id: 'str' = DEFAULT_MODEL_ID,
) -> 'typing.List[typing.Dict[str, object]]':
    chunks: 'typing.List[typing.Dict[str, object]]' = []
    for index, part in enumerate(arguments_parts):
        chunk: 'typing.Dict[str, object]' = {
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


def build_test_app(
    capture_store: 'CaptureStore',
    stream_chunks: 'typing.Union[typing.List[typing.Dict[str, object]], typing.List[typing.List[typing.Dict[str, object]]]]',
    model_id: 'str' = DEFAULT_MODEL_ID,
) -> 'FastAPI':
    app = FastAPI(title="FakeChat", version="0.1.0")
    chat_completion_count = 0

    async def write_capture(request: 'Request', body: 'object') -> 'None':
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

    @app.get("/models")
    @app.get("/v1/models")
    async def models(request: 'Request'):
        await write_capture(request, None)
        return {
            "object": "list",
            "data": [{"id": model_id, "object": "model"}],
        }

    @app.post("/chat/completions")
    @app.post("/v1/chat/completions")
    async def chat_completions(request: 'Request'):
        nonlocal chat_completion_count
        try:
            decoded_body: 'object' = await request.json()
        except Exception:
            decoded_body = (await request.body()).decode("utf-8", errors="replace")
        await write_capture(request, decoded_body)
        chat_completion_count += 1
        selected_chunks = _select_stream_chunks(stream_chunks, chat_completion_count)

        def event_stream():
            for chunk in selected_chunks:
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "close",
            },
        )

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def fallback(path: 'str', request: 'Request'):
        body: 'object' = None
        if request.method != "GET":
            try:
                body = await request.json()
            except Exception:
                body = (await request.body()).decode("utf-8", errors="replace")
        await write_capture(request, body)
        return JSONResponse({"ok": True})

    return app


def build_test_server(
    capture_store: 'CaptureStore',
    stream_chunks: 'typing.Union[typing.List[typing.Dict[str, object]], typing.List[typing.List[typing.Dict[str, object]]]]',
    model_id: 'str' = DEFAULT_MODEL_ID,
) -> 'RunningFastAPITestServer':
    return RunningFastAPITestServer(
        build_test_app(capture_store, stream_chunks, model_id=model_id)
    )


def _select_stream_chunks(
    stream_chunks: 'typing.Union[typing.List[typing.Dict[str, object]], typing.List[typing.List[typing.Dict[str, object]]]]',
    request_index: 'int',
) -> 'typing.List[typing.Dict[str, object]]':
    if stream_chunks and isinstance(stream_chunks[0], list):
        stream_sequence = stream_chunks
        selected_index = min(request_index - 1, len(stream_sequence) - 1)
        selected = stream_sequence[selected_index]
        if isinstance(selected, list):
            return selected
    return stream_chunks


def _reserve_free_port() -> 'int':
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])
    finally:
        sock.close()
