
import argparse
import asyncio
from dataclasses import replace
import json
import socket
import threading
import time
from typing import Iterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from .config import CompatServerConfig
from .server import ResponseServer
from .stream_router import OutcommingChatError, UnsupportedIncommingFeature
import typing


def _run_uvicorn_server(server) -> 'None':
    asyncio.set_event_loop(asyncio.new_event_loop())
    server.run()


def _format_sse_event(event_name: 'str', payload: 'typing.Dict[str, object]') -> 'bytes':
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event_name}\ndata: {data}\n\n".encode("utf-8")


def _stream_events(response_server: 'ResponseServer', request_body: 'typing.Dict[str, object]', request_headers: 'typing.Dict[str, str]') -> 'Iterator[bytes]':
    try:
        event_iter = response_server.start_response_stream(request_body, request_headers)
        for event_name, payload in event_iter:
            yield _format_sse_event(event_name, payload)
    except OutcommingChatError as exc:
        
        import traceback
        yield _format_sse_event(
            "response.failed",
            {
                "type": "response.failed",
                "response": {
                    "error": {
                        "message": '\n'.join(traceback.format_exception(exc)),
                    }
                },
            },
        )


def build_parser() -> 'argparse.ArgumentParser':
    parser = argparse.ArgumentParser(
        prog="python -m responses_server",
        description=(
            "Standalone localhost `/v1/responses` server that translates the "
            "Codex/Responses subset onto an outcomming `/v1/chat/completions` backend."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--outcomming-base-url", required=True)
    parser.add_argument("--outcomming-api-key-env", default=None)
    parser.add_argument("--model-provider", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser


def run_server(config: 'CompatServerConfig') -> 'None':
    uvicorn.run(
        ManagedResponseServer.build_app(config),
        host=config.host,
        port=config.port,
        log_level="info",
    )


def launch_chat_completion_compat_server(
    base_url: 'str',
    api_key_env: 'typing.Union[str, None]' = None,
    model_provider: 'typing.Union[str, None]' = None,
):
    config = CompatServerConfig.from_base_url(
        base_url,
        api_key_env,
        model_provider=model_provider,
    )
    server = ManagedResponseServer(config)
    server.start()
    return server


class ManagedResponseServer:
    @staticmethod
    def build_app(
        config: 'CompatServerConfig',
        session_store=None,
        stream_router=None,
    ) -> 'FastAPI':
        response_server = ResponseServer(
            config,
            session_store=session_store,
            stream_router=stream_router,
        )
        app = FastAPI(title="ResponsesCompat", version="0.1.0")
        app.state.response_server = response_server

        @app.get("/health")
        @app.get("/healthz")
        async def health() -> 'typing.Dict[str, bool]':
            return {"ok": True}

        @app.get("/models")
        @app.get("/v1/models")
        async def list_models():
            try:
                return response_server.list_models()
            except OutcommingChatError as exc:
                return JSONResponse(
                    {"error": {"message": str(exc)}},
                    status_code=502,
                )

        @app.post("/responses")
        @app.post("/v1/responses")
        async def responses(request: 'Request'):
            try:
                request_body = await request.json()
            except Exception as exc:
                return JSONResponse(
                    {"error": {"message": f"invalid JSON body: {exc}"}},
                    status_code=400,
                )
            if not isinstance(request_body, dict):
                return JSONResponse(
                    {"error": {"message": "request body must be a JSON object"}},
                    status_code=400,
                )

            request_headers = {
                str(key).lower(): str(value)
                for key, value in request.headers.items()
            }
            try:
                response_server.stream_router.validate_incomming_request(request_body)
            except UnsupportedIncommingFeature as exc:
                return JSONResponse(
                    {"error": {"message": str(exc)}},
                    status_code=501,
                )

            return StreamingResponse(
                _stream_events(response_server, request_body, request_headers),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                },
            )

        return app

    def __init__(self, config: 'CompatServerConfig') -> 'None':
        port = config.port or _reserve_free_port()
        self._config = replace(config, port=port)
        self._app = self.build_app(self._config)
        self._uvicorn_config = uvicorn.Config(
            self._app,
            host=self._config.host,
            port=self._config.port,
            log_level="error",
            access_log=False,
        )
        self._server = uvicorn.Server(self._uvicorn_config)
        self._thread = threading.Thread(
            target=_run_uvicorn_server,
            args=(self._server,),
            daemon=True,
        )

    @property
    def base_url(self) -> 'str':
        return f"http://{self._config.host}:{self._config.port}/v1"

    def start(self, timeout_seconds: 'float' = 10.0) -> 'None':
        self._thread.start()
        deadline = time.time() + timeout_seconds
        while not self._server.started:
            if time.time() >= deadline:
                raise RuntimeError(
                    "timed out waiting for managed responses server to start"
                )
            time.sleep(0.01)

    def stop(self, timeout_seconds: 'float' = 5.0) -> 'None':
        self._server.should_exit = True
        self._thread.join(timeout=timeout_seconds)
        if self._thread.is_alive():
            raise RuntimeError(
                "timed out waiting for managed responses server to stop"
            )


def main() -> 'None':
    args = build_parser().parse_args()
    run_server(
        CompatServerConfig(
            host=args.host,
            port=args.port,
            outcomming_base_url=args.outcomming_base_url,
            outcomming_api_key_env=args.outcomming_api_key_env,
            model_provider=args.model_provider,
            timeout_seconds=args.timeout_seconds,
        )
    )


if __name__ == "__main__":
    main()


def _reserve_free_port() -> 'int':
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])
    finally:
        sock.close()
