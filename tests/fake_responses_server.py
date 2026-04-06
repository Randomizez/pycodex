"""Local Responses API capture server for alignment work.

This helper supports two modes:

- fake mode: return a fixed local response while recording requests
- proxy mode: forward requests to a real upstream Responses API endpoint while
  recording both the request and the upstream response

It lives under `tests/` because it is test/support tooling, not runtime code.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
import urllib.error
import urllib.request
import typing

from pycodex.compat import ThreadingHTTPServer

DEFAULT_PORT = 8765
DEFAULT_MODEL_ID = "gpt-5.4"
DEFAULT_OUTPUT_ROOT = Path(".tmp") / "prompt_capture"
DEFAULT_RESPONSE_TEXT = "OK"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0


def build_parser() -> 'argparse.ArgumentParser':
    parser = argparse.ArgumentParser(
        prog="python -m tests.fake_responses_server",
        description=(
            "Capture local Responses API traffic. By default returns a fixed fake "
            "response; with --proxy-base-url it forwards to a real upstream."
        ),
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory used to store captured request JSON files.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port to listen on.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Model id returned from /models in fake mode.",
    )
    parser.add_argument(
        "--response-text",
        default=DEFAULT_RESPONSE_TEXT,
        help="Assistant text returned from the fake /responses SSE stream.",
    )
    parser.add_argument(
        "--proxy-base-url",
        default=None,
        help="When set, proxy requests to this upstream base URL instead of using fake responses.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="Timeout used when proxying upstream requests.",
    )
    return parser


class CaptureStore:
    def __init__(self, root: 'Path') -> 'None':
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._counter_path = self._root / "counter.txt"

    @property
    def root(self) -> 'Path':
        return self._root

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
        response_status: 'int',
        response_headers: 'typing.Dict[str, str]',
        response_body: 'object',
    ) -> 'None':
        parsed = urlparse(path)
        safe_name = parsed.path.strip("/").replace("/", "_") or "root"
        filename = self._root / f"{request_id:03d}_{method}_{safe_name}.json"
        filename.write_text(
            json.dumps(
                {
                    "method": method,
                    "path": path,
                    "headers": headers,
                    "body": body,
                    "response": {
                        "status": response_status,
                        "headers": response_headers,
                        "body": response_body,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def _decode_body(body_bytes: 'bytes', content_type: 'typing.Union[str, None]' = None) -> 'object':
    text = body_bytes.decode("utf-8", errors="replace")
    if content_type and "application/json" in content_type.lower():
        try:
            return json.loads(text)
        except Exception:
            return text
    try:
        return json.loads(text)
    except Exception:
        return text


def _write_response(
    handler: 'BaseHTTPRequestHandler',
    status: 'int',
    headers: 'typing.Dict[str, str]',
    body_bytes: 'bytes',
) -> 'None':
    handler.send_response(status)
    for key, value in headers.items():
        lowered = key.lower()
        if lowered in {"content-length", "connection", "transfer-encoding"}:
            continue
        handler.send_header(key, value)
    handler.send_header("Content-Length", str(len(body_bytes)))
    handler.end_headers()
    handler.wfile.write(body_bytes)


def _request_headers_for_proxy(headers) -> 'typing.Dict[str, str]':
    forwarded: 'typing.Dict[str, str]' = {}
    for key, value in headers.items():
        lowered = key.lower()
        if lowered in {"host", "content-length", "connection"}:
            continue
        forwarded[key] = value
    return forwarded


def _build_upstream_url(upstream_base_url: 'str', request_path: 'str') -> 'str':
    parsed_base = urlparse(upstream_base_url)
    base_origin = f"{parsed_base.scheme}://{parsed_base.netloc}"
    base_path = parsed_base.path.rstrip("/")
    parsed_request = urlparse(request_path)
    request_only_path = parsed_request.path or "/"

    if base_path and request_only_path.startswith(f"{base_path}/"):
        path = request_only_path
    elif base_path and request_only_path == base_path:
        path = request_only_path
    else:
        path = urljoin(f"{base_path}/", request_only_path.lstrip("/"))

    url = f"{base_origin}{path}"
    if parsed_request.query:
        return f"{url}?{parsed_request.query}"
    return url


def build_fake_handler(
    capture_store: 'CaptureStore',
    model_id: 'str',
    response_text: 'str',
):
    class Handler(BaseHTTPRequestHandler):
        server_version = "PromptCapture/0.1"

        def log_message(self, format: 'str', *args) -> 'None':
            del format, args
            return

        def do_GET(self) -> 'None':
            request_id = capture_store.next_request_id()
            parsed = urlparse(self.path)
            if parsed.path.endswith("/models") or parsed.path == "/models":
                payload = {
                    "object": "list",
                    "data": [{"id": model_id, "object": "model"}],
                }
                body_bytes = json.dumps(payload).encode("utf-8")
                response_headers = {"Content-Type": "application/json"}
                capture_store.write_capture(
                    request_id,
                    self.command,
                    self.path,
                    dict(self.headers),
                    None,
                    200,
                    response_headers,
                    payload,
                )
                _write_response(self, 200, response_headers, body_bytes)
                return

            payload = {"ok": True}
            body_bytes = json.dumps(payload).encode("utf-8")
            response_headers = {"Content-Type": "application/json"}
            capture_store.write_capture(
                request_id,
                self.command,
                self.path,
                dict(self.headers),
                None,
                200,
                response_headers,
                payload,
            )
            _write_response(self, 200, response_headers, body_bytes)

        def do_POST(self) -> 'None':
            length = int(self.headers.get("Content-Length", "0"))
            request_body_bytes = self.rfile.read(length)
            decoded_request_body = _decode_body(request_body_bytes, self.headers.get("Content-Type"))
            request_id = capture_store.next_request_id()
            parsed = urlparse(self.path)

            if parsed.path.endswith("/responses") or parsed.path == "/responses":
                response_text_payload = "".join(
                    [
                        "event: response.created\n"
                        f'data: {json.dumps({"type": "response.created", "response": {"id": "resp_mock", "object": "response", "status": "in_progress", "model": model_id}}, ensure_ascii=False)}\n\n',
                        "event: response.output_item.done\n"
                        f'data: {json.dumps({"type": "response.output_item.done", "item": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": response_text}]}}, ensure_ascii=False)}\n\n',
                        'event: response.completed\n'
                        'data: {"type":"response.completed","response":{"id":"resp_mock","output":[]}}\n\n',
                    ]
                )
                response_body_bytes = response_text_payload.encode("utf-8")
                response_headers = {
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                }
                capture_store.write_capture(
                    request_id,
                    self.command,
                    self.path,
                    dict(self.headers),
                    decoded_request_body,
                    200,
                    response_headers,
                    response_text_payload,
                )
                _write_response(self, 200, response_headers, response_body_bytes)
                return

            payload = {"ok": True}
            response_body_bytes = json.dumps(payload).encode("utf-8")
            response_headers = {"Content-Type": "application/json"}
            capture_store.write_capture(
                request_id,
                self.command,
                self.path,
                dict(self.headers),
                decoded_request_body,
                200,
                response_headers,
                payload,
            )
            _write_response(self, 200, response_headers, response_body_bytes)

    return Handler


def build_proxy_handler(
    capture_store: 'CaptureStore',
    upstream_base_url: 'str',
    timeout_seconds: 'float',
):
    class Handler(BaseHTTPRequestHandler):
        server_version = "PromptProxy/0.1"

        def log_message(self, format: 'str', *args) -> 'None':
            del format, args
            return

        def do_GET(self) -> 'None':
            self._forward()

        def do_POST(self) -> 'None':
            self._forward()

        def _forward(self) -> 'None':
            request_id = capture_store.next_request_id()
            length = int(self.headers.get("Content-Length", "0"))
            request_body_bytes = self.rfile.read(length) if length else b""
            decoded_request_body = _decode_body(
                request_body_bytes,
                self.headers.get("Content-Type"),
            ) if request_body_bytes else None

            target_url = _build_upstream_url(upstream_base_url, self.path)

            forwarded_headers = _request_headers_for_proxy(self.headers)
            request = urllib.request.Request(
                target_url,
                data=request_body_bytes if self.command != "GET" else None,
                headers=forwarded_headers,
                method=self.command,
            )

            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    response_status = getattr(response, "status", 200)
                    response_headers = dict(response.headers.items())
                    response_body_bytes = response.read()
            except urllib.error.HTTPError as exc:
                response_status = exc.code
                response_headers = dict(exc.headers.items())
                response_body_bytes = exc.read()

            decoded_response_body = _decode_body(
                response_body_bytes,
                response_headers.get("Content-Type"),
            )
            capture_store.write_capture(
                request_id,
                self.command,
                self.path,
                dict(self.headers),
                decoded_request_body,
                response_status,
                response_headers,
                decoded_response_body,
            )
            _write_response(self, response_status, response_headers, response_body_bytes)

    return Handler


def build_handler(
    capture_store: 'CaptureStore',
    model_id: 'str',
    response_text: 'str',
):
    """Backward-compatible alias used by existing tests."""

    return build_fake_handler(capture_store, model_id, response_text)


def main() -> 'None':
    args = build_parser().parse_args()
    capture_store = CaptureStore(Path(args.root))
    if args.proxy_base_url:
        handler = build_proxy_handler(
            capture_store,
            args.proxy_base_url,
            args.request_timeout_seconds,
        )
        mode = f"proxy -> {args.proxy_base_url}"
    else:
        handler = build_fake_handler(
            capture_store,
            args.model_id,
            args.response_text,
        )
        mode = "fake"
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"listening on http://127.0.0.1:{args.port}", flush=True)
    print(f"mode: {mode}", flush=True)
    print(f"capturing into {capture_store.root}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
