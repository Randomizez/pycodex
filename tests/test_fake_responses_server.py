
import json
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
import urllib.request

from pycodex.http_compat import ThreadingHTTPServer
from tests.fake_responses_server import CaptureStore, build_proxy_handler
import typing


def test_proxy_handler_forwards_and_captures_response(tmp_path) -> 'None':
    upstream_requests: 'typing.List[typing.Dict[str, object]]' = []

    class UpstreamHandler(BaseHTTPRequestHandler):
        def log_message(self, format: 'str', *args) -> 'None':
            del format, args
            return

        def do_POST(self) -> 'None':
            length = int(self.headers.get("Content-Length", "0"))
            body_bytes = self.rfile.read(length)
            upstream_requests.append(
                {
                    "path": self.path,
                    "headers": dict(self.headers),
                    "body": json.loads(body_bytes.decode("utf-8")),
                }
            )
            payload = "".join(
                [
                    'event: response.created\n'
                    'data: {"type":"response.created"}\n\n',
                    'event: response.output_item.done\n'
                    'data: {"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"PROXIED"}]}}\n\n',
                    'event: response.completed\n'
                    'data: {"type":"response.completed"}\n\n',
                ]
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    upstream_httpd = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream_httpd.serve_forever, daemon=True)
    upstream_thread.start()

    capture_root = tmp_path / "capture"
    capture_store = CaptureStore(capture_root)
    proxy_httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        build_proxy_handler(
            capture_store,
            f"http://127.0.0.1:{upstream_httpd.server_port}/v1",
            5.0,
        ),
    )
    proxy_thread = threading.Thread(target=proxy_httpd.serve_forever, daemon=True)
    proxy_thread.start()

    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{proxy_httpd.server_port}/v1/responses",
            data=json.dumps({"model": "demo-model", "input": []}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "Authorization": "Bearer test-key",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5.0) as response:
            body_text = response.read().decode("utf-8")
            content_type = response.headers.get("Content-Type")
    finally:
        proxy_httpd.shutdown()
        proxy_thread.join(timeout=5)
        proxy_httpd.server_close()
        upstream_httpd.shutdown()
        upstream_thread.join(timeout=5)
        upstream_httpd.server_close()

    assert content_type == "text/event-stream"
    assert "PROXIED" in body_text
    assert len(upstream_requests) == 1
    upstream_request = upstream_requests[0]
    assert upstream_request["path"] == "/v1/responses"
    assert upstream_request["body"] == {"model": "demo-model", "input": []}
    upstream_headers = upstream_request["headers"]
    assert upstream_headers["Content-Type"] == "application/json"
    assert upstream_headers["Accept"] == "text/event-stream"
    assert upstream_headers["Authorization"] == "Bearer test-key"

    capture_files = sorted(capture_root.glob("*_POST_*.json"))
    assert len(capture_files) == 1
    capture = json.loads(capture_files[0].read_text())
    assert capture["path"] == "/v1/responses"
    assert capture["body"] == {"model": "demo-model", "input": []}
    assert capture["response"]["status"] == 200
    assert capture["response"]["headers"]["Content-Type"] == "text/event-stream"
    assert "PROXIED" in capture["response"]["body"]
