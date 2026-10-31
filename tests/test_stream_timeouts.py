import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler

import pytest

from pycodex import Prompt, ResponsesModelClient, ResponsesProviderConfig, UserMessage
from pycodex.events import ModelRequestTimingEvent
from pycodex.model import ResponsesRetryableError
from pycodex.utils.compat import ThreadingHTTPServer


def make_client(base_url="https://example.com/v1", retries=0, first_data=None, idle=2000):
    return ResponsesModelClient(ResponsesProviderConfig(
        model="primary", provider_name="test", base_url=base_url, api_key_env=None,
        stream_max_retries=retries, stream_first_data_timeout_ms=first_data,
        stream_idle_timeout_ms=idle,
    ))

@pytest.mark.parametrize("value", [0, -1, True, 1.5, "100"])
def test_first_data_timeout_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="stream_first_data_timeout_ms"):
        make_client(first_data=value)

@pytest.mark.parametrize(
    "mode,protocol",
    [
        ("no_headers", "HTTP/1.1"),
        ("no_body", "HTTP/1.1"),
        ("healthy_pause", "HTTP/1.0"),
        ("healthy_pause", "HTTP/1.1"),
        ("healthy_fragment", "HTTP/1.1"),
        ("midstream_stall", "HTTP/1.1"),
    ],
)
@pytest.mark.asyncio
async def test_first_data_and_midstream_timeouts_use_different_budgets(
    mode, protocol, monkeypatch
):
    seen = []
    monkeypatch.setattr(ResponsesModelClient, "_retry_delay_seconds", lambda *args: 0)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = protocol

        def log_message(self, *args):
            pass

        def body(self, body):
            if protocol == "HTTP/1.1":
                body = ("%x\r\n" % len(body)).encode() + body + b"\r\n"
            self.wfile.write(body)
            self.wfile.flush()

        def event(self, payload):
            self.body(("data: " + json.dumps(payload) + "\n\n").encode())

        def do_POST(self):
            model = json.loads(self.rfile.read(int(self.headers["Content-Length"])))[
                "model"
            ]
            seen.append(model)
            try:
                if model == "primary" and mode == "no_headers":
                    time.sleep(0.6)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                if protocol == "HTTP/1.1":
                    self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.flush()
                if model == "primary" and mode == "no_body":
                    time.sleep(0.6)
                if model == "primary" and mode == "healthy_fragment":
                    self.body(b"data: ")
                    time.sleep(0.6)
                    self.body(b'{"type": "response.created"}\n\n')
                else:
                    self.event({"type": "response.created"})
                if model == "primary" and mode in ("healthy_pause", "midstream_stall"):
                    time.sleep(0.6)
                self.event(
                    {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "done"}],
                        },
                    }
                )
                self.event({"type": "response.completed"})
                if protocol == "HTTP/1.1":
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = make_client(
        "http://127.0.0.1:%s/v1" % server.server_port,
        retries=1,
        first_data=200,
        idle=200 if mode == "midstream_stall" else 2000,
    )
    events = []
    try:
        if mode in ("healthy_pause", "healthy_fragment"):
            result = await client.complete(Prompt(input=[UserMessage("hi")], tools=[]), events.append)
            assert result.items[0].text == "done"
        else:
            with pytest.raises(ResponsesRetryableError):
                await client.complete(Prompt(input=[UserMessage("hi")], tools=[]), events.append)
    finally:
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        thread.join(timeout=2)
    timings = [e for e in events if isinstance(e, ModelRequestTimingEvent)]
    if mode in ("healthy_pause", "healthy_fragment"):
        assert seen == ["primary"]
        assert timings[0].elapsed_seconds >= 0.6
        assert timings[0].first_data_seconds < 0.2
        assert timings[0].outcome == "completed"
    elif mode == "midstream_stall":
        assert seen == ["primary", "primary"]
        assert timings[0].first_data_seconds is not None
        assert timings[0].outcome == "failed"
    else:
        assert seen == ["primary", "primary"]
        assert timings[0].first_data_seconds is None
        assert timings[0].outcome == "failed"
