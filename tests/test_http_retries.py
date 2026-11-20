import json

import pytest
import requests

from pycodex import Prompt, ResponsesModelClient, ResponsesProviderConfig


class Response:
    def __init__(self, status):
        self.status_code = status
        self.text = '{"error":{"message":"Please retry after 50ms."}}'

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_lines(self, **kwargs):
        yield ('data: ' + json.dumps({"type": "response.completed"})).encode()
        yield b""


@pytest.mark.parametrize("status", [408, 429, 503])
@pytest.mark.asyncio
async def test_http_errors_use_retry_budget(monkeypatch, status):
    attempts = []

    def send(*args, **kwargs):
        attempts.append(status)
        return Response(status if len(attempts) == 1 else 200)

    monkeypatch.setattr(requests.Session, "send", send)
    client = ResponsesModelClient(ResponsesProviderConfig(
        "demo", "test", "https://example.com/v1", None, stream_max_retries=1,
    ))
    events = []
    await client.complete(Prompt(input=[], tools=[]), events.append)
    assert len(attempts) == 2
    delays = [event.delay_seconds for event in events if event.kind == "stream_error"]
    assert delays == ([0.05] if status == 429 else [0.2])


@pytest.mark.parametrize("text,seconds", [
    ("try again in 250ms", 0.25),
    ("retry after 8 seconds", 8.0),
    ("retry after 1.5s", 1.5),
])
def test_rate_limit_delay_units(text, seconds):
    client = ResponsesModelClient(ResponsesProviderConfig(
        "demo", "test", "https://example.com/v1", None,
    ))
    assert client._try_parse_retry_after_seconds("rate_limit_exceeded", text) == seconds
