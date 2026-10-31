import json

import pytest
import requests

from pycodex import Prompt, ResponsesModelClient, ResponsesProviderConfig


class Response:
    def __init__(self, status):
        self.status_code = status
        self.text = '{"error":{"message":"Please retry after 8 seconds."}}'

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
    delays = []

    def send(*args, **kwargs):
        attempts.append(status)
        return Response(status if len(attempts) == 1 else 200)

    async def sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(requests.Session, "send", send)
    monkeypatch.setattr("pycodex.model.asyncio.sleep", sleep)
    client = ResponsesModelClient(ResponsesProviderConfig(
        "demo", "test", "https://example.com/v1", None, stream_max_retries=1,
    ))
    await client.complete(Prompt(input=[], tools=[]))
    assert len(attempts) == 2
    assert delays == ([8.0] if status == 429 else [0.2])


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
