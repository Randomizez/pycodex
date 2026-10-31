import json
from dataclasses import replace

import pytest
import requests

from pycodex import (
    AssistantMessage,
    Prompt,
    ResponsesModelClient,
    ResponsesProviderConfig,
    ToolCall,
    ToolResult,
    ToolSpec,
    UserMessage,
)
from pycodex.model import (
    ContextLengthExceeded,
    ResponsesApiError,
    ResponsesIncompleteError,
    ResponsesRetryableError,
)

MODELS = (
    "primary",
    "backup-one",
    "backup-two",
    "backup-three",
)


class Response:
    def __init__(self, status=200, events=(), text="unavailable"):
        self.status_code = status
        self.events = events
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_lines(self, **kwargs):
        for event in self.events:
            yield ("data: " + json.dumps(event)).encode()
            yield b""


def success():
    return Response(
        events=[
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "name": "echo",
                    "call_id": "next",
                    "arguments": '{"text":"done"}',
                },
            },
            {"type": "response.completed"},
        ]
    )


def client(monkeypatch, retries=1, cooldown=900):
    monkeypatch.setattr(ResponsesModelClient, "_retry_delay_seconds", lambda *args: 0)
    return ResponsesModelClient(
        ResponsesProviderConfig(
            model=MODELS[0],
            provider_name="test",
            base_url="https://example.com/v1",
            api_key_env=None,
            stream_max_retries=retries,
            fallback_models=MODELS[1:],
            fallback_cooldown_seconds=cooldown,
        ),
        session_id="same-session",
    )


def test_fallback_config(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        'model = "primary"\nmodel_provider = "test"\n'
        '[model_providers.test]\nbase_url = "https://example.com/v1"\n'
        'fallback_models = ["backup", "backup", "last"]\n',
        encoding="utf-8",
    )
    config = ResponsesProviderConfig.from_codex_config(path)
    assert config.fallback_models == ("backup", "last")
    assert config.fallback_cooldown_seconds == 900
    assert replace(config, model="other").fallback_models == ("backup", "last")


@pytest.mark.parametrize("value", ['"backup"', '[""]', "[1]", '["has space"]'])
def test_invalid_fallback_config(tmp_path, value):
    path = tmp_path / "config.toml"
    path.write_text(
        'model = "primary"\n[model_providers.openai]\n'
        'base_url = "https://example.com/v1"\nfallback_models = ' + value,
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fallback_models"):
        ResponsesProviderConfig.from_codex_config(path)


@pytest.mark.asyncio
async def test_ordered_fallback_preserves_history_tools_and_primary(monkeypatch):
    model_client = client(monkeypatch, cooldown=0)
    seen = []
    recovered = [False]

    def send(self, request, **kwargs):
        body = json.loads(request.body)
        seen.append((body, dict(request.headers)))
        if not recovered[0] and body["model"] != MODELS[-1]:
            return Response(503)
        return success()

    monkeypatch.setattr(requests.Session, "send", send)
    prompt = Prompt(
        input=[
            UserMessage(text="continue"),
            ToolCall(name="echo", arguments={"text": "before"}, call_id="previous"),
            ToolResult(name="echo", call_id="previous", output="before"),
        ],
        tools=[
            ToolSpec(name="echo", description="echo", input_schema={"type": "object"})
        ],
    )
    events = []
    result = await model_client.complete(prompt, events.append)
    assert [body["model"] for body, _ in seen] == [
        MODELS[0],
        MODELS[0],
        MODELS[1],
        MODELS[1],
        MODELS[2],
        MODELS[2],
        MODELS[3],
    ]
    assert isinstance(result.items[0], ToolCall)
    assert result.items[0].call_id == "next"
    first_body, first_headers = seen[0]
    for body, headers in seen:
        assert body["input"] == first_body["input"]
        tools = body.get("tools")
        if tools is None:
            tools = next(
                item["tools"]
                for item in body["input"]
                if item["type"] == "additional_tools"
            )
        assert tools == [tool.serialize() for tool in prompt.tools]
        assert body["prompt_cache_key"] == "same-session"
        assert headers["session_id"] == first_headers["session_id"]
    assert (
        len(
            [
                event
                for event in events
                if event.kind == "stream_error" and "Falling back:" in event.message
            ]
        )
        == 3
    )
    assert not any(event.kind == "model_request_started" for event in events)
    assert model_client.model == MODELS[0]
    recovered[0] = True
    await model_client.complete(prompt)
    assert seen[-1][0]["model"] == MODELS[0]


@pytest.mark.parametrize(
    "failure",
    [
        403,
        404,
        408,
        429,
        500,
        "disconnect",
        "truncated",
        "model_not_found",
    ],
)
@pytest.mark.asyncio
async def test_fallback_on_access_transport_and_stream_failures(monkeypatch, failure):
    model_client = client(monkeypatch, retries=0)
    seen = []

    def send(self, request, **kwargs):
        seen.append(json.loads(request.body)["model"])
        if len(seen) > 1:
            return success()
        if failure == "disconnect":
            raise requests.ConnectionError("offline")
        if failure == "truncated":
            return Response(
                events=[
                    {
                        "type": "response.output_text.delta",
                        "delta": "partial",
                    }
                ]
            )
        if failure == "model_not_found":
            return Response(
                events=[
                    {
                        "type": "response.failed",
                        "response": {
                            "error": {"code": failure, "message": "not available"}
                        },
                    }
                ]
            )
        return Response(failure)

    monkeypatch.setattr(requests.Session, "send", send)
    result = await model_client.complete(Prompt(input=[], tools=[]))
    assert seen == list(MODELS[:2])
    assert len(result.items) == 1
    assert not any(isinstance(item, AssistantMessage) for item in result.items)


@pytest.mark.asyncio
async def test_fallback_exhaustion_is_bounded_and_deduplicated(monkeypatch):
    model_client = client(monkeypatch)
    model_client._config = replace(model_client._config, fallback_models=MODELS * 2)
    seen = []

    def send(self, request, **kwargs):
        seen.append(json.loads(request.body)["model"])
        return Response(503)

    monkeypatch.setattr(requests.Session, "send", send)
    with pytest.raises(ResponsesRetryableError):
        await model_client.complete(Prompt(input=[], tools=[]))
    assert seen == [model for model in MODELS for _ in range(2)]


@pytest.mark.parametrize(
    "failure,error",
    [
        (Response(400), ResponsesApiError),
        (Response(401), ResponsesApiError),
        (Response(400, text="context_length_exceeded"), ContextLengthExceeded),
        (
            Response(
                events=[
                    {
                        "type": "response.incomplete",
                        "response": {
                            "incomplete_details": {"reason": "max_output_tokens"}
                        },
                    }
                ]
            ),
            ResponsesIncompleteError,
        ),
    ],
)
@pytest.mark.asyncio
async def test_fallback_does_not_mask_terminal_errors(monkeypatch, failure, error):
    model_client = client(monkeypatch)
    seen = []

    def send(self, request, **kwargs):
        seen.append(json.loads(request.body)["model"])
        return failure

    monkeypatch.setattr(requests.Session, "send", send)
    with pytest.raises(error):
        await model_client.complete(Prompt(input=[], tools=[]))
    assert seen == [MODELS[0]]


@pytest.fixture
def cooldown_clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("pycodex.model.monotonic", lambda: now[0])
    return now


@pytest.mark.parametrize("value", ["-1", "1.5", "true", '"900"'])
def test_invalid_cooldown_config(tmp_path, value):
    path = tmp_path / "config.toml"
    path.write_text(
        'model = "primary"\n[model_providers.openai]\n'
        'base_url = "https://example.com/v1"\nfallback_cooldown_seconds = ' + value,
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fallback_cooldown_seconds"):
        ResponsesProviderConfig.from_codex_config(path)


@pytest.mark.parametrize("recovers", [True, False])
@pytest.mark.asyncio
async def test_cooldown_skips_primary_and_reprobes_only_after_expiry(
    monkeypatch, cooldown_clock, recovers
):
    model_client = client(monkeypatch)
    seen, events = [], []
    primary_down = [True]

    def send(self, request, **kwargs):
        model = json.loads(request.body)["model"]
        seen.append(model)
        return Response(503) if model == MODELS[0] and primary_down[0] else success()

    monkeypatch.setattr(requests.Session, "send", send)
    prompt = Prompt(input=[UserMessage(text="continue")], tools=[])
    await model_client.complete(prompt, events.append)
    assert seen == [MODELS[0], MODELS[0], MODELS[1]]
    for _ in range(7):
        await model_client.complete(prompt, events.append)
    assert seen[3:] == [MODELS[1]] * 7
    switches = [event for event in events if event.kind == "stream_error" and event.message.startswith("Falling back:")]
    assert len(switches) == 1
    assert model_client._model_cooldowns[MODELS[0]] == 1900

    cooldown_clock[0] += 899
    await model_client.complete(prompt)
    assert seen[-1] == MODELS[1]
    primary_down[0] = not recovers
    cooldown_clock[0] += 1
    before = len(seen)
    await model_client.complete(prompt, events.append)
    assert seen[before:] == (
        [MODELS[0]] if recovers else [MODELS[0], MODELS[0], MODELS[1]]
    )
    assert model_client._model_cooldowns.get(MODELS[0]) == (None if recovers else 2800)
    before = len(seen)
    await model_client.complete(prompt)
    assert seen[before:] == [MODELS[0] if recovers else MODELS[1]]


@pytest.mark.asyncio
async def test_unavailable_backups_are_also_cooled_down(monkeypatch, cooldown_clock):
    model_client = client(monkeypatch, retries=0)
    seen = []

    def send(self, request, **kwargs):
        model = json.loads(request.body)["model"]
        seen.append(model)
        return Response(403) if model in MODELS[:2] else success()

    monkeypatch.setattr(requests.Session, "send", send)
    prompt = Prompt(input=[], tools=[])
    await model_client.complete(prompt)
    await model_client.complete(prompt)
    assert seen == [MODELS[0], MODELS[1], MODELS[2], MODELS[2]]


@pytest.mark.asyncio
async def test_all_models_cooling_down_resets_and_retries_full_chain(
    monkeypatch, cooldown_clock
):
    model_client = client(monkeypatch, retries=0)
    seen = []

    def send(self, request, **kwargs):
        seen.append(json.loads(request.body)["model"])
        return Response(503)

    monkeypatch.setattr(requests.Session, "send", send)
    prompt = Prompt(input=[], tools=[])
    with pytest.raises(ResponsesRetryableError):
        await model_client.complete(prompt)
    assert seen == list(MODELS)
    cooldown_clock[0] += 100
    events = []
    with pytest.raises(ResponsesRetryableError):
        await model_client.complete(prompt, events.append)
    assert seen == list(MODELS) * 2
    switches = [e.message for e in events if e.kind == "stream_error" and e.message.startswith("Falling back:")]
    assert switches == ["Falling back: %s -> %s" % pair for pair in zip(MODELS, MODELS[1:])]


@pytest.mark.parametrize("recovered_model", [MODELS[0], MODELS[2]])
@pytest.mark.asyncio
async def test_user_can_retry_during_cooldown_and_recover(monkeypatch, recovered_model):
    model_client = client(monkeypatch, retries=0)
    seen = []
    recovered = [False]

    def send(self, request, **kwargs):
        model = json.loads(request.body)["model"]
        seen.append(model)
        if recovered[0] and model == recovered_model:
            return success()
        return Response(503)

    monkeypatch.setattr(requests.Session, "send", send)
    prompt = Prompt(input=[], tools=[])
    with pytest.raises(ResponsesRetryableError):
        await model_client.complete(prompt)
    recovered[0] = True
    events = []
    await model_client.complete(prompt, events.append)
    assert seen[len(MODELS) :] == list(MODELS[: MODELS.index(recovered_model) + 1])
    assert recovered_model not in model_client._model_cooldowns


@pytest.mark.asyncio
async def test_rate_limit_retry_after_is_honored_before_fallback(monkeypatch):
    model_client = client(monkeypatch)
    seen = []
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    def send(self, request, **kwargs):
        model = json.loads(request.body)["model"]
        seen.append(model)
        if len(seen) == 1:
            return Response(
                429,
                text='{"error":{"code":"rate_limit_exceeded","message":'
                '"capacity reached. Please retry after 8 seconds."}}',
            )
        return success()

    monkeypatch.setattr(requests.Session, "send", send)
    monkeypatch.setattr("pycodex.model.asyncio.sleep", fake_sleep)
    events = []
    await model_client.complete(Prompt(input=[], tools=[]), events.append)
    assert seen == [MODELS[0], MODELS[0]]
    assert sleeps == [8]
    retry = next(e for e in events if e.kind == "stream_error")
    assert retry.delay_seconds == 8
    assert retry.message == "Reconnecting... 1/1"


@pytest.mark.asyncio
async def test_fallback_event_uses_provider_reason_and_attempt_timing(monkeypatch):
    model_client = client(monkeypatch)
    seen = []

    async def fake_sleep(seconds):
        assert seconds in (8, 12)

    def send(self, request, **kwargs):
        model = json.loads(request.body)["model"]
        seen.append(model)
        if model == MODELS[0]:
            delay = 8 if len(seen) == 1 else 12
            return Response(
                429,
                text='{"error":{"code":"rate_limit_exceeded","message":'
                '"The service capacity limit has been reached: 600000 tokens '
                'per 60 seconds. Please retry after %s seconds."}}' % delay,
            )
        return success()

    monkeypatch.setattr(requests.Session, "send", send)
    monkeypatch.setattr("pycodex.model.asyncio.sleep", fake_sleep)
    events = []
    await model_client.complete(Prompt(input=[], tools=[]), events.append)
    assert seen == [MODELS[0], MODELS[0], MODELS[1]]
    fallback = next(e for e in events if e.kind == "stream_error" and e.message.startswith("Falling back:"))
    assert fallback.message == "Falling back: primary -> backup-one"
    assert "retry after 12 seconds" in fallback.error


@pytest.mark.asyncio
async def test_cooldown_is_session_local_and_manual_selection_clears_it(
    monkeypatch, cooldown_clock
):
    model_client = client(monkeypatch, retries=0)
    seen = []
    primary_down = [True]

    def send(self, request, **kwargs):
        model = json.loads(request.body)["model"]
        seen.append(model)
        return Response(503) if model == MODELS[0] and primary_down[0] else success()

    monkeypatch.setattr(requests.Session, "send", send)
    prompt = Prompt(input=[], tools=[])
    await model_client.complete(prompt)
    primary_down[0] = False
    other = model_client.with_overrides(session_id="other-session")
    await other.complete(prompt)
    assert seen[-1] == MODELS[0]
    await model_client.complete(prompt)
    assert seen[-1] == MODELS[1]
    model_client.model = MODELS[0]
    await model_client.complete(prompt)
    assert seen[-1] == MODELS[0]
