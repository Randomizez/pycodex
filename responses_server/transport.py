"""Downstream HTTP transport, provider boundary adapters and SSE diagnostics."""

import http.client
import json
import ssl
import time
import typing
import urllib.error
import urllib.request
from contextlib import contextmanager

from .config import CompatServerConfig
from .messages_api import MessagesAPIAdapterError, build_messages_request
from .messages_api import iter_chat_chunks as iter_chat_chunks_from_messages
from .messages_api import saw_message_stop as messages_saw_message_stop


class OutcommingChatError(RuntimeError):
    def __init__(
        self,
        message: "str",
        error_type: "typing.Union[str, None]" = None,
    ) -> "None":
        super().__init__(message)
        self.error_type = error_type


class OutcommingTransport:
    def __init__(self, config: "CompatServerConfig") -> "None":
        self.config = config

    def list_models(self) -> "typing.Dict[str, object]":
        request = urllib.request.Request(
            self.config.outcomming_models_url(),
            headers=self._build_headers(accept="application/json"),
            method="GET",
        )
        return self._request_json(request)

    def open_stream(self, outcomming_request: "typing.Dict[str, object]"):
        outcomming_api = self.config.normalized_outcomming_api()
        if outcomming_api == "messages":
            return self._open_outcomming_messages_stream(outcomming_request)
        if outcomming_api != "chat_completions":
            raise OutcommingChatError(
                f"unsupported outcomming API: {self.config.outcomming_api!r}"
            )
        return self._open_outcomming_chat_stream(outcomming_request)

    def _open_outcomming_chat_stream(
        self,
        outcomming_request: "typing.Dict[str, object]",
    ):
        request = urllib.request.Request(
            self.config.outcomming_chat_completions_url(),
            data=json.dumps(outcomming_request).encode("utf-8"),
            headers=self._build_headers(accept="text/event-stream"),
            method="POST",
        )
        with self._request_diagnostics(request) as progress:
            yield from self._read_outcomming_chat_stream(request, progress)

    def _read_outcomming_chat_stream(self, request, progress):
        try:
            with urllib.request.urlopen(
                request,
                context=ssl.create_default_context(),
                timeout=self.config.timeout_seconds,
            ) as response:
                progress["headers_at"] = time.monotonic()
                progress["status"] = response.status
                progress["phase"] = "waiting_first_chunk"
                try:
                    saw_done = False
                    for _event_name, data in self._iter_sse_events(response):
                        if not data:
                            continue
                        if data == "[DONE]":
                            saw_done = True
                            break
                        payload = json.loads(data)
                        self._received_chunk(progress)
                        if isinstance(payload, dict) and payload.get("error") is not None:
                            raise OutcommingChatError(
                                "outcomming chat stream returned an error: "
                                + json.dumps(payload["error"], ensure_ascii=False)[:500]
                            )
                        yield payload
                    if not saw_done:
                        raise OutcommingChatError(
                            "outcomming chat stream ended before [DONE]"
                        )
                except (
                    ConnectionError,
                    EOFError,
                    OSError,
                    http.client.HTTPException,
                    json.JSONDecodeError,
                ) as exc:
                    raise OutcommingChatError(
                        "outcomming chat stream failed while reading response body: "
                        f"{exc}"
                    ) from exc
        except urllib.error.HTTPError as exc:
            progress["headers_at"] = time.monotonic()
            progress["status"] = exc.code
            progress["phase"] = "reading_error_body"
            body = exc.read().decode("utf-8", errors="replace")
            raise OutcommingChatError(
                f"outcomming chat request failed with status {exc.code}: {body[:500]}"
            ) from exc
        except TimeoutError as exc:
            raise OutcommingChatError(
                "outcomming chat request timed out after "
                f"{self.config.timeout_seconds:g}s"
            ) from exc
        except urllib.error.URLError as exc:
            raise OutcommingChatError(
                f"outcomming chat request failed: {exc.reason}"
            ) from exc
        except (OSError, EOFError, http.client.HTTPException) as exc:
            raise OutcommingChatError(
                "outcomming chat request failed while opening response: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _open_outcomming_messages_stream(
        self,
        outcomming_request: "typing.Dict[str, object]",
    ):
        try:
            messages_request = build_messages_request(outcomming_request)
        except MessagesAPIAdapterError as exc:
            raise OutcommingChatError(str(exc)) from exc

        request = urllib.request.Request(
            self.config.outcomming_messages_url(),
            data=json.dumps(messages_request).encode("utf-8"),
            headers=self._build_headers(accept="text/event-stream"),
            method="POST",
        )
        with self._request_diagnostics(request) as progress:
            yield from self._read_outcomming_messages_stream(request, progress)

    def _read_outcomming_messages_stream(self, request, progress):
        try:
            with urllib.request.urlopen(
                request,
                context=ssl.create_default_context(),
                timeout=self.config.timeout_seconds,
            ) as response:
                progress["headers_at"] = time.monotonic()
                progress["status"] = response.status
                progress["phase"] = "waiting_first_chunk"
                try:
                    stream_state: "typing.Dict[str, object]" = {}
                    for event_name, data in self._iter_sse_events(response):
                        if not data:
                            continue
                        payload = json.loads(data)
                        if not isinstance(payload, dict):
                            continue
                        self._received_chunk(progress)
                        for chunk in iter_chat_chunks_from_messages(
                            event_name,
                            payload,
                            stream_state,
                        ):
                            yield chunk
                    if not messages_saw_message_stop(stream_state):
                        raise OutcommingChatError(
                            "outcomming messages stream ended before `message_stop`"
                        )
                except (
                    ConnectionError,
                    EOFError,
                    OSError,
                    http.client.HTTPException,
                    json.JSONDecodeError,
                    MessagesAPIAdapterError,
                ) as exc:
                    raise OutcommingChatError(
                        "outcomming messages stream failed while reading response body: "
                        f"{exc}"
                    ) from exc
        except urllib.error.HTTPError as exc:
            progress["headers_at"] = time.monotonic()
            progress["status"] = exc.code
            progress["phase"] = "reading_error_body"
            body = exc.read().decode("utf-8", errors="replace")
            raise OutcommingChatError(
                f"outcomming messages request failed with status {exc.code}: {body[:500]}"
            ) from exc
        except TimeoutError as exc:
            raise OutcommingChatError(
                "outcomming messages request timed out after "
                f"{self.config.timeout_seconds:g}s"
            ) from exc
        except urllib.error.URLError as exc:
            raise OutcommingChatError(
                f"outcomming messages request failed: {exc.reason}"
            ) from exc
        except (OSError, EOFError, http.client.HTTPException) as exc:
            raise OutcommingChatError(
                "outcomming messages request failed while opening response: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    @contextmanager
    def _request_diagnostics(self, request):
        started = time.monotonic()
        progress = {
            "phase": "waiting_response_headers",
            "headers_at": None,
            "first_chunk_at": None,
            "last_chunk_at": None,
            "chunks": 0,
            "status": None,
        }
        try:
            yield progress
        except OutcommingChatError as exc:
            now = time.monotonic()
            cause = exc.__cause__ or exc
            if isinstance(cause, urllib.error.URLError) and isinstance(
                cause.reason, BaseException
            ):
                cause = cause.reason

            def timing(received_at):
                if received_at is None:
                    return "not_received"
                return f"{received_at - started:.2f}s"

            idle = (
                f"{now - progress['last_chunk_at']:.2f}s"
                if progress["last_chunk_at"] is not None
                else "not_started"
            )
            details = (
                f"phase={progress['phase']}, elapsed={now - started:.2f}s, "
                f"timeout={self.config.timeout_seconds:g}s\n"
                f"headers={timing(progress['headers_at'])}, "
                f"first_chunk={timing(progress['first_chunk_at'])}, "
                f"chunks={progress['chunks']}, idle={idle}\n"
                f"status={progress['status']}, "
                f"request_bytes={len(request.data or b'')}, "
                f"exception={type(cause).__name__}"
            )
            exc.args = (f"{exc}\n{details}",)
            raise

    def _received_chunk(self, progress):
        now = time.monotonic()
        if progress["chunks"] == 0:
            progress["first_chunk_at"] = now
        progress["last_chunk_at"] = now
        progress["chunks"] += 1
        progress["phase"] = "reading_stream"

    def _request_json(
        self, request: "urllib.request.Request"
    ) -> "typing.Dict[str, object]":
        with self._request_diagnostics(request) as progress:
            return self._read_json_response(request, progress)

    def _read_json_response(self, request, progress):
        try:
            with urllib.request.urlopen(
                request,
                context=ssl.create_default_context(),
                timeout=self.config.timeout_seconds,
            ) as response:
                progress["headers_at"] = time.monotonic()
                progress["status"] = response.status
                progress["phase"] = "reading_response_body"
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            progress["headers_at"] = time.monotonic()
            progress["status"] = exc.code
            progress["phase"] = "reading_error_body"
            body = exc.read().decode("utf-8", errors="replace")
            raise OutcommingChatError(
                f"outcomming request failed with status {exc.code}: {body[:500]}"
            ) from exc
        except TimeoutError as exc:
            raise OutcommingChatError(
                "outcomming request timed out after "
                f"{self.config.timeout_seconds:g}s"
            ) from exc
        except urllib.error.URLError as exc:
            raise OutcommingChatError(
                f"outcomming request failed: {exc.reason}"
            ) from exc
        except (OSError, EOFError, http.client.HTTPException) as exc:
            raise OutcommingChatError(
                f"outcomming request failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _build_headers(self, accept: "str") -> "typing.Dict[str, str]":
        headers = {
            "Accept": accept,
            "Content-Type": "application/json",
        }
        api_key = self.config.outcomming_api_key()
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _iter_sse_events(self, response):
        event_name: "typing.Union[str, None]" = None
        data_lines: "typing.List[str]" = []

        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line == "":
                if data_lines:
                    yield event_name or "message", "\n".join(data_lines)
                event_name = None
                data_lines = []
                continue

            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].lstrip()
                continue
            if line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].lstrip())

        if data_lines:
            yield event_name or "message", "\n".join(data_lines)
