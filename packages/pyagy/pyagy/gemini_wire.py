"""AGY Gemini wire transport and local ManagedResponseServer integration."""

import base64
from copy import deepcopy
from dataclasses import replace
import gzip
import http.client
import io
import json
import ssl
import threading
import time
import typing
import urllib.error
import urllib.parse
import urllib.request

import uvicorn

from pycodex.model import (
    DEFAULT_ORIGINATOR,
    Prompt,
    ResponsesModelClient,
    ResponsesProviderConfig,
)
from pycodex.protocol import ToolResult, UserMessage
from responses_server.app import ManagedResponseServer, _reserve_free_port, _run_uvicorn_server
from responses_server.config import CompatServerConfig
from responses_server.gemini_api import (
    GeminiAPIAdapterError,
    SKIP_THOUGHT_SIGNATURE_VALIDATOR,
    _content_parts,
    continuation_chunk as gemini_continuation_chunk,
    iter_chat_chunks as iter_chat_chunks_from_gemini,
)
from responses_server.request_translation import inline_history_instructions
from responses_server.server import PreparedResponse, ResponseServer
from responses_server.session_store import SessionStore
from responses_server.stream_router import StreamRouter
from responses_server.transport import (
    OutcommingChatError,
    OutcommingTransport,
    encode_gemini_request_body,
    is_gzip_request_rejected_http_error,
    mark_gzip_unsupported,
    should_gzip_gemini_request,
)

from .context import AGY_COMPACTION_PROMPT_NAME, SYSTEM_MESSAGE_PREAMBLE
from .models import (
    AGY_USER_AGENT,
    OFFICIAL_MODELS,
    normalize_effort_level,
    resolve_agy_model,
)


class AgyResponsesModelClient(ResponsesModelClient):
    """ResponsesModelClient propagating ToolResult.is_error and AGY effort metadata."""

    def __init__(
        self,
        config: "ResponsesProviderConfig",
        timeout_seconds: "float" = 120.0,
        session_id: "typing.Union[str, None]" = None,
        originator: "str" = DEFAULT_ORIGINATOR,
        user_agent: "typing.Union[str, None]" = None,
        openai_subagent: "typing.Union[str, None]" = None,
        config_effort: "typing.Union[str, None]" = None,
        explicit_effort: "typing.Union[str, None]" = None,
        effort_change_callback: "typing.Optional[typing.Callable[[typing.Optional[str]], None]]" = None,
    ) -> "None":
        super().__init__(
            config=config,
            timeout_seconds=timeout_seconds,
            session_id=session_id,
            originator=originator,
            user_agent=user_agent,
            openai_subagent=openai_subagent,
        )
        self._config_effort = config_effort
        self._explicit_effort = explicit_effort
        self._effort_change_callback = effort_change_callback

    def set_effort_change_callback(
        self,
        callback: "typing.Optional[typing.Callable[[typing.Optional[str]], None]]",
    ) -> "None":
        self._effort_change_callback = callback

    @property
    def explicit_effort(self) -> "typing.Union[str, None]":
        return self._explicit_effort

    @explicit_effort.setter
    def explicit_effort(self, effort: "typing.Union[str, None]") -> "None":
        self._explicit_effort = effort

    @ResponsesModelClient.reasoning_effort.setter
    def reasoning_effort(self, effort: "typing.Union[str, None]") -> "None":
        if effort is not None:
            normalize_effort_level(effort)
        ResponsesModelClient.reasoning_effort.fset(self, effort)
        self._explicit_effort = effort
        if self._effort_change_callback is not None:
            self._effort_change_callback(effort)

    def list_models_sync(self) -> "typing.List[str]":
        seen: "typing.Set[str]" = set()
        models: "typing.List[str]" = []
        for candidate in (self.model,) + tuple(slug for slug, _ in OFFICIAL_MODELS):
            if candidate and candidate not in seen:
                seen.add(candidate)
                models.append(candidate)
        return models

    def with_overrides(
        self,
        model: "typing.Union[str, None]" = None,
        reasoning_effort: "typing.Union[str, None]" = None,
        session_id: "typing.Union[str, None]" = None,
        openai_subagent: "typing.Union[str, None]" = None,
    ) -> "AgyResponsesModelClient":
        next_explicit = (
            self._explicit_effort if reasoning_effort is None else reasoning_effort
        )
        return AgyResponsesModelClient(
            config=replace(
                self._config,
                model=model or self.model,
                reasoning_effort=(
                    self._config.reasoning_effort
                    if reasoning_effort is None
                    else reasoning_effort
                ),
            ),
            timeout_seconds=self._timeout_seconds,
            session_id=self._session_id if session_id is None else session_id,
            originator=self.recording_metadata.originator,
            user_agent=self._user_agent,
            openai_subagent=(
                self._openai_subagent if openai_subagent is None else openai_subagent
            ),
            config_effort=self._config_effort,
            explicit_effort=next_explicit,
        )

    def _build_payload(self, prompt: "Prompt") -> "typing.Dict[str, object]":
        payload = super()._build_payload(prompt)
        error_call_ids = [
            item.call_id
            for item in prompt.input
            if isinstance(item, ToolResult) and item.is_error
        ]
        if error_call_ids:
            payload["error_call_ids"] = error_call_ids
        if self._config_effort is not None:
            payload["agy_config_effort"] = self._config_effort
        if self._explicit_effort is not None:
            payload["agy_explicit_effort"] = self._explicit_effort
        if (
            prompt.input
            and isinstance(prompt.input[-1], UserMessage)
            and prompt.input[-1].name == AGY_COMPACTION_PROMPT_NAME
        ):
            payload["agy_is_compaction"] = True
        return payload


def _sort_json_keys(value: "object") -> "object":
    if isinstance(value, dict):
        return {k: _sort_json_keys(value[k]) for k in sorted(value.keys())}
    if isinstance(value, list):
        return [_sort_json_keys(item) for item in value]
    return deepcopy(value)


def _unsigned_agy_function_call_part(call: "typing.Dict[str, object]") -> "typing.Dict[str, object]":
    function = call["function"]
    raw_args = function.get("arguments", "{}")
    arguments = json.loads(raw_args) if isinstance(raw_args, str) else deepcopy(raw_args)
    if not isinstance(arguments, dict):
        raise GeminiAPIAdapterError("Gemini function call arguments must decode to an object")
    fc_obj: "typing.Dict[str, object]" = {
        "args": _sort_json_keys(arguments),
        "id": call["id"],
        "name": function["name"],
    }
    return {"functionCall": fc_obj}


def build_agy_gemini_request(
    request: "typing.Dict[str, object]",
) -> "typing.Tuple[str, typing.Dict[str, object]]":
    raw_model = str(request.get("model", "") or "").strip()
    config_effort = (
        str(request["agy_config_effort"])
        if request.get("agy_config_effort") is not None
        else None
    )
    explicit_effort = (
        str(request["agy_explicit_effort"])
        if request.get("agy_explicit_effort") is not None
        else None
    )
    model_spec = resolve_agy_model(
        raw_model,
        config_effort=config_effort,
        explicit_effort=explicit_effort,
    )
    is_compaction = bool(request.get("agy_is_compaction"))

    error_call_ids = set(str(cid) for cid in (request.get("error_call_ids") or []))
    contents: "typing.List[typing.Dict[str, object]]" = []
    system_parts: "typing.List[typing.Dict[str, object]]" = []
    calls: "typing.Dict[str, str]" = {}
    prev_was_tool = False

    for message in inline_history_instructions(request.get("messages", [])):
        orig_role = message["role"]
        if orig_role in {"system", "developer"}:
            system_parts.extend(_content_parts(message.get("content")))
            prev_was_tool = False
            continue

        if orig_role == "assistant":
            tool_calls = message.get("tool_calls") or []
            has_tool_calls = bool(tool_calls)
            active_call_ids: "typing.Set[str]" = set()
            for call in tool_calls:
                calls[call["id"]] = call["function"]["name"]
                active_call_ids.add(call["id"])

            native = [
                entry
                for entry in (message.get("provider_data") or [])
                if isinstance(entry, dict) and entry.get("provider") == "gemini"
            ]
            parts: "typing.List[typing.Dict[str, object]]" = []
            if native:
                # Collect any trailing empty-text thoughtSignature attached to a call_id.
                deferred_call_sigs: "typing.Dict[str, str]" = {}
                for entry in native:
                    part = entry.get("part", {})
                    cid = entry.get("call_id")
                    if (
                        cid
                        and "functionCall" not in part
                        and part.get("text", "") == ""
                        and "thoughtSignature" in part
                    ):
                        deferred_call_sigs[cid] = part["thoughtSignature"]

                native_call_ids: "typing.Set[str]" = set()
                thought_chunks: "typing.List[str]" = []
                thought_sig_bytes = bytearray()
                text_chunks: "typing.List[str]" = []
                text_sig_bytes = bytearray()
                fc_parts: "typing.List[typing.Dict[str, object]]" = []

                for entry in native:
                    cid = entry.get("call_id")
                    if cid is not None and cid not in active_call_ids:
                        continue
                    part = deepcopy(entry["part"])
                    if "functionCall" in part:
                        if cid is not None:
                            native_call_ids.add(cid)
                        fc_raw = part["functionCall"]
                        fc_clean: "typing.Dict[str, object]" = {
                            "args": _sort_json_keys(fc_raw.get("args", {})),
                        }
                        if "id" in fc_raw:
                            fc_clean["id"] = fc_raw["id"]
                        elif cid:
                            fc_clean["id"] = cid
                        fc_clean["name"] = fc_raw["name"]
                        out_part: "typing.Dict[str, object]" = {"functionCall": fc_clean}
                        sig = part.get("thoughtSignature") or (
                            deferred_call_sigs.get(cid) if cid else None
                        )
                        if sig is not None:
                            out_part["thoughtSignature"] = sig
                        fc_parts.append(out_part)
                    elif part.get("thought") is True:
                        t_txt = part.get("text", "")
                        if isinstance(t_txt, str) and t_txt:
                            thought_chunks.append(t_txt)
                        t_sig = part.get("thoughtSignature")
                        if isinstance(t_sig, str) and t_sig:
                            thought_sig_bytes.extend(base64.b64decode(t_sig))
                    else:
                        if cid and part.get("text", "") == "" and "thoughtSignature" in part:
                            continue
                        c_txt = part.get("text", "")
                        if isinstance(c_txt, str) and c_txt:
                            text_chunks.append(c_txt)
                        c_sig = part.get("thoughtSignature")
                        if isinstance(c_sig, str) and c_sig:
                            text_sig_bytes.extend(base64.b64decode(c_sig))

                for call in tool_calls:
                    if call["id"] not in native_call_ids:
                        fc_parts.append(_unsigned_agy_function_call_part(call))

                combined_thought = "".join(thought_chunks)
                combined_text = "".join(text_chunks)
                if fc_parts:
                    if combined_thought and len(thought_sig_bytes) > 0:
                        parts.append({"text": combined_thought, "thought": True})
                    if combined_text:
                        txt_entry: "typing.Dict[str, object]" = {"text": combined_text}
                        if text_sig_bytes:
                            txt_entry["thoughtSignature"] = base64.b64encode(
                                bytes(text_sig_bytes)
                            ).decode("ascii")
                        parts.append(txt_entry)
                    has_any_sig = len(thought_sig_bytes) > 0 or any(
                        "thoughtSignature" in fp for fp in fc_parts
                    )
                    if not has_any_sig:
                        fc_parts[0]["thoughtSignature"] = SKIP_THOUGHT_SIGNATURE_VALIDATOR
                    parts.extend(fc_parts)
                else:
                    if combined_thought:
                        parts.append({"text": combined_thought, "thought": True})
                    merged_sig = bytes(thought_sig_bytes) + bytes(text_sig_bytes)
                    if combined_text or merged_sig:
                        txt_entry = {}
                        if combined_text:
                            txt_entry["text"] = combined_text
                        if merged_sig:
                            txt_entry["thoughtSignature"] = base64.b64encode(
                                merged_sig
                            ).decode("ascii")
                        parts.append(txt_entry)
            else:
                reasoning_text = message.get("reasoning")
                if not has_tool_calls and isinstance(reasoning_text, str) and reasoning_text:
                    parts.append({"text": reasoning_text, "thought": True})
                parts.extend(_content_parts(message.get("content")))
                fallback_fc_parts = [
                    _unsigned_agy_function_call_part(call) for call in tool_calls
                ]
                if fallback_fc_parts:
                    fallback_fc_parts[0]["thoughtSignature"] = SKIP_THOUGHT_SIGNATURE_VALIDATOR
                    parts.extend(fallback_fc_parts)

            if parts:
                contents.append({"parts": parts, "role": "model"})
            prev_was_tool = False
            continue

        if orig_role == "tool":
            call_id = str(message["tool_call_id"])
            if call_id not in calls:
                raise GeminiAPIAdapterError("Gemini tool result has no matching function call")
            output_parts = _content_parts(message.get("content"))
            text_val = "".join(p["text"] for p in output_parts if "text" in p)
            resp_key = "error" if call_id in error_call_ids else "output"
            media_parts = [
                {
                    "inlineData": {
                        "data": p["inlineData"]["data"],
                        "mimeType": p["inlineData"]["mimeType"],
                    }
                }
                for p in output_parts
                if "inlineData" in p
            ]
            function_response: "typing.Dict[str, object]" = {
                "id": call_id,
                "name": calls[call_id],
            }
            if media_parts:
                function_response["parts"] = media_parts
            function_response["response"] = {resp_key: text_val}
            part_entry = {"functionResponse": function_response}
            # AGY uses outer role "model" for functionResponse and merges consecutive tool results,
            # never merging into the preceding functionCall model turn!
            if prev_was_tool and contents:
                contents[-1]["parts"].append(part_entry)
            else:
                contents.append({"parts": [part_entry], "role": "model"})
            prev_was_tool = True
            continue

        if orig_role == "user":
            user_parts = _content_parts(message.get("content"))
            if user_parts:
                if (
                    prev_was_tool
                    and contents
                    and len(user_parts) == 1
                    and isinstance(user_parts[0].get("text"), str)
                    and user_parts[0]["text"].startswith(SYSTEM_MESSAGE_PREAMBLE)
                ):
                    last_fr = contents[-1]["parts"][-1]["functionResponse"]["response"]
                    target_key = "error" if "error" in last_fr else "output"
                    prev_val = str(last_fr.get(target_key, ""))
                    sys_txt = str(user_parts[0]["text"])
                    last_fr[target_key] = f"{prev_val}\n\n{sys_txt}" if prev_val else sys_txt
                    continue
                contents.append({"parts": user_parts, "role": "user"})
            prev_was_tool = False
            continue

        raise GeminiAPIAdapterError(f"unsupported Gemini message role: {orig_role!r}")

    declarations = []
    for tool in request.get("tools", []) or []:
        if tool.get("type") != "function":
            raise GeminiAPIAdapterError("Gemini backend requires function-style tools")
        function = tool["function"]
        declarations.append(
            {
                "description": function.get("description", ""),
                "name": function["name"],
                "parametersJsonSchema": deepcopy(function.get("parameters", {"type": "object"})),
            }
        )

    payload: "typing.Dict[str, object]" = {
        "contents": contents,
        "generationConfig": (
            model_spec.compaction_generation_config()
            if is_compaction
            else model_spec.generation_config()
        ),
    }
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts, "role": "user"}
    if declarations:
        payload["tools"] = [{"functionDeclarations": declarations}]

    choice = request.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") == "function":
        payload["toolConfig"] = {
            "functionCallingConfig": {
                "mode": "ANY",
                "allowedFunctionNames": [choice["function"]["name"]],
            }
        }
    elif choice in ("required", "none"):
        payload["toolConfig"] = {
            "functionCallingConfig": {
                "mode": "ANY" if choice == "required" else "NONE",
            }
        }

    return model_spec.api_model, payload


class AgyOutcommingTransport(OutcommingTransport):
    """Gemini SSE transport reproducing exact agy 1.5.13 headers, URL and body."""

    def _build_agy_stream_headers(
        self, gzip_request: "bool" = False
    ) -> "typing.Dict[str, str]":
        headers: "typing.Dict[str, str]" = {
            "Content-Type": "application/json",
            "User-agent": AGY_USER_AGENT,
            "x-goog-api-client": AGY_USER_AGENT,
            "Accept-Encoding": "gzip",
        }
        if gzip_request:
            headers["Content-Encoding"] = "gzip"
        for name, value in self.config.outcomming_http_headers.items():
            if name.lower() in ("user-agent", "x-goog-api-client"):
                continue
            headers[name] = value
        api_key = self.config.outcomming_api_key()
        if api_key is not None:
            headers["x-goog-api-key"] = api_key
        return headers

    def _open_outcomming_gemini_stream(self, outcomming_request: "typing.Dict[str, object]"):
        try:
            api_model, payload = build_agy_gemini_request(outcomming_request)
            url = self.config.outcomming_gemini_url(api_model)
        except ValueError as exc:
            raise OutcommingChatError(str(exc)) from exc

        raw_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        use_gzip = should_gzip_gemini_request(self.config.outcomming_base_url)

        def _make_request(compress: "bool") -> "urllib.request.Request":
            return urllib.request.Request(
                url,
                data=encode_gemini_request_body(raw_bytes, compress),
                headers=self._build_agy_stream_headers(gzip_request=compress),
                method="POST",
            )

        request = _make_request(use_gzip)
        try:
            with self._request_diagnostics(request) as progress:
                yield from self._read_agy_gemini_stream(request, progress)
            return
        except OutcommingChatError as exc:
            if not (
                use_gzip
                and progress["chunks"] == 0
                and is_gzip_request_rejected_http_error(exc.http_status, str(exc))
            ):
                raise
            mark_gzip_unsupported(self.config.outcomming_base_url)

        fallback_request = _make_request(False)
        with self._request_diagnostics(fallback_request) as progress:
            yield from self._read_agy_gemini_stream(fallback_request, progress)

    def _read_agy_gemini_stream(self, request: "urllib.request.Request", progress: "typing.Dict[str, object]"):
        try:
            with urllib.request.urlopen(
                request,
                context=ssl.create_default_context(),
                timeout=self.config.timeout_seconds,
            ) as response:
                progress["headers_at"] = time.monotonic()
                progress["status"] = response.status
                progress["phase"] = "waiting_first_chunk"
                encoding = (response.headers.get("Content-Encoding") or "").strip().lower()
                line_source: "typing.Iterable[bytes]" = (
                    gzip.GzipFile(fileobj=response) if encoding == "gzip" else response
                )
                try:
                    stream_state: "typing.Dict[str, object]" = {}
                    for _event_name, data in self._iter_sse_events(line_source):
                        if not data:
                            yield {}
                            continue
                        if data == "[DONE]":
                            break
                        payload = json.loads(data)
                        self._received_chunk(progress)
                        chunks = list(iter_chat_chunks_from_gemini(payload, stream_state))
                        yield from chunks or [{}]
                    if not stream_state.get("finished"):
                        raise OutcommingChatError(
                            "outcomming gemini stream ended before `finishReason`",
                            error_code="502",
                        )
                    yield gemini_continuation_chunk(stream_state)
                except (
                    ConnectionError,
                    EOFError,
                    OSError,
                    http.client.HTTPException,
                    json.JSONDecodeError,
                    GeminiAPIAdapterError,
                ) as exc:
                    raise OutcommingChatError(
                        f"outcomming gemini stream failed while reading response body: {exc}"
                    ) from exc
        except urllib.error.HTTPError as exc:
            progress["headers_at"] = time.monotonic()
            progress["status"] = exc.code
            progress["phase"] = "reading_error_body"
            raw_err = exc.read()
            if (exc.headers.get("Content-Encoding") or "").strip().lower() == "gzip":
                raw_err = gzip.decompress(raw_err)
            body = raw_err.decode("utf-8", errors="replace")
            raise OutcommingChatError(
                f"outcomming gemini request failed with status {exc.code}: {body[:500]}",
                http_status=exc.code,
            ) from exc
        except TimeoutError as exc:
            raise OutcommingChatError(
                f"outcomming gemini request timed out after {self.config.timeout_seconds:g}s"
            ) from exc
        except urllib.error.URLError as exc:
            raise OutcommingChatError(
                f"outcomming gemini request failed: {exc.reason}"
            ) from exc
        except (OSError, EOFError, http.client.HTTPException) as exc:
            raise OutcommingChatError(
                f"outcomming gemini request failed while opening response: {type(exc).__name__}: {exc}"
            ) from exc


class AgyResponseServer(ResponseServer):
    """ResponseServer forwarding error_call_ids and AGY effort metadata."""

    def prepare_response(
        self,
        request_body: "typing.Dict[str, object]",
    ) -> "PreparedResponse":
        prepared = super().prepare_response(request_body)
        error_call_ids = request_body.get("error_call_ids")
        if isinstance(error_call_ids, list):
            prepared.outcomming_request["error_call_ids"] = [
                str(cid) for cid in error_call_ids
            ]
        for key in ("agy_config_effort", "agy_explicit_effort"):
            if key in request_body and request_body[key] is not None:
                prepared.outcomming_request[key] = request_body[key]
        if request_body.get("agy_is_compaction"):
            prepared.outcomming_request["agy_is_compaction"] = True
        return prepared


class AgyManagedResponseServer:
    """In-process Starlette/Uvicorn server translating /v1/responses -> AGY Gemini wire."""

    @staticmethod
    def build_app(config: "CompatServerConfig"):
        stream_router = StreamRouter(AgyOutcommingTransport(config))
        app = ManagedResponseServer.build_app(
            config,
            session_store=SessionStore(),
            stream_router=stream_router,
        )
        # Swap the ResponseServer instance with AgyResponseServer so error_call_ids & effort pass through.
        agy_server = AgyResponseServer(
            config,
            session_store=app.state.response_server.session_store,
            stream_router=stream_router,
        )
        # Re-bind routes onto agy_server cleanly via ManagedResponseServer-compatible routes.
        from starlette.applications import Starlette
        from starlette.exceptions import HTTPException
        from starlette.requests import Request
        from starlette.responses import JSONResponse, StreamingResponse
        from starlette.routing import Route
        from responses_server.app import _http_exception_response, _stream_events
        from responses_server.request_translation import UnsupportedIncommingFeature

        async def health(_request: "Request"):
            return JSONResponse({"ok": True})

        async def list_models(_request: "Request"):
            try:
                return JSONResponse(agy_server.list_models())
            except OutcommingChatError as exc:
                return JSONResponse({"error": {"message": str(exc)}}, status_code=502)

        async def responses(request: "Request"):
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
                str(key).lower(): str(value) for key, value in request.headers.items()
            }
            try:
                prepared = agy_server.prepare_response(request_body)
            except UnsupportedIncommingFeature as exc:
                return JSONResponse({"error": {"message": str(exc)}}, status_code=501)
            return StreamingResponse(
                _stream_events(agy_server, prepared, request_headers),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "close"},
            )

        starlette_app = Starlette(
            routes=[
                Route("/health", health, methods=["GET"]),
                Route("/healthz", health, methods=["GET"]),
                Route("/models", list_models, methods=["GET"]),
                Route("/v1/models", list_models, methods=["GET"]),
                Route("/responses", responses, methods=["POST"]),
                Route("/v1/responses", responses, methods=["POST"]),
            ],
            exception_handlers={HTTPException: _http_exception_response},
        )
        starlette_app.state.response_server = agy_server
        return starlette_app

    def __init__(self, config: "CompatServerConfig") -> "None":
        port = config.port or _reserve_free_port()
        self._config = replace(config, port=port)
        self._app = self.build_app(self._config)
        self._uvicorn_config = uvicorn.Config(
            self._app,
            host=self._config.host,
            port=self._config.port,
            loop="asyncio",
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
    def base_url(self) -> "str":
        return f"http://{self._config.host}:{self._config.port}/v1"

    def start(self, timeout_seconds: "float" = 10.0) -> "None":
        self._thread.start()
        deadline = time.time() + timeout_seconds
        while not self._server.started:
            if time.time() >= deadline:
                raise RuntimeError("timed out waiting for AGY managed responses server to start")
            time.sleep(0.01)

    def stop(self, timeout_seconds: "float" = 5.0) -> "None":
        self._server.should_exit = True
        self._thread.join(timeout=timeout_seconds)
        if self._thread.is_alive():
            raise RuntimeError("timed out waiting for AGY managed responses server to stop")
