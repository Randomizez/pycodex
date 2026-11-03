"""Gemini generateContent request and streaming response adapter."""

import json
import uuid
from copy import deepcopy


class GeminiAPIAdapterError(ValueError):
    pass


def build_gemini_request(request):
    contents = []
    system_parts = []
    calls = {}
    for message in request.get("messages", []):
        role = message["role"]
        if role in {"system", "developer"}:
            system_parts.extend(_content_parts(message.get("content")))
            continue
        if role == "assistant":
            active_call_ids = set()
            for call in message.get("tool_calls", []):
                calls[call["id"]] = call["function"]["name"]
                active_call_ids.add(call["id"])
            native = message.get("provider_data", [])
            native = [entry for entry in native if entry["provider"] == "gemini"]
            if native:
                # Compaction may prune a call/result pair while retaining the
                # opaque reasoning item. Never replay a removed function call.
                parts = [deepcopy(entry["part"]) for entry in native
                         if "call_id" not in entry or entry["call_id"] in active_call_ids]
            else:
                parts = _content_parts(message.get("content"))
                for call in message.get("tool_calls", []):
                    function = call["function"]
                    arguments = json.loads(function["arguments"])
                    if not isinstance(arguments, dict):
                        raise GeminiAPIAdapterError("Gemini function arguments must be objects")
                    parts.append({"functionCall": {
                        "name": function["name"], "args": arguments, "id": call["id"],
                    }})
            role = "model"
        elif role == "tool":
            call_id = message["tool_call_id"]
            if call_id not in calls:
                raise GeminiAPIAdapterError("Gemini tool result has no matching function call")
            parts = [{"functionResponse": {
                "name": calls[call_id], "id": call_id,
                "response": {"result": message.get("content", "")},
            }}]
            role = "user"
        elif role == "user":
            parts = _content_parts(message.get("content"))
        else:
            raise GeminiAPIAdapterError("unsupported Gemini message role: {!r}".format(role))
        if parts:
            # Batch parallel function responses, keeping subsequent image input
            # in its own user content as provided by the canonical translator.
            if message["role"] == "tool" and contents and all(
                "functionResponse" in part for part in contents[-1]["parts"]
            ):
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})

    payload = {"contents": contents}
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}
    generation = {"thinkingConfig": {"includeThoughts": True}}
    if "max_tokens" in request:
        generation["maxOutputTokens"] = request["max_tokens"]
    effort = request.get("chat_template_kwargs", {}).get("reasoning_effort")
    if effort is not None:
        levels = {
            "none": "minimal", "minimal": "minimal", "low": "low",
            "medium": "medium", "high": "high", "xhigh": "high", "max": "high",
        }
        if effort not in levels:
            raise GeminiAPIAdapterError("unsupported Gemini reasoning effort: {!r}".format(effort))
        generation["thinkingConfig"]["thinkingLevel"] = levels[effort]
    payload["generationConfig"] = generation

    declarations = []
    for tool in request.get("tools", []):
        if tool["type"] != "function":
            raise GeminiAPIAdapterError("Gemini backend requires function-style tools")
        function = tool["function"]
        declarations.append({
            "name": function["name"], "description": function.get("description", ""),
            "parametersJsonSchema": deepcopy(function.get("parameters", {"type": "object"})),
        })
    if declarations:
        payload["tools"] = [{"functionDeclarations": declarations}]
        choice = request.get("tool_choice", "auto")
        if isinstance(choice, dict) and choice.get("type") == "function":
            config = {"mode": "ANY", "allowedFunctionNames": [choice["function"]["name"]]}
        elif choice in ("auto", "required", "none"):
            config = {"mode": {"auto": "AUTO", "required": "ANY", "none": "NONE"}[choice]}
        else:
            raise GeminiAPIAdapterError("unsupported Gemini tool_choice")
        payload["toolConfig"] = {"functionCallingConfig": config}
    return payload


def _content_parts(content):
    if not content:
        return []
    if isinstance(content, str):
        return [{"text": content}]
    parts = []
    for part in content:
        if part["type"] == "text":
            parts.append({"text": part["text"]})
        elif part["type"] == "image_url":
            url = part["image_url"]["url"]
            if url.startswith("data:"):
                header, separator, data = url.partition(",")
                mime_type = header[5:].split(";")[0]
                if not separator or not header.endswith(";base64") or not mime_type or not data:
                    raise GeminiAPIAdapterError("Gemini images require a base64 data URL with a media type")
                parts.append({"inlineData": {"mimeType": mime_type, "data": data}})
            else:
                raise GeminiAPIAdapterError("Gemini images require a base64 data URL with a media type")
        else:
            raise GeminiAPIAdapterError("unsupported Gemini content part: {!r}".format(part["type"]))
    return parts


def iter_chat_chunks(payload, state):
    if not isinstance(payload, dict):
        raise GeminiAPIAdapterError("Gemini stream payload must be an object")
    if "error" in payload:
        error = payload["error"]
        message = error.get("message", "Gemini API error") if isinstance(error, dict) else str(error)
        raise GeminiAPIAdapterError(message)
    blocked = payload.get("promptFeedback", {}).get("blockReason")
    if blocked:
        raise GeminiAPIAdapterError("Gemini prompt blocked: {}".format(blocked))
    candidates = payload.get("candidates", [])
    if len(candidates) > 1:
        raise GeminiAPIAdapterError("Gemini adapter requires a single candidate")
    for candidate in candidates:
        if candidate.get("index", 0) != 0:
            raise GeminiAPIAdapterError("unexpected Gemini candidate index")
        finish = candidate.get("finishReason")
        if finish and finish != "STOP":
            raise GeminiAPIAdapterError("Gemini generation ended with {}".format(finish))
        for part in candidate.get("content", {}).get("parts", []):
            entry = {"provider": "gemini", "part": deepcopy(part)}
            state.setdefault("provider_data", []).append(entry)
            if "functionCall" in part:
                call = part["functionCall"]
                if not call.get("name") or not isinstance(call.get("args", {}), dict):
                    raise GeminiAPIAdapterError("invalid Gemini function call")
                tool_index = state.get("tool_index", -1) + 1
                state["tool_index"] = tool_index
                call_id = call.get("id") or "call_" + uuid.uuid4().hex
                entry["call_id"] = call_id
                state["last_call_id"] = call_id
                yield _delta({"tool_calls": [{
                    "index": tool_index, "id": call_id, "type": "function",
                    "function": {"name": call["name"], "arguments": json.dumps(
                        call.get("args", {}), ensure_ascii=False, separators=(",", ":"),
                    )},
                }]})
            elif "text" in part or "thoughtSignature" in part:
                text = part.get("text", "")
                if not text and state.get("last_call_id"):
                    entry["call_id"] = state["last_call_id"]
                if text:
                    state.pop("last_call_id", None)
                    kind = "reasoning_content" if part.get("thought") else "content"
                    yield _delta({kind: text})
            else:
                raise GeminiAPIAdapterError("unsupported Gemini output part: {}".format(
                    ", ".join(sorted(part)),
                ))
        if finish:
            state["finished"] = True
            yield {"choices": [{"index": 0, "delta": {}, "finish_reason": (
                "tool_calls" if "tool_index" in state else "stop"
            )}]}
    usage = payload.get("usageMetadata", {})
    token_counts = state.setdefault("token_counts", {})
    for key in ("candidatesTokenCount", "thoughtsTokenCount"):
        if key in usage:
            token_counts[key] = usage[key]
    translated = {}
    for source, target in (("promptTokenCount", "prompt_tokens"), ("totalTokenCount", "total_tokens")):
        if source in usage:
            translated[target] = usage[source]
    if "candidatesTokenCount" in usage or "thoughtsTokenCount" in usage:
        translated["completion_tokens"] = (
            token_counts.get("candidatesTokenCount", 0)
            + token_counts.get("thoughtsTokenCount", 0)
        )
    if "thoughtsTokenCount" in usage:
        translated["completion_tokens_details"] = {"reasoning_tokens": usage["thoughtsTokenCount"]}
    if "cachedContentTokenCount" in usage:
        translated["prompt_tokens_details"] = {"cached_tokens": usage["cachedContentTokenCount"]}
    if translated:
        yield {"choices": [], "usage": translated}


def _delta(delta):
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


def continuation_chunk(state):
    """Use the existing opaque reasoning-item contract for durable native state."""
    return {"reasoning_item": {
        "type": "reasoning", "summary": [],
        "provider_data": deepcopy(state.get("provider_data", [])),
    }}
