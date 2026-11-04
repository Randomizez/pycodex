"""Gemini generateContent request and streaming response adapter."""

import json
import re
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
        config["streamFunctionCallArguments"] = True
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
            if "functionCall" in part:
                chunk = _consume_function_call(part, state)
                if chunk is not None:
                    yield chunk
            elif "text" in part or "thoughtSignature" in part:
                entry = {"provider": "gemini", "part": deepcopy(part)}
                state.setdefault("provider_data", []).append(entry)
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
            if "function_call" in state:
                raise GeminiAPIAdapterError("Gemini stream ended with an incomplete function call")
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


def _consume_function_call(part, state):
    call = part["functionCall"]
    active = state.get("function_call")
    if active is None:
        if not isinstance(call.get("name"), str) or not call["name"] or not isinstance(call.get("args", {}), dict):
            raise GeminiAPIAdapterError("invalid Gemini function call")
        native = {key: deepcopy(value) for key, value in call.items()
                  if key not in {"partialArgs", "willContinue"}}
        entry = {"provider": "gemini", "part": dict(deepcopy(part), functionCall=native),
                 "call_id": call.get("id") or "call_" + uuid.uuid4().hex}
        state.setdefault("provider_data", []).append(entry)
        state["tool_index"] = state.get("tool_index", -1) + 1
        active = {"entry": entry, "pending": {}}
        state["function_call"] = active
        state["last_call_id"] = entry["call_id"]
    else:
        native = active["entry"]["part"]["functionCall"]
        if ("name" in call and call["name"] != native["name"]
                or "id" in call and call["id"] != native.get("id")
                or "args" in call):
            raise GeminiAPIAdapterError("conflicting Gemini function call fragment")
        if "thoughtSignature" in part:
            state["provider_data"].append({
                "provider": "gemini", "call_id": active["entry"]["call_id"],
                "part": {"text": "", "thoughtSignature": part["thoughtSignature"]},
            })
    for partial in call.get("partialArgs", []):
        _consume_partial_argument(partial, native.setdefault("args", {}), active["pending"])
    if call.get("willContinue", False):
        return None
    if active["pending"]:
        raise GeminiAPIAdapterError("Gemini function call ended with incomplete arguments")
    del state["function_call"]
    return _delta({"tool_calls": [{
        "index": state["tool_index"], "id": active["entry"]["call_id"], "type": "function",
        "function": {"name": native["name"], "arguments": json.dumps(
            native.get("args", {}), ensure_ascii=False, separators=(",", ":"),
        )},
    }]})


def _consume_partial_argument(partial, arguments, pending):
    raw_path = partial.get("jsonPath")
    container = raw_path[-2:] if isinstance(raw_path, str) else None
    is_container = container in ("{}", "[]")
    path = _argument_path(raw_path[:-2] if is_container else raw_path)
    fields = [key for key in ("stringValue", "numberValue", "boolValue", "nullValue")
              if key in partial]
    if len(fields) != (0 if is_container else 1):
        raise GeminiAPIAdapterError("Gemini partial argument must contain one value")
    field = fields[0] if fields else None
    value = partial[field] if field is not None else ([] if container == "[]" else {})
    if field == "stringValue":
        if not isinstance(value, str):
            raise GeminiAPIAdapterError("invalid Gemini string argument")
        chunks = pending.setdefault(path, [])
        chunks.append(value)
        if partial.get("willContinue", False):
            return
        value = "".join(pending.pop(path))
    else:
        if path in pending or partial.get("willContinue", False):
            raise GeminiAPIAdapterError("invalid Gemini scalar argument continuation")
        if field == "numberValue" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise GeminiAPIAdapterError("invalid Gemini number argument")
        if field == "boolValue" and not isinstance(value, bool):
            raise GeminiAPIAdapterError("invalid Gemini boolean argument")
        if field == "nullValue":
            if isinstance(value, bool) or value not in (None, "NULL_VALUE", 0):
                raise GeminiAPIAdapterError("invalid Gemini null argument")
            value = None
    node = arguments
    for index, key in enumerate(path):
        last = index == len(path) - 1
        child = value if last else ([] if isinstance(path[index + 1], int) else {})
        if isinstance(node, dict) and isinstance(key, str):
            if key not in node:
                node[key] = child
            elif last:
                raise GeminiAPIAdapterError("duplicate Gemini argument path")
        elif isinstance(node, list) and isinstance(key, int) and key <= len(node):
            if key == len(node):
                node.append(child)
            elif last:
                raise GeminiAPIAdapterError("duplicate Gemini argument path")
        else:
            raise GeminiAPIAdapterError("invalid Gemini argument path structure")
        node = node[key]


def _argument_path(path):
    # Function arguments use concrete member/index paths, never JSONPath queries.
    pattern = r'''\.([^\W\d]\w*)|\[(0|[1-9][0-9]*|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')\]'''
    if not isinstance(path, str) or not path.startswith("$"):
        raise GeminiAPIAdapterError("invalid Gemini argument path")
    keys = []
    offset = 1
    while offset < len(path):
        match = re.match(pattern, path[offset:])
        if match is None:
            raise GeminiAPIAdapterError("invalid Gemini argument path")
        member, bracket = match.groups()
        if member is not None:
            key = member
        elif bracket[0] in ("'", '"'):
            if bracket[0] == "'":
                bracket = '"' + re.sub(
                    r'''\\(.)|"''',
                    lambda item: "'" if item.group(1) == "'" else
                    '\\"' if item.group(0) == '"' else item.group(0),
                    bracket[1:-1],
                ) + '"'
            try:
                key = json.loads(bracket)
            except ValueError as exc:
                raise GeminiAPIAdapterError("invalid Gemini argument path") from exc
        else:
            key = int(bracket)
        keys.append(key)
        offset += match.end()
    if not keys or not isinstance(keys[0], str):
        raise GeminiAPIAdapterError("Gemini function arguments must be objects")
    return tuple(keys)


def _delta(delta):
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


def continuation_chunk(state):
    """Use the existing opaque reasoning-item contract for durable native state."""
    return {"reasoning_item": {
        "type": "reasoning", "summary": [],
        "provider_data": deepcopy(state.get("provider_data", [])),
    }}
