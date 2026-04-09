import json
import typing

DEFAULT_MESSAGES_MAX_TOKENS = 32000


class MessagesAPIAdapterError(ValueError):
    pass


def build_messages_request(
    outcomming_request: 'typing.Dict[str, object]',
) -> 'typing.Dict[str, object]':
    model = str(outcomming_request.get("model", "")).strip()
    if not model:
        raise MessagesAPIAdapterError("outcomming request is missing `model`")

    raw_messages = outcomming_request.get("messages") or []
    if not isinstance(raw_messages, list):
        raise MessagesAPIAdapterError("outcomming request `messages` must be a list")

    system_blocks: 'typing.List[typing.Dict[str, object]]' = []
    messages: 'typing.List[typing.Dict[str, object]]' = []
    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            raise MessagesAPIAdapterError(
                "outcomming request messages must be objects"
            )
        role = str(raw_message.get("role", "")).strip()
        if role in {"developer", "system"}:
            text = str(raw_message.get("content", "") or "")
            if text:
                system_blocks.append({"type": "text", "text": text})
            continue
        if role == "user":
            messages.append(
                {
                    "role": "user",
                    "content": _build_text_blocks(raw_message.get("content")),
                }
            )
            continue
        if role == "assistant":
            messages.append(
                {
                    "role": "assistant",
                    "content": _build_assistant_blocks(raw_message),
                }
            )
            continue
        if role == "tool":
            messages.append(
                {
                    "role": "user",
                    "content": [_build_tool_result_block(raw_message)],
                }
            )
            continue
        raise MessagesAPIAdapterError(
            f"unsupported outcomming message role for messages API: {role!r}"
        )

    payload: 'typing.Dict[str, object]' = {
        "model": model,
        "messages": messages,
        "max_tokens": _resolve_max_tokens(outcomming_request),
        "stream": bool(outcomming_request.get("stream", True)),
    }
    if system_blocks:
        payload["system"] = system_blocks

    tools = _translate_tools(outcomming_request.get("tools"))
    if tools:
        payload["tools"] = tools
        tool_choice = _translate_tool_choice(
            outcomming_request.get("tool_choice"),
            outcomming_request.get("parallel_tool_calls"),
        )
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
    return payload


def iter_chat_chunks(
    event_name: 'typing.Union[str, None]',
    payload: 'typing.Dict[str, object]',
    state: 'typing.Dict[str, object]',
) -> 'typing.List[typing.Dict[str, object]]':
    event_type = str(payload.get("type") or event_name or "").strip()
    chunks: 'typing.List[typing.Dict[str, object]]' = []

    if event_type == "message_start":
        message = payload.get("message") or {}
        if isinstance(message, dict):
            usage_chunk = _usage_chunk(message.get("usage"))
            if usage_chunk is not None:
                chunks.append(usage_chunk)
        return chunks

    if event_type == "content_block_start":
        block_index = _normalize_index(payload.get("index"))
        content_block = payload.get("content_block") or {}
        if not isinstance(content_block, dict):
            return chunks
        content_blocks = state.setdefault("content_blocks", {})
        if not isinstance(content_blocks, dict):
            raise MessagesAPIAdapterError("messages stream state is corrupted")
        content_blocks[block_index] = str(content_block.get("type", "")).strip()

        block_type = str(content_block.get("type", "")).strip()
        if block_type == "text":
            text = str(content_block.get("text", "") or "")
            if text:
                chunks.append(_chat_text_chunk(text))
            return chunks
        if block_type == "thinking":
            thinking = str(content_block.get("thinking", "") or "")
            if thinking:
                chunks.append(_chat_reasoning_chunk(thinking))
            return chunks
        if block_type == "tool_use":
            arguments = _dump_json(content_block.get("input") or {})
            chunks.append(
                _chat_tool_chunk(
                    block_index,
                    call_id=str(content_block.get("id", "")).strip(),
                    name=str(content_block.get("name", "")).strip(),
                    arguments=arguments if arguments != "{}" else "",
                )
            )
            return chunks
        return chunks

    if event_type == "content_block_delta":
        block_index = _normalize_index(payload.get("index"))
        delta = payload.get("delta") or {}
        if not isinstance(delta, dict):
            return chunks
        delta_type = str(delta.get("type", "")).strip()
        if delta_type == "text_delta":
            text = str(delta.get("text", "") or "")
            if text:
                chunks.append(_chat_text_chunk(text))
            return chunks
        if delta_type == "thinking_delta":
            thinking = str(delta.get("thinking", "") or "")
            if thinking:
                chunks.append(_chat_reasoning_chunk(thinking))
            return chunks
        if delta_type == "input_json_delta":
            partial_json = str(delta.get("partial_json", "") or "")
            chunks.append(_chat_tool_chunk(block_index, arguments=partial_json))
            return chunks
        return chunks

    if event_type == "message_delta":
        usage_chunk = _usage_chunk(payload.get("usage"))
        if usage_chunk is not None:
            chunks.append(usage_chunk)
        delta = payload.get("delta") or {}
        if not isinstance(delta, dict):
            return chunks
        finish_reason = _translate_stop_reason(delta.get("stop_reason"))
        if finish_reason and not bool(state.get("finish_emitted")):
            state["finish_reason"] = finish_reason
            state["finish_emitted"] = True
            chunks.append(_chat_finish_chunk(finish_reason))
        return chunks

    if event_type == "message_stop":
        if not bool(state.get("finish_emitted")):
            finish_reason = str(state.get("finish_reason") or "stop")
            state["finish_emitted"] = True
            chunks.append(_chat_finish_chunk(finish_reason))
        state["saw_message_stop"] = True
        return chunks

    if event_type == "error":
        error = payload.get("error")
        if isinstance(error, dict):
            message = str(error.get("message", "") or "").strip()
            if message:
                raise MessagesAPIAdapterError(message)
        raise MessagesAPIAdapterError(_dump_json(payload))

    return chunks


def saw_message_stop(state: 'typing.Dict[str, object]') -> 'bool':
    return bool(state.get("saw_message_stop"))


def _build_text_blocks(raw_content: 'object') -> 'typing.List[typing.Dict[str, object]]':
    text = str(raw_content or "")
    if not text:
        return []
    return [{"type": "text", "text": text}]


def _build_assistant_blocks(
    raw_message: 'typing.Dict[str, object]',
) -> 'typing.List[typing.Dict[str, object]]':
    blocks: 'typing.List[typing.Dict[str, object]]' = []
    reasoning = str(raw_message.get("reasoning", "") or "")
    if reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning})

    text = str(raw_message.get("content", "") or "")
    if text:
        blocks.append({"type": "text", "text": text})

    raw_tool_calls = raw_message.get("tool_calls") or []
    if raw_tool_calls:
        if not isinstance(raw_tool_calls, list):
            raise MessagesAPIAdapterError("assistant `tool_calls` must be a list")
        for raw_tool_call in raw_tool_calls:
            if not isinstance(raw_tool_call, dict):
                raise MessagesAPIAdapterError("assistant tool calls must be objects")
            function = raw_tool_call.get("function") or {}
            if not isinstance(function, dict):
                raise MessagesAPIAdapterError(
                    "assistant tool call is missing function payload"
                )
            blocks.append(
                {
                    "type": "tool_use",
                    "id": str(raw_tool_call.get("id", "")).strip(),
                    "name": str(function.get("name", "")).strip(),
                    "input": _parse_json_object(function.get("arguments")),
                }
            )
    return blocks


def _build_tool_result_block(
    raw_message: 'typing.Dict[str, object]',
) -> 'typing.Dict[str, object]':
    return {
        "type": "tool_result",
        "tool_use_id": str(raw_message.get("tool_call_id", "")).strip(),
        "content": str(raw_message.get("content", "") or ""),
    }


def _translate_tools(
    raw_tools: 'object',
) -> 'typing.List[typing.Dict[str, object]]':
    translated: 'typing.List[typing.Dict[str, object]]' = []
    if not isinstance(raw_tools, list):
        return translated
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict) or raw_tool.get("type") != "function":
            raise MessagesAPIAdapterError(
                "messages API backend only supports function-style tools"
            )
        function = raw_tool.get("function") or {}
        if not isinstance(function, dict):
            raise MessagesAPIAdapterError("tool definition is missing function payload")
        name = str(function.get("name", raw_tool.get("name", ""))).strip()
        if not name:
            raise MessagesAPIAdapterError("tool definition is missing `name`")
        translated.append(
            {
                "name": name,
                "description": str(function.get("description", "") or ""),
                "input_schema": function.get("parameters") or {"type": "object"},
            }
        )
    return translated


def _translate_tool_choice(
    raw_tool_choice: 'object',
    parallel_tool_calls: 'object',
) -> 'typing.Union[typing.Dict[str, object], None]':
    if raw_tool_choice is None:
        if parallel_tool_calls is False:
            return {
                "type": "auto",
                "disable_parallel_tool_use": True,
            }
        return None

    translated: 'typing.Dict[str, object]'
    if isinstance(raw_tool_choice, str):
        choice = raw_tool_choice.strip()
        if choice == "auto":
            translated = {"type": "auto"}
        elif choice == "required":
            translated = {"type": "any"}
        elif choice == "none":
            return None
        else:
            raise MessagesAPIAdapterError(
                f"unsupported tool_choice for messages API: {raw_tool_choice!r}"
            )
    elif isinstance(raw_tool_choice, dict):
        choice_type = str(raw_tool_choice.get("type", "")).strip()
        if choice_type == "function":
            function = raw_tool_choice.get("function") or {}
            name = ""
            if isinstance(function, dict):
                name = str(function.get("name", "")).strip()
            if not name:
                name = str(raw_tool_choice.get("name", "")).strip()
            if not name:
                raise MessagesAPIAdapterError(
                    "function tool_choice is missing `name`"
                )
            translated = {
                "type": "tool",
                "name": name,
            }
        else:
            raise MessagesAPIAdapterError(
                f"unsupported tool_choice for messages API: {raw_tool_choice!r}"
            )
    else:
        raise MessagesAPIAdapterError(
            f"unsupported tool_choice for messages API: {raw_tool_choice!r}"
        )

    if parallel_tool_calls is False:
        translated["disable_parallel_tool_use"] = True
    return translated


def _parse_json_object(raw_value: 'object') -> 'typing.Dict[str, object]':
    if isinstance(raw_value, dict):
        return dict(raw_value)
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MessagesAPIAdapterError(
                f"tool arguments must be valid JSON objects for messages API: {exc}"
            ) from exc
        if isinstance(parsed, dict):
            return dict(parsed)
        raise MessagesAPIAdapterError(
            "tool arguments must decode to JSON objects for messages API"
        )
    raise MessagesAPIAdapterError(
        "tool arguments must be strings or objects for messages API"
    )


def _resolve_max_tokens(outcomming_request: 'typing.Dict[str, object]') -> 'int':
    raw_value = outcomming_request.get("max_tokens")
    if isinstance(raw_value, bool):
        return DEFAULT_MESSAGES_MAX_TOKENS
    if isinstance(raw_value, int) and raw_value > 0:
        return raw_value
    return DEFAULT_MESSAGES_MAX_TOKENS


def _usage_chunk(raw_usage: 'object') -> 'typing.Union[typing.Dict[str, object], None]':
    usage = _translate_usage(raw_usage)
    if not usage:
        return None
    return {
        "choices": [],
        "usage": usage,
    }


def _translate_usage(raw_usage: 'object') -> 'typing.Dict[str, object]':
    if not isinstance(raw_usage, dict):
        return {}
    usage: 'typing.Dict[str, object]' = {}
    input_tokens = raw_usage.get("input_tokens")
    output_tokens = raw_usage.get("output_tokens")
    if isinstance(input_tokens, int):
        usage["input_tokens"] = input_tokens
    if isinstance(output_tokens, int):
        usage["output_tokens"] = output_tokens
    total_tokens = raw_usage.get("total_tokens")
    if isinstance(total_tokens, int):
        usage["total_tokens"] = total_tokens
    elif isinstance(input_tokens, int) and isinstance(output_tokens, int):
        usage["total_tokens"] = input_tokens + output_tokens

    input_details: 'typing.Dict[str, int]' = {}
    cache_creation = raw_usage.get("cache_creation_input_tokens")
    if isinstance(cache_creation, int):
        input_details["cache_creation_input_tokens"] = cache_creation
    cache_read = raw_usage.get("cache_read_input_tokens")
    if isinstance(cache_read, int):
        input_details["cache_read_input_tokens"] = cache_read
    if input_details:
        usage["input_tokens_details"] = input_details
    return usage


def _normalize_index(raw_index: 'object') -> 'int':
    if isinstance(raw_index, int):
        return raw_index
    try:
        return int(raw_index)
    except (TypeError, ValueError):
        return 0


def _translate_stop_reason(raw_stop_reason: 'object') -> 'typing.Union[str, None]':
    if not isinstance(raw_stop_reason, str):
        return None
    stop_reason = raw_stop_reason.strip()
    if not stop_reason:
        return None
    if stop_reason == "tool_use":
        return "tool_calls"
    if stop_reason == "max_tokens":
        return "length"
    if stop_reason in {"end_turn", "stop_sequence"}:
        return "stop"
    return stop_reason


def _chat_text_chunk(text: 'str') -> 'typing.Dict[str, object]':
    return _chat_delta_chunk({"content": text})


def _chat_reasoning_chunk(reasoning: 'str') -> 'typing.Dict[str, object]':
    return _chat_delta_chunk({"reasoning_content": reasoning})


def _chat_tool_chunk(
    index: 'int',
    call_id: 'str' = "",
    name: 'str' = "",
    arguments: 'str' = "",
) -> 'typing.Dict[str, object]':
    tool_call: 'typing.Dict[str, object]' = {
        "index": index,
        "function": {},
    }
    if call_id:
        tool_call["id"] = call_id
    if name:
        tool_call["type"] = "function"
        tool_call["function"] = {"name": name}
    function = tool_call.get("function")
    if not isinstance(function, dict):
        function = {}
        tool_call["function"] = function
    if arguments:
        function["arguments"] = arguments
    return _chat_delta_chunk({"tool_calls": [tool_call]})


def _chat_delta_chunk(delta: 'typing.Dict[str, object]') -> 'typing.Dict[str, object]':
    return {
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": None,
            }
        ]
    }


def _chat_finish_chunk(finish_reason: 'str') -> 'typing.Dict[str, object]':
    return {
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }
        ]
    }


def _dump_json(raw_value: 'object') -> 'str':
    return json.dumps(raw_value, ensure_ascii=False, separators=(",", ":"))
