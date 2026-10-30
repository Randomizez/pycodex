"""Pure Responses-to-chat request translation and request-local tool names."""

import json
import typing

from .tools import WebSearchTool
from .tools.custom_adapter import CustomToolAdapterError
from .tools.custom_adapter import build_tool_call as build_custom_tool_call
from .tools.custom_adapter import build_tool_definition as build_custom_tool_definition
from .tools.web_search import build_tool_definition


class UnsupportedIncommingFeature(ValueError):
    pass


class ToolNameMap:
    """Request-local identities for flattened namespace tools and their calls."""

    def __init__(self) -> "None":
        self._identities: "typing.Dict[str, typing.Tuple[typing.Optional[str], str]]" = {}
        self.custom_tool_names: "typing.Set[str]" = set()

    def qualify(self, name: "object", namespace: "object" = None) -> "str":
        if not isinstance(name, str) or not name.strip():
            raise UnsupportedIncommingFeature("tool is missing a non-empty `name`")
        name = name.strip()
        if namespace is not None:
            if not isinstance(namespace, str) or not namespace.strip():
                raise UnsupportedIncommingFeature(
                    "tool namespace must be a non-empty string"
                )
            namespace = namespace.strip()
        qualified = f"{namespace}__{name}" if namespace is not None else name
        identity = (namespace, name)
        if qualified in self._identities and self._identities[qualified] != identity:
            raise UnsupportedIncommingFeature(
                f"flattened tool name collision: {qualified!r}"
            )
        self._identities[qualified] = identity
        return qualified

    def restore_output_item(
        self, item: "typing.Dict[str, object]"
    ) -> "typing.Dict[str, object]":
        if item.get("type") not in {"function_call", "custom_tool_call"}:
            return item
        name = item["name"]
        namespace, original_name = self._identities.get(name, (None, name))
        if namespace is None:
            return item
        return dict(item, namespace=namespace, name=original_name)


def build_outcomming_request(
    incomming_request: "typing.Dict[str, object]",
    tool_names: "typing.Union[ToolNameMap, None]" = None,
) -> "typing.Dict[str, object]":
    model = str(incomming_request.get("model", "")).strip()
    if not model:
        raise UnsupportedIncommingFeature("incomming request is missing `model`")

    stream = incomming_request.get("stream", True)
    if stream is not True:
        raise UnsupportedIncommingFeature(
            "only streaming incomming `/responses` requests are supported"
        )

    instructions = str(incomming_request.get("instructions", "") or "")
    input_items = incomming_request.get("input") or []
    if not isinstance(input_items, list):
        raise UnsupportedIncommingFeature("incomming `input` must be a list")

    if tool_names is None:
        tool_names = ToolNameMap()
    tools = incomming_request.get("tools") or []
    if not isinstance(tools, list):
        raise UnsupportedIncommingFeature("incomming `tools` must be a list")
    translated_tools = _translate_tools(tools, tool_names)
    payload: "typing.Dict[str, object]" = {
        "model": model,
        "messages": _responses_input_to_chat_messages(
            instructions,
            input_items,
            tool_names,
        ),
        "stream": True,
    }
    reasoning = incomming_request.get("reasoning")
    if isinstance(reasoning, dict):
        reasoning_effort = reasoning.get("effort")
        if isinstance(reasoning_effort, str) and reasoning_effort:
            # vLLM's Responses renderer consumes effort as a template
            # kwarg; keeping it out of the chat schema also preserves
            # provider-specific values such as `max`.
            payload["chat_template_kwargs"] = {
                "reasoning_effort": reasoning_effort,
            }

    max_tokens = _coerce_positive_int(
        incomming_request.get("max_output_tokens")
    )
    if max_tokens is None:
        max_tokens = _coerce_positive_int(incomming_request.get("max_tokens"))
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    payload["stream_options"] = {"include_usage": True}

    if translated_tools:
        payload["tools"] = translated_tools

    tool_choice = incomming_request.get("tool_choice")
    if tool_choice is not None:
        payload["tool_choice"] = _translate_tool_choice(
            tool_choice, tool_names,
        )

    parallel_tool_calls = incomming_request.get("parallel_tool_calls")
    if isinstance(parallel_tool_calls, bool):
        payload["parallel_tool_calls"] = parallel_tool_calls

    return payload


def _responses_input_to_chat_messages(
    instructions: "str",
    input_items: "typing.List[object]",
    tool_names: "ToolNameMap",
) -> "typing.List[typing.Dict[str, object]]":
    messages: "typing.List[typing.Dict[str, object]]" = []
    if instructions:
        messages.append({"role": "developer", "content": instructions})

    pending_assistant: "typing.Union[typing.Dict[str, object], None]" = None
    pending_tool_images: "typing.List[typing.Dict[str, object]]" = []

    def flush_pending_assistant() -> "None":
        nonlocal pending_assistant
        if pending_assistant is None:
            return
        if (
            "content" not in pending_assistant
            and "reasoning" not in pending_assistant
            and "tool_calls" not in pending_assistant
        ):
            pending_assistant = None
            return
        messages.append(pending_assistant)
        pending_assistant = None

    def flush_pending_tool_images() -> "None":
        if not pending_tool_images:
            return
        messages.append({"role": "user", "content": list(pending_tool_images)})
        del pending_tool_images[:]

    for raw_item in input_items:
        if not isinstance(raw_item, dict):
            raise UnsupportedIncommingFeature(
                "all incomming `input` items must be objects"
            )
        item_type = raw_item.get("type")
        if item_type not in {"function_call_output", "custom_tool_call_output"}:
            flush_pending_tool_images()

        if item_type == "agent_message":
            if not all(
                isinstance(raw_item.get(key), str) for key in ("author", "recipient")
            ):
                raise UnsupportedIncommingFeature(
                    "agent_message requires string `author` and `recipient`"
                )
            text, image_parts = _split_content_parts(raw_item.get("content"))
            if image_parts:
                raise UnsupportedIncommingFeature(
                    "agent_message content must contain only input_text parts"
                )
            flush_pending_assistant()
            messages.append({
                "role": "user",
                "content": (
                    f"Agent message from {raw_item['author']} to "
                    f"{raw_item['recipient']}:\n{text}"
                ),
            })
            continue

        if item_type == "message":
            role = str(raw_item.get("role", "")).strip()
            if role not in {"developer", "user", "assistant", "system"}:
                raise UnsupportedIncommingFeature(
                    f"unsupported incomming message role: {role or '<empty>'}"
                )
            text, image_parts = _split_content_parts(raw_item.get("content"))
            if role == "assistant":
                if image_parts:
                    raise UnsupportedIncommingFeature(
                        "assistant messages cannot carry `input_image` content parts"
                    )
                if pending_assistant is None:
                    pending_assistant = {"role": "assistant"}
                if text:
                    pending_assistant["content"] = (
                        str(pending_assistant.get("content", "")) + text
                    )
                continue
            flush_pending_assistant()
            if image_parts:
                content: "object" = (
                    [{"type": "text", "text": text}] if text else []
                ) + image_parts
            else:
                content = text
            messages.append({"role": role, "content": content})
            continue

        if item_type == "reasoning":
            if pending_assistant is None:
                pending_assistant = {"role": "assistant"}
            reasoning_text = coalesce_reasoning_text(raw_item)
            if reasoning_text:
                pending_assistant["reasoning"] = (
                    str(pending_assistant.get("reasoning", "")) + reasoning_text
                )
            continue

        if item_type == "function_call":
            if pending_assistant is None:
                pending_assistant = {"role": "assistant"}
            tool_calls = pending_assistant.setdefault("tool_calls", [])
            if not isinstance(tool_calls, list):
                raise UnsupportedIncommingFeature(
                    "assistant tool calls must be a list"
                )
            tool_calls.append(
                {
                    "id": str(raw_item.get("call_id", "")).strip(),
                    "type": "function",
                    "function": {
                        "name": tool_names.qualify(
                            raw_item.get("name"), raw_item.get("namespace"),
                        ),
                        "arguments": str(raw_item.get("arguments", "") or "{}"),
                    },
                }
            )
            continue

        if item_type in {"function_call_output", "custom_tool_call_output"}:
            flush_pending_assistant()
            text, image_parts = _split_tool_output_parts(
                raw_item.get("output")
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(raw_item.get("call_id", "")).strip(),
                    "content": text,
                }
            )
            pending_tool_images.extend(image_parts)
            continue

        if item_type == "custom_tool_call":
            if pending_assistant is None:
                pending_assistant = {"role": "assistant"}
            tool_calls = pending_assistant.setdefault("tool_calls", [])
            if not isinstance(tool_calls, list):
                raise UnsupportedIncommingFeature(
                    "assistant tool calls must be a list"
                )
            try:
                tool_call = build_custom_tool_call(raw_item)
            except CustomToolAdapterError as exc:
                raise UnsupportedIncommingFeature(str(exc)) from exc
            tool_call["function"]["name"] = tool_names.qualify(
                raw_item.get("name"), raw_item.get("namespace"),
            )
            tool_calls.append(tool_call)
            continue

        raise UnsupportedIncommingFeature(
            f"unsupported incomming input item type: {item_type!r}"
        )

    flush_pending_assistant()
    flush_pending_tool_images()
    return messages


def _coerce_positive_int(raw_value: "object") -> "typing.Union[int, None]":
    if isinstance(raw_value, bool):
        return None
    if isinstance(raw_value, int) and raw_value > 0:
        return raw_value
    return None


def _split_content_parts(
    raw_content: "object",
) -> "typing.Tuple[str, typing.List[typing.Dict[str, object]]]":
    if raw_content is None:
        return "", []
    if isinstance(raw_content, str):
        return raw_content, []
    if not isinstance(raw_content, list):
        raise UnsupportedIncommingFeature(
            "message `content` must be a list or string"
        )

    text_parts: "typing.List[str]" = []
    image_parts: "typing.List[typing.Dict[str, object]]" = []
    for part in raw_content:
        if not isinstance(part, dict):
            raise UnsupportedIncommingFeature(
                "message content parts must be objects"
            )
        part_type = part.get("type")
        if part_type in {"input_text", "output_text"}:
            text_parts.append(str(part.get("text", "")))
            continue
        if part_type == "input_image":
            image_parts.append(_build_chat_image_part(part))
            continue
        raise UnsupportedIncommingFeature(
            f"content part type `{part_type}` is not yet supported by the chat backend"
        )
    return "".join(text_parts), image_parts


def _build_chat_image_part(
    part: "typing.Dict[str, object]",
) -> "typing.Dict[str, object]":
    image_url = str(part.get("image_url", "") or "").strip()
    if not image_url:
        raise UnsupportedIncommingFeature(
            "`input_image` content parts must carry a non-empty `image_url`"
        )
    image_payload: "typing.Dict[str, object]" = {"url": image_url}
    detail = part.get("detail")
    if isinstance(detail, str) and detail in {"auto", "low", "high"}:
        image_payload["detail"] = detail
    return {"type": "image_url", "image_url": image_payload}


def _split_tool_output_parts(
    raw_output: "object",
) -> "typing.Tuple[str, typing.List[typing.Dict[str, object]]]":
    if isinstance(raw_output, str):
        return raw_output, []
    if isinstance(raw_output, list):
        return _split_content_parts(raw_output)
    return json.dumps(raw_output, ensure_ascii=False), []


def coalesce_reasoning_text(raw_item: "typing.Dict[str, object]") -> "str":
    content = raw_item.get("content")
    if isinstance(content, list):
        text_parts: "typing.List[str]" = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in {"reasoning_text", "summary_text"}:
                text_parts.append(str(part.get("text", "")))
        if text_parts:
            return "".join(text_parts)

    summary = raw_item.get("summary")
    if isinstance(summary, list):
        text_parts = []
        for part in summary:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "summary_text":
                text_parts.append(str(part.get("text", "")))
        if text_parts:
            return "".join(text_parts)

    for key in ("reasoning", "reasoning_content", "text"):
        value = raw_item.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _translate_tools(
    incomming_tools: "typing.List[object]",
    tool_names: "ToolNameMap",
    namespace: "typing.Union[str, None]" = None,
    namespace_description: "str" = "",
) -> "typing.List[typing.Dict[str, object]]":
    translated: "typing.List[typing.Dict[str, object]]" = []
    for raw_tool in incomming_tools:
        if not isinstance(raw_tool, dict):
            raise UnsupportedIncommingFeature("tool definitions must be objects")
        tool_type = raw_tool.get("type")
        if namespace is not None and tool_type not in {"function", "custom"}:
            raise UnsupportedIncommingFeature(
                "namespace tools must be function or custom tools"
            )
        if tool_type == "namespace":
            namespace_name = raw_tool.get("name")
            if not isinstance(namespace_name, str) or not namespace_name.strip():
                raise UnsupportedIncommingFeature(
                    "namespace is missing a non-empty `name`"
                )
            nested_tools = raw_tool.get("tools")
            if not isinstance(nested_tools, list):
                raise UnsupportedIncommingFeature("namespace `tools` must be a list")
            translated.extend(
                _translate_tools(
                    nested_tools,
                    tool_names,
                    namespace_name,
                    str(raw_tool.get("description", "") or ""),
                )
            )
            continue
        if tool_type == "function":
            name = tool_names.qualify(raw_tool.get("name"), namespace)
            tool = {
                "type": "function",
                "name": name,
                "function": {
                    "name": name,
                    "description": str(raw_tool.get("description", "") or ""),
                    "parameters": raw_tool.get("parameters") or {"type": "object"},
                    "strict": bool(raw_tool.get("strict", False)),
                },
            }
        elif tool_type == "web_search":
            tool_names.qualify(WebSearchTool.name)
            translated.append(build_tool_definition(WebSearchTool()))
            continue
        elif tool_type == "custom":
            try:
                tool = build_custom_tool_definition(raw_tool)
            except CustomToolAdapterError as exc:
                raise UnsupportedIncommingFeature(str(exc)) from exc
            name = tool_names.qualify(raw_tool.get("name"), namespace)
            tool["name"] = name
            tool["function"]["name"] = name
            tool_names.custom_tool_names.add(name)
        else:
            raise UnsupportedIncommingFeature(
                f"unsupported incomming tool type: {tool_type!r}"
            )
        if namespace_description:
            tool["function"]["description"] = (
                namespace_description + "\n\n" + tool["function"]["description"]
            )
        translated.append(tool)
    return translated


def _translate_tool_choice(
    raw_tool_choice: "object", tool_names: "ToolNameMap",
) -> "object":
    if isinstance(raw_tool_choice, str):
        return raw_tool_choice
    if not isinstance(raw_tool_choice, dict):
        raise UnsupportedIncommingFeature("unsupported `tool_choice` shape")

    choice_type = raw_tool_choice.get("type")
    if choice_type in {"function", "custom"}:
        name = raw_tool_choice.get("name")
        if not isinstance(name, str) or not name.strip():
            raise UnsupportedIncommingFeature(
                f"{choice_type} tool_choice is missing `name`"
            )
        return {
            "type": "function",
            "function": {
                "name": tool_names.qualify(name, raw_tool_choice.get("namespace")),
            },
        }
    raise UnsupportedIncommingFeature(
        f"unsupported incomming tool_choice type: {choice_type!r}"
    )
