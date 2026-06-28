"""Shared truncation helpers for model-visible tool output."""

import math

from ..protocol import JSONValue, ToolResult
import typing

DEFAULT_MAX_OUTPUT_TOKENS = 10_000
TRUNCATION_SERIALIZATION_BUDGET_MULTIPLIER = 1.2
HISTORY_TOOL_OUTPUT_TOKENS = int(
    math.ceil(DEFAULT_MAX_OUTPUT_TOKENS * TRUNCATION_SERIALIZATION_BUDGET_MULTIPLIER)
)
APPROX_BYTES_PER_TOKEN = 4


def approx_token_count(text: 'str') -> 'int':
    """Estimate token count using the upstream Codex 4-bytes-per-token rule."""
    if not text:
        return 0
    byte_length = len(text.encode("utf-8"))
    return max(
        1,
        (byte_length + APPROX_BYTES_PER_TOKEN - 1) // APPROX_BYTES_PER_TOKEN,
    )


def formatted_truncate_text(text: 'str', max_tokens: 'int') -> 'str':
    """Format a direct tool response with line count plus middle truncation."""
    byte_budget = _approx_bytes_for_tokens(max_tokens)
    if len(text.encode("utf-8")) <= byte_budget:
        return text

    total_lines = len(text.splitlines())
    return f"Total output lines: {total_lines}\n\n{_truncate_text(text, max_tokens)}"


def truncate_tool_result_for_history(
    result: 'ToolResult',
) -> 'ToolResult':
    """Truncate model-visible ToolResult content before storing it in history."""
    if result.content_items is not None:
        truncated_content_items = _truncate_content_items(
            result.content_items,
            HISTORY_TOOL_OUTPUT_TOKENS,
        )
        if truncated_content_items == result.content_items:
            return result
        return ToolResult(
            call_id=result.call_id,
            name=result.name,
            output=result.output,
            content_items=truncated_content_items,
            success=result.success,
            is_error=result.is_error,
            tool_type=result.tool_type,
        )

    output_text = _tool_output_text(result.output)
    truncated_output = _truncate_text(output_text, HISTORY_TOOL_OUTPUT_TOKENS)
    if truncated_output == output_text:
        return result
    return ToolResult(
        call_id=result.call_id,
        name=result.name,
        output=truncated_output,
        success=result.success,
        is_error=result.is_error,
        tool_type=result.tool_type,
    )


def truncate_tool_results_for_history(
    results: 'typing.Iterable[ToolResult]',
) -> 'typing.List[ToolResult]':
    """Apply history-layer truncation to a batch of completed tool results."""
    return [
        truncate_tool_result_for_history(result)
        for result in results
    ]


def _tool_output_text(output: 'JSONValue') -> 'str':
    if isinstance(output, str):
        return output

    import json

    return json.dumps(
        output,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _truncate_content_items(
    content_items: 'typing.Tuple[typing.Dict[str, typing.Any], ...]',
    token_limit: 'int',
) -> 'typing.Tuple[typing.Dict[str, typing.Any], ...]':
    output: 'typing.List[typing.Dict[str, typing.Any]]' = []
    remaining_budget = token_limit
    omitted_text_items = 0

    for item in content_items:
        if item.get("type") != "input_text":
            output.append(dict(item))
            continue

        text = str(item.get("text", ""))
        if remaining_budget <= 0:
            omitted_text_items += 1
            continue

        cost = approx_token_count(text)
        if cost <= remaining_budget:
            output.append(dict(item))
            remaining_budget -= cost
            continue

        truncated_text = _truncate_text(text, remaining_budget)
        if truncated_text:
            next_item = dict(item)
            next_item["text"] = truncated_text
            output.append(next_item)
        else:
            omitted_text_items += 1
        remaining_budget = 0

    if omitted_text_items > 0:
        output.append(
            {
                "type": "input_text",
                "text": f"[omitted {omitted_text_items} text items ...]",
            }
        )
    return tuple(output)


def _approx_tokens_from_byte_count(byte_count: 'int') -> 'int':
    if byte_count <= 0:
        return 0
    return (byte_count + APPROX_BYTES_PER_TOKEN - 1) // APPROX_BYTES_PER_TOKEN


def _approx_bytes_for_tokens(token_count: 'int') -> 'int':
    return max(token_count, 0) * APPROX_BYTES_PER_TOKEN


def _split_budget(byte_budget: 'int') -> 'typing.Tuple[int, int]':
    left_budget = byte_budget // 2
    return left_budget, byte_budget - left_budget


def _split_string(
    text: 'str',
    beginning_bytes: 'int',
    end_bytes: 'int',
) -> 'typing.Tuple[str, str]':
    if not text:
        return "", ""

    total_bytes = len(text.encode("utf-8"))
    tail_start_target = max(total_bytes - end_bytes, 0)
    prefix_end = 0
    suffix_start = len(text)
    suffix_started = False
    current_byte = 0

    for index, char in enumerate(text):
        char_bytes = len(char.encode("utf-8"))
        char_start = current_byte
        char_end = current_byte + char_bytes
        if char_end <= beginning_bytes:
            prefix_end = index + 1
            current_byte = char_end
            continue
        if char_start >= tail_start_target:
            if not suffix_started:
                suffix_start = index
                suffix_started = True
            current_byte = char_end
            continue
        current_byte = char_end

    if suffix_start < prefix_end:
        suffix_start = prefix_end

    return text[:prefix_end], text[suffix_start:]


def _truncate_text(text: 'str', max_tokens: 'int') -> 'str':
    if not text:
        return ""

    max_bytes = _approx_bytes_for_tokens(max_tokens)
    total_bytes = len(text.encode("utf-8"))
    if total_bytes <= max_bytes:
        return text

    removed_tokens = _approx_tokens_from_byte_count(total_bytes - max_bytes)
    marker = f"\u2026{removed_tokens} tokens truncated\u2026"
    if max_bytes == 0:
        return marker

    left_budget, right_budget = _split_budget(max_bytes)
    prefix, suffix = _split_string(text, left_budget, right_budget)
    return f"{prefix}{marker}{suffix}"
