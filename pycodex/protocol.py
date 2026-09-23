"""pycodex 的最小协议层定义。

核心抽象：
- Agent：内层 turn 主循环，负责维护历史、调用模型、执行工具，并在
  `ToolCall -> ToolResult -> 下一轮模型调用` 之间反复闭环，直到得到稳定回复。
- AgentRuntime：外层调度与前端接口，负责按顺序处理用户请求；
  队列生命周期不作为请求入队。
- ContextManager：负责把基础指令、AGENTS.md、环境信息等上下文拼成每轮模型看到的
  `Prompt` 前缀，但这些注入内容不会写回长期会话历史。
- ModelClient：模型后端抽象，接收 `Prompt`，返回 `ModelResponse`。
- ToolRegistry：工具执行抽象，接收 `ToolCall`，产出 `ToolResult`。

本文件只定义这些抽象之间传递的数据结构，不包含具体执行逻辑。
"""

import json
import typing
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .compat import Literal, TypeAlias

JSONValue: "TypeAlias" = Any
JSONDict: "TypeAlias" = typing.Dict[str, Any]


@dataclass(
    frozen=True,
)
class ToolSpec:
    """何时：Agent 准备发起一轮模型请求时，随 `Prompt.tools` 一起发送。
    发送方：Agent。
    接收方：ModelClient。
    """

    name: "str"
    description: "str"
    input_schema: "typing.Union[JSONDict, None]" = None
    tool_type: 'Literal["function", "custom", "web_search"]' = "function"
    format: "typing.Union[JSONDict, None]" = None
    options: "typing.Union[JSONDict, None]" = None
    output_schema: "typing.Union[JSONDict, None]" = None
    supports_parallel: "bool" = True

    def serialize(self) -> "JSONDict":
        if self.tool_type == "web_search":
            payload = {"type": "web_search"}
            if self.options is not None:
                payload.update(self.options)
            return payload

        if self.tool_type == "custom":
            if self.format is None:
                raise ValueError("custom tools require `format`")
            return {
                "type": "custom",
                "name": self.name,
                "description": self.description,
                "format": self.format,
            }

        if self.input_schema is None:
            raise ValueError("function tools require `input_schema`")

        payload = {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
            "strict": False,
        }
        return payload


@dataclass(
    frozen=True,
)
class UserMessage:
    """何时：外部发起一个新的用户 turn 时创建，并写入会话历史。
    发送方：外部调用方创建，Agent 转发。
    接收方：Agent 先接收，随后 ModelClient 在 `Prompt.input` 中看到它。
    """

    text: "str"
    role: 'Literal["user"]' = "user"
    id: "typing.Union[str, None]" = None

    def serialize(self) -> "JSONDict":
        payload = {
            "type": "message",
            "role": self.role,
            "content": [{"type": "input_text", "text": self.text}],
        }
        if self.id is not None:
            payload["id"] = self.id
        return payload


@dataclass(
    frozen=True,
)
class AssistantMessage:
    """何时：模型要直接输出自然语言内容时产生，可作为中间文本或最终回复。
    发送方：ModelClient。
    接收方：Agent。
    """

    text: "str"
    role: 'Literal["assistant"]' = "assistant"
    id: "typing.Union[str, None]" = None
    phase: "typing.Union[str, None]" = None
    content_items: "typing.Union[typing.Tuple[JSONDict, ...], None]" = None

    @classmethod
    def from_response_item(cls, payload: "JSONDict") -> "AssistantMessage":
        content = tuple(
            {"type": part["type"], "text": str(part.get("text", ""))}
            for part in payload.get("content", [])
            if part.get("type") == "output_text"
        )
        return cls(
            text="".join(part["text"] for part in content),
            id=payload.get("id"),
            phase=payload.get("phase"),
            content_items=content if len(content) != 1 else None,
        )

    def serialize(self) -> "JSONDict":
        payload = {
            "type": "message",
            "role": self.role,
            "content": (
                deepcopy(list(self.content_items))
                if self.content_items is not None
                else [{"type": "output_text", "text": self.text}]
            ),
        }
        if self.id is not None:
            payload["id"] = self.id
        if self.phase is not None:
            payload["phase"] = self.phase
        return payload


@dataclass(
    frozen=True,
)
class ContextMessage:
    """何时：注入额外上下文或用 compact 摘要替换历史时构造。
    发送方：ContextManager 或 compactor。
    接收方：ModelClient。
    """

    text: "typing.Union[str, None]" = None
    role: 'Literal["user", "developer"]' = "user"
    content_items: "typing.Union[typing.Tuple[JSONDict, ...], None]" = None
    id: "typing.Union[str, None]" = None

    def serialize(self) -> "JSONDict":
        if self.content_items is not None:
            content = list(self.content_items)
        else:
            if self.text is None:
                raise ValueError("ContextMessage requires `text` or `content_items`")
            content = [{"type": "input_text", "text": self.text}]
        payload = {
            "type": "message",
            "role": self.role,
            "content": content,
        }
        if self.id is not None:
            payload["id"] = self.id
        return payload


@dataclass(
    frozen=True,
)
class ToolCall:
    """何时：模型决定调用工具而不是只输出文本时产生。
    发送方：ModelClient。
    接收方：Agent，随后由它转给 ToolRegistry 执行。
    """

    call_id: "str"
    name: "str"
    arguments: "JSONValue"
    tool_type: 'Literal["function", "custom"]' = "function"
    kind: 'Literal["tool_call"]' = "tool_call"
    id: "typing.Union[str, None]" = None
    raw_arguments: "typing.Union[str, None]" = None
    namespace: "typing.Union[str, None]" = None
    status: "typing.Union[str, None]" = None

    def serialize(self) -> "JSONDict":
        if self.tool_type == "custom":
            payload = {
                "type": "custom_tool_call",
                "name": self.name,
                "input": str(self.arguments),
                "call_id": self.call_id,
            }
        else:
            payload = {
                "type": "function_call",
                "name": self.name,
                "arguments": (
                    self.raw_arguments
                    if self.raw_arguments is not None
                    else json.dumps(
                        self.arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
                "call_id": self.call_id,
            }
        if self.id is not None:
            payload["id"] = self.id
        if self.namespace is not None:
            payload["namespace"] = self.namespace
        if self.tool_type == "custom" and self.status is not None:
            payload["status"] = self.status
        return payload


@dataclass(
    frozen=True,
)
class ReasoningItem:
    """何时：模型在一次 Responses 采样里产出 reasoning item 时产生。
    发送方：ModelClient。
    接收方：Agent；它会把该 item 保留进 history，并在后续请求里原样回传给
    ModelClient。
    """

    payload: "JSONDict"
    kind: 'Literal["reasoning"]' = "reasoning"

    def serialize(self) -> "JSONDict":
        payload = deepcopy(self.payload)
        payload.setdefault("content", None)
        payload.setdefault("encrypted_content", None)
        if payload["content"] is not None and not any(
            item.get("type") == "reasoning_text" for item in payload["content"]
        ):
            del payload["content"]
        return payload


@dataclass(
    frozen=True,
)
class ToolResult:
    """何时：某个 `ToolCall` 执行完成后产生，用于喂回下一轮模型调用。
    发送方：ToolRegistry 产出，Agent 追加并转发。
    接收方：Agent 先接收，随后 ModelClient 在下一轮 `Prompt.input` 中看到它。
    """

    call_id: "str"
    name: "str"
    output: "JSONValue"
    content_items: "typing.Union[typing.Tuple[JSONDict, ...], None]" = None
    success: "typing.Union[bool, None]" = None
    is_error: "bool" = False
    tool_type: 'Literal["function", "custom"]' = "function"
    kind: 'Literal["tool_result"]' = "tool_result"
    id: "typing.Union[str, None]" = None

    def output_text(self) -> "str":
        if self.content_items is not None:
            text_parts = [
                str(item.get("text", ""))
                for item in self.content_items
                if item.get("type") == "input_text"
            ]
            if text_parts:
                return "\n".join(text_parts)
            if isinstance(self.output, str):
                return self.output
            return json.dumps(self.output, ensure_ascii=False)
        if isinstance(self.output, str):
            return self.output
        return json.dumps(self.output, ensure_ascii=False)

    def serialize(self) -> "JSONDict":
        payload_output: "JSONValue"
        if self.content_items is not None:
            payload_output = list(self.content_items)
        elif isinstance(self.output, str):
            payload_output = self.output
        else:
            payload_output = json.dumps(
                self.output,
                ensure_ascii=False,
                separators=(",", ":"),
            )

        item_type = (
            "custom_tool_call_output"
            if self.tool_type == "custom"
            else "function_call_output"
        )
        payload = {
            "type": item_type,
            "call_id": self.call_id,
            "output": payload_output,
        }
        if self.success is not None:
            payload["success"] = self.success
        if self.id is not None:
            payload["id"] = self.id
        if self.tool_type == "custom":
            payload["name"] = self.name
        return payload


ConversationItem: "TypeAlias" = typing.Union[
    typing.Union[
        typing.Union[
            typing.Union[typing.Union[UserMessage, AssistantMessage], ContextMessage],
            ToolCall,
        ],
        ReasoningItem,
    ],
    ToolResult,
]
ModelOutputItem: "TypeAlias" = typing.Union[
    typing.Union[AssistantMessage, ToolCall], ReasoningItem
]


@dataclass(
    frozen=True,
)
class Prompt:
    """何时：Agent 每发起一轮模型采样前构造。
    发送方：Agent。
    接收方：ModelClient。
    """

    input: "typing.List[ConversationItem]"
    tools: "typing.List[ToolSpec]"
    parallel_tool_calls: "bool" = True
    base_instructions: "typing.Union[str, None]" = None
    turn_id: "typing.Union[str, None]" = None
    turn_metadata: "typing.Union[JSONDict, None]" = None


@dataclass(
    frozen=True,
)
class ModelResponse:
    """何时：ModelClient 完成一轮 `Prompt` 处理后返回。
    发送方：ModelClient。
    接收方：Agent。
    """

    items: "typing.List[ModelOutputItem]"

    def __post_init__(self) -> "None":
        for item in self.items:
            if not isinstance(item, (AssistantMessage, ToolCall, ReasoningItem)):
                raise TypeError(f"invalid model output item: {type(item).__name__}")


@dataclass(
    frozen=True,
)
class TurnResult:
    """何时：一个 turn 已经收敛，Agent 决定结束本轮时返回。
    发送方：Agent。
    接收方：外部调用方。
    """

    turn_id: "str"
    output_text: "typing.Union[str, None]"
    iterations: "int"
    response_items: "typing.Tuple[ModelOutputItem, ...]"
    history: "typing.Tuple[ConversationItem, ...]"
