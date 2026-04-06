"""pycodex 的最小协议层定义。

核心抽象：
- AgentLoop：内层 turn 主循环，负责维护历史、调用模型、执行工具，并在
  `ToolCall -> ToolResult -> 下一轮模型调用` 之间反复闭环，直到得到稳定回复。
- AgentRuntime：外层提交队列，负责按顺序接收 `Submission`，驱动
  `UserTurnOp` / `ShutdownOp` 这类运行时操作。
- ContextManager：负责把基础指令、AGENTS.md、环境信息等上下文拼成每轮模型看到的
  `Prompt` 前缀，但这些注入内容不会写回长期会话历史。
- ModelClient：模型后端抽象，接收 `Prompt`，返回 `ModelResponse`。
- ModelStreamEvent：模型在一次 `Prompt` 处理中途产生的流式事件，例如文本增量。
- ToolRegistry：工具执行抽象，接收 `ToolCall`，产出 `ToolResult`。

本文件只定义这些抽象之间传递的数据结构，不包含具体执行逻辑。
"""

from copy import deepcopy
import json
from dataclasses import dataclass, field
from typing import Any
from .compat import Literal, TypeAlias
import typing

JSONValue: 'TypeAlias' = Any
JSONDict: 'TypeAlias' = typing.Dict[str, Any]


@dataclass(frozen=True, )
class ToolSpec:
    """何时：AgentLoop 准备发起一轮模型请求时，随 `Prompt.tools` 一起发送。
    发送方：AgentLoop。
    接收方：ModelClient。
    """

    name: 'str'
    description: 'str'
    input_schema: 'typing.Union[JSONDict, None]' = None
    tool_type: 'Literal["function", "custom", "web_search"]' = "function"
    format: 'typing.Union[JSONDict, None]' = None
    options: 'typing.Union[JSONDict, None]' = None
    output_schema: 'typing.Union[JSONDict, None]' = None
    supports_parallel: 'bool' = True
    raw_payload: 'typing.Union[JSONDict, None]' = None

    def serialize(self) -> 'JSONDict':
        if self.raw_payload is not None:
            return deepcopy(self.raw_payload)
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
        if self.output_schema is not None:
            payload["output_schema"] = self.output_schema
        return payload


@dataclass(frozen=True, )
class UserMessage:
    """何时：外部发起一个新的用户 turn 时创建，并写入会话历史。
    发送方：外部调用方创建，AgentLoop 转发。
    接收方：AgentLoop 先接收，随后 ModelClient 在 `Prompt.input` 中看到它。
    """

    text: 'str'
    role: 'Literal["user"]' = "user"

    def serialize(self) -> 'JSONDict':
        return {
            "type": "message",
            "role": self.role,
            "content": [{"type": "input_text", "text": self.text}],
        }


@dataclass(frozen=True, )
class AssistantMessage:
    """何时：模型要直接输出自然语言内容时产生，可作为中间文本或最终回复。
    发送方：ModelClient。
    接收方：AgentLoop。
    """

    text: 'str'
    role: 'Literal["assistant"]' = "assistant"

    def serialize(self) -> 'JSONDict':
        return {
            "type": "message",
            "role": self.role,
            "content": [{"type": "output_text", "text": self.text}],
        }


@dataclass(frozen=True, )
class ContextMessage:
    """何时：ContextManager 为单轮模型请求注入额外上下文时构造。
    发送方：ContextManager。
    接收方：ModelClient。
    """

    text: 'typing.Union[str, None]' = None
    role: 'Literal["user", "developer"]' = "user"
    content_items: 'typing.Union[typing.Tuple[JSONDict, ...], None]' = None

    def serialize(self) -> 'JSONDict':
        if self.content_items is not None:
            content = list(self.content_items)
        else:
            if self.text is None:
                raise ValueError("ContextMessage requires `text` or `content_items`")
            content = [{"type": "input_text", "text": self.text}]
        return {
            "type": "message",
            "role": self.role,
            "content": content,
        }


@dataclass(frozen=True, )
class ToolCall:
    """何时：模型决定调用工具而不是只输出文本时产生。
    发送方：ModelClient。
    接收方：AgentLoop，随后由它转给 ToolRegistry 执行。
    """

    call_id: 'str'
    name: 'str'
    arguments: 'JSONValue'
    tool_type: 'Literal["function", "custom"]' = "function"
    kind: 'Literal["tool_call"]' = "tool_call"

    def serialize(self) -> 'JSONDict':
        if self.tool_type == "custom":
            return {
                "type": "custom_tool_call",
                "name": self.name,
                "input": str(self.arguments),
                "call_id": self.call_id,
            }
        return {
            "type": "function_call",
            "name": self.name,
            "arguments": json.dumps(
                self.arguments,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "call_id": self.call_id,
        }


@dataclass(frozen=True, )
class ReasoningItem:
    """何时：模型在一次 Responses 采样里产出 reasoning item 时产生。
    发送方：ModelClient。
    接收方：AgentLoop；它会把该 item 保留进 history，并在后续请求里原样回传给
    ModelClient。
    """

    payload: 'JSONDict'
    kind: 'Literal["reasoning"]' = "reasoning"

    def serialize(self) -> 'JSONDict':
        return deepcopy(self.payload)


@dataclass(frozen=True, )
class ToolResult:
    """何时：某个 `ToolCall` 执行完成后产生，用于喂回下一轮模型调用。
    发送方：ToolRegistry 产出，AgentLoop 追加并转发。
    接收方：AgentLoop 先接收，随后 ModelClient 在下一轮 `Prompt.input` 中看到它。
    """

    call_id: 'str'
    name: 'str'
    output: 'JSONValue'
    content_items: 'typing.Union[typing.Tuple[JSONDict, ...], None]' = None
    success: 'typing.Union[bool, None]' = None
    is_error: 'bool' = False
    tool_type: 'Literal["function", "custom"]' = "function"
    kind: 'Literal["tool_result"]' = "tool_result"

    def output_text(self) -> 'str':
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

    def serialize(self) -> 'JSONDict':
        payload_output: 'JSONValue'
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
        if self.tool_type == "custom":
            payload["name"] = self.name
        return payload


ConversationItem: 'TypeAlias' = typing.Union[typing.Union[typing.Union[typing.Union[typing.Union[UserMessage, AssistantMessage], ContextMessage], ToolCall], ReasoningItem], ToolResult]
ModelOutputItem: 'TypeAlias' = typing.Union[typing.Union[AssistantMessage, ToolCall], ReasoningItem]
Operation: 'TypeAlias' = "UserTurnOp | ShutdownOp"


@dataclass(frozen=True, )
class Prompt:
    """何时：AgentLoop 每发起一轮模型采样前构造。
    发送方：AgentLoop。
    接收方：ModelClient。
    """

    input: 'typing.List[ConversationItem]'
    tools: 'typing.List[ToolSpec]'
    parallel_tool_calls: 'bool' = True
    base_instructions: 'typing.Union[str, None]' = None
    turn_id: 'typing.Union[str, None]' = None
    turn_metadata: 'typing.Union[JSONDict, None]' = None


@dataclass(frozen=True, )
class ModelResponse:
    """何时：ModelClient 完成一轮 `Prompt` 处理后返回。
    发送方：ModelClient。
    接收方：AgentLoop。
    """

    items: 'typing.List[ModelOutputItem]'


@dataclass(frozen=True, )
class ModelStreamEvent:
    """何时：ModelClient 处理 `Prompt` 的过程中有流式中间结果时产生。
    发送方：ModelClient。
    接收方：AgentLoop。
    """

    kind: 'str'
    payload: 'JSONDict' = field(default_factory=dict)


@dataclass(frozen=True, )
class TurnResult:
    """何时：一个 turn 已经收敛，AgentLoop 决定结束本轮时返回。
    发送方：AgentLoop。
    接收方：外部调用方。
    """

    turn_id: 'str'
    output_text: 'typing.Union[str, None]'
    iterations: 'int'
    response_items: 'typing.Tuple[ModelOutputItem, ...]'
    history: 'typing.Tuple[ConversationItem, ...]'


@dataclass(frozen=True, )
class AgentEvent:
    """何时：主循环运行过程中发生阶段性事件时发出，例如模型调用、工具开始/结束。
    发送方：AgentLoop。
    接收方：可选的事件观察者 / 回调。
    """

    kind: 'str'
    turn_id: 'str'
    payload: 'typing.Dict[str, object]' = field(default_factory=dict)


@dataclass(frozen=True, )
class UserTurnOp:
    """何时：运行时要提交一个新的用户请求时创建。
    发送方：外部运行时调用方。
    接收方：AgentRuntime。
    """

    texts: 'typing.List[str]'


@dataclass(frozen=True, )
class ShutdownOp:
    """何时：运行时要停止外层提交循环时创建。
    发送方：外部运行时调用方。
    接收方：AgentRuntime。
    """

    pass


@dataclass(frozen=True, )
class Submission:
    """何时：任意运行时操作要进入 AgentRuntime 队列时创建。
    发送方：外部运行时调用方。
    接收方：AgentRuntime 的提交队列 / 外层循环。
    """

    id: 'str'
    op: 'Operation'
