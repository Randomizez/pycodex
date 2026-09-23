"""Shared tool abstractions for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the registry/handler layer behind Rust Codex tool routing
  rather than one single end-user tool.

Expected behavior:
- `BaseTool` defines the contract every local Python tool must implement.
- `ToolRegistry` stores concrete tool instances, exposes tool specs to the
  model, and dispatches `ToolCall` executions back into `ToolResult`s.
"""

import asyncio
import json
import traceback
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..events import Event
from ..protocol import (
    ConversationItem,
    JSONDict,
    JSONValue,
    ToolCall,
    ToolResult,
    ToolSpec,
    UserMessage,
)
from ..runtime_services import AgentRuntimeEnvironment
from ..utils import get_debug_dir

if typing.TYPE_CHECKING:
    from ..agent import Agent


@dataclass(
    frozen=True,
)
class ToolContext:
    turn_id: "str"
    history: "typing.Tuple[ConversationItem, ...]"


class StructuredToolOutput:
    def __init__(
        self,
        output: "JSONValue",
        content_items: "typing.Union[typing.Union[typing.Tuple[JSONDict, ...], typing.List[JSONDict]], None]" = None,
        success: "typing.Union[bool, None]" = None,
    ) -> "None":
        self.output = output
        self.content_items = None if content_items is None else tuple(content_items)
        self.success = success


class BaseTool(ABC):
    name: "str"
    description: "str"
    input_schema: "typing.Union[JSONDict, None]" = None
    tool_type: "str" = "function"
    format: "typing.Union[JSONDict, None]" = None
    options: "typing.Union[JSONDict, None]" = None
    output_schema: "typing.Union[JSONDict, None]" = None
    supports_parallel: "bool" = True

    def spec(self) -> "ToolSpec":
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            tool_type=self.tool_type,
            format=self.format,
            options=self.options,
            output_schema=self.output_schema,
            supports_parallel=self.supports_parallel,
        )

    def serialize(self) -> "JSONDict":
        return self.spec().serialize()

    @abstractmethod
    async def run(
        self, context: "ToolContext", args: "JSONValue"
    ) -> "typing.Union[JSONValue, StructuredToolOutput]":
        raise NotImplementedError

    def follow_up_messages(
        self, output: "JSONValue"
    ) -> "typing.Tuple[UserMessage, ...]":
        return ()

    def bind_agent(self, agent: "Agent") -> "None":
        pass

    def handle_agent_event(self, event: "Event") -> "None":
        pass

    def shutdown(self) -> "None":
        pass

    def background_work_count(self, after_reply: "bool") -> "int":
        return 0


class ToolRegistry:
    def __init__(
        self, runtime_environment: "typing.Union[AgentRuntimeEnvironment, None]" = None
    ) -> "None":
        self._tools: "typing.Dict[str, BaseTool]" = {}
        self.runtime_environment = runtime_environment or AgentRuntimeEnvironment()

    def register(self, tool: "BaseTool") -> "None":
        self._tools[tool.name] = tool

    def model_visible_specs(self) -> "typing.List[ToolSpec]":
        return [tool.spec() for tool in self._tools.values()]

    def supports_parallel(self, tool_name: "str") -> "bool":
        tool = self._tools.get(tool_name)
        return False if tool is None else tool.supports_parallel

    async def execute(self, call: "ToolCall", context: "ToolContext") -> "ToolResult":
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                output={"error": f"unknown tool: {call.name}"},
                is_error=True,
                tool_type=call.tool_type,
            )

        try:
            output = await tool.run(context, call.arguments)
            if isinstance(output, StructuredToolOutput):
                return ToolResult(
                    call_id=call.call_id,
                    name=call.name,
                    output=output.output,
                    content_items=output.content_items,
                    success=output.success,
                    tool_type=call.tool_type,
                )
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                output=output,
                tool_type=call.tool_type,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            debug_dir = get_debug_dir()
            if debug_dir is not None:
                with (debug_dir / "tool_errors.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(
                        json.dumps(
                            {
                                "tool": call.name,
                                "call_id": call.call_id,
                                "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc(),
                            },
                            ensure_ascii=False,
                        )
                    )
                    handle.write("\n")
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                output={"error": f"{type(exc).__name__}: {exc}"},
                is_error=True,
                tool_type=call.tool_type,
            )

    def follow_up_messages(
        self, results: "typing.Iterable[ToolResult]"
    ) -> "typing.Tuple[UserMessage, ...]":
        messages = []
        for result in results:
            if not result.is_error:
                messages.extend(
                    self._tools[result.name].follow_up_messages(result.output)
                )
        return tuple(messages)

    def __contains__(self, tool_name: "str") -> "bool":
        return tool_name in self._tools

    def __len__(self) -> "int":
        return len(self._tools)

    def names(self) -> "typing.Tuple[str, ...]":
        return tuple(self._tools)

    def get_tool(self, tool_name: "str") -> "typing.Union[BaseTool, None]":
        return self._tools.get(tool_name)

    def tools(self) -> "typing.Tuple[BaseTool, ...]":
        return tuple(self._tools.values())


Registry = ToolRegistry
