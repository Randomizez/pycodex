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
from dataclasses import dataclass, replace
from pathlib import Path

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
from ..utils import get_debug_dir
from .runtime_services import AgentRuntimeEnvironment

if typing.TYPE_CHECKING:
    from ..runtime import AgentRuntime


@dataclass(
    frozen=True,
)
class ToolContext:
    turn_id: "str"
    history: "typing.Tuple[ConversationItem, ...]"
    cwd: "Path"
    agent_id: "typing.Union[str, None]" = None
    encrypted_messages: "bool" = True
    call_id: "typing.Union[str, None]" = None
    session_id: "typing.Union[str, None]" = None

    def resolve_path(self, path=None) -> "Path":
        return (self.cwd / Path("." if path is None else path).expanduser()).resolve()


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
    namespace: "typing.Union[str, None]" = None
    namespace_description: "str" = ""
    exec_only: "bool" = True
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
            namespace=self.namespace,
            namespace_description=self.namespace_description,
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

    def bind_runtime(self, runtime: "AgentRuntime") -> "None":
        pass

    def handle_agent_event(self, event: "Event") -> "None":
        pass

    def shutdown(self) -> "None":
        pass

    def background_work_count(self, after_reply: "bool") -> "int":
        return 0

    def background_work_snapshot(self) -> "typing.List[JSONDict]":
        return []

    def set_state_listener(self, listener) -> "None":
        pass


class ToolRegistry:
    def __init__(
        self, runtime_environment: "typing.Union[AgentRuntimeEnvironment, None]" = None
    ) -> "None":
        self._tools: "typing.Dict[str, BaseTool]" = {}
        self.runtime_environment = runtime_environment or AgentRuntimeEnvironment()

    def register(self, tool: "BaseTool") -> "None":
        if "." in tool.name:
            raise ValueError("tool names must not contain '.'")
        name = (
            tool.name if tool.namespace in (None, "", "functions")
            else f"{tool.namespace}.{tool.name}"
        )
        self._tools[name] = tool

    def model_visible_specs(self, code_mode_only: "bool" = False) -> "typing.List[ToolSpec]":
        if not code_mode_only:
            return [
                tool.spec() for name, tool in self._tools.items()
                if name not in ("exec", "wait")
            ]
        direct_tools = {
            name: tool for name, tool in self._tools.items() if not tool.exec_only
        }
        if len(direct_tools) != len(self._tools):
            if not {"exec", "wait"}.issubset(direct_tools):
                raise ValueError("CodeModeOnly requires registered exec and wait tools")
        return [tool.spec() for tool in direct_tools.values()]

    def supports_parallel(self, tool_name: "str") -> "bool":
        tool = self._tools.get(tool_name)
        return False if tool is None else tool.supports_parallel

    async def execute(self, call: "ToolCall", context: "ToolContext") -> "ToolResult":
        tool = self._tools.get(call.name)
        if tool is None:
            self.runtime_environment.tool_completed()
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                output={"error": f"unknown tool: {call.name}"},
                is_error=True,
                tool_type=call.tool_type,
            )

        try:
            output = await tool.run(
                replace(context, call_id=call.call_id),
                call.arguments,
            )
            if isinstance(output, StructuredToolOutput):
                return ToolResult(
                    call_id=call.call_id,
                    name=call.name,
                    output=output.output,
                    content_items=output.content_items,
                    success=output.success,
                    is_error=output.success is False,
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
        finally:
            self.runtime_environment.tool_completed()

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
