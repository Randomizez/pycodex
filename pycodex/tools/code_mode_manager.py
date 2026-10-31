"""Host for Python exec cells and their nested tool calls."""

import asyncio
import json
import math
import traceback
import typing
import uuid
from dataclasses import dataclass, field, replace

from loguru import logger

from ..protocol import JSONDict, JSONValue, ToolCall, ToolResult
from ..utils.truncation import DEFAULT_MAX_OUTPUT_TOKENS
from .base_tool import StructuredToolOutput, ToolContext, ToolRegistry
from .exec_runtime import ExecRuntime
from .python_tool_schema import python_tool_declaration

DEFAULT_EXEC_YIELD_TIME_MS = 30_000
DEFAULT_WAIT_YIELD_TIME_MS = 10_000
CHARS_PER_TOKEN = 4
EXEC_PRAGMA_PREFIX = "# @exec:"
WAIT_COMPLETION_GRACE_SECONDS = 0.02


@dataclass
class ExecCell:
    cell_id: "str"
    started_at: "float"
    call_id: "typing.Union[str, None]"
    cwd: "str"
    generation: "int"
    runtime: "typing.Union[ExecRuntime, None]" = None
    task: "typing.Union[asyncio.Task, None]" = None
    output_items: "typing.List[JSONDict]" = field(default_factory=list)
    delivered_count: "int" = 0
    yield_event: "asyncio.Event" = field(default_factory=asyncio.Event)
    output_event: "asyncio.Event" = field(default_factory=asyncio.Event)
    done_event: "asyncio.Event" = field(default_factory=asyncio.Event)
    terminated: "bool" = False
    error_text: "typing.Union[str, None]" = None


@dataclass(
    frozen=True,
)
class ParsedExecSource:
    code: "str"
    yield_time_ms: "int"
    max_output_tokens: "typing.Union[int, None]"


class CodeModeManager:
    def __init__(self, registry: "ToolRegistry") -> "None":
        self._registry = registry
        self._stored_values: "typing.Dict[str, JSONValue]" = {}
        self._cells: "typing.Dict[str, ExecCell]" = {}
        self._retired_cells: "typing.List[ExecCell]" = []
        self._generation = 0
        self._output_handler = None
        self._state_listener = None
        self._closed = False

    def bind(self, invoke, output, state_changed):
        self._output_handler = output
        self._state_listener = state_changed

    def restart(self):
        self._generation += 1
        for cell in self._cells.values():
            self._stop_cell(cell)
            self._retired_cells.append(cell)
        self._cells.clear()
        self._stored_values = {}
        self._closed = False

    def set_state_listener(self, listener):
        self._state_listener = listener

    def _state_changed(self):
        if self._state_listener is not None:
            self._state_listener()

    def background_work_snapshot(self):
        return [
            {
                "cell_id": cell.cell_id,
                "cwd": cell.cwd,
                "running": not cell.done_event.is_set(),
            }
            for cell in self._cells.values()
        ]

    def shutdown(self):
        self._closed = True
        self._output_handler = None
        self._state_listener = None
        for cell in tuple(self._cells.values()) + tuple(self._retired_cells):
            self._stop_cell(cell)

    async def close(self):
        self._closed = True
        cells = tuple(self._cells.values()) + tuple(self._retired_cells)
        if cells:
            await asyncio.gather(*(self._terminate_cell(cell) for cell in cells))
        self._cells.clear()
        self._retired_cells.clear()
        self._stored_values.clear()
        self._state_changed()

    def _stop_cell(self, cell):
        if not cell.done_event.is_set():
            cell.terminated = True
            cell.runtime.request_stop()

    async def _terminate_cell(self, cell):
        self._stop_cell(cell)
        await asyncio.shield(cell.task)

    def background_work_count(self, after_reply) -> "int":
        return sum(not cell.done_event.is_set() for cell in self._cells.values())

    async def exec(
        self, source: "str", context: "ToolContext"
    ) -> "typing.Union[StructuredToolOutput, str]":
        if self._closed:
            raise RuntimeError("exec runtime is closed")
        try:
            parsed = self._parse_exec_source(source)
        except ValueError as exc:
            return f"Error: {exc}"

        cell = await self._start_cell(parsed.code, context)
        await self._wait_for_exec(cell, parsed.yield_time_ms)
        return await self._snapshot_cell(cell, parsed.max_output_tokens)

    async def wait(
        self,
        cell_id: "str",
        yield_time_ms: "int",
        max_tokens: "typing.Union[int, None]",
        terminate: "bool",
    ) -> "typing.Union[StructuredToolOutput, str]":
        cell = self._cells.get(cell_id)
        if cell is None:
            return f"Error: unknown exec cell `{cell_id}`."

        if terminate:
            await self._terminate_cell(cell)

        await self._wait_for_wait(cell, yield_time_ms)
        return await self._snapshot_cell(cell, max_tokens)

    def describe_tools(self):
        sections = []
        for tool in self._registry.exec_tools():
            sections.append(
                "\n\n### `" + tool["namespace"] + "." + tool["name"] + "`\n"
                + tool["description"]
                + "\n\nexec tool declaration:\n```python\n"
                + python_tool_declaration(tool) + "\n```"
            )
        return "".join(sections)

    async def _start_cell(self, code: "str", context: "ToolContext") -> "ExecCell":
        tools = self._registry.exec_tools()
        cell_id = uuid.uuid4().hex[:10]
        cell = ExecCell(
            cell_id=cell_id,
            started_at=asyncio.get_running_loop().time(),
            call_id=context.call_id,
            cwd=str(context.cwd),
            generation=self._generation,
        )
        cell.runtime = ExecRuntime(
            code, tools, self._stored_values, context.cwd,
            lambda message: self._receive(cell, message),
            lambda spec, request_id, args: self._invoke_tool(
                cell, context, spec, request_id, args,
            ),
        )
        self._cells[cell_id] = cell
        cell.task = asyncio.create_task(self._run_cell(cell))
        self._state_changed()
        logger.debug("exec start cell_id={} cwd={}", cell_id, context.cwd)
        return cell

    async def _run_cell(self, cell):
        try:
            cell.error_text = await cell.runtime.run()
        except Exception:
            cell.error_text = traceback.format_exc()
        finally:
            cell.done_event.set()
            cell.output_event.set()
            self._state_changed()

    def _receive(self, cell, message):
        if cell.generation != self._generation:
            return
        msg_type = message["type"]
        if msg_type == "content":
            cell.output_items.append(message["item"])
            cell.output_event.set()
        elif msg_type == "yield":
            cell.yield_event.set()
        elif msg_type == "notify":
            if self._output_handler is None or cell.call_id is None:
                raise RuntimeError("notify requires an attached Runtime")
            self._output_handler(ToolResult(
                call_id=cell.call_id, name="exec",
                output=message["text"], tool_type="custom",
            ))
            cell.yield_event.set()
        else:
            raise ValueError("unknown exec cell message: " + msg_type)

    async def _invoke_tool(self, cell, context, spec, request_id, arguments):
        result = await self._registry.execute(
            ToolCall(
                call_id=f"{cell.cell_id}_{request_id}",
                name=spec["tool_name"],
                arguments=arguments,
                tool_type=spec["tool_type"],
                namespace=self._registry.get_tool(spec["tool_name"]).namespace,
            ),
            replace(context, encrypted_messages=False),
        )
        if result.is_error:
            raise RuntimeError(result.output_text())
        return result.output

    async def _wait_for_exec(
        self, cell: "ExecCell", yield_time_ms: "int"
    ) -> "None":
        done_task = asyncio.create_task(cell.done_event.wait())
        yield_task = asyncio.create_task(cell.yield_event.wait())
        tasks = {done_task, yield_task}
        try:
            await asyncio.wait_for(
                asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED),
                timeout=max(yield_time_ms, 1) / 1000.0,
            )
        except asyncio.TimeoutError:
            return
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        cell.yield_event.clear()

    async def _wait_for_wait(self, cell: "ExecCell", yield_time_ms: "int") -> "None":
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(yield_time_ms, 1) / 1000.0
        initial_count = cell.delivered_count

        while True:
            if cell.done_event.is_set():
                break
            if cell.yield_event.is_set():
                break
            if len(cell.output_items) > initial_count:
                remaining = deadline - loop.time()
                if remaining > 0:
                    await self._wait_for_completion_grace(
                        cell,
                        min(remaining, WAIT_COMPLETION_GRACE_SECONDS),
                    )
                break

            remaining = deadline - loop.time()
            if remaining <= 0:
                break

            done_task = asyncio.create_task(cell.done_event.wait())
            output_task = asyncio.create_task(cell.output_event.wait())
            yield_task = asyncio.create_task(cell.yield_event.wait())
            tasks = {done_task, output_task, yield_task}
            try:
                await asyncio.wait(
                    tasks,
                    timeout=min(remaining, 0.05),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

        cell.output_event.clear()
        cell.yield_event.clear()

    async def _wait_for_completion_grace(
        self,
        cell: "ExecCell",
        timeout_seconds: "float",
    ) -> "None":
        if timeout_seconds <= 0:
            return
        done_task = asyncio.create_task(cell.done_event.wait())
        yield_task = asyncio.create_task(cell.yield_event.wait())
        tasks = {done_task, yield_task}
        try:
            await asyncio.wait(
                tasks,
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _snapshot_cell(
        self,
        cell: "ExecCell",
        max_tokens: "typing.Union[int, None]",
    ) -> "StructuredToolOutput":
        if cell.done_event.is_set():
            await cell.task

        wall_time = asyncio.get_running_loop().time() - cell.started_at
        new_items = list(cell.output_items[cell.delivered_count :])
        cell.delivered_count = len(cell.output_items)

        if cell.error_text:
            new_items.append(
                {
                    "type": "input_text",
                    "text": f"Script error:\n{cell.error_text}",
                }
            )

        header = {
            "type": "input_text",
            "text": (
                f"{self._status_text(cell)}\n"
                f"Wall time {wall_time:.1f} seconds\n"
                "Output:\n"
            ),
        }
        content_items = [header] + new_items
        content_items = self._truncate_content_items(content_items, max_tokens)
        output_text = "\n".join(
            item.get("text", "")
            for item in content_items
            if item.get("type") == "input_text"
        )
        cell.output_event.clear()
        cell.yield_event.clear()

        if cell.done_event.is_set():
            self._cells.pop(cell.cell_id, None)
            self._state_changed()

        return StructuredToolOutput(
            output=output_text, content_items=tuple(content_items),
            success=not bool(cell.error_text),
        )

    def _truncate_content_items(
        self,
        items: "typing.List[JSONDict]",
        max_tokens: "typing.Union[int, None]",
    ) -> "typing.List[JSONDict]":
        token_budget = DEFAULT_MAX_OUTPUT_TOKENS if max_tokens is None else max_tokens
        max_chars = max(1, token_budget) * CHARS_PER_TOKEN
        total_chars = 0
        truncated: "typing.List[JSONDict]" = []
        for item in items:
            if item.get("type") != "input_text":
                truncated.append(item)
                continue
            text = str(item.get("text", ""))
            remaining = max_chars - total_chars
            if remaining <= 0:
                continue
            if len(text) > remaining:
                truncated.append(
                    {
                        "type": "input_text",
                        "text": text[:remaining] + "\n...[truncated]...",
                    }
                )
                total_chars = max_chars
                continue
            truncated.append(item)
            total_chars += len(text)
        return truncated

    def _status_text(self, cell: "ExecCell") -> "str":
        if not cell.done_event.is_set():
            return f"Script running with cell ID {cell.cell_id}"
        if cell.terminated:
            return "Script terminated"
        if cell.error_text:
            return "Script failed"
        return "Script completed"

    def _parse_exec_source(self, input_text: "str") -> "ParsedExecSource":
        if not input_text.strip():
            raise ValueError("exec expects non-empty Python source text")
        code = input_text
        yield_time_ms = DEFAULT_EXEC_YIELD_TIME_MS
        max_output_tokens = None
        lines = input_text.split("\n", 1)
        first_line = lines[0].lstrip()
        if first_line.startswith(EXEC_PRAGMA_PREFIX):
            if len(lines) == 1 or not lines[1].strip():
                raise ValueError(
                    "exec pragma must be followed by Python source on subsequent lines"
                )
            pragma_text = first_line[len(EXEC_PRAGMA_PREFIX) :].strip()
            try:
                value = json.loads(pragma_text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "exec pragma must be valid JSON with supported fields `yield_time_ms` and `max_output_tokens`: "
                    f"{exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    "exec pragma must be a JSON object with supported fields `yield_time_ms` and `max_output_tokens`"
                )
            for key in value:
                if key not in {"yield_time_ms", "max_output_tokens"}:
                    raise ValueError(f"unsupported exec pragma field `{key}`")
            for key, number in value.items():
                if (
                    isinstance(number, bool) or not isinstance(number, (int, float))
                    or not math.isfinite(number) or int(number) != number or number < 0
                ):
                    raise ValueError("exec pragma field " + key + " must be a non-negative integer")
            if "yield_time_ms" in value:
                yield_time_ms = int(value["yield_time_ms"])
            if "max_output_tokens" in value:
                max_output_tokens = int(value["max_output_tokens"])
            code = lines[1]
        return ParsedExecSource(
            code=code,
            yield_time_ms=yield_time_ms,
            max_output_tokens=max_output_tokens,
        )
