"""Shared runtime for Codex `exec` / `wait` tools.

Original Codex mapping:
- Corresponds to the code-mode runtime behind Codex `exec` and `wait`.

Expected behavior:
- Run raw JavaScript source in a background cell.
- Let JavaScript call nested local tools through a `tools` object.
- Support yielding/running cells and later polling or terminating them via
  `wait`.
"""

import asyncio
import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from ..compat import is_ascii, stream_writer_is_closing
from ..protocol import JSONDict, JSONValue, ToolCall
from .base_tool import StructuredToolOutput, ToolContext, ToolRegistry
import typing

DEFAULT_WAIT_YIELD_TIME_MS = 10_000
DEFAULT_MAX_OUTPUT_TOKENS = 10_000
CHARS_PER_TOKEN = 4
EXEC_PRAGMA_PREFIX = "// @exec:"
WAIT_COMPLETION_GRACE_SECONDS = 0.02


@dataclass
class ExecCell:
    cell_id: 'str'
    process: 'asyncio.subprocess.Process'
    started_at: 'float'
    output_items: 'typing.List[JSONDict]' = field(default_factory=list)
    delivered_count: 'int' = 0
    reader_task: 'typing.Union[asyncio.Task, None]' = None
    stderr_task: 'typing.Union[asyncio.Task, None]' = None
    yield_event: 'asyncio.Event' = field(default_factory=asyncio.Event)
    output_event: 'asyncio.Event' = field(default_factory=asyncio.Event)
    done_event: 'asyncio.Event' = field(default_factory=asyncio.Event)
    completed: 'bool' = False
    terminated: 'bool' = False
    error_text: 'typing.Union[str, None]' = None
    stderr_chunks: 'typing.List[str]' = field(default_factory=list)


@dataclass(frozen=True, )
class ParsedExecSource:
    code: 'str'
    yield_time_ms: 'typing.Union[int, None]'
    max_output_tokens: 'typing.Union[int, None]'


class CodeModeManager:
    def __init__(self, registry: 'ToolRegistry', cwd: 'typing.Union[typing.Union[str, Path], None]' = None) -> 'None':
        self._registry = registry
        self._default_cwd = Path(cwd or Path.cwd()).resolve()
        self._runtime_script = Path(__file__).with_name("exec_runtime.js")
        self._stored_values: 'typing.Dict[str, JSONValue]' = {}
        self._cells: 'typing.Dict[str, ExecCell]' = {}
        self._lock = asyncio.Lock()

    async def exec(self, source: 'str', context: 'ToolContext') -> 'typing.Union[StructuredToolOutput, str]':
        try:
            parsed = self._parse_exec_source(source)
        except ValueError as exc:
            return f"Error: {exc}"

        cell = await self._start_cell(parsed.code, context)
        await self._wait_for_exec(cell, parsed.yield_time_ms)
        return await self._snapshot_cell(cell, parsed.max_output_tokens)

    async def wait(
        self,
        cell_id: 'str',
        yield_time_ms: 'int',
        max_tokens: 'typing.Union[int, None]',
        terminate: 'bool',
    ) -> 'typing.Union[StructuredToolOutput, str]':
        cell = self._cells.get(cell_id)
        if cell is None:
            return f"Error: unknown exec cell `{cell_id}`."

        if terminate and cell.process.returncode is None:
            cell.terminated = True
            cell.process.terminate()
            try:
                await asyncio.wait_for(cell.process.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                cell.process.kill()
                await cell.process.wait()

        await self._wait_for_wait(cell, yield_time_ms)
        return await self._snapshot_cell(cell, max_tokens)

    def enabled_tools(self) -> 'typing.List[typing.Dict[str, str]]':
        enabled: 'typing.List[typing.Dict[str, str]]' = []
        for tool in self._registry.tools():
            if tool.name in {"exec", "wait"}:
                continue
            if tool.tool_type not in {"function", "custom"}:
                continue
            enabled.append(
                {
                    "tool_name": tool.name,
                    "js_name": self._normalize_identifier(tool.name),
                    "description": tool.description,
                    "tool_type": tool.tool_type,
                }
            )
        enabled.sort(key=lambda item: item["tool_name"])
        return enabled

    async def _start_cell(self, code: 'str', context: 'ToolContext') -> 'ExecCell':
        cell_id = uuid.uuid4().hex[:10]
        process = await asyncio.create_subprocess_exec(
            "node",
            str(self._runtime_script),
            cwd=str(self._default_cwd),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        cell = ExecCell(
            cell_id=cell_id,
            process=process,
            started_at=asyncio.get_running_loop().time(),
        )
        self._cells[cell_id] = cell
        cell.reader_task = asyncio.create_task(self._read_stdout(cell, context))
        cell.stderr_task = asyncio.create_task(self._read_stderr(cell))
        await self._send_message(
            cell,
            {
                "type": "init",
                "cell_id": cell_id,
                "source": code,
                "stored_values": self._stored_values,
                "tools": self.enabled_tools(),
            },
        )
        logger.debug("exec start cell_id={} cwd={}", cell_id, self._default_cwd)
        return cell

    async def _read_stdout(self, cell: 'ExecCell', context: 'ToolContext') -> 'None':
        stream = cell.process.stdout
        if stream is None:
            cell.error_text = "missing stdout pipe"
            cell.done_event.set()
            return

        while True:
            line = await stream.readline()
            if not line:
                break
            try:
                message = json.loads(line.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                cell.output_items.append(
                    {
                        "type": "input_text",
                        "text": line.decode("utf-8", errors="replace").rstrip("\n"),
                    }
                )
                cell.output_event.set()
                continue

            msg_type = message.get("type")
            if msg_type == "output_text":
                cell.output_items.append(
                    {
                        "type": "input_text",
                        "text": str(message.get("text", "")),
                    }
                )
                cell.output_event.set()
                continue
            if msg_type == "output_image":
                image_item: 'JSONDict' = {
                    "type": "input_image",
                    "image_url": str(message.get("image_url", "")),
                }
                detail = message.get("detail")
                if detail is not None:
                    image_item["detail"] = detail
                cell.output_items.append(image_item)
                cell.output_event.set()
                continue
            if msg_type == "yield":
                cell.yield_event.set()
                continue
            if msg_type == "tool_call":
                await self._handle_nested_tool_call(cell, context, message)
                continue
            if msg_type == "result":
                cell.completed = True
                cell.error_text = self._coerce_optional_text(message.get("error_text"))
                stored_values = message.get("stored_values")
                if isinstance(stored_values, dict):
                    async with self._lock:
                        self._stored_values = stored_values
                cell.done_event.set()
                cell.output_event.set()
                continue

        await cell.process.wait()
        if cell.stderr_task is not None:
            await cell.stderr_task
        if not cell.done_event.is_set():
            stderr_text = "".join(cell.stderr_chunks).strip()
            if stderr_text:
                cell.error_text = stderr_text
            elif cell.process.returncode not in (0, None):
                cell.error_text = f"process exited with code {cell.process.returncode}"
            cell.done_event.set()
            cell.output_event.set()

    async def _read_stderr(self, cell: 'ExecCell') -> 'None':
        stream = cell.process.stderr
        if stream is None:
            return
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            cell.stderr_chunks.append(chunk.decode("utf-8", errors="replace"))

    async def _handle_nested_tool_call(
        self,
        cell: 'ExecCell',
        context: 'ToolContext',
        message: 'JSONDict',
    ) -> 'None':
        tool_name = str(message.get("tool_name", ""))
        request_id = str(message.get("id", ""))
        tool = self._registry.get_tool(tool_name)
        if tool is None:
            await self._send_message(
                cell,
                {
                    "type": "tool_result",
                    "id": request_id,
                    "ok": False,
                    "error": f"unknown tool: {tool_name}",
                },
            )
            return
        if tool.tool_type not in {"function", "custom"}:
            await self._send_message(
                cell,
                {
                    "type": "tool_result",
                    "id": request_id,
                    "ok": False,
                    "error": f"tool `{tool_name}` is not available inside exec",
                },
            )
            return

        result = await self._registry.execute(
            ToolCall(
                call_id=f"{cell.cell_id}_{request_id}",
                name=tool_name,
                arguments=message.get("arguments"),
                tool_type=tool.tool_type,
            ),
            ToolContext(turn_id=context.turn_id, history=context.history),
        )
        if result.is_error:
            payload = {
                "type": "tool_result",
                "id": request_id,
                "ok": False,
                "error": result.output_text(),
            }
        else:
            payload = {
                "type": "tool_result",
                "id": request_id,
                "ok": True,
                "result": result.output,
            }
        await self._send_message(cell, payload)

    async def _send_message(self, cell: 'ExecCell', payload: 'JSONDict') -> 'None':
        stdin = cell.process.stdin
        if stdin is None or stream_writer_is_closing(stdin):
            return
        stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        await stdin.drain()

    async def _wait_for_exec(self, cell: 'ExecCell', yield_time_ms: 'typing.Union[int, None]') -> 'None':
        done_task = asyncio.create_task(cell.done_event.wait())
        yield_task = asyncio.create_task(cell.yield_event.wait())
        tasks = {done_task, yield_task}
        try:
            if yield_time_ms is None:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            else:
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
        cell.yield_event.clear()

    async def _wait_for_wait(self, cell: 'ExecCell', yield_time_ms: 'int') -> 'None':
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

        cell.output_event.clear()
        cell.yield_event.clear()

    async def _wait_for_completion_grace(
        self,
        cell: 'ExecCell',
        timeout_seconds: 'float',
    ) -> 'None':
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

    async def _snapshot_cell(
        self,
        cell: 'ExecCell',
        max_tokens: 'typing.Union[int, None]',
    ) -> 'StructuredToolOutput':
        if cell.process.returncode is not None and cell.reader_task is not None:
            await cell.reader_task

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

        return StructuredToolOutput(output=output_text, content_items=tuple(content_items))

    def _truncate_content_items(
        self,
        items: 'typing.List[JSONDict]',
        max_tokens: 'typing.Union[int, None]',
    ) -> 'typing.List[JSONDict]':
        token_budget = DEFAULT_MAX_OUTPUT_TOKENS if max_tokens is None else max_tokens
        max_chars = max(1, token_budget) * CHARS_PER_TOKEN
        total_chars = 0
        truncated: 'typing.List[JSONDict]' = []
        for item in items:
            if item.get("type") != "input_text":
                truncated.append(item)
                continue
            text = str(item.get("text", ""))
            remaining = max_chars - total_chars
            if remaining <= 0:
                break
            if len(text) > remaining:
                truncated.append(
                    {
                        "type": "input_text",
                        "text": text[:remaining] + "\n...[truncated]...",
                    }
                )
                total_chars = max_chars
                break
            truncated.append(item)
            total_chars += len(text)
        return truncated

    def _status_text(self, cell: 'ExecCell') -> 'str':
        if cell.terminated:
            return "Script terminated"
        if not cell.done_event.is_set():
            return f"Script running with cell ID {cell.cell_id}"
        if cell.error_text:
            return "Script failed"
        return "Script completed"

    def _parse_exec_source(self, input_text: 'str') -> 'ParsedExecSource':
        if not input_text.strip():
            raise ValueError(
                "exec expects raw JavaScript source text (non-empty)."
            )
        code = input_text
        yield_time_ms = None
        max_output_tokens = None
        lines = input_text.split("\n", 1)
        first_line = lines[0].lstrip()
        if first_line.startswith(EXEC_PRAGMA_PREFIX):
            if len(lines) == 1 or not lines[1].strip():
                raise ValueError(
                    "exec pragma must be followed by JavaScript source on subsequent lines"
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
            if value.get("yield_time_ms") is not None:
                yield_time_ms = int(value["yield_time_ms"])
            if value.get("max_output_tokens") is not None:
                max_output_tokens = int(value["max_output_tokens"])
            code = lines[1]
        return ParsedExecSource(
            code=code,
            yield_time_ms=yield_time_ms,
            max_output_tokens=max_output_tokens,
        )

    def _normalize_identifier(self, tool_name: 'str') -> 'str':
        identifier = []
        for index, char in enumerate(tool_name):
            is_valid = (
                char == "_"
                or char == "$"
                or (is_ascii(char) and char.isalnum() and (index != 0 or char.isalpha()))
            )
            if is_valid:
                identifier.append(char)
            else:
                identifier.append("_")
        return "".join(identifier) or "_"

    def _coerce_optional_text(self, value: 'JSONValue') -> 'typing.Union[str, None]':
        if value in (None, ""):
            return None
        return str(value)
