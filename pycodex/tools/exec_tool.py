"""Python implementation of the CodeModeOnly `exec` entrypoint."""

from dataclasses import replace

from ..protocol import JSONValue
from .base_tool import BaseTool, ToolContext
from .code_mode_manager import CodeModeManager

EXEC_FREEFORM_GRAMMAR = r"""start: pragma_source | plain_source
pragma_source: PRAGMA_LINE NEWLINE SOURCE
plain_source: SOURCE

PRAGMA_LINE: /[ \t]*# @exec:[^\r\n]*/
NEWLINE: /\r?\n/
SOURCE: /[\s\S]+/
"""


class ExecTool(BaseTool):
    name = "exec"
    exec_only = False
    description = (
        "Run Python code to orchestrate and compose tool calls.\n"
        "- Submit raw Python source with top-level `await`; the runner executes "
        "it in the host process with a fresh namespace for each cell. "
        "Imports and helper definitions are allowed. "
        "Do not wrap the source in a main function, JSON, or Markdown.\n"
        "- The final top-level expression is displayed with `repr()` unless its "
        "value is None. Assignment-only cells do not display a value. "
        "`print()` and `text()` remain available for explicit output.\n"
        "- To run a shell command, submit Python such as "
        '`await tools.exec_command(cmd="git status --short")`.\n'
        "- Call ordinary nested tools through `tools.<name>(...)` and agent "
        "tools through `collaboration.<name>(...)` inside this Python source. "
        "Collaboration tools also remain available as direct tool calls. "
        "Both namespaces are listed below. Follow the Python signatures: use "
        "positional or keyword arguments, and `**arguments` to unpack a dict. "
        "Freeform tools accept one string, either "
        "positionally or with `input=...`.\n"
        "- Tool calls are asynchronous: use `await tools.exec_command(cmd=\"pwd\")`. "
        "Use `await asyncio.gather(...)` for independent calls, or "
        "`await asyncio.gather(..., return_exceptions=True)` to inspect every "
        "success and failure. Keep dependent operations sequential.\n"
        "- `asyncio` and the Python standard library are available. Tasks created "
        "through asyncio from cell tasks, and timers created with `set_timeout`, "
        "are cleaned up when the script finishes. Already issued host tool calls "
        "finish before the cell is closed.\n"
        "- Execution and stopping are cooperative. Use async operations and run "
        "blocking work through tools.exec_command. Synchronous code blocks the host; "
        "a non-yielding loop cannot be stopped independently. Imported modules and "
        "process state are shared.\n"
        "- `cwd` is the cell's workspace Path; use `cwd / \"file\"` for direct file "
        "access. Nested tools use that workspace. The host's working directory "
        "and sys.stdout are not changed. Plain `print()` writes to this cell; "
        "output from imported modules is not captured.\n"
        "- Optional first line: "
        '`# @exec: {"yield_time_ms": 30000, "max_output_tokens": 1000}`. '
        "The defaults are 30000 ms and 10000 tokens. A running script returns a "
        "cell_id; use wait for subsequent output.\n"
        "\n"
        "Global helpers:\n"
        "- `text(value)`: append text; non-string values use JSON encoding. "
        "Use `text()` or `print()` for explicit output.\n"
        "- `image(value, detail=...)`: append a base64 data-URL image, an object "
        "with image_url/detail, or an individual MCP image block. An explicit "
        "detail overrides the object's detail or codex/imageDetail metadata.\n"
        "- `audio(value)`: append a base64 data-URL audio item, an object with "
        "audio_url, or an individual MCP audio block.\n"
        "- `generated_image(result)`: append the generated image_url and optional "
        "output_hint. The URL must be a base64 data URL.\n"
        "- `store(key, value)` / `load(key)`: save and retrieve JSON-serializable "
        "values across exec cells in this session; a missing key returns None.\n"
        "- `notify(value)`: immediately submit an additional output for this exec "
        "call through the attached Runtime.\n"
        "- `set_timeout(callback, delay_ms=0)` / `clear_timeout(handle)`: schedule "
        "or cancel a callback; timers alone do not keep the script alive. Use "
        "`await asyncio.sleep(seconds)` to wait.\n"
        "- `await yield_control()`: return accumulated output to the model while "
        "the script continues running.\n"
        "- `exit()`: finish the current script successfully.\n"
        "- `ALL_TOOLS`: the available nested tools as qualified name/description records.\n"
        "\n"
        "Example:\n"
        "```python\n"
        "result = await tools.exec_command(cmd=\"pwd\")\n"
        "result\n"
        "```"
    )
    tool_type = "custom"
    format = {
        "type": "grammar",
        "syntax": "lark",
        "definition": EXEC_FREEFORM_GRAMMAR,
    }
    supports_parallel = False

    def __init__(self, manager: "CodeModeManager") -> "None":
        self._manager = manager
        self._runtime = None

    def spec(self):
        return replace(
            super().spec(),
            description=self.description + self._manager.describe_tools(),
        )

    def bind_runtime(self, runtime):
        self._manager.restart()
        self._manager.set_output_handler(runtime.receive_tool_output)
        if self._runtime is not runtime:
            runtime.add_close_handler(self._manager.close)
            self._runtime = runtime

    def shutdown(self):
        self._manager.shutdown()

    def set_state_listener(self, listener):
        self._manager.set_state_listener(listener)

    def background_work_snapshot(self):
        return self._manager.snapshot()

    def background_work_count(self, after_reply: "bool") -> "int":
        return self._manager.running_cell_count()

    async def run(self, context: "ToolContext", args: "JSONValue") -> "JSONValue":
        if not isinstance(args, str):
            raise ValueError("exec expects Python source text")
        return await self._manager.exec(args, context)
