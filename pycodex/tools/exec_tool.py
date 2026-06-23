"""`exec` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex code-mode `exec` tool.

Expected behavior:
- Accept raw JavaScript source text as a freeform/custom tool payload.
- Run the script inside a background exec cell with helper functions and nested
  tool access.
- Return either a completed result or a running `cell_id` status that can be
  resumed via `wait`.
"""

from ..protocol import JSONValue
from .base_tool import BaseTool, ToolContext
from .code_mode_manager import CodeModeManager

EXEC_FREEFORM_GRAMMAR = r"""start: pragma_source | plain_source
pragma_source: PRAGMA_LINE NEWLINE SOURCE
plain_source: SOURCE

PRAGMA_LINE: /[ \t]*\/\/ @exec:[^\r\n]*/
NEWLINE: /\r?\n/
SOURCE: /[\s\S]+/
"""


class ExecTool(BaseTool):
    name = "exec"
    description = (
        "Run JavaScript code to orchestrate/compose tool calls\n"
        "- Evaluates the provided JavaScript code in a fresh V8 isolate as an "
        "async module.\n"
        "- All nested tools are available on the global `tools` object, for "
        "example `await tools.exec_command(...)`.\n"
        "- Nested tool methods take either a string or an object as their input "
        "argument.\n"
        "- Runs raw JavaScript -- no Node, no file system, no network access, "
        "no console.\n"
        "- Accepts raw JavaScript source text, not JSON, quoted strings, or "
        "markdown code fences.\n"
        "- You may optionally start the tool input with a first-line pragma "
        "like `// @exec: {\"yield_time_ms\": 10000, "
        "\"max_output_tokens\": 1000}`.\n"
        "- `yield_time_ms` asks `exec` to yield early if the script is still "
        "running. Defaults to 10000 ms.\n"
        "- `max_output_tokens` sets the token budget for direct `exec` results. "
        "Defaults to 10000 tokens."
    )
    tool_type = "custom"
    format = {
        "type": "grammar",
        "syntax": "lark",
        "definition": EXEC_FREEFORM_GRAMMAR,
    }
    supports_parallel = False

    def __init__(self, manager: 'CodeModeManager') -> 'None':
        self._manager = manager

    async def run(self, context: 'ToolContext', args: 'JSONValue') -> 'JSONValue':
        return await self._manager.exec(str(args), context)
