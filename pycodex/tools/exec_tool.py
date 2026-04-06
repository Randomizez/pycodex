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

from __future__ import annotations

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
        "Runs raw JavaScript in an isolated context. Send raw JavaScript "
        "source text, not JSON, quoted strings, or markdown code fences."
    )
    tool_type = "custom"
    format = {
        "type": "grammar",
        "syntax": "lark",
        "definition": EXEC_FREEFORM_GRAMMAR,
    }
    supports_parallel = False

    def __init__(self, manager: CodeModeManager) -> None:
        self._manager = manager

    async def run(self, context: ToolContext, args: JSONValue) -> JSONValue:
        return await self._manager.exec(str(args), context)
