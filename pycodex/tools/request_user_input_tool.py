"""`request_user_input` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex `request_user_input` collaboration tool.

Expected behavior:
- Keep the tool declaration available for upstream request compatibility.
- Return the upstream Default-mode unavailable message without requesting input.
"""

from ..protocol import JSONDict, JSONValue
from .base_tool import BaseTool, ToolContext

MIN_AUTO_RESOLUTION_MS = 60_000
MAX_AUTO_RESOLUTION_MS = 240_000

REQUEST_USER_INPUT_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {
            "type": "string",
            "description": "Stable identifier for mapping answers (snake_case).",
        },
        "header": {
            "type": "string",
            "description": "Short header label shown in the UI (12 or fewer chars).",
        },
        "question": {
            "type": "string",
            "description": "Single-sentence prompt shown to the user.",
        },
        "options": {
            "type": "array",
            "description": (
                "Provide 2-3 mutually exclusive choices. Put the recommended option "
                'first and suffix its label with "(Recommended)". Do not include '
                'an "Other" option in this list; the client will add a free-form '
                "Other option automatically."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "User-facing label (1-5 words).",
                    },
                    "description": {
                        "type": "string",
                        "description": "One short sentence explaining impact/tradeoff if selected.",
                    },
                },
                "required": ["label", "description"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["id", "header", "question", "options"],
    "additionalProperties": False,
}


class RequestUserInputTool(BaseTool):
    name = "request_user_input"
    description = (
        "Request user input for one to three short questions and wait for the "
        f"response. Set autoResolutionMs, from {MIN_AUTO_RESOLUTION_MS} to "
        f"{MAX_AUTO_RESOLUTION_MS} milliseconds, only when the question is "
        "useful but non-blocking and continuing with best judgment is "
        "acceptable if the user does not answer; omit it when explicit user "
        "input is required. This tool is unavailable in Default mode."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "description": "Questions to show the user. Prefer 1 and do not exceed 3",
                "items": REQUEST_USER_INPUT_QUESTION_SCHEMA,
            },
            "autoResolutionMs": {
                "type": "number",
                "description": (
                    "Optional auto-resolution window in milliseconds, from "
                    f"{MIN_AUTO_RESOLUTION_MS} to {MAX_AUTO_RESOLUTION_MS}. "
                    "Include this only when the question is useful but "
                    "non-blocking and continuing with best judgment is "
                    "acceptable if the user does not answer; omit it when "
                    "explicit user input is required before continuing. Use "
                    f"{MIN_AUTO_RESOLUTION_MS} for lightly helpful context and "
                    f"up to {MAX_AUTO_RESOLUTION_MS} when the answer would "
                    "materially unblock better work."
                ),
            },
        },
        "required": ["questions"],
        "additionalProperties": False,
    }
    supports_parallel = False

    async def run(self, context: "ToolContext", args: "JSONDict") -> "JSONValue":
        del context, args
        return "request_user_input is unavailable in Default mode"
