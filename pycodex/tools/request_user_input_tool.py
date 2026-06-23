"""`request_user_input` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex `request_user_input` collaboration tool.

Expected behavior:
- In upstream Codex this tool is only actually usable in Plan mode.
- In the current `pycodex` default-mode path, runtime invocation returns the
  same fixed unavailable error string as upstream Codex.
- In Plan mode, the tool validates its question payload, forces `isOther=true`
  on every question, and returns a JSON-string `function_call_output` with
  `success=true`.
"""

import json

from ..collaboration import collaboration_mode_display_name
from ..protocol import JSONDict, JSONValue
from ..runtime_services import RequestUserInputManager
from .base_tool import BaseTool, StructuredToolOutput, ToolContext
import typing

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
                "first and suffix its label with \"(Recommended)\". Do not include "
                "an \"Other\" option in this list; the client will add a free-form "
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


def request_user_input_is_available(
    mode: 'str',
    default_mode_request_user_input: 'bool' = False,
) -> 'bool':
    normalized = mode.strip().lower()
    return normalized == "plan" or (
        default_mode_request_user_input and normalized == "default"
    )


def request_user_input_unavailable_message(
    mode: 'str',
    default_mode_request_user_input: 'bool' = False,
) -> 'typing.Union[str, None]':
    if request_user_input_is_available(mode, default_mode_request_user_input):
        return None
    return (
        "request_user_input is unavailable in "
        f"{collaboration_mode_display_name(mode)} mode"
    )


def request_user_input_tool_description(
    default_mode_request_user_input: 'bool' = False,
) -> 'str':
    if default_mode_request_user_input:
        allowed_modes = "Default or Plan mode"
    else:
        allowed_modes = "Plan mode"
    return (
        "Request user input for one to three short questions and wait for the "
        f"response. Set autoResolutionMs, from {MIN_AUTO_RESOLUTION_MS} to "
        f"{MAX_AUTO_RESOLUTION_MS} milliseconds, only when the question is "
        "useful but non-blocking and continuing with best judgment is "
        "acceptable if the user does not answer; omit it when explicit user "
        f"input is required. This tool is only available in {allowed_modes}."
    )


class RequestUserInputTool(BaseTool):
    name = "request_user_input"
    description = request_user_input_tool_description()
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
            }
        },
        "required": ["questions"],
        "additionalProperties": False,
    }
    supports_parallel = False

    def __init__(
        self,
        request_manager: 'RequestUserInputManager',
        default_mode_request_user_input: 'bool' = False,
    ) -> 'None':
        self._request_manager = request_manager
        self._default_mode_request_user_input = default_mode_request_user_input
        self.description = request_user_input_tool_description(
            default_mode_request_user_input
        )

    async def run(self, context: 'ToolContext', args: 'JSONDict') -> 'JSONValue':
        unavailable = request_user_input_unavailable_message(
            context.collaboration_mode,
            self._default_mode_request_user_input,
        )
        if unavailable is not None:
            return unavailable

        questions = args.get("questions")
        if not isinstance(questions, list):
            raise ValueError("questions must be a list")

        if any(
            not isinstance(question, dict)
            or not isinstance(question.get("options"), list)
            or not question["options"]
            for question in questions
        ):
            return "request_user_input requires non-empty options for every question"

        request_payload = {
            "questions": [
                {
                    **question,
                    "isOther": True,
                }
                for question in questions
            ]
        }
        auto_resolution_ms = args.get("autoResolutionMs")
        if auto_resolution_ms not in (None, ""):
            request_payload["autoResolutionMs"] = min(
                max(int(auto_resolution_ms), MIN_AUTO_RESOLUTION_MS),
                MAX_AUTO_RESOLUTION_MS,
            )
        response = await self._request_manager.request(request_payload)
        if response is None:
            return "request_user_input was cancelled before receiving a response"
        return StructuredToolOutput(
            json.dumps(response, ensure_ascii=False, separators=(",", ":")),
            success=True,
        )
