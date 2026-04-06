
from copy import deepcopy
import json
import typing


class CustomToolAdapterError(ValueError):
    pass


# Mirrors the chat-completions apply_patch function tool text from
# `codex-rs/core/src/tools/handlers/apply_patch.rs`.
APPLY_PATCH_NAME = "apply_patch"
APPLY_PATCH_CHAT_INPUT_DESCRIPTION = "The entire contents of the apply_patch command"
APPLY_PATCH_CHAT_DESCRIPTION = """Use the `apply_patch` tool to edit files.
Your patch language is a stripped-down, file-oriented diff format designed to be easy to parse and safe to apply. You can think of it as a high-level envelope:

*** Begin Patch
[ one or more file sections ]
*** End Patch

Within that envelope, you get a sequence of file operations.
You MUST include a header to specify the action you are taking.
Each operation starts with one of three headers:

*** Add File: <path> - create a new file. Every following line is a + line (the initial contents).
*** Delete File: <path> - remove an existing file. Nothing follows.
*** Update File: <path> - patch an existing file in place (optionally with a rename).

May be immediately followed by *** Move to: <new path> if you want to rename the file.
Then one or more "hunks", each introduced by @@ (optionally followed by a hunk header).
Within a hunk each line starts with:

For instructions on [context_before] and [context_after]:
- By default, show 3 lines of code immediately above and 3 lines immediately below each change. If a change is within 3 lines of a previous change, do NOT duplicate the first change's [context_after] lines in the second change's [context_before] lines.
- If 3 lines of context is insufficient to uniquely identify the snippet of code within the file, use the @@ operator to indicate the class or function to which the snippet belongs. For instance, we might have:
@@ class BaseClass
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

- If a code block is repeated so many times in a class or function such that even a single `@@` statement and 3 lines of context cannot uniquely identify the snippet of code, you can use multiple `@@` statements to jump to the right context. For instance:

@@ class BaseClass
@@ \t def method():
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

The full grammar definition is below:
Patch := Begin { FileOp } End
Begin := "*** Begin Patch" NEWLINE
End := "*** End Patch" NEWLINE
FileOp := AddFile | DeleteFile | UpdateFile
AddFile := "*** Add File: " path NEWLINE { "+" line NEWLINE }
DeleteFile := "*** Delete File: " path NEWLINE
UpdateFile := "*** Update File: " path NEWLINE [ MoveTo ] { Hunk }
MoveTo := "*** Move to: " newPath NEWLINE
Hunk := "@@" [ header ] NEWLINE { HunkLine } [ "*** End of File" NEWLINE ]
HunkLine := (" " | "-" | "+") text NEWLINE

A full patch can combine several operations:

*** Begin Patch
*** Add File: hello.txt
+Hello world
*** Update File: src/app.py
*** Move to: src/main.py
@@ def greet():
-print("Hi")
+print("Hello, world!")
*** Delete File: obsolete.txt
*** End Patch

It is important to remember:

- You must include a header with your intended action (Add/Delete/Update)
- You must prefix new lines with `+` even when creating a new file
- File references can only be relative, NEVER ABSOLUTE.
"""


def collect_custom_tool_names(raw_tools: 'object') -> 'typing.Set[str]':
    names: 'typing.Set[str]' = set()
    if not isinstance(raw_tools, list):
        return names
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict) or raw_tool.get("type") != "custom":
            continue
        name = _tool_name(raw_tool)
        if name:
            names.add(name)
    return names


def build_tool_definition(raw_tool: 'typing.Dict[str, object]') -> 'typing.Dict[str, object]':
    name = _required_tool_name(raw_tool)
    description = _build_description(raw_tool)
    input_description = (
        APPLY_PATCH_CHAT_INPUT_DESCRIPTION
        if name == APPLY_PATCH_NAME
        else "Raw tool input. Pass the freeform payload verbatim as a single string."
    )
    return {
        "type": "function",
        "name": name,
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": input_description,
                    }
                },
                "required": ["input"],
                "additionalProperties": False,
            },
            "strict": False,
        },
    }


def build_tool_call(raw_item: 'typing.Dict[str, object]') -> 'typing.Dict[str, object]':
    name = _required_item_name(raw_item)
    return {
        "id": str(raw_item.get("call_id", "")).strip() or name,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(
                {"input": str(raw_item.get("input", "") or "")},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    }


def build_output_item(tool_call: 'typing.Dict[str, object]', index: 'int') -> 'typing.Dict[str, object]':
    function = tool_call.get("function") or {}
    if not isinstance(function, dict):
        raise CustomToolAdapterError(
            "outcomming custom tool call is missing function payload"
        )
    name = str(function.get("name", "")).strip()
    if not name:
        raise CustomToolAdapterError(
            "outcomming custom tool call is missing `name`"
        )
    return {
        "type": "custom_tool_call",
        "call_id": str(tool_call.get("id", "")).strip() or f"call_{index}",
        "name": name,
        "input": extract_input_text(function.get("arguments")),
    }


def extract_input_text(raw_arguments: 'object') -> 'str':
    if isinstance(raw_arguments, dict):
        parsed = deepcopy(raw_arguments)
    else:
        parsed = None

    raw_text = raw_arguments if isinstance(raw_arguments, str) else None
    if parsed is None and raw_text is not None:
        try:
            parsed = json.loads(raw_text or "{}")
        except json.JSONDecodeError:
            return raw_text

    if isinstance(parsed, dict) and "input" in parsed:
        value = parsed.get("input")
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    if raw_text is not None:
        return raw_text
    if parsed is not None:
        return json.dumps(parsed, ensure_ascii=False)
    return str(raw_arguments or "")


def _build_description(raw_tool: 'typing.Dict[str, object]') -> 'str':
    name = _tool_name(raw_tool)
    if name == APPLY_PATCH_NAME:
        return APPLY_PATCH_CHAT_DESCRIPTION

    description = str(raw_tool.get("description", "") or "").strip()
    parts = [description] if description else []
    parts.append(
        "Chat-completions compatibility: provide the raw tool payload in the "
        "`input` string field."
    )

    raw_format = raw_tool.get("format")
    if isinstance(raw_format, dict):
        format_lines: 'typing.List[str]' = []
        format_type = str(raw_format.get("type", "")).strip()
        syntax = str(raw_format.get("syntax", "")).strip()
        definition = str(raw_format.get("definition", "") or "").strip()
        if format_type:
            format_lines.append(f"Input format type: {format_type}")
        if syntax:
            format_lines.append(f"Input format syntax: {syntax}")
        if definition:
            format_lines.append("Input format definition:")
            format_lines.append(definition)
        if format_lines:
            parts.append("\n".join(format_lines))

    return "\n\n".join(parts)


def _tool_name(raw_tool: 'typing.Dict[str, object]') -> 'str':
    return str(raw_tool.get("name", "")).strip()


def _required_tool_name(raw_tool: 'typing.Dict[str, object]') -> 'str':
    name = _tool_name(raw_tool)
    if not name:
        raise CustomToolAdapterError("custom tool definition is missing `name`")
    return name


def _required_item_name(raw_item: 'typing.Dict[str, object]') -> 'str':
    name = str(raw_item.get("name", "")).strip()
    if not name:
        raise CustomToolAdapterError("custom tool call is missing `name`")
    return name
