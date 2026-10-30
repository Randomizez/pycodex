"""Render tool schemas as Python function signatures for code mode."""

import inspect
import keyword


def python_tool_signature(tool):
    parameters = []
    if tool["tool_type"] == "custom":
        parameters.append(inspect.Parameter(
            "input", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation="str",
        ))
    else:
        schema = {} if tool["parameters"] is None else tool["parameters"]
        properties = schema.get("properties", {})
        required = set(schema.get("required", ()))
        for name in sorted(properties, key=lambda key: (key not in required, key)):
            if not name.isidentifier() or keyword.iskeyword(name):
                raise ValueError("invalid Python tool parameter name: " + name)
            parameter = properties[name]
            default = inspect.Parameter.empty if name in required else (
                parameter.get("default", ...) if isinstance(parameter, dict) else ...
            )
            parameters.append(inspect.Parameter(
                name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default, annotation=_annotation(parameter),
            ))
    return inspect.Signature(parameters, return_annotation=_annotation(tool["output_schema"]))


def python_tool_declaration(tool):
    signature = python_tool_signature(tool)
    result = signature.return_annotation
    if tool["tool_type"] == "custom":
        lines = ["async def {0}(input: str) -> {1}:".format(tool["name"], result)]
    else:
        schema = {} if tool["parameters"] is None else tool["parameters"]
        properties = schema.get("properties", {})
        if not properties and not _comments(tool["output_schema"]):
            return "async def {0}() -> {1}: ...".format(tool["name"], result)
        lines = ["async def " + tool["name"] + "("]
        for parameter in signature.parameters.values():
            lines.extend("  # " + line for line in _comments(properties[parameter.name]))
            default = ""
            if parameter.default is not inspect.Parameter.empty:
                default = " = " + (
                    "..." if parameter.default is ... else repr(parameter.default)
                )
            lines.append("  {0}: {1}{2},".format(
                parameter.name, parameter.annotation, default,
            ))
        lines.append(") -> " + result + ":")
    result_comments = _comments(tool["output_schema"])
    if not result_comments:
        lines[-1] += " ..."
        return "\n".join(lines)
    lines.append("  # Returns:")
    lines.extend("  # " + line for line in result_comments)
    lines.append("  ...")
    return "\n".join(lines)


def _annotation(schema):
    if schema is None or schema is True or schema == {}:
        return "Any"
    if schema is False:
        return "NoReturn"
    if "$ref" in schema:
        raise ValueError("Python tool declarations require resolved schema references")
    if "const" in schema:
        return "Literal[{0!r}]".format(schema["const"])
    if "enum" in schema:
        return "Literal[" + ", ".join(repr(value) for value in schema["enum"]) + "]"
    for union in ("anyOf", "oneOf"):
        if union in schema:
            return _union([_annotation(item) for item in schema[union]])
    kind = schema.get("type")
    if isinstance(kind, list):
        return _union([_annotation(dict(schema, type=item)) for item in kind])
    if kind == "object" or "properties" in schema:
        additional = schema.get("additionalProperties")
        if not schema.get("properties") and isinstance(additional, dict):
            return "Dict[str, " + _annotation(additional) + "]"
        return "dict"
    if kind == "array":
        return "List[" + _annotation(schema.get("items")) + "]"
    types = {
        None: "Any", "string": "str", "integer": "int",
        "number": "Union[int, float]", "boolean": "bool", "null": "None",
    }
    return types[kind]


def _comments(schema):
    if schema is None or isinstance(schema, bool):
        return []
    lines = schema.get("description", "").splitlines()
    for key in (
        "default", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
        "minLength", "maxLength", "pattern", "format", "minItems", "maxItems",
        "uniqueItems",
    ):
        if key in schema:
            lines.append("{0} = {1!r}".format(key, schema[key]))
    required = set(schema.get("required", ()))
    for name, value in sorted(schema.get("properties", {}).items()):
        lines.append("{0}: {1} ({2})".format(
            name, _annotation(value), "required" if name in required else "optional",
        ))
        lines.extend("    " + line for line in _comments(value))
    if schema.get("type") == "array":
        items = _comments(schema.get("items"))
        if items:
            lines.append("Items:")
            lines.extend("    " + line for line in items)
    return lines


def _union(annotations):
    annotations = list(dict.fromkeys(annotations))
    if len(annotations) == 1:
        return annotations[0]
    return "Union[" + ", ".join(annotations) + "]"
