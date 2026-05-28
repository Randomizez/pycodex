"""Optional tool that executes code in the current IPython kernel."""

from ..utils.toolcall_visualize import colorize_tool_message, tool_summary
from .base_tool import BaseTool, ToolContext


class IPythonTool(BaseTool):
    name = "ipython"
    description = (
        "Execute Python code in the current IPython kernel namespace. "
        "Use this to inspect live Python variables in the user's IPython session."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute in the current IPython session.",
            },
        },
        "required": ["code"],
        "additionalProperties": False,
    }
    supports_parallel = False

    def __init__(self, shell, name="ipython"):
        self._shell = shell
        self.name = name

    async def run(self, context: 'ToolContext', args):
        del context
        if not isinstance(args, dict):
            return {"error": "arguments must be an object"}

        code = args.get("code", "")
        if not isinstance(code, str) or not code.strip():
            return {"error": "`code` must be a non-empty string"}

        from IPython.display import Code, display
        from IPython.utils.capture import capture_output

        display(Code(code, language="python"))
        with capture_output() as captured:
            result = self._shell.run_cell(code, store_history=False)
        captured.show()

        output = {
            "stdout": captured.stdout,
            "stderr": captured.stderr,
        }

        display_outputs = []
        for item in captured.outputs:
            data = getattr(item, "data", None)
            if isinstance(data, dict):
                display_outputs.append(
                    data.get("text/plain")
                    or data.get("text/html")
                    or repr(data)
                )
            else:
                display_outputs.append(repr(item))
        if display_outputs:
            output["display"] = display_outputs

        if getattr(result, "result", None) is not None:
            output["result"] = repr(result.result)

        error = result.error_before_exec or result.error_in_exec
        if error is not None:
            output["error"] = f"{type(error).__name__}: {error}"

        return output


def _install_agent_shortcut(shell):
    existing = shell.user_ns.get("_pycodex_agent_shortcut_transform")
    if existing in shell.input_transformers_cleanup:
        return existing

    def transform(lines):
        if len(lines) != 1:
            return lines

        line = lines[0]
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if not stripped.startswith("@"):
            return lines

        body = stripped[1:].rstrip("\n")
        parts = body.split(None, 1)
        if not parts:
            return lines

        agent_name = parts[0]
        if not agent_name.isidentifier():
            return lines

        agent = shell.user_ns.get(agent_name)
        if agent is None or not callable(getattr(agent, "run_turn", None)):
            return lines

        prompt = "" if len(parts) == 1 else parts[1]
        if not prompt.strip():
            return [f"{indent}print('Usage: @{agent_name} <prompt>')\n"]

        return [
            f"{indent}print((await {agent_name}.run_turn([{prompt!r}]))"
            f".output_text)\n"
        ]

    shell.input_transformers_cleanup.append(transform)
    shell.user_ns["_pycodex_agent_shortcut_transform"] = transform
    return transform


def attach_ipython_event_printer(agent, color=True):
    def handle_event(event):
        if event.kind == "tool_started":
            tool_name = str(event.payload.get("tool_name", "")).strip()
            if tool_name:
                message = f"[{tool_name}] running"
            else:
                message = "[tool] running"
            print(colorize_tool_message(message, color, tool_name), flush=True)
            return

        if event.kind == "tool_completed":
            tool_name = str(event.payload.get("tool_name", "")).strip()
            message = tool_summary(event.payload)
            for line in message.splitlines() or [""]:
                print(colorize_tool_message(line, color, tool_name), flush=True)

    agent.set_event_handler(handle_event)
    return handle_event


def attach_ipython_tool(
    agent, name="ipython", shortcut=True, print_tool_events=True, color=True
):
    from IPython import get_ipython

    shell = get_ipython()
    if shell is None:
        raise RuntimeError("not running inside IPython")

    tool = IPythonTool(shell, name=name)
    agent._tool_registry.register(tool)
    if shortcut:
        _install_agent_shortcut(shell)
    if print_tool_events:
        attach_ipython_event_printer(agent, color=color)
    return tool
