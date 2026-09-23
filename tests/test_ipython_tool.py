import sys
import types

from pycodex.events import AssistantDeltaEvent, ToolCompletedEvent, ToolStartedEvent
from pycodex.protocol import ToolCall, ToolResult
from pycodex.tools.ipython_tool import IPythonTool, attach_ipython_event_printer


class FakeAgent:
    def __init__(self):
        self.event_handler = None


def test_ipython_event_printer_prints_completed_tool_events(capsys) -> "None":
    agent = FakeAgent()

    handler = attach_ipython_event_printer(agent, color=False)

    assert agent.event_handler is handler

    call = ToolCall("call_1", "exec_command", {"cmd": "pwd"})
    handler(AssistantDeltaEvent("hello", "turn_1"))
    handler(ToolStartedEvent("turn_1", call))
    handler(
        ToolCompletedEvent(
            "turn_1",
            call,
            ToolResult(
                "call_1", "exec_command", "Exit code: 0\nOutput:\n/data/pycodex\n"
            ),
        )
    )

    assert capsys.readouterr().out == "[exec_command] pwd -> /data/pycodex\n"


async def test_ipython_tool_prints_io_without_storing_history(monkeypatch) -> "None":
    displayed = []

    class FakeCode:
        def __init__(self, code, language=None):
            self.code = code
            self.language = language

    def fake_display(value):
        displayed.append(value)

    display_module = types.ModuleType("IPython.display")
    display_module.Code = FakeCode
    display_module.display = fake_display

    class FakeCaptured:
        def __init__(self):
            self.stdout = "printed\n"
            self.stderr = ""
            self.outputs = []
            self.show_called = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def show(self):
            self.show_called = True

    captured = FakeCaptured()
    capture_module = types.ModuleType("IPython.utils.capture")
    capture_module.capture_output = lambda: captured

    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    monkeypatch.setitem(sys.modules, "IPython.utils.capture", capture_module)

    class FakeResult:
        result = 3
        error_before_exec = None
        error_in_exec = None

    class FakeShell:
        def __init__(self):
            self.calls = []

        def run_cell(self, code, store_history=False):
            self.calls.append((code, store_history))
            return FakeResult()

    shell = FakeShell()

    output = await IPythonTool(shell).run(None, {"code": "print('x')\n1 + 2"})

    assert shell.calls == [("print('x')\n1 + 2", False)]
    assert isinstance(displayed[0], FakeCode)
    assert displayed[0].code == "print('x')\n1 + 2"
    assert displayed[0].language == "python"
    assert captured.show_called is True
    assert output == {"stdout": "printed\n", "stderr": "", "result": "3"}
