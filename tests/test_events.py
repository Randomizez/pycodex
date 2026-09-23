import json
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from pycodex.events import (
    AssistantDeltaEvent,
    AutoCompactCompletedEvent,
    AutoCompactFailedEvent,
    AutoCompactStartedEvent,
    CommandCompletedEvent,
    CommandFailedEvent,
    CompactCompletedEvent,
    CompactFailedEvent,
    CompactStartedEvent,
    Event,
    EventDisplay,
    InputQueuedEvent,
    InputRequestedEvent,
    InputResolvedEvent,
    ModelCalledEvent,
    ModelCompletedEvent,
    SessionClosedEvent,
    SessionStateEvent,
    StreamErrorEvent,
    TerminalEvent,
    TokenCountEvent,
    ToolCalledEvent,
    ToolCompletedEvent,
    ToolStartedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
    TurnInterruptedEvent,
    TurnStartedEvent,
)
from pycodex.protocol import ToolCall, ToolResult, TurnResult
from pycodex.utils.event_helpers import (
    colorize_cli_message,
    colorize_tool_message,
    format_command_result,
    format_error,
    render_result,
)


def make_display(color=False):
    operations = []
    display = EventDisplay(
        lambda text: operations.append(("log", text)),
        lambda text: operations.append(("status", text)),
        lambda text: operations.append(("prompt", text)),
        color,
    )
    return display, operations


def test_event_fields_are_explicit_and_frozen():
    event = TurnStartedEvent("turn", ("first", "second"))
    assert isinstance(event, Event)
    assert event.kind == "turn_started"
    assert event.user_texts == ("first", "second")
    assert not hasattr(event, "payload")
    assert not hasattr(event, "user_text")
    assert event.visualize() == "first\nsecond"
    with pytest.raises(TypeError):
        TurnStartedEvent("turn")
    with pytest.raises(TypeError):
        TurnStartedEvent("turn", ("first",), extra="unknown")
    with pytest.raises(FrozenInstanceError):
        event.user_texts = ("changed",)


@pytest.mark.parametrize(
    "event,text",
    [
        (TurnInterruptedEvent("turn", 1, "partial", 0), "partial"),
        (TurnCompletedEvent("turn", 1, None, 0), ""),
        (TurnCompletedEvent("turn", 1, "answer\nnext", 0), "answer\nnext"),
        (TurnFailedEvent("turn", 1, "failed", "ValueError", 0), "failed"),
        (AssistantDeltaEvent("delta"), "delta"),
        (TokenCountEvent({"total_tokens": 12}), ""),
        (StreamErrorEvent("Reconnecting", 1, 5, 0.2, "lost"), "Reconnecting"),
        (
            CommandFailedEvent("submission", "model", "unknown model", "cli"),
            "unknown model",
        ),
        (SessionClosedEvent(), ""),
    ],
)
def test_event_text_is_plain_and_stateless(event, text):
    assert event.visualize() == text
    assert event.visualize() == text
    assert "\x1b" not in event.visualize()


@pytest.mark.parametrize(
    "details,text",
    [
        ({}, "[web_search] browsing"),
        ({"action_type": "search"}, "[web_search] searched"),
        (
            {"action_type": "search", "query": "  question "},
            "[web_search] searched: question",
        ),
        (
            {"action_type": "search", "query": " ", "queries": (" first ", "second")},
            "[web_search] searched: first",
        ),
        (
            {"action_type": "open_page", "url": " example "},
            "[web_search] opened: example",
        ),
        ({"action_type": "open_page"}, "[web_search] opened"),
        (
            {"action_type": "find_in_page", "pattern": " match ", "url": " example "},
            "[web_search] found: match @ example",
        ),
        (
            {"action_type": "find_in_page", "pattern": "match"},
            "[web_search] found: match",
        ),
        ({"action_type": "find_in_page"}, "[web_search] found in page"),
    ],
)
def test_provider_tool_text(details, text):
    assert ToolCalledEvent("call", "web_search", **details).visualize() == text
    assert ToolCalledEvent("call", "other", **details).visualize() == ""


@pytest.mark.parametrize(
    "name,arguments,output,summary",
    [
        (
            "exec_command",
            {"cmd": "pwd"},
            "Exit code: 0\nOutput:\n/work\nignored",
            "pwd -> /work",
        ),
        (
            "exec_command",
            {"cmd": "work"},
            "Process running with session ID 12\nOutput:\nstarted",
            "work -> session_id=12",
        ),
        (
            "write_stdin",
            {"session_id": "12"},
            "Output:\ndone",
            "poll session 12 -> done",
        ),
        (
            "write_stdin",
            {"session_id": 12, "chars": "hello\n"},
            "",
            "session 12 <- hello",
        ),
        ("shell_command", {"command": "pwd"}, "ok", "pwd -> ok"),
        ("shell", {"command": ["echo", "two words"]}, "ok", "echo 'two words' -> ok"),
        ("exec", {"code": "text(1)"}, {"value": 1}, '{"value":1}'),
        ("wait", {"id": "run"}, ["done"], '["done"]'),
        ("read_file", {"file_path": "/work/file"}, "line\nnext", "/work/file -> line"),
        ("list_dir", {"dir_path": "/work"}, "file", "/work -> file"),
        (
            "grep_files",
            {"pattern": "match", "path": "/work"},
            "file",
            "match @ /work -> file",
        ),
        (
            "view_image",
            {"path": "/work/image"},
            [{}, {}],
            "/work/image -> 2 image item(s)",
        ),
        ("update_plan", {"plan": []}, {}, "0 steps"),
        ("update_plan", {}, {"plan": [{}, {}]}, "2 steps"),
        (
            "spawn_agent",
            {},
            {"agent_id": "1234567890abcdef", "nickname": "Ada"},
            "Ada (12345678...cdef)",
        ),
        (
            "send_input",
            {"id": "1234567890abcdef", "message": "go"},
            {"submission_id": "abcdefgh12345678"},
            "12345678...cdef <- go -> queued abcdefgh...5678",
        ),
        ("wait_agent", {}, {"timed_out": True}, "timed out"),
        (
            "wait_agent",
            {},
            {"status": {"agent": {"completed": "done"}}},
            "agent=completed: done",
        ),
        (
            "resume_agent",
            {"id": "agent"},
            {"status": "pending_init"},
            "agent -> pending_init",
        ),
        (
            "close_agent",
            {"id": "agent"},
            {"status": {"errored": "failed"}},
            "agent -> errored: failed",
        ),
        (
            "other",
            "freeform",
            "Wall time: 1\nCommand: run\nExit code: 0\nhello",
            "hello",
        ),
    ],
)
@pytest.mark.parametrize("is_error", [False, True])
def test_tool_text(name, arguments, output, summary, is_error):
    call = ToolCall("call", name, arguments)
    result = ToolResult("call", name, output, is_error=is_error)
    event = ToolCompletedEvent("turn", call, result)
    if is_error:
        expected = f"[error] {name} failed: {summary}"
    else:
        prefix = f"[{name}] spawned" if name == "spawn_agent" else f"[{name}]"
        expected = prefix + " " + summary
    assert event.visualize() == expected
    assert ToolStartedEvent("turn", call).visualize() == f"calling {name}({arguments})"


@pytest.mark.parametrize(
    "name",
    [
        "exec_command",
        "write_stdin",
        "shell_command",
        "shell",
        "exec",
        "wait",
        "read_file",
        "list_dir",
        "grep_files",
        "view_image",
        "update_plan",
        "spawn_agent",
        "send_input",
        "wait_agent",
        "resume_agent",
        "close_agent",
        "other",
    ],
)
def test_tool_empty_and_error_text(name):
    call = ToolCall("call", name, {})
    result = ToolResult("call", name, "")
    event = ToolCompletedEvent("turn", call, result)
    expected = f"[{name}]"
    if name == "spawn_agent":
        expected += " spawned"
    elif name == "update_plan":
        expected += " Plan updated"
    assert event.visualize() == expected
    assert (
        replace(event, result=replace(result, is_error=True)).visualize()
        == f"[error] {name} failed"
    )


@pytest.mark.parametrize("session_id", [7, "not-a-session"])
def test_write_stdin_summary_preserves_failed_tool_arguments(session_id):
    call = ToolCall("call_stdin", "write_stdin", {"session_id": session_id})
    result = ToolResult(
        "call_stdin", "write_stdin", {"error": "invalid session"}, is_error=True
    )
    summary = ToolCompletedEvent("turn", call, result).visualize()
    assert summary.startswith(f"[error] write_stdin failed: poll session {session_id}")
    assert "invalid session" in summary


def test_exec_preview_preserves_python_heredoc_and_truncates_other_commands():
    command = "python3 - <<'PY'\nprint('" + "x" * 220 + "')\nPY"
    call = ToolCall("call", "exec_command", {"cmd": command + "\necho after"})
    result = ToolResult("call", "exec_command", "Output:\nhello")
    event = ToolCompletedEvent("turn", call, result)
    assert event.visualize() == "[exec_command] " + command + " -> hello"
    call = replace(call, arguments={"cmd": "x" * 210})
    assert (
        replace(event, call=call).visualize()
        == "[exec_command] " + "x" * 197 + "... -> hello"
    )


def test_plan_text_contains_progress_and_steps_without_error_details_duplication():
    plan = [
        {"step": "first", "status": "completed"},
        {"step": "second", "status": "in_progress"},
        {"step": "third", "status": "pending"},
    ]
    event = ToolCompletedEvent(
        "turn",
        ToolCall("call", "update_plan", {"plan": plan}),
        ToolResult("call", "update_plan", {}),
    )
    assert (
        event.visualize()
        == "[update_plan] Working on 2/3\n  [x] first\n  [>] second\n  [ ] third"
    )
    assert (
        replace(event, result=replace(event.result, is_error=True)).visualize()
        == "[error] update_plan failed: Working on 2/3"
    )
    call = replace(
        event.call, arguments={"plan": [{"step": "done", "status": "completed"}]}
    )
    assert replace(event, call=call).visualize() == "[update_plan] Done 1/1\n  [x] done"
    call = replace(
        event.call,
        arguments={"plan": [None, {"step": " unknown ", "status": "unknown"}, {}]},
    )
    assert (
        replace(event, call=call).visualize()
        == "[update_plan] Planned 0/3\n  [ ] unknown"
    )


@pytest.mark.parametrize(
    "original,retained,pruned,text",
    [
        (1, 0, 0, "compact(1 item) -> 0 items + [summary]"),
        (
            5,
            1,
            1,
            "compact(5 items) -> 1 item + [summary] (dropped 1 old tool response)",
        ),
        (
            8,
            2,
            2,
            "compact(8 items) -> 2 items + [summary] (dropped 2 old tool responses)",
        ),
    ],
)
def test_compact_summary_is_derived_for_events_and_commands(
    original, retained, pruned, text
):
    event = CompactCompletedEvent(
        "turn", "manual", None, None, original, retained, pruned, 0
    )
    automatic = AutoCompactCompletedEvent(
        "turn", "pre_turn", 100, 90, original, retained, pruned
    )
    assert event.summary == event.visualize() == text
    assert automatic.summary == text
    assert automatic.visualize() == "[status] " + text
    assert isinstance(event, TerminalEvent)
    assert not isinstance(automatic, TerminalEvent)
    assert isinstance(TurnInterruptedEvent("turn", 1, None, 0), TerminalEvent)
    assert "summary" not in event.__dict__
    assert replace(event, original_item_count=original + 1).summary != event.summary
    assert (
        format_command_result(
            {
                "kind": "compacted",
                "original_item_count": original,
                "retained_item_count": retained,
                "pruned_tool_results": pruned,
            }
        )
        == text
    )


def test_compact_progress_and_failures():
    assert (
        CompactStartedEvent("turn", "manual", None, None).visualize()
        == "Compacting conversation history..."
    )
    assert (
        CompactFailedEvent(
            "turn", "manual", None, None, "failed", "ValueError", 0
        ).visualize()
        == "failed"
    )
    assert (
        AutoCompactStartedEvent("turn", "pre_turn", 100, 90).visualize()
        == "[status] auto-compact: 100/90 tokens"
    )
    assert (
        AutoCompactStartedEvent("turn", "pre_turn", None, 90).visualize()
        == "[status] auto-compact"
    )
    failure = AutoCompactFailedEvent(
        "turn", "pre_turn", None, None, " failed ", "ValueError"
    )
    assert failure.visualize() == "[error] auto-compact failed: failed"
    assert replace(failure, error=" ").visualize() == "[error] auto-compact failed"


@pytest.mark.parametrize(
    "result,text",
    [
        (
            {"kind": "help", "commands": ["help", "exit"]},
            "Extra commands: /help, /exit",
        ),
        ({"kind": "title", "title": ""}, "Session: untitled"),
        ({"kind": "title_changed", "title": "demo"}, "Session: demo"),
        (
            {"kind": "models", "model": "current", "models": ["current", "next"]},
            "Current model: current\nAvailable models: current, next",
        ),
        ({"kind": "model_changed", "model": "next"}, "Switched model to next."),
        ({"kind": "sessions", "sessions": []}, "No resumable sessions found."),
        (
            {
                "kind": "sessions",
                "sessions": [{"preview": "first"}, {"preview": "second"}],
            },
            "Available sessions:\n[1] first\n[2] second",
        ),
        ({"kind": "history", "state": {"title": "", "history": []}}, "No history yet."),
        (
            {"kind": "resumed", "state": {"title": "demo", "history": []}},
            "Resumed session: demo\nNo history yet.",
        ),
        (
            {
                "kind": "history",
                "state": {
                    "title": "demo",
                    "history": [("问", "答\nnext"), ("pending", None)],
                },
            },
            "Session: demo\n[1]U> 问\n[1]A> 答\nnext\n[2]U> pending",
        ),
        ({"kind": "compact_empty"}, "Nothing to compact."),
        ({"kind": "forked", "session_id": "fork"}, "Forked session: fork"),
        ({"kind": "closed"}, ""),
        ({"kind": "linked", "lines": ["first", "second"]}, "first\nsecond"),
    ],
)
def test_command_text(result, text):
    assert (
        CommandCompletedEvent("submission", "command", result, "cli").visualize()
        == text
    )
    assert format_command_result(result) == text


def test_input_request_text():
    event = InputRequestedEvent(
        "request",
        "questions",
        False,
        question={
            "header": "Choice",
            "question": "Which?",
            "options": [{"label": "First", "description": "Use first"}],
        },
    )
    assert event.visualize() == (
        "[request_user_input] waiting for user response\n[Choice] Which?\n"
        "  1. First - Use first\n  0. Other"
    )
    assert (
        replace(event, other=True).visualize() == "Enter your answer (blank to cancel):"
    )
    event = InputRequestedEvent(
        "request",
        "permissions",
        False,
        permissions={
            "reason": "network required",
            "permissions": {"network": True},
        },
    )
    assert event.visualize() == (
        "[request_permissions] user approval required\nReason: network required\n"
        'Requested permissions:\n{\n  "network": true\n}\n'
        "Choose: [n] deny / [t] grant for turn / [s] grant for session"
    )


@pytest.mark.parametrize(
    "message,tool_name,color",
    [
        ("[error] exec_command failed", None, "\x1b[31m"),
        ("  [x] done", None, "\x1b[36m"),
        ("[exec_command] run -> session_id=12", None, "\x1b[2m"),
        ("print('hello')", "exec_command", ""),
        ("[exec_command] pwd", None, "\x1b[33m"),
        ("[write_stdin] poll", None, "\x1b[35m"),
        ("[spawn_agent] spawned Ada", None, "\x1b[34m"),
        ("[read_file] file", None, "\x1b[2m"),
    ],
)
def test_tool_colors_are_selected_by_event_presentation(message, tool_name, color):
    assert (
        colorize_tool_message(message, True, tool_name)
        == "\x1b[1m" + color + message + "\x1b[0m"
    )
    assert colorize_tool_message(message, False, tool_name) == message


def test_cli_colors():
    for kind, color in [("assistant", "32"), ("status", "36"), ("error", "31")]:
        assert (
            colorize_cli_message("text", kind, True)
            == "\x1b[1m\x1b[" + color + "mtext\x1b[0m"
        )
        assert colorize_cli_message("text", kind, False) == "text"
    assert colorize_cli_message("text", "other", True) == "text"


@pytest.mark.parametrize("outcome", ["success", "retry", "failure"])
def test_stream_rendering_uses_completed_attempt_only(outcome):
    display, operations = make_display()
    TurnStartedEvent("turn", ("hello",)).render(display)
    AssistantDeltaEvent("partial", "turn").render(display)
    assert ("log", "assistant> partial") not in operations
    if outcome == "retry":
        StreamErrorEvent("Reconnecting", 1, 5, 0, "disconnected", "turn").render(
            display
        )
        assert display.stream_buffer == ""
        AssistantDeltaEvent("final", "turn").render(display)
    event = (
        TurnFailedEvent("turn", 1, "failed", "RuntimeError", 0)
        if outcome == "failure"
        else TurnCompletedEvent(
            "turn", 1, "final" if outcome == "retry" else "partial", 0
        )
    )
    event.render(display)
    expected = "final" if outcome == "retry" else "partial"
    assert [
        text
        for channel, text in operations
        if channel == "log" and text.startswith("assistant>")
    ] == ["assistant> " + expected]
    assert (("log", "Error: failed") in operations) == (outcome == "failure")
    assert operations[-1] == ("status", None)


def test_render_keeps_subscriber_state_and_coloring_out_of_events():
    colored, colored_operations = make_display(True)
    plain, plain_operations = make_display()
    call = ToolCall("call", "exec_command", {"cmd": "pwd"})
    event = ToolCompletedEvent(
        "turn", call, ToolResult("call", "exec_command", "/work")
    )
    original = asdict(event)
    event.render(colored)
    event.render(plain)
    assert asdict(event) == original
    assert event.visualize() == "[exec_command] pwd -> /work"
    assert colored_operations[-1] == (
        "log",
        "\x1b[1m\x1b[33m[exec_command] pwd -> /work\x1b[0m",
    )
    assert plain_operations[-1] == ("log", event.visualize())
    AssistantDeltaEvent("partial").render(colored)
    AssistantDeltaEvent("partial").render(plain)
    StreamErrorEvent("Retrying", 1, 2, 0, "lost").render(colored)
    InputQueuedEvent("queued", "next", "enqueue", "cli", True, True).render(colored)
    assert colored.stream_buffer == ""
    assert plain.stream_buffer == "partial"
    assert colored.queued_steer_prompts == {"queued": ["next"]}
    assert not plain.queued_steer_prompts


def test_tool_render_orders_stream_flush_status_and_log_and_remembers_names():
    display, operations = make_display()
    AssistantDeltaEvent("checking").render(display)
    operations.clear()
    call = ToolCall("call", "spawn_agent", {})
    ToolStartedEvent("turn", call).render(display)
    ToolCompletedEvent(
        "turn",
        call,
        ToolResult(
            "call",
            "spawn_agent",
            {"agent_id": "1234567890abcdef", "nickname": "Ada"},
        ),
    ).render(display)
    assert operations == [
        ("log", "assistant> checking"),
        ("status", "calling spawn_agent({})"),
        ("status", "called spawn_agent"),
        ("log", "[spawn_agent] spawned Ada (12345678...cdef)"),
    ]
    call = ToolCall("send", "send_input", {"id": "1234567890abcdef", "message": "go"})
    ToolCompletedEvent("turn", call, ToolResult("send", "send_input", "")).render(
        display
    )
    assert operations[-1] == ("log", "[send_input] Ada <- go")
    other, _operations = make_display()
    assert other.agent_names == {}


@pytest.mark.parametrize("explicit", [False, True])
def test_queue_render_consumes_feedback_only_when_input_starts(explicit):
    display, operations = make_display()
    InputQueuedEvent("submission", "next", "enqueue", "cli", explicit, True).render(
        display
    )
    assert operations == ([("log", "[steer] queued: next")] if explicit else [])
    TurnStartedEvent("turn", ("next",), "submission").render(display)
    assert operations[-3:] == [
        ("status", "turn_started"),
        ("log", "[steer] inserted: next"),
        ("log", "user> next"),
    ]
    assert display.queued_steer_prompts == display.inserted_steer_prompts == {}


def test_input_render_switches_prompt_and_preserves_context_hint():
    display, operations = make_display()
    display.set_context_window_tokens(100000)
    TokenCountEvent({"total_tokens": 56000}).render(display)
    assert operations == [("prompt", "pyco(50%)> ")]
    event = InputRequestedEvent(
        "request",
        "questions",
        False,
        question={
            "header": "Choice",
            "question": "Which?",
            "options": [{"label": "First", "description": "Use first"}],
        },
    )
    AssistantDeltaEvent("before question").render(display)
    operations.clear()
    event.render(display)
    assert operations[0] == ("log", "assistant> before question")
    assert operations[-2:] == [("status", None), ("prompt", "answer> ")]
    replace(event, other=True).render(display)
    assert operations[-1] == ("prompt", "other> ")
    replace(event, request_kind="permissions", permissions={}).render(display)
    assert operations[-1] == ("prompt", "permissions> ")
    InputResolvedEvent("request").render(display)
    assert operations[-1] == ("prompt", "pyco(50%)> ")


@pytest.mark.parametrize("failed", [False, True])
def test_manual_compact_render_leaves_result_logging_to_command(failed):
    display, operations = make_display()
    CompactStartedEvent("turn", "manual", None, None).render(display)
    operations.clear()
    if failed:
        terminal = CompactFailedEvent(
            "turn", "manual", None, None, "failed", "ValueError", 0
        )
        command = CommandFailedEvent("submission", "/compact", "failed", "cli")
        expected = "Error: failed"
    else:
        terminal = CompactCompletedEvent("turn", "manual", None, None, 5, 0, 0, 0)
        command = CommandCompletedEvent(
            "submission",
            "/compact",
            {
                "kind": "compacted",
                "original_item_count": 5,
                "retained_item_count": 0,
                "pruned_tool_results": 0,
            },
            "cli",
        )
        expected = terminal.summary
    terminal.render(display)
    assert operations == [("status", None)]
    command.render(display)
    assert operations == [("status", None), ("log", expected)]


def test_render_flush_boundaries_and_auto_compact_status():
    display, operations = make_display()
    AssistantDeltaEvent("partial").render(display)
    operations.clear()
    ModelCompletedEvent("turn", 1, 1).render(display)
    Event().render(display)
    assert operations == []
    ModelCalledEvent("turn", 2, 1, 0).render(display)
    assert operations == [("log", "assistant> partial")]
    AutoCompactStartedEvent("turn", "mid_turn", 100, 90).render(display)
    assert operations[-1] == ("status", "compacting")
    AutoCompactCompletedEvent("turn", "mid_turn", 100, 90, 5, 0, 0).render(display)
    assert operations[-1] == ("status", "compacted")
    TurnCompletedEvent("turn", 2, None, 1).render(display)
    assert operations[-1] == ("status", "idle: sleeping")
    AssistantDeltaEvent("last").render(display)
    SessionClosedEvent().render(display)
    assert display.closed
    assert operations[-1] == ("log", "assistant> last")


def test_snapshot_render_restores_active_output_and_clears_old_queue_feedback():
    display, operations = make_display()
    InputQueuedEvent("old", "old input", "enqueue", "cli", True, True).render(display)
    event = SessionStateEvent(
        "attach",
        {
            "title": "restored",
            "context_window": 100000,
            "usage_tokens": 56000,
            "input_request": None,
            "active_turn": {
                "turn_id": "turn",
                "submission_id": "current",
                "user_texts": ["live"],
                "assistant_text": "part",
            },
        },
    )
    original = asdict(event)
    event.render(display)
    assert display.title == "restored"
    assert display.context_remaining_percent == 50
    assert not display.queued_steer_prompts
    assert display.stream_buffer == "part"
    AssistantDeltaEvent("ial").render(display)
    TurnCompletedEvent("turn", 1, "partial", 0).render(display)
    assert operations.count(("log", "assistant> partial")) == 1
    assert asdict(event) == original


@pytest.mark.parametrize("accepts_input", [False, True])
def test_admission_render_explains_graceful_close_without_finishing_output(
    accepts_input,
):
    display, operations = make_display()
    display.stream_buffer = "partial"
    SessionStateEvent(
        "admission",
        {
            "title": "",
            "accepts_input": accepts_input,
        },
    ).render(display)
    assert operations == (
        []
        if accepts_input
        else [
            ("log", "[closing] Waiting for accepted work and cleanup to finish."),
        ]
    )
    assert display.stream_buffer == "partial"
    assert not display.closed


def test_non_event_presentation_and_status_frames():
    display, operations = make_display(True)
    display.start(["help", "exit"])
    assert operations == [
        ("log", "pycodex interactive mode. Type /exit or press Ctrl+C to quit."),
        ("log", "Extra commands: /help, /exit"),
    ]
    result = TurnResult("turn", "answer", 1, (), ())
    render_result("turn", result, True, display.log)
    assert json.loads(operations[-1][1])["output_text"] == "answer"
    assert "\x1b" not in operations[-1][1]
    render_result("turn", result, False, display.log)
    assert operations[-1] == ("log", "answer")
    count = len(operations)
    render_result("command", {"kind": "closed"}, False, display.log)
    assert len(operations) == count
    assert format_error("first\nsecond") == "Error: first\n  second"
    assert format_error("first\nsecond", False) == "Error: first\nsecond"
    assert EventDisplay.status_frame(None, 1) is None
    assert EventDisplay.status_frame("talking", 1) == "⠙ talking"


def test_closed_state_is_recorded_even_when_terminal_output_fails():
    display, _operations = make_display()
    display.stream_buffer = "partial"

    def fail_output(text):
        raise OSError("terminal closed")

    display.log = fail_output
    with pytest.raises(OSError, match="terminal closed"):
        SessionClosedEvent().render(display)
    assert display.closed
