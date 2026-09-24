import argparse
import asyncio
import inspect
import os
import shlex
import signal
import sys
import tempfile
import threading
import traceback
import typing
from contextlib import contextmanager

from prompt_toolkit import PromptSession
from prompt_toolkit.enums import DEFAULT_BUFFER
from prompt_toolkit.filters import has_focus
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout

from .bootstrap import build_agent, build_model, build_runtime, configure_loguru
from .events import DEFAULT_MAIN_PROMPT, EventDisplay
from .model import DEFAULT_CODEX_CONFIG_PATH
from .portable import bootstrap_called_home, upload_codex_home
from .utils import get_debug_dir
from .utils.event_helpers import format_error, render_result


def build_parser():
    parser = argparse.ArgumentParser(
        prog="pycodex",
        description="Minimal Codex-style local CLI backed by ~/.codex/config.toml.",
    )
    parser.add_argument(
        "prompt", nargs="*", help="Prompt text. If omitted, read from stdin."
    )
    parser.add_argument(
        "--put",
        default=None,
        metavar="PATH@SERVER",
        help="Upload a Codex home using `--put @host:port` or `--put /path/.codex@host:port`.",
    )
    parser.add_argument(
        "--call",
        default=None,
        help="Download and use a stored Codex home via <secret>-<call_id>@<host:port>.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CODEX_CONFIG_PATH),
        help="Path to Codex config.toml.",
    )
    parser.add_argument(
        "--profile", default=None, help="Optional profile name from config.toml."
    )
    parser.add_argument(
        "--vllm-endpoint",
        default=None,
        help="Start a local responses compat server for a chat-completions-backed vLLM endpoint.",
    )
    parser.add_argument(
        "--use-chat-completion",
        default=False,
        action="store_true",
        help="Start a local responses compat server for this session.",
    )
    parser.add_argument(
        "--use-messages",
        default=False,
        action="store_true",
        help="Route the local responses compat server to a downstream /v1/messages backend.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional base instructions override passed to the model.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="HTTP timeout for one model call.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the full TurnResult as JSON."
    )
    return parser


def should_run_interactive(prompt_parts, stdin_is_tty):
    return not prompt_parts and stdin_is_tty


def resolve_prompt_text(prompt_parts):
    if prompt_parts:
        return " ".join(prompt_parts).strip()
    if not sys.stdin.isatty():
        prompt_text = sys.stdin.read().strip()
        if prompt_text:
            return prompt_text
    raise ValueError("prompt is required either as argv text or stdin")


async def run_cli(args):
    runtime = None
    debug_dir = get_debug_dir()
    phase_handle = (
        None
        if debug_dir is None
        else (debug_dir / "phase.log").open("a", encoding="utf-8")
    )

    def phase(message):
        if phase_handle is not None:
            phase_handle.write(message + "\n")
            phase_handle.flush()

    try:
        if args.put is not None and args.call:
            raise ValueError("--put and --call cannot be combined")
        if args.put is not None and args.prompt:
            raise ValueError("--put does not accept prompt text")
        configure_loguru()
        config_path = args.config
        if args.put is not None:

            def emit_put_log(message):
                print(message, flush=True)

            call_spec = upload_codex_home(args.put, event_handler=emit_put_log)
            emit_put_log(f"[put] testing call: {call_spec}")
            with tempfile.TemporaryDirectory(prefix="pycodex-put-call-test-") as tmpdir:
                config_path = bootstrap_called_home(call_spec, storage_root=tmpdir)
            emit_put_log(f"[put] call test ok: {config_path.name}")
            print("[put] one-click start:", flush=True)
            print(f"pycodex --call {shlex.quote(call_spec)}", flush=True)
            return 0
        if args.call:
            phase("bootstrap_called_home:start")
            config_path = bootstrap_called_home(args.call)
            phase("bootstrap_called_home:done")
            os.environ["CODEX_HOME"] = str(config_path.parent)
        phase("build_model:start")
        model = build_model(
            config_path=str(config_path),
            profile=args.profile,
            timeout_seconds=args.timeout_seconds,
            vllm_endpoint=args.vllm_endpoint,
            use_chat_completion=args.use_chat_completion or None,
            use_messages=args.use_messages,
        )
        phase("build_model:done")
        phase("build_agent:start")
        agent = build_agent(
            model,
            config_path=str(config_path),
            profile=args.profile,
            system_prompt=args.system_prompt,
        )
        phase("build_agent:done")
        runtime = build_runtime(agent)
        if should_run_interactive(args.prompt, sys.stdin.isatty()):
            return await run_interactive_session(runtime, args.json, str(config_path))
        prompt_text = resolve_prompt_text(args.prompt)
        await runtime.start(str(config_path))
        phase("submit_input:start")
        receipt = await runtime.submit_input(prompt_text, "cli")
        result = await receipt.future
        phase("submit_input:done")
        render_result(receipt.kind, result, args.json, print)
        return 0
    except Exception as exc:
        phase("fatal_exception")
        if debug_dir is not None:
            (debug_dir / "fatal_error.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        print(format_error(exc, indent=False), file=sys.stderr)
        return 1
    finally:
        if phase_handle is not None:
            phase_handle.close()
        if runtime is not None:
            await runtime.close()


@contextmanager
def _cli_sigint_handler(runtime):
    if threading.current_thread() is not threading.main_thread():
        yield lambda: None
        return

    def handle_sigint(signum, frame):
        if not runtime.accepts_input:
            # A second interrupt must not re-enter asyncio's shutdown waits.
            os._exit(130)
        signal.default_int_handler(signum, frame)

    def install_handler():
        signal.signal(signal.SIGINT, handle_sigint)

    previous_handler = signal.getsignal(signal.SIGINT)
    install_handler()
    try:
        yield install_handler
    finally:
        signal.signal(signal.SIGINT, previous_handler)


async def run_interactive_session(runtime, json_mode, config_path=None, view=None):
    if view is None:
        view = CliSessionView()
    await runtime.start(config_path)
    frontend_id = runtime.attach(view.handle_event)

    def show_result(future):
        if not future.cancelled() and future.exception() is None:
            render_result("turn", future.result(), True, view.write_line)

    view.display.start(runtime.commands())
    with _cli_sigint_handler(runtime) as install_sigint_handler:
        try:
            while not view.display.closed:
                try:
                    raw_line = await view.poll_prompt()
                except EOFError:
                    break
                if raw_line is None:
                    await asyncio.sleep(0.05)
                    continue
                install_sigint_handler()
                try:
                    receipt = await runtime.submit_input(raw_line, sender="cli")
                except Exception as exc:
                    view.display.show_error(str(exc))
                    continue
                if json_mode and receipt.kind == "turn":
                    receipt.future.add_done_callback(show_result)
        finally:
            # Older prompt_toolkit versions reset SIGINT when the prompt exits.
            install_sigint_handler()
            try:
                await runtime.close()
            finally:
                runtime.detach(frontend_id)
                view.close()
    return 0


class Prompter:
    def __init__(self, prompt: str = DEFAULT_MAIN_PROMPT, lock=None):
        self.lock = lock or threading.Lock()
        self._prompt_session = PromptSession(**prompt_session_kwargs())
        self.prompt = prompt
        self._status = None
        self._status_frame_index = 0
        self._prompt_task = None

    def set_prompt(self, prompt):
        self.prompt = prompt

    def set_status(self, text):
        self._status = text

    async def poll_input(self) -> "typing.Union[str, None]":
        if self._prompt_task is None:
            self._prompt_task = asyncio.create_task(self._block_prompt())
        done, _pending = await asyncio.wait(
            {self._prompt_task},
            timeout=0.05,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            return None
        prompt_task, self._prompt_task = self._prompt_task, None
        try:
            return prompt_task.result()
        except asyncio.CancelledError:
            return None

    async def _block_prompt(self):
        with patch_stdout(raw=True):
            return await self._prompt_session.prompt_async(
                lambda: self.prompt,
                refresh_interval=0.12,
                bottom_toolbar=self._get_status,
                set_exception_handler=False,
            )

    def _get_status(self):
        self._status_frame_index += 1
        return EventDisplay.status_frame(self._status, self._status_frame_index)

    def close(self) -> "None":
        if self._prompt_task is not None and not self._prompt_task.done():
            self._prompt_task.cancel()
            self._prompt_task = None


def prompt_session_kwargs() -> "typing.Dict[str, object]":
    key_bindings = KeyBindings()

    @key_bindings.add("c-c", filter=has_focus(DEFAULT_BUFFER))
    @key_bindings.add("<sigint>")
    def exit_prompt(event):
        event.app.exit(exception=EOFError, style="class:aborting")

    kwargs = {
        "erase_when_done": True,
        "enable_system_prompt": True,
        "key_bindings": key_bindings,
    }
    try:
        parameters = inspect.signature(PromptSession.__init__).parameters
    except (TypeError, ValueError):
        return kwargs
    if "show_frame" in parameters:
        kwargs["show_frame"] = True
    return kwargs


class CliSessionView:
    """Execute terminal I/O; events own presentation decisions and state."""

    def __init__(self, context_window_tokens=None):
        self._line_output = print
        self._terminal_lock = threading.RLock()
        self.prompter = Prompter(lock=self._terminal_lock)
        color_enabled = sys.stdout.isatty() and os.environ.get(
            "PYCODEX_NO_COLOR",
            "",
        ).strip().lower() not in {"1", "true", "yes", "on"}
        self.display = EventDisplay(
            self.write_line,
            self.prompter.set_status,
            self.prompter.set_prompt,
            color_enabled,
            context_window_tokens,
        )

    def handle_event(self, event):
        with self._terminal_lock:
            event.render(self.display)

    def write_line(self, text):
        with self._terminal_lock:
            self._line_output(text)

    async def poll_prompt(self, prompt=None):
        if prompt:
            self.prompter.set_prompt(prompt)
        return await self.prompter.poll_input()

    def close(self):
        self.prompter.close()


def ipython_agent(config_path=DEFAULT_CODEX_CONFIG_PATH):
    from loguru import logger

    from .tools.ipython_tool import attach_ipython_tool

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    model = build_model(config_path)
    agent = build_agent(client=model, config_path=config_path)
    attach_ipython_tool(agent)
    return agent


def main(argv=None):
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    if raw_args and raw_args[0] == "doctor":
        from .doctor import build_doctor_parser, run_doctor_cli

        parser = build_doctor_parser()
        args = parser.parse_args(raw_args[1:])
        try:
            return asyncio.run(run_doctor_cli(args))
        except ValueError as exc:
            parser.error(str(exc))
        except KeyboardInterrupt:
            return 130
        return 0
    parser = build_parser()
    args = parser.parse_args(raw_args)
    try:
        return asyncio.run(run_cli(args))
    except ValueError as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
