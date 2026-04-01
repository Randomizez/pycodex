"""Compare Plan-mode `request_user_input` round-trip captures.

This helper drives:

- upstream Codex through a tmux-backed inline TUI session (`/plan`, then prompt,
  then a numeric answer)
- `pycodex` through its local runtime in explicit Plan mode

Both clients talk to the same deterministic origin Responses server through the
proxy capture server from `tests.fake_responses_server`, so the comparison is:

- deterministic at the model-output level
- still recorded through the same proxy-mode capture path we use elsewhere

Like `tests/compare_tool_schemas.py`, this is intentionally a support script
under `tests/` rather than an always-on pytest, because it depends on a local
Codex CLI installation plus tmux.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import json
import random
import shlex
import shutil
import string
import subprocess
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

from pycodex.cli import _build_model_client, build_runtime
from pycodex.runtime_services import get_runtime_environment
from tests.compare_tool_schemas import (
    build_proxy_config_copy,
    load_provider_info,
    rewrite_config_base_url,
)
from tests.fake_responses_server import CaptureStore, build_proxy_handler

DEFAULT_CONFIG_PATH = Path.home() / ".codex" / "config.toml"
DEFAULT_OUTPUT_ROOT = Path(".tmp") / "request_user_input_roundtrip_compare"
DEFAULT_PROMPT = "Please ask exactly one question via request_user_input."
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MODEL_ID = "gpt-5.4"
DEFAULT_CALL_ID = "user-input-call"
DEFAULT_QUESTION_ID = "confirm_path"
DEFAULT_UPSTREAM_COMMAND = "codex"


@dataclass(frozen=True, slots=True)
class RunCapture:
    label: str
    request_paths: tuple[Path, Path]
    request_bodies: tuple[dict[str, object], dict[str, object]]
    collaboration_text: str | None
    function_call_item: dict[str, object] | None
    function_call_output_item: dict[str, object] | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uv run python tests/compare_request_user_input_roundtrip.py",
        description=(
            "Capture a deterministic Plan-mode `request_user_input` round trip "
            "from upstream Codex and pycodex through proxy-mode fake servers."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the real Codex config.toml used as the source configuration.",
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory used to store captures, logs, and comparison output.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt sent after switching upstream Codex into Plan mode.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Timeout applied to each capture run.",
    )
    parser.add_argument(
        "--upstream-command",
        default=DEFAULT_UPSTREAM_COMMAND,
        help=(
            "Command used to launch upstream Codex. Examples: `codex`, "
            "`npx -y @openai/codex@0.117.0`."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    output_root = Path(args.root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    provider_name, _upstream_base_url = load_provider_info(config_path)
    upstream_version = detect_upstream_version(args.upstream_command)

    sequence = build_request_user_input_sequence(
        model_id=DEFAULT_MODEL_ID,
        call_id=DEFAULT_CALL_ID,
        question_id=DEFAULT_QUESTION_ID,
    )

    with scripted_origin_server(
        model_id=DEFAULT_MODEL_ID,
        response_bodies=sequence,
    ) as upstream_origin:
        upstream_capture = run_upstream_codex_capture(
            config_path=config_path,
            provider_name=provider_name,
            output_root=output_root,
            prompt=args.prompt,
            timeout_seconds=args.timeout_seconds,
            upstream_command=args.upstream_command,
            origin_base_url=upstream_origin.base_url,
        )

    with scripted_origin_server(
        model_id=DEFAULT_MODEL_ID,
        response_bodies=sequence,
    ) as pycodex_origin:
        pycodex_capture = run_pycodex_capture(
            config_path=config_path,
            output_root=output_root,
            prompt=args.prompt,
            timeout_seconds=args.timeout_seconds,
            origin_base_url=pycodex_origin.base_url,
        )

    comparison = {
        "upstream_command": args.upstream_command,
        "upstream_version": upstream_version,
        "upstream": serialize_run_capture(upstream_capture),
        "pycodex": serialize_run_capture(pycodex_capture),
        "comparison": {
            "plan_mode_prompt_equal": (
                upstream_capture.collaboration_text == pycodex_capture.collaboration_text
            ),
            "function_call_equal": (
                upstream_capture.function_call_item == pycodex_capture.function_call_item
            ),
            "function_call_output_equal": (
                upstream_capture.function_call_output_item
                == pycodex_capture.function_call_output_item
            ),
        },
    }
    comparison_path = output_root / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2))

    print(f"Upstream version: {upstream_version or 'unknown'}")
    print(f"Upstream first request:  {upstream_capture.request_paths[0]}")
    print(f"Upstream second request: {upstream_capture.request_paths[1]}")
    print(f"pycodex first request:   {pycodex_capture.request_paths[0]}")
    print(f"pycodex second request:  {pycodex_capture.request_paths[1]}")
    print(f"Comparison JSON:         {comparison_path}")
    print()
    print(
        "Plan prompt equal:       "
        f"{'yes' if comparison['comparison']['plan_mode_prompt_equal'] else 'no'}"
    )
    print(
        "Function call equal:     "
        f"{'yes' if comparison['comparison']['function_call_equal'] else 'no'}"
    )
    print(
        "Function call output equal: "
        f"{'yes' if comparison['comparison']['function_call_output_equal'] else 'no'}"
    )
    if not comparison["comparison"]["function_call_output_equal"]:
        print()
        print("Upstream function_call_output:")
        print(json.dumps(upstream_capture.function_call_output_item, ensure_ascii=False, indent=2))
        print()
        print("pycodex function_call_output:")
        print(json.dumps(pycodex_capture.function_call_output_item, ensure_ascii=False, indent=2))


def detect_upstream_version(upstream_command: str) -> str | None:
    command = shlex.split(upstream_command) + ["--version"]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=20.0,
        )
    except Exception:
        return None
    text = (result.stdout or result.stderr).strip()
    return text or None


def build_request_user_input_sequence(
    *,
    model_id: str,
    call_id: str,
    question_id: str,
) -> tuple[str, str]:
    first = "".join(
        [
            "event: response.created\n",
            f'data: {json.dumps({"type": "response.created", "response": {"id": "resp_1", "object": "response", "status": "in_progress", "model": model_id}}, ensure_ascii=False)}\n\n',
            "event: response.output_item.done\n",
            "data: "
            + json.dumps(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": "request_user_input",
                        "arguments": json.dumps(
                            {
                                "questions": [
                                    {
                                        "id": question_id,
                                        "header": "Confirm",
                                        "question": "Proceed with the plan?",
                                        "options": [
                                            {
                                                "label": "Yes (Recommended)",
                                                "description": "Continue the current plan.",
                                            },
                                            {
                                                "label": "No",
                                                "description": "Stop and revisit the approach.",
                                            },
                                        ],
                                    }
                                ]
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                },
                ensure_ascii=False,
            )
            + "\n\n",
            "event: response.completed\n",
            'data: {"type":"response.completed","response":{"id":"resp_1","output":[]}}\n\n',
        ]
    )
    second = "".join(
        [
            "event: response.created\n",
            f'data: {json.dumps({"type": "response.created", "response": {"id": "resp_2", "object": "response", "status": "in_progress", "model": model_id}}, ensure_ascii=False)}\n\n',
            "event: response.output_item.done\n",
            'data: {"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"thanks"}]}}\n\n',
            "event: response.completed\n",
            'data: {"type":"response.completed","response":{"id":"resp_2","output":[]}}\n\n',
        ]
    )
    return first, second


class ScriptedOriginServer:
    def __init__(self, *, model_id: str, response_bodies: tuple[str, ...]) -> None:
        self._model_id = model_id
        self._response_bodies = response_bodies
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._httpd is None:
            raise RuntimeError("origin server has not started")
        return f"http://127.0.0.1:{self._httpd.server_port}/v1"

    def start(self) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            counter = 0

            def log_message(self, format: str, *args) -> None:
                del format, args
                return

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path.endswith("/models") or parsed.path == "/models":
                    body = json.dumps(
                        {
                            "object": "list",
                            "data": [{"id": outer._model_id, "object": "model"}],
                        }
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                body = json.dumps({"ok": True}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                if length:
                    self.rfile.read(length)
                idx = Handler.counter
                Handler.counter += 1
                payload = outer._response_bodies[min(idx, len(outer._response_bodies) - 1)]
                body = payload.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._httpd is not None:
            self._httpd.server_close()


@contextmanager
def scripted_origin_server(
    *,
    model_id: str,
    response_bodies: tuple[str, ...],
):
    server = ScriptedOriginServer(
        model_id=model_id,
        response_bodies=response_bodies,
    )
    server.start()
    try:
        yield server
    finally:
        server.stop()


def run_upstream_codex_capture(
    *,
    config_path: Path,
    provider_name: str,
    output_root: Path,
    prompt: str,
    timeout_seconds: float,
    upstream_command: str,
    origin_base_url: str,
) -> RunCapture:
    capture_root = output_root / "upstream"
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    proxy_url = run_proxy_capture_with_tmux(
        capture_root=capture_root,
        origin_base_url=origin_base_url,
        timeout_seconds=timeout_seconds,
        provider_name=provider_name,
        upstream_command=upstream_command,
        prompt=prompt,
        terminal_log_path=log_root / "upstream_tmux.log",
    )
    del proxy_url, config_path
    return load_roundtrip_capture("upstream", capture_root)


def run_pycodex_capture(
    *,
    config_path: Path,
    output_root: Path,
    prompt: str,
    timeout_seconds: float,
    origin_base_url: str,
) -> RunCapture:
    capture_root = output_root / "pycodex"
    config_root = output_root / "config"
    temp_config_path = build_proxy_config_copy(config_path, config_root)

    def run_with_proxy(proxy_url: str) -> None:
        rewrite_config_base_url(temp_config_path, proxy_url)
        asyncio.run(
            run_pycodex_plan_session(
                config_path=temp_config_path,
                prompt=prompt,
                timeout_seconds=timeout_seconds,
            )
        )

    run_proxy_capture_inline(
        capture_root=capture_root,
        origin_base_url=origin_base_url,
        timeout_seconds=timeout_seconds,
        runner=run_with_proxy,
    )
    return load_roundtrip_capture("pycodex", capture_root)


def run_proxy_capture_with_tmux(
    *,
    capture_root: Path,
    origin_base_url: str,
    timeout_seconds: float,
    provider_name: str,
    upstream_command: str,
    prompt: str,
    terminal_log_path: Path,
) -> str:
    if capture_root.exists():
        shutil.rmtree(capture_root)
    capture_root.mkdir(parents=True, exist_ok=True)
    proxy_store = CaptureStore(capture_root)
    proxy = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        build_proxy_handler(proxy_store, origin_base_url, timeout_seconds),
    )
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    proxy_url = f"http://127.0.0.1:{proxy.server_port}/v1"

    session_name = "codex_rui_" + random_suffix()
    upstream_shell = (
        f"{upstream_command} --no-alt-screen "
        f"-c model_providers.{provider_name}.base_url={shlex.quote(proxy_url)}"
    )

    try:
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session_name, upstream_shell],
            check=True,
        )
        time.sleep(5.0)
        tmux_send_line(session_name, "/plan")
        time.sleep(2.0)
        tmux_send_line(session_name, prompt)
        wait_for_post_count(capture_root, 1, timeout_seconds)
        time.sleep(2.0)
        tmux_send_keys(session_name, "1")
        wait_for_post_count(capture_root, 2, timeout_seconds)
        terminal_log_path.write_text(tmux_capture_pane(session_name))
    finally:
        subprocess.run(["tmux", "kill-session", "-t", session_name], check=False)
        proxy.shutdown()
        proxy_thread.join(timeout=5)
        proxy.server_close()
    return proxy_url


def run_proxy_capture_inline(
    *,
    capture_root: Path,
    origin_base_url: str,
    timeout_seconds: float,
    runner,
) -> None:
    if capture_root.exists():
        shutil.rmtree(capture_root)
    capture_root.mkdir(parents=True, exist_ok=True)
    proxy_store = CaptureStore(capture_root)
    proxy = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        build_proxy_handler(proxy_store, origin_base_url, timeout_seconds),
    )
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    proxy_url = f"http://127.0.0.1:{proxy.server_port}/v1"
    try:
        runner(proxy_url)
    finally:
        proxy.shutdown()
        proxy_thread.join(timeout=5)
        proxy.server_close()


async def run_pycodex_plan_session(
    *,
    config_path: Path,
    prompt: str,
    timeout_seconds: float,
) -> None:
    args = SimpleNamespace(
        config=str(config_path),
        profile=None,
        vllm_endpoint=None,
        use_chat_completion=False,
        system_prompt=None,
        timeout_seconds=timeout_seconds,
        json=False,
        prompt=[],
    )
    client = _build_model_client(
        args.config,
        args.profile,
        args.timeout_seconds,
    )
    runtime = build_runtime(
        args.config,
        args.profile,
        args.system_prompt,
        client,
        session_mode="tui",
        collaboration_mode="plan",
    )
    worker = asyncio.create_task(runtime.run_forever())
    runtime_environment = get_runtime_environment()

    async def answer_handler(payload: dict[str, object]) -> dict[str, object]:
        questions = payload.get("questions")
        answers: dict[str, dict[str, list[str]]] = {}
        if isinstance(questions, list):
            for question in questions:
                if not isinstance(question, dict):
                    continue
                question_id = str(question.get("id", "")).strip()
                options = question.get("options")
                if not question_id or not isinstance(options, list) or not options:
                    continue
                first_option = options[0]
                if not isinstance(first_option, dict):
                    continue
                label = str(first_option.get("label", "")).strip()
                if label:
                    answers[question_id] = {"answers": [label]}
        return {"answers": answers}

    runtime_environment.request_user_input_manager.set_handler(answer_handler)
    try:
        await runtime.submit_user_turn(prompt)
    finally:
        runtime_environment.request_user_input_manager.set_handler(None)
        await runtime.shutdown()
        await worker


def load_roundtrip_capture(label: str, capture_root: Path) -> RunCapture:
    request_files = sorted(capture_root.glob("*_POST_*.json"))
    captures = [
        (path, json.loads(path.read_text()))
        for path in request_files
    ]
    first_index = None
    for index, (_path, capture) in enumerate(captures):
        response_body = capture.get("response", {}).get("body")
        if isinstance(response_body, str) and DEFAULT_CALL_ID in response_body:
            first_index = index
            break
    if first_index is None or first_index + 1 >= len(captures):
        raise RuntimeError(
            "failed to locate a full request_user_input round trip for "
            f"{label} under {capture_root}"
        )

    first_path, first_capture = captures[first_index]
    second_path, second_capture = captures[first_index + 1]
    first_body = first_capture["body"]
    second_body = second_capture["body"]
    return RunCapture(
        label=label,
        request_paths=(first_path, second_path),
        request_bodies=(first_body, second_body),
        collaboration_text=extract_collaboration_text(first_body),
        function_call_item=extract_function_call(second_body),
        function_call_output_item=extract_function_call_output(second_body),
    )


def extract_collaboration_text(request_body: dict[str, object]) -> str | None:
    for item in request_body.get("input", []):
        if not isinstance(item, dict) or item.get("role") != "developer":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            text = str(part.get("text", ""))
            if text.startswith("<collaboration_mode>"):
                return text
    return None


def extract_function_call(request_body: dict[str, object]) -> dict[str, object] | None:
    for item in request_body.get("input", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call" and item.get("name") == "request_user_input":
            return item
    return None


def extract_function_call_output(request_body: dict[str, object]) -> dict[str, object] | None:
    for item in request_body.get("input", []):
        if not isinstance(item, dict):
            continue
        if (
            item.get("type") == "function_call_output"
            and item.get("call_id") == DEFAULT_CALL_ID
        ):
            return item
    return None


def serialize_run_capture(capture: RunCapture) -> dict[str, object]:
    return {
        "label": capture.label,
        "request_paths": [str(path) for path in capture.request_paths],
        "collaboration_text": capture.collaboration_text,
        "function_call_item": capture.function_call_item,
        "function_call_output_item": capture.function_call_output_item,
    }


def wait_for_post_count(capture_root: Path, expected_count: int, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if len(list(capture_root.glob("*_POST_*.json"))) >= expected_count:
            return
        time.sleep(0.1)
    raise RuntimeError(
        f"timed out waiting for {expected_count} POST captures under {capture_root}"
    )


def tmux_send_line(session_name: str, text: str) -> None:
    tmux_send_keys(session_name, text)
    time.sleep(0.5)
    subprocess.run(["tmux", "send-keys", "-t", f"{session_name}:", "Enter"], check=True)


def tmux_send_keys(session_name: str, text: str) -> None:
    subprocess.run(["tmux", "send-keys", "-t", f"{session_name}:", text], check=True)


def tmux_capture_pane(session_name: str) -> str:
    result = subprocess.run(
        ["tmux", "capture-pane", "-pt", f"{session_name}:"],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def random_suffix(length: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


if __name__ == "__main__":
    main()
