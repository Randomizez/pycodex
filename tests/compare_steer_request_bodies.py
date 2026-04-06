"""Compare upstream Codex vs `pycodex` request bodies for a steer flow.

This helper drives both CLIs in interactive mode through tmux while routing
traffic through the local proxy capture server from `tests.fake_responses_server`.
The origin server is scripted so that:

- the first `/responses` request stays open briefly, giving us time to submit a
  steer prompt while the model is still busy
- the second `/responses` request completes immediately

The comparison target is the outbound request body shape, especially the second
request after steer.
"""

import argparse
from contextlib import contextmanager
import json
import random
import re
import shlex
import shutil
import string
import subprocess
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

from tests.compare_request_user_input_roundtrip import (
    tmux_capture_pane,
    tmux_send_line,
    wait_for_post_count,
)
from tests.compare_tool_schemas import (
    build_proxy_config_copy,
    load_provider_info,
    rewrite_config_base_url,
)
from pycodex.compat import ThreadingHTTPServer
from tests.fake_responses_server import CaptureStore, build_proxy_handler
import typing

DEFAULT_CONFIG_PATH = Path.home() / ".codex" / "config.toml"
DEFAULT_OUTPUT_ROOT = Path(".tmp") / "steer_request_body_compare"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MODEL_ID = "gpt-5.4"
DEFAULT_UPSTREAM_COMMAND = "codex"
DEFAULT_INITIAL_PROMPT = "say hi"
DEFAULT_STEER_PROMPT = "say bye instead"
DEFAULT_FIRST_RESPONSE_TEXT = "hi"
DEFAULT_SECOND_RESPONSE_TEXT = "bye"
DEFAULT_FIRST_RESPONSE_DELAY_SECONDS = 3.0


@dataclass(frozen=True, )
class CapturedRequest:
    path: 'Path'
    body: 'typing.Dict[str, object]'
    headers: 'typing.Dict[str, str]'


@dataclass(frozen=True, )
class RunCapture:
    label: 'str'
    first: 'CapturedRequest'
    second: 'CapturedRequest'
    terminal_log_path: 'Path'


def build_parser() -> 'argparse.ArgumentParser':
    parser = argparse.ArgumentParser(
        prog="uv run python tests/compare_steer_request_bodies.py",
        description=(
            "Capture a deterministic steer flow from upstream Codex and pycodex "
            "through the fake/proxy Responses servers, then compare request bodies."
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
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Timeout for each capture run.",
    )
    parser.add_argument(
        "--upstream-command",
        default=DEFAULT_UPSTREAM_COMMAND,
        help="Command used to launch upstream Codex.",
    )
    parser.add_argument(
        "--initial-prompt",
        default=DEFAULT_INITIAL_PROMPT,
        help="Initial prompt submitted before steering.",
    )
    parser.add_argument(
        "--steer-prompt",
        default=DEFAULT_STEER_PROMPT,
        help="Prompt submitted while the first request is still running.",
    )
    return parser


def main() -> 'None':
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    output_root = Path(args.root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    provider_name, _upstream_base_url = load_provider_info(config_path)
    response_bodies = build_steer_response_sequence(
        model_id=DEFAULT_MODEL_ID,
        first_text=DEFAULT_FIRST_RESPONSE_TEXT,
        second_text=DEFAULT_SECOND_RESPONSE_TEXT,
    )

    with scripted_origin_server(
        model_id=DEFAULT_MODEL_ID,
        response_bodies=response_bodies,
        first_delay_seconds=DEFAULT_FIRST_RESPONSE_DELAY_SECONDS,
    ) as upstream_origin:
        upstream_capture = run_upstream_codex_capture(
            config_path=config_path,
            provider_name=provider_name,
            output_root=output_root,
            timeout_seconds=args.timeout_seconds,
            upstream_command=args.upstream_command,
            origin_base_url=upstream_origin.base_url,
            initial_prompt=args.initial_prompt,
            steer_prompt=args.steer_prompt,
            upstream_origin_waiter=upstream_origin.wait_for_post_count,
        )

    with scripted_origin_server(
        model_id=DEFAULT_MODEL_ID,
        response_bodies=response_bodies,
        first_delay_seconds=DEFAULT_FIRST_RESPONSE_DELAY_SECONDS,
    ) as pycodex_origin:
        pycodex_capture = run_pycodex_capture(
            config_path=config_path,
            output_root=output_root,
            timeout_seconds=args.timeout_seconds,
            origin_base_url=pycodex_origin.base_url,
            initial_prompt=args.initial_prompt,
            steer_prompt=args.steer_prompt,
            pycodex_origin_waiter=pycodex_origin.wait_for_post_count,
        )

    comparison = build_comparison(upstream_capture, pycodex_capture)
    comparison_path = output_root / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2))

    print(f"Upstream first request:  {upstream_capture.first.path}")
    print(f"Upstream second request: {upstream_capture.second.path}")
    print(f"pycodex first request:   {pycodex_capture.first.path}")
    print(f"pycodex second request:  {pycodex_capture.second.path}")
    print(f"Comparison JSON:         {comparison_path}")
    print()
    print(
        "First body equal (sans prompt_cache_key):  "
        f"{'yes' if comparison['first']['equal_ignoring_prompt_cache_key'] else 'no'}"
    )
    print(
        "Second body equal (sans prompt_cache_key): "
        f"{'yes' if comparison['second']['equal_ignoring_prompt_cache_key'] else 'no'}"
    )
    print(
        "Upstream same turn_id across steer:        "
        f"{'yes' if comparison['upstream']['same_turn_id_across_requests'] else 'no'}"
    )
    print(
        "pycodex same turn_id across steer:         "
        f"{'yes' if comparison['pycodex']['same_turn_id_across_requests'] else 'no'}"
    )
    if comparison["second"]["diffs"]:
        print()
        print("Second body diffs (sans prompt_cache_key):")
        for diff in comparison["second"]["diffs"][:20]:
            print(f"- {diff}")


def build_steer_response_sequence(
    *,
    model_id: 'str',
    first_text: 'str',
    second_text: 'str',
) -> 'typing.Tuple[str, str]':
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
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": first_text}],
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
            "data: "
            + json.dumps(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": second_text}],
                    },
                },
                ensure_ascii=False,
            )
            + "\n\n",
            "event: response.completed\n",
            'data: {"type":"response.completed","response":{"id":"resp_2","output":[]}}\n\n',
        ]
    )
    return first, second


class ScriptedOriginServer:
    def __init__(
        self,
        *,
        model_id: 'str',
        response_bodies: 'typing.Tuple[str, ...]',
        first_delay_seconds: 'float',
    ) -> 'None':
        self._model_id = model_id
        self._response_bodies = response_bodies
        self._first_delay_seconds = first_delay_seconds
        self._httpd: 'typing.Union[ThreadingHTTPServer, None]' = None
        self._thread: 'typing.Union[threading.Thread, None]' = None
        self._request_count = 0
        self._condition = threading.Condition()

    @property
    def base_url(self) -> 'str':
        if self._httpd is None:
            raise RuntimeError("origin server has not started")
        return f"http://127.0.0.1:{self._httpd.server_port}/v1"

    def start(self) -> 'None':
        outer = self

        class Handler(BaseHTTPRequestHandler):
            counter = 0

            def log_message(self, format: 'str', *args) -> 'None':
                del format, args
                return

            def do_GET(self) -> 'None':
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

            def do_POST(self) -> 'None':
                length = int(self.headers.get("Content-Length", "0"))
                if length:
                    self.rfile.read(length)
                idx = Handler.counter
                Handler.counter += 1
                with outer._condition:
                    outer._request_count += 1
                    outer._condition.notify_all()
                if idx == 0 and outer._first_delay_seconds > 0:
                    time.sleep(outer._first_delay_seconds)
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

    def stop(self) -> 'None':
        if self._httpd is not None:
            self._httpd.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._httpd is not None:
            self._httpd.server_close()

    def wait_for_post_count(self, expected_count: 'int', timeout_seconds: 'float') -> 'None':
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._request_count < expected_count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"timed out waiting for origin POST count {expected_count}"
                    )
                self._condition.wait(timeout=remaining)


@contextmanager
def scripted_origin_server(
    *,
    model_id: 'str',
    response_bodies: 'typing.Tuple[str, ...]',
    first_delay_seconds: 'float',
):
    server = ScriptedOriginServer(
        model_id=model_id,
        response_bodies=response_bodies,
        first_delay_seconds=first_delay_seconds,
    )
    server.start()
    try:
        yield server
    finally:
        server.stop()


def run_upstream_codex_capture(
    *,
    config_path: 'Path',
    provider_name: 'str',
    output_root: 'Path',
    timeout_seconds: 'float',
    upstream_command: 'str',
    origin_base_url: 'str',
    initial_prompt: 'str',
    steer_prompt: 'str',
    upstream_origin_waiter,
) -> 'RunCapture':
    capture_root = output_root / "upstream"
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    config_root = output_root / "upstream_config"
    temp_config_path = build_proxy_config_copy(config_path, config_root)
    enable_feature_flag(temp_config_path, "steer", True)
    remove_toplevel_key(temp_config_path, "service_tier")
    proxy_url = run_proxy_capture_with_tmux(
        capture_root=capture_root,
        origin_base_url=origin_base_url,
        timeout_seconds=timeout_seconds,
        command=(
            f"env CODEX_HOME={shlex.quote(str(config_root))} "
            f"{upstream_command} --no-alt-screen"
        ),
        terminal_log_path=log_root / "upstream_tmux.log",
        initial_prompt=initial_prompt,
        steer_prompt=steer_prompt,
        prepare_proxy_url=lambda proxy_url: rewrite_config_base_url(
            temp_config_path,
            proxy_url,
        ),
        wait_for_origin_post=upstream_origin_waiter,
    )
    del proxy_url
    return load_steer_capture("upstream", capture_root, log_root / "upstream_tmux.log")


def run_pycodex_capture(
    *,
    config_path: 'Path',
    output_root: 'Path',
    timeout_seconds: 'float',
    origin_base_url: 'str',
    initial_prompt: 'str',
    steer_prompt: 'str',
    pycodex_origin_waiter,
) -> 'RunCapture':
    capture_root = output_root / "pycodex"
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    config_root = output_root / "config"
    temp_config_path = build_proxy_config_copy(config_path, config_root)
    enable_feature_flag(temp_config_path, "steer", True)
    remove_toplevel_key(temp_config_path, "service_tier")
    proxy_url = run_proxy_capture_with_tmux(
        capture_root=capture_root,
        origin_base_url=origin_base_url,
        timeout_seconds=timeout_seconds,
        command=(
            f"uv run pycodex --config {shlex.quote(str(temp_config_path))}"
        ),
        terminal_log_path=log_root / "pycodex_tmux.log",
        initial_prompt=initial_prompt,
        steer_prompt=steer_prompt,
        prepare_proxy_url=lambda proxy_url: rewrite_config_base_url(
            temp_config_path,
            proxy_url,
        ),
        wait_for_origin_post=pycodex_origin_waiter,
    )
    del proxy_url
    return load_steer_capture("pycodex", capture_root, log_root / "pycodex_tmux.log")


def run_proxy_capture_with_tmux(
    *,
    capture_root: 'Path',
    origin_base_url: 'str',
    timeout_seconds: 'float',
    command: 'str',
    terminal_log_path: 'Path',
    initial_prompt: 'str',
    steer_prompt: 'str',
    prepare_proxy_url,
    wait_for_origin_post,
) -> 'str':
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
    if prepare_proxy_url is not None:
        prepare_proxy_url(proxy_url)

    session_name = "codex_steer_" + random_suffix()
    final_command = command.replace(proxy_url_placeholder(), proxy_url)

    try:
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session_name, final_command],
            check=True,
        )
        time.sleep(5.0)
        tmux_send_line(session_name, initial_prompt)
        wait_for_origin_post(1, timeout_seconds)
        time.sleep(0.5)
        tmux_send_line(session_name, steer_prompt)
        wait_for_post_count(capture_root, 2, timeout_seconds)
        time.sleep(0.5)
        terminal_log_path.write_text(tmux_capture_pane(session_name))
    finally:
        subprocess.run(["tmux", "kill-session", "-t", session_name], check=False)
        proxy.shutdown()
        proxy_thread.join(timeout=5)
        proxy.server_close()
    return proxy_url


def load_steer_capture(label: 'str', capture_root: 'Path', terminal_log_path: 'Path') -> 'RunCapture':
    request_files = sorted(capture_root.glob("*_POST_*.json"))
    if len(request_files) < 2:
        raise RuntimeError(
            f"expected at least 2 POST captures for {label} under {capture_root}"
        )
    first_path, second_path = request_files[:2]
    first_capture = json.loads(first_path.read_text())
    second_capture = json.loads(second_path.read_text())
    return RunCapture(
        label=label,
        first=CapturedRequest(
            path=first_path,
            body=first_capture["body"],
            headers=first_capture["headers"],
        ),
        second=CapturedRequest(
            path=second_path,
            body=second_capture["body"],
            headers=second_capture["headers"],
        ),
        terminal_log_path=terminal_log_path,
    )


def build_comparison(upstream: 'RunCapture', pycodex: 'RunCapture') -> 'typing.Dict[str, object]':
    upstream_meta = extract_turn_metadata(upstream)
    pycodex_meta = extract_turn_metadata(pycodex)
    first_upstream_body = normalize_body_for_compare(upstream.first.body)
    first_pycodex_body = normalize_body_for_compare(pycodex.first.body)
    second_upstream_body = normalize_body_for_compare(upstream.second.body)
    second_pycodex_body = normalize_body_for_compare(pycodex.second.body)
    return {
        "upstream": {
            "first_path": str(upstream.first.path),
            "second_path": str(upstream.second.path),
            "first_turn_metadata": upstream_meta[0],
            "second_turn_metadata": upstream_meta[1],
            "same_turn_id_across_requests": same_turn_id(upstream_meta),
            "terminal_log_path": str(upstream.terminal_log_path),
        },
        "pycodex": {
            "first_path": str(pycodex.first.path),
            "second_path": str(pycodex.second.path),
            "first_turn_metadata": pycodex_meta[0],
            "second_turn_metadata": pycodex_meta[1],
            "same_turn_id_across_requests": same_turn_id(pycodex_meta),
            "terminal_log_path": str(pycodex.terminal_log_path),
        },
        "first": {
            "equal_ignoring_prompt_cache_key": first_upstream_body == first_pycodex_body,
            "diffs": diff_values(first_upstream_body, first_pycodex_body),
        },
        "second": {
            "equal_ignoring_prompt_cache_key": (
                second_upstream_body == second_pycodex_body
            ),
            "diffs": diff_values(second_upstream_body, second_pycodex_body),
        },
    }


def normalize_body_for_compare(body: 'typing.Dict[str, object]') -> 'typing.Dict[str, object]':
    normalized = json.loads(json.dumps(body))
    normalized.pop("prompt_cache_key", None)
    return normalized


def extract_turn_metadata(capture: 'RunCapture') -> 'typing.Tuple[typing.Dict[str, object], typing.Dict[str, object]]':
    first = json.loads(capture.first.headers["x-codex-turn-metadata"])
    second = json.loads(capture.second.headers["x-codex-turn-metadata"])
    return first, second


def same_turn_id(metadata_pair: 'typing.Tuple[typing.Dict[str, object], typing.Dict[str, object]]') -> 'bool':
    first, second = metadata_pair
    first_turn_id = str(first.get("turn_id", "")).strip()
    second_turn_id = str(second.get("turn_id", "")).strip()
    return bool(first_turn_id) and first_turn_id == second_turn_id


def diff_values(left, right, path: 'str' = "body") -> 'typing.List[str]':
    diffs: 'typing.List[str]' = []
    if type(left) is not type(right):
        return [f"{path}: type {type(left).__name__} != {type(right).__name__}"]
    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        for key in sorted(left_keys - right_keys):
            diffs.append(f"{path}.{key}: missing on right")
        for key in sorted(right_keys - left_keys):
            diffs.append(f"{path}.{key}: missing on left")
        for key in sorted(left_keys & right_keys):
            diffs.extend(diff_values(left[key], right[key], f"{path}.{key}"))
        return diffs
    if isinstance(left, list):
        if len(left) != len(right):
            diffs.append(f"{path}: length {len(left)} != {len(right)}")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=False)):
            diffs.extend(diff_values(left_item, right_item, f"{path}[{index}]"))
        return diffs
    if left != right:
        diffs.append(f"{path}: {left!r} != {right!r}")
    return diffs


def enable_feature_flag(config_path: 'Path', feature_name: 'str', enabled: 'bool') -> 'None':
    raw_text = config_path.read_text()
    feature_line = f"{feature_name} = {'true' if enabled else 'false'}"
    section_pattern = re.compile(r"(?ms)(^\[features\]\s*$)(.*?)(?=^\[|\Z)")
    feature_pattern = re.compile(rf"(?m)^{re.escape(feature_name)}\s*=\s*(true|false)\s*$")
    match = section_pattern.search(raw_text)
    if match is None:
        suffix = "\n" if raw_text.endswith("\n") else "\n\n"
        config_path.write_text(raw_text + f"{suffix}[features]\n{feature_line}\n")
        return

    header = match.group(1)
    body = match.group(2)
    replaced_body, count = feature_pattern.subn(feature_line, body, count=1)
    if count == 0:
        replaced_body = body + ("" if body.endswith("\n") or not body else "\n") + feature_line + "\n"
    rewritten = raw_text[: match.start()] + header + replaced_body + raw_text[match.end() :]
    config_path.write_text(rewritten)


def remove_toplevel_key(config_path: 'Path', key: 'str') -> 'None':
    raw_text = config_path.read_text()
    pattern = re.compile(rf"(?m)^{re.escape(key)}\s*=.*\n?")
    rewritten, _count = pattern.subn("", raw_text, count=1)
    config_path.write_text(rewritten)


def proxy_url_placeholder() -> 'str':
    return "__PYCODEX_PROXY_BASE_URL__"


def random_suffix(length: 'int' = 8) -> 'str':
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


if __name__ == "__main__":
    main()
