"""Live side-by-side wire parity tests: official `agy 1.5.13` vs `pyagy`."""

import base64
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gzip
import difflib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import typing

import pytest

from pyagy.cli import main as pyagy_main
from pyagy.models import AGY_USER_AGENT

AGY_BIN = Path(shutil.which("agy") or (Path.home() / ".local" / "bin" / "agy"))

PNG_1X1_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)

UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)
LOCAL_ISO_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2})"
)
UTC_ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")

REQUIRED_WIRE_HEADERS = (
    "content-type",
    "user-agent",
    "x-goog-api-client",
    "accept-encoding",
    "x-goog-api-key",
)


def encode_sse_frames(
    frames: "typing.Sequence[typing.Sequence[typing.Dict[str, object]]]",
    gzip_compress: "bool" = False,
    prompt_tokens: "int" = 12,
    candidates_tokens: "int" = 6,
) -> "bytes":
    chunks: "typing.List[str]" = []
    total = len(frames)
    for idx, parts in enumerate(frames):
        candidate: "typing.Dict[str, object]" = {
            "content": {"role": "model", "parts": list(parts)},
        }
        payload: "typing.Dict[str, object]" = {"candidates": [candidate]}
        if idx == total - 1:
            candidate["finishReason"] = "STOP"
            payload["usageMetadata"] = {
                "promptTokenCount": prompt_tokens,
                "candidatesTokenCount": candidates_tokens,
                "totalTokenCount": prompt_tokens + candidates_tokens,
            }
        chunks.append(
            "data: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n\n"
        )
    raw = "".join(chunks).encode("utf-8")
    return gzip.compress(raw) if gzip_compress else raw


class MockGeminiServer:
    """Local HTTP server capturing Gemini `streamGenerateContent` requests."""

    def __init__(
        self,
        responder: "typing.Callable[[typing.Dict[str, object], bool, int], typing.Sequence[typing.Sequence[typing.Dict[str, object]]]]",
        gzip_responses: "bool" = False,
        usage_responder: "typing.Optional[typing.Callable[[typing.Dict[str, object], bool, int], typing.Tuple[int, int]]]" = None,
        before_request: "typing.Optional[typing.Callable[[typing.Dict[str, object], bool], None]]" = None,
        after_response: "typing.Optional[typing.Callable[[typing.Dict[str, object], bool, int], None]]" = None,
    ) -> "None":
        self._responder = responder
        self._gzip = gzip_responses
        self._usage_responder = usage_responder
        self._before_request = before_request
        self._after_response = after_response
        self._lock = threading.Lock()
        self._parent_idx = 0
        self._sub_idx = 0
        self.agent_requests: "typing.List[typing.Dict[str, object]]" = []
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: "str", *args: "object") -> "None":
                return

            def do_POST(self) -> "None":
                length = int(self.headers.get("Content-Length") or 0)
                raw_bytes = self.rfile.read(length)
                lower_headers = {str(k).lower(): str(v) for k, v in self.headers.items()}
                if lower_headers.get("content-encoding", "").strip().lower() == "gzip":
                    raw_bytes = gzip.decompress(raw_bytes)
                body = json.loads(raw_bytes.decode("utf-8"))

                if not body.get("tools"):
                    # Official `agy` safety-rater side-call (`tools` omitted).
                    sse = encode_sse_frames(
                        [[{"text": '{"reason":"routine","severity":0,"confidence":10}'}]]
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(sse)))
                    self.end_headers()
                    self.wfile.write(sse)
                    return

                sys_txt = "".join(
                    str(p.get("text", ""))
                    for p in (body.get("systemInstruction") or {}).get("parts", [])
                )
                is_subagent = "<subagent_reminder>" in sys_txt
                if server_ref._before_request is not None:
                    server_ref._before_request(body, is_subagent)
                with server_ref._lock:
                    if is_subagent:
                        turn_idx = server_ref._sub_idx
                        server_ref._sub_idx += 1
                    else:
                        turn_idx = server_ref._parent_idx
                        server_ref._parent_idx += 1
                    server_ref.agent_requests.append(
                        {
                            "path": self.path,
                            "headers": lower_headers,
                            "is_subagent": is_subagent,
                            "top_keys": list(body.keys()),
                            "body": body,
                        }
                    )
                    frames = server_ref._responder(body, is_subagent, turn_idx)
                    prompt_tok, cand_tok = (
                        server_ref._usage_responder(body, is_subagent, turn_idx)
                        if server_ref._usage_responder is not None
                        else (12, 6)
                    )

                sse_bytes = encode_sse_frames(
                    frames,
                    gzip_compress=server_ref._gzip,
                    prompt_tokens=prompt_tok,
                    candidates_tokens=cand_tok,
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                if server_ref._gzip:
                    self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(sse_bytes)))
                self.end_headers()
                self.wfile.write(sse_bytes)
                if server_ref._after_response is not None:
                    server_ref._after_response(body, is_subagent, turn_idx)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> "str":
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def __enter__(self) -> "MockGeminiServer":
        self._thread.start()
        return self

    def __exit__(self, exc_type: "object", exc: "object", tb: "object") -> "None":
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5.0)


def _prepare_isolated_tree(
    root_dir: "Path",
    mock_base_url: "str",
) -> "typing.Tuple[Path, Path, Path]":
    home_dir = root_dir / "home"
    work_dir = root_dir / "work"
    app_data = home_dir / ".gemini" / "antigravity-cli"
    app_data.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    (app_data / "settings.json").write_text(
        json.dumps({"modelProvider": "gemini"}),
        encoding="utf-8",
    )
    codex_dir = home_dir / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    config_toml = codex_dir / "config.toml"
    config_toml.write_text(
        'model = "gemini-4-carbon-high"\n'
        'model_provider = "gemini-mock"\n\n'
        "[model_providers.gemini-mock]\n"
        'name = "Gemini Mock"\n'
        f'base_url = "{mock_base_url}/v1beta"\n'
        'env_key = "PYAGY_MOCK_API_KEY"\n'
        'compat_api = "gemini"\n',
        encoding="utf-8",
    )
    return home_dir, work_dir, config_toml


def normalize_request_sequence(
    requests: "typing.Sequence[typing.Dict[str, object]]",
    tmp_root: "Path",
) -> "typing.List[typing.Dict[str, object]]":
    uuid_map: "typing.Dict[str, str]" = {}
    root_str = str(tmp_root.resolve())

    def _norm_str(val: "str") -> "str":
        text = val.replace(root_str, "<TMP_ROOT>")
        text = LOCAL_ISO_RE.sub("<LOCAL_TS>", text)
        text = UTC_ISO_RE.sub("<UTC_TS>", text)

        def _sub_uuid(match: "re.Match[str]") -> "str":
            raw_u = match.group(0)
            if raw_u not in uuid_map:
                uuid_map[raw_u] = f"<UUID_{len(uuid_map)}>"
            return uuid_map[raw_u]

        return UUID_RE.sub(_sub_uuid, text)

    def _norm_obj(obj: "object") -> "object":
        if isinstance(obj, str):
            return _norm_str(obj)
        if isinstance(obj, list):
            return [_norm_obj(item) for item in obj]
        if isinstance(obj, dict):
            return {k: _norm_obj(v) for k, v in obj.items()}
        return deepcopy(obj)

    normalized: "typing.List[typing.Dict[str, object]]" = []
    for req in requests:
        raw_headers = req["headers"]
        assert isinstance(raw_headers, dict)
        selected_headers = {h: raw_headers.get(h) for h in REQUIRED_WIRE_HEADERS}
        assert selected_headers["user-agent"] == AGY_USER_AGENT
        assert selected_headers["x-goog-api-client"] == AGY_USER_AGENT

        body = req["body"]
        assert isinstance(body, dict)
        fr_key_orders: "typing.List[typing.List[str]]" = []
        for content_entry in body.get("contents", []):
            if isinstance(content_entry, dict):
                for part in content_entry.get("parts", []):
                    if isinstance(part, dict) and "functionResponse" in part:
                        fr_key_orders.append(list(part["functionResponse"].keys()))

        normalized.append(
            {
                "path": req["path"],
                "headers": selected_headers,
                "is_subagent": req["is_subagent"],
                "top_keys": req["top_keys"],
                "fr_key_orders": fr_key_orders,
                "body": _norm_obj(body),
            }
        )
    return normalized


def _collect_json_diffs(
    actual: "object",
    expected: "object",
    path: "str" = "root",
    diffs: "typing.Optional[typing.List[str]]" = None,
) -> "typing.List[str]":
    out = [] if diffs is None else diffs
    if len(out) >= 20:
        return out
    if type(actual) is not type(expected):
        out.append(f"TYPE MISMATCH at {path}: pyagy={type(actual).__name__} ({actual!r:.120}) vs agy={type(expected).__name__} ({expected!r:.120})")
        return out
    if isinstance(actual, dict):
        assert isinstance(expected, dict)
        actual_keys = list(actual.keys())
        expected_keys = list(expected.keys())
        if actual_keys != expected_keys:
            missing = [k for k in expected_keys if k not in actual]
            extra = [k for k in actual_keys if k not in expected]
            out.append(f"DICT KEYS at {path}: missing_in_pyagy={missing}, extra_in_pyagy={extra}, pyagy_order={actual_keys}, agy_order={expected_keys}")
        for key in expected_keys:
            if key in actual:
                _collect_json_diffs(actual[key], expected[key], f"{path}.{key}", out)
        return out
    if isinstance(actual, list):
        assert isinstance(expected, list)
        if len(actual) != len(expected):
            out.append(f"LIST LEN at {path}: pyagy={len(actual)} vs agy={len(expected)}")
        for idx in range(min(len(actual), len(expected))):
            _collect_json_diffs(actual[idx], expected[idx], f"{path}[{idx}]", out)
        return out
    if actual != expected:
        if isinstance(actual, str) and isinstance(expected, str) and (len(actual) > 120 or "\n" in actual or "\n" in expected):
            udiff = "".join(
                difflib.unified_diff(
                    expected.splitlines(keepends=True),
                    actual.splitlines(keepends=True),
                    fromfile="agy",
                    tofile="pyagy",
                    n=2,
                )
            )
            out.append(f"STR DIFF at {path}:\n{udiff[:1500]}")
        else:
            out.append(f"VALUE DIFF at {path}: pyagy={actual!r} vs agy={expected!r}")
    return out


def assert_wire_equal(actual: "object", expected: "object") -> "None":
    diffs = _collect_json_diffs(actual, expected)
    if diffs or actual != expected:
        raise AssertionError("Wire parity mismatch:\n" + "\n".join(diffs))


def _run_official_agy(
    args: "typing.Sequence[str]",
    home_dir: "Path",
    cwd: "Path",
    mock_base_url: "str",
    stdin_text: "typing.Optional[str]" = None,
) -> "subprocess.CompletedProcess[str]":
    env = {
        "HOME": str(home_dir),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "GOOGLE_GEMINI_BASE_URL": mock_base_url,
        "GEMINI_API_KEY": "dummy",
    }
    proc = subprocess.run(
        [str(AGY_BIN), "--dangerously-skip-permissions", *args],
        input=stdin_text,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert proc.returncode == 0, f"agy failed ({proc.returncode}): {proc.stderr}\n{proc.stdout}"
    return proc


def _run_pyagy_cli(
    args: "typing.Sequence[str]",
    home_dir: "Path",
    config_toml: "Path",
    cwd: "Path",
    stdin_text: "typing.Optional[str]" = None,
) -> "str":
    app_data = home_dir / ".gemini" / "antigravity-cli"
    in_buf = io.StringIO(stdin_text or "")
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    prev_key = os.environ.get("PYAGY_MOCK_API_KEY")
    os.environ["PYAGY_MOCK_API_KEY"] = "dummy"
    try:
        rc = pyagy_main(
            argv=[
                "--config",
                str(config_toml),
                "--app-data-dir",
                str(app_data),
                *args,
            ],
            stdin=in_buf,
            stdout=out_buf,
            stderr=err_buf,
            cwd=cwd,
        )
    finally:
        if prev_key is None:
            os.environ.pop("PYAGY_MOCK_API_KEY", None)
        else:
            os.environ["PYAGY_MOCK_API_KEY"] = prev_key
    assert rc == 0, f"pyagy failed ({rc}): {err_buf.getvalue()}\n{out_buf.getvalue()}"
    return out_buf.getvalue()


pytestmark = pytest.mark.skipif(
    not AGY_BIN.is_file(),
    reason="official agy binary not found on PATH or ~/.local/bin/agy",
)


def test_first_turn_models_rules_skills_and_workspaces_parity(tmp_path: "Path") -> "None":
    def _setup_rich_workspace(root_dir: "Path", mock_url: "str") -> "typing.Tuple[Path, Path, Path, Path]":
        home_dir, work_dir, cfg = _prepare_isolated_tree(root_dir, mock_url)
        global_skill = home_dir / ".gemini" / "config" / "skills" / "global_helper"
        global_skill.mkdir(parents=True)
        (global_skill / "SKILL.md").write_text(
            "---\nname: global-helper\ndescription: Global skill description.\n---\nBody\n",
            encoding="utf-8",
        )
        repo_dir = work_dir / "repo"
        sub_dir = repo_dir / "packages" / "sub"
        extra_dir = work_dir / "extra_ws"
        (repo_dir / ".git").mkdir(parents=True)
        ((repo_dir / ".git") / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        sub_dir.mkdir(parents=True)
        extra_dir.mkdir(parents=True)

        (repo_dir / "AGENTS.md").write_text("# Root Agents Rule\nFollow root.\n", encoding="utf-8")
        (repo_dir / "GEMINI.md").write_text("# Root Gemini Rule\nBe concise.\n", encoding="utf-8")
        (sub_dir / "AGENTS.md").write_text("# Subdir Rule\nScoped rule.\n", encoding="utf-8")

        ws_skill = repo_dir / ".agents" / "skills" / "alpha"
        ws_skill.mkdir(parents=True)
        (ws_skill / "SKILL.md").write_text(
            "---\nname: alpha-skill\ndescription: Workspace alpha skill.\n---\nAlpha\n",
            encoding="utf-8",
        )
        disabled_skill = sub_dir / ".agent" / "skills" / "hidden"
        disabled_skill.mkdir(parents=True)
        (disabled_skill / "SKILL.md").write_text(
            "---\nname: hidden-skill\ndescription: Hidden.\ndisable-model-invocation: true\n---\n",
            encoding="utf-8",
        )
        return home_dir, sub_dir, extra_dir, cfg

    def _simple_responder(
        _body: "typing.Dict[str, object]",
        _is_sub: "bool",
        _idx: "int",
    ) -> "typing.Sequence[typing.Sequence[typing.Dict[str, object]]]":
        return [[{"text": "Hello back!", "thoughtSignature": "c2lnMA=="}]]

    # Run both subcases (3.1-pro-low empty workspace & 4-carbon-high rich git workspace + gzip SSE).
    def _capture_agy_empty() -> "typing.List[typing.Dict[str, object]]":
        root = tmp_path / "agy_empty"
        with MockGeminiServer(_simple_responder) as srv:
            home_d, work_d, _ = _prepare_isolated_tree(root, srv.base_url)
            _run_official_agy(["--model", "gemini-3.1-pro-low", "-p", "ping 3.1"], home_d, work_d, srv.base_url)
            return normalize_request_sequence(srv.agent_requests, root)

    def _capture_pyagy_empty() -> "typing.List[typing.Dict[str, object]]":
        root = tmp_path / "pyagy_empty"
        with MockGeminiServer(_simple_responder) as srv:
            home_d, work_d, cfg = _prepare_isolated_tree(root, srv.base_url)
            _run_pyagy_cli(["--model", "gemini-3.1-pro-low", "-p", "ping 3.1"], home_d, cfg, work_d)
            return normalize_request_sequence(srv.agent_requests, root)

    def _capture_agy_rich() -> "typing.List[typing.Dict[str, object]]":
        root = tmp_path / "agy_rich"
        with MockGeminiServer(_simple_responder, gzip_responses=True) as srv:
            home_d, sub_d, extra_d, _ = _setup_rich_workspace(root, srv.base_url)
            _run_official_agy(
                ["--model", "gemini-4-carbon-high", "--add-dir", str(extra_d), "-p", "inspect repo"],
                home_d,
                sub_d,
                srv.base_url,
            )
            return normalize_request_sequence(srv.agent_requests, root)

    def _capture_pyagy_rich() -> "typing.List[typing.Dict[str, object]]":
        root = tmp_path / "pyagy_rich"
        with MockGeminiServer(_simple_responder, gzip_responses=True) as srv:
            home_d, sub_d, extra_d, cfg = _setup_rich_workspace(root, srv.base_url)
            _run_pyagy_cli(
                ["--model", "gemini-4-carbon-high", "--add-dir", str(extra_d), "-p", "inspect repo"],
                home_d,
                cfg,
                sub_d,
            )
            return normalize_request_sequence(srv.agent_requests, root)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_agy_empty = pool.submit(_capture_agy_empty)
        f_agy_rich = pool.submit(_capture_agy_rich)
        pyagy_empty = _capture_pyagy_empty()
        pyagy_rich = _capture_pyagy_rich()
        agy_empty = f_agy_empty.result()
        agy_rich = f_agy_rich.result()

    assert_wire_equal(pyagy_empty, agy_empty)
    assert_wire_equal(pyagy_rich, agy_rich)


def test_multiturn_thoughts_signatures_parallel_errors_and_png_parity(tmp_path: "Path") -> "None":
    def _build_responder(work_dir: "Path"):
        png_path = work_dir / "dot.png"
        empty_path = work_dir / "empty.txt"
        png_path.write_bytes(PNG_1X1_BYTES)
        empty_path.write_bytes(b"")

        def _responder(
            _body: "typing.Dict[str, object]",
            _is_sub: "bool",
            idx: "int",
        ) -> "typing.Sequence[typing.Sequence[typing.Dict[str, object]]]":
            if idx == 0:
                return [
                    [
                        {"text": "Signed thought. ", "thought": True, "thoughtSignature": "dGhvdWdodDE="},
                        {"text": "Trailing thought.", "thought": True},
                        {
                            "functionCall": {
                                "id": "c_png",
                                "name": "read_file",
                                "args": {"Path": str(png_path)},
                            },
                            "thoughtSignature": "c2lnX3BuZw==",
                        },
                        {
                            "functionCall": {
                                "id": "c_empty",
                                "name": "read_file",
                                "args": {"Path": str(empty_path)},
                            }
                        },
                        {
                            "functionCall": {
                                "id": "c_err",
                                "name": "read_file",
                                "args": {"Path": "rel.txt"},
                            }
                        },
                    ]
                ]
            if idx == 1:
                # Unsigned thought before functionCall -> dropped; fc2 unsigned -> kept unsigned
                return [
                    [{"text": "Ephemeral tool thought.", "thought": True}],
                    [
                        {
                            "functionCall": {
                                "id": "fc1",
                                "name": "read_file",
                                "args": {"Path": str(empty_path)},
                            },
                            "thoughtSignature": "c2lnX2ZjMQ==",
                        },
                        {
                            "functionCall": {
                                "id": "fc2",
                                "name": "read_file",
                                "args": {"Path": "bad_rel.py"},
                            }
                        },
                    ],
                ]
            if idx == 2:
                # Completely unsigned tool-call turn -> SKIP_THOUGHT_SIGNATURE_VALIDATOR on first call
                return [
                    [
                        {
                            "functionCall": {
                                "id": "fc_unsigned_1",
                                "name": "read_file",
                                "args": {"Path": str(empty_path)},
                            }
                        },
                        {
                            "functionCall": {
                                "id": "fc_unsigned_2",
                                "name": "read_file",
                                "args": {"Path": str(empty_path)},
                            }
                        },
                    ]
                ]
            if idx == 3:
                # Multi-chunk streamed thought + multi-chunk text with signature
                return [
                    [{"text": "Final ", "thought": True}],
                    [{"text": "reasoning.", "thought": True}],
                    [{"text": "Turn 1 "}],
                    [{"text": "complete.", "thoughtSignature": "c2lnX3Qxfg=="}],
                ]
            return [[{"text": "Turn 2 complete.", "thoughtSignature": "c2lnX3Qyfg=="}]]

        return _responder

    ndjson_turns = (
        json.dumps({"event": "user", "message": {"content": "run checks"}})
        + "\n"
        + json.dumps({"event": "user", "message": {"content": "follow up"}})
        + "\n"
    )

    def _run_agy() -> "typing.List[typing.Dict[str, object]]":
        root = tmp_path / "agy_mt"
        home_d, work_d, _ = _prepare_isolated_tree(root, "http://127.0.0.1:1")
        with MockGeminiServer(_build_responder(work_d)) as srv:
            _run_official_agy(
                [
                    "--model",
                    "gemini-4-carbon-high",
                    "--input-format",
                    "stream-json",
                    "--output-format",
                    "stream-json",
                    "-p=",
                ],
                home_d,
                work_d,
                srv.base_url,
                stdin_text=ndjson_turns,
            )
            return normalize_request_sequence(srv.agent_requests, root)

    def _run_pyagy() -> "typing.List[typing.Dict[str, object]]":
        root = tmp_path / "pyagy_mt"
        home_d, work_d, cfg = _prepare_isolated_tree(root, "http://127.0.0.1:1")
        with MockGeminiServer(_build_responder(work_d)) as srv:
            _, _, cfg = _prepare_isolated_tree(root, srv.base_url)
            _run_pyagy_cli(
                [
                    "--model",
                    "gemini-4-carbon-high",
                    "--input-format",
                    "stream-json",
                    "--output-format",
                    "stream-json",
                    "-p=",
                ],
                home_d,
                cfg,
                work_d,
                stdin_text=ndjson_turns,
            )
            return normalize_request_sequence(srv.agent_requests, root)

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_run_agy)
        pyagy_reqs = _run_pyagy()
        agy_reqs = fut.result()

    assert len(pyagy_reqs) == 5
    assert_wire_equal(pyagy_reqs, agy_reqs)


def test_stateful_tools_background_tasks_and_research_subagent_parity(tmp_path: "Path") -> "None":
    def _build_responder(work_dir: "Path"):
        hello_py = work_dir / "hello.py"

        def _responder(
            body: "typing.Dict[str, object]",
            is_sub: "bool",
            idx: "int",
        ) -> "typing.Sequence[typing.Sequence[typing.Dict[str, object]]]":
            if is_sub:
                return [[{"text": "Verified hello.py looks clean.", "thoughtSignature": "c3ViX29r"}]]

            if idx == 0:
                return [
                    [
                        {
                            "functionCall": {
                                "id": "w1",
                                "name": "write_file",
                                "args": {"Path": str(hello_py), "Content": "x = 1\ny = 2\n"},
                            },
                            "thoughtSignature": "c2lnX3cx",
                        },
                        {
                            "functionCall": {
                                "id": "d1",
                                "name": "define_subagent",
                                "args": {
                                    "Name": "helper",
                                    "Description": "Helper agent",
                                    "SystemPrompt": "You are a helper.",
                                    "EnableWriteTools": False,
                                },
                            }
                        },
                    ]
                ]
            if idx == 1:
                return [
                    [
                        {
                            "functionCall": {
                                "id": "e1",
                                "name": "edit_file",
                                "args": {
                                    "Path": str(hello_py),
                                    "Target": "x = 1",
                                    "Replacement": "x = 42\nz = 99",
                                },
                            },
                            "thoughtSignature": "c2lnX2Ux",
                        },
                        {
                            "functionCall": {
                                "id": "r1",
                                "name": "read_file",
                                "args": {"Path": str(hello_py)},
                            }
                        },
                        {
                            "functionCall": {
                                "id": "sync_cmd",
                                "name": "run_command",
                                "args": {
                                    "CommandLine": "true",
                                    "WaitMsBeforeAsync": 5000,
                                    "toolSummary": "No-op command",
                                },
                            }
                        },
                        {
                            "functionCall": {
                                "id": "bg_cmd",
                                "name": "run_command",
                                "args": {
                                    "CommandLine": "sleep 2",
                                    "WaitMsBeforeAsync": 500,
                                    "toolSummary": "Background sleep",
                                },
                            }
                        },
                        {
                            "functionCall": {
                                "id": "sched1",
                                "name": "schedule",
                                "args": {
                                    "DurationSeconds": 120,
                                    "Prompt": "Later reminder",
                                },
                            }
                        },
                    ]
                ]
            if idx == 2:
                sched_task_id = ""
                for entry in body.get("contents", []):
                    if isinstance(entry, dict):
                        for part in entry.get("parts", []):
                            fr = part.get("functionResponse") if isinstance(part, dict) else None
                            if isinstance(fr, dict) and fr.get("id") == "sched1":
                                out_txt = str((fr.get("response") or {}).get("output", ""))
                                for ln in out_txt.splitlines():
                                    if "task id: " in ln:
                                        sched_task_id = ln.split("task id: ", 1)[1].strip()
                return [
                    [
                        {
                            "functionCall": {
                                "id": "mt_list",
                                "name": "manage_task",
                                "args": {"Action": "list"},
                            },
                            "thoughtSignature": "c2lnX210",
                        },
                        {
                            "functionCall": {
                                "id": "mt_kill",
                                "name": "manage_task",
                                "args": {"Action": "kill", "TaskId": sched_task_id},
                            }
                        },
                        {
                            "functionCall": {
                                "id": "inv_res",
                                "name": "invoke_subagent",
                                "args": {
                                    "TypeName": "research",
                                    "Role": "Code Checker",
                                    "Prompt": "Check hello.py",
                                },
                            }
                        },
                    ]
                ]
            if idx == 3:
                return [[{"text": "Waiting for subagent and task.", "thoughtSignature": "c2lnXzM="}]]
            if idx == 4:
                return [[{"text": "Subagent noted; waiting for sleep.", "thoughtSignature": "c2lnXzQ="}]]
            return [[{"text": "All async work finished.", "thoughtSignature": "c2lnXzU="}]]

        return _responder

    def _run_agy() -> "typing.Tuple[typing.List[typing.Dict[str, object]], typing.List[typing.Dict[str, object]]]":
        root = tmp_path / "agy_stateful"
        home_d, work_d, _ = _prepare_isolated_tree(root, "http://127.0.0.1:1")
        with MockGeminiServer(_build_responder(work_d)) as srv:
            _run_official_agy(
                ["--model", "gemini-4-carbon-high", "-p", "orchestrate tools"],
                home_d,
                work_d,
                srv.base_url,
            )
            parent_reqs = [r for r in srv.agent_requests if not r["is_subagent"]]
            sub_reqs = [r for r in srv.agent_requests if r["is_subagent"]]
            return (
                normalize_request_sequence(parent_reqs, root),
                normalize_request_sequence(sub_reqs, root),
            )

    def _run_pyagy() -> "typing.Tuple[typing.List[typing.Dict[str, object]], typing.List[typing.Dict[str, object]]]":
        root = tmp_path / "pyagy_stateful"
        home_d, work_d, _ = _prepare_isolated_tree(root, "http://127.0.0.1:1")
        with MockGeminiServer(_build_responder(work_d)) as srv:
            _, _, cfg = _prepare_isolated_tree(root, srv.base_url)
            _run_pyagy_cli(
                ["--model", "gemini-4-carbon-high", "-p", "orchestrate tools"],
                home_d,
                cfg,
                work_d,
            )
            parent_reqs = [r for r in srv.agent_requests if not r["is_subagent"]]
            sub_reqs = [r for r in srv.agent_requests if r["is_subagent"]]
            return (
                normalize_request_sequence(parent_reqs, root),
                normalize_request_sequence(sub_reqs, root),
            )

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_run_agy)
        pyagy_parent, pyagy_sub = _run_pyagy()
        agy_parent, agy_sub = fut.result()

    assert len(pyagy_sub) == 1
    assert_wire_equal(pyagy_sub, agy_sub)
    assert len(pyagy_parent) == 6
    assert_wire_equal(pyagy_parent, agy_parent)


def test_cross_process_continue_and_model_switch_parity(tmp_path: "Path") -> "None":
    def _responder(
        _body: "typing.Dict[str, object]",
        _is_sub: "bool",
        idx: "int",
    ) -> "typing.Sequence[typing.Sequence[typing.Dict[str, object]]]":
        return [
            [
                {"text": f"thought-{idx}", "thought": True, "thoughtSignature": "dGhpbms="},
                {"text": f"answer-{idx}", "thoughtSignature": "YW5z"},
            ]
        ]

    def _run_agy() -> "typing.List[typing.Dict[str, object]]":
        root = tmp_path / "agy_cont"
        with MockGeminiServer(_responder) as srv:
            home_d, work_d, _ = _prepare_isolated_tree(root, srv.base_url)
            _run_official_agy(
                ["--model", "gemini-4-carbon-high", "-p", "first question"],
                home_d,
                work_d,
                srv.base_url,
            )
            _run_official_agy(
                ["--model", "gemini-3.1-pro-low", "--continue", "-p", "second question"],
                home_d,
                work_d,
                srv.base_url,
            )
            return normalize_request_sequence(srv.agent_requests, root)

    def _run_pyagy() -> "typing.List[typing.Dict[str, object]]":
        root = tmp_path / "pyagy_cont"
        with MockGeminiServer(_responder) as srv:
            home_d, work_d, cfg = _prepare_isolated_tree(root, srv.base_url)
            _run_pyagy_cli(
                ["--model", "gemini-4-carbon-high", "-p", "first question"],
                home_d,
                cfg,
                work_d,
            )
            _run_pyagy_cli(
                ["--model", "gemini-3.1-pro-low", "--continue", "-p", "second question"],
                home_d,
                cfg,
                work_d,
            )
            return normalize_request_sequence(srv.agent_requests, root)

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_run_agy)
        pyagy_reqs = _run_pyagy()
        agy_reqs = fut.result()

    assert len(pyagy_reqs) == 2
    assert_wire_equal(pyagy_reqs, agy_reqs)


def test_compaction_auto_manual_and_subagent_to_root_message_parity(tmp_path: "Path") -> "None":
    def _make_scenario_hooks(work_dir: "Path"):
        note_file = work_dir / "note.txt"
        note_file.write_text("ok\n", encoding="utf-8")
        root_idle_after_spawn = threading.Event()
        root_idle_after_ping = threading.Event()
        root_idle_after_final = threading.Event()
        sub_seen = [0]
        sub_seen_lock = threading.Lock()

        def _before(body: "typing.Dict[str, object]", is_sub: "bool") -> "None":
            if not is_sub:
                return
            with sub_seen_lock:
                cur = sub_seen[0]
                sub_seen[0] += 1
            if cur == 0:
                root_idle_after_spawn.wait(timeout=5.0)
                time.sleep(0.08)
            elif cur == 1:
                root_idle_after_ping.wait(timeout=5.0)
                time.sleep(0.08)
            elif cur == 3:
                root_idle_after_final.wait(timeout=5.0)
                time.sleep(0.08)

        def _responder(
            body: "typing.Dict[str, object]",
            is_sub: "bool",
            idx: "int",
        ) -> "typing.Sequence[typing.Sequence[typing.Dict[str, object]]]":
            if is_sub:
                sys_txt = "".join(
                    str(p.get("text", ""))
                    for p in (body.get("systemInstruction") or {}).get("parts", [])
                )
                caller_match = UUID_RE.search(sys_txt.split("<subagent_reminder>", 1)[1])
                assert caller_match is not None
                caller_id = caller_match.group(0)
                if idx == 0:
                    return [
                        [
                            {
                                "functionCall": {
                                    "id": "sub_sm1",
                                    "name": "send_message",
                                    "args": {
                                        "Recipient": caller_id,
                                        "Message": "Progress ping from subagent",
                                    },
                                },
                                "thoughtSignature": "c3ViX3NtMQ==",
                            }
                        ]
                    ]
                if idx == 1:
                    return [
                        [
                            {
                                "text": "<summary>\nSubagent auto summary.\n</summary>",
                                "thoughtSignature": "c3ViX2NtcA==",
                            }
                        ]
                    ]
                if idx == 2:
                    return [
                        [
                            {
                                "functionCall": {
                                    "id": "sub_sm2",
                                    "name": "send_message",
                                    "args": {
                                        "Recipient": caller_id,
                                        "Message": "Final report after subagent compact",
                                    },
                                },
                                "thoughtSignature": "c3ViX3NtMg==",
                            }
                        ]
                    ]
                return [[{"text": "Subagent done.", "thoughtSignature": "c3ViX2RvbmU="}]]

            if idx == 0:
                return [
                    [
                        {
                            "functionCall": {
                                "id": "inv1",
                                "name": "invoke_subagent",
                                "args": {
                                    "TypeName": "self",
                                    "Role": "Worker",
                                    "Prompt": "Check status and ping root",
                                },
                            },
                            "thoughtSignature": "cm9vdF9pbnYx",
                        }
                    ]
                ]
            if idx == 1:
                return [[{"text": "Waiting for worker.", "thoughtSignature": "cm9vdF93YWl0"}]]
            if idx == 2:
                return [
                    [
                        {
                            "functionCall": {
                                "id": "root_cmd1",
                                "name": "run_command",
                                "args": {
                                    "CommandLine": "true",
                                    "WaitMsBeforeAsync": 5000,
                                    "toolSummary": "verify",
                                },
                            },
                            "thoughtSignature": "cm9vdF9jbWQx",
                        }
                    ]
                ]
            if idx == 3:
                return [
                    [
                        {
                            "text": "<summary>\nRoot auto summary.\n</summary>",
                            "thoughtSignature": "cm9vdF9jbXAx",
                        }
                    ]
                ]
            if idx == 4:
                return [
                    [
                        {
                            "text": "Got progress ping after root auto-compact.",
                            "thoughtSignature": "cm9vdF9waW5n",
                        }
                    ]
                ]
            if idx == 5:
                return [
                    [
                        {
                            "text": "All worker updates received.",
                            "thoughtSignature": "cm9vdF90MWRvbmU=",
                        }
                    ]
                ]
            if idx == 6:
                return [
                    [
                        {
                            "text": "<summary>\nManual bare summary.\n</summary>",
                            "thoughtSignature": "cm9vdF9tY21wMQ==",
                        }
                    ]
                ]
            if idx == 7:
                return [
                    [
                        {
                            "text": "<summary>\nManual arg summary.\n</summary>",
                            "thoughtSignature": "cm9vdF9tY21wMg==",
                        }
                    ]
                ]
            if idx == 8:
                return [[{"text": "Status summarized.", "thoughtSignature": "cm9vdF90M2RvbmU="}]]
            if idx == 9:
                return [
                    [
                        {
                            "functionCall": {
                                "id": "r_note",
                                "name": "read_file",
                                "args": {"Path": str(note_file)},
                            },
                            "thoughtSignature": "cm9vdF9ybm90ZQ==",
                        }
                    ]
                ]
            if idx == 10:
                return [
                    [
                        {
                            "text": "<summary>\nPro 3.1 compact summary.\n</summary>",
                            "thoughtSignature": "cm9vdF9wcm9fY21w",
                        }
                    ]
                ]
            return [
                [
                    {
                        "text": "Continued and compacted on 3.1 Pro.",
                        "thoughtSignature": "cm9vdF9jb250X2RvbmU=",
                    }
                ]
            ]

        def _usage(
            _body: "typing.Dict[str, object]",
            is_sub: "bool",
            idx: "int",
        ) -> "typing.Tuple[int, int]":
            if is_sub and idx == 0:
                return (512000, 50)
            if not is_sub and idx == 2:
                return (512000, 80)
            if not is_sub and idx == 9:
                return (256000, 40)
            return (12, 6)

        def _after(_body: "typing.Dict[str, object]", is_sub: "bool", idx: "int") -> "None":
            if not is_sub:
                if idx == 1:
                    root_idle_after_spawn.set()
                elif idx == 4:
                    root_idle_after_ping.set()
                elif idx == 5:
                    root_idle_after_final.set()

        return _responder, _usage, _before, _after

    ndjson_in = (
        json.dumps({"event": "user", "message": {"content": "start task with worker"}})
        + "\n"
        + json.dumps({"event": "user", "message": {"content": "/compact"}})
        + "\n"
        + json.dumps({"event": "user", "message": {"content": "/compact now summarize status"}})
        + "\n"
    )
    stream_args = [
        "--model",
        "gemini-4-carbon-high",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "-p=",
    ]
    cont_args = [
        "--model",
        "gemini-3.1-pro-low",
        "--continue",
        "-p",
        "final check after continue",
    ]

    def _run_agy() -> "typing.Tuple[typing.List[typing.Dict[str, object]], typing.List[typing.Dict[str, object]], str, str]":
        root = tmp_path / "agy_cmp"
        home_d, work_d, _ = _prepare_isolated_tree(root, "http://127.0.0.1:1")
        resp_fn, usage_fn, before_fn, after_fn = _make_scenario_hooks(work_d)
        with MockGeminiServer(
            resp_fn,
            usage_responder=usage_fn,
            before_request=before_fn,
            after_response=after_fn,
        ) as srv:
            p1 = _run_official_agy(stream_args, home_d, work_d, srv.base_url, stdin_text=ndjson_in)
            p2 = _run_official_agy(cont_args, home_d, work_d, srv.base_url)
            parent_reqs = [r for r in srv.agent_requests if not r["is_subagent"]]
            sub_reqs = [r for r in srv.agent_requests if r["is_subagent"]]
            return (
                normalize_request_sequence(parent_reqs, root),
                normalize_request_sequence(sub_reqs, root),
                p1.stdout,
                p2.stdout,
            )

    def _run_pyagy() -> "typing.Tuple[typing.List[typing.Dict[str, object]], typing.List[typing.Dict[str, object]], str, str]":
        root = tmp_path / "pyagy_cmp"
        home_d, work_d, _ = _prepare_isolated_tree(root, "http://127.0.0.1:1")
        resp_fn, usage_fn, before_fn, after_fn = _make_scenario_hooks(work_d)
        with MockGeminiServer(
            resp_fn,
            usage_responder=usage_fn,
            before_request=before_fn,
            after_response=after_fn,
        ) as srv:
            _, _, cfg = _prepare_isolated_tree(root, srv.base_url)
            out1 = _run_pyagy_cli(stream_args, home_d, cfg, work_d, stdin_text=ndjson_in)
            out2 = _run_pyagy_cli(cont_args, home_d, cfg, work_d)
            parent_reqs = [r for r in srv.agent_requests if not r["is_subagent"]]
            sub_reqs = [r for r in srv.agent_requests if r["is_subagent"]]
            return (
                normalize_request_sequence(parent_reqs, root),
                normalize_request_sequence(sub_reqs, root),
                out1,
                out2,
            )

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_run_agy)
        pyagy_parent, pyagy_sub, pyagy_out1, pyagy_out2 = _run_pyagy()
        agy_parent, agy_sub, _agy_out1, agy_out2 = fut.result()

    assert len(pyagy_sub) == 4
    assert_wire_equal(pyagy_sub, agy_sub)
    assert len(pyagy_parent) == 12
    assert_wire_equal(pyagy_parent, agy_parent)
    assert pyagy_out2 == agy_out2 == "Continued and compacted on 3.1 Pro.\n"

    events = [json.loads(ln) for ln in pyagy_out1.strip().splitlines() if ln.strip()]
    checkpoint_updates = [
        ev["step_update"]
        for ev in events
        if ev.get("event") == "step_update"
        and ev.get("step_update", {}).get("step_type") == "checkpoint"
    ]
    assert len(checkpoint_updates) == 3
    result_responses = [
        ev["result"]["response"] for ev in events if ev.get("event") == "result"
    ]
    assert result_responses == [
        "All worker updates received.\n",
        "",
        "Status summarized.\n",
    ]


def test_oversized_sync_and_background_run_command_truncation_parity(tmp_path: "Path") -> "None":
    sync_cmd = (
        "python3 -c 'import sys; "
        'sys.stdout.write("SYNC_HEAD_" + "A" * 8000 + "中文段落" * 600 + "_SYNC_TAIL\\n")\''
    )
    bg_cmd = (
        "sleep 0.7 && python3 -c 'import sys; "
        'sys.stdout.write("BG_HEAD_" + "B" * 9000 + "异步输出" * 600 + "_BG_TAIL\\n")\''
    )

    def _responder(
        _body: "typing.Dict[str, object]",
        _is_sub: "bool",
        idx: "int",
    ) -> "typing.Sequence[typing.Sequence[typing.Dict[str, object]]]":
        if idx == 0:
            return [
                [
                    {
                        "functionCall": {
                            "id": "cmd_sync_big",
                            "name": "run_command",
                            "args": {
                                "CommandLine": sync_cmd,
                                "WaitMsBeforeAsync": 5000,
                                "toolSummary": "Sync big output",
                            },
                        },
                        "thoughtSignature": "c2lnX3N5bmM=",
                    },
                    {
                        "functionCall": {
                            "id": "cmd_bg_big",
                            "name": "run_command",
                            "args": {
                                "CommandLine": bg_cmd,
                                "WaitMsBeforeAsync": 500,
                                "toolSummary": "Background big output",
                            },
                        },
                    },
                ]
            ]
        if idx == 1:
            return [[{"text": "Waiting for background task.", "thoughtSignature": "c2lnX3dhaXQ="}]]
        return [[{"text": "Both truncated outputs verified.", "thoughtSignature": "c2lnX2RvbmU="}]]

    def _run_agy() -> "typing.List[typing.Dict[str, object]]":
        root = tmp_path / "agy_trunc"
        home_d, work_d, _ = _prepare_isolated_tree(root, "http://127.0.0.1:1")
        with MockGeminiServer(_responder) as srv:
            _run_official_agy(
                ["--model", "gemini-4-carbon-high", "-p", "run oversized commands"],
                home_d,
                work_d,
                srv.base_url,
            )
            return normalize_request_sequence(srv.agent_requests, root)

    def _run_pyagy() -> "typing.List[typing.Dict[str, object]]":
        root = tmp_path / "pyg_trunc"
        home_d, work_d, _ = _prepare_isolated_tree(root, "http://127.0.0.1:1")
        with MockGeminiServer(_responder) as srv:
            _, _, cfg = _prepare_isolated_tree(root, srv.base_url)
            _run_pyagy_cli(
                ["--model", "gemini-4-carbon-high", "-p", "run oversized commands"],
                home_d,
                cfg,
                work_d,
            )
            return normalize_request_sequence(srv.agent_requests, root)

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_run_agy)
        pyagy_reqs = _run_pyagy()
        agy_reqs = fut.result()

    assert len(pyagy_reqs) == 3
    assert_wire_equal(pyagy_reqs, agy_reqs)
