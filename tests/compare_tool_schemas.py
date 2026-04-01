"""Compare captured tool schemas between upstream Codex and `pycodex`.

This helper uses the local proxy capture server in `tests/fake_responses_server.py`
to record one real upstream Codex request and one `pycodex` request under the
same provider configuration, then compares the request-visible tool schemas in
the order recorded by `tests/TESTS.md`.

It is intentionally a support script under `tests/`, not an always-on pytest,
because it depends on a real local Codex CLI setup and a reachable upstream
Responses provider.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path

from tests.fake_responses_server import CaptureStore, build_proxy_handler

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


DEFAULT_CONFIG_PATH = Path.home() / ".codex" / "config.toml"
DEFAULT_OUTPUT_ROOT = Path(".tmp") / "tool_schema_proxy_compare"
DEFAULT_PROMPT = "Reply with exactly OK. Do not call any tools."
DEFAULT_TIMEOUT_SECONDS = 180.0
TESTS_MD_PATH = Path(__file__).with_name("TESTS.md")


@dataclass(frozen=True, slots=True)
class RunCapture:
    label: str
    request_path: Path
    request_body: dict[str, object]
    request_headers: dict[str, str]
    tool_map: dict[str, dict[str, object]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uv run python tests/compare_tool_schemas.py",
        description=(
            "Capture one upstream Codex request and one pycodex request through "
            "the local proxy server, then compare the request-visible tool schemas."
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
        help="Prompt sent to both CLIs. The first outbound request is compared.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Timeout applied to each CLI invocation and the proxy upstream forwarding.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    output_root = Path(args.root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    provider_name, upstream_base_url = load_provider_info(config_path)
    tool_names = load_tool_names_from_tests_md(TESTS_MD_PATH)

    upstream_capture = run_upstream_codex_capture(
        config_path=config_path,
        provider_name=provider_name,
        upstream_base_url=upstream_base_url,
        output_root=output_root,
        prompt=args.prompt,
        timeout_seconds=args.timeout_seconds,
    )
    pycodex_capture = run_pycodex_capture(
        config_path=config_path,
        upstream_base_url=upstream_base_url,
        output_root=output_root,
        prompt=args.prompt,
        timeout_seconds=args.timeout_seconds,
    )

    comparison_rows = compare_tool_maps(
        tool_names,
        upstream_capture.tool_map,
        pycodex_capture.tool_map,
    )

    comparison_payload = {
        "tool_names": tool_names,
        "upstream_request": str(upstream_capture.request_path),
        "pycodex_request": str(pycodex_capture.request_path),
        "rows": comparison_rows,
    }
    comparison_path = output_root / "comparison.json"
    comparison_path.write_text(
        json.dumps(comparison_payload, ensure_ascii=False, indent=2)
    )

    print(f"Upstream request: {upstream_capture.request_path}")
    print(f"pycodex request:  {pycodex_capture.request_path}")
    print(f"Comparison JSON:  {comparison_path}")
    print()
    print("| tool name | upstream | pycodex | equal | note |")
    print("|---|---|---|---|---|")
    for row in comparison_rows:
        print(
            "| {tool_name} | {upstream_present} | {pycodex_present} | {equal} | {note} |".format(
                **row
            )
        )


def load_provider_info(config_path: Path) -> tuple[str, str]:
    data = tomllib.loads(config_path.read_text())
    provider_name = str(data["model_provider"])
    provider = data["model_providers"][provider_name]
    upstream_base_url = str(provider["base_url"])
    return provider_name, upstream_base_url


def load_tool_names_from_tests_md(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    tool_names: list[str] = []
    inside_table = False
    for line in lines:
        if line.strip() == "| tool name | test prompt | expected behavior |":
            inside_table = True
            continue
        if not inside_table:
            continue
        stripped = line.strip()
        if not stripped:
            break
        if stripped.startswith("|---|"):
            continue
        if not stripped.startswith("|"):
            break
        parts = [part.strip() for part in stripped.strip("|").split("|")]
        if len(parts) < 3:
            break
        tool_name = parts[0].strip("`")
        if tool_name:
            tool_names.append(tool_name)
    if not tool_names:
        raise RuntimeError(f"failed to parse tool names from {path}")
    return tool_names


def run_upstream_codex_capture(
    config_path: Path,
    provider_name: str,
    upstream_base_url: str,
    output_root: Path,
    prompt: str,
    timeout_seconds: float,
) -> RunCapture:
    capture_root = output_root / "upstream"
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    proxy_url = run_proxy_capture(
        capture_root=capture_root,
        upstream_base_url=upstream_base_url,
        timeout_seconds=timeout_seconds,
        command=build_upstream_codex_command(provider_name, prompt),
        command_log_path=log_root / "upstream_codex.log",
        cwd=Path.cwd(),
        placeholder_resolver=lambda _: None,
    )
    del proxy_url
    return load_first_post_capture("upstream", capture_root)


def run_pycodex_capture(
    config_path: Path,
    upstream_base_url: str,
    output_root: Path,
    prompt: str,
    timeout_seconds: float,
) -> RunCapture:
    capture_root = output_root / "pycodex"
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    temp_config_path = build_proxy_config_copy(config_path, output_root / "config")

    def resolve_proxy_url(proxy_url: str) -> None:
        rewrite_config_base_url(temp_config_path, proxy_url)

    proxy_url = run_proxy_capture(
        capture_root=capture_root,
        upstream_base_url=upstream_base_url,
        timeout_seconds=timeout_seconds,
        command=[
            "uv",
            "run",
            "pycodex",
            "--config",
            str(temp_config_path),
            prompt,
        ],
        command_log_path=log_root / "pycodex.log",
        cwd=Path.cwd(),
        placeholder_resolver=resolve_proxy_url,
    )
    del proxy_url
    return load_first_post_capture("pycodex", capture_root)


def run_proxy_capture(
    capture_root: Path,
    upstream_base_url: str,
    timeout_seconds: float,
    command: list[str],
    command_log_path: Path,
    cwd: Path,
    placeholder_resolver,
) -> str:
    if capture_root.exists():
        shutil.rmtree(capture_root)
    capture_store = CaptureStore(capture_root)
    proxy_httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        build_proxy_handler(capture_store, upstream_base_url, timeout_seconds),
    )
    proxy_thread = threading.Thread(target=proxy_httpd.serve_forever, daemon=True)
    proxy_thread.start()

    proxy_url = f"http://127.0.0.1:{proxy_httpd.server_port}/v1"
    placeholder_resolver(proxy_url)
    final_command = [
        argument.replace(proxy_url_placeholder(), proxy_url) for argument in command
    ]

    timed_out = False
    saw_first_post = False
    started_at = time.monotonic()
    process = subprocess.Popen(
        final_command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = started_at + timeout_seconds
        while True:
            if list(capture_root.glob("*_POST_*.json")):
                saw_first_post = True
                break
            if process.poll() is not None:
                break
            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(0.1)
    finally:
        if process.poll() is None and (saw_first_post or timed_out):
            terminate_process_group(process.pid)
        try:
            stdout, stderr = process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            terminate_process_group(process.pid, force=True)
            stdout, stderr = process.communicate(timeout=5.0)
        proxy_httpd.shutdown()
        proxy_thread.join(timeout=5)
        proxy_httpd.server_close()

    if timed_out and not saw_first_post:
        command_log_path.write_text(
            json.dumps(
                {
                    "command": final_command,
                    "timeout_seconds": timeout_seconds,
                    "stdout": stdout,
                    "stderr": stderr,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise RuntimeError(
            f"command timed out: {' '.join(final_command)}\n"
            f"see log: {command_log_path}"
        )

    command_log_path.write_text(
        json.dumps(
            {
                "command": final_command,
                "returncode": process.returncode,
                "terminated_after_first_post": saw_first_post,
                "stdout": stdout,
                "stderr": stderr,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not saw_first_post:
        raise RuntimeError(
            f"no POST capture found after command: {' '.join(final_command)}\n"
            f"see log: {command_log_path}"
        )
    return proxy_url


def proxy_url_placeholder() -> str:
    return "__PYCODEX_PROXY_BASE_URL__"


def build_upstream_codex_command(provider_name: str, prompt: str) -> list[str]:
    inner_command = shlex.join(
        [
            "codex",
            "--no-alt-screen",
            "-c",
            f'model_providers.{provider_name}.base_url="{proxy_url_placeholder()}"',
            prompt,
        ]
    )
    return ["script", "-qfec", inner_command, "/dev/null"]


def build_proxy_config_copy(config_path: Path, config_root: Path) -> Path:
    if config_root.exists():
        shutil.rmtree(config_root)
    config_root.mkdir(parents=True, exist_ok=True)
    target_config_path = config_root / "config.toml"
    target_config_path.write_text(config_path.read_text())

    dotenv_path = config_path.parent / ".env"
    if dotenv_path.exists():
        shutil.copy2(dotenv_path, config_root / ".env")
    agents_path = config_path.parent / "AGENTS.md"
    if agents_path.exists():
        target_agents_path = config_root / "AGENTS.md"
        if target_agents_path.exists() or target_agents_path.is_symlink():
            target_agents_path.unlink()
        target_agents_path.symlink_to(agents_path)
    skills_path = config_path.parent / "skills"
    if skills_path.exists():
        target_skills_path = config_root / "skills"
        if target_skills_path.exists() or target_skills_path.is_symlink():
            if target_skills_path.is_symlink() or target_skills_path.is_file():
                target_skills_path.unlink()
            else:
                shutil.rmtree(target_skills_path)
        target_skills_path.symlink_to(skills_path, target_is_directory=True)
    return target_config_path


def rewrite_config_base_url(config_path: Path, proxy_url: str) -> None:
    data = tomllib.loads(config_path.read_text())
    provider_name = str(data["model_provider"])
    section_pattern = re.compile(
        rf"(?ms)(^\[model_providers\.{re.escape(provider_name)}\]\s*$)(.*?)(?=^\[|\Z)"
    )
    base_url_pattern = re.compile(r'(?m)^base_url\s*=\s*".*?"\s*$')
    raw_text = config_path.read_text()
    match = section_pattern.search(raw_text)
    if match is None:
        raise RuntimeError(f"provider section not found for {provider_name}")
    header = match.group(1)
    body = match.group(2)
    replaced_body, count = base_url_pattern.subn(
        f'base_url = "{proxy_url}"',
        body,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"base_url entry not found for provider {provider_name}")
    rewritten = raw_text[: match.start()] + header + replaced_body + raw_text[match.end() :]
    config_path.write_text(rewritten)


def load_first_post_capture(label: str, capture_root: Path) -> RunCapture:
    request_files = sorted(capture_root.glob("*_POST_*.json"))
    if not request_files:
        raise RuntimeError(f"no POST capture found for {label} under {capture_root}")
    request_path = request_files[0]
    capture = json.loads(request_path.read_text())
    request_body = capture["body"]
    request_headers = capture["headers"]
    tools = request_body.get("tools", [])
    tool_map: dict[str, dict[str, object]] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_key = str(tool.get("name") or tool.get("type") or "")
        if tool_key:
            tool_map[tool_key] = tool
    return RunCapture(
        label=label,
        request_path=request_path,
        request_body=request_body,
        request_headers=request_headers,
        tool_map=tool_map,
    )


def compare_tool_maps(
    tool_names: list[str],
    upstream_map: dict[str, dict[str, object]],
    pycodex_map: dict[str, dict[str, object]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for tool_name in tool_names:
        upstream_schema = upstream_map.get(tool_name)
        pycodex_schema = pycodex_map.get(tool_name)
        upstream_present = upstream_schema is not None
        pycodex_present = pycodex_schema is not None
        equal = upstream_schema == pycodex_schema if upstream_present and pycodex_present else False
        if upstream_present and pycodex_present:
            note = "same" if equal else schema_note(upstream_schema, pycodex_schema)
        elif upstream_present:
            note = "missing in pycodex captured request"
        elif pycodex_present:
            note = "missing in upstream captured request"
        else:
            note = "not exposed by the compared captured path"
        rows.append(
            {
                "tool_name": tool_name,
                "upstream_present": "yes" if upstream_present else "no",
                "pycodex_present": "yes" if pycodex_present else "no",
                "equal": "yes" if equal else ("n/a" if not upstream_present or not pycodex_present else "no"),
                "note": note,
            }
        )
    return rows


def schema_note(
    upstream_schema: dict[str, object],
    pycodex_schema: dict[str, object],
) -> str:
    upstream_keys = sorted(str(key) for key in upstream_schema.keys())
    pycodex_keys = sorted(str(key) for key in pycodex_schema.keys())
    if upstream_keys != pycodex_keys:
        return (
            "different keys: upstream="
            f"{','.join(upstream_keys)}; pycodex={','.join(pycodex_keys)}"
        )
    return "same keys, different values"


def terminate_process_group(pid: int, force: bool = False) -> None:
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        return


if __name__ == "__main__":
    main()
