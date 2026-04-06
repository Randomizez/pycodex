from __future__ import annotations

import importlib.metadata
import os
import platform
import re
from datetime import datetime
from pathlib import Path
import subprocess


def get_shell_name() -> str:
    shell_path = os.environ.get("SHELL")
    if shell_path:
        return Path(shell_path).name or shell_path
    return "bash"


def get_timezone_name() -> str:
    timezone_env = os.environ.get("TZ")
    if timezone_env:
        return timezone_env
    zoneinfo_root = Path("/usr/share/zoneinfo")
    localtime = Path("/etc/localtime")
    try:
        resolved = localtime.resolve()
    except OSError:
        resolved = None
    if resolved is not None and zoneinfo_root in resolved.parents:
        return str(resolved.relative_to(zoneinfo_root))
    timezone = datetime.now().astimezone().tzinfo
    if timezone is None:
        return "Etc/UTC"
    name = str(timezone)
    return name or "Etc/UTC"
def get_sandbox_tag(sandbox_mode: str | None) -> str:
    if sandbox_mode == "danger-full-access":
        return "none"
    if sandbox_mode == "read-only":
        return "read-only"
    if sandbox_mode == "workspace-write":
        return "workspace-write"
    return "none"


def get_workspace_turn_metadata(cwd: str | Path) -> dict[str, object] | None:
    resolved_cwd = Path(cwd).resolve()
    repo_root = _git_output(
        resolved_cwd,
        ["rev-parse", "--show-toplevel"],
    )
    if repo_root is None:
        return None

    workspace: dict[str, object] = {}
    head = _git_output(resolved_cwd, ["rev-parse", "HEAD"])
    if head is not None:
        workspace["latest_git_commit_hash"] = head

    remotes = _git_remote_urls(resolved_cwd)
    if remotes:
        workspace["associated_remote_urls"] = remotes

    has_changes = _git_has_changes(resolved_cwd)
    if has_changes is not None:
        workspace["has_changes"] = has_changes

    if not workspace:
        return None
    return {"workspaces": {repo_root: workspace}}


def build_user_agent(originator: str) -> str:
    version = get_package_version()
    terminal = get_terminal_user_agent_token()
    os_name, os_version = get_os_info()
    arch = platform.machine() or "unknown"
    suffix = _user_agent_suffix(originator, version)
    return f"{originator}/{version} ({os_name} {os_version}; {arch}) {terminal}{suffix}"


def get_package_version() -> str:
    detected = _detect_upstream_codex_version()
    if detected is not None:
        return detected
    for distribution_name in ("python-codex", "pycodex"):
        try:
            return importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    local_version = _read_local_package_version()
    if local_version is not None:
        return local_version
    return "0.1.0"


def get_os_info() -> tuple[str, str]:
    os_release = Path("/etc/os-release")
    if os_release.is_file():
        values: dict[str, str] = {}
        for line in os_release.read_text().splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
        name = values.get("NAME")
        version = values.get("VERSION_ID")
        if name and version:
            return name, _normalize_os_version(version)
    return platform.system(), platform.release()


def get_terminal_user_agent_token() -> str:
    term_program = os.environ.get("TERM_PROGRAM", "")
    if term_program.lower() == "tmux":
        client_termname = _tmux_display_message("#{client_termname}")
        if client_termname:
            return _sanitize_header_token(client_termname)

    term = os.environ.get("TERM")
    if term:
        return _sanitize_header_token(term)
    return "unknown"


def _git_output(cwd: Path, args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _git_remote_urls(cwd: Path) -> dict[str, str]:
    remote_names = _git_output(cwd, ["remote"])
    if remote_names is None:
        return {}
    remotes: dict[str, str] = {}
    for name in remote_names.splitlines():
        remote_name = name.strip()
        if not remote_name:
            continue
        url = _git_output(cwd, ["remote", "get-url", remote_name])
        if url is not None:
            remotes[remote_name] = url
    return remotes


def _git_has_changes(cwd: Path) -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(completed.stdout.strip())


def _user_agent_suffix(originator: str, version: str) -> str:
    if originator == "codex_exec":
        return f" (codex-exec; {version})"
    if originator == "codex-tui":
        return f" (codex-tui; {version})"
    return ""


def _normalize_os_version(version: str) -> str:
    parts = version.split(".")
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        major, minor = parts
        return f"{int(major)}.{int(minor)}.0"
    return version


def _read_local_package_version() -> str | None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject_path.is_file():
        return None
    match = re.search(
        r'^\s*version\s*=\s*"([^"]+)"\s*$',
        pyproject_path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if match is None:
        return None
    return match.group(1).strip() or None


def _tmux_display_message(fmt: str) -> str | None:
    try:
        output = subprocess.run(
            ["tmux", "display-message", "-p", fmt],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = output.stdout.strip()
    return value or None


def _sanitize_header_token(value: str) -> str:
    return "".join(
        character
        if (character.isalnum() or character in {"-", "_", ".", "/"})
        else "_"
        for character in value
    )


def _detect_upstream_codex_version() -> str | None:
    try:
        output = subprocess.run(
            ["codex", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    match = re.search(r"\b(\d+\.\d+\.\d+)\b", output.stdout)
    if match is None:
        return None
    return match.group(1)
