import os
import platform
import shutil
import subprocess
import typing
from datetime import datetime
from pathlib import Path, PureWindowsPath


def resolve_default_shell_path(default_posix: "str" = "/bin/bash") -> "str":
    shell_path = (os.environ.get("SHELL") or "").strip()
    if shell_path:
        if os.name != "nt":
            return shell_path
        if not shell_path.startswith("/") or os.path.isfile(shell_path):
            return shell_path
        win_equiv = shutil.which(PureWindowsPath(shell_path).name)
        if win_equiv:
            return win_equiv
    if os.name == "nt":
        for candidate in ("powershell.exe", "powershell", "pwsh.exe", "pwsh"):
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        return os.environ.get("COMSPEC") or "cmd.exe"
    return default_posix


def build_shell_command(
    cmd: "str",
    shell: "typing.Union[str, None]" = None,
    login: "bool" = True,
    default_posix: "str" = "/bin/bash",
) -> "typing.List[str]":
    shell_path = shell or resolve_default_shell_path(default_posix=default_posix)
    shell_name = (
        PureWindowsPath(shell_path).name.lower()
        if os.name == "nt"
        else Path(shell_path).name.lower()
    )
    if shell_name in {"cmd", "cmd.exe"}:
        return [shell_path, "/C", cmd]
    if "powershell" in shell_name or shell_name in {"pwsh", "pwsh.exe"}:
        return [shell_path, "-NoProfile", "-Command", cmd]
    return [shell_path, "-lc" if login else "-c", cmd]


def get_shell_name() -> "str":
    if os.name != "nt":
        shell_path = os.environ.get("SHELL")
        if shell_path:
            return Path(shell_path).name or shell_path
        return "bash"
    resolved = resolve_default_shell_path()
    name = PureWindowsPath(resolved).name or resolved
    lower = name.lower()
    if "powershell" in lower:
        return "powershell"
    if lower in {"pwsh", "pwsh.exe"}:
        return "pwsh"
    if lower in {"cmd", "cmd.exe"}:
        return "cmd"
    if lower.endswith(".exe"):
        return name[:-4]
    return name


def get_timezone_name() -> "str":
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


def get_sandbox_tag(sandbox_mode: "typing.Union[str, None]") -> "str":
    if sandbox_mode == "danger-full-access":
        return "none"
    if sandbox_mode == "read-only":
        return "read-only"
    if sandbox_mode == "workspace-write":
        return "workspace-write"
    return "none"


def get_workspace_turn_metadata(
    cwd: "typing.Union[str, Path]",
) -> "typing.Union[typing.Dict[str, object], None]":
    resolved_cwd = Path(cwd).resolve()
    repo_root = _git_output(
        resolved_cwd,
        ["rev-parse", "--show-toplevel"],
    )
    if repo_root is None:
        return None

    workspace: "typing.Dict[str, object]" = {}
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


def build_user_agent(originator: "str") -> "str":
    version = get_package_version()
    terminal = get_terminal_user_agent_token()
    os_name, os_version = get_os_info()
    arch = platform.machine() or "unknown"
    suffix = _user_agent_suffix(originator, version)
    return f"{originator}/{version} ({os_name} {os_version}; {arch}) {terminal}{suffix}"


def get_package_version() -> "str":
    """Return the upstream alignment version recorded in docs/ALIGNMENT.md."""
    return "0.153.4"


def get_os_info() -> "typing.Tuple[str, str]":
    os_release = Path("/etc/os-release")
    if os_release.is_file():
        values: "typing.Dict[str, str]" = {}
        os_release_text = os_release.read_text(
            encoding="utf-8",
            errors="replace",
        )
        for line in os_release_text.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
        name = values.get("NAME")
        version = values.get("VERSION_ID")
        if name and version:
            return name, _normalize_os_version(version)
    return platform.system(), platform.release()


def get_terminal_user_agent_token() -> "str":
    term_program = os.environ.get("TERM_PROGRAM", "")
    if term_program.lower() == "tmux":
        client_termname = _tmux_display_message("#{client_termname}")
        if client_termname:
            return _sanitize_header_token(client_termname)

    term = os.environ.get("TERM")
    if term:
        return _sanitize_header_token(term)
    return "unknown"


def _git_output(cwd: "Path", args: "typing.List[str]") -> "typing.Union[str, None]":
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _git_remote_urls(cwd: "Path") -> "typing.Dict[str, str]":
    remote_names = _git_output(cwd, ["remote"])
    if remote_names is None:
        return {}
    remotes: "typing.Dict[str, str]" = {}
    for name in remote_names.splitlines():
        remote_name = name.strip()
        if not remote_name:
            continue
        url = _git_output(cwd, ["remote", "get-url", remote_name])
        if url is not None:
            remotes[remote_name] = url
    return remotes


def _git_has_changes(cwd: "Path") -> "typing.Union[bool, None]":
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(completed.stdout.strip())


def _user_agent_suffix(originator: "str", version: "str") -> "str":
    if originator == "codex_exec":
        return f" (codex-exec; {version})"
    if originator == "codex-tui":
        return f" (codex-tui; {version})"
    return ""


def _normalize_os_version(version: "str") -> "str":
    parts = version.split(".")
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        major, minor = parts
        return f"{int(major)}.{int(minor)}.0"
    return version


def _tmux_display_message(fmt: "str") -> "typing.Union[str, None]":
    try:
        output = subprocess.run(
            ["tmux", "display-message", "-p", fmt],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = output.stdout.strip()
    return value or None


def _sanitize_header_token(value: "str") -> "str":
    return "".join(
        character if (character.isalnum() or character in {"-", "_", ".", "/"}) else "_"
        for character in value
    )
