"""Background command and schedule task manager matching agy 1.5.13."""

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import errno
import json
import os
from pathlib import Path, PureWindowsPath
import random
import shutil
import signal
import subprocess
import tempfile
import threading
from time import time
import typing

pty: "typing.Any"
if os.name == "nt":
    pty = None
else:
    try:
        import pty
    except ImportError:
        pty = None

from .context import format_single_system_message, format_utc_iso
from .transcript import truncate_agy_tool_output

if typing.TYPE_CHECKING:
    from .transcript import AgyTranscriptStore


NotifyCallback = typing.Callable[..., typing.Optional[bool]]


@dataclass(frozen=True)
class ParsedCron:
    minutes: "typing.FrozenSet[int]"
    hours: "typing.FrozenSet[int]"
    days_of_month: "typing.FrozenSet[int]"
    months: "typing.FrozenSet[int]"
    days_of_week: "typing.FrozenSet[int]"
    dom_wildcard: "bool"
    dow_wildcard: "bool"


def _parse_cron_field(
    field_str: "str",
    min_val: "int",
    max_val: "int",
    normalize_sunday: "bool" = False,
) -> "typing.Tuple[typing.FrozenSet[int], bool]":
    raw = field_str.strip()
    if not raw:
        raise ValueError("empty cron field")
    is_wildcard = raw == "*"
    values: "typing.Set[int]" = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            raise ValueError(f"invalid cron token in {field_str!r}")
        step = 1
        if "/" in token:
            range_part, step_part = token.split("/", 1)
            step = int(step_part)
            if step <= 0:
                raise ValueError(f"cron step must be positive: {token!r}")
        else:
            range_part = token

        if range_part == "*":
            start, end = min_val, max_val
        elif "-" in range_part:
            lo_s, hi_s = range_part.split("-", 1)
            start, end = int(lo_s), int(hi_s)
            if start > end:
                raise ValueError(f"invalid cron range: {token!r}")
        else:
            val = int(range_part)
            start = val
            end = max_val if "/" in token else val

        if start < min_val or end > max_val:
            raise ValueError(
                f"cron value out of bounds [{min_val}, {max_val}]: {token!r}"
            )
        for current in range(start, end + 1, step):
            values.add(0 if (normalize_sunday and current == 7) else current)
    if not values:
        raise ValueError(f"cron field matched no values: {field_str!r}")
    return frozenset(values), is_wildcard


def parse_cron_expression(expr: "str") -> "ParsedCron":
    parts = expr.strip().split()
    if len(parts) != 5:
        raise ValueError(
            f"cron expression must have exactly 5 fields, got {len(parts)}: {expr!r}"
        )
    minutes, _ = _parse_cron_field(parts[0], 0, 59)
    hours, _ = _parse_cron_field(parts[1], 0, 23)
    dom, dom_wild = _parse_cron_field(parts[2], 1, 31)
    months, _ = _parse_cron_field(parts[3], 1, 12)
    dow, dow_wild = _parse_cron_field(parts[4], 0, 7, normalize_sunday=True)
    return ParsedCron(
        minutes=minutes,
        hours=hours,
        days_of_month=dom,
        months=months,
        days_of_week=dow,
        dom_wildcard=dom_wild,
        dow_wildcard=dow_wild,
    )


def cron_matches(parsed: "ParsedCron", dt: "datetime") -> "bool":
    if dt.minute not in parsed.minutes:
        return False
    if dt.hour not in parsed.hours:
        return False
    if dt.month not in parsed.months:
        return False
    cron_dow = (dt.weekday() + 1) % 7
    dom_ok = dt.day in parsed.days_of_month
    dow_ok = cron_dow in parsed.days_of_week
    if not parsed.dom_wildcard and not parsed.dow_wildcard:
        return dom_ok or dow_ok
    return dom_ok and dow_ok


def next_cron_fire_time(parsed: "ParsedCron", now: "datetime") -> "datetime":
    candidate = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    # Search up to 4 years + leap day (minutes = 2_103_840); step by day/hour when mismatched.
    limit = candidate + timedelta(days=366 * 4)
    while candidate <= limit:
        if candidate.month not in parsed.months:
            if candidate.month == 12:
                candidate = candidate.replace(year=candidate.year + 1, month=1, day=1, hour=0, minute=0)
            else:
                candidate = candidate.replace(month=candidate.month + 1, day=1, hour=0, minute=0)
            continue
        cron_dow = (candidate.weekday() + 1) % 7
        dom_ok = candidate.day in parsed.days_of_month
        dow_ok = cron_dow in parsed.days_of_week
        day_ok = (dom_ok or dow_ok) if (not parsed.dom_wildcard and not parsed.dow_wildcard) else (dom_ok and dow_ok)
        if not day_ok:
            candidate = (candidate + timedelta(days=1)).replace(hour=0, minute=0)
            continue
        if candidate.hour not in parsed.hours:
            candidate = (candidate + timedelta(hours=1)).replace(minute=0)
            continue
        if candidate.minute not in parsed.minutes:
            candidate = candidate + timedelta(minutes=1)
            continue
        return candidate
    raise ValueError("cron expression has no matching occurrence within 4 years")


def format_command_result_body(exit_code: "int", output: "str") -> "str":
    if not output:
        return f"Command exited with code {exit_code}.\n\nNo output from the command."
    return f"Command exited with code {exit_code}.\n\nCommand output:\n{output}"


def format_schedule_duration_label(total_seconds: "int") -> "str":
    hours, rem = divmod(int(total_seconds), 3600)
    mins, secs = divmod(rem, 60)
    parts: "typing.List[str]" = []
    if hours:
        parts.append(f"{hours}h")
    if mins:
        parts.append(f"{mins}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def format_cron_summary(expr: "str") -> "str":
    cleaned = " ".join(expr.strip().split())
    tokens = cleaned.split(" ")
    if len(tokens) == 5:
        minute, hour, dom, month, dow = tokens
        if cleaned == "* * * * *":
            return "cron (every minute)"
        if minute.startswith("*/") and hour == "*" and dom == "*" and month == "*" and dow == "*":
            n = minute[2:]
            return "cron (every minute)" if n == "1" else f"cron (every {n} minutes)"
        if minute == "0" and hour == "*" and dom == "*" and month == "*" and dow == "*":
            return "cron (every hour)"
    return f"cron ({cleaned})"


def format_background_task_started(
    task_id: "str",
    log_path: "Path",
    api_model: "str",
) -> "str":
    base = (
        f"Tool is running as a background task with task id: {task_id}\n"
        f"Task logs are available at: file://{log_path}"
    )
    if api_model == "gemini-3.1-pro-preview":
        return base
    action_guidance = (
        "\nYOU MUST TAKE ONE OF THE FOLLOWING TWO ACTIONS: A) either proceed to other "
        "relevant work (if any) or, B) simply update the user with a short message "
        "(that you have launched the command and will wait for it to finish) and end the turn.\n"
        " DO NOTHING ELSE."
    )
    if api_model.startswith("gemini-3."):
        return base + action_guidance
    return (
        base
        + action_guidance
        + "\n When updating the user, always respond in the language of the user "
        "(defaulting to English). Never mix two or more languages in the same response."
    )


class _CommandProcessHandle:
    """Manages PTY/pipe IO and process lifecycle for one shell command."""

    def __init__(
        self,
        proc: "subprocess.Popen[bytes]",
        master_fd: "typing.Optional[int]",
        loop: "asyncio.AbstractEventLoop",
    ) -> "None":
        self.proc = proc
        self.tty: "bool" = master_fd is not None
        self._master_fd = master_fd
        self._loop = loop
        self._lock = threading.Lock()
        self._buffer = bytearray()
        self._log_path: "typing.Optional[Path]" = None
        self.done_event = asyncio.Event()
        self.exit_code: "typing.Optional[int]" = None

    def attach_log_file(self, log_path: "Path") -> "None":
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._log_path = log_path
            log_path.write_bytes(bytes(self._buffer))

    def append_chunk(self, data: "bytes") -> "None":
        if not data:
            return
        with self._lock:
            self._buffer.extend(data)
            if self._log_path is not None:
                with self._log_path.open("ab") as fh:
                    fh.write(data)

    def output_text(self) -> "str":
        with self._lock:
            return bytes(self._buffer).decode("utf-8", errors="replace")

    def write_input(self, data: "str") -> "None":
        encoded = data.encode("utf-8")
        with self._lock:
            if self._master_fd is not None:
                os.write(self._master_fd, encoded)
                return
            if self.proc.stdin is not None:
                self.proc.stdin.write(encoded)
                self.proc.stdin.flush()
                return
        raise RuntimeError("task stdin is already closed")

    def close_master(self) -> "None":
        with self._lock:
            if self._master_fd is not None:
                fd = self._master_fd
                self._master_fd = None
                try:
                    os.close(fd)
                except OSError:
                    pass

    def terminate_group(self) -> "None":
        if self.proc.poll() is not None:
            self.close_master()
            return
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                self.proc.kill()
            try:
                self.proc.wait(timeout=0.3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.close_master()
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=0.3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except OSError:
                self.proc.kill()
        self.close_master()

    def mark_finished(self, code: "int") -> "None":
        self.exit_code = code
        self.close_master()
        try:
            self._loop.call_soon_threadsafe(self.done_event.set)
        except RuntimeError:
            pass


@dataclass
class AgyTaskRecord:
    task_id: "str"
    step_index: "int"
    tool_name: "str"
    tool_summary: "str"
    description: "str"
    start_time_utc: "str"
    log_path: "Path"
    is_daemon: "bool"
    status: "str" = "RUNNING"
    timer_condition: "typing.Optional[str]" = None
    command_handle: "typing.Optional[_CommandProcessHandle]" = None
    async_task: "typing.Optional[asyncio.Task[None]]" = None

    def to_list_item(self) -> "typing.Dict[str, str]":
        return {
            "taskId": self.task_id,
            "toolName": self.tool_name,
            "toolSummary": self.tool_summary,
            "description": self.description,
            "startTime": self.start_time_utc,
            "logUri": f"file://{self.log_path}",
        }


@dataclass
class _Schedule:
    step_index: int
    prompt: str
    start_time_utc: str
    next_fire_at: float
    duration_seconds: "typing.Optional[int]" = None
    timer_condition: "typing.Optional[str]" = None
    cron_expression: "typing.Optional[str]" = None
    exact_timing: bool = False
    is_daemon: bool = False
    max_iterations: "typing.Optional[int]" = None
    fired: int = 0

    def save(self, path: Path) -> None:
        file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(path.parent), delete=False)
        temporary = Path(file.name)
        try:
            with file:
                json.dump(asdict(self), file, ensure_ascii=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(str(temporary), str(path))
        finally:
            if temporary.exists():
                temporary.unlink()


def resolve_agy_shell_command(
    command_line: "str",
) -> "typing.Tuple[str, typing.List[str]]":
    """Return `(shell_label, argv)` for executing `command_line` on the host OS."""
    if os.name != "nt":
        return "bash", ["bash", "-c", command_line]

    env_shell = (os.environ.get("SHELL") or "").strip()
    resolved_env: "typing.Optional[str]" = None
    if env_shell:
        if env_shell.startswith("/"):
            if os.path.isfile(env_shell):
                resolved_env = env_shell
            else:
                resolved_env = shutil.which(PureWindowsPath(env_shell).name)
        else:
            resolved_env = shutil.which(env_shell) or env_shell

    if resolved_env:
        env_name = PureWindowsPath(resolved_env).name.lower()
        if env_name in {"cmd", "cmd.exe"}:
            return "cmd", [resolved_env, "/C", command_line]
        if "powershell" in env_name or env_name in {"pwsh", "pwsh.exe"}:
            label = "pwsh" if env_name in {"pwsh", "pwsh.exe"} else "powershell"
            return label, [
                resolved_env,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command_line,
            ]
        stem = env_name[:-4] if env_name.endswith(".exe") else env_name
        return (stem or "bash"), [resolved_env, "-c", command_line]

    for candidate in ("powershell.exe", "powershell", "pwsh.exe", "pwsh"):
        found = shutil.which(candidate)
        if found:
            found_name = PureWindowsPath(found).name.lower()
            label = "pwsh" if found_name in {"pwsh", "pwsh.exe"} else "powershell"
            return label, [
                found,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command_line,
            ]

    comspec = os.environ.get("COMSPEC") or "cmd.exe"
    return "cmd", [comspec, "/C", command_line]


def _next_cron_deadline(expression: str, exact_timing: bool) -> float:
    now = datetime.now().astimezone()
    target = next_cron_fire_time(parse_cron_expression(expression), now).timestamp()
    delay = target - now.timestamp()
    if not exact_timing and delay > 10:
        target += random.uniform(0.0, min(5.0, delay * 0.05))
    return target


class AgyTaskManager:
    """Conversation-scoped background task manager."""

    def __init__(
        self,
        conversation_id: "str",
        tasks_dir: "Path",
        api_model_getter: "typing.Callable[[], str]",
        notify_callback: "NotifyCallback",
        transcript_store: "typing.Optional[AgyTranscriptStore]" = None,
        clock: "typing.Optional[typing.Callable[[], datetime]]" = None,
    ) -> "None":
        self.conversation_id = conversation_id
        self.tasks_dir = tasks_dir
        self._api_model_getter = api_model_getter
        self._notify_callback = notify_callback
        self.transcript_store = transcript_store
        self._clock = clock
        self._tasks: "typing.Dict[str, AgyTaskRecord]" = {}
        self._state_change_event: "typing.Optional[asyncio.Event]" = None
        self._state_listener: "typing.Optional[typing.Callable[[], None]]" = None
        self._closed = False

    def bind(
        self,
        invoke: "typing.Any",
        output: "typing.Any",
        state_changed: "typing.Optional[typing.Callable[[], None]]",
    ) -> "None":
        del invoke, output
        self._state_listener = state_changed

    def set_state_listener(
        self,
        listener: "typing.Optional[typing.Callable[[], None]]",
    ) -> "None":
        self._state_listener = listener

    def shutdown(self) -> "None":
        self._state_listener = None

    def restart(self) -> "None":
        had_tasks = bool(self._tasks)
        for rec in list(self._tasks.values()):
            if rec.status == "RUNNING":
                rec.status = "CANCELLED"
            if rec.command_handle is not None:
                rec.command_handle.terminate_group()
            if rec.async_task is not None and not rec.async_task.done():
                rec.async_task.cancel()
        self._tasks.clear()
        self._closed = False
        if had_tasks:
            self._signal_state_change()
        elif self._state_change_event is not None:
            self._state_change_event.set()

    def background_work_count(self, after_reply: "bool") -> "int":
        del after_reply
        return sum(rec.status == "RUNNING" for rec in self._tasks.values())

    def background_work_snapshot(self) -> "typing.List[typing.Dict[str, object]]":
        items: "typing.List[typing.Dict[str, object]]" = []
        for rec in sorted(self._tasks.values(), key=lambda r: r.step_index):
            if rec.status != "RUNNING":
                continue
            entry: "typing.Dict[str, object]" = {
                "kind": "exec" if rec.tool_name == "run_command" else "schedule",
                "session_id": rec.step_index,
                "task_id": rec.task_id,
                "summary": rec.tool_summary,
                "start_time": rec.start_time_utc,
                "is_daemon": rec.is_daemon,
                "status": "running",
            }
            if rec.command_handle is not None:
                entry["pid"] = rec.command_handle.proc.pid
                entry["tty"] = rec.command_handle.tty
                entry["exit_code"] = None
            items.append(entry)
        return items

    def _now(self) -> "typing.Optional[datetime]":
        return self._clock() if self._clock is not None else None

    def _step_output_path(self, step_index: "int") -> "Path":
        if self.transcript_store is not None:
            return self.transcript_store.step_output_path(step_index)
        return self.tasks_dir.parent / "steps" / str(int(step_index)) / "output.txt"

    def _signal_state_change(self) -> "None":
        if self._state_change_event is not None:
            self._state_change_event.set()
        if self._state_listener is not None:
            self._state_listener()

    def has_active_non_daemon_tasks(self) -> "bool":
        return any(
            rec.status == "RUNNING" and not rec.is_daemon
            for rec in self._tasks.values()
        )

    async def wait_state_change(self, timeout_seconds: "float" = 1.0) -> "None":
        if self._state_change_event is None:
            self._state_change_event = asyncio.Event()
        self._state_change_event.clear()
        if not self.has_active_non_daemon_tasks():
            return
        try:
            await asyncio.wait_for(self._state_change_event.wait(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return

    def notify_incoming_message(self, sender: "str") -> "None":
        """Cancel any one-shot timer whose TimerCondition matches `sender`."""
        for rec in list(self._tasks.values()):
            if rec.status != "RUNNING" or rec.tool_name != "schedule":
                continue
            cond = (rec.timer_condition or "never").strip()
            if cond in ("", "never"):
                continue
            if cond == "any" or cond == sender:
                self.kill(rec.task_id)

    def _spawn_shell_process(
        self,
        command_line: "str",
        cwd: "Path",
        loop: "asyncio.AbstractEventLoop",
    ) -> "_CommandProcessHandle":
        _, shell_argv = resolve_agy_shell_command(command_line)
        master_fd: "typing.Optional[int]" = None
        slave_fd: "typing.Optional[int]" = None
        if pty is not None and os.name != "nt":
            try:
                master_fd, slave_fd = pty.openpty()
            except OSError:
                master_fd, slave_fd = None, None

        if master_fd is not None and slave_fd is not None:
            try:
                proc = subprocess.Popen(
                    shell_argv,
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    cwd=str(cwd),
                    start_new_session=True,
                    close_fds=True,
                )
            except BaseException:
                os.close(master_fd)
                os.close(slave_fd)
                raise
            os.close(slave_fd)
            handle = _CommandProcessHandle(proc=proc, master_fd=master_fd, loop=loop)

            def _read_pty() -> "None":
                try:
                    while True:
                        try:
                            chunk = os.read(master_fd, 4096)
                        except OSError as exc:
                            if exc.errno in (errno.EIO, errno.EBADF):
                                break
                            break
                        if not chunk:
                            break
                        handle.append_chunk(chunk)
                finally:
                    code = proc.wait()
                    handle.mark_finished(code)

            threading.Thread(target=_read_pty, daemon=True).start()
            return handle

        popen_kwargs: "typing.Dict[str, typing.Any]" = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "bufsize": 0,
            "cwd": str(cwd),
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        else:
            popen_kwargs["start_new_session"] = True
            popen_kwargs["close_fds"] = True
        proc = subprocess.Popen(shell_argv, **popen_kwargs)
        handle = _CommandProcessHandle(proc=proc, master_fd=None, loop=loop)

        def _read_pipe() -> "None":
            try:
                assert proc.stdout is not None
                while True:
                    chunk = proc.stdout.read(4096)
                    if not chunk:
                        break
                    handle.append_chunk(chunk)
            finally:
                code = proc.wait()
                handle.mark_finished(code)

        threading.Thread(target=_read_pipe, daemon=True).start()
        return handle

    async def run_command(
        self,
        command_line: "str",
        tool_summary: "str",
        wait_ms: "int",
        is_daemon: "bool",
        cwd: "Path",
        step_index: "int",
    ) -> "typing.Tuple[bool, str]":
        """Run `command_line`. Returns `(completed_synchronously, formatted_body)`."""
        if self._closed:
            raise RuntimeError("task manager is closed")
        clamped_ms = max(500, min(10000, int(wait_ms)))
        loop = asyncio.get_running_loop()
        handle = self._spawn_shell_process(command_line=command_line, cwd=cwd, loop=loop)

        try:
            await asyncio.wait_for(handle.done_event.wait(), timeout=clamped_ms / 1000.0)
            code = 0 if handle.exit_code is None else int(handle.exit_code)
            return True, format_command_result_body(code, handle.output_text())
        except asyncio.TimeoutError:
            pass

        if self._closed:
            handle.terminate_group()
            raise RuntimeError("task manager is closed")

        task_id = f"{self.conversation_id}/task-{step_index}"
        log_path = self.tasks_dir / f"task-{step_index}.log"
        handle.attach_log_file(log_path)
        record = AgyTaskRecord(
            task_id=task_id,
            step_index=step_index,
            tool_name="run_command",
            tool_summary=tool_summary,
            description=command_line,
            start_time_utc=format_utc_iso(self._now()),
            log_path=log_path,
            is_daemon=bool(is_daemon),
            status="RUNNING",
            command_handle=handle,
        )

        async def _watch_completion() -> "None":
            await handle.done_event.wait()
            if self._closed or record.status != "RUNNING":
                self._signal_state_change()
                return
            record.status = "DONE"
            code = 0 if handle.exit_code is None else int(handle.exit_code)
            raw_result_body = format_command_result_body(code, handle.output_text())
            result_body = truncate_agy_tool_output(
                raw_result_body,
                output_path=self._step_output_path(step_index),
                always_persist=True,
            )
            msg_content = (
                f'Task id "{task_id}" finished with result:\n'
                f"{result_body}\n"
                f"Log: file://{log_path}"
            )
            self._notify_callback(
                msg_content,
                task_id,
                "MESSAGE_PRIORITY_HIGH",
                True,
                "Task done",
            )
            self._signal_state_change()

        self._tasks[task_id] = record
        record.async_task = asyncio.create_task(_watch_completion())
        self._signal_state_change()
        return False, format_background_task_started(
            task_id=task_id,
            log_path=log_path,
            api_model=self._api_model_getter(),
        )

    async def schedule(
        self,
        prompt: "str",
        step_index: "int",
        duration_seconds: "typing.Optional[int]" = None,
        cron_expression: "typing.Optional[str]" = None,
        exact_timing: "bool" = False,
        is_daemon: "bool" = False,
        max_iterations: "typing.Optional[int]" = None,
        timer_condition: "typing.Optional[str]" = None,
    ) -> "str":
        has_duration = duration_seconds is not None
        has_cron = cron_expression is not None and bool(cron_expression.strip())
        if has_duration == has_cron:
            raise ValueError(
                "exactly one of DurationSeconds or CronExpression must be provided"
            )
        if self._closed:
            raise RuntimeError("task manager is closed")

        task_id = f"{self.conversation_id}/task-{step_index}"
        log_path = self.tasks_dir / f"task-{step_index}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if has_duration:
            assert duration_seconds is not None
            if max_iterations is not None:
                raise ValueError("MaxIterations cannot be used with DurationSeconds")
            if int(duration_seconds) <= 0:
                raise ValueError("DurationSeconds must be a positive integer")
            secs = int(duration_seconds)
            cond = (timer_condition or "never").strip() or "never"
            schedule = _Schedule(
                step_index, prompt, format_utc_iso(self._now()), time() + secs,
                duration_seconds=secs, timer_condition=cond,
            )
            log_line = f"Scheduled one-shot timer for {secs}s (condition={cond})."
        else:
            assert cron_expression is not None
            if timer_condition is not None and timer_condition.strip():
                raise ValueError("TimerCondition cannot be used with CronExpression")
            if max_iterations is not None:
                max_iterations = int(max_iterations)
                if max_iterations <= 0:
                    raise ValueError("MaxIterations must be a positive integer")
            schedule = _Schedule(
                step_index, prompt, format_utc_iso(self._now()), _next_cron_deadline(cron_expression, exact_timing),
                cron_expression=cron_expression.strip(), exact_timing=exact_timing,
                is_daemon=is_daemon, max_iterations=max_iterations,
            )
            log_line = f"Scheduled cron {cron_expression!r} (max_iterations={max_iterations}, daemon={is_daemon})."

        log_path.write_text(log_line + "\n", encoding="utf-8")
        schedule.save(log_path.with_suffix(".schedule.json"))
        self._arm_schedule(schedule)
        self._signal_state_change()
        return format_background_task_started(task_id, log_path, self._api_model_getter())

    def restore_schedules(self) -> None:
        pending = []
        for path in self.tasks_dir.glob("task-*.schedule.json"):
            schedule = _Schedule(**json.loads(path.read_text(encoding="utf-8")))
            if path.name != f"task-{schedule.step_index}.schedule.json" or type(schedule.step_index) is not int:
                raise ValueError(f"invalid schedule ID in {path}")
            task_id = f"{self.conversation_id}/task-{schedule.step_index}"
            if task_id not in self._tasks:
                pending.append(schedule)
        for schedule in sorted(pending, key=lambda s: (s.next_fire_at, s.step_index)):
            self._arm_schedule(schedule)
            if self.transcript_store is not None:
                self.transcript_store.set_step_index(schedule.step_index + 1)
        if pending:
            self._signal_state_change()

    def _arm_schedule(self, schedule: _Schedule) -> None:
        if schedule.duration_seconds is not None:
            duration = format_schedule_duration_label(schedule.duration_seconds)
            summary = f"{duration} timer"
            description = f"Timer: {duration}, Prompt: {schedule.prompt}"
        else:
            summary = format_cron_summary(schedule.cron_expression)
            description = f"Cron: {schedule.cron_expression}, Prompt: {schedule.prompt}"
        task_id = f"{self.conversation_id}/task-{schedule.step_index}"
        record = AgyTaskRecord(
            task_id=task_id, step_index=schedule.step_index, tool_name="schedule",
            tool_summary=summary, description=description, start_time_utc=schedule.start_time_utc,
            log_path=self.tasks_dir / f"task-{schedule.step_index}.log",
            is_daemon=schedule.is_daemon, timer_condition=schedule.timer_condition,
        )
        self._tasks[task_id] = record
        record.async_task = asyncio.create_task(self._run_schedule(record, schedule))

    async def _run_schedule(self, record: AgyTaskRecord, schedule: _Schedule) -> None:
        path = record.log_path.with_suffix(".schedule.json")
        while not self._closed and record.status == "RUNNING":
            await asyncio.sleep(max(0.0, schedule.next_fire_at - time()))
            schedule.fired += 1
            final = schedule.cron_expression is None or schedule.fired == schedule.max_iterations
            if final:
                # An expiring one-shot must not cancel itself on its own notification.
                record.status = "DONE"
            accepted = self._notify_callback(
                schedule.prompt,
                record.task_id,
                "MESSAGE_PRIORITY_HIGH",
                True,
                "Wake-up",
            )
            if accepted is False:
                return  # Admission closed. Leave the saved occurrence for resume.
            if final:
                path.unlink()
            else:
                schedule.next_fire_at = _next_cron_deadline(schedule.cron_expression, schedule.exact_timing)
                schedule.save(path)
            event = "Fired" if schedule.cron_expression is None else f"Cron trigger #{schedule.fired}"
            with record.log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"{event} at {format_utc_iso(self._now())}: {schedule.prompt}\n")
            self._signal_state_change()

    def list_tasks(self) -> "str":
        running_records = sorted(
            (rec for rec in self._tasks.values() if rec.status == "RUNNING"),
            key=lambda r: r.step_index,
        )
        running = [rec.to_list_item() for rec in running_records]
        if not running:
            return "No background tasks are currently running."
        encoded = json.dumps(running, ensure_ascii=False, separators=(",", ":"))
        return f"You have {len(running)} background task(s) currently running:\n{encoded}"

    def status(self, task_id: "str") -> "str":
        rec = self._tasks.get(task_id)
        if rec is None:
            raise ValueError(f"task {task_id!r} not found")
        tail = ""
        if rec.command_handle is not None:
            tail = rec.command_handle.output_text()[-4096:]
        elif rec.log_path.is_file():
            tail = rec.log_path.read_text(encoding="utf-8", errors="replace")[-4096:]
        return (
            f"Task: {task_id}\n"
            f"Status: {rec.status}\n"
            f"Log: file://{rec.log_path}\n"
            f"{tail}"
        )

    def send_input(self, task_id: "str", data: "str") -> "str":
        rec = self._tasks.get(task_id)
        if rec is None:
            raise ValueError(f"task {task_id!r} not found")
        if rec.status != "RUNNING":
            raise ValueError(f"task {task_id!r} is not running (status={rec.status})")
        if rec.command_handle is None:
            raise ValueError(f"task {task_id!r} does not accept stdin")
        rec.command_handle.write_input(data)
        return f'Input sent to task "{task_id}".'

    def kill(self, task_id: "str") -> "str":
        rec = self._tasks.get(task_id)
        if rec is None:
            raise ValueError(f"task {task_id!r} not found")
        was_running = rec.status == "RUNNING"
        if was_running:
            if rec.tool_name == "schedule":
                try:
                    rec.log_path.with_suffix(".schedule.json").unlink()
                except FileNotFoundError:
                    pass  # A timer created before schedule persistence has only a log.
            rec.status = "CANCELLED"
            if rec.command_handle is not None:
                rec.command_handle.terminate_group()
            if rec.async_task is not None and not rec.async_task.done():
                rec.async_task.cancel()
            self._signal_state_change()
        if was_running and rec.tool_name == "run_command":
            cancel_content = (
                f'Task id "{task_id}" was canceled with result:\n'
                "Tool execution was canceled\n"
                f"Log: file://{rec.log_path}"
            )
            self._notify_callback(
                cancel_content,
                task_id,
                "MESSAGE_PRIORITY_LOW",
                False,
                "Task canceled",
            )
        return f'Task "{task_id}" cancelled.'

    async def close(self) -> "None":
        self._closed = True
        had_running = False
        pending_tasks: "typing.List[asyncio.Task[None]]" = []
        for rec in self._tasks.values():
            if rec.status == "RUNNING":
                rec.status = "CANCELLED"
                had_running = True
            if rec.command_handle is not None:
                rec.command_handle.terminate_group()
            if rec.async_task is not None and not rec.async_task.done():
                rec.async_task.cancel()
                pending_tasks.append(rec.async_task)
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        if had_running:
            self._signal_state_change()
        elif self._state_change_event is not None:
            self._state_change_event.set()
