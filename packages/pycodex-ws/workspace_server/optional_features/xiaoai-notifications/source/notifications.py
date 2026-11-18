"""Optional spoken session notifications, owned by the workspace lifecycle."""
import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import functools
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import threading
import time


BEIJING = timezone(timedelta(hours=8))
KINDS = {
    "turn_completed": "completed", "turn_failed": "error",
}


def quiet_hours(now=None):
    now = now or datetime.now(BEIJING)
    return 1 <= now.astimezone(BEIJING).hour < 10


def validate_speech(text):
    if not isinstance(text, str):
        raise ValueError("speech must be text")
    text = text.strip()
    if not 1 <= len(text) <= 50 or "\n" in text or "\r" in text:
        raise ValueError("speech must be one line of at most 50 characters")
    return text


def session_key(snapshot, session_id):
    return str(snapshot.get("recorded_rollout_path") or
               snapshot.get("rollout_path") or session_id)


def shorten(text, limit):
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[:limit // 3] + "\n[…]\n" + text[-(limit * 2 // 3):]


class SessionNotifications:
    def __init__(self, state_path, config=None):
        if config is None:
            config_path = Path.home() / ".pycodex" / "notifications.json"
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                config = {}
        allowed = config.get("workspace_state_paths")
        self.enabled = bool(config.get("enabled")) and (
            not allowed or str(state_path) in allowed)
        self.config = config
        self.queue = None
        self.worker = None
        self.sessions = {}
        self.lock = threading.RLock()
        self.last_speech_end = 0
        self.db = None
        self.closed = False
        if self.enabled:
            for key in ("prepare_command", "deliver_command"):
                if not isinstance(config.get(key), list) or not config[key]:
                    raise ValueError(key + " must be a nonempty argument list")
            path = str(state_path) + ".pycodex-notifications.sqlite3" if state_path else ":memory:"
            self.db = sqlite3.connect(path, check_same_thread=False)
            if path != ":memory:":
                os.chmod(path, 0o600)
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS notifications "
                "(id TEXT PRIMARY KEY, session_key TEXT, turn_id TEXT, state TEXT, "
                "speech TEXT, created REAL, error TEXT)")
            self.db.commit()
            # A process exit during synthesis or sending must never replay speech.
            self.db.execute(
                "UPDATE notifications SET state='delivery_unknown' "
                "WHERE state IN ('preparing','prepared','sending')")
            self.db.commit()

    def bind(self, session_id, session):
        if self.enabled:
            self.sessions[session_id] = session
            session.set_notification_records(self.records(session_id, session.snapshot()))

    def unbind(self, session_id):
        self.sessions.pop(session_id, None)

    def observe(self, session_id, session, event):
        if not self.enabled or self.closed or event.get("kind") not in KINDS:
            return
        snapshot = event.get("snapshot") or {}
        kind = event["kind"]
        payload = event.get("payload") or {}
        # Steer interruptions and input requests are intermediate events, not
        # notifications. A reply with queued/background work is intermediate too.
        if snapshot.get("queued_inputs"):
            return
        if kind == "turn_completed" and (
                payload.get("background_work_count") or
                snapshot.get("spinner") == "idle: sleeping"):
            return
        turn_id = str(event.get("turn_id") or "")
        if not turn_id:
            return
        key = session_key(snapshot, session_id)
        event_id = hashlib.sha256((key + "\0" + turn_id + "\0" + kind).encode()).hexdigest()
        turn = next((item for item in reversed(snapshot.get("turns") or [])
                     if item.get("turn_id") == event.get("turn_id")), {})
        with self.lock:
            cursor = self.db.execute(
                "INSERT OR IGNORE INTO notifications "
                "(id,session_key,turn_id,state,created) VALUES(?,?,?,?,?)",
                (event_id, key, turn_id, "preparing", time.time()))
            self.db.commit()
            if not cursor.rowcount:
                return
        item = {
            "event_id": event_id, "session_id": session_id, "turn_id": turn_id,
            "kind": kind, "title": shorten(snapshot.get("title"), 180),
            "runtime_status": KINDS[kind],
            "background_active": bool(payload.get("background_work_count")) or
                                 snapshot.get("spinner") == "idle: sleeping",
            "latest_request": shorten(
                turn.get("prompt") or "\n".join(
                    message.get("text", "") for message in turn.get("messages", [])
                ), 1200),
            "latest_response": shorten(turn.get("response"), 5000),
            "error": shorten(turn.get("error"), 500),
        }
        if self.queue is None:
            self.queue = asyncio.Queue()
        self.queue.put_nowait((session, item))
        if self.worker is None:
            self.worker = asyncio.create_task(self._run())

    def records(self, session_id, snapshot):
        if self.db is None:
            return []
        key = session_key(snapshot, session_id)
        with self.lock:
            rows = self.db.execute(
                "SELECT id,turn_id,state,speech,created,error FROM notifications "
                "WHERE session_key=? AND state!='preparing' "
                "AND NOT (state='superseded' AND speech IS NULL) ORDER BY created",
                (key,)).fetchall()
        return [dict(zip(("id", "turn_id", "state", "speech", "created", "error"), row))
                for row in rows]

    def update(self, item, state, speech=None, error=None):
        with self.lock:
            self.db.execute(
                "UPDATE notifications SET state=?,speech=COALESCE(?,speech),error=? WHERE id=?",
                (state, speech, error, item["event_id"]))
            self.db.commit()
        session = self.sessions.get(item["session_id"])
        if session is not None:
            session.set_notification_records(
                self.records(item["session_id"], session.snapshot()))

    async def command(self, command, payload, timeout):
        environment = os.environ.copy()
        environment.pop("PYCODEX_DEBUG_STDERR", None)
        environment.pop("PYCODEX_DEBUG_DIR", None)
        # Python 3.7/3.8 ASGI threads may have no asyncio child watcher.
        # Timeout only terminates this isolated helper, never an Agent turn.
        result = await asyncio.get_running_loop().run_in_executor(
            None, functools.partial(
                subprocess.run, command,
                input=json.dumps(payload, ensure_ascii=False).encode(),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=environment, timeout=timeout,
            ))
        if result.returncode:
            raise RuntimeError("notification command failed")
        return result.stdout.decode("utf-8").strip()

    def latest(self, item):
        # Steer can reuse a turn id, so compare event order within the rollout.
        # rowid is insertion order even when two events have the same timestamp.
        with self.lock:
            row = self.db.execute(
                "SELECT id FROM notifications WHERE session_key=("
                "SELECT session_key FROM notifications WHERE id=?) "
                "ORDER BY rowid DESC LIMIT 1", (item["event_id"],)).fetchone()
        return bool(row and row[0] == item["event_id"])

    def fresh(self, session, item):
        if item["kind"] not in KINDS or not self.latest(item):
            return False
        snapshot = session.snapshot()
        if snapshot.get("queued_inputs"):
            return False
        if snapshot.get("running"):
            return False
        # Compaction and command output do not replace the assistant's result.
        turns = [turn for turn in snapshot.get("turns") or []
                 if turn.get("kind") not in ("compaction", "control")]
        return bool(turns and turns[-1].get("turn_id") == item["turn_id"])

    async def _deliver(self, session, item):
        if not self.fresh(session, item):
            self.update(item, "superseded")
            return
        speech = validate_speech(await self.command(
            self.config["prepare_command"], item, 75))
        # Even suppressed, superseded or failed speech is kept in the chat.
        if not self.latest(item):
            self.update(item, "superseded", speech)
            return
        self.update(item, "prepared", speech)
        if quiet_hours():
            self.update(item, "quiet_hours")
            return
        delay = max(0, self.last_speech_end - time.monotonic())
        if delay:
            await asyncio.sleep(delay)
        # Check again after synthesis and after waiting for other speech.
        if quiet_hours():
            self.update(item, "quiet_hours")
            return
        if not self.fresh(session, item):
            self.update(item, "superseded")
            return
        self.update(item, "sending")
        result = json.loads(await self.command(
            self.config["deliver_command"], {"speech": speech}, 45))
        state = result.get("state")
        if state not in ("accepted", "quiet_hours"):
            raise RuntimeError("speaker did not accept the notification")
        self.update(item, state)
        if state == "accepted":
            self.last_speech_end = time.monotonic() + len(speech) / 4.5 + 1

    async def _run(self):
        while True:
            job = await self.queue.get()
            try:
                if job is None:
                    return
                session, item = job
                try:
                    await self._deliver(session, item)
                except Exception as error:
                    self.update(item, "failed", error=type(error).__name__)
            finally:
                self.queue.task_done()

    async def close(self):
        self.closed = True
        if self.worker is not None:
            self.queue.put_nowait(None)
            await self.worker
            self.worker = None
        if self.db is not None:
            self.db.close()
            self.db = None
        self.sessions.clear()
