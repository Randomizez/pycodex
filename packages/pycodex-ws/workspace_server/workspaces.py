import asyncio
import json
import typing
from dataclasses import dataclass
from pathlib import Path

from pycodex.utils import uuid7_string
from pycodex_ws.workspace_config import (
    WorkspaceDefinition,
    default_state_path,
    load_workspace_definitions,
    next_workspace_id as _next_workspace_id,
    normalize_workspace_id,
    resolve_workspace_path as _resolve_workspace_path,
    save_workspace_definitions,
    workspace_file_path,
)

from .workspace_views import WorkspaceViewStore

SessionFactory = typing.Callable[[], object]


class WorkspaceStateStore:
    def __init__(self, path: "typing.Union[Path, None]") -> None:
        self.path = None if path is None else Path(path)

    def load_tabs(self) -> "typing.List[typing.Dict[str, object]]":
        if self.path is None or not self.path.exists():
            return []

        tabs = json.loads(self.path.read_text(encoding="utf-8", errors="replace"))["tabs"]
        if not isinstance(tabs, list):
            raise ValueError("workspace state `tabs` must be a list")
        if any(tab.get("fork") is True for tab in tabs):
            raise ValueError("workspace state contains unsupported pending fork tabs")
        return tabs

    def save_tabs(self, tabs: "typing.Iterable[typing.Dict[str, object]]") -> None:
        if self.path is None:
            return

        payload = (
            json.dumps(
                {"version": 1, "tabs": list(tabs)},
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(payload, encoding="utf-8")


class WorkspaceSessionManager:
    def __init__(
        self,
        session_factory: "SessionFactory",
        state_path: "typing.Union[Path, None]" = None,
        persist_callback: "typing.Union[typing.Callable[[], None], None]" = None,
        views=None,
        profile_factories: "typing.Optional[typing.Dict[str, SessionFactory]]" = None,
        default_profile: "typing.Optional[str]" = None,
    ) -> None:
        self._session_factory = session_factory
        self.profile_factories = dict(profile_factories or {})
        self.default_profile = default_profile
        if default_profile is not None and default_profile not in self.profile_factories:
            raise ValueError("unknown Codex profile: {0}".format(default_profile))
        self._sessions: "typing.Dict[str, object]" = {}
        self._session_order: "typing.List[str]" = []
        self._state_watchers: "typing.Dict[str, asyncio.Task]" = {}
        self._tab_states: "typing.Dict[str, typing.Dict[str, object]]" = {}
        self._read_final_ids: "typing.Dict[str, str]" = {}
        self._lock = None
        self._state_store = WorkspaceStateStore(
            workspace_file_path(state_path, ".pycodex-ws.json")
        )
        self._persist_callback = persist_callback
        self.views = views if views is not None else WorkspaceViewStore(
            workspace_file_path(state_path, ".pycodex-views.json")
        )

    def _get_lock(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def set_persist_callback(
        self,
        callback: "typing.Union[typing.Callable[[], None], None]",
    ) -> None:
        self._persist_callback = callback

    async def start(self) -> None:
        state_tabs = self._state_store.load_tabs()
        if not state_tabs:
            await self.create_session()
            return
        for tab in state_tabs:
            await self._create_session(
                title=tab["title"],
                rollout_path=tab["rollout_path"],
                cwd=tab.get("cwd"),
                profile=tab.get("profile", self.default_profile),
            )
        for session_id in self._session_order:
            self._subscribe_session_state(session_id)
        self.persist_workspace_state()

    async def close(self) -> None:
        sessions = list(self._sessions.values())
        watchers = list(self._state_watchers.values())
        try:
            for session in sessions:
                await session.close()
            if watchers:
                await asyncio.gather(*watchers)
        finally:
            for watcher in watchers:
                watcher.cancel()
            if watchers:
                await asyncio.gather(*watchers, return_exceptions=True)
            self._sessions.clear()
            self._session_order = []
            self._state_watchers.clear()
            self._tab_states.clear()
            self._read_final_ids.clear()

    async def create_session(
        self,
        title: str = "",
        rollout_path: str = "",
        profile: "typing.Optional[str]" = None,
    ) -> str:
        session_id = await self._create_session(
            title, rollout_path,
            profile=self.default_profile if profile is None else profile,
        )
        self._subscribe_session_state(session_id)
        self.persist_workspace_state()
        return session_id

    async def _create_session(
        self,
        title: str = "",
        rollout_path: str = "",
        cwd: "typing.Optional[str]" = None,
        profile: "typing.Optional[str]" = None,
    ) -> str:
        if profile is not None and (
            not isinstance(profile, str) or profile not in self.profile_factories
        ):
            raise ValueError("unknown Codex profile: {0}".format(profile))
        factory = (
            self._session_factory if profile is None else self.profile_factories[profile]
        )
        async with self._get_lock():
            session_id = uuid7_string()
            session = factory()
            await session.start()

            try:
                if rollout_path or title or cwd is not None:
                    await session.restore_from_rollout(
                        rollout_path, title=title, cwd=cwd
                    )
            except BaseException:
                await session.close()
                raise

            self._sessions[session_id] = session
            self._session_order.append(session_id)
            summary = session_summary(session)
            self._tab_states[session_id] = {
                "title": str(summary.get("title") or ""),
                "cwd": summary["cwd"],
                "rollout_path": rollout_path,
                "profile": profile,
            }
            return session_id

    def _subscribe_session_state(self, session_id: str) -> None:
        session = self._sessions[session_id]
        subscriber = session.subscribe()
        self._state_watchers[session_id] = asyncio.create_task(
            self._watch_session_state(session_id, session, subscriber)
        )

    async def close_session(self, session_id: str) -> None:
        async with self._get_lock():
            if len(self._session_order) <= 1:
                raise ValueError("cannot close the last session")
            session = self._sessions.pop(session_id, None)
            if session is None:
                raise KeyError(session_id)
            watcher = self._state_watchers.pop(session_id, None)
            self._tab_states.pop(session_id, None)
            self._read_final_ids.pop(session_id, None)
            self._session_order = [
                item for item in self._session_order if item != session_id
            ]
        if watcher is not None:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        await session.close()
        self.persist_workspace_state()

    async def _watch_session_state(self, session_id: str, session, subscriber) -> None:
        try:
            while True:
                event = await subscriber.get()
                if event is None:
                    return
                if not isinstance(event, dict) or "snapshot" not in event:
                    continue
                previous = self._tab_states[session_id]
                state = dict(previous)
                snapshot = event["snapshot"]
                state["title"] = str(snapshot.get("title") or "")
                state["cwd"] = snapshot["cwd"]
                state["rollout_path"] = str(snapshot["recorded_rollout_path"] or "")
                self._tab_states[session_id] = state
                if state != previous:
                    self.persist_workspace_state()
        finally:
            session.unsubscribe(subscriber)

    def persist_workspace_state(self) -> None:
        self._state_store.save_tabs(
            self._tab_states[session_id] for session_id in self._session_order
        )
        if self._persist_callback is not None:
            self._persist_callback()

    def get(self, session_id: "typing.Union[str, None]" = None) -> object:
        return self._sessions[self.resolve_session_id(session_id)]

    def resolve_session_id(self, session_id: "typing.Union[str, None]" = None) -> str:
        if session_id:
            return str(session_id)
        if not self._session_order:
            raise KeyError("no sessions")
        return self._session_order[0]

    def mark_read(self, session_id: str, final_id: str) -> None:
        if not isinstance(final_id, str) or not 0 < len(final_id) <= 128:
            raise ValueError("final_id must be a nonempty string of at most 128 characters")
        session = self._sessions[session_id]
        if final_id == session_summary(session).get("last_final_id"):
            self._read_final_ids[session_id] = final_id

    def list_sessions(self) -> "typing.List[typing.Dict[str, object]]":
        result = []
        for session_id in self._session_order:
            session = self._sessions[session_id]
            summary = session_summary(session)
            final_id = summary.get("last_final_id") or ""
            result.append(
                {
                    "id": session_id,
                    "title": summary.get("title") or "pycodex",
                    "cwd": summary["cwd"],
                    "profile": self._tab_states[session_id]["profile"],
                    "running": bool(summary.get("running")),
                    "last_final_id": final_id,
                    "unread": bool(
                        final_id and final_id != self._read_final_ids.get(session_id)
                    ),
                    "spinner": summary.get("spinner") or "",
                    "turn_count": summary.get("turn_count") or 0,
                    "last_assistant": summary.get("last_assistant") or "",
                    "context_remaining_percent": summary.get(
                        "context_remaining_percent"
                    ),
                }
            )
        return result


@dataclass(frozen=True)
class WorkspaceEntry:
    definition: "WorkspaceDefinition"
    manager: "WorkspaceSessionManager"

    def to_dict(self) -> "typing.Dict[str, object]":
        return {
            "id": self.definition.workspace_id,
            "state_path": str(self.definition.state_path),
            "enable_board": self.definition.enable_board,
            "board_path": (
                ""
                if self.definition.board_path is None
                else str(self.definition.board_path)
            ),
            "work_dir": str(self.definition.work_dir),
            "sessions": self.manager.list_sessions(),
        }


WorkspaceEntryFactory = typing.Callable[
    [WorkspaceDefinition, "typing.Union[typing.Callable[[], None], None]"],
    WorkspaceEntry,
]


class WorkspaceRegistry:
    def __init__(
        self,
        entries: "typing.Iterable[WorkspaceEntry]",
        config_path: "typing.Union[str, Path, None]" = None,
        entry_factory: "typing.Union[WorkspaceEntryFactory, None]" = None,
    ) -> None:
        self._entries: "typing.Dict[str, WorkspaceEntry]" = {}
        self._name_to_key: "typing.Dict[str, str]" = {}
        self._order: "typing.List[str]" = []
        self._started = False
        self._config_path = (
            None if config_path is None else Path(config_path).expanduser().resolve()
        )
        self._entry_factory = entry_factory
        for entry in entries:
            self._register(entry)
        if not self._entries and self._entry_factory is None:
            raise ValueError("at least one workspace is required")

    def _register(self, entry: "WorkspaceEntry") -> None:
        workspace_id = entry.definition.workspace_id
        key = self._key_for_definition(entry.definition)
        if key in self._entries:
            raise ValueError(
                "duplicate workspace state_path: {0}".format(entry.definition.state_path)
            )
        if workspace_id in self._name_to_key:
            raise ValueError("duplicate workspace name: {0}".format(workspace_id))
        self._entries[key] = entry
        self._name_to_key[workspace_id] = key
        self._order.append(key)
        entry.manager.set_persist_callback(self.persist_definitions)

    def _key_for_definition(self, definition: "WorkspaceDefinition") -> str:
        return str(definition.state_path.resolve())

    async def start(self) -> None:
        self._started = True
        for key in self._order:
            await self._entries[key].manager.start()

    async def close(self) -> None:
        for key in reversed(self._order):
            await self._entries[key].manager.close()
        self._started = False

    def get(self, workspace_id: str) -> "WorkspaceEntry":
        workspace_id = normalize_workspace_id(workspace_id)
        return self._entries[self._name_to_key[workspace_id]]

    async def add_workspace(
        self,
        name: str,
        work_dir: str = "./",
        state_path: "typing.Union[str, None]" = None,
        enable_board: bool = True,
    ) -> "WorkspaceEntry":
        if self._entry_factory is None:
            raise ValueError("workspace creation is unavailable")

        base_dir = (
            self._config_path.parent if self._config_path is not None else Path.cwd()
        )
        resolved_work_dir = _resolve_workspace_path(str(work_dir or "./"), base_dir)
        resolved_work_dir.mkdir(parents=True, exist_ok=True)

        if state_path is not None and (
            not isinstance(state_path, str) or not state_path.strip()
        ):
            raise ValueError("state_path must be a non-empty path string")
        if not isinstance(enable_board, bool):
            raise ValueError("enable_board must be a boolean")
        resolved_state_path = (
            _resolve_workspace_path(state_path, base_dir)
            if state_path is not None
            else default_state_path()
        )
        state_key = str(resolved_state_path.resolve())
        if state_key in self._entries:
            raise ValueError(
                "workspace state_path already exists: {0}".format(resolved_state_path)
            )

        workspace_id = normalize_workspace_id(name)
        if not workspace_id:
            workspace_id = _next_workspace_id(self._name_to_key)
        if workspace_id in self._name_to_key:
            raise ValueError("workspace name already exists: {0}".format(workspace_id))

        definition = WorkspaceDefinition(
            workspace_id=workspace_id,
            state_path=resolved_state_path,
            work_dir=resolved_work_dir,
            enable_board=enable_board,
        )
        entry = self._entry_factory(definition, self.persist_definitions)
        self._register(entry)
        if self._started:
            await entry.manager.start()
        self.persist_definitions()
        return entry

    async def delete_workspace(self, name: str) -> "WorkspaceDefinition":
        workspace_id = normalize_workspace_id(name)
        key = self._name_to_key.pop(workspace_id)
        entry = self._entries.pop(key)
        self._order = [item for item in self._order if item != key]
        await entry.manager.close()
        self.persist_definitions()
        return entry.definition

    def persist_definitions(self) -> None:
        if self._config_path is None:
            return
        save_workspace_definitions(
            self._config_path,
            [self._entries[key].definition for key in self._order],
        )

    def list_workspaces(self) -> "typing.List[typing.Dict[str, object]]":
        return [self._entries[key].to_dict() for key in self._order]


def session_snapshot(session) -> "typing.Dict[str, object]":
    return typing.cast("typing.Dict[str, object]", session.snapshot())


def session_summary(session) -> "typing.Dict[str, object]":
    summary = getattr(session, "summary", None)
    if callable(summary):
        return typing.cast("typing.Dict[str, object]", summary())
    snapshot = session_snapshot(session)
    return {
        "title": snapshot.get("title") or "",
        "running": bool(snapshot.get("running")),
        "last_final_id": snapshot.get("last_final_id") or "",
        "spinner": snapshot.get("spinner") or "",
        "turn_count": len(snapshot.get("turns") or []),
        "last_assistant": _last_assistant_text(snapshot.get("turns") or []),
        "context_remaining_percent": snapshot.get("context_remaining_percent"),
    }


def _last_assistant_text(turns: "typing.Iterable[typing.Dict[str, object]]") -> str:
    for turn in reversed(list(turns)):
        if str(turn.get("kind") or "assistant") == "control":
            continue
        response = str(turn.get("response") or "").strip()
        if response:
            return response
    return ""
