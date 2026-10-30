"""Workspace Runtime adapters, receipt observation and worker-thread dispatch."""

import asyncio
import threading
import typing
from collections import deque

from starlette.concurrency import run_in_threadpool

from pycodex.events import (
    CommandCompletedEvent,
    CommandFailedEvent,
    CommandStartedEvent,
    RuntimeUpdatedEvent,
    SessionStateEvent,
    TurnStartedEvent,
)
from pycodex.utils.event_helpers import shorten_title

from .session_api import history_page
from .session_view import SPINNER_STATUS_PREVIEW_LIMIT, WebSessionView, _json_safe

SessionFactory = typing.Callable[[], object]
ThreadedSessionFactory = typing.Callable[[], "WorkspaceInteractiveSession"]


class _SubmissionStore:
    """Pending receipts and the last 500 completed results on the session loop."""

    def __init__(self):
        self.records = {}
        self._finished = deque()

    def handle_event(self, event, runtime, view):
        if (
            isinstance(event, (CommandStartedEvent, CommandCompletedEvent, CommandFailedEvent))
            and event.submission_id not in self.records
        ):
            self.records[event.submission_id] = {
                "submission_id": event.submission_id,
                "agent_id": runtime.agent.agent_id,
                "kind": "command", "command": event.command.lstrip("/"),
                "status": "queued", "result": None, "error": None,
            }
        if (
            isinstance(event, (TurnStartedEvent, CommandStartedEvent))
            and event.submission_id in self.records
        ):
            record = self.records[event.submission_id]
            record["status"] = "running"
            self._publish(record, view)
        elif isinstance(event, CommandCompletedEvent):
            self._finish(
                self.records[event.submission_id], view, result=event.result,
            )
        elif isinstance(event, CommandFailedEvent):
            self._finish(self.records[event.submission_id], view, error={
                "type": event.error_type, "message": event.error,
            })

    def _publish(self, record, view):
        view.publish({"type": "submission", "submission": _json_safe(record)})

    def _finish(self, record, view, result=None, error=None):
        record.update(
            status="failed" if error is not None else "completed",
            result=_json_safe(result), error=error,
        )
        self._finished.append(record["submission_id"])
        while len(self._finished) > 500:
            del self.records[self._finished.popleft()]
        self._publish(record, view)

    def track(self, receipt, runtime, view, command=None):
        if receipt.submission_id in self.records:
            return
        record = {
            "submission_id": receipt.submission_id,
            "agent_id": runtime.agent.agent_id,
            "kind": receipt.kind, "status": "queued", "result": None, "error": None,
        }
        if command is not None:
            record["command"] = command
        self.records[receipt.submission_id] = record
        self._publish(record, view)
        if receipt.kind == "command":
            return

        def completed(future):
            try:
                result = future.result()
                if receipt.kind == "turn":
                    result = {
                        "turn_id": result.turn_id, "output_text": result.output_text,
                        "iterations": result.iterations,
                    }
            except Exception as error:
                self._finish(record, view, error={
                    "type": type(error).__name__, "message": str(error),
                })
            else:
                self._finish(record, view, result=result)

        receipt.future.add_done_callback(completed)


class WorkspaceInteractiveSession:
    def __init__(
        self,
        runtime,
        config_path: "typing.Union[str, None]" = None,
    ) -> None:
        self.runtime = runtime
        self.config_path = config_path
        self.view = WebSessionView()
        self._frontend_id = None
        self._child_views = {}
        self.submissions = _SubmissionStore()

    async def start(self) -> "WorkspaceInteractiveSession":
        await self.runtime.start(self.config_path)
        if self._frontend_id is None:
            self._frontend_id = self.runtime.attach(self._handle_event)
        return self

    def _handle_event(self, event):
        self._handle_runtime_event(event, self.runtime, self.view)

    def _handle_runtime_event(self, event, runtime, view):
        if runtime is self.runtime and (
            isinstance(event, RuntimeUpdatedEvent) and event.reason == "agents"
            or isinstance(event, SessionStateEvent) and event.reason == "attach"
        ):
            agents = runtime.agents_manager.list_agents(
                runtime.agent.agent_id, runtime.agent.agent_path,
            )
            agent_ids = {agent["agent_id"] for agent in agents}
            for agent_id in tuple(self._child_views):
                if agent_id not in agent_ids:
                    child, frontend_id, child_view = self._child_views.pop(agent_id)
                    child.detach(frontend_id)
                    child_view.close()
            for agent in agents:
                self._observe_runtime(runtime.agents_manager.get_runtime(agent["agent_id"]))
        view.handle_event(event, runtime.state_snapshot())
        self.submissions.handle_event(event, runtime, view)

    async def close(self) -> None:
        try:
            await self.runtime.close()
        finally:
            self.detach()

    def detach(self):
        if self._frontend_id is not None:
            self.runtime.detach(self._frontend_id)
            self._frontend_id = None
        for runtime, frontend_id, view in self._child_views.values():
            runtime.detach(frontend_id)
            view.close()
        self._child_views.clear()
        self.view.close()

    async def submit(
        self, prompt: str, sender: str = "web"
    ) -> "typing.Dict[str, object]":
        try:
            receipt = await self.runtime.submit_input(prompt, sender)
        except (ValueError, RuntimeError) as exc:
            return {"ok": False, "error": str(exc), "snapshot": self.snapshot()}
        self.submissions.track(receipt, self.runtime, self.view)
        return {
            "ok": True,
            "type": "submitted",
            "submission_id": receipt.submission_id,
            "snapshot": self.snapshot(),
        }

    def _runtime_for_agent(self, agent_id):
        if agent_id is None or agent_id == self.runtime.agent.agent_id:
            return self.runtime
        runtime = self.runtime.agents_manager.get_runtime(agent_id)
        if not runtime.agent.agent_path.startswith(self.runtime.agent.agent_path + "/"):
            raise KeyError(agent_id)
        return runtime

    def _observe_runtime(self, runtime):
        agent_id = runtime.agent.agent_id
        if runtime is self.runtime:
            return self.view
        if agent_id not in self._child_views:
            view = WebSessionView()
            view.attach_server_loop(self.view.server_loop)
            frontend_id = runtime.attach(
                lambda event: self._handle_runtime_event(event, runtime, view),
                interactive=False,
            )
            self._child_views[agent_id] = (runtime, frontend_id, view)
        return self._child_views[agent_id][2]

    async def event_view(self, agent_id=None):
        runtime = self._runtime_for_agent(agent_id)
        view = self._observe_runtime(runtime)
        view.synchronize_state(runtime.state_snapshot())
        return view

    async def api_query(self, resource, options):
        if resource == "submission":
            return {"submission": _json_safe(self.submissions.records[options["submission_id"]])}
        agent_id = options.get("agent_id")
        runtime = self._runtime_for_agent(agent_id)
        if resource == "state":
            view = await self.event_view(agent_id)
            return view.api_snapshot()
        if resource == "models":
            return {
                "model": runtime.state_snapshot()["model"],
                "models": await runtime.list_models(),
            }
        if resource == "history":
            agent = runtime.agent
            return await run_in_threadpool(
                history_page, agent.session_id, agent.recorded_session_file_path,
                agent.history, options.get("before"), options["limit"],
            )
        if resource == "agents":
            agents = {}
            for agent in runtime.agents_manager.list_agents(
                runtime.agent.agent_id, runtime.agent.agent_path,
            ):
                view = await self.event_view(agent["agent_id"])
                agents[agent["agent_id"]] = dict(
                    agent,
                    elapsed_seconds=runtime.agents_manager.get_runtime(
                        agent["agent_id"]
                    ).turn_elapsed_seconds,
                    last_message=shorten_title(
                        view.summary()["last_assistant"],
                        limit=SPINNER_STATUS_PREVIEW_LIMIT,
                    ),
                )
            return {
                "root": agents.pop(runtime.agent.agent_id),
                "agents": list(agents.values()),
            }
        if resource == "background-work":
            return {"background_work": runtime.background_work_snapshot()}
        raise ValueError("unknown session resource: " + resource)

    async def submit_command(self, command, argument="", agent_id=None):
        runtime = self._runtime_for_agent(agent_id)
        if command not in runtime.commands() or command == "queue":
            raise ValueError("expected an available session command other than queue")
        if not runtime.accepts_input:
            raise RuntimeError("agent is shutting down")
        view = await self.event_view(agent_id)
        receipt = await runtime.submit_input(
            "/" + command + (" " + argument if argument else ""),
            sender="web", wait_for_command=False,
        )
        self.submissions.track(receipt, runtime, view, command)
        return {
            "submission": _json_safe(self.submissions.records[receipt.submission_id]),
            "submission_id": receipt.submission_id,
        }

    async def answer_input(self, request_id, answer):
        try:
            self.runtime.answer_input(request_id, answer)
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "snapshot": self.snapshot()}
        return {"ok": True, "type": "answered", "snapshot": self.snapshot()}

    def subscribe(self) -> "asyncio.Queue":
        return self.view.subscribe()

    def unsubscribe(self, queue: "asyncio.Queue") -> None:
        self.view.unsubscribe(queue)

    def snapshot(self) -> "typing.Dict[str, object]":
        return self.view.snapshot()

    def summary(self) -> "typing.Dict[str, object]":
        return self.view.summary()

    def rollout_path(self) -> str:
        return self.view.snapshot()["rollout_path"]

    async def restore_from_rollout(
        self,
        rollout_path: str,
        title: str = "",
        cwd: "typing.Optional[str]" = None,
    ) -> None:
        if cwd is not None:
            self.runtime.set_cwd(cwd)
        if rollout_path:
            self.runtime.resume(rollout_path, title)
        else:
            self.runtime.set_title(title)


class ThreadedWorkspaceInteractiveSession:
    def __init__(
        self,
        session_factory: "ThreadedSessionFactory",
        server_loop: "asyncio.AbstractEventLoop",
    ) -> None:
        self._session_factory = session_factory
        self._server_loop = server_loop
        self._view = WebSessionView()
        self._view.attach_server_loop(server_loop)
        self._thread: "typing.Union[threading.Thread, None]" = None
        self._worker_loop: "typing.Union[asyncio.AbstractEventLoop, None]" = None
        self._ready = threading.Event()
        self._startup_error: "typing.Union[BaseException, None]" = None
        self._session: "typing.Union[WorkspaceInteractiveSession, None]" = None

    async def start(self) -> "ThreadedWorkspaceInteractiveSession":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._thread_main,
            name="pycodex-workspace-session",
            daemon=True,
        )
        self._thread.start()
        await asyncio.to_thread(self._ready.wait)
        if self._startup_error is not None:
            raise RuntimeError(
                "workspace session thread failed to start"
            ) from self._startup_error
        return self

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._worker_loop = loop
        asyncio.set_event_loop(loop)
        try:
            session = self._session_factory()
            session.view = self._view
            self._session = session
            try:
                loop.run_until_complete(session.start())
            except BaseException:
                loop.run_until_complete(session.close())
                raise
            self._ready.set()
            loop.run_forever()
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            asyncio.set_event_loop(None)
            loop.close()

    async def close(self) -> None:
        session = self._session
        loop = self._worker_loop
        try:
            if session is not None and loop is not None and loop.is_running():
                future = asyncio.wrap_future(
                    asyncio.run_coroutine_threadsafe(session.close(), loop)
                )
                try:
                    await asyncio.shield(future)
                except asyncio.CancelledError:
                    await future
                    raise
        finally:
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
            thread = self._thread
            if thread is not None:
                await asyncio.to_thread(thread.join)
            self._thread = None

    async def _call(self, coroutine):
        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(coroutine, self._worker_loop)
        )

    async def submit(
        self, prompt: str, sender: str = "web"
    ) -> "typing.Dict[str, object]":
        return await self._call(self._session.submit(prompt, sender))

    async def answer_input(self, request_id, answer):
        return await self._call(self._session.answer_input(request_id, answer))

    async def api_query(self, resource, options):
        return await self._call(self._session.api_query(resource, options))

    async def submit_command(self, command, argument="", agent_id=None):
        return await self._call(self._session.submit_command(command, argument, agent_id))

    async def event_view(self, agent_id=None):
        return await self._call(self._session.event_view(agent_id))

    def subscribe(self) -> "asyncio.Queue":
        return self._view.subscribe()

    def unsubscribe(self, queue: "asyncio.Queue") -> None:
        self._view.unsubscribe(queue)

    def snapshot(self) -> "typing.Dict[str, object]":
        return self._view.snapshot()

    def summary(self) -> "typing.Dict[str, object]":
        return self._view.summary()

    def rollout_path(self) -> str:
        if self._session is None:
            return ""
        return self._session.rollout_path()

    async def restore_from_rollout(
        self,
        rollout_path: str,
        title: str = "",
        cwd: "typing.Optional[str]" = None,
    ) -> None:
        session = self._session
        loop = self._worker_loop
        if session is None or loop is None:
            return

        await self._call(session.restore_from_rollout(rollout_path, title=title, cwd=cwd))
