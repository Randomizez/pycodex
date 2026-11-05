"""In-process Python cells with expression display and asyncio task ownership."""

import asyncio
import base64
import builtins
from copy import deepcopy
import inspect
import io
import json
import traceback
from types import SimpleNamespace
import wave

from ..utils.compat import compile_async_cell
from .python_tool_schema import python_tool_signature

_BLOCKED_MODULES = frozenset(("ctypes", "_ctypes", "cffi"))


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and isinstance(name, str):
        root = name.split(".", 1)[0]
        if root in _BLOCKED_MODULES:
            raise ImportError(
                f"'{root}' is disabled in in-process exec cells; "
                "use `await tools.exec_command(...)` to run native code in a subprocess."
            )
    return builtins.__import__(name, globals, locals, fromlist, level)


class CellTaskFactory:
    """Track descendants of cell tasks without owning the host's tool tasks.

    One factory serves the active cells on an event loop. The previous factory
    is restored after the last cell has cleaned up; no process-global state or
    changes to the asyncio module are needed.
    """

    def __init__(self, loop, previous):
        self._loop = loop
        self._previous = previous
        self._owners = {}
        self._users = 0

    @classmethod
    def acquire(cls, loop):
        factory = loop.get_task_factory()
        if not isinstance(factory, cls):
            factory = cls(loop, factory)
            loop.set_task_factory(factory)
        factory._users += 1
        return factory

    def release(self):
        self._users -= 1
        if self._users == 0 and self._loop.get_task_factory() is self:
            self._loop.set_task_factory(self._previous)

    def __call__(self, loop, coroutine, **kwargs):
        owner = self._owners.get(asyncio.current_task(loop=loop))
        return self.create_task(coroutine, owner, **kwargs)

    def create_task(self, coroutine, owner=None, **kwargs):
        if self._previous is None:
            task = asyncio.Task(coroutine, loop=self._loop, **kwargs)
        else:
            task = self._previous(self._loop, coroutine, **kwargs)
        if owner is not None:
            self._owners[task] = owner
            owner._tasks.add(task)
            task.add_done_callback(self._forget)
        return task

    def _forget(self, task):
        self._owners.pop(task, None)


class ExecExit(BaseException):
    pass


class TextWriter:
    def __init__(self, emit):
        self._emit = emit
        self._buffer = ""

    def write(self, value):
        self._buffer += value
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line)
        return len(value)

    def flush(self):
        if self._buffer:
            self._emit(self._buffer)
            self._buffer = ""


class ExecRuntime:
    def __init__(self, source, tools, stored_values, cwd, emit, invoke_tool):
        self._loop = asyncio.get_running_loop()
        self._source = source
        self._values = stored_values
        self._tool_specs = tools
        self._cwd = cwd
        self._emit = emit
        self._invoke_tool = invoke_tool
        self._writer = TextWriter(self.text)
        self._next_request_id = 0
        self._tasks = set()
        self._tool_tasks = []
        self._factory = None
        self._main_task = None
        self._stop_requested = False
        self._closing = False

    def send(self, message):
        self._emit(message)

    def request_stop(self):
        if not self._stop_requested:
            self._stop_requested = True
            if self._main_task is not None and not self._main_task.done():
                self._main_task.cancel()

    def print(self, *values, **kwargs):
        if kwargs.get("file") is None:
            kwargs["file"] = self._writer
        builtins.print(*values, **kwargs)

    def content(self, item):
        self.send({"type": "content", "item": item})

    def text(self, value):
        text = value if isinstance(value, str) else json.dumps(
            value, ensure_ascii=False, allow_nan=False,
        )
        self.content({"type": "input_text", "text": text})

    def notify(self, value):
        text = value if isinstance(value, str) else json.dumps(
            value, ensure_ascii=False, allow_nan=False,
        )
        self.send({"type": "notify", "text": text})

    def image(self, value, detail=...):
        url, item = self._media_url(value, "image")
        if detail is ...:
            detail = item.get("detail", item.get("_meta", {}).get("codex/imageDetail"))
        if detail not in (None, "auto", "low", "high", "original"):
            raise ValueError("unsupported image detail")
        output = {"type": "input_image", "image_url": url}
        if detail is not None:
            output["detail"] = detail
        self.content(output)

    def audio(self, value):
        url, _item = self._media_url(value, "audio")
        mime_type, encoded = url[5:].split(";base64,", 1)
        if mime_type in ("audio/wav", "audio/x-wav"):
            with wave.open(io.BytesIO(base64.b64decode(encoded)), "rb") as clip:
                if clip.getnframes() / clip.getframerate() < 0.025:
                    self.text(
                        "Audio output omitted because the clip is shorter than "
                        "25 ms; use a longer clip."
                    )
                    return
        self.content({"type": "input_audio", "audio_url": url})

    def generated_image(self, result):
        self.image(result["image_url"])
        if "output_hint" in result:
            if not isinstance(result["output_hint"], str):
                raise TypeError("output_hint must be a string")
            self.text(result["output_hint"])

    def _media_url(self, value, kind):
        item = {}
        if isinstance(value, str):
            url = value
        elif isinstance(value, dict):
            item = value
            if value.get("type") == kind:
                url = "data:{0};base64,{1}".format(value["mimeType"], value["data"])
            else:
                url = value[kind + "_url"]
        else:
            raise TypeError(kind + " expects a data URL or an individual media object")
        if not isinstance(url, str) or not url.startswith("data:" + kind + "/"):
            raise ValueError(kind + " expects a base64 data URL")
        _header, encoded = url.split(";base64,", 1)
        if not encoded:
            raise ValueError(kind + " data must not be empty")
        base64.b64decode(encoded, validate=True)
        return url, item

    def store(self, key, value):
        if not isinstance(key, str):
            raise TypeError("store key must be a string")
        self._values[key] = json.loads(json.dumps(value, allow_nan=False))

    def load(self, key):
        if not isinstance(key, str):
            raise TypeError("load key must be a string")
        return deepcopy(self._values.get(key))

    def exit(self):
        raise ExecExit()

    async def yield_control(self):
        self.send({"type": "yield"})
        await asyncio.sleep(0)

    def _tool_caller(self, spec):
        signature = python_tool_signature(spec)

        def call(*args, **kwargs):
            arguments = dict(signature.bind(*args, **kwargs).arguments)
            if spec["tool_type"] == "custom":
                arguments = arguments["input"]
                if not isinstance(arguments, str):
                    raise TypeError(spec["name"] + " expects one source string")
            return self._call_tool(spec, arguments)
        call.__name__ = spec["name"]
        call.__signature__ = signature
        return call

    async def _call_tool(self, spec, arguments):
        if self._stop_requested or self._closing:
            raise RuntimeError("exec cell is stopping")
        self._next_request_id += 1
        request_id = str(self._next_request_id)
        task = self._factory.create_task(self._invoke_tool(
            spec, request_id, deepcopy(arguments),
        ))
        self._tool_tasks.append(task)
        # Once issued, host tool calls finish even if the cell is cancelled.
        return deepcopy(await asyncio.shield(task))

    async def run(self):
        if self._stop_requested:
            return None
        tool_namespaces = {
            "tools": SimpleNamespace(), "collaboration": SimpleNamespace(),
        }
        for spec in self._tool_specs:
            setattr(
                tool_namespaces[spec["namespace"]], spec["name"], self._tool_caller(spec),
            )
        namespace = {
            "__name__": "__codex_exec__",
            "__builtins__": dict(
                vars(builtins), print=self.print, __import__=_guarded_import,
            ),
            "cwd": self._cwd,
            "asyncio": asyncio,
            "text": self.text, "image": self.image, "audio": self.audio,
            "generated_image": self.generated_image,
            "store": self.store, "load": self.load, "notify": self.notify,
            "exit": self.exit, "yield_control": self.yield_control,
            "ALL_TOOLS": tuple(
                {
                    "name": spec["namespace"] + "." + spec["name"],
                    "description": spec["description"],
                }
                for spec in self._tool_specs
            ),
        }
        namespace.update(tool_namespaces)
        error_text = None
        self._factory = CellTaskFactory.acquire(self._loop)
        try:
            self._main_task = self._factory.create_task(
                self._execute(namespace), self,
            )
            try:
                error_text = await self._main_task
            except asyncio.CancelledError:
                if not self._stop_requested:
                    error_text = "exec cell was cancelled"
        finally:
            self._closing = True
            try:
                # A cancelled task's finally block can create another task.
                while True:
                    pending = [task for task in self._tasks if not task.done()]
                    if not pending:
                        break
                    for task in pending:
                        task.cancel()
                    outcomes = await asyncio.gather(*pending, return_exceptions=True)
                    for outcome in outcomes:
                        if (
                            isinstance(outcome, BaseException)
                            and not isinstance(outcome, (asyncio.CancelledError, ExecExit))
                            and error_text is None
                        ):
                            error_text = "".join(traceback.format_exception(
                                type(outcome), outcome, outcome.__traceback__,
                            ))
                await asyncio.gather(*self._tool_tasks, return_exceptions=True)
                self._writer.flush()
            finally:
                self._tasks.clear()
                self._tool_tasks.clear()
                self._factory.release()
        return error_text

    async def _execute(self, namespace):
        try:
            coro = eval(compile_async_cell(self._source), namespace)
            if "__exec_cell__" in namespace:
                coro = namespace.pop("__exec_cell__")()
            if inspect.iscoroutine(coro):
                result = await coro
            else:
                result = None
            if result is None:
                result = namespace.pop("__exec_result__", None)
            if result is not None:
                output = repr(result)
                self._writer.flush()
                self.text(output)
        except ExecExit:
            return None
        except asyncio.CancelledError:
            raise
        except BaseException:
            # User code (including sys.exit()) must report through its cell.
            return traceback.format_exc()
        return None
