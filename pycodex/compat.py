import asyncio
import functools
import shlex

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - Python 3.6 path
    import importlib_metadata  # type: ignore

try:
    from typing import Literal, Protocol, TypeAlias
except ImportError:  # pragma: no cover - Python 3.6 path
    from typing_extensions import Literal, Protocol  # type: ignore
    try:
        from typing_extensions import TypeAlias  # type: ignore
    except ImportError:  # pragma: no cover - old typing_extensions
        TypeAlias = object


def patch_asyncio():
    if not hasattr(asyncio, "create_task"):
        asyncio.create_task = asyncio.ensure_future

    if not hasattr(asyncio, "get_running_loop"):
        def get_running_loop():
            return asyncio.get_event_loop()

        asyncio.get_running_loop = get_running_loop

    if not hasattr(asyncio, "to_thread"):
        async def to_thread(func, *args, **kwargs):
            loop = asyncio.get_event_loop()
            call = functools.partial(func, *args, **kwargs)
            return await loop.run_in_executor(None, call)

        asyncio.to_thread = to_thread

    if not hasattr(asyncio, "run"):
        def run(main):
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(main)
            finally:
                all_tasks = getattr(asyncio.Task, "all_tasks", None)
                if all_tasks is not None:
                    pending = all_tasks(loop=loop)
                else:
                    pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                shutdown_asyncgens = getattr(loop, "shutdown_asyncgens", None)
                if shutdown_asyncgens is not None:
                    loop.run_until_complete(shutdown_asyncgens())
                asyncio.set_event_loop(None)
                loop.close()

        asyncio.run = run


def shlex_join(parts):
    join = getattr(shlex, "join", None)
    if join is not None:
        return join(parts)
    return " ".join(shlex.quote(part) for part in parts)


def stream_writer_is_closing(writer):
    method = getattr(writer, "is_closing", None)
    if callable(method):
        return method()
    transport = getattr(writer, "transport", None)
    if transport is None:
        return False
    transport_is_closing = getattr(transport, "is_closing", None)
    if callable(transport_is_closing):
        return transport_is_closing()
    return False


def is_ascii(text):
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True
