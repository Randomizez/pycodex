import asyncio
import inspect


def run_async(awaitable):
    """Run an awaitable from synchronous code and return its real result.

    This helper is intended for sync call sites (for example, IPython cells)
    that need to call async agent APIs without managing the event loop
    themselves. It returns the underlying result directly and raises on
    failure instead of swallowing exceptions.
    """
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is not None:
        try:
            import nest_asyncio
        except ImportError as exc:
            if inspect.iscoroutine(awaitable):
                awaitable.close()
            raise RuntimeError(
                "run_async() cannot block on a running event loop without "
                "nest_asyncio; install nest_asyncio or await the object directly."
            ) from exc
        nest_asyncio.apply(running_loop)
        return running_loop.run_until_complete(awaitable)

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(awaitable)
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    if loop.is_running():
        try:
            import nest_asyncio
        except ImportError as exc:
            if inspect.iscoroutine(awaitable):
                awaitable.close()
            raise RuntimeError(
                "run_async() cannot block on a running event loop without "
                "nest_asyncio; install nest_asyncio or await the object directly."
            ) from exc
        nest_asyncio.apply(loop)

    return loop.run_until_complete(awaitable)
