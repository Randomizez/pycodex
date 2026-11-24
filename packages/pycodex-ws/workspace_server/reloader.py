"""Fresh-import support for backend reload, with preflight and package upgrade."""

import asyncio
import importlib
import importlib.util
import os
import subprocess
import sys
import traceback
import types
import typing
from functools import partial
from pathlib import Path

RELOAD_PREFIXES = (
    "pycodex",
    "responses_server",
    "pycodex_ws",
    "pycodex_router",
    "workspace_server",
    "pyagy",
)


def _is_reloadable_module(name: str) -> bool:
    return name in RELOAD_PREFIXES or name.startswith(
        tuple(prefix + "." for prefix in RELOAD_PREFIXES)
    )


def loaded_reloadable_modules() -> "typing.List[typing.Tuple[str, types.ModuleType]]":
    result = []
    for name, module in list(sys.modules.items()):
        if (
            isinstance(module, types.ModuleType)
            and name != __name__
            and _is_reloadable_module(name)
            and getattr(module, "__file__", None)
        ):
            result.append((name, module))
    return result


def _evict_same_second_bytecode(source_path: "typing.Optional[str]") -> None:
    if not source_path:
        return
    source = Path(source_path)
    pyc = Path(importlib.util.cache_from_source(str(source)))
    if pyc.exists() and int(source.stat().st_mtime) >= int(pyc.stat().st_mtime):
        pyc.unlink()


async def _run_python(arguments, env=None) -> None:
    # Python 3.7's asyncio subprocess watcher cannot run on an ASGI worker
    # thread. Run the child off-loop without owning asyncio pipe transports.
    process = await asyncio.get_running_loop().run_in_executor(
        None,
        partial(
            subprocess.run,
            [sys.executable] + list(arguments),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        ),
    )
    if process.returncode != 0:
        detail = process.stdout.decode("utf-8", errors="replace").strip()
        lines = [line for line in detail.splitlines() if line.strip()]
        raise RuntimeError("\n".join(lines[-8:]) or "exit status {0}".format(process.returncode))


async def upgrade_packages(requirement: str) -> None:
    """Upgrade pycodex-ws, which pins the same-version core and pyagy packages.

    Windows locks the running console-script launcher against replacement but
    allows renaming it, so the launcher moves aside while pip installs. After
    an earlier upgrade the running launcher already is the locked aside file.
    """
    launcher = Path(sys.argv[0]).with_suffix(".exe")
    aside = launcher.with_suffix(".exe.old")
    moved = sys.platform == "win32" and launcher.is_file()
    if moved and aside.exists():
        try:
            aside.unlink()
        except PermissionError:
            moved = False
    if moved:
        launcher.rename(aside)
    try:
        await _run_python(["-m", "pip", "install", "--upgrade", requirement])
    except BaseException:
        if moved and not launcher.exists():
            aside.rename(launcher)
        raise


async def preflight_check() -> None:
    loaded = loaded_reloadable_modules()
    for _name, module in loaded:
        file_path = getattr(module, "__file__", None)
        if not file_path or not str(file_path).endswith(".py"):
            continue
        _evict_same_second_bytecode(file_path)
        try:
            source = Path(file_path).read_bytes()
            compile(source, str(file_path), "exec")
        except Exception as exc:
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            raise RuntimeError(detail or str(exc)) from exc
    names = [name for name, _module in loaded]
    if not names:
        return
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
    script = (
        "import importlib, sys\n"
        "for name in sys.argv[1:]:\n"
        "    importlib.import_module(name)\n"
    )
    await _run_python(["-c", script] + names, env)


def purge_reloadable_modules() -> "typing.Dict[str, types.ModuleType]":
    """Remove loaded backend modules so the next import reads current files."""
    importlib.invalidate_caches()
    purged = {}
    for name, module in loaded_reloadable_modules():
        _evict_same_second_bytecode(getattr(module, "__file__", None))
        purged[name] = sys.modules.pop(name)
    return purged


def restore_modules(purged: "typing.Dict[str, types.ModuleType]") -> None:
    """Replace any partially imported generation with the purged modules."""
    for name, _module in loaded_reloadable_modules():
        del sys.modules[name]
    sys.modules.update(purged)
