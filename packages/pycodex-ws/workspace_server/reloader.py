"""In-place backend code reload with preflight import verification."""

import asyncio
import copy
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

RELOAD_PREFIXES = ("pycodex", "pycodex_ws", "pycodex_router", "workspace_server")


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
    # Python 3.7's asyncio subprocess watcher cannot run on an ASGI worker
    # thread. Run the import probe off-loop without owning asyncio pipe transports.
    process = await asyncio.get_running_loop().run_in_executor(
        None,
        partial(
            subprocess.run,
            [sys.executable, "-c", script, *names],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        ),
    )
    stderr = process.stderr
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        lines = [line for line in detail.splitlines() if line.strip()]
        summary = "\n".join(lines[-8:]) if lines else "module reload preflight failed"
        raise RuntimeError(summary)


def _rebind_closure(obj: object, mapping: "typing.Dict[object, object]") -> None:
    if isinstance(obj, types.FunctionType):
        if obj.__closure__:
            for cell in obj.__closure__:
                try:
                    target = cell.cell_contents
                except ValueError:
                    continue
                if isinstance(target, (type, types.FunctionType)) and target in mapping:
                    cell.cell_contents = mapping[target]
        if obj.__defaults__:
            obj.__defaults__ = tuple(
                mapping.get(item, item)
                if isinstance(item, (type, types.FunctionType))
                else item
                for item in obj.__defaults__
            )
        if obj.__kwdefaults__:
            for key, item in list(obj.__kwdefaults__.items()):
                if isinstance(item, (type, types.FunctionType)) and item in mapping:
                    obj.__kwdefaults__[key] = mapping[item]
    elif isinstance(obj, (classmethod, staticmethod)):
        _rebind_closure(obj.__func__, mapping)
    elif isinstance(obj, property):
        _rebind_closure(obj.fget, mapping)
        _rebind_closure(obj.fset, mapping)
        _rebind_closure(obj.fdel, mapping)


def _capture_closure_state(
    fn: types.FunctionType,
) -> "typing.List[typing.Tuple[object, bool, object]]":
    cells = []
    for cell in fn.__closure__ or ():
        try:
            cells.append((cell, True, cell.cell_contents))
        except ValueError:
            cells.append((cell, False, None))
    return cells


def _save_function_state(
    fn: types.FunctionType,
    saved_functions: "typing.Dict[types.FunctionType, typing.Tuple[object, ...]]",
) -> None:
    if fn in saved_functions:
        return
    saved_functions[fn] = (
        fn.__code__,
        fn.__defaults__,
        dict(fn.__kwdefaults__) if fn.__kwdefaults__ is not None else None,
        dict(fn.__annotations__) if fn.__annotations__ is not None else {},
        fn.__doc__,
        _capture_closure_state(fn),
    )


def _update_function_in_place(
    old_fn: types.FunctionType,
    new_fn: types.FunctionType,
    mapping: "typing.Dict[object, object]",
    saved_functions: "typing.Dict[types.FunctionType, typing.Tuple[object, ...]]",
) -> types.FunctionType:
    _rebind_closure(new_fn, mapping)
    if old_fn.__code__.co_freevars == new_fn.__code__.co_freevars:
        _save_function_state(old_fn, saved_functions)
        old_fn.__code__ = new_fn.__code__
        old_fn.__defaults__ = new_fn.__defaults__
        old_fn.__kwdefaults__ = new_fn.__kwdefaults__
        old_fn.__annotations__ = new_fn.__annotations__
        old_fn.__doc__ = new_fn.__doc__
        if old_fn.__closure__ and new_fn.__closure__:
            for old_cell, new_cell in zip(old_fn.__closure__, new_fn.__closure__):
                try:
                    old_cell.cell_contents = new_cell.cell_contents
                except ValueError:
                    pass
        return old_fn
    return new_fn


def _update_callable(
    old_val: object,
    new_val: object,
    mapping: "typing.Dict[object, object]",
    saved_functions: "typing.Dict[types.FunctionType, typing.Tuple[object, ...]]",
) -> object:
    if isinstance(old_val, types.FunctionType) and isinstance(
        new_val, types.FunctionType
    ):
        return _update_function_in_place(old_val, new_val, mapping, saved_functions)
    _rebind_closure(new_val, mapping)
    return new_val


def _update_class_in_place(
    old_cls: type,
    new_cls: type,
    mapping: "typing.Dict[object, object]",
    saved_classes: "typing.Dict[type, typing.Dict[str, object]]",
    saved_functions: "typing.Dict[types.FunctionType, typing.Tuple[object, ...]]",
) -> None:
    if old_cls not in saved_classes:
        saved_classes[old_cls] = {
            k: v
            for k, v in vars(old_cls).items()
            if k not in ("__dict__", "__weakref__")
        }
    for attr in list(vars(old_cls)):
        if attr not in vars(new_cls) and attr not in (
            "__dict__",
            "__weakref__",
            "__doc__",
            "__module__",
        ):
            delattr(old_cls, attr)
    for attr, value in list(vars(new_cls).items()):
        if attr in ("__dict__", "__weakref__"):
            continue
        old_val = vars(old_cls).get(attr)
        if isinstance(old_val, classmethod) and isinstance(value, classmethod):
            value = classmethod(
                _update_callable(
                    old_val.__func__, value.__func__, mapping, saved_functions
                )
            )
        elif isinstance(old_val, staticmethod) and isinstance(value, staticmethod):
            value = staticmethod(
                _update_callable(
                    old_val.__func__, value.__func__, mapping, saved_functions
                )
            )
        elif isinstance(old_val, property) and isinstance(value, property):
            value = property(
                _update_callable(old_val.fget, value.fget, mapping, saved_functions),
                _update_callable(old_val.fset, value.fset, mapping, saved_functions),
                _update_callable(old_val.fdel, value.fdel, mapping, saved_functions),
                value.__doc__,
            )
        else:
            value = _update_callable(old_val, value, mapping, saved_functions)
        setattr(old_cls, attr, value)


def _rebind_container(value: object, mapping: "typing.Dict[object, object]") -> object:
    if isinstance(value, (type, types.FunctionType)) and value in mapping:
        return mapping[value]
    if isinstance(value, tuple):
        return tuple(_rebind_container(item, mapping) for item in value)
    if isinstance(value, list):
        value[:] = [_rebind_container(item, mapping) for item in value]
        return value
    if isinstance(value, dict):
        updated = {
            _rebind_container(k, mapping): _rebind_container(v, mapping)
            for k, v in list(value.items())
        }
        value.clear()
        value.update(updated)
        return value
    return value


def reload_modules_in_place() -> "typing.Tuple[int, typing.Callable[[], None]]":
    importlib.invalidate_caches()
    modules = loaded_reloadable_modules()
    mapping: "typing.Dict[object, object]" = {}
    saved_modules: "typing.List[typing.Tuple[types.ModuleType, typing.Dict[str, object]]]" = []
    saved_classes: "typing.Dict[type, typing.Dict[str, object]]" = {}
    saved_functions: "typing.Dict[types.FunctionType, typing.Tuple[object, ...]]" = {}
    for name, module in modules:
        items = list(vars(module).items())
        saved_modules.append((module, dict(items)))
        old_functions = {
            attr: obj
            for attr, obj in items
            if isinstance(obj, types.FunctionType) and obj.__module__ == name
        }
        old_classes = {
            attr: obj
            for attr, obj in items
            if isinstance(obj, type) and obj.__module__ == name
        }
        _evict_same_second_bytecode(getattr(module, "__file__", None))
        spec = getattr(module, "__spec__", None)
        try:
            importlib.reload(module)
        except ModuleNotFoundError:
            if spec is not None and spec.loader is not None:
                module.__spec__ = spec
                spec.loader.exec_module(module)
            else:
                raise
        sys.modules[name] = sys.modules.pop(name)
        for attr, new_obj in list(vars(module).items()):
            if getattr(new_obj, "__module__", None) != name:
                continue
            if attr in old_classes and isinstance(new_obj, type):
                mapping[new_obj] = old_classes[attr]
            elif (
                attr in old_functions
                and isinstance(new_obj, types.FunctionType)
                and old_functions[attr].__code__.co_freevars
                == new_obj.__code__.co_freevars
            ):
                mapping[new_obj] = old_functions[attr]
        for attr, value in list(vars(module).items()):
            if getattr(value, "__module__", None) == name and isinstance(
                value, types.FunctionType
            ):
                setattr(
                    module,
                    attr,
                    _update_callable(
                        old_functions.get(attr), value, mapping, saved_functions
                    ),
                )
            elif getattr(value, "__module__", None) == name and isinstance(value, type):
                if attr in old_classes:
                    _update_class_in_place(
                        old_classes[attr],
                        value,
                        mapping,
                        saved_classes,
                        saved_functions,
                    )
                    setattr(module, attr, old_classes[attr])
                else:
                    for cls_attr, cls_val in list(vars(value).items()):
                        if cls_attr not in ("__dict__", "__weakref__"):
                            _rebind_closure(cls_val, mapping)
            elif not attr.startswith("__") and isinstance(value, (tuple, list, dict)):
                setattr(module, attr, _rebind_container(value, mapping))

    def rollback() -> None:
        for fn, state in saved_functions.items():
            (
                code,
                defaults,
                kwdefaults,
                annotations,
                doc,
                closure_state,
            ) = state
            fn.__code__ = code
            fn.__defaults__ = defaults
            fn.__kwdefaults__ = kwdefaults
            fn.__annotations__ = annotations
            fn.__doc__ = doc
            for cell, has_val, val in closure_state:
                if has_val:
                    cell.cell_contents = val
        for cls, cls_dict in saved_classes.items():
            for attr in list(vars(cls)):
                if attr not in cls_dict and attr not in (
                    "__dict__",
                    "__weakref__",
                    "__doc__",
                    "__module__",
                ):
                    delattr(cls, attr)
            for attr, val in cls_dict.items():
                setattr(cls, attr, val)
        for module, mod_dict in saved_modules:
            module.__dict__.clear()
            module.__dict__.update(mod_dict)

    return len(modules), rollback


def _merge_reprojected_turn(
    old_turn: "typing.Dict[str, object]",
    staged_turn: "typing.Optional[typing.Dict[str, object]]",
    template: "typing.Dict[str, object]",
) -> "typing.Dict[str, object]":
    merged = copy.deepcopy(template)
    if staged_turn is not None:
        merged.update(copy.deepcopy(staged_turn))
    for key, val in old_turn.items():
        if val is None or (val in ("", [], {}) and merged.get(key)):
            continue
        merged[key] = copy.deepcopy(val)
    return merged


def _reproject_view(runtime: object, view: object) -> "typing.Callable[[], None]":
    from pycodex.events import SessionStateEvent
    from .session_view import WebSessionView, _json_safe

    fresh_view = WebSessionView()
    added_attrs = []
    for attr, val in vars(fresh_view).items():
        if attr not in vars(view):
            setattr(view, attr, copy.deepcopy(val))
            added_attrs.append(attr)

    with view._lock:
        saved_state = (
            copy.deepcopy(view._turns),
            dict(view._turns_by_submission_id),
            dict(view._turns_by_turn_id),
            view._last_final_id,
            copy.deepcopy(view._runtime_state),
        )

    def restore_view() -> None:
        with view._lock:
            (
                view._turns,
                view._turns_by_submission_id,
                view._turns_by_turn_id,
                view._last_final_id,
                view._runtime_state,
            ) = saved_state
            for attr in added_attrs:
                vars(view).pop(attr, None)

    staged = WebSessionView()
    staged._apply_runtime_event(SessionStateEvent("attach", runtime.snapshot()))
    template = staged._ensure_turn("", "", "")
    runtime_state = runtime.state_snapshot()

    with view._lock:
        old_turns = list(view._turns)
        old_non_control = [
            t for t in old_turns if t.get("kind", "assistant") != "control"
        ]
        staged_turns = list(staged._turns)
        paired_staged: "typing.Dict[int, typing.Dict[str, object]]" = {}
        if old_non_control and staged_turns:
            count = min(len(old_non_control), len(staged_turns))
            for old_t, staged_t in zip(
                old_non_control[-count:], staged_turns[-count:]
            ):
                paired_staged[id(old_t)] = staged_t

        new_turns = []
        by_submission = {}
        by_turn = {}
        for old_t in old_turns:
            merged = _merge_reprojected_turn(
                old_t, paired_staged.get(id(old_t)), template
            )
            new_turns.append(merged)
            sub_id = str(merged.get("submission_id") or "").strip()
            t_id = str(merged.get("turn_id") or "").strip()
            if sub_id:
                by_submission[sub_id] = merged
            if t_id:
                by_turn[t_id] = merged

        view._turns = new_turns
        view._turns_by_submission_id = by_submission
        view._turns_by_turn_id = by_turn
        view._runtime_state = _json_safe(runtime_state)
        view.snapshot()
        view.summary()

    return restore_view


async def _reproject_interactive_session(
    session: object,
) -> "typing.List[typing.Callable[[], None]]":
    restorers = []
    try:
        restorers.append(_reproject_view(session.runtime, session.view))
        for child_runtime, _fid, child_view in list(
            getattr(session, "_child_views", {}).values()
        ):
            restorers.append(_reproject_view(child_runtime, child_view))
    except BaseException:
        for restore in reversed(restorers):
            restore()
        raise
    return restorers


async def sync_reloaded_sessions(
    managers: "typing.Iterable[object]",
) -> "typing.Callable[[], None]":
    restorers: "typing.List[typing.Callable[[], None]]" = []
    try:
        for manager in managers:
            for session in list(getattr(manager, "_sessions", {}).values()):
                if getattr(session, "_worker_loop", None) is not None and getattr(
                    session, "_session", None
                ) is not None:
                    session_restorers = await session._call(
                        _reproject_interactive_session(session._session)
                    )
                    restorers.extend(session_restorers)
                elif getattr(session, "runtime", None) is not None and getattr(
                    session, "view", None
                ) is not None:
                    session_restorers = await _reproject_interactive_session(session)
                    restorers.extend(session_restorers)
    except BaseException:
        for restore in reversed(restorers):
            restore()
        raise

    def restore_all() -> None:
        for restore in reversed(restorers):
            restore()

    return restore_all
