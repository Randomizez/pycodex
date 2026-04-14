import os
from pathlib import Path
import typing


def get_debug_dir() -> 'typing.Union[Path, None]':
    value = os.environ.get("PYCODEX_DEBUG_LOG", "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path
