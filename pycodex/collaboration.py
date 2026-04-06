
from .compat import Literal
import typing

CollaborationMode = Literal["default", "plan", "execute", "pair_programming"]

DEFAULT_COLLABORATION_MODE: 'CollaborationMode' = "default"
PLAN_COLLABORATION_MODE: 'CollaborationMode' = "plan"

_MODE_DISPLAY_NAMES: 'typing.Dict[str, str]' = {
    "default": "Default",
    "plan": "Plan",
    "execute": "Execute",
    "pair_programming": "Pair Programming",
}


def collaboration_mode_display_name(mode: 'typing.Union[str, None]') -> 'str':
    normalized = (mode or DEFAULT_COLLABORATION_MODE).strip().lower()
    return _MODE_DISPLAY_NAMES.get(normalized, normalized.replace("_", " ").title())
