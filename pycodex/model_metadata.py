"""Shared loader for vendored Codex model metadata."""

from functools import lru_cache
import json
from pathlib import Path
import typing

from .protocol import JSONDict


DEFAULT_MODELS_PATH = Path(__file__).resolve().parent / "prompts" / "models.json"


@lru_cache(maxsize=1)
def load_models_by_slug() -> 'typing.Dict[str, JSONDict]':
    payload = json.loads(DEFAULT_MODELS_PATH.read_text(encoding="utf-8"))
    models = payload.get("models", [])
    by_slug: 'typing.Dict[str, JSONDict]' = {}
    for model in models:
        if not isinstance(model, dict):
            continue
        slug = model.get("slug")
        if isinstance(slug, str):
            by_slug[slug] = model
    return by_slug


def model_metadata(slug: 'typing.Union[str, None]') -> 'typing.Union[JSONDict, None]':
    if slug is None:
        return None
    return load_models_by_slug().get(slug)
