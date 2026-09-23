"""Shared loader for vendored Codex model metadata."""

import json
import re
import typing
from functools import lru_cache
from pathlib import Path

from .protocol import JSONDict

DEFAULT_MODELS_PATH = Path(__file__).resolve().parent / "prompts" / "models.json"


@lru_cache(maxsize=1)
def load_models_by_slug() -> "typing.Dict[str, JSONDict]":
    payload = json.loads(DEFAULT_MODELS_PATH.read_text(encoding="utf-8"))
    models = payload.get("models", [])
    by_slug: "typing.Dict[str, JSONDict]" = {}
    for model in models:
        if not isinstance(model, dict):
            continue
        slug = model.get("slug")
        if isinstance(slug, str):
            by_slug[slug] = model
    return by_slug


def model_metadata(slug: "typing.Union[str, None]") -> "typing.Union[JSONDict, None]":
    if slug is None:
        return None
    models = load_models_by_slug()
    candidates = [name for name in models if slug.startswith(name)]
    if not candidates:
        namespace, separator, suffix = slug.partition("/")
        if (
            separator
            and "/" not in suffix
            and re.fullmatch(r"[A-Za-z0-9_-]+", namespace)
        ):
            candidates = [name for name in models if suffix.startswith(name)]
    if not candidates:
        return None
    return models[max(candidates, key=len)]
