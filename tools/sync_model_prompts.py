"""Refresh shared model prompts without replacing local models or runtime metadata."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream_models", type=Path)
    args = parser.parse_args()
    upstream = json.loads(args.upstream_models.read_text(encoding="utf-8"))
    source_models = {model["slug"]: model for model in upstream["models"]}
    destination = (
        Path(__file__).resolve().parents[1] / "pycodex" / "prompts" / "models.json"
    )
    local = json.loads(destination.read_text(encoding="utf-8"))
    for model in local["models"]:
        source = source_models.get(model["slug"])
        if source is None:
            continue
        changed = False
        for key in ("base_instructions", "model_messages"):
            if model.get(key) == source.get(key):
                continue
            if key in source:
                model[key] = source[key]
            else:
                del model[key]
            changed = True
        if changed:
            print(model["slug"])
    destination.write_text(
        json.dumps(local, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
