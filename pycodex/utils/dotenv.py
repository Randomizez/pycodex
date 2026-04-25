
import os
from pathlib import Path
import typing

ILLEGAL_ENV_VAR_PREFIX = "CODEX_"
DOTENV_FILENAME = ".env"
_LOADED_CODEX_DOTENV_HOMES: 'typing.Set[str]' = set()


def load_codex_dotenv(config_path: 'typing.Union[str, Path]') -> 'None':
    codex_home = str(Path(config_path).resolve().parent)
    if codex_home in _LOADED_CODEX_DOTENV_HOMES:
        return

    dotenv_path = Path(codex_home) / DOTENV_FILENAME
    if not dotenv_path.is_file():
        _LOADED_CODEX_DOTENV_HOMES.add(codex_home)
        return

    for key, value in parse_dotenv(
        dotenv_path.read_text(encoding="utf-8", errors="replace")
    ).items():
        if key.upper().startswith(ILLEGAL_ENV_VAR_PREFIX):
            continue
        os.environ[key] = value

    _LOADED_CODEX_DOTENV_HOMES.add(codex_home)


def parse_dotenv(text: 'str') -> 'typing.Dict[str, str]':
    values: 'typing.Dict[str, str]' = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue

        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = parse_dotenv_value(raw_value.strip())
    return values


def parse_dotenv_value(raw_value: 'str') -> 'str':
    if not raw_value:
        return ""

    quote = raw_value[0]
    if quote in {'"', "'"}:
        if len(raw_value) >= 2 and raw_value[-1] == quote:
            inner = raw_value[1:-1]
        else:
            inner = raw_value[1:]
        if quote == "'":
            return inner
        return bytes(inner, "utf-8").decode("unicode_escape")

    if " #" in raw_value:
        raw_value = raw_value.split(" #", 1)[0].rstrip()
    return raw_value
