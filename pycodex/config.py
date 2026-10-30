"""Load startup configuration once, then pass each component its own config."""

import typing
from dataclasses import dataclass
from pathlib import Path

from .context import ContextConfig
from .model import ResponsesProviderConfig
from .utils.codex_config import DEFAULT_CODEX_CONFIG_PATH, read_codex_config
from .utils.dotenv import load_codex_dotenv


@dataclass(frozen=True)
class CodexConfig:
    path: "Path"
    profile: "typing.Union[str, None]"
    provider: "ResponsesProviderConfig"
    context: "ContextConfig"

    @classmethod
    def from_env(
        cls,
        config_path: "typing.Union[str, Path]" = DEFAULT_CODEX_CONFIG_PATH,
        profile: "typing.Union[str, None]" = None,
        use_default_profile: "bool" = True,
    ) -> "CodexConfig":
        """Load the selected TOML and adjacent dotenv; retain dynamic key lookup."""
        path = Path(config_path)
        load_codex_dotenv(path)
        selected = read_codex_config(path, profile, use_default_profile)
        return cls(
            path=path,
            profile=selected["profile"],
            provider=ResponsesProviderConfig.from_mapping(selected),
            context=ContextConfig.from_mapping(selected, path),
        )
