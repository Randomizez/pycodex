"""Antigravity (agy 1.5.13) model catalog and generationConfig resolution."""

from dataclasses import dataclass
import os
from types import MappingProxyType
import typing


AGY_USER_AGENT = (
    "google-genai-sdk/1.79.1 gl-go/go1.28-20261102-RC01 "
    "cl/1012564305 +298f53e483 X:fieldtrack,boringcrypto,nogolistexportnewformat"
)

_LEVEL_DISPLAY = MappingProxyType(
    {
        "low": "Low",
        "medium": "Medium",
        "high": "High",
        "xhigh": "XHigh",
    }
)

_EFFORT_NORMALIZATION = MappingProxyType(
    {
        "none": "low",
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",
        "max": "high",
    }
)

_KNOWN_BASES = MappingProxyType(
    {
        "gemini-4-neon": ("gemini-4-neon", "Gemini 4 Neon", 65536, "level", 32768, 512000),
        "gemini-4-carbon": ("gemini-4-carbon", "Gemini 4 Carbon", 65536, "level", 32768, 512000),
        "gemini-4-hydrogen": ("gemini-4-hydrogen", "Gemini 4 Hydrogen", 65536, "level", 16384, 256000),
        "gemini-4-argon": ("gemini-4-argon", "Gemini 4 Argon", 65536, "level", 32768, 512000),
        "gemini-3.8-flash": ("gemini-3.8-flash", "Gemini 3.8 Flash", 65536, "level", 32768, 256000),
        "gemini-3.7-flash": ("gemini-3.7-flash", "Gemini 3.7 Flash", 65536, "level", 16384, 256000),
        "gemini-3.6-flash": ("gemini-3.6-flash", "Gemini 3.6 Flash", 65536, "level", 16384, 256000),
        "gemini-3.1-pro": ("gemini-3.1-pro-preview", "Gemini 3.1 Pro", 65535, "budget", 16384, 256000),
        "gemini-3.1-pro-preview": ("gemini-3.1-pro-preview", "Gemini 3.1 Pro", 65535, "budget", 16384, 256000),
    }
)

_XHIGH_BASES = frozenset({"gemini-4-neon", "gemini-4-hydrogen"})

OFFICIAL_MODELS: "typing.Tuple[typing.Tuple[str, str], ...]" = (
    ("gemini-4-neon-high", "Gemini 4 Neon (High)"),
    ("gemini-4-neon-xhigh", "Gemini 4 Neon (XHigh)"),
    ("gemini-4-neon-medium", "Gemini 4 Neon (Medium)"),
    ("gemini-4-neon-low", "Gemini 4 Neon (Low)"),
    ("gemini-4-carbon-high", "Gemini 4 Carbon (High)"),
    ("gemini-4-carbon-medium", "Gemini 4 Carbon (Medium)"),
    ("gemini-4-carbon-low", "Gemini 4 Carbon (Low)"),
    ("gemini-4-hydrogen-high", "Gemini 4 Hydrogen (High)"),
    ("gemini-4-hydrogen-xhigh", "Gemini 4 Hydrogen (XHigh)"),
    ("gemini-4-hydrogen-medium", "Gemini 4 Hydrogen (Medium)"),
    ("gemini-4-hydrogen-low", "Gemini 4 Hydrogen (Low)"),
    ("gemini-4-argon-high", "Gemini 4 Argon (High)"),
    ("gemini-4-argon-medium", "Gemini 4 Argon (Medium)"),
    ("gemini-4-argon-low", "Gemini 4 Argon (Low)"),
    ("gemini-3.8-flash-high", "Gemini 3.8 Flash (High)"),
    ("gemini-3.8-flash-medium", "Gemini 3.8 Flash (Medium)"),
    ("gemini-3.8-flash-low", "Gemini 3.8 Flash (Low)"),
    ("gemini-3.7-flash-high", "Gemini 3.7 Flash (High)"),
    ("gemini-3.7-flash-medium", "Gemini 3.7 Flash (Medium)"),
    ("gemini-3.7-flash-low", "Gemini 3.7 Flash (Low)"),
    ("gemini-3.6-flash-high", "Gemini 3.6 Flash (High)"),
    ("gemini-3.6-flash-medium", "Gemini 3.6 Flash (Medium)"),
    ("gemini-3.6-flash-low", "Gemini 3.6 Flash (Low)"),
    ("gemini-3.1-pro-high", "Gemini 3.1 Pro (High)"),
    ("gemini-3.1-pro-low", "Gemini 3.1 Pro (Low)"),
)


@dataclass(frozen=True)
class AgyModelSpec:
    requested_model: "str"
    api_model: "str"
    display_name: "str"
    max_output_tokens: "int"
    thinking_config: "typing.Dict[str, object]"
    compaction_max_output_tokens: "int" = 32768
    auto_compact_threshold: "typing.Optional[int]" = None

    def generation_config(self) -> "typing.Dict[str, object]":
        return {
            "maxOutputTokens": self.max_output_tokens,
            "thinkingConfig": dict(self.thinking_config),
        }

    def compaction_generation_config(self) -> "typing.Dict[str, object]":
        return {
            "maxOutputTokens": self.compaction_max_output_tokens,
            "thinkingConfig": dict(self.thinking_config),
        }

    def resolve_auto_compact_threshold(
        self,
        env_override: "typing.Optional[str]" = None,
    ) -> "typing.Optional[int]":
        raw = (
            env_override
            if env_override is not None
            else os.environ.get("AGY_CLI_AUTO_COMPACT_OVERRIDE")
        )
        if raw is None or not str(raw).strip():
            return self.auto_compact_threshold
        spec = str(raw).strip()
        if spec.endswith("%"):
            num_text = spec[:-1].strip()
            if not num_text.isdigit():
                raise ValueError(f"invalid AGY_CLI_AUTO_COMPACT_OVERRIDE: {raw!r}")
            pct = int(num_text)
            if pct < 1 or pct > 100:
                raise ValueError(f"invalid AGY_CLI_AUTO_COMPACT_OVERRIDE: {raw!r}")
            window = (
                self.auto_compact_threshold * 2
                if self.auto_compact_threshold is not None
                else 1000000
            )
            return (pct * (window >> 1)) // 50
        if not spec.isdigit():
            raise ValueError(f"invalid AGY_CLI_AUTO_COMPACT_OVERRIDE: {raw!r}")
        val = int(spec)
        if val <= 0 or val > 0xFFFFFFFF:
            raise ValueError(f"invalid AGY_CLI_AUTO_COMPACT_OVERRIDE: {raw!r}")
        return val


def normalize_effort_level(effort: "typing.Optional[str]", default: "str" = "high") -> "str":
    if effort is None or not str(effort).strip():
        return default
    raw = str(effort).strip().lower()
    if raw not in _EFFORT_NORMALIZATION:
        raise ValueError(f"unsupported reasoning effort: {effort!r}")
    return _EFFORT_NORMALIZATION[raw]


def _split_known_suffix(model: "str") -> "typing.Tuple[str, typing.Optional[str]]":
    for suffix in ("xhigh", "high", "medium", "low"):
        token = "-" + suffix
        if model.endswith(token):
            base = model[: -len(token)]
            if base in _KNOWN_BASES:
                return base, suffix
    return model, None


def resolve_agy_model(
    model: "str",
    config_effort: "typing.Optional[str]" = None,
    explicit_effort: "typing.Optional[str]" = None,
) -> "AgyModelSpec":
    raw_model = str(model or "").strip() or "gemini-3.1-pro-high"
    model_part, route_sep, route_suffix = raw_model.partition(":")
    base, suffix_level = _split_known_suffix(model_part)

    if base in _KNOWN_BASES:
        (
            api_model,
            display_base,
            max_tokens,
            mode,
            compaction_max_tokens,
            auto_compact_threshold,
        ) = _KNOWN_BASES[base]
        effective_api_model = (
            f"{api_model}:{route_suffix}" if route_sep and route_suffix else api_model
        )
        if explicit_effort is not None:
            level = normalize_effort_level(explicit_effort)
        elif suffix_level is not None:
            level = suffix_level
        else:
            level = normalize_effort_level(config_effort, default="high")

        if mode == "budget":
            is_low = level == "low"
            budget = 1024 if is_low else -1
            label = "Low" if is_low else "High"
            return AgyModelSpec(
                requested_model=raw_model,
                api_model=effective_api_model,
                display_name=f"{display_base} ({label})",
                max_output_tokens=max_tokens,
                thinking_config={
                    "includeThoughts": True,
                    "thinkingBudget": budget,
                },
                compaction_max_output_tokens=compaction_max_tokens,
                auto_compact_threshold=auto_compact_threshold,
            )

        if level == "xhigh" and base not in _XHIGH_BASES:
            level = "high"
        title_level = _LEVEL_DISPLAY[level]
        return AgyModelSpec(
            requested_model=raw_model,
            api_model=effective_api_model,
            display_name=f"{display_base} ({title_level})",
            max_output_tokens=max_tokens,
            thinking_config={
                "includeThoughts": True,
                "thinkingLevel": level.upper(),
            },
            compaction_max_output_tokens=compaction_max_tokens,
            auto_compact_threshold=auto_compact_threshold,
        )

    effective_level = normalize_effort_level(
        explicit_effort if explicit_effort is not None else config_effort,
        default="high",
    )
    return AgyModelSpec(
        requested_model=raw_model,
        api_model=raw_model,
        display_name=raw_model,
        max_output_tokens=65536,
        thinking_config={
            "includeThoughts": True,
            "thinkingLevel": effective_level.upper(),
        },
        compaction_max_output_tokens=32768,
        auto_compact_threshold=None,
    )


def resolve_subagent_model(
    tier: "typing.Optional[str]",
    parent_model: "str",
) -> "str":
    normalized = (tier or "inherit").strip().lower()
    if normalized in ("", "inherit"):
        return parent_model
    if normalized == "flash_lite":
        return "gemini-3.6-flash-low"
    if normalized == "flash":
        return "gemini-3.8-flash-high"
    if normalized == "pro":
        return "gemini-4-carbon-high"
    raise ValueError(f"unsupported subagent Model tier: {tier!r}")
