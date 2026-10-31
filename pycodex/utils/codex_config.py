import re
import typing
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 and earlier
    import tomli as tomllib

DEFAULT_CODEX_CONFIG_PATH = Path.home() / ".codex" / "config.toml"


def read_codex_config(
    config_path: "typing.Union[str, Path]",
    profile: "typing.Union[str, None]" = None,
    use_default_profile: "bool" = True,
) -> "typing.Dict[str, typing.Any]":
    """Merge a file or embedded profile over the base config.

    Disable default selection when restoring an explicitly saved base config.
    The returned ``profile`` is the effective name, or None for the base config.
    """
    path = Path(config_path)
    selected = _read_config_file(path)
    profiles = _embedded_profiles(selected, path)
    if profile is None and use_default_profile:
        profile = selected.get("profile")
    if profile is not None:
        _validate_profile_name(profile)
        profile_path = path.parent / (profile + ".config.toml")
        if profile in profiles:
            if profile_path.exists():
                _raise_profile_conflict(path, profile)
            overrides = profiles[profile]
        else:
            overrides = _read_config_file(profile_path)
        if "profile" in overrides:
            raise ValueError(
                "Codex profile {!r} cannot select another profile".format(profile)
            )
        selected = _merge_config(selected, overrides)
    selected["profile"] = profile
    return selected


def list_codex_profiles(
    config_path: "typing.Union[str, Path]",
) -> "typing.List[str]":
    """List file and embedded profiles, rejecting ambiguous names."""
    config_path = Path(config_path)
    profiles = _embedded_profiles(_read_config_file(config_path), config_path)
    suffix = ".config.toml"
    files = {
        path.name[:-len(suffix)]
        for path in config_path.parent.glob("*" + suffix)
        if path.is_file()
        and re.fullmatch(r"[A-Za-z0-9_-]+", path.name[:-len(suffix)])
    }
    for profile in sorted(files.intersection(profiles)):
        _raise_profile_conflict(config_path, profile)
    return sorted(files.union(profiles))


def _validate_profile_name(profile: "str") -> "None":
    if not isinstance(profile, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", profile):
        raise ValueError("invalid Codex profile name: {!r}".format(profile))


def _embedded_profiles(
    data: "typing.Dict[str, typing.Any]", path: "Path",
) -> "typing.Dict[str, typing.Any]":
    profiles = data.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("profiles in {} must be a table".format(path))
    for name, settings in profiles.items():
        _validate_profile_name(name)
        if not isinstance(settings, dict):
            raise ValueError("[profiles.{}] in {} must be a table".format(name, path))
    return profiles


def _raise_profile_conflict(path: "Path", profile: "str") -> "None":
    raise ValueError(
        "ambiguous Codex profile {0!r}: both [profiles.{0}] in {1} and {2} "
        "exist; keep only one definition".format(
            profile, path, path.parent / (profile + ".config.toml"),
        )
    )


def _merge_config(
    base: "typing.Dict[str, typing.Any]",
    overrides: "typing.Dict[str, typing.Any]",
) -> "typing.Dict[str, typing.Any]":
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_config_file(path: "Path") -> "typing.Dict[str, typing.Any]":
    return tomllib.loads(path.read_text(encoding="utf-8"))
