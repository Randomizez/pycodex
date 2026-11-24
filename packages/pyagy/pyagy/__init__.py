"""Antigravity CLI (agy 1.5.14) wire-compatible mode built on pycodex."""

from pycodex.utils.compat import importlib_metadata

try:
    __version__ = importlib_metadata.version("pyagy")
except importlib_metadata.PackageNotFoundError:  # pragma: no cover - uninstalled checkout
    __version__ = "0.7.4"
