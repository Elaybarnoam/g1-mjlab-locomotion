"""G1 standing and walking research workflows built on mjlab.

Importing this package is deliberately CPU-only. Simulator imports live in
``g1_mjlab.runtime`` and are loaded only by commands that need them.
"""

from importlib.metadata import PackageNotFoundError, version

from .config import ResolvedRunConfig, load_config

__all__ = ["ResolvedRunConfig", "load_config"]
try:
    __version__ = version("g1-mjlab-locomotion")
except PackageNotFoundError:  # Source checkout without an installed distribution.
    __version__ = "0+unknown"
