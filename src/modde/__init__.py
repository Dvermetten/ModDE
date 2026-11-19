"""modde: Modular Differential Evolution.

This is the development version of the C++-backed rewrite. The public API is
added in Phase 3 of the development plan (``dev/README.md``). Until then the
0.0.4 names are forwarded to :mod:`modde.legacy` with a ``DeprecationWarning``.
"""
from __future__ import annotations

import warnings
from importlib import metadata as _metadata

from . import _core

try:
    __version__ = _metadata.version("modde")
except _metadata.PackageNotFoundError:  # running from a source tree without install
    __version__ = "0+unknown"

# 0.0.4 names that are forwarded to modde.legacy until the new API replaces them.
_LEGACY_NAMES = frozenset(
    {"ModularDE", "AskTellDE", "Parameters", "TrackedStats", "Population"}
)


def __getattr__(name: str):
    if name in _LEGACY_NAMES:
        warnings.warn(
            f"modde.{name} currently refers to the deprecated 0.0.4 implementation "
            f"(modde.legacy.{name}). Import it from modde.legacy explicitly; the "
            "top-level name will point to the new API in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from . import legacy
        return getattr(legacy, name)
    raise AttributeError(f"module 'modde' has no attribute {name!r}")


__all__ = ("__version__",)
