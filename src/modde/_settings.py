"""Build core ``Settings`` from keyword arguments (registry names).

Private helper; Phase 3 builds the public ``settings_from_dict`` /
``settings_from_config`` on top of it.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import _core

_MODULE_FIELDS = frozenset(n for n in dir(_core.Modules) if not n.startswith("_"))
_READ_ONLY = frozenset({"resolved", "provenance", "warnings", "modules", "dim"})
_VECTORS = frozenset({"lb", "ub", "initial_fitness"})


def _plain(value: Any) -> Any:
    """numpy scalars (e.g. from ConfigSpace) -> Python scalars."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def make_settings(dim: int, **kwargs: Any) -> _core.Settings:
    """Create ``_core.Settings`` for ``dim`` from registry-named keyword arguments.

    Module options may be given as strings (case-insensitive) or core enums.
    ``lb``/``ub`` may be scalars (broadcast to ``dim``).
    """
    settings = _core.Settings(int(dim))
    modules = _core.Modules()
    for name, value in kwargs.items():
        value = _plain(value)
        if name in _MODULE_FIELDS:
            current = getattr(modules, name)
            if isinstance(value, str):
                enum_type = type(current)
                try:
                    value = enum_type[value.upper()]
                except KeyError:
                    options = [e.name for e in enum_type]
                    raise ValueError(f"{name}: unknown option {value!r}; options are {options}") from None
            setattr(modules, name, value)
        elif name in _READ_ONLY:
            raise ValueError(f"{name} cannot be set here")
        elif hasattr(settings, name):
            if value is not None and name in _VECTORS:
                value = np.broadcast_to(np.asarray(value, dtype=float), (dim,)).copy() \
                    if name != "initial_fitness" else np.asarray(value, dtype=float).ravel()
            elif value is not None and name == "initial_population":
                value = np.atleast_2d(np.asarray(value, dtype=float))
            setattr(settings, name, value)
        else:
            raise ValueError(f"unknown setting {name!r}")
    settings.modules = modules
    return settings
