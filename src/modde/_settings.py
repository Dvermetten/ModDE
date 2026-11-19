"""Conversion between keyword arguments (registry names) and core ``Settings``."""
from __future__ import annotations

from typing import Any

import numpy as np

from . import _core
from ._registry import PARAMS

_MODULE_FIELDS = frozenset(n for n in dir(_core.Modules) if not n.startswith("_"))
_READ_ONLY = frozenset({"resolved", "provenance", "warnings", "modules", "dim"})
_VECTORS = frozenset({"lb", "ub"})


def _plain(value: Any) -> Any:
    """numpy scalars (e.g. from ConfigSpace) -> Python scalars."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def settings_from_dict(dim: int, **kwargs: Any) -> _core.Settings:
    """Create (unresolved) ``Settings`` for a ``dim``-dimensional problem.

    Keyword arguments use the registry names (see ``modde._registry``), e.g.
    ``settings_from_dict(10, lb=-5, ub=5, mutation_base="target", lambda0=50)``.
    Module options may be strings (case-insensitive) or core enums; ``lb``/``ub``
    may be scalars (broadcast to ``dim``). Resolution (defaults, validation,
    provenance) happens when an optimizer is created, or via ``Settings.resolve()``.
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
                value = np.broadcast_to(np.asarray(value, dtype=float), (int(dim),)).copy()
            elif value is not None and name == "initial_fitness":
                value = np.asarray(value, dtype=float).ravel()
            elif value is not None and name == "initial_population":
                value = np.atleast_2d(np.asarray(value, dtype=float))
            setattr(settings, name, value)
        else:
            raise ValueError(f"unknown setting {name!r}")
    settings.modules = modules
    return settings


#: Backwards-compatible private alias (used by the tests).
make_settings = settings_from_dict


def settings_to_dict(settings: _core.Settings) -> dict[str, Any]:
    """All registry-named values of ``settings`` that are set (modules as option names)."""
    out: dict[str, Any] = {}
    for p in PARAMS:
        if p.name in _MODULE_FIELDS:
            value = getattr(settings.modules, p.name)
            out[p.name] = value.name if hasattr(value, "name") else value
        elif hasattr(settings, p.name):
            value = getattr(settings, p.name)
            if value is not None:
                out[p.name] = value
    return out
