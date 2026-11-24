"""Conversion between keyword arguments (registry names) and core ``Settings``."""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from . import _core
from ._registry import PARAMS

_MODULE_FIELDS = frozenset(n for n in dir(_core.Modules) if not n.startswith("_"))
_READ_ONLY = frozenset({"resolved", "provenance", "warnings", "modules", "problem_fields"})
_VECTORS = frozenset({"lb", "ub"})

# 0.0.4 names -> new names, from the registry's `replaces` notes (for error messages).
_OLD_NAMES: dict[str, str] = {}
for _p in PARAMS:
    if _p.replaces:
        _match = re.match(r"[A-Za-z_]+", _p.replaces)
        if _match and _match.group(0) != _p.name:
            _OLD_NAMES.setdefault(_match.group(0), _p.name)


def _plain(value: Any) -> Any:
    """numpy scalars (e.g. from ConfigSpace) -> Python scalars."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _as_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{name}: expected True or False, got {value!r}")


def _set(target: Any, name: str, value: Any) -> None:
    """setattr with a readable error message (instead of nanobind's signature dump)."""
    try:
        setattr(target, name, value)
        return
    except TypeError:
        pass
    # whole-number floats (e.g. 1e4) are accepted for integer settings
    if isinstance(value, float) and value.is_integer():
        try:
            setattr(target, name, int(value))
            return
        except TypeError:
            pass
    current = getattr(target, name)
    expected = type(current).__name__ if current is not None else "a value of the documented type"
    p = next((p for p in PARAMS if p.name == name), None)
    if p is not None:
        expected = {"int": "a non-negative integer", "float": "a number", "bool": "True or False",
                    "vector": "a number or an array of length dim", "matrix": "an (n, dim) array",
                    "categorical": f"one of {list(p.options)}"}.get(p.kind, expected)
    raise ValueError(f"{name}: expected {expected}, got {value!r}") from None


def settings_from_dict(dim: int, **kwargs: Any) -> _core.Settings:
    """Create (unresolved) ``Settings`` for a ``dim``-dimensional problem.

    Keyword arguments use the registry names (see the Parameters page), e.g.
    ``settings_from_dict(10, lb=-5, ub=5, mutation_base="target", lambda0=50)``.
    Module options may be strings (case-insensitive) or core enums; booleans may
    also be ``"true"``/``"false"``; ``lb``/``ub`` may be scalars (broadcast to
    ``dim``). Resolution (defaults, validation, provenance) happens when an
    optimizer is created, or via ``Settings.resolve()``.
    """
    settings = _core.Settings(int(dim))
    modules = _core.Modules()
    for name, value in kwargs.items():
        value = _plain(value)
        if name == "dim":
            if value is not None and int(value) != int(dim):
                raise ValueError(f"dim = {value} does not match dim = {dim}")
            continue
        if name in _MODULE_FIELDS:
            current = getattr(modules, name)
            if isinstance(current, bool):
                value = _as_bool(name, value)
            elif isinstance(value, str) and hasattr(type(current), "__members__"):
                enum_type = type(current)
                try:
                    value = enum_type[value.upper()]
                except KeyError:
                    options = [e.name for e in enum_type]
                    raise ValueError(f"{name}: unknown option {value!r}; options are {options}") from None
            _set(modules, name, value)
        elif name in _READ_ONLY:
            raise ValueError(f"{name} cannot be set here")
        elif hasattr(settings, name):
            if value is not None and name in _VECTORS:
                value = np.broadcast_to(np.asarray(value, dtype=float), (int(dim),)).copy()
            elif value is not None and name == "initial_fitness":
                value = np.asarray(value, dtype=float).ravel()
            elif value is not None and name == "initial_population":
                value = np.atleast_2d(np.asarray(value, dtype=float))
            elif value is not None and name == "track_evaluations":
                value = _as_bool(name, value)
            _set(settings, name, value)
        else:
            hint = f"; in 0.0.4 this was {name!r}, now use {_OLD_NAMES[name]!r}" if name in _OLD_NAMES else ""
            raise ValueError(f"unknown setting {name!r}{hint} (see the Parameters page for all settings)")
    settings.modules = modules
    return settings


#: Backwards-compatible private alias (used by the tests).
make_settings = settings_from_dict


def settings_to_dict(settings: _core.Settings, user_only: bool = False) -> dict[str, Any]:
    """The registry-named values of ``settings`` (modules as option names).

    By default every value that is set is included: for resolved settings (e.g.
    ``opt.settings``) that is the complete configuration of a run, including
    defaults and the seed. With ``user_only=True`` only what was set explicitly is
    returned: module options that differ from the default, values with provenance
    ``USER``, and the seed. That dictionary reproduces the run when passed back
    (``settings_from_dict(dim, **d)`` or ``ModularDE(problem, **d)``) without
    turning defaults into explicit values (which could trigger "no effect"
    warnings). ``dim`` is never included.

    Values are as stored in the core, i.e. in minimisation sign. For an optimizer
    on a maximisation problem use ``opt.settings_dict()``, which converts ``target``
    and ``initial_fitness`` back to the problem's sign.
    """
    out: dict[str, Any] = {}
    provenance = dict(settings.provenance) if settings.resolved else {}
    defaults = _core.Modules()
    for p in PARAMS:
        if p.name == "dim":
            continue
        if p.name in _MODULE_FIELDS:
            value = getattr(settings.modules, p.name)
            if user_only and value == getattr(defaults, p.name):
                continue
            out[p.name] = value.name if hasattr(value, "name") else value
        elif hasattr(settings, p.name):
            value = getattr(settings, p.name)
            if value is None:
                continue
            if user_only and settings.resolved and p.name != "seed" and \
                    provenance.get(p.name) not in (_core.Provenance.USER, None):
                continue
            out[p.name] = value
    return out
