"""Helpers shared by ``ModularDE`` and ``AskTellDE``: input checks, building the
settings from preset / config / keywords, warnings, and the tracked attributes."""
from __future__ import annotations

import numbers
import warnings
from typing import Any

import numpy as np

from . import _core
from . import _registry as R
from ._settings import settings_from_dict, settings_to_dict

#: Attributes of the optimizer state that an ioh logger can watch (D-B8).
TRACKED = ("curr_F", "curr_CR", "corrected", "corr_so_far", "CS", "ED", "popmean", "popstd",
           "F_memory_mean", "generation")


def as_int(name: str, value: Any, minimum: int = 0) -> int:
    """An integer argument; whole-number floats (1e4) are accepted, others are an error."""
    value = value.item() if isinstance(value, np.generic) else value
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if int(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value!r}")
    return int(value)


def collect_values(dim: int, preset: str | None, config: Any, kwargs: dict[str, Any]
                   ) -> tuple[dict[str, Any], set[str]]:
    """Settings from a preset or a configuration, with keyword arguments on top.

    Returns the values and the names that only the preset set (not the user)."""
    from .configspace import preset_settings

    if preset is not None and config is not None:
        raise ValueError("pass either `preset` or `config`, not both (a configuration is complete "
                         "and would replace the preset; get_presets(dim)[name] gives a preset as a "
                         "configuration to edit)")
    values: dict[str, Any] = {}
    preset_only: set[str] = set()
    if preset is not None:
        values.update(preset_settings(preset, dim))
        preset_only = set(values) - set(kwargs)
    if config is not None:
        values.update(dict(config))
    values.update(kwargs)
    return values, preset_only


def check_settings_alone(**given: Any) -> None:
    """With complete core ``settings``, the other ways of configuring must not be used."""
    used = [name for name, value in given.items()
            if value is not None and not (isinstance(value, dict) and not value)]
    if used:
        raise ValueError(f"`settings` is complete; it cannot be combined with {', '.join(used)} "
                         "(set these values in the settings, e.g. with settings_from_dict)")


def emit_warnings(core: _core.ModularDE, preset_only: set[str], user_modules: set[str],
                  stacklevel: int = 3) -> list[str]:
    """Warn about settings without effect (after resolution).

    - Warnings from the core about values that came from the preset (not from the
      user) are dropped: the preset is not wrong just because the user changed a module.
    - Module options set by the user that are inactive with the chosen modules
      (e.g. ``archive_content`` without ``use_archive``) are reported too.
    """
    messages = [m for m in core.settings.warnings
                if not any(m == f"{name} has no effect with the current modules" for name in preset_only)]
    full = settings_to_dict(core.settings)
    for name in sorted(user_modules):
        p = R.BY_NAME.get(name)
        if p is not None and p.is_module and p.active_when and not R.is_active(p, full):
            messages.append(f"{name} has no effect with the current modules")
    for message in messages:
        warnings.warn(message, UserWarning, stacklevel=stacklevel)
    return messages


def warn_nan(core: _core.ModularDE, state: dict) -> None:
    """Warn once per optimizer when the objective returned NaN (treated as +inf)."""
    n = int(core.stats.nan_evaluations)
    if n and not state.get("nan_warned"):
        state["nan_warned"] = True
        warnings.warn(f"the objective returned NaN ({n} time{'s' if n > 1 else ''} so far); NaN is "
                      "treated as +inf (worse than any number)", RuntimeWarning, stacklevel=4)


class TrackedMixin:
    """The tracked attributes, read from the core statistics (D-B8).

    They describe the **last evaluated point** (``curr_F``, ``curr_CR``, ``corrected``,
    ``CS``, ``ED``; per point only with ``track_evaluations=True``, NaN otherwise) or the
    **population after the last generation** (``popmean``, ``popstd``, ``F_memory_mean``).
    """

    core: _core.ModularDE

    @property
    def curr_F(self) -> float:
        """F used for the last evaluated point (NaN unless ``track_evaluations=True``)."""
        return self.core.stats.curr_F

    @property
    def curr_CR(self) -> float:
        """CR used for the last evaluated point (NaN unless ``track_evaluations=True``)."""
        return self.core.stats.curr_CR

    @property
    def corrected(self) -> float:
        """1.0 if bound correction changed the last evaluated point, else 0.0."""
        return float(self.core.stats.corrected)

    @property
    def corr_so_far(self) -> int:
        """Number of evaluated points that bound correction changed so far."""
        return self.core.stats.corr_so_far

    @property
    def CS(self) -> float:
        """Cosine similarity of the steps from the parent to the corrected and to the uncorrected
        trial (normalised space; 0 if the last point was not corrected)."""
        return self.core.stats.CS

    @property
    def ED(self) -> float:
        """Euclidean distance between the corrected and the uncorrected trial (normalised space;
        0 if the last point was not corrected)."""
        return self.core.stats.ED

    @property
    def popmean(self) -> float:
        """Mean over all coordinates of all individuals (one number)."""
        return self.core.stats.popmean

    @property
    def popstd(self) -> float:
        """Standard deviation over all coordinates of all individuals pooled (not a diversity measure)."""
        return self.core.stats.popstd

    @property
    def F_memory_mean(self) -> float:
        """Mean of the SHADE F memory (NaN without SHADE F adaptation)."""
        return self.core.stats.F_memory_mean

    @property
    def generation(self) -> int:
        """Number of completed generations."""
        return self.core.stats.generation


def writable_core_properties() -> frozenset[str]:
    """Writable state properties of the core optimizer (population, fitness, F, ...)."""
    return frozenset(name for name in dir(_core.ModularDE)
                     if isinstance(getattr(_core.ModularDE, name, None), property)
                     and getattr(_core.ModularDE, name).fset is not None)
