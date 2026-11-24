"""modde: Modular Differential Evolution with a C++ core.

Quick start::

    import numpy as np, modde
    x, f, evals, opt = modde.fmin(lambda x: float(np.sum(x**2)), budget=2000, dim=5, lb=-5, ub=5)

Main entry points: :func:`fmin`, :class:`ModularDE`, :class:`AskTellDE`,
:func:`get_configspace`, :func:`get_presets`. The 0.0.4 implementation is in
:mod:`modde.legacy` (deprecated).
"""
from __future__ import annotations

from importlib import metadata as _metadata

from . import _core
from ._core import (AdaptationCR, AdaptationF, ArchiveContent, BaseSampler, BoundCorrectionType,
                    CrossoverType,
                    Modules, MutationBase, MutationReference, OppositionalInitialization,
                    PbestSampling, PopulationReduction, Provenance, Settings)
from ._settings import settings_from_dict, settings_to_dict
from .asktell import AskTellDE
from .configspace import (config_from_settings, get_configspace, get_presets, preset_names,
                          preset_settings, settings_from_config)
from .optimizer import FminResult, ModularDE, fmin, wrap_problem
from . import irace  # irace export (D-C10)

try:
    __version__ = _metadata.version("modde")
except _metadata.PackageNotFoundError:  # running from a source tree without install
    __version__ = "0+unknown"

# 0.0.4 names without a counterpart in the new API.
_REMOVED = frozenset({"Parameters", "TrackedStats", "Population"})


def __getattr__(name: str):
    if name in _REMOVED:
        raise AttributeError(
            f"modde.{name} belonged to the 0.0.4 API. Use modde.legacy.{name} (deprecated), or the "
            "new API: modde.ModularDE / modde.settings_from_dict (see the migration guide).")
    raise AttributeError(f"module 'modde' has no attribute {name!r}")


__all__ = (
    "__version__",
    "fmin", "FminResult", "ModularDE", "AskTellDE", "wrap_problem",
    "settings_from_dict", "settings_to_dict",
    "get_configspace", "get_presets", "preset_names", "preset_settings",
    "settings_from_config", "config_from_settings", "irace",
    "Settings", "Modules", "Provenance",
    "BaseSampler", "OppositionalInitialization", "MutationBase", "MutationReference",
    "PbestSampling", "ArchiveContent", "CrossoverType", "BoundCorrectionType", "AdaptationF",
    "AdaptationCR", "PopulationReduction",
)

del annotations  # from the __future__ import; not part of the API
