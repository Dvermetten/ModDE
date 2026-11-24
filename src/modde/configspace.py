"""ConfigSpace interface for algorithm configuration (D-C*) and named presets (D-D2)."""
from __future__ import annotations

from typing import Any

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.util import deactivate_inactive_hyperparameters

from . import _core
from . import _registry as R
from ._settings import settings_from_dict, settings_to_dict


def get_configspace(
    dim: int,
    add_popsize: bool = True,
    add_initial_FCR: bool = True,
    add_module_params: bool = True,
) -> ConfigurationSpace:
    """The configuration space of modde for a ``dim``-dimensional problem.

    Always contains every module (categorical). Numeric groups can be left out:

    - ``add_popsize``: ``lambda0`` (integer, log scale; range depends on ``dim``)
    - ``add_initial_FCR``: ``F0``, ``CR0``
    - ``add_module_params``: parameters of individual modules (archive rate,
      memory size, p-best rates, jDE/SHADE constants, ...), each conditional on
      its module being active.

    The default configuration is classic DE/rand/1/bin. Every configuration in
    the space is valid (no forbidden clauses are needed).
    """
    groups = ["modules"]
    if add_popsize:
        groups.append("population")
    if add_initial_FCR:
        groups.append("initial_FCR")
    if add_module_params:
        groups.append("module_params")
    return R._build_configspace(int(dim), tuple(groups))


def preset_names() -> list[str]:
    """Names of the available presets."""
    return list(R.PRESETS)


def preset_settings(name: str, dim: int) -> dict[str, Any]:
    """The settings (registry names, only non-defaults) of a preset for ``dim``.
    Names are matched case-insensitively."""
    by_lower = {key.lower(): key for key in R.PRESETS}
    if str(name).lower() not in by_lower:
        raise ValueError(f"unknown preset {name!r}; available: {preset_names()}")
    preset = R.PRESETS[by_lower[str(name).lower()]]
    return dict(preset.settings(int(dim)))


def get_presets(dim: int, cs: ConfigurationSpace | None = None) -> dict[str, Configuration]:
    """All presets as configurations of ``get_configspace(dim)`` (D-D2).

    A preset configuration can be edited and passed to ``settings_from_config``,
    ``ModularDE(config=...)`` or ``fmin(config=...)``, or used to warm-start a tuner.
    """
    cs = cs or get_configspace(dim)
    return {name: R._preset_configuration(name, int(dim), cs) for name in R.PRESETS}


def settings_from_config(dim: int, config: Configuration | dict, **overrides: Any) -> _core.Settings:
    """Settings from a ConfigSpace configuration; ``overrides`` (e.g. ``lb``, ``ub``,
    ``budget``, ``seed``) are added on top. Every configured value counts as set by
    the user (provenance USER)."""
    values = dict(config)
    values.update(overrides)
    return settings_from_dict(dim, **values)


def config_from_settings(settings: _core.Settings, cs: ConfigurationSpace | None = None) -> Configuration:
    """The configuration of ``cs`` (default: ``get_configspace(settings.dim)``) that
    corresponds to ``settings``. Unset numeric values take their defaults."""
    cs = cs or get_configspace(settings.dim)
    given = settings_to_dict(settings)
    values = {hp.name: hp.default_value for hp in cs.values()}
    for name in values:
        if name in given:
            values[name] = given[name]
    try:
        return deactivate_inactive_hyperparameters(values, cs)
    except Exception as exc:  # out-of-range values etc.
        raise ValueError(f"settings do not fit the configuration space: {exc}") from exc
