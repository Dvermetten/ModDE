"""Ask-tell interface (D-A12): generation-level ``ask``/``tell`` plus a
per-individual convenience layer. Uses the same core building blocks as
``ModularDE.run``, so the same seed gives the same trajectory (D-D12)."""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from . import _core
from ._common import (TrackedMixin, as_int, check_settings_alone, collect_values, emit_warnings,
                      warn_nan)
from ._settings import _MODULE_FIELDS, settings_from_dict


class AskTellDE(TrackedMixin):
    """Modular DE with external evaluation (minimisation).

    >>> at = AskTellDE(dim=5, lb=-5, ub=5, budget=1000)
    >>> while not at.stop:
    ...     X = at.ask()                       # (n, d) points
    ...     at.tell([f(x) for x in X])          # their values, same order

    Or one point at a time: ``i, x = at.ask_one(); at.tell_one(i, f(x))``.
    Use one of the two styles per batch.

    - The budget (default ``10000 * dim``) is counted by the optimizer. Near its end,
      ``ask()`` returns fewer points than the population size.
    - A batch can be trial vectors or, with oppositional generation, opposite points;
      both are simply evaluated and told.
    - ``ask()`` raises ``StopIteration`` once the run is finished (check ``stop``).
    - The optimizer minimises; to maximise, tell ``-f``.
    - NaN values are treated as +inf (with a warning).

    Settings as for :class:`ModularDE`: ``preset`` or ``config``, plus keyword
    settings, or complete core ``settings`` (then ``dim``, ``lb`` and ``ub`` come from
    them and nothing else may be given).
    """

    def __init__(self, dim: int | None = None, lb: Any = None, ub: Any = None, *,
                 budget: int | None = None, preset: str | None = None, config: Any = None,
                 settings: _core.Settings | None = None, **kwargs: Any) -> None:
        object.__setattr__(self, "_state", {})
        preset_only: set[str] = set()
        user_modules: set[str] = set()
        if settings is not None:
            check_settings_alone(lb=lb, ub=ub, budget=budget, preset=preset, config=config,
                                 **{"keyword settings": kwargs})
            if dim is not None and as_int("dim", dim) != settings.dim:
                raise ValueError(f"dim = {dim} does not match settings.dim = {settings.dim}")
        else:
            if dim is None or lb is None or ub is None:
                raise ValueError("dim, lb and ub are required")
            dim = as_int("dim", dim, minimum=1)
            values, preset_only = collect_values(dim, preset, config, kwargs)
            user_modules = {k for k in (set(kwargs) | set(dict(config or {}))) if k in _MODULE_FIELDS}
            if budget is not None:
                values["budget"] = as_int("budget", budget, minimum=1)
            settings = settings_from_dict(dim, lb=lb, ub=ub, **values)
        self.core = _core.ModularDE(settings)
        emit_warnings(self.core, preset_only, user_modules)
        self._pending: tuple[str, np.ndarray, Any] | None = None
        self._single: dict[str, Any] | None = None

    # ------------------------------------------------------------------ batch interface

    @property
    def stop(self) -> bool:
        """True once a termination condition holds (and nothing is pending)."""
        return self._pending is None and self.core.initialized and self.core.break_conditions()

    def ask(self) -> np.ndarray:
        """The next points to evaluate, shape (n, d). Asking again before
        ``tell`` returns the same points. Raises StopIteration when finished."""
        if self._pending is not None:
            return self._pending[1].copy()
        core = self.core
        if not core.initialized:
            C = core.initial_candidates()
            known = core.settings.initial_fitness
            if core.settings.initial_population is not None and known is not None:
                core.finish_initialization(C, known)  # nothing to evaluate
            else:
                k = core.evaluable(len(C))
                self._pending = ("init", C[:k], C)
                return C[:k].copy()
        if core.break_conditions():
            raise StopIteration("termination condition reached")
        core.sample_parameters()
        core.mutate()
        core.crossover()
        core.correct_bounds()
        k = core.evaluable(core.lambda_)
        T = core.trials_corrected[:k]
        self._pending = ("trials", T, None)
        return T.copy()

    def tell(self, f: Any) -> None:
        """Objective values of the last asked points, in the same order."""
        if self._pending is None:
            raise RuntimeError("tell() without a preceding ask()")
        kind, points, extra = self._pending
        f = np.asarray(f, dtype=float).ravel()
        if len(f) != len(points):
            raise ValueError(f"expected {len(points)} values, got {len(f)}")
        core = self.core
        self._pending = None
        self._single = None  # a batch told as a whole ends any one-at-a-time state

        if kind == "init":
            core.record_evaluations(points, f)
            full = np.full(len(extra), np.inf)
            full[: len(f)] = f
            core.finish_initialization(extra, full)
        elif kind == "trials":
            core.tell_trials(f)
            core.select()
            if not core.break_conditions():
                if core.draw_opposition():
                    O = core.opposites
                    k = core.evaluable(len(O))
                    self._pending = ("opposition", O[:k], None)
                    warn_nan(core, self._state)
                    return  # adapt() follows after the opposite points are told
                core.adapt()
            core.update_stats()
        else:  # opposition
            core.record_evaluations(points, f)
            core.finish_opposition(f)
            if not core.break_conditions():
                core.adapt()
            core.update_stats()
        warn_nan(core, self._state)

    # ------------------------------------------------------------------ one point at a time

    def ask_one(self) -> tuple[int, np.ndarray]:
        """``(index, x)`` of the next point. Points are matched by index, not by value."""
        if self._single is None:
            X = self.ask()
            self._single = {"X": X, "f": np.zeros(len(X)), "told": np.zeros(len(X), dtype=bool), "next": 0}
        s = self._single
        if s["next"] >= len(s["X"]):
            missing = [int(i) for i in np.flatnonzero(~s["told"])]
            raise RuntimeError(f"all points of this batch were asked; tell the values of indices "
                               f"{missing} first")
        i = s["next"]
        s["next"] += 1
        return i, s["X"][i].copy()

    def tell_one(self, index: int, f: float) -> None:
        """Value of the point returned by ``ask_one`` with this ``index``."""
        if self._single is None:
            raise RuntimeError("tell_one() without a preceding ask_one()")
        s = self._single
        if not 0 <= index < s["next"]:
            raise ValueError(f"index {index} has not been asked")
        if s["told"][index]:
            warnings.warn(f"value for index {index} told twice; using the last one", UserWarning)
        s["f"][index] = float(f)
        s["told"][index] = True
        if s["told"].all():
            self.tell(s["f"])

    # ------------------------------------------------------------------ results and state

    @property
    def best_x(self) -> np.ndarray:
        """Best point found."""
        return np.asarray(self.core.stats.best_x)

    @property
    def best_f(self) -> float:
        """Best objective value found."""
        return float(self.core.stats.best_f)

    @property
    def evaluations(self) -> int:
        """Values told so far (= evaluations used)."""
        return int(self.core.stats.evaluations)

    def __getattr__(self, name: str) -> Any:
        # population, fitness, archive, memories, settings, ... from the core (read-only here)
        if name.startswith("_") or name == "core":
            raise AttributeError(name)
        try:
            return getattr(self.core, name)
        except AttributeError:
            raise AttributeError(f"'AskTellDE' object has no attribute {name!r}") from None
