"""Ask-tell interface (D-A12): generation-level ``ask``/``tell`` plus a
per-individual convenience layer. Uses the same core building blocks as
``ModularDE.run``, so the same seed gives the same trajectory (D-D12)."""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from . import _core
from ._settings import settings_from_dict
from .configspace import preset_settings


class AskTellDE:
    """Modular DE with external evaluation.

    >>> at = AskTellDE(dim=5, lb=-5, ub=5, budget=1000)
    >>> while not at.stop:
    ...     X = at.ask()                       # (n, d) points
    ...     at.tell([f(x) for x in X])          # their values, same order

    Or one point at a time: ``i, x = at.ask_one(); at.tell_one(i, f(x))``.
    The budget is counted by the optimizer (there is no problem object).
    """

    def __init__(self, dim: int, lb: Any, ub: Any, *, budget: int | None = None,
                 preset: str | None = None, config: Any = None,
                 settings: _core.Settings | None = None, **kwargs: Any) -> None:
        if settings is None:
            values: dict[str, Any] = {}
            if preset is not None:
                values.update(preset_settings(preset, dim))
            if config is not None:
                values.update(dict(config))
            values.update(kwargs)
            if budget is not None:
                values["budget"] = int(budget)
            settings = settings_from_dict(dim, lb=lb, ub=ub, **values)
        self.core = _core.ModularDE(settings)
        for message in self.core.settings.warnings:
            warnings.warn(message, UserWarning, stacklevel=2)
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

        if kind == "init":
            core.record_evaluations(points, f)
            full = np.full(len(extra), np.inf)
            full[: len(f)] = f
            core.finish_initialization(extra, full)
            return

        if kind == "trials":
            core.tell_trials(f)
            core.select()
            if not core.break_conditions():
                if core.draw_opposition():
                    O = core.opposites
                    k = core.evaluable(len(O))
                    self._pending = ("opposition", O[:k], None)
                    return  # adapt() follows after the opposite points are told
                core.adapt()
            core.update_stats()
            return

        # kind == "opposition"
        core.record_evaluations(points, f)
        core.finish_opposition(f)
        if not core.break_conditions():
            core.adapt()
        core.update_stats()

    # ------------------------------------------------------------------ one point at a time

    def ask_one(self) -> tuple[int, np.ndarray]:
        """``(index, x)`` of the next point. Points are matched by index, not by value."""
        if self._single is None:
            X = self.ask()
            self._single = {"X": X, "f": np.full(len(X), np.nan), "next": 0}
        s = self._single
        if s["next"] >= len(s["X"]):
            raise RuntimeError("all points of this batch were asked; tell their values first")
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
        if not np.isnan(s["f"][index]):
            warnings.warn(f"value for index {index} told twice; using the last one", UserWarning)
        s["f"][index] = float(f)
        if not np.isnan(s["f"]).any():
            self._single = None
            self.tell(s["f"])

    # ------------------------------------------------------------------ results

    @property
    def best_x(self) -> np.ndarray:
        return np.asarray(self.core.stats.best_x)

    @property
    def best_f(self) -> float:
        return float(self.core.stats.best_f)

    @property
    def evaluations(self) -> int:
        return int(self.core.stats.evaluations)

    def __getattr__(self, name: str) -> Any:
        if name == "core":
            raise AttributeError(name)
        return getattr(self.core, name)
