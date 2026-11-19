"""The user-facing optimizer: ``ModularDE`` and ``fmin``."""
from __future__ import annotations

import itertools
import warnings
from typing import Any, Callable, NamedTuple

import numpy as np

from . import _core
from ._settings import settings_from_dict
from .configspace import preset_settings

_wrap_counter = itertools.count()


def is_ioh_problem(obj: Any) -> bool:
    """Duck-typed check for an ioh (real-valued, single-objective) problem."""
    return all(hasattr(obj, a) for a in ("meta_data", "state", "bounds"))


def wrap_problem(func: Callable, dim: int, lb: Any, ub: Any, name: str | None = None):
    """Wrap a plain function ``f(x) -> float`` as an ioh problem with box bounds.

    Wrapping lets the problem count the evaluation budget (D-B3) and makes ioh's
    loggers (IOHanalyzer output, tracked attributes) available at no extra effort.
    """
    import ioh

    lb = np.broadcast_to(np.asarray(lb, dtype=float), (dim,)).copy()
    ub = np.broadcast_to(np.asarray(ub, dtype=float), (dim,)).copy()
    base = name or getattr(func, "__name__", "objective")
    unique = f"modde_{base}_{next(_wrap_counter)}"  # ioh keeps a registry of wrapped problems
    problem = ioh.wrap_problem(func, unique, ioh.ProblemClass.REAL, dimension=int(dim),
                               lb=float(lb.min()), ub=float(ub.max()))
    problem.bounds.lb = lb
    problem.bounds.ub = ub
    return problem


class FminResult(NamedTuple):
    """Result of :func:`fmin` (unpacks as ``x, f, evaluations, optimizer``)."""

    x: np.ndarray
    f: float
    evaluations: int
    optimizer: "ModularDE"


class ModularDE:
    """Modular Differential Evolution bound to a problem.

    Parameters
    ----------
    problem:
        An ioh problem, or a plain callable ``f(x) -> float``. A plain callable
        needs ``dim``, ``lb`` and ``ub`` (D-B2) and is wrapped with ioh by
        default (``wrap=True``), so that the problem counts the budget (D-B3).
    budget:
        Maximum number of evaluations, counted by the problem. If the problem
        already has evaluations, a warning is issued and they count towards it.
    preset, config, settings:
        Start from a named preset (``modde.preset_names()``), a ConfigSpace
        configuration, or complete core ``Settings``. Keyword arguments
        (registry names, e.g. ``mutation_base="TARGET"``) are applied on top.
    vectorized:
        Whether the objective accepts an ``(n, d)`` array and returns ``n``
        values. Default: True for ioh problems, False for plain callables.

    Notes
    -----
    Attributes such as ``curr_F``, ``curr_CR``, ``corrected`` and ``popmean`` can
    be watched by an ioh logger; per-evaluation values need
    ``track_evaluations=True``. All state of the core optimizer (``population``,
    ``fitness``, ``archive``, ``memory_F``, ...) is available as attributes too.
    """

    def __init__(self, problem: Any, *, budget: int | None = None, dim: int | None = None,
                 lb: Any = None, ub: Any = None, preset: str | None = None, config: Any = None,
                 settings: _core.Settings | None = None, wrap: bool = True,
                 vectorized: bool | None = None, **kwargs: Any) -> None:
        user_lb, user_ub = lb, ub
        if is_ioh_problem(problem):
            n = int(problem.meta_data.n_variables)
            if dim is not None and int(dim) != n:
                raise ValueError(f"dim = {dim} does not match the problem dimension {n}")
            dim = n
            self.problem = problem
        elif callable(problem):
            if dim is None or lb is None or ub is None:
                raise ValueError("dim, lb and ub are required for a plain callable (they are read "
                                 "from the problem for ioh problems)")
            self.problem = wrap_problem(problem, int(dim), lb, ub) if wrap else problem
        else:
            raise TypeError("problem must be an ioh problem or a callable")

        self._ioh = is_ioh_problem(self.problem)
        self._sign = 1.0
        if self._ioh:
            import ioh

            if self.problem.meta_data.optimization_type == ioh.OptimizationType.MAX:
                self._sign = -1.0  # minimise -f (D-B10)

        if settings is None:
            values: dict[str, Any] = {}
            if preset is not None:
                values.update(preset_settings(preset, dim))
            if config is not None:
                values.update(dict(config))
            values.update(kwargs)
            problem_fields = set()
            if self._ioh and user_lb is None:
                values["lb"] = np.asarray(self.problem.bounds.lb, dtype=float)
                problem_fields.add("lb")
            elif user_lb is not None:
                values["lb"] = user_lb
            if self._ioh and user_ub is None:
                values["ub"] = np.asarray(self.problem.bounds.ub, dtype=float)
                problem_fields.add("ub")
            elif user_ub is not None:
                values["ub"] = user_ub
            if budget is not None:
                values["budget"] = int(budget)
            settings = settings_from_dict(dim, **values)
            settings.problem_fields = problem_fields
        elif kwargs or preset is not None or config is not None:
            raise ValueError("pass either `settings` or preset/config/keyword settings, not both")

        self.core = _core.ModularDE(settings)
        for message in self.core.settings.warnings:
            warnings.warn(message, UserWarning, stacklevel=2)

        if self._ioh:
            problem_ = self.problem
            sign = self._sign
            self._f = (lambda X: problem_(X)) if sign > 0 else \
                (lambda X: -np.asarray(problem_(X), dtype=float))
            self._vectorized = True if vectorized is None else bool(vectorized)
        else:
            self._f = self.problem
            self._vectorized = bool(vectorized)
        self._started = False
        self._target_hit = False

    # ------------------------------------------------------------------ running

    def _start(self) -> None:
        if self._started:
            return
        self._started = True
        if self._ioh:
            used = int(self.problem.state.evaluations)
            if used > 0:
                warnings.warn(f"the problem already has {used} evaluations; they count towards the "
                              f"budget of {self.core.settings.budget}", UserWarning, stacklevel=3)
                self.core.stats.evaluations = used

    def _check_target(self) -> bool:
        if self._ioh:
            state = self.problem.state
            self._target_hit = bool(getattr(state, "final_target_found", False))
        return self._target_hit

    def step(self) -> bool:
        """One generation (the first call also initialises). False once finished."""
        self._start()
        if self.break_conditions():
            return False
        more = self.core.step(self._f, self._vectorized)
        return more and not self._check_target()

    def run(self) -> "ModularDE":
        """Step until a termination condition holds."""
        while self.step():
            pass
        return self

    def break_conditions(self) -> bool:
        return self._target_hit or (self.core.initialized and self.core.break_conditions())

    # split step API with the bound objective (others are forwarded to the core)
    def initialize(self) -> None:
        self._start()
        self.core.initialize(self._f, self._vectorized)
        self._check_target()

    def evaluate(self) -> None:
        self.core.evaluate(self._f, self._vectorized)
        self._check_target()

    def oppositional_generation(self) -> bool:
        jumped = self.core.oppositional_generation(self._f, self._vectorized)
        self._check_target()
        return jumped

    def __getattr__(self, name: str) -> Any:
        # population, fitness, donors, trials, archive, memories, settings, ...
        if name in ("core", "problem"):
            raise AttributeError(name)
        return getattr(self.core, name)

    # ------------------------------------------------------------------ results

    @property
    def best_x(self) -> np.ndarray:
        return np.asarray(self.core.stats.best_x)

    @property
    def best_f(self) -> float:
        return self._sign * float(self.core.stats.best_f)

    @property
    def evaluations(self) -> int:
        return int(self.problem.state.evaluations) if self._ioh else int(self.core.stats.evaluations)

    # ------------------------------------------------------------------ tracked attributes (D-B8)

    @property
    def curr_F(self) -> float:
        return self.core.stats.curr_F

    @property
    def curr_CR(self) -> float:
        return self.core.stats.curr_CR

    @property
    def corrected(self) -> float:
        return float(self.core.stats.corrected)

    @property
    def corr_so_far(self) -> int:
        return self.core.stats.corr_so_far

    @property
    def CS(self) -> float:
        return self.core.stats.CS

    @property
    def ED(self) -> float:
        return self.core.stats.ED

    @property
    def popmean(self) -> float:
        return self.core.stats.popmean

    @property
    def popstd(self) -> float:
        return self.core.stats.popstd

    @property
    def F_memory_mean(self) -> float:
        return self.core.stats.F_memory_mean

    @property
    def generation(self) -> int:
        return self.core.stats.generation

    def __repr__(self) -> str:
        return (f"<modde.ModularDE dim={self.core.settings.dim} lambda={self.core.lambda_} "
                f"evaluations={self.evaluations} best_f={self.best_f:.6g}>")


def fmin(func: Any, budget: int, *, dim: int | None = None, lb: Any = None, ub: Any = None,
         preset: str | None = None, config: Any = None, seed: int | None = None,
         vectorized: bool | None = None, wrap: bool = True, **kwargs: Any) -> FminResult:
    """Minimise ``func`` with modular DE (classic DE/rand/1/bin unless configured).

    ``func`` is an ioh problem or a callable ``f(x) -> float`` (then ``dim``, ``lb``
    and ``ub`` are required). ``budget`` is required. Returns
    ``FminResult(x, f, evaluations, optimizer)``, which also unpacks as a tuple.

    Example
    -------
    >>> import numpy as np, modde
    >>> x, f, evals, opt = modde.fmin(lambda x: float(np.sum(x**2)), budget=2000, dim=5, lb=-5, ub=5)
    """
    if seed is not None:
        kwargs["seed"] = seed
    opt = ModularDE(func, budget=budget, dim=dim, lb=lb, ub=ub, preset=preset, config=config,
                    vectorized=vectorized, wrap=wrap, **kwargs)
    opt.run()
    return FminResult(opt.best_x, opt.best_f, opt.evaluations, opt)
