"""The user-facing optimizer: ``ModularDE`` and ``fmin``."""
from __future__ import annotations

import itertools
import warnings
from typing import Any, Callable, NamedTuple

import numpy as np

from . import _core
from ._common import (TrackedMixin, as_int, check_settings_alone, collect_values, emit_warnings,
                      warn_nan, writable_core_properties)
from ._settings import _MODULE_FIELDS, settings_from_dict, settings_to_dict

_wrap_counter = itertools.count()
_CORE_WRITABLE = writable_core_properties()
_SIGNED = frozenset({"fitness", "trial_fitness"})  # exposed in the problem's sign


def is_ioh_problem(obj: Any) -> bool:
    """Duck-typed check for an ioh (real-valued, single-objective) problem."""
    return all(hasattr(obj, a) for a in ("meta_data", "state", "bounds"))


def wrap_problem(func: Callable, dim: int, lb: Any, ub: Any, name: str | None = None):
    """Wrap a plain function ``f(x) -> float`` as an ioh problem with box bounds.

    Wrapping lets the problem count the evaluation budget and makes ioh's
    loggers (IOHanalyzer output, tracked attributes) available. ioh calls the
    function with one point (a 1-D array of length ``dim``) at a time.
    """
    import ioh

    dim = as_int("dim", dim, minimum=1)
    lb = np.broadcast_to(np.asarray(lb, dtype=float), (dim,)).copy()
    ub = np.broadcast_to(np.asarray(ub, dtype=float), (dim,)).copy()
    base = name or getattr(func, "__name__", "objective")
    unique = f"modde_{base}_{next(_wrap_counter)}"  # ioh keeps a registry of wrapped problems
    problem = ioh.wrap_problem(func, unique, ioh.ProblemClass.REAL, dimension=dim,
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


class ModularDE(TrackedMixin):
    """Modular Differential Evolution bound to a problem.

    Parameters
    ----------
    problem:
        An ioh problem, or a plain callable ``f(x) -> float``. A plain callable
        needs ``dim``, ``lb`` and ``ub``. By default it is wrapped as an ioh problem
        (``wrap=True``), which counts the budget and supports ioh logging; ioh then
        calls it with one point (a 1-D array) at a time.
    budget:
        Maximum number of evaluations (default ``10000 * dim``), counted by the
        problem. If the problem already has evaluations, a warning is issued and they
        count towards the budget.
    dim, lb, ub:
        Dimension and bounds; required for plain callables. For ioh problems they are
        taken from the problem (``lb``/``ub`` given here replace the problem's bounds,
        with a warning if they differ).
    preset, config:
        Start from a named preset (``modde.preset_names()``) **or** a ConfigSpace
        configuration (not both). Keyword arguments (registry names, e.g.
        ``mutation_base="TARGET"``, ``seed=1``) are applied on top.
    settings:
        Complete core ``Settings`` (e.g. from ``settings_from_dict``), used as they are.
        Cannot be combined with ``budget``, ``lb``, ``ub``, ``preset``, ``config`` or
        keyword settings. Its values are in the optimizer's minimisation sign.
    wrap:
        Wrap a plain callable as an ioh problem (default True).
    vectorized:
        Evaluate a whole generation per call. For ioh problems (default True) the
        problem receives an ``(n, d)`` array. For a plain callable this needs
        ``wrap=False``: the callable then receives an ``(n, d)`` array and returns
        ``n`` values, and the optimizer counts the budget itself.

    Notes
    -----
    - **Stopping:** a run ends when the budget is used, when ``target`` is reached
      (``f <= target``; ``f >= target`` for maximisation problems), when an ioh
      problem's final target is found, or after ``max_generations``.
    - **Maximisation:** for ioh maximisation problems the optimizer minimises
      ``-f`` internally. ``best_f``, ``fitness``, ``trial_fitness``, ``target`` and
      ``initial_fitness`` are all in the problem's own sign.
    - **NaN** objective values are treated as +inf (with a warning).
    - **State:** the core state (``population``, ``fitness``, ``F``, ``CR``,
      ``F_used``, ``CR_used``, ``donors``, ``trials``, ``archive``, ``memory_F``,
      ...) is available as attributes; writable ones can be assigned between steps
      (shapes are checked). ``opt.settings`` returns a copy of the resolved settings.
    - **Logging:** ``curr_F``, ``curr_CR``, ``corrected``, ``popmean``, ... can be
      watched by an ioh logger; per-evaluation values need ``track_evaluations=True``.
    """

    def __init__(self, problem: Any, *, budget: int | None = None, dim: int | None = None,
                 lb: Any = None, ub: Any = None, preset: str | None = None, config: Any = None,
                 settings: _core.Settings | None = None, wrap: bool = True,
                 vectorized: bool | None = None, **kwargs: Any) -> None:
        state: dict[str, Any] = {}
        object.__setattr__(self, "_state", state)
        if settings is not None:
            check_settings_alone(budget=budget, lb=lb, ub=ub, preset=preset, config=config,
                                 **{"keyword settings": kwargs})
            if dim is not None and as_int("dim", dim) != settings.dim:
                raise ValueError(f"dim = {dim} does not match settings.dim = {settings.dim}")
            dim = settings.dim
        if budget is not None:
            budget = as_int("budget", budget, minimum=1)

        if is_ioh_problem(problem):
            n = int(problem.meta_data.n_variables)
            if dim is not None and as_int("dim", dim) != n:
                raise ValueError(f"dim = {dim} does not match the problem dimension {n}")
            dim = n
            self.problem = problem
        elif callable(problem):
            if settings is not None:
                lb, ub = settings.lb, settings.ub
            if dim is None or lb is None or ub is None:
                raise ValueError("dim, lb and ub are required for a plain callable (they are read "
                                 "from the problem for ioh problems)")
            dim = as_int("dim", dim, minimum=1)
            if wrap and vectorized:
                raise ValueError("vectorized=True needs wrap=False for a plain callable: an ioh-wrapped "
                                 "function is called with one point at a time")
            self.problem = wrap_problem(problem, dim, lb, ub) if wrap else problem
            if settings is not None:
                lb = ub = None
        else:
            raise TypeError("problem must be an ioh problem or a callable")

        self._ioh = is_ioh_problem(self.problem)
        self._sign = 1.0
        if self._ioh:
            import ioh

            if self.problem.meta_data.optimization_type == ioh.OptimizationType.MAX:
                self._sign = -1.0  # minimise -f

        preset_only: set[str] = set()
        user_modules: set[str] = set()
        if settings is None:
            values, preset_only = collect_values(dim, preset, config, kwargs)
            user_modules = {k for k in (set(kwargs) | set(dict(config or {}))) if k in _MODULE_FIELDS}
            if self._sign < 0:  # user values in the problem's sign -> minimisation sign
                if values.get("target") is not None:
                    values["target"] = -float(values["target"])
                if values.get("initial_fitness") is not None:
                    values["initial_fitness"] = -np.asarray(values["initial_fitness"], dtype=float)
            problem_fields = set()
            for name, given in (("lb", lb), ("ub", ub)):
                if given is not None:
                    values[name] = given
                    if self._ioh:
                        own = np.asarray(getattr(self.problem.bounds, name), dtype=float)
                        if not np.allclose(np.broadcast_to(np.asarray(given, dtype=float), own.shape), own):
                            warnings.warn(f"{name} given here differs from the problem's bounds; the "
                                          f"search uses {name} = {given}", UserWarning, stacklevel=2)
                elif self._ioh:
                    values[name] = np.asarray(getattr(self.problem.bounds, name), dtype=float)
                    problem_fields.add(name)
            if budget is not None:
                values["budget"] = budget
            settings = settings_from_dict(dim, **values)
            settings.problem_fields = problem_fields
        elif self._ioh and (settings.lb is None or settings.ub is None):
            raise ValueError("these settings have no bounds: set lb and ub with settings_from_dict, "
                             "or use the keyword interface, which takes them from the problem")

        self.core = _core.ModularDE(settings)
        emit_warnings(self.core, preset_only, user_modules)

        if self._ioh:
            problem_, core, sign = self.problem, self.core, self._sign

            def f(X):
                values = problem_(X)
                if problem_.state.final_target_found:
                    core.stop_requested = True  # ends the generation at the next check
                return values if sign > 0 else -np.asarray(values, dtype=float)

            self._f = f
            self._vectorized = True if vectorized is None else bool(vectorized)
        else:
            self._f = self.problem
            self._vectorized = bool(vectorized)
        self._started = False

    # ------------------------------------------------------------------ running

    def _start(self) -> None:
        if self._started:
            return
        self._started = True
        if self._ioh:
            used = int(self.problem.state.evaluations)
            if used > 0:
                warnings.warn(f"the problem already has {used} evaluations; they count towards the "
                              f"budget of {self.core.settings.budget}. Call problem.reset() to start "
                              "from zero", UserWarning, stacklevel=3)
                self.core.stats.evaluations = used
            if self.problem.state.final_target_found:
                warnings.warn("the problem has already reached its final target, so the run stops "
                              "immediately; call problem.reset() first", UserWarning, stacklevel=3)
                self.core.stop_requested = True

    def step(self) -> bool:
        """One generation (the first call also initialises). Returns False once finished."""
        self._start()
        if self.break_conditions():
            return False
        more = self.core.step(self._f, self._vectorized)
        warn_nan(self.core, self._state)
        return more and not self.break_conditions()

    def run(self) -> "ModularDE":
        """Step until a termination condition holds. Returns the optimizer itself."""
        while self.step():
            pass
        return self

    def break_conditions(self) -> bool:
        """True once the run is finished (budget, target, max_generations or the problem's target)."""
        return bool(self.core.stop_requested) or (self.core.initialized and self.core.break_conditions())

    # split step API with the bound objective (the other steps are forwarded to the core)
    def initialize(self) -> None:
        """Sample, evaluate and select the initial population."""
        self._start()
        self.core.initialize(self._f, self._vectorized)
        warn_nan(self.core, self._state)

    def evaluate(self) -> None:
        """Evaluate the corrected trials (at most the remaining budget)."""
        self.core.evaluate(self._f, self._vectorized)
        warn_nan(self.core, self._state)

    def oppositional_generation(self) -> bool:
        """Draw and, if it happens, perform an oppositional jump. True if it jumped."""
        jumped = self.core.oppositional_generation(self._f, self._vectorized)
        warn_nan(self.core, self._state)
        return jumped

    # ------------------------------------------------------------------ state access

    def __getattr__(self, name: str) -> Any:
        # population, F, archive, memories, settings, ... from the core
        if name.startswith("_") or name in ("core", "problem"):
            raise AttributeError(name)
        try:
            return getattr(self.core, name)
        except AttributeError:
            raise AttributeError(f"'ModularDE' object has no attribute {name!r}") from None

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _CORE_WRITABLE:
            if name in _SIGNED:
                value = self._sign * np.asarray(value, dtype=float)
            setattr(self.core, name, value)
        elif isinstance(getattr(type(self), name, None), property) or (
                name not in ("core", "problem") and not name.startswith("_")
                and hasattr(_core.ModularDE, name)):
            raise AttributeError(f"{name} is read-only")
        else:
            object.__setattr__(self, name, value)

    @property
    def fitness(self) -> np.ndarray:
        """Objective values of the population (in the problem's sign)."""
        return self._sign * np.asarray(self.core.fitness)

    @property
    def trial_fitness(self) -> np.ndarray:
        """Objective values of the current trials (in the problem's sign; ±inf if not evaluated)."""
        return self._sign * np.asarray(self.core.trial_fitness)

    def settings_dict(self, user_only: bool = False) -> dict[str, Any]:
        """The settings of this run as keyword arguments, in the problem's sign.

        ``user_only=True`` gives what was set explicitly plus the seed;
        ``ModularDE(problem, **opt.settings_dict(user_only=True))`` repeats the run.
        Unlike ``settings_to_dict(opt.settings)``, ``target`` and ``initial_fitness``
        are converted back to the problem's sign for maximisation problems.
        """
        values = settings_to_dict(self.core.settings, user_only=user_only)
        if self._sign < 0:
            for name in ("target", "initial_fitness"):
                if values.get(name) is not None:
                    values[name] = -np.asarray(values[name], dtype=float) if name == "initial_fitness"                         else -float(values[name])
        return values

    # ------------------------------------------------------------------ results

    @property
    def best_x(self) -> np.ndarray:
        """Best point found."""
        return np.asarray(self.core.stats.best_x)

    @property
    def best_f(self) -> float:
        """Best objective value found (in the problem's sign)."""
        return self._sign * float(self.core.stats.best_f)

    @property
    def evaluations(self) -> int:
        """Evaluations used (counted by the problem for ioh problems)."""
        return int(self.problem.state.evaluations) if self._ioh else int(self.core.stats.evaluations)

    def __repr__(self) -> str:
        return (f"<modde.ModularDE dim={self.core.settings.dim} lambda={self.core.lambda_} "
                f"evaluations={self.evaluations} best_f={self.best_f:.6g}>")


def fmin(func: Any, budget: int, *, dim: int | None = None, lb: Any = None, ub: Any = None,
         preset: str | None = None, config: Any = None, seed: int | None = None,
         vectorized: bool | None = None, wrap: bool = True, **kwargs: Any) -> FminResult:
    """Minimise ``func`` with modular DE (classic DE/rand/1/bin unless configured).

    ``func`` is an ioh problem or a callable ``f(x) -> float`` (then ``dim``, ``lb``
    and ``ub`` are required). ``budget`` (number of evaluations) is required. Other
    arguments as for :class:`ModularDE`. Returns ``FminResult(x, f, evaluations,
    optimizer)``, which also unpacks as a tuple.

    Example
    -------
    >>> import numpy as np, modde
    >>> x, f, evals, opt = modde.fmin(lambda x: float(np.sum(x**2)), budget=2000, dim=5, lb=-5, ub=5)
    """
    if budget is None:
        raise ValueError("fmin needs a budget (number of evaluations)")
    if seed is not None:
        kwargs["seed"] = seed
    opt = ModularDE(func, budget=budget, dim=dim, lb=lb, ub=ub, preset=preset, config=config,
                    vectorized=vectorized, wrap=wrap, **kwargs)
    opt.run()
    return FminResult(opt.best_x, opt.best_f, opt.evaluations, opt)
