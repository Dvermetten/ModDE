"""Parameter registry: the single source of truth for modde settings (D-C3).

Every setting the optimizer accepts is described exactly once here: its type,
valid values, default rule, when it is active, whether and how it is tuned,
and its documentation. From this module we derive (or test against):

* the C++ ``Settings::resolve()`` defaults (hand-written; a test checks agreement),
* the ConfigSpace (``_build_configspace``; public wrapper comes in Phase 3),
* the named presets (``PRESETS``, D-D2),
* the module reference tables in the docs.

Semantics of each module are specified in ``dev/spec/`` in the repository.
This module has no dependency on the compiled extension.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

# --------------------------------------------------------------------------- types


@dataclass(frozen=True)
class Condition:
    """The parameter is active when ``parent`` takes one of ``values``."""

    parent: str
    values: tuple


@dataclass(frozen=True)
class Param:
    """Description of one setting.

    Attributes
    ----------
    name: name of the setting (Python keyword argument / ConfigSpace name)
    category: documentation section, matches a file in ``dev/spec/``
    kind: ``"categorical" | "bool" | "int" | "float" | "vector" | "matrix"``
    doc: one-line description
    default: value, or ``callable(dim) -> value``; ``None`` = no static default
    default_doc: human-readable default rule (for docs)
    options: allowed values of a categorical (first = default)
    valid: ``(lo, hi)`` valid range for numeric values (``None`` = unbounded side),
        or ``callable(dim) -> (lo, hi)``
    tune: ConfigSpace range ``(lo, hi)`` or ``callable(dim) -> (lo, hi)``;
        ``None`` = not tunable / not in the ConfigSpace
    log: sample on log scale in the ConfigSpace
    tune_doc: human-readable tuning range when ``tune`` depends on ``dim``
    group: ConfigSpace group (see ``GROUPS``); ignored if ``tune`` is None and
        the parameter is not categorical/bool
    active_when: parameter is active if ANY condition holds (empty = always)
    derived: default is DERIVED from other settings (see ``derive_defaults``)
    replaces: the 0.0.4 name(s), for the migration guide
    references: literature
    """

    name: str
    category: str
    kind: str
    doc: str
    default: Any = None
    default_doc: str = ""
    options: tuple = ()
    valid: Any = None
    tune: Any = None
    log: bool = False
    tune_doc: str = ""
    group: str | None = None
    active_when: tuple[Condition, ...] = ()
    derived: bool = False
    replaces: str = ""
    references: tuple[str, ...] = ()

    # helpers -----------------------------------------------------------------
    def default_for(self, dim: int) -> Any:
        return self.default(dim) if callable(self.default) else self.default

    def valid_for(self, dim: int):
        return self.valid(dim) if callable(self.valid) else self.valid

    def tune_for(self, dim: int):
        return self.tune(dim) if callable(self.tune) else self.tune

    @property
    def is_module(self) -> bool:
        return self.kind in ("categorical", "bool")


#: ConfigSpace groups. ``modules`` holds every categorical/bool switch plus
#: ``mutation_n_comps``; the numeric groups can be switched on/off (D-C2).
GROUPS = ("modules", "population", "initial_FCR", "module_params")

# --------------------------------------------------------------------------- references

REF = {
    "storn1997": "Storn & Price, Differential evolution: a simple and efficient heuristic for global optimization over continuous spaces, J. Global Optimization 11, 1997.",
    "zhang2009": "Zhang & Sanderson, JADE: adaptive differential evolution with optional external archive, IEEE TEC 13(5), 2009.",
    "tanabe2013": "Tanabe & Fukunaga, Success-history based parameter adaptation for differential evolution (SHADE), IEEE CEC 2013.",
    "tanabe2014": "Tanabe & Fukunaga, Improving the search performance of SHADE using linear population size reduction (L-SHADE), IEEE CEC 2014.",
    "brest2006": "Brest et al., Self-adapting control parameters in differential evolution: a comparative study on numerical benchmark problems (jDE), IEEE TEC 10(6), 2006.",
    "brest2017": "Brest, Maucec & Boskovic, Single objective real-parameter optimization: algorithm jSO, IEEE CEC 2017.",
    "rahnamayan2008": "Rahnamayan, Tizhoosh & Salama, Opposition-based differential evolution (ODE), IEEE TEC 12(1), 2008.",
    "rahnamayan2007": "Rahnamayan, Tizhoosh & Salama, Quasi-oppositional differential evolution (QODE), IEEE CEC 2007.",
    "guo2015": "Guo & Yang, Enhancing differential evolution utilizing eigenvector-based crossover operator, IEEE TEC 19(1), 2015.",
    "vermetten2023": "Vermetten, Caraffini, Kononova & Baeck, Modular differential evolution, GECCO 2023.",
    # Bound correction (as cited / attributed in vermetten2023 and kononova2022)
    "biedrzycki2019": "Biedrzycki, Arabas & Jagodzinski, Bound constraints handling in differential evolution: an experimental study, Swarm and Evolutionary Computation 50, 2019.",
    "kononova2022": "Kononova, Vermetten, Caraffini, Mitran & Zaharie, The importance of being constrained: dealing with infeasible solutions in differential evolution and beyond, arXiv:2203.03512, 2022.",
    "caraffini2019": "Caraffini, Kononova & Corne, Infeasibility and structural bias in differential evolution, Information Sciences 496, 2019.",
    "kononova2020a": "Kononova, Caraffini, Wang & Baeck, Can compact optimisation algorithms be structurally biased?, PPSN XVI, 2020.",
    "kononova2020b": "Kononova, Caraffini, Wang & Baeck, Can single solution optimisation methods be structurally biased?, IEEE CEC 2020.",
    "vermetten2022c": "Vermetten, van Stein, Kononova & Caraffini, Analysis of structural bias in differential evolution configurations, in: Differential Evolution: From Theory to Practice, Springer, 2022.",
    "padhye2015": "Padhye, Mittal & Deb, Feasibility preserving constraint-handling strategies for real parameter evolutionary optimization, Computational Optimization and Applications 62, 2015.",
    "alvarezbenitez2005": "Alvarez-Benitez, Everson & Fieldsend, A MOPSO algorithm based exclusively on Pareto dominance concepts, EMO 2005.",
}

# --------------------------------------------------------------------------- parameters

_ADAPT_SHADE_F = ("SHADE", "SHADE_MODIFIED")
_ADAPT_SHADE_CR = ("SHADE", "LSHADE")

PARAMS: tuple[Param, ...] = (
    # ---- problem / run ------------------------------------------------------
    Param("dim", "settings", "int", "Search-space dimension. Taken from the problem when it is an ioh problem.",
          valid=(1, None)),
    Param("budget", "settings", "int",
          "Maximum number of evaluations, counted by the problem (D-B3).",
          default=lambda d: 10_000 * d, default_doc="10 000·d (fmin requires a budget)", valid=(1, None)),
    Param("lb", "settings", "vector", "Lower bounds. From the problem; required for plain callables (D-B2).",
          default_doc="from the problem"),
    Param("ub", "settings", "vector", "Upper bounds. From the problem; required for plain callables (D-B2).",
          default_doc="from the problem"),
    Param("target", "settings", "float",
          "Stop when this objective value is reached. ioh problems use their final target.",
          default_doc="ioh final target, else none"),
    Param("max_generations", "settings", "int", "Optional cap on the number of generations.", valid=(1, None)),
    Param("seed", "settings", "int", "Seed of the optimizer's random number generator (D-A13).",
          default_doc="random"),
    Param("track_evaluations", "settings", "bool",
          "Per-evaluation statistics (forces one objective call per point, D-B4).",
          default=False, replaces="init_stats"),
    Param("initial_population", "initialization", "matrix",
          "Optional user-provided initial population (n × d). Replaces sampling (D-A14).",
          replaces="inialize_custom_pop / initialize_custom_population()"),
    Param("initial_fitness", "initialization", "vector",
          "Optional known fitness of initial_population (skips their evaluation)."),

    # ---- population size ----------------------------------------------------
    Param("lambda0", "population_size", "int",
          "Initial population size. A true parameter; no module changes its default (D-D6).",
          default=lambda d: 10 * d, default_doc="10·d",
          valid=(1, None), tune=lambda d: (7, max(200, 50 * d)), log=True, group="population",
          tune_doc="[7, max(200, 50·d)], log scale",
          replaces="lambda_", references=("storn1997",)),
    Param("population_reduction", "population_size", "categorical",
          "Population size schedule during the run.",
          options=("NONE", "LINEAR"), group="modules", replaces="lpsr (bool)",
          references=("tanabe2014",)),
    Param("lambda_min", "population_size", "int",
          "Final population size of the reduction schedule.",
          default_doc="min(lambda0, max(4, parents needed + 1))", derived=True, valid=(1, None),
          active_when=(Condition("population_reduction", ("LINEAR",)),),
          references=("tanabe2014",)),

    # ---- initialization -----------------------------------------------------
    Param("base_sampler", "initialization", "categorical", "Sampler for the initial population.",
          options=("UNIFORM", "GAUSSIAN", "SOBOL", "HALTON", "LHS"), group="modules",
          replaces="base_sampler ('LHC' renamed to LHS)"),
    Param("oppositional_initialization", "initialization", "categorical",
          "Add (quasi-)opposite points of the sampled points to the initial candidates.",
          options=("NONE", "OPPOSITE", "QUASI_OPPOSITE"), group="modules",
          replaces="oppositional_initialization (bool)",
          references=("rahnamayan2008", "rahnamayan2007")),
    Param("oppositional_initialization_factor", "initialization", "float",
          "Scales the opposite point towards the centre: 1 = full opposition / QODE, 0 = centre (KI-37).",
          default=1.0, valid=(0.0, 1.0), tune=(0.0, 1.0), group="module_params",
          active_when=(Condition("oppositional_initialization", ("OPPOSITE", "QUASI_OPPOSITE")),)),
    Param("oversampling_factor", "initialization", "float",
          "Evaluate ceil(λ0·(1 + factor)) initial candidates and keep the best λ0 (KI-4 fixed).",
          default=0.0, valid=(0.0, None), tune=(0.0, 2.0), group="module_params"),

    # ---- mutation -----------------------------------------------------------
    Param("mutation_base", "mutation", "categorical", "Base vector of the mutation.",
          options=("RAND", "BEST", "TARGET"), group="modules", references=("storn1997",)),
    Param("mutation_reference", "mutation", "categorical",
          "Optional '-to-X' term F·(x_X − x_i) added to the base vector.",
          options=("NONE", "RAND", "BEST", "PBEST"), group="modules", references=("zhang2009",)),
    Param("mutation_n_comps", "mutation", "int",
          "Number of difference vectors F·(x_r − x_s). Integer ≥ 0 in the API (D-A6).",
          default=1, valid=(0, None), tune=(1, 2), group="modules"),
    Param("pbest_sampling", "mutation", "categorical",
          "How p is chosen for p-best: RANDOM per individual p_i ~ U(2/λ, pbest_rate_max) (SHADE), or FIXED p (JADE/L-SHADE).",
          options=("RANDOM", "FIXED"), group="modules",
          active_when=(Condition("mutation_reference", ("PBEST",)),),
          replaces="pbest_value (None → RANDOM, float → FIXED)", references=("tanabe2013", "zhang2009")),
    Param("pbest_rate", "mutation", "float", "Fixed p: p-best is drawn from the top max(1, round(p·λ)) (KI-25).",
          default=0.11, valid=(0.0, 1.0), tune=(0.02, 0.5), group="module_params",
          active_when=(Condition("pbest_sampling", ("FIXED",)),), replaces="pbest_value",
          references=("tanabe2014",)),
    Param("pbest_rate_max", "mutation", "float", "Upper bound of the per-individual random p.",
          default=0.2, valid=(0.0, 1.0), tune=(0.05, 0.5), group="module_params",
          active_when=(Condition("pbest_sampling", ("RANDOM",)),), references=("tanabe2013",)),
    Param("mutation_use_weighted_F", "mutation", "bool",
          "jSO weighted F on the '-to-X' term only: 0.7F / 0.8F / 1.2F by budget progress (KI-24).",
          default=False, group="modules", references=("brest2017",)),
    Param("use_archive", "archive", "bool",
          "External archive (of replaced parents by default, see archive_content); the second vector of the first difference term is drawn from P ∪ A (KI-27).",
          default=False, group="modules", references=("zhang2009",)),
    Param("archive_content", "archive", "categorical",
          "What enters the archive: the replaced PARENTS (JADE/SHADE/L-SHADE papers) or the successful "
          "TRIALS (Tanabe's L-SHADE code, which produced the published CEC2014 results; KI-48, D-D15).",
          options=("PARENTS", "TRIALS"), group="modules",
          active_when=(Condition("use_archive", (True,)),), references=("zhang2009", "tanabe2014")),
    Param("archive_rate", "archive", "float", "Archive capacity = round(rate · current λ) (D-D4).",
          default=1.0, valid=(0.0, None), tune=(0.5, 3.0), group="module_params",
          active_when=(Condition("use_archive", (True,)),), replaces="archive_size (was overwritten, KI-30)",
          references=("zhang2009", "tanabe2014")),

    # ---- crossover ----------------------------------------------------------
    Param("crossover", "crossover", "categorical", "Crossover operator.",
          options=("BIN", "EXP"), group="modules", references=("storn1997",)),
    Param("eigenvalue_crossover", "crossover", "bool",
          "Perform crossover in the eigenbasis of the population covariance.",
          default=False, group="modules", references=("guo2015",)),

    # ---- bound correction ---------------------------------------------------
    Param("bound_correction", "bound_correction", "categorical",
          "Repair of out-of-bounds coordinates of trial vectors.",
          options=("SATURATE", "NONE", "UNIF_RESAMPLE", "COTN", "TOROIDAL", "MIRROR",
                   "HVB", "EXPC_TARGET", "EXPC_CENTER", "EXPS"),
          group="modules", references=("biedrzycki2019", "kononova2022", "vermetten2023")),
    Param("hvb_alpha", "bound_correction", "float",
          "HVB: weight of the violated bound; y' = α·b + (1 − α)·y_target (KI-34).",
          default=0.5, valid=(0.0, 1.0), tune=(0.05, 0.95), group="module_params",
          active_when=(Condition("bound_correction", ("HVB",)),)),

    # ---- F / CR -------------------------------------------------------------
    Param("F0", "adaptation", "float",
          "Initial scale factor; also the initial SHADE F memory value.",
          default=0.5, valid=(0.0, None), tune=(0.1, 1.5), group="initial_FCR", replaces="F (array)",
          references=("storn1997",)),
    Param("CR0", "adaptation", "float",
          "Initial crossover rate; also the initial SHADE CR memory value.",
          default=0.9, valid=(0.0, 1.0), tune=(0.0, 1.0), group="initial_FCR", replaces="CR (array; default was 0.5)",
          references=("storn1997",)),
    Param("adaptation_method_F", "adaptation", "categorical", "Adaptation of F.",
          options=("NONE", "SHADE", "SHADE_MODIFIED", "JDE"), group="modules",
          references=("tanabe2013", "brest2006")),
    Param("adaptation_method_CR", "adaptation", "categorical",
          "Adaptation of CR. LSHADE = Lehmer-mean memory + terminal value (D-D10).",
          options=("NONE", "SHADE", "LSHADE", "JDE"), group="modules",
          references=("tanabe2013", "tanabe2014", "brest2006")),
    Param("memory_size", "adaptation", "int", "Number of SHADE memory slots H (D-D4).",
          default=6, valid=(1, None), tune=(1, 100), log=True, group="module_params",
          active_when=(Condition("adaptation_method_F", _ADAPT_SHADE_F),
                       Condition("adaptation_method_CR", _ADAPT_SHADE_CR)),
          replaces="memory_size (default was 100)", references=("tanabe2013", "tanabe2014")),
    Param("shade_scale_F", "adaptation", "float", "Scale of the Cauchy distribution for F (KI-34).",
          default=0.1, valid=(0.0, None), tune=(0.01, 0.5), log=True, group="module_params",
          active_when=(Condition("adaptation_method_F", _ADAPT_SHADE_F),), references=("tanabe2013",)),
    Param("shade_scale_CR", "adaptation", "float", "Standard deviation of the normal distribution for CR (KI-34).",
          default=0.1, valid=(0.0, None), tune=(0.01, 0.5), log=True, group="module_params",
          active_when=(Condition("adaptation_method_CR", _ADAPT_SHADE_CR),), references=("tanabe2013",)),
    Param("jde_tau_F", "adaptation", "float", "jDE: probability of regenerating F (KI-34).",
          default=0.1, valid=(0.0, 1.0), tune=(0.01, 0.5), group="module_params",
          active_when=(Condition("adaptation_method_F", ("JDE",)),), references=("brest2006",)),
    Param("jde_F_lower", "adaptation", "float", "jDE: new F = jde_F_lower + U(0,1)·jde_F_width (KI-34).",
          default=0.1, valid=(0.0, None), tune=(0.0, 0.5), group="module_params",
          active_when=(Condition("adaptation_method_F", ("JDE",)),), references=("brest2006",)),
    Param("jde_F_width", "adaptation", "float", "jDE: width of the F range (KI-34).",
          default=0.9, valid=(0.0, None), tune=(0.1, 1.5), group="module_params",
          active_when=(Condition("adaptation_method_F", ("JDE",)),), references=("brest2006",)),
    Param("jde_tau_CR", "adaptation", "float", "jDE: probability of regenerating CR (KI-34).",
          default=0.1, valid=(0.0, 1.0), tune=(0.01, 0.5), group="module_params",
          active_when=(Condition("adaptation_method_CR", ("JDE",)),), references=("brest2006",)),
    Param("use_jso_caps", "adaptation", "bool",
          "jSO schedule: CR ≥ 0.7 (t < 0.25), CR ≥ 0.6 (t < 0.5), F ≤ 0.7 (t < 0.6) (KI-21).",
          default=False, group="modules", references=("brest2017",)),

    # ---- oppositional generation -------------------------------------------
    Param("oppositional_generation", "oppositional_generation", "bool",
          "Generation jumping: with probability jump_rate, add opposite points and keep the best λ (D-D11).",
          default=False, group="modules", replaces="oppositional_generation_probability > 0",
          references=("rahnamayan2008",)),
    Param("oppositional_jump_rate", "oppositional_generation", "float", "Probability of a jump per generation.",
          default=0.3, valid=(0.0, 1.0), tune=(0.05, 0.6), group="module_params",
          active_when=(Condition("oppositional_generation", (True,)),),
          replaces="oppositional_generation_probability", references=("rahnamayan2008",)),
)

BY_NAME: Mapping[str, Param] = {p.name: p for p in PARAMS}

#: References for individual options of a categorical setting, in addition to
#: the setting's own ``references``. An empty tuple means "defined by modde".
OPTION_REFERENCES: Mapping[tuple[str, str], tuple[str, ...]] = {
    ("bound_correction", "SATURATE"): ("caraffini2019",),
    ("bound_correction", "UNIF_RESAMPLE"): ("vermetten2022c",),
    ("bound_correction", "COTN"): ("caraffini2019",),
    ("bound_correction", "TOROIDAL"): ("caraffini2019",),
    ("bound_correction", "MIRROR"): ("kononova2020a", "kononova2020b"),
    ("bound_correction", "HVB"): ("kononova2022", "tanabe2013", "biedrzycki2019"),
    ("bound_correction", "EXPC_TARGET"): ("alvarezbenitez2005", "padhye2015"),
    ("bound_correction", "EXPC_CENTER"): (),  # modde variant of EXPC_TARGET with the centre as reference
    ("bound_correction", "EXPS"): ("padhye2015",),
    ("archive_content", "PARENTS"): ("zhang2009", "tanabe2013", "tanabe2014"),
    ("archive_content", "TRIALS"): ("tanabe2014",),  # the code of the L-SHADE paper, not its text (D-D15)
}

# --------------------------------------------------------------------------- derived rules


def n_parents_needed(settings: Mapping[str, Any]) -> int:
    """Distinct population members (besides the target) the mutation needs.

    Worst case: the archive term's second vector may have to come from P (the
    archive starts empty), so it counts like a normal difference vector.
    """
    get = lambda k: settings.get(k, BY_NAME[k].default_for(1) if BY_NAME[k].default is not None
                                 else BY_NAME[k].options[0])
    n = 2 * int(get("mutation_n_comps"))
    n += get("mutation_base") == "RAND"
    n += get("mutation_reference") == "RAND"
    return n


def derive_defaults(settings: Mapping[str, Any]) -> dict[str, Any]:
    """DERIVED defaults (provenance ``DERIVED``) for the given settings."""
    lambda0 = settings.get("lambda0")
    lambda_min = max(4, n_parents_needed(settings) + 1)
    if lambda0 is not None:
        lambda_min = min(int(lambda0), lambda_min)  # a derived value never exceeds lambda0
    return {"lambda_min": lambda_min}


def is_active(param: Param, settings: Mapping[str, Any]) -> bool:
    """Whether ``param`` is active given (complete) module settings."""
    if not param.active_when:
        return True
    for c in param.active_when:
        if c.parent in settings and settings[c.parent] in c.values and is_active(BY_NAME[c.parent], settings):
            return True
    return False


def module_defaults(dim: int) -> dict[str, Any]:
    """Default value of every module and tunable parameter (the classic DE, D-D2)."""
    out = {}
    for p in PARAMS:
        if p.is_module:
            out[p.name] = p.options[0] if p.kind == "categorical" else p.default
        elif p.tune is not None or p.name == "mutation_n_comps":
            out[p.name] = p.default_for(dim)
    return out

# --------------------------------------------------------------------------- presets (D-D2)


@dataclass(frozen=True)
class Preset:
    name: str
    doc: str
    settings: Callable[[int], dict]  # dim -> non-default settings
    exact: bool = True  # False: approximation, some components of the paper are missing
    references: tuple[str, ...] = ()
    notes: str = ""


def _jso_lambda(d: int) -> int:
    return max(7, round(25 * math.log(d) * math.sqrt(d))) if d > 1 else 7


PRESETS: Mapping[str, Preset] = {
    p.name: p
    for p in (
        Preset("DE", "Classic DE/rand/1/bin (the default).", lambda d: {}, references=("storn1997",)),
        Preset("DE/best/1/bin", "Best base vector.", lambda d: {"mutation_base": "BEST"},
               references=("storn1997",)),
        Preset("DE/rand/2/bin", "Two difference vectors.", lambda d: {"mutation_n_comps": 2},
               references=("storn1997",)),
        Preset("DE/current-to-best/1/bin", "Target base, '-to-best' term.",
               lambda d: {"mutation_base": "TARGET", "mutation_reference": "BEST"},
               references=("storn1997",)),
        Preset("jDE", "Self-adaptive F and CR.",
               lambda d: {"lambda0": 100, "adaptation_method_F": "JDE", "adaptation_method_CR": "JDE"},
               references=("brest2006",),
               notes="Bound handling not specified in the paper; uses the default (SATURATE)."),
        Preset("ODE", "Opposition-based DE: opposite initialisation + generation jumping.",
               lambda d: {"lambda0": 100, "oppositional_initialization": "OPPOSITE",
                          "oppositional_generation": True},
               references=("rahnamayan2008",)),
        Preset("SHADE", "Success-history based adaptation, current-to-pbest/1 with archive.",
               lambda d: {"lambda0": 100, "mutation_base": "TARGET", "mutation_reference": "PBEST",
                          "pbest_sampling": "RANDOM", "use_archive": True, "archive_rate": 1.0,
                          "adaptation_method_F": "SHADE", "adaptation_method_CR": "SHADE",
                          "memory_size": 100, "F0": 0.5, "CR0": 0.5, "bound_correction": "HVB"},
               references=("tanabe2013",)),
        Preset("L-SHADE", "SHADE with linear population size reduction (paper-accurate, D-D10).",
               lambda d: {"lambda0": round(18 * d), "population_reduction": "LINEAR", "lambda_min": 4,
                          "mutation_base": "TARGET", "mutation_reference": "PBEST",
                          "pbest_sampling": "FIXED", "pbest_rate": 0.11,
                          "use_archive": True, "archive_rate": 2.6,
                          "adaptation_method_F": "SHADE", "adaptation_method_CR": "LSHADE",
                          "memory_size": 6, "F0": 0.5, "CR0": 0.5, "bound_correction": "HVB"},
               references=("tanabe2014",),
               notes="Archives the replaced parents, as in the paper. The authors' code, which produced the "
                     "published CEC2014 results, archives the successful trials: add "
                     "archive_content=\"TRIALS\" to reproduce it (KI-48)."),
        Preset("JADE-like", "Approximation of JADE with the available modules.",
               lambda d: {"lambda0": 100, "mutation_base": "TARGET", "mutation_reference": "PBEST",
                          "pbest_sampling": "FIXED", "pbest_rate": 0.05, "use_archive": True,
                          "archive_rate": 1.0, "adaptation_method_F": "SHADE",
                          "adaptation_method_CR": "SHADE", "memory_size": 1, "F0": 0.5, "CR0": 0.5,
                          "bound_correction": "HVB"},
               exact=False, references=("zhang2009",),
               notes="JADE updates μ_F/μ_CR by exponential smoothing (c = 0.1); a 1-slot SHADE memory "
                     "replaces instead of smoothing."),
        Preset("jSO-like", "Approximation of jSO with the available modules.",
               lambda d: {"lambda0": _jso_lambda(d), "population_reduction": "LINEAR", "lambda_min": 4,
                          "mutation_base": "TARGET", "mutation_reference": "PBEST",
                          "pbest_sampling": "FIXED", "pbest_rate": 0.17,
                          "mutation_use_weighted_F": True, "use_archive": True, "archive_rate": 1.0,
                          "adaptation_method_F": "SHADE", "adaptation_method_CR": "LSHADE",
                          "memory_size": 5, "F0": 0.3, "CR0": 0.8, "use_jso_caps": True,
                          "bound_correction": "HVB"},
               exact=False, references=("brest2017",),
               notes="Missing: jSO's linearly decreasing p (0.25 → 0.125; approximated by its mean "
                     "0.17), the memory slot fixed at 0.9, averaging new memory values with the old ones, "
                     "and pbest ≠ target in the first half (FR-10)."),
    )
}

# --------------------------------------------------------------------------- ConfigSpace


def _build_configspace(dim: int, groups: tuple[str, ...] = GROUPS):
    """Build the ConfigSpace from the registry (prototype of ``get_configspace``)."""
    from ConfigSpace import (Categorical, ConfigurationSpace, EqualsCondition, Float,
                             InCondition, Integer, OrConjunction)

    cs = ConfigurationSpace()
    included = []
    hp: Any
    for p in PARAMS:
        if p.group is None or p.group not in groups:
            continue
        if p.kind == "categorical":
            hp = Categorical(p.name, p.options, default=p.options[0])
        elif p.kind == "bool":
            hp = Categorical(p.name, (p.default, not p.default), default=p.default)
        elif p.tune is None:
            continue
        elif p.kind == "int":
            hp = Integer(p.name, p.tune_for(dim), default=p.default_for(dim), log=p.log)
        else:
            hp = Float(p.name, p.tune_for(dim), default=float(p.default_for(dim)), log=p.log)
        cs.add(hp)
        included.append(p)

    for p in included:
        conds: list[Any] = []
        for c in p.active_when:
            if c.parent not in cs:
                continue
            parent = cs[c.parent]
            if len(c.values) == 1:
                conds.append(EqualsCondition(cs[p.name], parent, c.values[0]))
            else:
                conds.append(InCondition(cs[p.name], parent, list(c.values)))
        if len(conds) == 1:
            cs.add(conds[0])
        elif len(conds) > 1:
            cs.add(OrConjunction(*conds))
    return cs


def _preset_configuration(name: str, dim: int, cs=None):
    """A preset as a ``Configuration`` of ``_build_configspace(dim)`` (D-D2)."""
    from ConfigSpace.util import deactivate_inactive_hyperparameters

    cs = cs or _build_configspace(dim)
    values = {hp.name: hp.default_value for hp in cs.values()}
    values.update({k: v for k, v in PRESETS[name].settings(dim).items() if k in values})
    return deactivate_inactive_hyperparameters(values, cs)
