# Decision Register

Every open design decision for the next version, grouped by where it comes
from. Each entry lists the options, a recommendation and a status.

- 🔴 **Blocking**: must be decided in Phase 0, before code is written.
- 🟠 **Before the relevant phase**.
- 🟢 **Can be deferred**.

Sources inspected: modcma 1.2.0 (installed wheel + GitHub `setup.py`,
`pyproject.toml`, `CMakeLists.txt`, `include/*.hpp`, publish workflow),
ioh 0.3.22, ConfigSpace 1.2.2, modde 0.0.4.

## Summary of blocking decisions

| ID | Question | Recommendation |
|----|----------|----------------|
| D-A1 | nanobind or pybind11? | ✅ **DECIDED:** nanobind |
| D-A2 | Build backend | ✅ **DECIDED:** scikit-build-core + CMake |
| D-A3 | Eigen? How to include it? | ✅ **DECIDED:** Eigen as git submodule |
| D-A5 | Fate of the pure-Python implementation and the old API | ✅ **DECIDED:** keep 0.0.4 code as deprecated legacy (`modde.legacy`, proposed); remove once stable |
| D-A6 | Adopt modcma's Modules/Settings/Parameters split? | ✅ **DECIDED:** yes; `lambda0`/`F0`/`CR0`; integer `mutation_n_comps` |
| D-A9 | Population memory layout | ✅ **DECIDED:** `d × n` internally, `n × d` in Python |
| D-A11 | Objective calling convention | ✅ **DECIDED:** batch default + per-individual fallback; add speed test |
| D-A13 | RNG and reproducibility guarantee | ✅ **DECIDED:** same seed → same result on the same platform/version only (documented) |
| D-A18 | Python versions / platforms | ✅ **DECIDED:** Python ≥ 3.10; CI on Windows + Linux, Python 3.11; no `-march=native` |
| D-B1 | Accepted problem types | ✅ **DECIDED:** any callable + dim + bounds; the front end wraps plain callables with ioh by default (amended via D-B3) |
| D-D1 | Parameter-conflict policy | ✅ **DECIDED:** user values never overridden; conflicts raise errors; derived values carry provenance |
| D-D3 | Literature fidelity vs. keep 0.0.4 behaviour | ✅ **DECIDED:** bugs (KI-1…15) are fixed; paper deviations (KI-20…28) move to the paper version, each with a before/after impact report |

---

## A. Framework architecture (modcma as template)

### D-A1 🔴 Binding library: nanobind vs. pybind11
- **Context:** modcma's C++ backend uses **pybind11**, not nanobind (confirmed in the compiled `.pyd` and `setup.py`). Choosing nanobind means following modcma's *structure* while rewriting the binding layer from scratch.
- **Options:** (a) nanobind; (b) pybind11 (copy-paste friendly with modcma).
- **nanobind pros:** smaller binaries, lower call overhead, faster compiles, **stable-ABI (abi3) wheels for Python ≥ 3.12**, built-in stub generation (`nanobind.stubgen` / `nanobind_add_stub`), Eigen casters.
- **nanobind cons:** no setuptools helper (CMake required), differences from pybind11 in casters and ownership semantics (stricter about implicit conversions), less code reuse from modcma.
- **Recommendation:** (a) nanobind. The structural reuse from modcma is what matters, and the binding file is small.
- **Status:** DECIDED (2026-11-19): nanobind. modcma is inspiration, not a template: its decisions are not adopted by default, each is weighed on its own merits.

### D-A2 🔴 Build system
- **Options:** (a) scikit-build-core + CMake (nanobind's recommended path); (b) setuptools calling CMake; (c) meson-python.
- **Recommendation:** (a). The same `CMakeLists.txt` also builds the C++-only targets (benchmark driver, optional C++ tests), which modcma keeps in a separate CMake file.
- **Status:** DECIDED (2026-11-19): scikit-build-core + CMake.

### D-A3 🔴 Linear algebra library and how it is included
- **Context:** modcma vendors Eigen in `external/`. Eigen is needed for eigenvector crossover (covariance + `SelfAdjointEigenSolver`) and is handy for population ops.
- **Options:** (a) Eigen vendored (modcma); (b) Eigen as a git submodule; (c) CMake `FetchContent` at build time; (d) no Eigen (own `std::vector` matrices).
- **Recommendation:** Eigen via (b) or (c). Vendoring bloats the repo, and FetchContent needs network access in sdist builds (fine with cibuildwheel; less so for offline installs).
- **Status:** DECIDED (2026-11-19): Eigen as a **git submodule** (b). No strong reason for FetchContent: a submodule works offline, pins an exact Eigen commit, and its sources end up in the sdist. Contributors need `git clone --recursive` (document in the developer guide).

### D-A4 🟠 C++ standard
- **Options:** C++17 (modcma, max compatibility) or C++20 (concepts, `std::span`, designated initialisers).
- **Recommendation:** C++17 unless a compelling C++20 feature appears. nanobind supports both.
- **Status:** DECIDED (2026-11-19): C++17. C++20 adds no feature that would give major speedups for this code; performance comes from Eigen, memory layout and compiler flags. Revisit only if a concrete need appears.

### D-A5 🔴 Package layout and the fate of the pure-Python implementation and the 0.0.4 API
- **Context:** modcma keeps its Python version as a "legacy, not actively developed" implementation next to `modcma.c_maes`. Our 0.0.4 Python code has the bugs listed in KNOWN_ISSUES.
- **Options:**
  1. Remove it; the C++ core is the only implementation.
  2. Keep it as `modde.legacy` (frozen, documented as such).
  3. Keep it only as a test oracle.
  4. Keep the old `ModularDE(problem, **kwargs)` signature as a thin compatibility shim over the C++ core, translating old kwargs.
- **Sub-decision (module naming):** a flat `modde._core` with re-exports in `modde`, or a modcma-style subpackage (`modde.c_de`)?
- **Recommendation:** (1) for the package. Write a fresh NumPy reference *from the spec* for tests (WS3 §5.1). A kwargs shim (4) is cheap if the new `ModularDE` keeps accepting `(problem, **settings)`, so decide it together with D-E4. Use a flat `modde._core` with everything re-exported from `modde`.
- **Status:** DECIDED (2026-11-19): keep the current Python implementation as a **legacy** version that emits a `DeprecationWarning`, and remove it once the new package is fully stable. Proposed mechanics (to confirm): the 0.0.4 classes move to `modde.legacy` (`ModularDE`, `AskTellDE`, `Parameters`, `Population`), which warns on import. Top-level `modde.*` names belong to the new C++-backed API. The legacy code is frozen and gets no bug fixes; KNOWN_ISSUES documents its behaviour. The C++ extension is `modde._core`, re-exported from `modde`.

### D-A6 🔴 Object model
- **Context:** modcma separates `Modules` (enums/bools) → `Settings(dim, modules, **numerics)` → `Parameters` (runtime state, strategy objects) → `ModularCMAES`. Our 0.0.4 has one `Parameters` class mixing all of this.
- **Options:** (a) adopt modcma's model; (b) single settings object.
- **Recommendation:** (a), plus a `resolve()` step that records provenance (WS3 §2). This keeps the API familiar to modcma users.
- **Sub-decisions:**
  - **Option naming:** modcma uses UPPERCASE enum names (`SATURATE`). 0.0.4 uses lowercase strings (`'saturate'`, `'expc_center'`, `'COTN'`, `'LHC'`, `'jDE'`). Recommend UPPERCASE enums in C++, with case-insensitive string parsing in `settings_from_dict`.
  - **Setting names:** keep 0.0.4 names (`lambda_`, `mutation_base`, …) or align with modcma (`lambda0`, …)? Recommend `lambda0`, `F0`, `CR0` (signals "initial value"). Keep module names descriptive (`mutation_base`, `mutation_reference`, …).
  - **Split modules:** should `mutation_n_comps` stay a module (1, 2) or become an integer setting `n_difference_vectors ≥ 1`?
- **Status:** DECIDED (2026-11-19): (a) Modules/Settings/Parameters split. UPPERCASE enum options with case-insensitive string parsing. Settings `lambda0`, `F0`, `CR0`; descriptive module names kept. `mutation_n_comps` becomes an **integer** setting (≥ 0 allowed in the API, subject to resolution checks). Its validity depends on the population size: λ must cover the parents needed, otherwise resolution raises an error. In the ConfigSpace the default range is {1, 2}. Whether 0 gives working configurations (e.g. current-to-pbest/0, or rand base with no reference = pure crossover with a random individual) is checked in the interaction study (WS3 §5.3).

### D-A7 🟠 Strategy dispatch in C++
- **Options:** (a) virtual base classes + factories returning `shared_ptr` (modcma); (b) `std::variant` + `std::visit`; (c) templates / policy-based design (one type per configuration; very fast but combinatorial compile cost).
- **Recommendation:** (a). The virtual call happens once per generation per operator, not per coordinate, so it costs nothing measurable. It also allows Python-side subclasses (D-A8).
- **Status:** DECIDED (2026-11-19): (a) virtual base classes + factories returning `shared_ptr`.

### D-A8 🟢 User-defined modules in Python
- **Context:** modcma exposes a `PySampler`. nanobind supports Python subclasses of C++ classes via trampolines (`NB_TRAMPOLINE`/`NB_OVERRIDE`).
- **Options:** (a) no Python extensibility; (b) only selected hooks (sampler, bound correction, custom mutation); (c) every strategy overridable.
- **Recommendation:** (b), designed in from the start (base-class signatures must be binding-friendly) but implemented after 1.0 if needed.
- **Status:** DECIDED (2026-11-19): (b) selected Python-overridable hooks (sampler, bound correction, custom mutation), with base-class signatures designed to be binding-friendly from the start. More extensibility after 1.0 if needed (FEATURE_REQUESTS.md).

### D-A9 🔴 Population memory layout
- **Context:** 0.0.4 and modcma store individuals as **columns** (`d × n`). ioh's batch call expects **rows** (`n × d`; verified with ioh 0.3.22). numpy and scikit-learn users expect rows too.
- **Options:** (a) `d × n` column-major internally (Eigen default; each individual contiguous), transposed `n × d` views in Python; (b) `n × d` row-major Eigen internally.
- **Recommendation:** (a) internally. **Expose `n × d` everywhere in the Python API**, which is a breaking change from 0.0.4's `population.x` shape and must be documented.
- **Status:** DECIDED (2026-11-19): (a) column-major `d × n` internally; the Python API exposes `n × d` everywhere. Breaking changes vs 0.0.4 are acceptable when they simplify future development.

### D-A10 🟠 Granularity of the step API
- **Context:** modcma exposes `mutate/select/recombine/adapt`. 0.0.4's `bound_correction()` also evaluates (KI-45).
- **Options:** expose `mutate`, `crossover`, `correct_bounds`, `evaluate`, `select`, `adapt` separately, or only `step`/`run`.
- **Recommendation:** expose all of them. A separate `evaluate` is what makes ask-tell (D-A12) and custom evaluation loops trivial.
- **Status:** DECIDED (2026-11-19): expose `mutate`, `crossover`, `correct_bounds`, `evaluate`, `select`, `adapt`, plus `step`/`run`.

### D-A11 🔴 Objective-function calling convention
- **Context:** modcma calls `Callable[[Vector], float]` once per individual. Each call crosses Python ↔ C++ with GIL acquisition, which dominates runtime for cheap objectives.
- **Options:** (a) per-individual only; (b) batch only (`f(X: n×d) -> n`); (c) batch default, per-individual fallback (flag or auto-detection); (d) both, plus a C++ function-pointer path for C++ objectives.
- **Recommendation:** (c), with an explicit flag (`vectorized=True`) instead of fragile auto-detection. ioh problems are batch-capable. Interacts with D-B4.
- **Status:** DECIDED (2026-11-19): (c) batch evaluation by default with an explicit `vectorized` flag, per-individual fallback. Add a **speed test** comparing batch vs per-individual evaluation to the benchmark suite (D-A19).

### D-A12 🟠 Ask-tell interface
- **Context:** modcma's ask-tell is Python-only. Ours exists in 0.0.4 but is fragile (KI-10/11/15) and per-individual.
- **Options:** (a) drop it; (b) per-individual `ask()`/`tell(x, f)` (0.0.4 semantics), implemented in C++; (c) generation-level `ask() -> X (n×d)` / `tell(X, f)`, built on the split step API; (d) (c) plus a per-individual convenience wrapper in Python.
- **Recommendation:** (d). Match individuals by index, not by float equality of `x` (0.0.4 matches with `==`).
- **Status:** DECIDED (2026-11-19): (d) generation-level `ask() -> X` / `tell(X, f)` on top of the split step API, plus a per-individual convenience wrapper in Python. Individuals are matched by index, not by float equality.

### D-A13 🔴 RNG engine and reproducibility guarantee
- **Options (engine):** `std::mt19937_64`, PCG64 (same as numpy's default), xoshiro256**.
- **Options (guarantee):** (1) same seed → same result on the same platform; (2) the same on **every** platform, which needs our own distribution implementations (the `std::normal_distribution` family is implementation-defined); (3) parity with 0.0.4 (not feasible).
- **Recommendation:** per-instance engine (mt19937_64 or PCG64) and guarantee (2) within a minor version. Results change only in changelog-announced releases.
- **Sub-decision:** how QMC samplers (Sobol/Halton) are scrambled and seeded. modcma has a C++ Sobol (`src/sobol.cpp`) and Halton, MIT licensed and potentially reusable with attribution (D-E6).
- **Status:** DECIDED (2026-11-19): guarantee (1): same seed gives the same result on the same platform and version. No promise across platforms, so `std` distributions may be used. Per-instance engine. **Docs must state this guarantee** (see 02_documentation.md, Reproducibility). Bit-reproducibility across platforms is a possible later request (FEATURE_REQUESTS.md).

### D-A14 🟠 Initialisation samplers
- **Context:** 0.0.4 offers gaussian, sobol, halton, uniform, LHC (via scipy). The Gaussian sampler can leave the bounds (KI-9).
- **Decisions:** Keep Gaussian (clip, resample or truncate?) or drop it? Implement LHS in C++? Support user-provided initial populations (0.0.4 has `inialize_custom_pop`; note the typo) and `x0`-centred initialisation?
- **Recommendation:** keep all of them; truncate the Gaussian by resampling; keep custom-population support as `Settings.initial_population` (+ optional known fitness).
- **Status:** DECIDED (2026-11-19): keep all samplers (uniform, Gaussian, Sobol, Halton, LHS). Gaussian is truncated by resampling (fixes KI-9). Custom initial population supported via `Settings.initial_population` (+ optional known fitness).

### D-A15 🟢 Floating type
- **Options:** `double` only (modcma: `Float` typedef) or templated.
- **Recommendation:** a `double` typedef, as in modcma.
- **Status:** DECIDED (2026-11-19): `double` via a `Float` typedef.

### D-A16 🟢 Parallel evaluation and threading
- **Options:** none in the core (batch callback lets users parallelise) vs. a built-in thread pool for C++ objectives.
- **Recommendation:** none in the core. Release the GIL around pure C++ work.
- **Status:** DECIDED (2026-11-19): no parallelisation in the core for now. Release the GIL around pure C++ sections (cheap, keeps the door open).

### D-A17 🟢 Restarts and other new modules
- **Context:** modcma has restart strategies (IPOP/BIPOP), repelling restarts, center placement. DE currently has none.
- **Decision:** is v1 a faithful port of the existing module set (+ fixes), or does it add modules (restarts, rand-to-best, current-to-pbest-w as in jSO, non-linear population reduction, …)?
- **Recommendation:** freeze v1 scope to existing modules + fixes. Design `Parameters` so restarts can be added (cf. modcma's `perform_restart`).
- **Status:** DECIDED (2026-11-19): **no new modules in v1**. Extending the set of DE variants covered is a separate later stage: Workstream 4 / Phase 6 "Scope extension" in README.md. `Parameters` should still be designed so that restarts can be added later.

### D-A18 🔴 Supported Python versions, platforms and compiler flags
- **Context:** 0.0.4 CI tests 3.8–3.10. modcma requires ≥ 3.8 and builds Linux wheels with `-march=native`, which can make wheels crash on older CPUs.
- **Recommendation:** Python ≥ 3.9 (or ≥ 3.10; 3.8 is end-of-life); abi3 wheels for ≥ 3.12; Linux x86_64 + aarch64, Windows x64, macOS arm64 (x86_64 optional); **no** `-march=native` in distributed wheels (opt-in CMake flag for local builds).
- **Status:** DECIDED (2026-11-19): Python **≥ 3.10**. CI tests for now on **Windows x64 (windows-latest) and Linux x86_64 (ubuntu-latest), Python 3.11**, matching the main development machine (Windows 11, Python 3.11). Wheels are built for these two platforms first; macOS and aarch64 come later. No `-march=native` / host-specific flags in distributed wheels. abi3 wheels for Python ≥ 3.12 when wheel publishing starts.

### D-A19 🟠 Performance targets
- **Decision:** what counts as "fast enough"? Proposed metrics: overhead per evaluation (μs) on sphere with batch eval, for d ∈ {10, 100} and λ = 10d; speed-up vs 0.0.4. Proposed target: ≥ 20× speed-up over 0.0.4 for the default configuration at d = 10.
- **Status:** DECIDED (2026-11-19): speed is measured as (1) **overhead per evaluation** (μs, on a trivial objective) and (2) **total wall-clock time to optimise a function X times** (repeated full runs). Target ≈ 20× vs 0.0.4. **Maintainability beats the last few percent**: no added complexity for marginal gains.

### D-A20 🟠 Stats and tracking design
- **Context:** 0.0.4 tracks per-evaluation `curr_F`, `curr_CR`, `corrected`, `CS`, `ED`, `corr_so_far`, `popmean`, `popstd` and `F_Memory_mean`, for logging through ioh. It requires a ~50-line user wrapper.
- **Decisions:** which statistics to keep or add; per-generation vs per-evaluation granularity; opt-in flag (cost); how they are exposed (a struct readable from Python, a callback hook per generation, or a ready-made ioh logging adapter, D-B8).
- **Status:** DECIDED (2026-11-19): for v1, keep the current set of tracked statistics (0.0.4 `TrackedStats` equivalents) in a stats object readable from Python. The direction is a **customisable stats object** where users choose what to track; recorded as a post-1.0 feature request (FEATURE_REQUESTS.md).

### D-A21 🟠 Termination criteria and budget semantics
- **Options:** budget, target value, max generations, stagnation/convergence (none exist in 0.0.4), user callback.
- **Budget:** (a) stop exactly at the budget by evaluating only part of the last generation; (b) allow overshoot (0.0.4, KI-8).
- **Recommendation:** (a) exact, which is required for fair fixed-budget benchmarking. Unevaluated trials are discarded.
- **Status:** DECIDED (2026-11-19): implement exact budget termination **if it causes little slowdown** compared with allowing overshoot (measure it). Overshoot is not critical, since post-processing can handle it, so fall back to allowing it if exactness is costly. Other criteria: budget, target, max generations; no stagnation criteria in v1.

---

## B. Problem interface (ioh)

Facts from ioh 0.3.22: problems expose `meta_data.n_variables`,
`meta_data.optimization_type`, `meta_data.final_target`, `bounds.lb/ub`,
`state.evaluations`, `state.optimum_found` **and** `state.final_target_found`,
`state.current_best`. Calls accept 2D input (batch). `ioh.wrap_problem` wraps
custom functions. Suites include BBOB, CEC2013, CEC2022, PBO, ….

### D-B1 🔴 Accepted problem types
- **Options:** (a) ioh problems only (0.0.4's effective requirement: it reads `meta_data` and `state`); (b) any callable + `dim` + bounds, with ioh problems auto-detected; (c) callables are wrapped into ioh problems via `ioh.wrap_problem`.
- **Recommendation:** (b). ioh stays a dependency (or optional extra, D-B11), and the core never depends on it.
- **Status:** DECIDED (2026-11-19): (b). The core accepts any callable + `dim` + bounds, and ioh problems are auto-detected (dimension, bounds, target, optimisation type read from the problem). The core never depends on ioh. **Docs and examples must show that wrapping a custom function with `ioh.wrap_problem` is generally beneficial**, because it enables ioh logging (IOHanalyzer, tracked attributes) at no extra effort. Tutorial 1 shows the bare callable and the wrapped version side by side.
- **Amended (2026-11-19, via D-B3):** wrapping with `ioh.wrap_problem` becomes the *default* in the Python front end (opt-out possible), so the problem counts the budget. The C++ core still accepts any callable and never depends on ioh.

### D-B2 🟠 Source of bounds
- **Context:** 0.0.4 ignores problem bounds (KI-3).
- **Options:** for ioh problems, always use `problem.bounds` (an explicit user lb/ub is an error, or overrides with a warning?). For bare callables: bounds required, or default `[-5, 5]^d`?
- **Recommendation:** ioh → problem bounds. User override allowed but recorded as `USER` provenance. Callables → bounds **required** (no silent `[-5, 5]`).
- **Status:** DECIDED (2026-11-19): ioh problems supply their bounds. User overrides are recorded as `USER` provenance. For plain callables, bounds are **required** (no silent [-5, 5]) and are passed on to the ioh wrapper (D-B3).

### D-B3 🟠 Budget and target ownership
- **Context:** 0.0.4 stops on `problem.state.evaluations >= budget`, which breaks if a problem is reused without `reset()`, and on `state.optimum_found`. Budget defaults to `1e4 * d`.
- **Options:** count evaluations internally (recommended; works for callables) vs. read `problem.state.evaluations`. Target: use `state.final_target_found` (ioh's configurable final target) vs `optimum_found`. Is the budget required or defaulted?
- **Recommendation:** internal counter; use ioh's final target for ioh problems; budget required in `fmin`, and a documented default in the low-level API.
- **Status:** DECIDED (2026-11-19): **the evaluation budget is counted by the problem**, not by the optimizer, so no evaluation can go uncounted. Consequently **wrapping with ioh is the standard path**: the Python front end wraps a plain callable with `ioh.wrap_problem` (using the given bounds) unless the user opts out. Termination reads `problem.state.evaluations` and the ioh final target (`state.final_target_found`). The budget is required in `fmin`; the low-level default is 10 000·d. The C++ core stays independent of ioh: it receives a callable plus a way to query the evaluation count, so the speed tests can still run without ioh. **Reused problems (decided 2026-11-19):** if the problem already has evaluations when a run starts (e.g. when switching between algorithms on the same problem), modde **warns but does not fail**. Counting stays as normal: the budget applies to the problem's *total* evaluation count, so `problem.state.evaluations ≤ budget` always holds. Still to do: measure ioh's per-evaluation overhead in the D-A19 benchmarks.

### D-B4 🟠 Batch evaluation vs per-evaluation tracking with ioh loggers
- **Context:** ioh loggers can record attributes of the algorithm object (README example: `logged_attributes=['F', 'CR', 'corrected', …]`). With batch evaluation, the attribute values seen during a batch call are those of the generation, not of the individual.
- **Options:** (a) per-individual evaluation when tracking is on (slower, exact); (b) per-generation tracking only; (c) batch evaluation + a C++ hook that updates tracked attributes before each individual (requires per-individual calls into ioh anyway).
- **Recommendation:** (a): tracking mode switches to per-individual calls. Document the performance cost.
- **Status:** DEFERRED TO IMPLEMENTATION (2026-11-19): keep this conflict in mind while designing the evaluation path (WS1). Whatever design is chosen must pass the logging validation test in [03_module_validation.md §5.3a](03_module_validation.md#53a-ioh-logging-validation).

### D-B5 🟢 Direct C++ linkage with IOHexperimenter
- **Context:** IOHexperimenter's C++ core is header-only. A C++ driver could benchmark DE without Python overhead.
- **Options:** never / optional C++ benchmark target / bindings that accept ioh's C++ problem objects directly (zero GIL), which needs a shared ABI with the ioh pybind11 module and is hard.
- **Recommendation:** an optional C++ benchmark target later. No cross-module C++ object passing.
- **Status:** OPEN

### D-B6 🟢 Integer and mixed variables
- **Context:** ioh has integer problems. modcma added `integer_variables` (rounding + sigma floor).
- **Recommendation:** out of scope for v1. Keep `Settings` extensible.
- **Status:** OPEN

### D-B7 🟠 Interaction with ioh constraint handling
- **Context:** ioh problems carry `bounds` with `enforced` / `penalty` behaviour and `add_constraint`. With `bound_correction=NONE`, DE evaluates infeasible points, and what ioh returns depends on its enforcement settings.
- **Decision:** define and document the semantics of `bound_correction=NONE` (evaluate as-is, pass-through to ioh), and whether general constraints are supported at all.
- **Status:** DECIDED (2026-11-19): with `bound_correction=NONE`, infeasible points are evaluated as-is (whatever the ioh problem's own bound handling does). General constraints are not supported in v1.

### D-B8 🟠 Logging and analysis integration
- **Decision:** ship a ready-made adapter (e.g. `modde.ioh.tracked(optimizer)`, or the optimizer exposing tracked properties directly) so `ioh.logger.Analyzer(...).watch(...)` / `ioh.Experiment(logged_attributes=...)` works without the 0.0.4 wrapper class.
- **Status:** OPEN

### D-B9 🟠 Benchmark suites for validation
- **Context:** BBOB, CEC2013 and CEC2022 are available in ioh 0.3.22. CEC2014/2017, where the L-SHADE and jSO results were published, are **not**.
- **Options:** validate on available suites only; add CEC2014/2017 via external code (e.g. `wrap_problem` around a third-party implementation); compare against reference implementations instead of published tables.
- **Status:** DECIDED (2026-11-19): a **separate validation stage** (Phase 4b) runs the presets on CEC suites, wrapping external function implementations with `ioh.wrap_problem`. No explicit integration into ioh. Open sub-decision: which CEC implementation to wrap (see [03_module_validation.md §5.6](03_module_validation.md#56-cec-suite-validation-phase-4b)).

### D-B10 🟢 Maximisation problems
- **Context:** ioh problems carry `optimization_type`. 0.0.4 assumes minimisation.
- **Recommendation:** sign-flip internally when `optimization_type == MAX`. Report values in the original sign.
- **Status:** OPEN

### D-B11 🟠 ioh as a hard or optional dependency
- **Recommendation:** keep it as a hard dependency (it is light and the main use case), but the core never imports it, so it can become optional later.
- **Status:** DECIDED (2026-11-19): **ioh is a hard dependency** (it is the standard evaluation path, D-B3). The C++ core still never includes or links ioh.

---

## C. Tuning interface (ConfigSpace)

Facts: ConfigSpace 1.2.2 installed, modern 1.x API (`ConfigurationSpace.add`,
`Categorical`, `Float`, `Integer`, `EqualsCondition`, `InCondition`,
`Forbidden*Relation/Clause`). modcma's `get_configspace(dim, add_popsize,
add_sigma, add_learning_rates)`:
- builds categoricals automatically by introspecting `Modules` (bool → `(default, not default)`, enum → default first);
- uses `UniformIntegerHyperparameter(log=True)` for population sizes and `NormalFloatHyperparameter(default, 0.3·distance-to-bound)` for floats;
- uses a single `ForbiddenGreaterThanRelation(mu0, lambda0)` and **no conditions**;
- `settings_from_config` passes only non-default numeric values, so derived defaults still apply.

### D-C1 🟠 ConfigSpace version
- **Recommendation:** require `ConfigSpace >= 1.0` (the 1.x API). The 0.x API differs (`add_hyperparameter`, etc.).
- **Status:** OPEN

### D-C2 🟠 Scope of the generated space
- **Options:** modules only; modules + numerics, toggled by flags (like modcma's `add_popsize`, …). Which numerics: `lambda0`, `F0`, `CR0`, `memory_size`, `archive_rate`, `p_best`, `lambda_min`, jDE τ's, SHADE scales, …?
- **Recommendation:** flags per group (`add_popsize`, `add_initial_FCR`, `add_adaptation_params`, …). Each group generated from the registry.
- **Status:** OPEN

### D-C3 🟠 Source of truth for the space
- **Options:** (a) introspect bound enums (modcma); (b) the parameter registry (WS3), which also carries ranges, distributions and active-when conditions; (c) hand-written.
- **Recommendation:** (b). Introspection alone cannot know ranges or conditions.
- **Sub-decision:** where the registry lives: a C++ table bound to Python, or a Python/YAML file from which the C++ defaults are generated (or against which they are tested).
- **Status:** DECIDED (2026-11-19): (c). The registry is a single Python data module in the package (`src/modde/_registry.py`; plain Python, so no PyYAML dependency). C++ defaults are written by hand, and a test checks that the C++ defaults, the registry, the ConfigSpace and the generated docs tables agree.

### D-C4 🟠 Conditional hyperparameters
- **Context:** many parameters are only meaningful with a module: `memory_size` ↔ SHADE; `archive_rate` ↔ archive; `p_best` ↔ `mutation_reference=pbest`; jDE τ's ↔ jDE; `lambda_min` ↔ LPSR; opposition factor ↔ oppositional init.
- **Options:** (a) use ConfigSpace conditions (`EqualsCondition`, `InCondition`); (b) flat space where inactive parameters are ignored (modcma's approach).
- **Recommendation:** (a). It shrinks the effective search space, and SMAC handles conditions natively. Check that the target tuners (irace?) are fine with them.
- **Status:** DECIDED (2026-11-19): use ConfigSpace conditions wherever a parameter is only meaningful under a module setting. The "active when" column of the parameter registry is the source for these conditions.

### D-C5 🟠 How incompatibilities reach the tuner
- **Options:** (a) forbidden clauses generated from the interaction matrix; (b) design every combination to be valid (fix or redefine semantics) and use no forbiddens; (c) let resolution errors become crashed runs.
- **Recommendation:** (b) as the goal, with (a) only for truly impossible combinations. Avoid (c).
- **Status:** OPEN

### D-C6 🟠 Dimension-dependent ranges
- **Context:** modcma uses absolute `lambda0 ∈ [1, 50·dim]`, so a space is tied to one dimension.
- **Options:** absolute population size vs. a multiplier of d (e.g. `lambda_factor ∈ [2, 30]`, λ = round(factor·d)), or a log-scale absolute size.
- **Recommendation:** a multiplier, which transfers across dimensions and suits per-dimension tuning studies. Also allow an absolute value outside the configspace.
- **Status:** DECIDED (2026-11-19): the ConfigSpace tunes **`lambda0` directly as an integer** (log scale), not as a multiplier of d, because good population sizes need not scale linearly with dimension. The space therefore depends on `dim` (as in modcma: `get_configspace(dim)`). The proposed range is [parents needed + 1, 50·d] with default 10·d, to be fixed in the registry.

### D-C7 🟢 Prior distributions for numerics
- **Options:** uniform (unbiased) vs normal around the default (modcma) vs log-scale.
- **Recommendation:** uniform or log-uniform by default. Normal priors only if requested (they bias tuners such as SMAC's initial design).
- **Status:** OPEN

### D-C8 🟠 Round-trip API
- **Decisions:** functions `get_configspace`, `settings_from_config(dim, config, **overrides)`, `settings_from_dict`, and **also** `config_from_settings` (missing in modcma; useful for warm-starting tuners from presets). How a configuration distinguishes "explicitly tuned" from "default": with provenance (D-D1), every configured value is `USER`.
- **Status:** OPEN

### D-C9 🟠 ConfigSpace as a hard or optional dependency
- **Context:** modcma makes it a hard dependency.
- **New consideration (2026-11-19):** D-D2 defines the named presets as **ConfigSpace configurations**, so the presets (a core convenience) now need ConfigSpace. Options: (a) make ConfigSpace a hard dependency (simplest; it is pure Python and light); (b) keep it optional, storing presets internally as plain settings dicts and converting them to `Configuration` objects only when ConfigSpace is installed.
- **Recommendation (revised):** (a) hard dependency, since presets are a main entry point for users. (b) adds a second representation to keep in sync.
- **Status:** DECIDED (2026-11-19): (a) **ConfigSpace (≥ 1.0) is a hard dependency.**

### D-C10 🟢 Other tuners and formats
- **Decision:** support irace's parameter-file format (not provided by ConfigSpace), and Optuna/Hyperopt adapters?
- **Recommendation:** an irace exporter, if the group uses irace. Otherwise ConfigSpace only.
- **Status:** OPEN

### D-C11 🟠 Default configuration of the space
- **Decision:** which algorithm the space's default represents (classic DE/rand/1/bin? L-SHADE-like?). Tied to D-D2.
- **Status:** DECIDED (2026-11-19): the ConfigSpace default configuration is classic DE (see D-D2).

---

## D. Module semantics and parameter policy (WS3)

### D-D1 🔴 Policy for parameter conflicts and derived values
- **Options:** (a) silent adjust (0.0.4); (b) adjust + warning; (c) user values never adjusted, conflicts raise errors, derived values recorded with provenance and inspectable.
- **Recommendation:** (c).
- **Status:** DECIDED (2026-11-19): (c). User-set values are never adjusted. Conflicts raise a clear error naming the value and the module(s) involved. Derived values are recorded with their provenance (`USER` / `DEFAULT` / `DERIVED(by=…)`) and can be inspected on the resolved settings.

### D-D2 🟠 Default algorithm
- **Context:** the 0.0.4 defaults are uniform init, saturate, rand/1, bin, F=0.5, CR=0.5, λ=10d.
- **Options:** keep these, classic Storn & Price (F=0.5, CR=0.9), or an adaptive default (e.g. SHADE-like) for out-of-the-box performance.
- **Recommendation:** decide based on audience. For `fmin` users an adaptive default performs better. For a "baseline" the classic one is clearer. One option: modules default to classic, `fmin` defaults to a preset.
- **Status:** DECIDED (2026-11-19): the default (all modules at their defaults, and `fmin` without arguments) is **classic DE**: DE/rand/1/bin, F0 = 0.5, CR0 = 0.9, uniform initialisation, saturate bound correction, λ0 = 10·d. **Popular configurations are available as named aliases**: a dictionary mapping names (e.g. `"LSHADE"`, `"SHADE"`, `"JADE"`, `"jDE"`, `"DE/best/1/bin"`, …) to their ConfigSpace realizations (`Configuration` objects of `get_configspace(dim)`), usable directly with `settings_from_config` / `fmin`. Because some preset values depend on `dim` (e.g. L-SHADE's 18·d), the dictionary is built per dimension (e.g. `get_presets(dim)`). Exact preset list is fixed in Phase 1.

### D-D3 🔴 Literature fidelity vs. 0.0.4 behaviour
- **Context:** KNOWN_ISSUES §B (KI-20…28): Lehmer mean, jSO caps, jDE success rule, distinct parents, weighted-F scope, `pbest_value` semantics, archive init, archive term, LPSR minimum.
- **Options per item:** fix to the literature; keep the 0.0.4 behaviour as a named variant; keep the 0.0.4 behaviour and document it.
- **Recommendation:** the literature is the default. Keep a 0.0.4 variant only where the GECCO'23 results depend on it and reproducibility of those results matters. Bugs (KI-1…15) are simply fixed.
- **Status:** DECIDED (2026-11-19): all bugs in KNOWN_ISSUES §A (KI-1…15) are fixed in the port. All paper deviations (KI-20…28) move to the paper version; the 0.0.4 behaviour is not kept as an option. Each change is documented by a before/after impact report on BBOB d = 5, 20 (module alone + defaults, and L-SHADE with/without the change): [studies/paper_deviations/](studies/paper_deviations/).

### D-D4 🟠 Archive and memory sizing
- **Options:** archive as absolute size vs rate × λ (JADE: 1.0; L-SHADE: 2.6; 0.0.4: 2.0 forced). Behaviour under LPSR (shrink with λ?). Default `memory_size` (SHADE: H = N; L-SHADE: 6; 0.0.4: 100).
- **Recommendation:** `archive_rate` (default 1.0 or 2.6?) and capacity tracking current λ; a single `memory_size` default; presets carry paper values.
- **Status:** DECIDED (2026-11-19): `archive_rate` (capacity = round(rate × current λ), so the archive shrinks with LPSR as in L-SHADE), default 1.0 (JADE). `memory_size` default 6 (L-SHADE). Presets carry paper values.

### D-D5 🟠 LPSR end population
- **Options:** derived from the mutation's parent needs (0.0.4) vs explicit `lambda_min` with default 4 (L-SHADE) and validation against parent needs.
- **Recommendation:** explicit `lambda_min`.
- **Status:** DECIDED (2026-11-19): explicit `lambda_min` setting. Its default is derived as max(4, parents needed + 1) (`DERIVED` provenance). A user value below what the mutation needs raises an error (D-D1).

### D-D6 🟠 Population-size specification and default
- **Options:** absolute `lambda0`, `lambda_factor × d`, or both. Default 10d vs other (e.g. 18d L-SHADE, 25·log(d)·√d jSO). The default must not depend on LPSR (KI-31).
- **Status:** DECIDED (2026-11-19): **`lambda0` is a true integer parameter** with a single default of 10·d for every configuration. LPSR (or any other module) never changes that default (fixes KI-31); LPSR only uses `lambda0` as its starting size. Presets set their own values (e.g. L-SHADE 18·d). See D-C6 for tuning.

### D-D7 🟠 Budget enforcement mid-generation
- See D-A21. Additional question: in a partial final generation, are the evaluated trials still selected? (Yes, for any-time correctness.)
- **Status:** DECIDED (2026-11-19): covered by D-A21 (exact budget if cheap; overshoot acceptable otherwise).

### D-D8 🟠 Parent selection rules
- **Options:** require mutually distinct parents (classic DE) and raise an error if λ is too small, vs allow repeats (0.0.4). Is the archive drawn from P ∪ A or from A only (KI-27)?
- **Recommendation:** distinct parents; P ∪ A, following JADE.
- **Status:** DECIDED (2026-11-19): covered by KI-23 and KI-27 (paper versions). Parents are mutually distinct and distinct from the target; the archive term draws from P ∪ A. A population too small for the required distinct parents raises an error (D-D1).

### D-D9 🟠 F/CR specification API
- **Options:** scalars only; scalars or per-individual arrays; distributions (e.g. "F ~ U(0.4, 0.9) each generation" as a non-adaptive module).
- **Recommendation:** scalar `F0`/`CR0`. Arrays only via the low-level API.
- **Status:** DECIDED (2026-11-19): scalar `F0` and `CR0` in the Settings. Per-individual arrays only through the low-level runtime API.

### D-D10 🟠 v1 module scope
- See D-A17. List the modules to drop (if any: e.g. `shade_modified`, `exps`?) and to add.
- **Status:** DECIDED (2026-11-19): **keep every module option currently implemented** (incl. `shade_modified`, `oversampling_factor`, `eigenvalue_crossover`, all bound corrections). **Add a paper-accurate L-SHADE adaptation option** (e.g. `adaptation_method_CR = LSHADE`): weighted Lehmer mean for the CR memory plus L-SHADE's terminal-value rule (a memory slot whose successful CRs are all 0 becomes terminal, and CR = 0 is used from then on). It makes a faithful L-SHADE preset possible; it is the only option added before 1.0 (an exception to D-A17).

### D-D11 🟠 Oppositional generation semantics
- **Decisions:** where it sits in the generation (before/after adapt); whether selection is (λ + λ) truncation; whether it consumes budget (yes) and respects the budget limit; how F/CR/memory follow reordered individuals; whether the jumping probability is a module or a numeric setting.
- **Status:** DECIDED (2026-11-19): oppositional generation runs after selection and before adaptation. Opposite points use the current population's per-dimension bounds (KI-6 fix). The best λ of population ∪ opposite points are kept. Opposite points inherit F/CR from their source individual. Their evaluations count towards the budget and are cut off at the budget. Configured as an on/off module plus a `jump_rate` setting, default 0.3 (Rahnamayan et al., ODE).

### D-D12 🟠 Ask-tell parity
- **Decision:** must ask-tell produce **identical** trajectories to `run()` for the same seed? Recommended: yes, as a tested invariant.
- **Status:** OPEN

### D-D13 🟠 Reproducibility guarantee
- See D-A13. Also: what changes count as "behaviour-changing" and need a changelog entry and a new regression snapshot.
- **Status:** DECIDED (2026-11-19): covered by D-A13 (same seed → same result on the same platform and version; documented).

### D-D14 🟠 Validation acceptance criteria
- **Decision:** when a module or preset counts as "validated". Proposed: operator tests pass, invariants hold for N random configurations, and the preset is statistically indistinguishable (or not significantly worse, α = 0.05, ≥ 25 runs) from the reference implementation on the agreed suite.
- **Status:** OPEN

---

## E. Documentation and project

### D-E1 🟠 Documentation tooling
- **Options:** Sphinx (+ autodoc/napoleon, `myst-nb` for notebooks; modcma uses Read the Docs) vs MkDocs Material (+ `mkdocstrings`, `mkdocs-jupyter`).
- **Constraint:** the API docs must come from nanobind stubs and docstrings. Both handle `.pyi` with some setup.
- **Status:** OPEN

### D-E2 🟢 Hosting
- **Options:** Read the Docs vs GitHub Pages.
- **Status:** OPEN

### D-E3 🟠 Tutorial format
- **Options:** Jupyter notebooks (rendered, executed in CI) vs plain scripts (+ literalinclude).
- **Status:** OPEN

### D-E4 🟠 Versioning and naming
- **Options:** `1.0.0` (signals the rewrite and the breaking API) vs `0.1.0`. Same PyPI name `modde`.
- **Recommendation:** pre-releases `1.0.0a*` during Phases 3–4, then 1.0.0.
- **Status:** OPEN

### D-E5 🟢 Repository home
- **Context:** the repo is `Dvermetten/ModDE`; modcma lives under the `IOHprofiler` org.
- **Options:** keep it, or move it to `IOHprofiler/ModularDE` for discoverability next to modcma and ioh.
- **Status:** OPEN

### D-E6 🟠 Code reuse from modcma and licensing
- **Decision:** which modcma C++ code to reuse (Sobol/Halton samplers, bound corrections, `Population`, `ndtr_ndtri.cpp`), and how to attribute it. Both projects are MIT.
- **Finding (Phase 2, 2026-11-19):** modcma's `src/sobol.cpp` is John Burkardt's code under the **GNU LGPL**, not MIT, so it was not reused. modde has its own Sobol (Joe & Kuo direction numbers, BSD-style licence shipped in `src/modde/_licenses/`) and Halton. **No modcma code is currently used**; everything else was written from the spec.
- **Status:** OPEN (no reuse so far; decide only if reuse comes up again)

### D-E7 🟢 Docstring source for C++ objects
- **Options:** docstrings written in `bindings.cpp` (single place, appears in stubs) vs Doxygen in headers + Breathe.
- **Recommendation:** in `bindings.cpp`, generated from the registry where possible.
- **Status:** OPEN

### D-E8 🟢 Test framework
- **Options:** Python-only tests (pytest via the bindings) vs also C++ unit tests (Catch2/GoogleTest).
- **Recommendation:** pytest for almost everything; a minimal C++ test target only for things not reachable from Python.
- **Status:** OPEN

---

## Decision log

| Date | Decision | Affects |
|------|----------|---------|
| 2026-11-19 | Bindings use **nanobind**. modcma is inspiration only; its choices are not adopted by default. | D-A1, all of WS1 |
| 2026-11-19 | No `-march=native` / host-specific flags in distributed wheels. | D-A18 |
| 2026-11-19 | All bugs in KNOWN_ISSUES §A (KI-1…15) are fixed in the port. | D-D3, KNOWN_ISSUES |
| 2026-11-19 | Paper deviations (KI-20…28) all move to the **paper version**. Each gets a short before/after report (BBOB d = 5, 20; module + defaults; L-SHADE with/without). | D-D3, KNOWN_ISSUES §B, studies/paper_deviations |
| 2026-11-19 | Batch evaluation vs per-individual ioh logging: resolve during implementation; the result must pass a dedicated logging validation test. | D-B4, WS3 §5.3a |
| 2026-11-19 | ConfigSpace uses **conditional parameters** where a parameter only makes sense under a module setting. | D-C4 |
| 2026-11-19 | CEC2014/2017 (and other suites not in ioh) are tested in a **separate stage** (Phase 4b) via `ioh.wrap_problem`; no integration into ioh. | D-B9, WS3 §5.6 |
| 2026-11-19 | Build: scikit-build-core + CMake; Eigen as git submodule; C++17; virtual strategy classes. | D-A2, D-A3, D-A4, D-A7 |
| 2026-11-19 | 0.0.4 Python code kept as deprecated legacy, removed once the new package is stable. | D-A5 |
| 2026-11-19 | modcma-style object model; `lambda0`/`F0`/`CR0`; UPPERCASE enums; integer `mutation_n_comps` (ConfigSpace range {1, 2}; validity of 0 to be checked). | D-A6 |
| 2026-11-19 | Python extensibility limited to selected hooks; more after 1.0 if needed. | D-A8 |
| 2026-11-19 | `d × n` internally, `n × d` in Python; breaking changes acceptable. Full split step API exposed. | D-A9, D-A10 |
| 2026-11-19 | Batch evaluation default + per-individual fallback, with a speed test. Generation-level ask-tell + per-individual wrapper. | D-A11, D-A12 |
| 2026-11-19 | Reproducibility: same seed → same result on same platform/version only; documented. | D-A13 |
| 2026-11-19 | All samplers kept; double precision; no parallelisation in core. | D-A14, D-A15, D-A16 |
| 2026-11-19 | No new modules in v1; scope extension becomes Workstream 4 / Phase 6. | D-A17 |
| 2026-11-19 | Python ≥ 3.10; CI on Windows + Linux with Python 3.11. | D-A18 |
| 2026-11-19 | Speed metrics: overhead per evaluation + total time for repeated runs; target ≈ 20×; maintainability first. | D-A19 |
| 2026-11-19 | Stats: keep the 0.0.4 tracked set for v1; customisable stats object is a post-1.0 feature request. | D-A20 |
| 2026-11-19 | Exact budget termination if cheap; overshoot otherwise acceptable. | D-A21 |
| 2026-11-19 | Parameter policy: user values never overridden, conflicts raise errors, provenance recorded. | D-D1 |
| 2026-11-19 | Any callable accepted (ioh auto-detected); examples show that wrapping with `ioh.wrap_problem` is beneficial for logging. | D-B1, WS2 |
| 2026-11-19 | Parameter registry: one Python data module (`src/modde/_registry.py`); C++ defaults hand-written and checked by a test. | D-C3 |
| 2026-11-19 | Keep all current module options; add a paper-accurate L-SHADE adaptation option (Lehmer CR + terminal value). | D-D10 |
| 2026-11-19 | `lambda0` is a true parameter, default 10·d for every configuration, never changed by LPSR; tuned directly (not as a multiplier) in a dim-dependent ConfigSpace. | D-D6, D-C6 |
| 2026-11-19 | `archive_rate` default 1.0 (shrinks with λ); `memory_size` default 6; explicit `lambda_min` (default max(4, parents+1)); scalar `F0`/`CR0`. | D-D4, D-D5, D-D9 |
| 2026-11-19 | Default = classic DE (DE/rand/1/bin, F0 0.5, CR0 0.9), also for `fmin`; popular configurations as named aliases mapping to ConfigSpace configurations, built per dimension. | D-D2, D-C11 |
| 2026-11-19 | Oppositional generation semantics fixed (after selection, per-dimension bounds, (λ+λ) truncation, inherit F/CR, budgeted, `jump_rate` 0.3). | D-D11 |
| 2026-11-19 | Closed as covered by earlier decisions: D-D7 (D-A21), D-D8 (KI-23/27), D-D13 (D-A13). | D-D7, D-D8, D-D13 |
| 2026-11-19 | Budget is counted by the problem; the front end wraps plain callables with ioh by default; ioh becomes a hard dependency; bounds required for plain callables; NONE correction evaluates infeasible points as-is. | D-B1 (amended), D-B2, D-B3, D-B7, D-B11 |
| 2026-11-19 | ConfigSpace (≥ 1.0) is a hard dependency. | D-C9 |
| 2026-11-19 | Problem with existing evaluations at start: warn, don't fail; budget applies to the problem's total count (`evaluations ≤ budget`). | D-B3 |
| 2026-11-19 | Implement quasi-oppositional initialisation using `oppositional_initialization_factor` (KI-37): `oppositional_initialization ∈ {NONE, OPPOSITE, QUASI_OPPOSITE}`, factor scales the opposite point toward the centre. | D-D10, KI-37 |
| 2026-11-19 | Phase 1 spec review: renamed settings, fixed constant schedules, F0/CR0 as SHADE memory init, and the eigen-crossover note accepted. Bound-correction references taken from the ModDE paper; preset verification deferred. M1 reached. | spec/README.md |
