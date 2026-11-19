# Workstream 3: Module Validation, Interactions and Parameter Policy

## Objectives
1. **Correctness**: every module does exactly what its written spec says, and the spec matches the cited literature (or documents why not).
2. **Interactions**: every pair of module options is classified, and incompatibilities are documented (and enforced where needed).
3. **Explicit parameters**: no parameter is changed without the user being able to see it. In particular, a value the user sets is **never** overridden silently (e.g. population size, archive size, memory size).

## 1. Module inventory (0.0.4)

| Category | Setting (0.0.4 name) | Options | Extra parameters it pulls in | In `__modules__` / tested? |
|----------|----------------------|---------|------------------------------|---------------------------|
| Initialisation | `base_sampler` | gaussian, sobol, halton, uniform, LHC | — | yes |
| | `oppositional_initialization` | on/off | `oppositional_initialization_factor` (unused, KI-37) | yes |
| | `oversampling_factor` | float | — | **no** (and a no-op, KI-4) |
| Mutation | `mutation_base` | rand, best, target | — | yes |
| | `mutation_reference` | None, pbest, best, rand | `pbest_value` | yes (pbest_value: no) |
| | `mutation_n_comps` | 1, 2 | — | yes (but buggy, KI-1) |
| | `mutation_use_weighted_F` | on/off | hard-coded schedule | yes |
| | `use_archive` | on/off | `archive_size` (overridden, KI-30) | yes |
| Crossover | `crossover` | bin, exp | — | yes |
| | `eigenvalue_crossover` | on/off | — | yes |
| Bound correction | `bound_correction` | None, saturate, unif_resample, COTN, toroidal, mirror, hvb, expc_target, expc_center, exps | — | yes |
| Adaptation | `adaptation_method_F` | None, shade, shade_modified, jDE | `memory_size`, jDE constants | yes |
| | `adaptation_method_CR` | None, shade, jDE | `memory_size`, jDE constants | yes |
| | `use_jso_caps` | on/off | hard-coded thresholds | yes |
| Population size | `lpsr` | on/off | derived `min_lambda` | yes |
| Oppositional | `oppositional_generation_probability` | float | — | **no** |
| Strategy params | `lambda_`, `F`, `CR` | — | — | no |

## 2. Parameter policy (no silent overrides)

### Principles (proposed, see D-D1)
1. Every parameter has exactly one entry in a **parameter registry** (§3).
2. Each resolved value carries a **provenance**: `USER` (explicitly passed), `DEFAULT` (static default), or `DERIVED` (computed from other settings, recording which ones).
3. A `USER` value is **never** changed by module logic. If a module combination cannot work with the value, resolution raises a clear error naming both the value and the module, e.g. *"lambda0=4 is too small for mutation (rand/2 + archive needs ≥ 6 distinct individuals)"*.
4. Values that change **during the run by design** (population size under LPSR, F/CR under adaptation, archive capacity tracking population size) are modelled as *initial value + schedule*. The documentation and the resolved settings say so: `lambda0` is the initial size, `lambda_min` the final one.
5. Resolved settings are inspectable: `de.settings.explain()` (or similar) prints every value with its provenance.
6. Nothing depends on *which* modules are on unless the registry says so. For example, the default population size is the same with and without LPSR (fixes KI-31), unless we decide to keep a documented derived default.

### Known violations to remove (from KNOWN_ISSUES §C)
| Issue | Policy fix |
|-------|-----------|
| KI-30 `archive_size` overwritten | Replace with `archive_rate` (user value kept). Capacity = `round(rate × current λ)` is a documented schedule. |
| KI-31 λ default depends on LPSR | `lambda0` is a true parameter with one default (10·d) for every configuration; LPSR never changes it (D-D6). Presets set their paper values explicitly. |
| KI-32 `memory_size=100` default | Default 6 for every configuration (D-D4). |
| KI-33 LPSR end size derived from mutation | Explicit `lambda_min` setting. Default derived as max(4, parents needed + 1) (D-D5). Error if a user value is below the minimum the mutation needs. |
| KI-34 hard-coded constants | Promoted to settings with registry defaults: jDE τ_F, τ_CR, F_l, F_u; pbest range; SHADE scales; weighted-F schedule; jSO thresholds; HVB α. |
| KI-35 budget default | The budget is counted **by the problem** (plain callables are wrapped with ioh by default). Required in `fmin`; low-level default 10 000·d (D-B3). |
| KI-36 F/CR array typing | Scalar `F0`, `CR0` in the Settings; arrays only via the low-level runtime API (D-D9). Clearly "initial values" when adaptation is on. |
| KI-37 unused factor | Implement quasi-opposition or remove. |

## 3. Parameter registry (Phase 1 deliverable)

A single Python data module, `src/modde/_registry.py` (D-C3). C++ defaults
are written by hand, and a test checks that they agree with the registry.
Columns:

| name | type | range | default rule | active when | provenance rules | tunable? | ConfigSpace distribution | docs text | literature |
|------|------|-------|--------------|-------------|------------------|----------|--------------------------|-----------|------------|

It generates or validates:
- the C++ `Settings::resolve()` defaults (via a test that the two agree);
- the ConfigSpace (hyperparameters, conditions, forbiddens), which depends on `dim` because `lambda0` is tuned directly (D-C6);
- the named presets (`get_presets(dim)`, D-D2);
- the module reference tables in the docs.

## 4. Module spec (Phase 1 deliverable)

`dev/spec/<category>.md` for each category. For each option:
- formula, with indices and every random draw made explicit (what is drawn, from which distribution, how many, in what order);
- edge cases: population smaller than needed parents, all individuals identical, empty archive, budget exhausted mid-generation, d = 1;
- literature reference + deviations (from KNOWN_ISSUES §B triage);
- properties that tests will check (§5.1).

## 5. Test strategy

All tests use **pytest** (replacing the `unittest` metaclass generator) and
target the new Python API, so they cover the C++ core through the bindings.

### 5.1 Operator-level tests (deterministic)
Use the injectable random-draw source (WS1). Feed known draws, assert exact output against values computed by hand or by a tiny NumPy reference implementation of the spec, kept in `tests/reference/`.

Property tests per operator:
- **Mutation**: chosen parents ≠ target and pairwise distinct (if D-D8 says so). `rand/2` uses 4 distinct parents (regression for KI-1). The pbest index is in the top ⌈p·n⌉. The archive term draws from P ∪ A (if KI-27 is fixed that way).
- **Binomial crossover**: at least one coordinate from the donor (`j_rand`). Expected fraction ≈ CR (statistical).
- **Exponential crossover**: donor coordinates form a single cyclic contiguous block. Its length follows a truncated geometric distribution with parameter CR.
- **Eigen crossover**: equals plain crossover when the covariance is diagonal and isotropic.
- **Bound correction**: output within [lb, ub] for every method except NONE. In-bounds coordinates untouched. Saturate/mirror/toroidal match closed forms. Stochastic methods' distributions checked against their spec (KS test).
- **Selection**: per index, `f_new ≤ f_old`. Ties go to the trial (or as specified). Success indices are correct.
- **SHADE**: memory update equals the weighted Lehmer mean (F) and weighted mean (CR). Memory index wraps. F ∈ (0, 1], CR ∈ [0, 1].
- **jDE**: the regeneration rate ≈ τ (regression for KI-2). Kept only on success if KI-22 is fixed.
- **jSO caps**: per the verified paper semantics (KI-21).
- **LPSR**: λ(t) follows the linear schedule. λ = λ_min exactly at the end of the budget. The worst individuals are removed. F/CR/memory stay aligned.
- **Archive**: never exceeds capacity. Inserts exactly the replaced parents. Shrinks with LPSR.
- **Samplers**: points within bounds (regression for KI-9). Opposition produces lb + ub − x. Oversampling evaluates n(1+o) points and keeps the best n (regression for KI-4).
- **Oppositional generation**: per-dimension bounds (KI-6). State stays aligned afterwards (KI-7).

### 5.2 Invariant tests over random configurations
Use property-based testing (`hypothesis`) to sample complete configurations from the ConfigSpace and run short runs (e.g. d ∈ {1, 2, 5, 20}, budget ≈ 50·d). Check invariants **every generation**:
- evaluations ≤ budget, and exactly = budget at the end if no target was hit (KI-8);
- population size equals the resolved schedule; F/CR/arrays have matching lengths;
- with correction ≠ NONE, every evaluated point lies within bounds;
- no NaN/inf in state;
- the same seed gives an identical trajectory (determinism), and different seeds give different ones;
- `USER`-provenance settings are unchanged after the run (no silent override, enforced by tests).

### 5.3 Interaction matrix
- Enumerate **all pairs** of module options (covering-array style) against a baseline. Add targeted higher-order combinations where we suspect interactions (archive × LPSR × SHADE, oppositional generation × SHADE × archive, eigen crossover × exp crossover, small λ × many difference terms).
- Run each on a small set of functions (e.g. BBOB f1, f8, f15, f21 at d = 5, 20). Several seeds.
- Classify each pair as:
  - ✅ **compatible**;
  - ⚠️ **changes meaning**: works, but one module's semantics shift (e.g. weighted F with an archive term; best-base + best-reference makes the reference term a no-op?);
  - ⭕ **no effect**: one module is a no-op in this context (e.g. jSO caps without adaptation? CR adaptation with … );
  - ❌ **incompatible**: errors or is degenerate. Becomes a validation error in `resolve()` **and** a ConfigSpace forbidden clause, or is fixed so the combination becomes valid (preferred, D-C5).
- Output: `results/interaction_matrix.csv` → generated docs page.

Candidate interactions to check first (from the code review):
- `mutation_n_comps = 0` (now an integer, D-A6): which combinations still produce a working DE? Candidates: current-to-pbest/0 (`x + F·(x_pbest − x)`) and best-base + rand-reference look viable; rand base + no reference + no archive degenerates into crossover with a random individual. The result decides whether 0 is allowed by `resolve()` and whether the ConfigSpace range stays {1, 2}.
- `mutation_n_comps ≥ 3` × small or LPSR-shrunk λ: resolution must reject λ (or λ_min) below the number of distinct parents needed.
- `mutation_base=best` × `mutation_reference=best`: the reference term is identically zero.
- `mutation_base=target` × `mutation_reference=None` × `n_comps=1`: DE/current/1, which is valid but has weak exploration (document only).
- `use_archive` × `n_comps=1`: the archive term *replaces* the only difference term (KI-27).
- `lpsr` × small λ0 × `n_comps=2` + archive: λ_min vs parents needed.
- `oppositional_generation` × SHADE / archive / init_stats (KI-7, KI-14).
- `adaptation_method_F=None` × `use_jso_caps`: caps act on constant F/CR.
- `adaptation_method_F=jDE` × `adaptation_method_CR=shade`: a mixed scheme. Is it valid and meaningful?
- `eigenvalue_crossover` × `exp` crossover; × small populations (n < d gives a singular covariance).
- `bound_correction=None` × ioh problems: out-of-bounds evaluation, and ioh's own bound enforcement (D-B7).
- `base_sampler=gaussian` × `oppositional_initialization`.
- Ask-tell × every module (parity with `run`).

### 5.3a ioh logging validation
**Reminder from D-B4:** batch evaluation (fast) and per-individual logging of
internal state through ioh loggers (exact) pull in opposite directions. The
evaluation path is designed during implementation. Whatever design is chosen
must pass these checks:
- With an ioh logger watching tracked attributes (F, CR, corrected, CS, ED, …), every logged row carries the values belonging to **the individual that was evaluated**, not the generation's last or first individual.
- Do this for every evaluation mode we expose (batch, per-individual, ask-tell), and with oppositional initialisation/generation turned on (extra evaluations outside the normal generation order, cf. KI-14).
- The problem's evaluation count (which is what counts the budget, D-B3) equals the logged count and equals the budget at the end of the run.
- The cost of tracking mode is measured and documented (WS1 benchmarks).

### 5.4 Reference-variant validation
Recreate published algorithms as presets and compare statistically (e.g. 25–51 runs, Mann-Whitney U / equivalence test on final error):

| Variant | Compare against | Suite |
|---------|-----------------|-------|
| DE/rand/1/bin | textbook implementation (small NumPy reference) | BBOB d=5,20 |
| JADE | reported results / reference code | BBOB or CEC2013 |
| SHADE | authors' reference code (Tanabe) / CEC2013 results | CEC2013 (in ioh) |
| L-SHADE | CEC2014 published results / reference code | CEC2014 → separate stage, §5.6 |
| jDE | reference results | BBOB |
| jSO-style | CEC2017 results | CEC2017 → separate stage, §5.6 |

Acceptance criteria per D-D14.

### 5.5 Regression and performance
- Replace `expected.json` (bit-exact under 0.0.4) with a new snapshot from the C++ core, after the fixes, with a tolerance-based comparison. Regenerated deliberately, never automatically.
- Benchmark runs in CI (non-blocking) to catch performance regressions.

### 5.6 CEC-suite validation (Phase 4b)
A separate stage after the main validation campaign (decided 2026-11-19, D-B9).
The published L-SHADE and jSO results are on CEC2014 and CEC2017, which ioh
0.3.22 does not include.
- **Approach:** wrap an external implementation of the CEC functions with `ioh.wrap_problem` (setting the bounds and an optimum/offset per function), so the rest of the pipeline (loggers, `ioh.Experiment`, IOHanalyzer) works unchanged. **No integration into ioh itself.**
- **Open sub-decision:** which implementation to wrap. Candidates: a Python package of CEC suites (e.g. `opfunu`, which should be checked against the official C code) or bindings around the official C/C++ code. First validate the wrapped functions themselves: compare function values at fixed test points with the official implementation.
- **Protocol:** follow the competition settings of each suite (dimensions, budget = 10000·d, 51 runs, error threshold 1e-8) so results are directly comparable to the published tables.
- **Scope:** CEC2014 for L-SHADE, CEC2017 for jSO-style. CEC2013/CEC2022 runs (available in ioh) can go in the same report.
- **Output:** a section in the validation report, plus a comparison table per preset vs published results.
- **Lives outside the core test suite:** scripts in `benchmarks/cec/` (or a separate repo), optional dependencies, not run in CI.

## 6. Tasks

### Phase 1
- [x] Triage KNOWN_ISSUES §A: all bugs fixed in port (2026-11-19).
- [x] Triage KNOWN_ISSUES §B (paper deviations): all move to the paper version (2026-11-19).
- [x] Verify KI-21 (jSO caps) and KI-28 (L-SHADE N_min) against references.
- [ ] Before/after impact report per deviation ([studies/paper_deviations/](studies/paper_deviations/)). The paper-version implementations in `harness.py` serve as reference behaviour for the spec.
- [x] Write `dev/spec/*.md` (draft, awaiting author review).
- [x] Build the parameter registry (`src/modde/_registry.py`) and resolution rules (provenance, spec/settings.md).
- [x] Decide the D-D decisions needed for the spec (D-D1…D-D11, D-D13; D-D12/D-D14 later).

### Phase 2 (alongside the C++ port)
- [ ] `tests/reference/`: minimal NumPy reference operators written directly from the spec (not copied from 0.0.4).
- [ ] Operator-level tests (§5.1) per module as it lands.

### Phase 4
- [ ] Invariant/property tests (§5.2) in CI.
- [ ] Interaction matrix runs + classification + generated docs page.
- [ ] Reference-variant validation report (`dev/validation_report.md` or a docs page).
- [ ] New regression snapshot.
- [ ] ioh logging validation (§5.3a).

### Phase 4b
- [ ] Choose and validate a CEC2014/2017 implementation (§5.6).
- [ ] Wrap it with `ioh.wrap_problem`. Run the presets under the competition protocol.
- [ ] Compare against published results. Add to the validation report.
