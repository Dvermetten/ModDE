# Settings, resolution, problems, termination and statistics

## 1. Settings and provenance

A run is configured by a set of named settings (registry: `PARAMS`).
**Resolution** turns what the user passed into a complete, validated set in
which every value carries a **provenance** (D-D1):

| Provenance | Meaning |
|---|---|
| `USER` | passed explicitly by the user (or by a preset / configuration). The Python front end does not report "no effect" warnings for values that only the preset set. |
| `PROBLEM` | read from the problem (dimension, bounds, target) |
| `DEFAULT` | the registry's static default (may depend on `dim`, e.g. `lambda0 = 10·d`) |
| `DERIVED` | computed from other settings by a registry rule (`derive_defaults`, e.g. `lambda_min`) |

`PROBLEM` refines D-D1's three categories; it makes clear when a value came
from an ioh problem rather than from the user.

**Rule: a `USER` value is never changed.** If it cannot work, resolution raises
an error that names the value and the module(s) involved.

### Resolution steps
1. **Names, types and options.** An unknown name is an error (suggest the closest valid name). Strings for categoricals are matched case-insensitively and stored UPPERCASE. Numbers outside `valid` are errors.
2. **Problem values.**
   - For an ioh problem: `dim`, `lb`, `ub` and the final target have `PROBLEM` provenance. A user value for `lb`/`ub`/`target` overrides it (`USER`).
   - For a plain callable: `dim`, `lb` and `ub` are required (D-B2). The front end wraps the callable with `ioh.wrap_problem` by default (D-B3).
3. **Defaults** (`DEFAULT`) for every unset setting.
4. **Derived values** (`DERIVED`): `lambda_min = min(lambda0, max(4, n_parents_needed + 1))` (never above `lambda0`, so a derived value cannot cause an error).
5. **Activity.** A `USER` value of a parameter that is inactive (its `active_when` conditions don't hold, e.g. `memory_size` without SHADE) produces a **warning** ("has no effect"), not an error. Numeric parameters are checked by the core; module options with conditions (`archive_content`, `pbest_sampling`) by the Python front end, which knows which ones the user set.
6. **Consistency checks** (errors):
   - `lambda0 ≥ n_parents_needed + 1` (distinct parents, D-D8);
   - with `population_reduction = LINEAR`: `n_parents_needed + 1 ≤ lambda_min ≤ lambda0`;
   - `budget ≥` the number of initial evaluations: `ceil(lambda0·(1 + oversampling_factor))` candidates, or the number of given initial individuals (0 if `initial_fitness` is given). Evaluations the problem already had are not part of this check (step 7): if fewer remain, the unevaluated candidates get `f = +∞`;
   - `lb < ub` element-wise, both finite and of length `dim`;
   - `initial_population`: shape `(n, dim)` with `n ≥ lambda0`, inside the bounds; `initial_fitness` of length `n`.
7. **Problem state.** If the problem already has evaluations, emit a **warning**, not an error (D-B3). Counting continues from the problem's total.
8. **Result.** An immutable resolved-settings object. Its `explain()` method lists every value with its provenance.

**Warnings for modules that have no effect** (the list grows with the interaction study, WS3 §5.3):
- `use_archive` with `mutation_n_comps = 0`;
- `pbest_*` without `mutation_reference = PBEST` (covered by step 5);
- `mutation_use_weighted_F` with `mutation_reference = NONE`.
- `mutation_n_comps = 0` with `mutation_reference = NONE` (Phase 4 interaction study, D-A6): the donors contain no difference vector. With `mutation_base = BEST` the population collapses onto the best point (the one *conflict* in the study); with `TARGET` every donor equals its target and nothing ever changes; with `RAND` the search only recombines existing points and is clearly worse than classic DE. With a reference term (current-to-X/0) it works well.
- `use_archive` with `archive_rate = 0` (capacity 0).
- Module options whose conditions do not hold (`archive_content` without `use_archive`, `pbest_sampling` without `PBEST`).

Not warned about, but documented: with `d = 1`, binomial and exponential crossover always take the donor coordinate, so CR (and CR adaptation) has no effect; with `mutation_n_comps = 0` and a `BEST` reference, the best individual's donor is itself.

### `n_parents_needed`
The number of distinct population members, besides the target, that the mutation must draw:

`n = 2·mutation_n_comps + [mutation_base = RAND] + [mutation_reference = RAND]`

The archive term's second vector counts as a population member. The archive
starts empty, so in the worst case it must come from `P`. `BEST` and `PBEST`
vectors don't need to be distinct.

## 2. Problems, evaluation and budget

- The objective is **minimised**. For ioh maximisation problems the front end minimises `−f` (D-B10); every value the user sees or passes (`best_f`, `fitness`, `trial_fitness`, `target`, `initial_fitness`) is in the problem's own sign. Complete core `Settings` passed directly are in the minimisation sign.
- **NaN objective values are treated as +∞**: they lose every comparison and sort last. The core counts them (`stats.nan_evaluations`) and the front end warns once. (Without this rule, a NaN parent is never replaced, and sorting with NaN is undefined.)
- **The budget is counted by the problem** (D-B3): termination compares `E = problem.state.evaluations` with `budget`, so `E ≤ budget` always holds.
- **Evaluation of a batch** (step 4): with `r = budget − E` evaluations remaining, at most `r` trial vectors are evaluated, in index order. The rest keep `f = +∞` and **never win selection**, even against a parent with `f = +∞` (the core tracks how many trials were evaluated), so they never count as successes (KI-8 fixed). *Subject to D-A21:* if exact capping turns out to be measurably slow, the whole batch is evaluated instead and the overshoot is documented.
- A batch is evaluated with one call when the objective is vectorised, otherwise one call per individual (D-A11). In per-evaluation tracking mode, it is always one call per individual (D-B4).

## 3. Termination

Checked after initialisation and after every evaluation batch. The run stops when any of these holds:
1. `E ≥ budget`;
2. the target is reached: ioh `state.final_target_found` (the front end checks it after every call of the problem and sets the core's `stop_requested`), or best `f ≤ target` for a user target;
3. `generation ≥ max_generations` (if set).

Trials that were already evaluated are still selected (D-D7). The remaining
steps of the generation (oppositional jump, `adapt()`) are skipped. The first
call to `step()` also performs initialisation. Within a batch, all points are
evaluated even if the target is reached part-way (one call when vectorised).
If an ioh problem has already found its target when a run starts, the run stops
at once (with a warning).

**Rounding.** Wherever the spec says `round` (archive capacity, p-best count,
population size schedule), the core rounds half away from zero (`llround`), as
L-SHADE's code does. The registry's Python rules agree for every value that occurs.

**Exact budget (D-A21, measured in Phase 2):** capping a batch at the remaining
budget is a single `min()` per batch and costs nothing measurable, so the budget
is always exact.

## 4. Statistics (v1 set, D-A20)

**Per generation:** `generation`, `evaluations` (E), global best `x`/`f`, `lambda` (current λ), `popmean` / `popstd` (mean and standard deviation over all coordinates of X pooled, as in 0.0.4; not a measure of diversity), and `F_memory_mean` (mean of the SHADE F memory, if active, else NaN). They are updated by `step()`; when the steps are called one by one, `update_stats()` refreshes them.

**Per evaluation** (only in tracking mode, which forces one evaluation call per individual, D-B4; otherwise `curr_F`/`curr_CR` are NaN):
- `curr_F`, `curr_CR`: the `F̃_i`, `CR̃_i` used for this trial;
- `corrected`: whether any coordinate of `u_i` was corrected; `corr_so_far`: cumulative count;
- `ED = ‖y(u'_i) − y(u_i)‖`: Euclidean distance between the corrected and uncorrected trial, in normalised space;
- `CS = cos∠(y(u'_i) − y(x_i), y(u_i) − y(x_i))`: cosine similarity between the corrected and uncorrected steps. It is defined as 0 when either vector is zero (KI-14 fixed: was NaN).

`ED` and `CS` are 0 for individuals that were not corrected. During initialisation and oppositional jumps, `curr_F` and `curr_CR` are those of the individual being evaluated (for opposite points: inherited from the source), and `corrected` is false.

Customisable statistics come after 1.0 (FR-1).
