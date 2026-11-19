# modde module specification (Phase 1)

This folder specifies **exactly what each module does**, independent of any
implementation. The C++ core (Phase 2) is implemented against it, and the
operator-level tests (WS3 §5.1) are derived from it.

- **Parameters** (names, types, defaults, ranges, when active, tuning ranges) are
  defined once in the registry [`src/modde/_registry.py`](../../src/modde/_registry.py).
  These documents refer to parameters by name and do not repeat their table.
- **Paper versions** are used throughout (D-D3). Known 0.0.4 deviations are
  referenced as `KI-n` ([KNOWN_ISSUES.md](../KNOWN_ISSUES.md)).

| Document | Covers |
|---|---|
| [settings.md](settings.md) | Settings, resolution and provenance, problems and budget, termination, statistics |
| [initialization.md](initialization.md) | Samplers, (quasi-)oppositional initialisation, oversampling, custom populations |
| [mutation.md](mutation.md) | Base vector, reference term, difference vectors, archive term, p-best, weighted F, parent selection |
| [crossover.md](crossover.md) | Binomial, exponential, eigenvector crossover |
| [bound_correction.md](bound_correction.md) | All bound-correction methods |
| [selection_archive.md](selection_archive.md) | One-to-one selection, success records, archive maintenance |
| [adaptation.md](adaptation.md) | F/CR per generation: constant, SHADE, SHADE_MODIFIED, LSHADE, jDE, jSO caps |
| [population_size.md](population_size.md) | Population size schedules (NONE, LINEAR) |
| [oppositional_generation.md](oppositional_generation.md) | Generation jumping |
| [presets.md](presets.md) | Named configurations (D-D2) |

## Notation

| Symbol | Meaning |
|---|---|
| `d` | dimension |
| `λ` | current population size (`λ0` initially) |
| `X = (x_1 … x_λ)` | population; `f_i = f(x_i)` |
| `F_i, CR_i` | per-individual stored parameters (persist between generations) |
| `F̃_i, CR̃_i` | parameters *used* in the current generation (see adaptation.md) |
| `v_i, u_i` | donor (after mutation) and trial (after crossover) vectors |
| `lb, ub` | bounds; `c = (lb + ub)/2` the centre |
| `y = (x − lb)/(ub − lb)` | normalised coordinates (element-wise) |
| `t = min(1, E / budget)` | progress, `E` = evaluations counted by the problem (D-B3) |
| `U(a, b)`, `N(μ, σ)`, `Cauchy(μ, γ)` | uniform, normal, Cauchy distributions |
| `U{0 … k−1}` | uniform integer |

All randomness comes from the optimizer's single per-instance RNG (D-A13).
Each spec states *which* random draws an operator makes. The order of draws
inside an operator is an implementation detail, but it must be fixed so that
runs are reproducible on the same platform.

## One generation

```
0. sample_parameters()   F̃, CR̃ for this generation              adaptation.md
1. mutate()              V from X, F̃ (+ archive)                 mutation.md
2. crossover()           U from X, V, CR̃                         crossover.md
3. correct_bounds()      U' from U (+ out-of-bounds mask)         bound_correction.md
4. evaluate()            f(U'), at most the remaining budget      settings.md
5. select()              X ← winners; success record R            selection_archive.md
6. [oppositional jump]   with probability jump_rate               oppositional_generation.md
7. adapt()               archive ← R; memories ← R; jDE commit;   selection_archive.md,
                         population size schedule                 adaptation.md, population_size.md
8. check termination                                              settings.md
```

Differences from 0.0.4 in the *structure*:
- Evaluation is a step of its own (KI-45).
- F/CR for a generation are sampled at its start (step 0), not at the end of the previous one. This is equivalent for SHADE, and required for jDE's keep-on-success rule (KI-22).
- The success record `R` (successful F̃/CR̃, fitness gains, replaced parents) is fixed at selection time. Later steps that reorder the population (oppositional jump, population reduction) therefore cannot misalign it (fixes KI-7).

## Open points from writing the spec

Collected here for the author's review. Each is also marked ⚠️ in its document.

1. **Renamed / restructured settings** compared with 0.0.4 (migration guide):
   - `lpsr` (bool) → `population_reduction ∈ {NONE, LINEAR}` (room for non-linear schedules in Workstream 4);
   - `oppositional_generation_probability` → `oppositional_generation` (bool) + `oppositional_jump_rate` (D-D11);
   - `pbest_value` → `pbest_sampling ∈ {RANDOM, FIXED}` + `pbest_rate` / `pbest_rate_max`. The 0.0.4 "None = SHADE random p" behaviour becomes an explicit, tunable choice;
   - `base_sampler='LHC'` → `LHS`; all options UPPERCASE (D-A6).
2. **Constants that stay fixed** (not settings), because they define the module itself: the weighted-F schedule (0.7/0.8/1.2 at t = 0.2/0.4) and the jSO cap thresholds. All other KI-34 constants became settings.
3. **`F0`/`CR0` also initialise the SHADE memories**, so the paper initial memory values (0.5; jSO 0.3/0.8) are set through presets. Consequence: SHADE with the classic defaults starts its CR memory at 0.9, not 0.5.
4. **Eigenvector crossover (Guo & Yang 2015):** 0.0.4 applies it in every generation using the covariance of the whole population. As far as I recall, the paper applies it with a probability and possibly on a subset of the population. *To verify*; if confirmed, it is a further paper deviation to decide on.
5. **References for the bound corrections** (COTN, HVB, EXPC_*, EXPS) are to be filled in by the author.
6. **Preset details** in [presets.md](presets.md) are from memory of the papers and need checking. JADE-like and jSO-like are marked as approximations: keep them, or drop them?
