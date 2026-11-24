# Migrating from 0.0.4

modde 1.0 is a rewrite: the algorithm runs in C++, settings are validated, and
the modules follow their papers. Code written for 0.0.4 needs small changes,
and its results will differ (see [Behaviour changes](#behaviour-changes)).

## Keep the old code running

The 0.0.4 implementation is still included, unchanged, as `modde.legacy`. It
raises a `DeprecationWarning` and will be removed in a later release:

```python
from modde.legacy import ModularDE   # the 0.0.4 implementation
```

## The same run, old and new

=== "0.0.4"

    ```python
    import ioh
    from modde.legacy import ModularDE

    f = ioh.get_problem(23, 1, 5)
    lshade = ModularDE(f, mutation_base="target", mutation_reference="pbest",
                       lpsr=True, lambda_=18 * 5, memory_size=6, use_archive=True,
                       adaptation_method_F="shade", adaptation_method_CR="shade")
    lshade.run()
    ```

=== "1.0"

    ```python
    import ioh
    import modde

    f = ioh.get_problem(23, 1, 5)
    opt = modde.ModularDE(f, budget=50_000, preset="L-SHADE").run()
    print(opt.best_f)
    ```

The main differences:

- **The budget** is an argument (0.0.4: `budget` setting, default $10^4 d$; the
  default is the same, but passing it explicitly is clearer). It is counted by
  the problem and can never be exceeded.
- **Presets** replace long lists of settings. `preset="L-SHADE"` is the
  paper-exact L-SHADE; the [Presets](presets.md) page shows what each one sets.
- **Plain functions** work too: `modde.fmin(f, budget, dim=..., lb=..., ub=...)`
  or `modde.ModularDE(f, budget=..., dim=..., lb=..., ub=...)`.
- **Internal state** is available directly: `opt.curr_F` instead of
  `opt.parameters.stats.curr_F`, `opt.population` (an `(n, d)` array) instead of
  `opt.parameters.population` (`(d, n)`). An ioh logger can watch the optimizer
  directly (`logger.watch(opt, ["curr_F"])`); only `ioh.Experiment`, which
  creates a new run per problem, still needs a small adapter class (shown in the
  [benchmarking tutorial](notebooks/04_benchmarking.ipynb)).
- **Ask-tell:** `modde.AskTellDE` asks and tells whole generations (or single
  points with `ask_one` / `tell_one`); see the [ask-tell notebook](notebooks/03_ask_tell.ipynb).

## Renamed settings

| 0.0.4 | 1.0 |
|---|---|
| `lambda_` | `lambda0` |
| `F`, `CR` (arrays) | `F0`, `CR0` (scalars; the CR default changed from 0.5 to 0.9) |
| `lpsr=True` | `population_reduction="LINEAR"` (and `lambda_min`) |
| `archive_size` (was always overwritten) | `archive_rate` (capacity relative to the population size) |
| `pbest_value=None` | `pbest_sampling="RANDOM"` with `pbest_rate_max` |
| `pbest_value=p` | **different behaviour:** 0.0.4 drew $p_i \sim U(0, p)$ per individual. JADE's fixed p is `pbest_sampling="FIXED"` with `pbest_rate=p`; the closest to the old behaviour is `pbest_sampling="RANDOM"` with `pbest_rate_max=p` (lower end $2/\lambda$) |
| `oppositional_generation_probability=p` | `oppositional_generation=True` with `oppositional_jump_rate=p` |
| `oppositional_initialization=True` | `oppositional_initialization="OPPOSITE"` (or `"QUASI_OPPOSITE"`) |
| `base_sampler="LHC"` | `base_sampler="LHS"` |
| `init_stats=True` | `track_evaluations=True` |
| `inialize_custom_pop` (sic) / `initialize_custom_population()` | `initial_population` (and optionally `initial_fitness`) |
| `memory_size` (default 100) | `memory_size` (default 6) |
| lowercase options (`"shade"`, `"jDE"`) | UPPERCASE (`"SHADE"`, `"JDE"`); strings are matched case-insensitively, so old spellings still work |

New settings that were hard-coded constants in 0.0.4: `hvb_alpha`,
`shade_scale_F`, `shade_scale_CR`, `jde_tau_F`, `jde_F_lower`, `jde_F_width`,
`jde_tau_CR`, `oppositional_initialization_factor` (declared but unused in 0.0.4).
New modules: `archive_content`, `adaptation_method_CR="LSHADE"`,
`oppositional_initialization="QUASI_OPPOSITE"`.

Unknown or misspelled settings now raise an error (old names get a hint to the
new one), and a setting that has no effect with the chosen modules gives a warning.

## Behaviour changes

Results never match 0.0.4: the random number generator is different, and the
following bugs and deviations from the papers were fixed. Each module page has
a *Changes from 0.0.4* section with details.

**Problem and budget**

- The bounds are taken from the problem (0.0.4 ignored the problem's bounds and used `lb`/`ub` settings that defaulted to $[-5, 5]$).
- The initial population counts towards the budget, so budget-based schedules (population reduction, jSO limits, weighted F) end exactly at the budget.
- The default population size is $10d$ for every configuration (0.0.4 switched to $18d$ when `lpsr` was on).

**Mutation** ([details](modules/mutation.md#changes-from-004))

- `mutation_n_comps=2` uses two different difference vectors (0.0.4 added the same one twice).
- Random parents are distinct from each other.
- `pbest_value` was not JADE's fixed $p$; `pbest_rate` is.
- Weighted F scales only the p-best term; the archive term draws from population and archive together, as in JADE.

**Adaptation** ([details](modules/adaptation.md#changes-from-004))

- jDE: new values were drawn 90 % of the time instead of 10 %, and kept regardless of success.
- SHADE's F memory uses the weighted Lehmer mean; jSO's limits are lower bounds on CR.

**Other modules**

- Archive: starts empty, and its size setting is no longer overwritten.
- Eigenvector crossover rotates into the eigenbasis.
- Oppositional generation computes the population range per dimension and no longer misaligns F, CR and the success record.
- Oversampling and `oppositional_initialization_factor` work; the Gaussian sampler stays inside the bounds; COTN never returns out-of-bounds values.

## Reproducibility

A seed gives the same result with the same modde version on the same machine.
See [Reproducibility](reproducibility.md).
