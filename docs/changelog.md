# Changelog

## 1.0.0 (unreleased)

A rewrite of modde with a C++ core. See [Migrating from 0.0.4](migration.md).

**Core**

- The algorithm runs in C++17 (Eigen, nanobind bindings): 15–33× faster than
  0.0.4 for L-SHADE, 1.5–3× for classic DE (whose runtime is dominated by the
  objective function).
- Settings are resolved once: defaults are filled in, values are validated, and
  each value records where it came from (user, problem, default, derived). A
  value you set is never changed; conflicts are errors and settings without
  effect give warnings.
- Exact budget: the problem counts evaluations, including the initial
  population, and the budget is never exceeded.
- Seeded runs are reproducible with the same build on the same machine.
- Robust to awkward objective values: NaN counts as +inf (with a warning), and
  infinite values (penalties) do not disturb parameter adaptation.
- The run stops right after the batch that reaches the target, including an ioh
  problem's own target.

**Modules**

- Every module follows its paper; all bugs and deviations of 0.0.4 are fixed
  (listed in the migration guide).
- New: L-SHADE CR adaptation (`adaptation_method_CR="LSHADE"`, Lehmer mean and
  terminal value), quasi-oppositional initialisation, `archive_content`
  (archive the replaced parents or the successful trials), modde's own Sobol and
  Halton samplers.
- Constants that were hard-coded are parameters (`hvb_alpha`, SHADE scales,
  jDE rates and ranges, `lambda_min`, `pbest_rate`, ...).

**Interface**

- `modde.fmin` and `modde.ModularDE` for plain functions and ioh problems; plain
  functions are wrapped as ioh problems, so ioh logging works for them too,
  including modde's internal state (per evaluation with `track_evaluations=True`).
- Named presets (`preset="L-SHADE"`): DE variants, jDE, ODE, SHADE, L-SHADE,
  and approximations of JADE and jSO.
- Configuration space for tuning (`modde.get_configspace`), with conversion
  between configurations and settings, and an irace exporter.
- Ask-tell interface (`modde.AskTellDE`), following exactly the same trajectory
  as `run()` for the same seed.
- The full state (population, donors, trials, archive, memories) is readable and,
  between steps, writable (with shape checks); a generation can be run step by step.
- Maximisation problems: all values (`best_f`, `fitness`, `target`,
  `initial_fitness`) are in the problem's own sign.
- Clear errors: unknown settings (with hints for 0.0.4 names), wrong types,
  conflicting arguments (`settings` together with other settings, `preset`
  together with `config`); warnings for settings that have no effect.
- `opt.settings_dict(user_only=True)` gives the settings that reproduce a run.

**Documentation and validation**

- Documentation site with tutorials, a module reference, the parameter
  reference and the results of the module interaction study.
- Validated against independent implementations of DE, jDE, SHADE and L-SHADE,
  and against the published CEC 2014 results of L-SHADE (see [Validation](validation.md)).

**Removed**

- The 0.0.4 implementation is kept as `modde.legacy` (deprecated); the top-level
  names now refer to the new API.

## 0.0.4

Pure-Python implementation, available as `modde.legacy`.
