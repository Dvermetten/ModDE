# Migrating from 0.0.4

!!! note "Work in progress"
    The full migration guide is written in Phase 5. This page lists the changes
    that are already fixed.

The 0.0.4 code remains available as `modde.legacy` (it raises a
`DeprecationWarning` and will be removed once the new API is stable):

```python
from modde.legacy import ModularDE   # the old implementation, unchanged
```

## Renamed settings

| 0.0.4 | new |
|---|---|
| `lambda_` | `lambda0` |
| `F`, `CR` (arrays) | `F0`, `CR0` (scalars; default CR changed from 0.5 to 0.9) |
| `lpsr=True` | `population_reduction="LINEAR"` (+ `lambda_min`) |
| `archive_size` (was always overwritten) | `archive_rate` |
| `pbest_value` | `pbest_sampling="FIXED"` + `pbest_rate` (or `"RANDOM"` + `pbest_rate_max`) |
| `oppositional_generation_probability` | `oppositional_generation=True` + `oppositional_jump_rate` |
| `oppositional_initialization=True` | `oppositional_initialization="OPPOSITE"` (or `"QUASI_OPPOSITE"`) |
| `base_sampler="LHC"` | `"LHS"` |
| `init_stats` | `track_evaluations` |
| lowercase options (`"shade"`, `"jDE"`) | UPPERCASE (`"SHADE"`, `"JDE"`); strings are matched case-insensitively |

## Other changes
- Populations are `(n, d)` arrays (0.0.4: `(d, n)`).
- Bounds are taken from the problem (0.0.4 always used [-5, 5]).
- The budget is counted by the problem and never exceeded.
- The default population size no longer depends on LPSR.
- All bugs and paper deviations listed in the development notes (`dev/KNOWN_ISSUES.md`) are fixed, so results differ from 0.0.4.
