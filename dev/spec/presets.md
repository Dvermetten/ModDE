# Presets (named configurations, D-D2)

**Source of truth:** `PRESETS` in [`src/modde/_registry.py`](../../src/modde/_registry.py).
Each preset is a function `dim → settings` (only the values that differ from the
defaults). In Phase 3, `get_presets(dim)` will expose them as ConfigSpace
`Configuration`s; the prototype is `_preset_configuration`. A test checks that
every preset is a valid configuration for d ∈ {1, 2, 5, 20, 100}.

The default (no settings) is **classic DE**: DE/rand/1/bin, `F0 = 0.5`,
`CR0 = 0.9`, `λ0 = 10·d`, uniform initialisation, SATURATE bounds.

## Table (values shown for d = 10; generated from the registry)

| Preset | Paper-exact | Settings (d = 10) | Reference |
|---|---|---|---|
| DE | yes | (all defaults) | Storn & Price 1997 |
| DE/best/1/bin | yes | `mutation_base=BEST` | Storn & Price 1997 |
| DE/rand/2/bin | yes | `mutation_n_comps=2` | Storn & Price 1997 |
| DE/current-to-best/1/bin | yes | `mutation_base=TARGET`, `mutation_reference=BEST` | Storn & Price 1997 |
| jDE | yes | `lambda0=100`, `adaptation_method_F=JDE`, `adaptation_method_CR=JDE` | Brest et al. 2006 |
| ODE | yes | `lambda0=100`, `oppositional_initialization=OPPOSITE`, `oppositional_generation=True` | Rahnamayan et al. 2008 |
| SHADE | yes | `lambda0=100`, `mutation_base=TARGET`, `mutation_reference=PBEST`, `pbest_sampling=RANDOM`, `use_archive=True`, `archive_rate=1.0`, `adaptation_method_F=SHADE`, `adaptation_method_CR=SHADE`, `memory_size=100`, `F0=0.5`, `CR0=0.5`, `bound_correction=HVB` | Tanabe & Fukunaga 2013 |
| L-SHADE | yes | `lambda0=180` (18·d), `population_reduction=LINEAR`, `lambda_min=4`, `mutation_base=TARGET`, `mutation_reference=PBEST`, `pbest_sampling=FIXED`, `pbest_rate=0.11`, `use_archive=True`, `archive_rate=2.6`, `adaptation_method_F=SHADE`, `adaptation_method_CR=LSHADE`, `memory_size=6`, `F0=0.5`, `CR0=0.5`, `bound_correction=HVB` | Tanabe & Fukunaga 2014 |
| JADE-like | **approximate** | `lambda0=100`, current-to-pbest/1, `pbest_sampling=FIXED`, `pbest_rate=0.05`, archive rate 1.0, SHADE F/CR with `memory_size=1`, `F0=CR0=0.5`, HVB | Zhang & Sanderson 2009 |
| jSO-like | **approximate** | `lambda0=182` (25·ln(d)·√d), LINEAR to 4, current-to-pbest-w/1 (`mutation_use_weighted_F`), `pbest_rate=0.17`, archive rate 1.0, SHADE F + LSHADE CR, `memory_size=5`, `F0=0.3`, `CR0=0.8`, `use_jso_caps`, HVB | Brest et al. 2017 |

## Notes and what is missing for the approximate presets
- **JADE-like:** JADE updates `μ_F` (Lehmer mean) and `μ_CR` (arithmetic mean) by exponential smoothing with `c = 0.1`. A 1-slot SHADE memory *replaces* the value instead of smoothing it.
- **jSO-like:** jSO decreases `p` linearly from 0.25 to 0.125 (approximated by the mean, 0.17) and keeps one memory slot fixed at 0.9. Neither exists as a module. `lambda0` is at least 7 for small d.
- **jDE:** the paper doesn't specify bound handling, so the default (SATURATE) is used.
- **SHADE / L-SHADE bound handling:** the midpoint between the violated bound and the parent, which is `HVB` with `hvb_alpha = 0.5`.

## ⚠️ To verify against the papers (from memory)
- SHADE: N = 100, H = 100, archive size = N, `p_i ~ U(2/N, 0.2)`, initial memories 0.5.
- L-SHADE: `N_init = 18·d`, `N_min = 4`, H = 6, archive rate 2.6, p = 0.11, initial memories 0.5, CR memory with Lehmer mean and terminal value.
- jDE: N = 100, F0 = 0.5, CR0 = 0.9, τ1 = τ2 = 0.1, F_l = 0.1, F_u = 0.9.
- ODE: N = 100, F = 0.5, CR = 0.9, jumping rate 0.3, opposite initialisation.
- Bound handling of SHADE / L-SHADE (midpoint rule).
- Decide whether to keep the approximate presets (README open point 6).
