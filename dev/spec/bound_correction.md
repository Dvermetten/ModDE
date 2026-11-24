# Bound correction

Parameters: `bound_correction`, `hvb_alpha`.

Bound correction works per coordinate, in normalised space
`y = (u − lb)/(ub − lb)`. Only coordinates with `y_ij ∉ [0, 1]` are changed;
all other coordinates are left untouched. For a violating coordinate, let
`b = 1` if `y_ij > 1` (upper bound violated) and `b = 0` if `y_ij < 0`. Let
`τ_ij = (x_ij − lb_j)/(ub_j − lb_j)` be the normalised coordinate of the target
(parent) `x_i`. After correction, `u'_ij = lb_j + y'_ij (ub_j − lb_j)`.

| Option | Corrected value `y'` | Draws |
|---|---|---|
| `NONE` | `y` (unchanged; infeasible points are evaluated as-is, D-B7) | – |
| `SATURATE` (default) | `b` | – |
| `UNIF_RESAMPLE` | `U(0, 1)` | 1 |
| `COTN` | `\|b − \|z\|\|` with `z ~ N(0, 1/3)` (standard deviation 1/3), **redrawn until the result is in `[0, 1]`** (KI-17 fixed) | ≥ 1 |
| `TOROIDAL` | `y − ⌊y⌋` | – |
| `MIRROR` | `\|y − ⌊y⌋ − (⌊y⌋ mod 2)\|` (repeated reflection) | – |
| `HVB` | `α·b + (1 − α)·τ_ij`, with `α = hvb_alpha`. `α = 0.5`: midpoint between bound and parent, as in SHADE/L-SHADE | – |
| `EXPC_TARGET` | `b + (1 − 2b)·δ`, with `δ` drawn from Exp(1) truncated to `[0, a]` and `a = \|b − τ_ij\|` | 1 |
| `EXPC_CENTER` | as `EXPC_TARGET` with `a = 0.5` (towards the centre) | 1 |
| `EXPS` | as `EXPC_TARGET` with `a = 1` (whole domain) | 1 |

**Truncated exponential:** `δ = −ln(1 − ρ·(1 − e^(−a)))` with `ρ ~ U(0, 1)`. The
corrected point lies between the violated bound and the reference point (the
parent, the centre or the opposite bound), concentrated near the bound. This
is the closed form of 0.0.4's expression `|2b − (b − ln(1 + ρ(e^(−a) − 1)))|`;
the two are equivalent. If `a = 0` (the parent lies on the violated bound),
then `y' = b`.

**Outputs** (for statistics): the corrected trial `u'_i`, the uncorrected `u_i`,
and a per-individual flag "any coordinate corrected".

## References

The ModDE paper (Vermetten et al., GECCO 2023, §3.4) cites Biedrzycki et al.
2019 and Kononova et al. 2022 for the bound corrections as a group. The
per-option attributions below follow Kononova et al. 2022 and Padhye et al.
2015. Full entries: `REF` / `OPTION_REFERENCES` in the registry.

| Option | Full name | Reference |
|---|---|---|
| `SATURATE` | saturation / projection | Caraffini, Kononova & Corne 2019 |
| `UNIF_RESAMPLE` | uniform resampling | Vermetten et al. 2022c |
| `COTN` | complete one-sided truncated normal (stochastic counterpart of mirror) | Caraffini, Kononova & Corne 2019 |
| `TOROIDAL` | toroidal / wrapping | Caraffini, Kononova & Corne 2019 |
| `MIRROR` | mirror / reflection | Kononova et al. 2020a, 2020b |
| `HVB` | halfway-to-violated-bound (= SHADE's midpoint-target rule; component-wise projection to the midpoint) | Kononova et al. 2022; Tanabe & Fukunaga 2013; Biedrzycki et al. 2019 |
| `EXPC_TARGET` | exponentially confined (Exp-C) | Alvarez-Benitez, Everson & Fieldsend 2005; Padhye, Mittal & Deb 2015 |
| `EXPS` | exponential spread (Exp-S) | Padhye, Mittal & Deb 2015 |
| `EXPC_CENTER` | exponentially confined towards the centre | modde variant (no external source) |

**Notes:**
- `hvb_alpha` generalises HVB; `α = 0.5` is the literature version.
- Padhye et al. define Exp-C/Exp-S with density `∝ exp(|x − x_parent|)` in the original variable units. modde applies them in **normalised** coordinates (rate 1 per unit of `[0, 1]`), as 0.0.4 did. The shape is the same (mass concentrated near the violated bound); the concentration is relative to the domain width.
