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
| `COTN` | `\|b − \|z\|\|` with `z ~ N(0, 1/3)`, **redrawn until the result is in `[0, 1]`** (KI-17 fixed) | ≥ 1 |
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

⚠️ README open point 5: literature references for COTN, HVB, EXPC_* and EXPS (author).
