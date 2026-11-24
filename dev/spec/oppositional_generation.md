# Oppositional generation (generation jumping)

Parameters: `oppositional_generation`, `oppositional_jump_rate` (D-D11).

**When:** after selection (step 5) and before `adapt()`. With
`oppositional_generation = True`, draw `ρ ~ U(0, 1)` once per generation and
jump if `ρ < oppositional_jump_rate` (default 0.3, as in ODE).

**Jump:**
1. Per dimension, take the bounds of the current population: `a_j = min_i x_ij` and `b_j = max_i x_ij` (KI-6 fixed: 0.0.4 took them per individual).
2. Opposite points `x̆_i = a + b − x_i` for all `i`. They lie inside `[a, b]`, the population's range, so they are never corrected. With `bound_correction = NONE` the population (and therefore `[a, b]`) can extend beyond `[lb, ub]`.
3. Evaluate them in index order, at most the remaining budget. Unevaluated points get `f = +∞`.
4. From `X ∪ X̆`, keep the `λ` points with the lowest `f`. On ties, the existing population member wins, then the lower index.
5. A kept opposite point `x̆_i` **inherits `F_i, CR_i`** from `x_i`.

**Effect on other modules:**
- The success record `R` was fixed at selection, so the SHADE memories and the archive are not affected by the jump (KI-7 fixed).
- Jumps are not successes: they do not enter `S`, the archive or jDE commits.
- Evaluations count towards the budget, and termination is checked after the jump's evaluations.
- Per-evaluation statistics: see settings.md §4.

## Random draws
One uniform per generation.
