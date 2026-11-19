# Initialisation

Parameters: `base_sampler`, `oppositional_initialization`,
`oppositional_initialization_factor`, `oversampling_factor`,
`initial_population`, `initial_fitness`, `lambda0`, `F0`, `CR0`.

## 1. Candidates

Let `m = ceil(λ0 · (1 + oversampling_factor))` be the number of candidates.
Without oversampling, `m = λ0`.

**With `initial_population` given** (`n × d`, `n ≥ λ0`), these `n` points are
the candidates. Sampling, opposition and oversampling are not used; a `USER`
value for any of them produces a warning (settings.md §1, step 5).
`initial_fitness`, if given, is used as-is and those points are not evaluated.

**Otherwise:**
1. Let `m_b = ceil(m / 2)` if `oppositional_initialization ≠ NONE`, else `m_b = m`.
2. Draw `m_b` base points `z ∈ [0, 1]^d` with the sampler and map them to `x = lb + z ⊙ (ub − lb)`.
3. With opposition, append one (quasi-)opposite point per base point (§3), then drop the last one if `2·m_b > m`. The candidates are base points first, then opposites.

## 2. Samplers (`base_sampler`)

| Option | Draws for `m_b` points in `[0,1]^d` |
|---|---|
| `UNIFORM` | `z_ij ~ U(0, 1)` |
| `GAUSSIAN` | `z_ij ~ N(0.5, 1/6)`, **redrawn until `z_ij ∈ [0, 1]`** (truncated normal; KI-9 fixed) |
| `SOBOL` | scrambled Sobol sequence, seeded from the optimizer RNG, first `m_b` points |
| `HALTON` | scrambled Halton sequence, seeded from the optimizer RNG, first `m_b` points |
| `LHS` | Latin hypercube: for each dimension, a random permutation `π_j` of `0…m_b−1`, and `z_ij = (π_j(i) + U(0,1)) / m_b` |

- Sobol/Halton follow the definitions of the sequences. Scrambling method and direction numbers (e.g. reusing modcma's MIT-licensed C++ Sobol/Halton, D-E6) are an implementation choice. Results will differ from scipy (0.0.4).
- The power-of-two note for Sobol balance properties is documented, not warned about.

## 3. (Quasi-)opposite points (`oppositional_initialization`, KI-37)

For a base point `x`, with `c = (lb + ub)/2`, opposite `x̆ = lb + ub − x` and `φ = oppositional_initialization_factor`:

| Option | Point added |
|---|---|
| `NONE` | none |
| `OPPOSITE` | `x° = c + φ·(x̆ − c)`. `φ = 1`: classic opposition (ODE). `φ = 0`: the centre. |
| `QUASI_OPPOSITE` | `x°_j = c_j + w_j·φ·(x̆_j − c_j)` with `w_j ~ U(0, 1)`, i.e. uniform between the centre and the scaled opposite point. `φ = 1`: QODE. |

The points stay inside the bounds for any `φ ∈ [0, 1]`.

## 4. Evaluation and selection of the initial population

1. Evaluate the candidates in order, at most the remaining budget (resolution guarantees `budget ≥ λ0`). Unevaluated candidates are dropped.
2. Keep the `λ0` candidates with the lowest `f`. Ties are broken by candidate index (lower first).
3. With oversampling and/or opposition, every evaluated candidate counts towards the budget (KI-4 fixed: 0.0.4 evaluated only `λ0`).

## 5. Initial state
- `F_i = F0` and `CR_i = CR0` for every individual.
- SHADE memories (if active) are filled with `M_F = F0` and `M_CR = CR0` in all `memory_size` slots (see README open point 3).
- The archive is empty (KI-26).
- `generation = 0`.

## Random draws
Sampler draws (`m_b · d` values, plus scrambling and permutations); with
`QUASI_OPPOSITE`, `m_b · d` weights `w_j`; with `GAUSSIAN`, redraws as needed.
