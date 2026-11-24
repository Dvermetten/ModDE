# Initialisation

The initial population is created once, before the first generation. By
default, `lambda0` points are drawn uniformly from the search space and
evaluated.

## How it works

1. **Number of candidates.** $m = \lceil \lambda_0 \cdot (1 + \text{oversampling\_factor}) \rceil$;
   without oversampling $m = \lambda_0$.
2. **Base points.** With opposition, $m_b = \lceil m/2 \rceil$ base points are
   sampled, otherwise $m_b = m$. The sampler draws $z \in [0,1]^d$, which is
   mapped to $x = \mathrm{lb} + z \odot (\mathrm{ub} - \mathrm{lb})$.
3. **Opposite points** (optional): one (quasi-)opposite point is added per base
   point; if that gives one candidate too many, the last is dropped.
4. **Selection.** All candidates are evaluated (each evaluation counts towards
   the budget), and the $\lambda_0$ best are kept. Ties go to the earlier candidate.

Every individual starts with $F_i = $ `F0` and $CR_i = $ `CR0`; SHADE memories
are filled with these values, and the archive starts empty.

### Samplers (`base_sampler`)

| Option | Points in $[0,1]^d$ |
|---|---|
| `UNIFORM` | independent $U(0, 1)$ coordinates |
| `GAUSSIAN` | normal with mean 0.5 and standard deviation 1/6 per coordinate, redrawn until inside $[0, 1]$ (truncated normal) |
| `SOBOL` | scrambled Sobol sequence (Joe & Kuo direction numbers, random digital shift); at most 1111 dimensions |
| `HALTON` | scrambled Halton sequence (random digit permutation per dimension) |
| `LHS` | Latin hypercube: each dimension is split into $m_b$ strata, one point per stratum, strata permuted at random |

All samplers are seeded from the optimizer's random number generator, so a seed
fixes the whole run. Sobol points have their best balance properties when
$m_b$ is a power of two.

### Opposite points (`oppositional_initialization`)

With centre $c = (\mathrm{lb} + \mathrm{ub})/2$, the opposite of $x$ is
$\breve{x} = \mathrm{lb} + \mathrm{ub} - x$. With
$\varphi = $ `oppositional_initialization_factor`:

| Option | Point added for base point $x$ |
|---|---|
| `NONE` | none |
| `OPPOSITE` | $c + \varphi(\breve{x} - c)$; $\varphi = 1$ is classic opposition (ODE), $\varphi = 0$ the centre |
| `QUASI_OPPOSITE` | per coordinate $c_j + w_j \varphi (\breve{x}_j - c_j)$ with $w_j \sim U(0,1)$: uniform between the centre and the opposite point (QODE for $\varphi = 1$) |

### Your own initial population

`initial_population` (an $n \times d$ array, $n \ge \lambda_0$) replaces sampling.
The $\lambda_0$ best of its rows are kept. Pass `initial_fitness` as well to skip
evaluating them, for example to continue from an earlier run. Sampling, opposition
and oversampling settings then have no effect (you get a warning if you set them).

## Example

```python
import numpy as np
import modde

def sphere(x):
    return float(np.sum(x ** 2))

x, f, *_ = modde.fmin(sphere, budget=5000, dim=10, lb=-5, ub=5,
                      base_sampler="SOBOL", oppositional_initialization="QUASI_OPPOSITE")
```

## Options and parameters

<!-- modde:params initialization -->

## Interactions

<!-- modde:interactions initialization -->

## Changes from 0.0.4

- The `GAUSSIAN` sampler is truncated to $[0, 1]$; 0.0.4 did not restrict it, so some initial points lay outside the bounds.
- `oversampling_factor` works: 0.0.4 sampled the extra points but evaluated and kept only the first $\lambda_0$ (which also dropped the opposite points).
- `oppositional_initialization_factor` is used (0.0.4 declared it but ignored it), and `QUASI_OPPOSITE` is new.
- Sobol and Halton are modde's own implementations, so the points differ from scipy's; `LHC` is renamed `LHS`.
