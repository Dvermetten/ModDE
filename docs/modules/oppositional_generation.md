# Oppositional generation

Generation jumping from opposition-based DE (ODE): now and then, the opposite
points of the whole population are evaluated, and the best $\lambda$ of
population and opposites survive.

## How it works

After selection, with probability `oppositional_jump_rate` (default 0.3, as in
ODE):

1. Take the per-dimension range of the current population,
   $a_j = \min_i x_{ij}$ and $b_j = \max_i x_{ij}$.
2. Compute the opposite points $\breve{x}_i = a + b - x_i$. They lie inside the
   population's range, so they are not bound-corrected (with
   `bound_correction="NONE"` that range can extend beyond the bounds).
3. Evaluate them (they count towards the budget).
4. Keep the $\lambda$ best of $X \cup \breve{X}$; on ties the current member wins.
   A surviving opposite point inherits $F_i$ and $CR_i$ of the point it mirrors.

A jump is not a success for adaptation: SHADE memories and the archive are
updated from the selection before the jump only.

## Example

```python
import numpy as np
import modde

def rastrigin(x):
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))

x, f, *_ = modde.fmin(rastrigin, budget=10000, dim=10, lb=-5, ub=5, preset="ODE")
```

## Options and parameters

<!-- modde:params oppositional_generation -->

## Interactions

<!-- modde:interactions oppositional_generation -->

## Changes from 0.0.4

- The population range is taken per dimension (0.0.4 took it per individual).
- Reordering the population no longer misaligns F, CR and the success record used by SHADE and the archive.
