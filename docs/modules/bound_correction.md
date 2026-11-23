# Bound correction

Mutation can produce trial vectors outside the bounds. Bound correction
repairs the violating coordinates before evaluation; coordinates inside the
bounds are never changed. The choice matters more than one might think: some
methods bias the search towards the bounds or the centre (structural bias).

## How it works

Work in normalised coordinates $y = (u - \mathrm{lb}) / (\mathrm{ub} - \mathrm{lb})$.
For a coordinate with $y \notin [0, 1]$, let $b = 1$ if the upper bound is
violated and $b = 0$ for the lower bound, and let $\tau$ be the normalised
coordinate of the target (parent) $x_i$. The corrected value $y'$ is:

| Option | $y'$ |
|---|---|
| `SATURATE` (default) | $b$: the coordinate is set to the bound |
| `NONE` | $y$: infeasible points are evaluated as they are |
| `UNIF_RESAMPLE` | $U(0, 1)$ |
| `COTN` | $\lvert b - \lvert z\rvert \rvert$ with $z \sim N(0, 1/3)$, redrawn until inside $[0, 1]$ |
| `TOROIDAL` | $y - \lfloor y \rfloor$ (wrap around) |
| `MIRROR` | reflection at the bounds, repeated until inside |
| `HVB` | $\alpha b + (1 - \alpha)\tau$ with $\alpha = $ `hvb_alpha`; $\alpha = 0.5$ is the midpoint between bound and parent used by SHADE and L-SHADE |
| `EXPC_TARGET` | $b + (1 - 2b)\,\delta$, $\delta$ exponential truncated to $[0, \lvert b - \tau\rvert]$: between the bound and the parent, concentrated near the bound |
| `EXPC_CENTER` | as `EXPC_TARGET`, truncated to $[0, 0.5]$ (towards the centre) |
| `EXPS` | as `EXPC_TARGET`, truncated to $[0, 1]$ (the whole domain) |

The truncated exponential is drawn as $\delta = -\ln\!\left(1 - \rho\,(1 - e^{-a})\right)$,
$\rho \sim U(0, 1)$. The exponential methods work in normalised coordinates,
so their concentration is relative to the domain width.

The statistics `corrected` (any coordinate of the current trial repaired) and
`corr_so_far` count the corrections; they can be logged with ioh.

## Example

```python
import numpy as np
import modde

def sphere(x):
    return float(np.sum((x - 4.9) ** 2))   # optimum close to the upper bound

x, f, *_ = modde.fmin(sphere, budget=5000, dim=10, lb=-5, ub=5, bound_correction="MIRROR")
```

## Options and parameters

<!-- modde:params bound_correction -->

## Interactions

<!-- modde:interactions bound_correction -->

## Changes from 0.0.4

- `COTN` redraws until the result is inside the bounds (0.0.4 could return out-of-bounds values).
- The HVB weight is a parameter (`hvb_alpha`; 0.0.4 fixed it at 0.5).
