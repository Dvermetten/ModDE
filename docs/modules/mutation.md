# Mutation

Mutation creates one **donor vector** $v_i$ per individual from a base vector,
an optional "-to-X" term and a number of difference vectors. The default is
DE/rand/1: $v_i = x_{r_0} + F_i (x_{r_1} - x_{r_2})$.

## How it works

With $n = $ `mutation_n_comps`:

$$
v_i = b_i
\;+\; F^{\text{ref}}_i \,(x^{\text{ref}}_i - x_i)
\;+\; F_i\,(x_{r_1} - \tilde{x}_{r_2})
\;+\; \sum_{k=2}^{n} F_i\,(x_{r_{2k-1}} - x_{r_{2k}})
$$

- **Base vector** $b_i$ (`mutation_base`): `RAND` $x_{r_0}$, `BEST` the best
  individual, `TARGET` $x_i$ itself.
- **Reference term** (`mutation_reference`, left out for `NONE`):
  $x^{\text{ref}}_i$ is a random member (`RAND`), the best (`BEST`) or a
  p-best individual (`PBEST`). Together with `mutation_base="TARGET"` this gives
  the "current-to-X" strategies, e.g. JADE's current-to-pbest/1.
- **Difference vectors:** $n \ge 0$ terms, each with its own pair of random
  members. With $n = 0$ there are none (useful only together with a reference
  term).
- **Archive:** with `use_archive=True`, the second vector of the *first*
  difference term, $\tilde{x}_{r_2}$, is drawn from the population and the
  archive together ($P \cup A$), as in JADE. See
  [Selection and archive](selection_archive.md).

All random members of one donor ($r_0, r_1, \dots$) are distinct and differ from
$i$. The best and p-best individuals need not be.

### p-best (`mutation_reference="PBEST"`)

The population is sorted by $f$, and $x^{\text{ref}}_i$ is drawn uniformly from
the $\max(1, \operatorname{round}(p_i\lambda))$ best. `pbest_sampling` sets $p_i$:

- `RANDOM` (SHADE): $p_i \sim U(2/\lambda,\ \text{pbest\_rate\_max})$, drawn per individual;
- `FIXED` (JADE, L-SHADE): $p_i = $ `pbest_rate`.

### Weighted F (`mutation_use_weighted_F`, jSO)

Only the reference term uses $F^{\text{ref}}_i = w(t) F_i$, with $w = 0.7$ while
$t < 0.2$, $0.8$ while $t < 0.4$, and $1.2$ afterwards. Without it,
$F^{\text{ref}}_i = F_i$.

## Example

DE/current-to-pbest/1 with archive (the mutation of JADE, SHADE and L-SHADE):

```python
import numpy as np
import modde

def sphere(x):
    return float(np.sum(x ** 2))

x, f, *_ = modde.fmin(sphere, budget=5000, dim=10, lb=-5, ub=5,
                      mutation_base="TARGET", mutation_reference="PBEST",
                      pbest_sampling="FIXED", pbest_rate=0.11, use_archive=True)
```

## Options and parameters

<!-- modde:params mutation -->

## Interactions

<!-- modde:interactions mutation -->

## Changes from 0.0.4

- `mutation_n_comps=2` uses two different pairs of parents (0.0.4 added the same difference twice, so DE/rand/2 was DE/rand/1 with $2F$).
- Random parents are distinct from each other, not only from the target.
- `pbest_rate` is a fixed $p$ as in JADE (0.0.4's `pbest_value` drew $p_i \sim U(0, p)$).
- Weighted F applies to the reference term only, as in jSO (0.0.4 scaled every term).
- The archive term draws from $P \cup A$, as in JADE (0.0.4 drew from the archive only).
