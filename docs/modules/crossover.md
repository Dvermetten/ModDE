# Crossover

Crossover mixes the donor $v_i$ with the target $x_i$ into the **trial vector**
$u_i$. The crossover rate $CR_i$ controls how much comes from the donor.

## How it works

### Binomial (`BIN`, default)

Each coordinate comes from the donor with probability $CR_i$; one random
coordinate $j_{\text{rand}}$ always does:

$$
u_{ij} = \begin{cases} v_{ij} & \text{if } \rho_{ij} \le CR_i \text{ or } j = j_{\text{rand}},\\ x_{ij} & \text{otherwise,}\end{cases}
\qquad \rho_{ij} \sim U(0, 1).
$$

### Exponential (`EXP`)

A block of $L$ cyclically consecutive coordinates, starting at a random
position, comes from the donor. $L$ starts at 1 and grows while
$\rho < CR_i$, so $P(L \ge l) = CR_i^{\,l-1}$ (at most $d$).

### Eigenvector crossover (`eigenvalue_crossover`)

Crossover happens in the eigenbasis of the population covariance, which makes
it rotation-invariant. With $C = \operatorname{cov}(X) = B \Lambda B^\top$:

1. express target and donor in the eigenbasis: $x'_i = B^\top x_i$, $v'_i = B^\top v_i$;
2. apply binomial or exponential crossover to $x'_i, v'_i$;
3. rotate back: $u_i = B u'_i$.

With $\lambda \le d$ the covariance is singular; the decomposition still works.

## Example

```python
import numpy as np
import modde

def ellipsoid(x):
    return float(np.sum(10 ** (6 * np.arange(len(x)) / (len(x) - 1)) * x ** 2))

x, f, *_ = modde.fmin(ellipsoid, budget=5000, dim=10, lb=-5, ub=5, eigenvalue_crossover=True)
```

## Options and parameters

<!-- modde:params crossover -->

## Interactions

<!-- modde:interactions crossover -->

## Changes from 0.0.4

- Eigenvector crossover rotates into the eigenbasis ($B^\top x$). 0.0.4 used $B x$, a different basis.
