# Selection and archive

## Selection

DE uses one-to-one selection: each trial vector competes only with its own
target, and replaces it if it is at least as good (minimisation):

$$
x_i \leftarrow u_i \quad\text{if } f(u_i) \le f(x_i).
$$

Ties go to the trial, which helps on plateaus. At selection, modde records the
**strict improvements** $S = \{i : f(u_i) < f(x_i)\}$ together with their
$F_i$, $CR_i$ and improvement $\Delta f_i$. SHADE-type adaptation learns from
this record, and the archive is filled from it. (jDE keeps a new F/CR whenever
the trial is selected, ties included.)

## Archive (`use_archive`)

The external archive of JADE stores solutions that recently left the
population. It adds diversity to the first difference vector of the
[mutation](mutation.md): its second vector is drawn from population and archive
together.

- The archive starts empty.
- After every generation, $|S|$ vectors are added (see `archive_content` below).
- Its capacity is $\operatorname{round}(\text{archive\_rate} \cdot \lambda)$ with the
  *current* population size $\lambda$. When the archive is over capacity (after
  insertion, or because the population shrank), random members are removed until it fits.

### What enters the archive (`archive_content`)

| Option | Added for each improved individual $i \in S$ | Source |
|---|---|---|
| `PARENTS` (default) | the replaced parent $x_i$ | JADE, SHADE and L-SHADE papers |
| `TRIALS` | the successful trial vector $u_i$ | the L-SHADE authors' code |

The published CEC 2014 results of L-SHADE were produced with code that archives
the trial vectors. That code converges faster in the middle of a run. modde's
`L-SHADE` preset follows the paper; add `archive_content="TRIALS"` to reproduce
the code. Both are validated against independent implementations
(see *Validation* in the development notes).

## Example

```python
import numpy as np
import modde

def rastrigin(x):
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))

# L-SHADE as published in the CEC 2014 code
x, f, *_ = modde.fmin(rastrigin, budget=20000, dim=10, lb=-5, ub=5,
                      preset="L-SHADE", archive_content="TRIALS")
```

## Options and parameters

<!-- modde:params archive -->

## Interactions

<!-- modde:interactions archive -->

## Changes from 0.0.4

- The archive starts empty (0.0.4 started with a copy of the initial population).
- `archive_rate` sets the capacity relative to the current population size; 0.0.4's `archive_size` was always overwritten with $2\lambda$.
- `archive_content` is new.
