# F and CR control

The scale factor $F$ and crossover rate $CR$ can be constant or adapted during
the run. F and CR are controlled **independently**: any F method can be
combined with any CR method (for example SHADE for F and jDE for CR).

Every generation, the values *used* by each individual, $F_i$ and $CR_i$, are
drawn first (step 0). After selection, the method learns from the successful
individuals (step 7).

## How it works

### Constant (`NONE`, default)

$F_i = $ `F0` and $CR_i = $ `CR0` throughout.

### SHADE family

Success-history memories $M_F$ and $M_{CR}$ with $H = $ `memory_size` slots,
initialised with `F0` and `CR0`. Per individual a slot $r_i$ is drawn
uniformly, and

- $CR_i = \operatorname{clip}_{[0,1]}\big(N(M_{CR}[r_i],\ \text{shade\_scale\_CR})\big)$;
- $F_i \sim \operatorname{Cauchy}(M_F[r_i],\ \text{shade\_scale\_F})$, redrawn while
  $F_i \le 0$, then truncated to $F_i \le 1$. `SHADE_MODIFIED` uses the mean of
  all memory slots as location instead of slot $r_i$.

After selection, if there were improvements, one memory slot $k$ (cycling
through the memory) is overwritten with weighted means of the successful values,
with weights $w_j = \Delta f_j / \sum \Delta f$:

$$
M_F[k] = \frac{\sum w_j F_j^2}{\sum w_j F_j} \quad\text{(Lehmer mean)},\qquad
M_{CR}[k] = \begin{cases}\sum w_j CR_j & \texttt{SHADE}\\[2pt] \dfrac{\sum w_j CR_j^2}{\sum w_j CR_j} & \texttt{LSHADE}\end{cases}
$$

`LSHADE` adds L-SHADE's **terminal value**: if all successful CRs of a slot
update are 0, the slot becomes terminal and gives $CR_i = 0$ for the rest of the
run.

### jDE

Each individual keeps its own $F$ and $CR$. Each generation, with probability
`jde_tau_F`, a new $F_i = $ `jde_F_lower` $+\ U(0,1) \cdot$ `jde_F_width` is
tried (otherwise the stored value is used), and likewise, with probability
`jde_tau_CR`, a new $CR_i \sim U(0, 1)$. A new value is **kept only if the trial
vector is selected**.

### jSO limits (`use_jso_caps`)

Applied to the values drawn by any method: $CR_i \ge 0.7$ while $t < 0.25$,
$CR_i \ge 0.6$ while $t < 0.5$, and $F_i \le 0.7$ while $t < 0.6$. The limited
values are the ones used and learned from.

## Example

SHADE adaptation for F, jDE for CR:

```python
import numpy as np
import modde

def sphere(x):
    return float(np.sum(x ** 2))

x, f, *_ = modde.fmin(sphere, budget=5000, dim=10, lb=-5, ub=5,
                      adaptation_method_F="SHADE", adaptation_method_CR="JDE", memory_size=10)
```

With an ioh logger, `curr_F` and `curr_CR` (the values used for the individual
just evaluated) and `F_memory_mean` can be logged; see the
[tracking tutorial](../notebooks/05_tracking.ipynb).

## Options and parameters

<!-- modde:params adaptation -->

## Interactions

<!-- modde:interactions adaptation -->

## Changes from 0.0.4

- jDE draws new values with probability $\tau$ (0.0.4 inverted this, drawing new values 90 % of the time) and keeps them only after a successful trial.
- SHADE updates the F memory with the weighted **Lehmer** mean, as in the paper (0.0.4 used the arithmetic mean).
- The jSO limits are lower bounds on CR, as in jSO (0.0.4 capped CR from above).
- `LSHADE` (Lehmer mean for CR plus the terminal value) is new; it makes the `L-SHADE` preset paper-exact.
- The constants that 0.0.4 hard-coded (jDE $\tau$ and F range, SHADE scales) are parameters.
