# Population size

The population starts with `lambda0` individuals (default $10d$). The default is
the same for every configuration: no module changes it, so a preset or your
own value sets it explicitly.

## How it works

### Constant (`NONE`, default)

$\lambda = $ `lambda0` throughout.

### Linear reduction (`LINEAR`, L-SHADE's LPSR)

After each generation, with progress $t$ (evaluations used / budget):

$$
\lambda_{\text{next}} = \operatorname{round}\big(\lambda_0 + (\lambda_{\min} - \lambda_0)\, t\big).
$$

If $\lambda_{\text{next}} < \lambda$, the worst individuals are removed (with
their own F and CR), and the archive is trimmed to its new capacity. The
schedule depends on the budget, so pass the real budget of your experiment:
with a larger budget, the population shrinks more slowly. If the run ends
earlier (target reached, `max_generations`), the population has not reached
`lambda_min` yet.

`lambda_min` defaults to the smallest population the mutation can work with
(at least 4, but never more than `lambda0`): every donor needs enough distinct
random members.

## Example

```python
import numpy as np
import modde

def sphere(x):
    return float(np.sum(x ** 2))

x, f, evaluations, opt = modde.fmin(sphere, budget=10000, dim=10, lb=-5, ub=5,
                                    lambda0=180, population_reduction="LINEAR", lambda_min=4)
print(opt.lambda_)   # the final population size
```

## Options and parameters

<!-- modde:params population_size -->

## Interactions

<!-- modde:interactions population_size -->

## Changes from 0.0.4

- The default population size no longer changes to $18d$ when population reduction is on; set `lambda0` yourself (the `L-SHADE` preset does).
- The initial population counts towards the budget, so the schedule ends exactly at the budget.
- `lambda_min` is a parameter.
