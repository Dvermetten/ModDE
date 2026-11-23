# Presets

Presets are well-known DE variants expressed as module settings. Use one by
name, and override any of its settings:

```python
import numpy as np
import modde

def sphere(x):
    return float(np.sum(x ** 2))

x, f, *_ = modde.fmin(sphere, budget=10000, dim=10, lb=-5, ub=5, preset="L-SHADE")
x, f, *_ = modde.fmin(sphere, budget=10000, dim=10, lb=-5, ub=5, preset="L-SHADE", memory_size=10)
```

Presets are also available as configurations of the
[configuration space](notebooks/02_configurations.ipynb), for example to
warm-start a tuner: `modde.get_presets(dim)["L-SHADE"]`.

Settings that depend on the dimension (such as L-SHADE's $\lambda_0 = 18d$)
are computed for the problem's dimension. Below, they are shown for $d = 10$
and $d = 30$.

Presets marked **approximation** lack mechanisms of the original algorithm
that modde has no module for; their notes say which. `DE`, `jDE`, `SHADE` and
`L-SHADE` have been validated against independent implementations of the papers
(see [Validation](validation.md)).

<!-- modde:presets -->
