# modde: Modular Differential Evolution

modde builds Differential Evolution (DE) variants from interchangeable
**modules**: initialisation, mutation, crossover, bound correction, parameter
adaptation, population size control and more. Classic DE, JADE, SHADE,
L-SHADE, jDE and thousands of combinations no paper has tried are all
configurations of the same algorithm.

- **Fast:** the algorithm runs in C++; the interface is Python.
- **Faithful:** every module follows its paper, and the presets are validated against independent implementations and published results.
- **Configurable:** every module and parameter is part of a configuration space for algorithm configuration (ConfigSpace, irace).
- **Benchmark-ready:** built on [IOHexperimenter](https://iohprofiler.github.io/): any function becomes an ioh problem, so logging for IOHanalyzer, including modde's internal state, works out of the box.

**Documentation:** https://dvermetten.github.io/ModDE/

## Installation

```bash
pip install modde
```

Python 3.10 or newer. Wheels are available for Windows and Linux; elsewhere, pip
builds from source with a C++17 compiler (see the
[installation guide](https://dvermetten.github.io/ModDE/installation/)).

## Quickstart

```python
import numpy as np
import modde

def sphere(x):
    return float(np.sum(x ** 2))

result = modde.fmin(sphere, budget=10_000, dim=10, lb=-5, ub=5)
print(result.f, result.x)
```

Without further settings this runs classic DE/rand/1/bin. Your function is
wrapped as an ioh problem automatically: the problem counts the budget, and an
ioh logger can be attached to record the run
([Quickstart](https://dvermetten.github.io/ModDE/notebooks/01_quickstart/)).

Well-known variants are available by name, here on a BBOB problem from ioh:

```python
import ioh
import modde

problem = ioh.get_problem(15, instance=1, dimension=10)     # BBOB f15, Rastrigin
opt = modde.ModularDE(problem, budget=100_000, preset="L-SHADE", seed=1).run()
print(opt.best_f - problem.optimum.y)
```

Or choose the modules yourself:

```python
import ioh
import modde

problem = ioh.get_problem(15, instance=1, dimension=10)
opt = modde.ModularDE(problem, budget=100_000, mutation_base="TARGET", mutation_reference="PBEST",
                      use_archive=True, adaptation_method_F="SHADE", adaptation_method_CR="SHADE",
                      bound_correction="MIRROR").run()
```

## Modules

| Module | Options |
|---|---|
| [Initialisation](https://dvermetten.github.io/ModDE/modules/initialization/) | uniform, Gaussian, Sobol, Halton, LHS; (quasi-)opposition; oversampling |
| [Mutation](https://dvermetten.github.io/ModDE/modules/mutation/) | rand / best / target base; '-to-rand/best/pbest' term; any number of difference vectors; weighted F |
| [Crossover](https://dvermetten.github.io/ModDE/modules/crossover/) | binomial, exponential; eigenvector crossover |
| [Bound correction](https://dvermetten.github.io/ModDE/modules/bound_correction/) | saturation, resampling, COTN, toroidal, mirror, midpoint (HVB), exponential variants |
| [Selection and archive](https://dvermetten.github.io/ModDE/modules/selection_archive/) | external archive of parents or successful trials |
| [F and CR control](https://dvermetten.github.io/ModDE/modules/adaptation/) | constant, SHADE, L-SHADE, jDE; jSO limits |
| [Population size](https://dvermetten.github.io/ModDE/modules/population_size/) | constant, linear reduction |
| [Oppositional generation](https://dvermetten.github.io/ModDE/modules/oppositional_generation/) | generation jumping |

Presets: `DE`, `DE/best/1/bin`, `DE/rand/2/bin`, `DE/current-to-best/1/bin`,
`jDE`, `ODE`, `SHADE`, `L-SHADE`, `JADE-like`, `jSO-like`
([details](https://dvermetten.github.io/ModDE/presets/)).

## Upgrading from 0.0.4

Version 1.0 is a rewrite with a new interface; the old implementation remains
available as `modde.legacy` for now. See the
[migration guide](https://dvermetten.github.io/ModDE/migration/).

## Citation

If you use modde, please cite:

```bibtex
@inproceedings{vermetten2023modular,
  author    = {Diederick Vermetten and Fabio Caraffini and Anna V. Kononova and Thomas B{\"{a}}ck},
  title     = {Modular Differential Evolution},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference, {GECCO} 2023},
  pages     = {864--872},
  publisher = {{ACM}},
  year      = {2023},
  doi       = {10.1145/3583131.3590417}
}
```

The design of this package follows the
[Modular CMA-ES](https://github.com/IOHprofiler/ModularCMAES) package by Jacob de Nobel.

## Licence

MIT; see [LICENCE](LICENCE). modde uses [Eigen](https://eigen.tuxfamily.org)
(MPL 2.0), [nanobind](https://github.com/wjakob/nanobind) (BSD) and the Sobol
direction numbers of Joe & Kuo (BSD-style; licence in `src/modde/_licenses/`).
