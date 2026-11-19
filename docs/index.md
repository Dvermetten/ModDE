# modde: Modular Differential Evolution

modde builds Differential Evolution (DE) variants from interchangeable
**modules**: initialisation, mutation, crossover, bound correction, parameter
adaptation, population size control and more. Classic DE, SHADE, L-SHADE, jDE
and many combinations no paper has tried are all configurations of the same
algorithm. The computation runs in C++; the interface is Python.

!!! warning "Development version"
    This documentation describes the C++-backed rewrite (1.0 development
    series). The 0.0.4 implementation is still available as `modde.legacy`, but
    it is deprecated.

## Installation

```bash
pip install modde
```

Building from source needs a C++17 compiler and the Eigen submodule:

```bash
git clone --recursive https://github.com/Dvermetten/ModDE.git
cd ModDE && pip install .
```

## Quickstart

```python
import numpy as np
import modde

def sphere(x):
    return float(np.sum(x ** 2))

x, f, evaluations, optimizer = modde.fmin(sphere, budget=5000, dim=10, lb=-5, ub=5)
```

Without further settings this runs **classic DE** (DE/rand/1/bin). Popular
variants are one argument away:

```python
modde.fmin(sphere, budget=5000, dim=10, lb=-5, ub=5, preset="L-SHADE")
```

!!! tip "Your function becomes an ioh problem"
    A plain function is wrapped with [IOHexperimenter](https://iohprofiler.github.io/)
    automatically. The problem counts the evaluations, so the budget can never be
    exceeded, and attaching an ioh logger gives you IOHanalyzer-ready output,
    including modde's internal state. See the [Quickstart notebook](notebooks/01_quickstart.ipynb).

## Where to go next

- [**Quickstart**](notebooks/01_quickstart.ipynb): `fmin`, ioh problems, logging, stepping through a run.
- [**Configurations, presets and tuning**](notebooks/02_configurations.ipynb): settings, named variants, the configuration space.
- [**Ask-tell**](notebooks/03_ask_tell.ipynb): evaluate points yourself, e.g. in parallel.
- [**Parameters**](parameters.md): every module and parameter, with tuning ranges and references.

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

The design follows the [Modular CMA-ES](https://github.com/IOHprofiler/ModularCMAES) package.
