"""Write the getting-started notebooks in docs/notebooks/ (without outputs).

The docs build executes them (mkdocs-jupyter, execute: true), so every example
is checked on every build. Edit the cells here and re-run this script, or edit
the notebooks directly in Jupyter.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "notebooks"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n")}


def notebook(cells):
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}}}


QUICKSTART = [
    md("""
# Quickstart

This notebook shows the main ways to run modde:

1. minimise a Python function with `fmin`;
2. run on benchmark problems from [IOHexperimenter](https://iohprofiler.github.io/) (ioh);
3. log a run, including modde's internal state, for IOHanalyzer;
4. step through a run yourself.
"""),
    code("""
import numpy as np
import ioh
import modde

print("modde", modde.__version__)
"""),
    md("""
## 1. Minimise a function

`fmin` needs the function, a budget (number of evaluations), the dimension and
the bounds. Without further settings it runs **classic DE** (DE/rand/1/bin).
"""),
    code("""
def sphere(x):
    return float(np.sum(x ** 2))

result = modde.fmin(sphere, budget=20_000, dim=10, lb=-5, ub=5, seed=42)
print("best value:", result.f)
print("evaluations:", result.evaluations)
"""),
    md("""
The result also unpacks as a tuple: `x, f, evaluations, optimizer = modde.fmin(...)`.

A well-known variant is one argument away (see the next notebook for all presets):
"""),
    code("""
result = modde.fmin(sphere, budget=20_000, dim=10, lb=-5, ub=5, seed=42, preset="L-SHADE")
print("L-SHADE best value:", result.f)
"""),
    md("""
## 2. Your function is wrapped as an ioh problem

modde wrapped `sphere` as an ioh problem behind the scenes:
"""),
    code("""
problem = result.optimizer.problem
print(type(problem).__name__, "| evaluations counted by the problem:", problem.state.evaluations)
"""),
    md("""
Because **the problem counts the evaluations**, the budget can never be exceeded
by accident. The bigger benefit is that everything ioh offers now works,
including logging. ioh's benchmark suites (BBOB, CEC2013, CEC2022, ...) can be
passed directly; bounds and the stopping target come from the problem.
"""),
    code("""
f15 = ioh.get_problem(15, instance=1, dimension=5)   # BBOB Rastrigin
opt = modde.ModularDE(f15, budget=20_000, preset="L-SHADE", seed=1).run()
print("precision reached:", opt.best_f - f15.optimum.y, "after", f15.state.evaluations, "evaluations")
"""),
    md("""
## 3. Logging a run, including internal state

To log, wrap your function yourself with `modde.wrap_problem` (or use an ioh
problem), attach an ioh logger and tell it which attributes of the optimizer to
**watch**. With `track_evaluations=True` the values are recorded for every evaluation.
The output folder can be loaded in [IOHanalyzer](https://iohanalyzer.liacs.nl).
"""),
    code("""
import tempfile, pathlib

problem = modde.wrap_problem(sphere, dim=5, lb=-5, ub=5, name="sphere")
folder = tempfile.mkdtemp()
logger = ioh.logger.Analyzer(root=folder, folder_name="modde-run", algorithm_name="L-SHADE")

opt = modde.ModularDE(problem, budget=2000, preset="L-SHADE", seed=3, track_evaluations=True)
logger.watch(opt, ["curr_F", "curr_CR", "corrected", "popstd"])
problem.attach_logger(logger)
opt.run()
problem.detach_logger()
logger.close()

data_file = next(pathlib.Path(folder).rglob("*.dat"))
print("\\n".join(data_file.read_text().splitlines()[:6]))
"""),
    md("""
## 4. Stepping through a run

`step()` runs one generation and returns `False` when the run is finished. In
between, the whole state can be inspected: `population` (an `(n, d)` array),
`fitness`, the `archive`, the SHADE `memory_F`, the population size `lambda_`, and so on.
"""),
    code("""
opt = modde.ModularDE(sphere, budget=6000, dim=5, lb=-5, ub=5, preset="L-SHADE", seed=4)
history = []
while opt.step():
    history.append((opt.evaluations, opt.lambda_, opt.best_f, opt.F_memory_mean))

for evals, lam, best, mean_F in history[::40]:
    print(f"{evals:6d} evaluations | population {lam:3d} | best {best:9.2e} | mean F memory {mean_F:.3f}")
"""),
    md("""
Each generation is made of steps that can also be called one by one
(`sample_parameters`, `mutate`, `crossover`, `correct_bounds`, `evaluate`,
`select`, `adapt`). See [Concepts](../concepts.md).
"""),
]

CONFIGURATIONS = [
    md("""
# Configurations, presets and tuning

modde is configured with named **settings**: modules (e.g. `mutation_base`) and
numeric parameters (e.g. `lambda0`). This notebook shows how to set them,
inspect them, use named presets, and work with the configuration space for
automated algorithm configuration. All settings are listed in [Parameters](../parameters.md).
"""),
    code("""
import numpy as np
import ioh
import modde

def sphere(x):
    return float(np.sum(x ** 2))
"""),
    md("""
## Settings by name

Module options are strings (case-insensitive).
"""),
    code("""
opt = modde.ModularDE(sphere, budget=3000, dim=5, lb=-5, ub=5,
                      mutation_base="target", mutation_reference="pbest",
                      use_archive=True, adaptation_method_F="SHADE", lambda0=40)
s = opt.settings
print("lambda0:", s.lambda0, s.provenance["lambda0"])
print("CR0:", s.CR0, s.provenance["CR0"])
print("archive_rate:", s.archive_rate, s.provenance["archive_rate"])
"""),
    md("""
Values you set are never changed. Conflicts raise an error, and settings that
have no effect with the chosen modules produce a warning:
"""),
    code("""
try:
    modde.ModularDE(sphere, budget=100, dim=5, lb=-5, ub=5, mutation_n_comps=3, lambda0=6)
except ValueError as error:
    print("error:", error)

import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    modde.ModularDE(sphere, budget=100, dim=5, lb=-5, ub=5, memory_size=10)
print("warning:", caught[0].message)
"""),
    md("""
## Presets

Popular DE variants are available by name. `exact=False` marks approximations
(components of the original paper that are not available as modules).
"""),
    code("""
print(modde.preset_names())
print(modde.preset_settings("L-SHADE", dim=10))
"""),
    md("""
## The configuration space

`get_configspace(dim)` returns a [ConfigSpace](https://automl.github.io/ConfigSpace/)
`ConfigurationSpace` with every module and tunable parameter. Module-specific
parameters are *conditional*: they only appear when their module is active.
Its default configuration is classic DE, and every configuration in it is valid.
"""),
    code("""
cs = modde.get_configspace(dim=5)
print(len(list(cs.values())), "hyperparameters,", len(cs.conditions), "conditions")
print(cs.get_default_configuration())
"""),
    md("""
Presets are configurations of this space, so they can be edited and used as
starting points for a tuner:
"""),
    code("""
config = modde.get_presets(dim=5)["L-SHADE"]
values = dict(config)
values["archive_rate"] = 1.5
config = type(config)(cs, values=values)
res = modde.fmin(sphere, budget=3000, dim=5, lb=-5, ub=5, config=config, seed=1)
print("edited L-SHADE:", res.f)
"""),
    md("""
## A tiny tuning loop

Any tuner that speaks ConfigSpace (e.g. [SMAC](https://github.com/automl/SMAC3))
can tune modde directly. Here, plain random search on a BBOB function shows the
workflow; `settings_from_config` turns a configuration into settings.
"""),
    code("""
def evaluate(config, seed=0):
    problem = ioh.get_problem(15, instance=1, dimension=5)
    opt = modde.ModularDE(problem, budget=2000, config=config, seed=seed).run()
    return opt.best_f - problem.optimum.y

cs.seed(1)
candidates = [cs.get_default_configuration()] + list(cs.sample_configuration(8))
scores = [np.mean([evaluate(c, seed) for seed in range(3)]) for c in candidates]
best = candidates[int(np.argmin(scores))]
print("default (classic DE):", round(scores[0], 3))
print("best of the sample:  ", round(min(scores), 3))
print({k: v for k, v in dict(best).items() if k in ("mutation_base", "crossover", "adaptation_method_F", "lambda0")})
"""),
]

ASK_TELL = [
    md("""
# Ask-tell

With the ask-tell interface you evaluate the points yourself: in parallel, on a
cluster, or from a simulation. modde only proposes points and receives values.
For the same seed it follows exactly the same trajectory as `run()`.
"""),
    code("""
import numpy as np
import modde

def sphere(x):
    return float(np.sum(x ** 2))
"""),
    md("""
## A generation at a time

`ask()` returns an `(n, d)` array of points; `tell()` takes their values in the
same order. The loop ends when `stop` becomes true.
"""),
    code("""
at = modde.AskTellDE(dim=5, lb=-5, ub=5, budget=3000, preset="SHADE", seed=1)
while not at.stop:
    X = at.ask()
    values = [sphere(x) for x in X]      # e.g. evaluate in parallel here
    at.tell(values)
print("best:", at.best_f, "after", at.evaluations, "evaluations")
"""),
    md("""
## One point at a time

`ask_one()` returns an index and a point. Values are matched by index, so they
may be told in any order.
"""),
    code("""
at = modde.AskTellDE(dim=3, lb=-5, ub=5, budget=500, seed=2)
while not at.stop:
    index, x = at.ask_one()
    at.tell_one(index, sphere(x))
print("best:", at.best_f)
"""),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cells in (("01_quickstart", QUICKSTART), ("02_configurations", CONFIGURATIONS),
                        ("03_ask_tell", ASK_TELL)):
        (OUT / f"{name}.ipynb").write_text(json.dumps(notebook(cells), indent=1) + "\n", encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
