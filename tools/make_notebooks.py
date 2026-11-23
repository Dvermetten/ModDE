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
    md("""
## Tuning with irace

For [irace](https://mlopez-ibanez.github.io/irace/), `modde.irace` writes the
parameter file (with all conditions) and an initial-configurations file with
the default and all presets. In the target runner, `config_from_irace` turns
irace's command-line arguments back into a configuration.
"""),
    code("""
import tempfile
paths = modde.irace.write_irace_files(tempfile.mkdtemp(), dim=5)
print("".join(paths["parameters"].read_text().splitlines(keepends=True)[:8]))

config = modde.irace.config_from_irace("--mutation_base TARGET --mutation_reference PBEST --pbest_sampling FIXED --pbest_rate 0.1", cs)
print(config["mutation_base"], config["pbest_rate"])
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


BENCHMARKING = [
    md("""
# Benchmarking with ioh

This notebook sets up a benchmark experiment: several DE variants on the
[BBOB](https://numbbo.github.io/coco/testsuites/bbob) functions of
[IOHexperimenter](https://iohprofiler.github.io/) (ioh), with output that
[IOHanalyzer](https://iohanalyzer.liacs.nl) can load, including modde's
internal state.

The experiment here is deliberately tiny so that the documentation builds
quickly. Increase the functions, dimensions and repetitions for real use.
"""),
    code("""
import tempfile, pathlib
import numpy as np
import ioh
import modde
"""),
    md("""
## A quick comparison in a loop

Any ioh problem can be passed to `modde.ModularDE`; bounds and the final target
come from the problem. Here, three variants on three BBOB functions, 5 runs
each, with the median precision $f - f^*$ reached:
"""),
    code("""
variants = {"DE": {}, "SHADE": {"preset": "SHADE"}, "L-SHADE": {"preset": "L-SHADE"}}
dim, budget, runs = 5, 5000, 5

print(f"{'function':<24}" + "".join(f"{name:>12}" for name in variants))
for fid in (1, 8, 15):
    row = []
    for name, settings in variants.items():
        precision = []
        for run in range(runs):
            problem = ioh.get_problem(fid, instance=run + 1, dimension=dim)
            modde.ModularDE(problem, budget=budget, seed=run, **settings).run()
            precision.append(problem.state.current_best.y - problem.optimum.y)
        row.append(np.median(precision))
    label = f"f{fid} {ioh.get_problem(fid, 1, dim).meta_data.name}"
    print(f"{label:<24}" + "".join(f"{v:12.2e}" for v in row))
"""),
    md("""
Runs stop when the final target ($f^* + 10^{-8}$) is reached, so precisions
below $10^{-8}$ mean the function was solved.

## A full experiment with `ioh.Experiment`

`ioh.Experiment` runs an algorithm on every combination of functions,
instances and dimensions, and writes IOHanalyzer files. It expects an object
that can be **called with a problem**. For modde, a small class does it. It also
forwards the optimizer's state (`curr_F`, `curr_CR`, `corrected`, ...), so the
experiment can log it next to the objective values:
"""),
    code("""
class DEAlgorithm:
    \"\"\"One modde run per problem; budget = budget_per_dim x dimension.\"\"\"

    def __init__(self, budget_per_dim=2000, **settings):
        self.budget_per_dim, self.settings, self.opt = budget_per_dim, settings, None

    def __call__(self, problem):
        budget = self.budget_per_dim * problem.meta_data.n_variables
        self.opt = modde.ModularDE(problem, budget=budget, track_evaluations=True, **self.settings)
        self.opt.run()

    def __getattr__(self, name):
        # tracked attributes of the current run (0 before the first run)
        if name.startswith("_"):
            raise AttributeError(name)
        opt = self.__dict__.get("opt")
        return getattr(opt, name) if opt is not None else 0.0
"""),
    md("""
`track_evaluations=True` makes the logged values belong to the individual just
evaluated (per-evaluation statistics). Now run two variants. Each gets its own
folder; together they form one IOHanalyzer data set:
"""),
    code("""
output = tempfile.mkdtemp()
for name, settings in {"DE": {}, "L-SHADE": {"preset": "L-SHADE"}}.items():
    experiment = ioh.Experiment(
        algorithm=DEAlgorithm(**settings),
        fids=[1, 8, 15], iids=[1, 2], dims=[5], reps=2,
        output_directory=output, folder_name=name, algorithm_name=name,
        logged_attributes=["curr_F", "curr_CR", "corrected"],
        zip_output=False,
    )
    experiment()

for path in sorted(pathlib.Path(output).rglob("*.json")):
    print(path.relative_to(output))
"""),
    md("""
The data files contain the logged attributes as extra columns:
"""),
    code("""
data_file = next(pathlib.Path(output, "L-SHADE").rglob("*.dat"))
print("\\n".join(data_file.read_text().splitlines()[:5]))
"""),
    md("""
Zip the output folders (or keep ioh's default `zip_output=True`) and upload them to
[IOHanalyzer](https://iohanalyzer.liacs.nl) to compare the variants.

Things to change for a real experiment:

- `fids=range(1, 25)`, several dimensions, more instances and repetitions;
- `njobs=-1` to use all cores;
- `logger_triggers`: by default ioh logs on every improvement. Logging every
  evaluation (`ioh.logger.trigger.ALWAYS`) shows the full parameter dynamics but
  produces a lot of data;
- `experiment_attributes` to store, for example, the module settings with the data.
"""),
]

TRACKING = [
    md("""
# Inside a run: tracking modde's state

modde exposes its complete state, so you can follow what an algorithm does,
generation by generation or even step by step. This notebook follows L-SHADE
on a rotated Rastrigin function.
"""),
    code("""
import numpy as np
import matplotlib.pyplot as plt
import ioh
import modde

problem = ioh.get_problem(15, instance=1, dimension=10)   # BBOB f15, rotated Rastrigin
"""),
    md("""
## Generation by generation

`step()` runs one generation and returns `False` when the run is over. Between
generations everything can be read: the population (an `(n, d)` array), its
fitness, the population size `lambda_`, the archive, the SHADE memories, ...
"""),
    code("""
opt = modde.ModularDE(problem, budget=50_000, preset="L-SHADE", seed=1)
rows = []
while opt.step():
    memory_CR = opt.memory_CR                      # NaN marks a terminal slot (CR = 0)
    rows.append((opt.evaluations, opt.best_f - problem.optimum.y, opt.lambda_, len(opt.archive),
                 np.mean(opt.memory_F), np.mean(np.nan_to_num(memory_CR)),
                 opt.population.std(axis=0).mean()))      # diversity: mean spread per coordinate
evals, precision, lam, archive, mF, mCR, spread = map(np.array, zip(*rows))
print(f"{len(rows)} generations, final precision {precision[-1]:.2e}")
"""),
    code("""
fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
ax[0, 0].semilogy(evals, np.maximum(precision, 1e-10))
ax[0, 0].set_title("best precision f - f*")
ax[0, 1].plot(evals, lam, label="population")
ax[0, 1].plot(evals, archive, label="archive")
ax[0, 1].set_title("sizes")
ax[0, 1].legend()
ax[1, 0].plot(evals, mF, label="F")
ax[1, 0].plot(evals, mCR, label="CR (terminal slots as 0)")
ax[1, 0].set_title("mean of the SHADE memories")
ax[1, 0].legend()
ax[1, 0].set_xlabel("evaluations")
ax[1, 1].semilogy(evals, spread)
ax[1, 1].set_title("diversity: mean std per coordinate")
ax[1, 1].set_xlabel("evaluations")
fig.tight_layout()
"""),
    md("""
The population shrinks linearly (L-SHADE's population size reduction) and the
archive with it. Diversity is computed here from the population array itself;
the logged statistic `popstd` is something else (the standard deviation over
all coordinates of all individuals pooled together, as in 0.0.4).

## Step by step

A generation consists of separate steps (see [Concepts](../concepts.md)).
Each can be called on its own, and the intermediate results inspected:
"""),
    code("""
problem.reset()              # the problem counts evaluations; start from zero again
opt = modde.ModularDE(problem, budget=5_000, preset="L-SHADE", seed=2)
opt.initialize()

opt.sample_parameters()      # F and CR per individual
opt.mutate()                 # donor vectors
opt.crossover()              # trial vectors
opt.correct_bounds()         # repair coordinates outside the bounds
print("F used (first 5):  ", np.round(opt.F_used[:5], 3))
print("CR used (first 5): ", np.round(opt.CR_used[:5], 3))
print("donors:", opt.donors.shape, "| trials:", opt.trials.shape)
print("share of coordinates taken from the donor:", np.mean(opt.trials == opt.donors).round(3))

opt.evaluate()
opt.select()
print("improved individuals:", len(opt.success["S"]), "of", opt.lambda_)
opt.adapt()                  # archive, memories, population size
print("archive size after adapt:", len(opt.archive))
"""),
    md("""
The state is also writable between steps: for example, you can replace the
population (`opt.population = ...`) or the F values used (`opt.F_used = ...`)
to try your own operators while modde does the rest.

## Logging the state with ioh

To store these values for every evaluation instead, attach an ioh logger that
**watches** the optimizer and set `track_evaluations=True`; see the
[Quickstart](01_quickstart.ipynb) and [Benchmarking](04_benchmarking.ipynb)
notebooks. The attributes that can be watched are `curr_F`, `curr_CR`,
`corrected`, `corr_so_far`, `CS`, `ED`, `popmean`, `popstd`, `F_memory_mean`
and `generation`.
"""),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cells in (("01_quickstart", QUICKSTART), ("02_configurations", CONFIGURATIONS),
                        ("03_ask_tell", ASK_TELL), ("04_benchmarking", BENCHMARKING),
                        ("05_tracking", TRACKING)):
        (OUT / f"{name}.ipynb").write_text(json.dumps(notebook(cells), indent=1) + "\n", encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
