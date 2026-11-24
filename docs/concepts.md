# Concepts

## Modules, settings and the optimizer

- **Modules** are the algorithm's building blocks, chosen by option name, for
  example `mutation_base="TARGET"` or `bound_correction="MIRROR"`.
- **Settings** are the numbers: population size `lambda0`, initial `F0` and
  `CR0`, archive rate, and so on. Many belong to a module and only have an
  effect when that module is active.
- Both are passed by name; [Parameters](parameters.md) lists them all.

There are three ways to configure a run, and they combine as follows:

| You pass | Result |
|---|---|
| keyword settings only | classic DE with your changes |
| `preset="L-SHADE"` + keyword settings | the preset, with your settings on top |
| `config=...` (a ConfigSpace configuration) + keyword settings | the configuration, with your settings on top |
| `settings=...` (complete core settings) | exactly these settings; nothing else may be passed |

A preset and a configuration cannot be combined (a configuration is complete
and would replace the preset); use `modde.get_presets(dim)[name]` to get a preset
as a configuration you can edit.

When an optimizer is created, the settings are **resolved**: defaults are
filled in, derived values are computed, and everything is validated. Each
value records where it came from (`USER`, `PROBLEM`, `DEFAULT` or `DERIVED`):

```python
import numpy as np
import modde

def sphere(x):
    return float(np.sum(x ** 2))

opt = modde.ModularDE(sphere, budget=1000, dim=5, lb=-5, ub=5, lambda0=30)
print(opt.settings.provenance["lambda0"])   # Provenance.USER
print(opt.settings.provenance["CR0"])       # Provenance.DEFAULT
```

**A value you set is never changed.** If it cannot work with the chosen modules
(say, a population too small for the mutation), you get an error saying why. If
it has no effect (say, `memory_size` without SHADE adaptation), you get a
`UserWarning`. Values that only a preset set do not cause such warnings.

### Seeing the full settings of a run

```python
import numpy as np
import modde

def sphere(x):
    return float(np.sum(x ** 2))

opt = modde.ModularDE(sphere, budget=1000, dim=5, lb=-5, ub=5, preset="SHADE")
everything = modde.settings_to_dict(opt.settings)                # every value, incl. defaults and the seed
mine = modde.settings_to_dict(opt.settings, user_only=True)      # what was set explicitly, plus the seed
print(mine)
print(opt.settings.warnings)                                     # the warnings issued at creation

again = modde.ModularDE(sphere, dim=5, **mine)                     # the same run again (incl. bounds, budget, seed)
```

`opt.settings` returns a copy: changing it has no effect on the optimizer.

## One generation

```
0. sample_parameters   F and CR for this generation (constant, SHADE, jDE, ...)
1. mutate              donor vectors
2. crossover           trial vectors
3. correct_bounds      repair coordinates outside the bounds
4. evaluate            at most the remaining budget
5. select              one-to-one selection
6. [oppositional jump] optional
7. adapt               archive, parameter memories, population size
```

Each step can be called on its own (`opt.mutate()`, ...), and the state can be
inspected between steps: `opt.population`, `opt.fitness`, `opt.donors`,
`opt.trials` (before bound correction), `opt.trials_corrected` (the points that
are evaluated), `opt.archive`, `opt.memory_F`, ... Populations are `(n, d)`
arrays. Writable state (`population`, `fitness`, `F`, `CR`, `F_used`, `CR_used`,
`donors`, `trials`, `trial_fitness`, `memory_F`, `memory_CR`) can be assigned
between steps; shapes are checked. `step()` also refreshes the population
statistics; after calling the steps yourself, call `opt.update_stats()`.

## Problems and budget

`modde` works on [ioh](https://iohprofiler.github.io/) problems. A plain
function is wrapped as one automatically (`wrap=False` to opt out). The problem
counts the evaluations, and the budget applies to that count. If the problem
already has evaluations when a run starts, modde warns and counts them too.

- **Maximisation** problems (ioh `optimization_type=MAX`) are handled by
  minimising $-f$ internally. Everything you see or pass is in the problem's own
  sign: `best_f`, `fitness`, `trial_fitness`, `target` and `initial_fitness`.
- **NaN** objective values are treated as $+\infty$ (worse than any number), with
  a `RuntimeWarning`. Infinite values are allowed, e.g. as a penalty.

!!! tip "Speed"
    Wrapping a plain function as an ioh problem costs a few microseconds per evaluation, because ioh
    calls your function once per point. For cheap functions that accept a whole `(n, d)` array, pass
    `wrap=False, vectorized=True` to skip ioh and evaluate a whole generation per call. This is several
    times faster, but the optimizer then counts the budget itself, and ioh logging is not available.

## When does a run stop?

A run stops as soon as one of these holds (checked after every batch of evaluations):

1. the **budget** is used up;
2. the **target** is reached: `f <= target` for a `target` you passed (`f >= target` for
   maximisation problems), or the ioh problem's own final target (BBOB and other ioh
   suites have one; a wrapped plain function has none unless you pass `target`);
3. **`max_generations`** generations are done (if set).

The rest of the generation is then skipped. A batch is always evaluated
completely, so a run can use a few evaluations more than the point where the
target was hit (at most one population). If a problem has already reached its
target before the run (for example when it is reused), the run stops
immediately; call `problem.reset()` to start again.

## Tracked attributes

These attributes of the optimizer describe its state. An ioh logger can
**watch** them and write them next to the objective values
(`logger.watch(opt, ["curr_F", "corrected"])`).

| Attribute | Meaning |
|---|---|
| `curr_F`, `curr_CR` | F and CR used for the point just evaluated |
| `corrected` | 1 if bound correction changed the point just evaluated, else 0 |
| `corr_so_far` | number of evaluated points changed by bound correction so far |
| `ED` | distance between the corrected and the uncorrected point (normalised space; 0 if not corrected) |
| `CS` | cosine similarity of the steps from the parent to the corrected and to the uncorrected point (0 if not corrected) |
| `popmean`, `popstd` | mean and standard deviation over all coordinates of all individuals pooled (as in 0.0.4; this is not a diversity measure) |
| `F_memory_mean` | mean of the SHADE F memory (NaN without SHADE F adaptation) |
| `generation` | completed generations |

!!! warning "Per-evaluation values need `track_evaluations=True`"
    `curr_F`, `curr_CR`, `corrected`, `CS` and `ED` refer to a single point. They
    are only kept up to date with `track_evaluations=True`, which evaluates one
    point at a time; otherwise `curr_F` and `curr_CR` are NaN. The population
    statistics are updated once per generation either way.

## Ask-tell

`modde.AskTellDE` lets you evaluate the points yourself (see the
[ask-tell notebook](notebooks/03_ask_tell.ipynb)). It always minimises (tell
$-f$ to maximise), counts the budget itself, and returns smaller batches near
the end of the budget. A batch may consist of opposite points when
oppositional generation is on. Once the run is finished, `stop` is true and
`ask()` raises `StopIteration`. For the same seed it follows exactly the same
trajectory as `ModularDE.run()`.
