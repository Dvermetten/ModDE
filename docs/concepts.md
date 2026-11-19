# Concepts

## Modules, settings and the optimizer

- **Modules** are the algorithm's building blocks, chosen by option name, for
  example `mutation_base="TARGET"` or `bound_correction="MIRROR"`.
- **Settings** are the numbers: population size `lambda0`, initial `F0` and
  `CR0`, archive rate, and so on. Many belong to a module and only have an
  effect when that module is active.
- Both are passed by name; [Parameters](parameters.md) lists them all.
- When an optimizer is created, the settings are **resolved**: defaults are
  filled in, derived values are computed, and everything is validated. Each
  value records where it came from (`USER`, `PROBLEM`, `DEFAULT` or `DERIVED`):

```python
opt = modde.ModularDE(f, budget=1000, dim=5, lb=-5, ub=5, lambda0=30)
opt.settings.provenance["lambda0"]   # Provenance.USER
opt.settings.provenance["budget"]    # Provenance.USER (passed explicitly)
opt.settings.provenance["CR0"]       # Provenance.DEFAULT
```

**A value you set is never changed.** If it cannot work with the chosen modules
(say, a population too small for the mutation), you get an error saying why. If
it has no effect (say, `memory_size` without SHADE adaptation), you get a warning.

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
inspected between steps (`opt.population`, `opt.donors`, `opt.trials`,
`opt.archive`, ...). Populations are `(n, d)` arrays.

## Problems and budget

`modde` works on [ioh](https://iohprofiler.github.io/) problems. A plain
function is wrapped as one automatically (`wrap=False` to opt out). The problem
counts the evaluations, and the budget applies to that count. If the problem
already has evaluations when a run starts, modde warns and counts them too.
Maximisation problems are handled transparently.
