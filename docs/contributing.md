# Contributing

Contributions are welcome: bug reports, new modules, documentation. This page
explains how the code is organised and what a new module needs.

## Layout

| Path | Contents |
|---|---|
| `include/modde/`, `src/*.cpp` | the C++ core: module enums (`modules.hpp`), settings and their resolution (`settings.cpp`), the operators (`mutation.cpp`, `crossover.cpp`, `bounds.cpp`, `adaptation.cpp`, `sampling.cpp`) and the generation loop (`modde.cpp`) |
| `src/bindings.cpp` | the nanobind bindings (`modde._core`) |
| `src/modde/_registry.py` | the **parameter registry**: every setting with its type, options, default, valid and tuning range, activity condition and references |
| `src/modde/*.py` | the Python API: `fmin`, `ModularDE`, `AskTellDE`, ConfigSpace and irace support |
| `tests/` | pytest suite; `tests/regression/` holds the regression snapshots |
| `docs/`, `tools/` | this documentation and the scripts that generate parts of it |

The registry is the single source of truth for settings. The ConfigSpace, the
[Parameters](parameters.md) page, the tables on the module pages and the
presets are all generated from it, and tests check that the C++ defaults and
options agree with it.

## Building and testing

```bash
git clone --recursive https://github.com/Dvermetten/ModDE.git
cd ModDE
pip install ".[test,docs]"
pytest
```

After changing C++ code, run `pip install .` again. `mkdocs serve` builds the
documentation locally, executing all notebooks.

## Adding a module

A module option touches the following places. The tests fail if the registry
and the core disagree, so working through the list in order is the easiest way.

1. **Semantics first.** Write down exactly what the option does: the formula,
   every random draw, edge cases, the literature reference.
2. **C++ enum** in `include/modde/modules.hpp`: add the option to an existing
   enum, or add an enum (and a field of `Modules`) for a new module. The first
   enum value is the default and must keep classic DE unchanged.
3. **Implementation** in the operator's source file. If the module has numeric
   parameters, add them to `Settings` (`settings.hpp`) with a default in
   `Settings::resolve()` (`settings.cpp`), a range check, and a warning when a
   user sets them while the module is off.
4. **Bindings** in `src/bindings.cpp`: `bind_enum` for a new enum, `def_rw` for
   new fields.
5. **Registry** entry in `src/modde/_registry.py`: a `Param` with the options,
   documentation, tuning range or group, `active_when` conditions and
   references (add new references to `REF`, and per-option references to
   `OPTION_REFERENCES`). This makes the option tunable and documents it.
6. **Tests** in `tests/test_operators.py` (operator behaviour against the
   semantics) and, where needed, `tests/test_runs.py`.
7. **Regression snapshot:** a new option adds a regression case, so
   regenerate the snapshot with `python tools/make_regression_snapshot.py` and
   check in the diff that only the new case was added.
8. **Documentation:** describe the option on its [module page](modules/index.md)
   (the parameter tables update themselves) and add a line to the changelog.

The [interaction study](interactions.md) is re-run by the maintainers before a
release, so new options appear there later.

## Reproducibility and the regression snapshot

The same seed gives the same result with the same build on the same machine.
`tests/test_regression.py` compares the final results of every preset and
module option on BBOB with a stored snapshot per platform, with a small
tolerance for last-bit differences between compilers and CPUs. If a change
alters results on purpose, regenerate the snapshot and say so in the changelog.

## Reporting a bug

Please open an issue on [GitHub](https://github.com/Dvermetten/ModDE/issues)
with the settings, the seed, the modde version (`modde.__version__`) and,
for build problems, `modde._core.build_info()`.
