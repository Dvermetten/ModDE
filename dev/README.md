# ModDE vNext — Development Plan

This folder holds the planning documents for the next major version of `modde`.
It is the working space for developers, not user documentation. User-facing
docs will live in `docs/` (see [02_documentation.md](02_documentation.md)).

## Goals

| # | Workstream | Outcome |
|---|------------|---------|
| 1 | [Speed: C++ core](01_cpp_backend.md) | All per-generation DE computation runs in C++, exposed to Python through nanobind. Structure follows `modcma`. |
| 2 | [Documentation](02_documentation.md) | A user can install `modde` and run a meaningful optimisation in under 5 minutes. Every module is documented, with its parameters and interactions. |
| 3 | [Module validation](03_module_validation.md) | Every module is tested against a written spec. All module interactions are classified and documented. No parameter is ever changed silently. |
| 4 | Scope extension (after 1.0, Phase 6) | Extend the range of DE variants covered (new modules such as restarts). **No new modules before 1.0** (D-A17). |

Supporting documents:

- [DECISIONS.md](DECISIONS.md) lists every open design decision, with options and a recommendation. **Most of Phase 0 is resolving these.**
- [KNOWN_ISSUES.md](KNOWN_ISSUES.md) audits the current (0.0.4) Python code: bugs, deviations from the literature, silent overrides.
- [FEATURE_REQUESTS.md](FEATURE_REQUESTS.md) is the post-1.0 backlog of explicitly deferred features.
- `studies/` holds supporting experiments (e.g. the paper-deviation impact study). **It is local only, not in the repository** (see `.gitignore`).

## How the workstreams depend on each other

```
            ┌───────────────────────────┐
            │ Phase 0: decide + scaffold │  (DECISIONS.md, build skeleton, CI)
            └─────────────┬─────────────┘
                          │
            ┌─────────────▼─────────────┐
            │ Phase 1: module spec +     │  WS3 (spec part)
            │ parameter registry         │  -> single source of truth for
            └─────────────┬─────────────┘     C++, ConfigSpace and docs
                          │
            ┌─────────────▼─────────────┐
            │ Phase 2: C++ core          │  WS1
            │ (+ operator-level tests)   │  WS3 unit tests written per module
            └─────────────┬─────────────┘
                          │
            ┌─────────────▼─────────────┐
            │ Phase 3: Python API,       │  WS1 + docs skeleton (WS2)
            │ ioh + ConfigSpace          │
            └─────────────┬─────────────┘
                          │
            ┌─────────────▼─────────────┐
            │ Phase 4: validation        │  WS3 (interaction matrix,
            │ campaign + benchmarks      │  reference variants, perf)
            └─────────────┬─────────────┘
                          │
            ┌─────────────▼─────────────┐
            │ Phase 5: docs + release    │  WS2
            └───────────────────────────┘
```

Key principle: **the module spec comes before the port.** The current Python
code has several bugs and undocumented deviations
([KNOWN_ISSUES.md](KNOWN_ISSUES.md)). Translating it line by line into C++ would
carry them along. Each module is first written down precisely (Phase 1), then
implemented in C++ against that spec, with tests derived from it.

## Phases and milestones

### Phase 0: Decide and scaffold
- [x] Resolve the architecture decisions (D-A*). See the decision log at the end of [DECISIONS.md](DECISIONS.md).
- [x] Resolve the remaining blocking decisions D-D1 (parameter policy) and D-B1 (accepted problem types).
- [x] Repo hygiene: `.gitignore` (build/, dist/, *.egg-info, __pycache__, .ipynb_checkpoints, `Untitled*.ipynb`). The scratch notebook is kept locally, just ignored.
- [x] Move the 0.0.4 code to `modde/legacy/` with a `DeprecationWarning` on import (D-A5). Until Phase 3, top-level `modde.ModularDE` etc. forward to legacy with a warning. Legacy tests moved to `tests/legacy/` (882 pass).
- [x] `pyproject.toml` with scikit-build-core (D-A2), `requires-python >= 3.10` (D-A18); `setup.py`, `requirements.txt` and the old CI workflow removed. Version `1.0.0.dev0` (final scheme still open, D-E4).
- [x] Eigen 3.4.0 as a shallow git submodule under `external/eigen` (D-A3).
- [x] "Hello world" nanobind extension `modde._core` (C++17, `include/modde/common.hpp`, `src/bindings.cpp`). Builds locally with MSVC 19.39 + nanobind 3.1; 7 core tests pass, including the n × d NumPy → Eigen path.
- [x] CI workflow (`.github/workflows/ci.yml`): Windows + Linux, Python 3.11. **Not yet run on GitHub** (branch not pushed).
- [x] Wheel dry-run workflow (`.github/workflows/wheels.yml`, manual trigger, artifacts only): cibuildwheel for Windows x64 + Linux x86_64, plus an sdist check. The sdist (1 MB: Eigen headers + licences) was verified locally by building and installing from it (889 tests pass). **Not yet run on GitHub.**
- [x] src layout: Python package in `src/modde/`. `dev/studies/` (raw data, reports) excluded from git and kept locally.
- [x] Branch pushed; **CI passes on GitHub for Windows and Linux** (2026-11-19). This also confirms the Linux build and the Eigen submodule checkout.
- [ ] Run the manual "Wheels (dry run)" workflow once and check the wheel and sdist artifacts.
- **M0:** an empty C++ extension builds, imports and produces wheels on Windows and Linux. *Build and tests confirmed in CI on both platforms; the wheel dry run is still to be run.*

### Phase 1: Spec and parameter registry (WS3, part 1)
- [x] Triage [KNOWN_ISSUES.md](KNOWN_ISSUES.md): bugs fixed; paper deviations move to the paper version.
- [x] Before/after impact report per paper deviation (local only: `studies/paper_deviations/reports/`).
- [x] Write [`dev/spec/`](spec/README.md): one document per module category (exact math, every random draw, parameters by registry name, literature), plus settings/resolution, termination, statistics and presets. *Draft; open points listed in [spec/README.md](spec/README.md).*
- [x] Build the **parameter registry** [`src/modde/_registry.py`](../src/modde/_registry.py): 43 settings with type, valid range, default rule, activity conditions, ConfigSpace range and group, 0.0.4 name, references. It also holds the derived rules (`n_parents_needed`, `lambda_min`), the 10 presets, and a prototype ConfigSpace builder. Tested by `tests/test_registry.py`, which covers: every sampled configuration is valid (D-C5), the default configuration is classic DE, and every preset is a valid configuration for d ∈ {1, 2, 5, 20, 100}.
- [x] Define the parameter resolution policy ([spec/settings.md](spec/settings.md) §1: USER / PROBLEM / DEFAULT / DERIVED provenance, errors vs warnings).
- [x] **Author review** of the spec, the registry names and the open points (2026-11-19): points 1–4 accepted; bound-correction references added from the ModDE paper; preset verification deferred (does not block Phase 2).
- **M1:** spec and registry reviewed. ✅ The C++ work can start.

### Phase 2: C++ core (WS1)
- [x] Core types: `Population`, `Rng`, `Settings` (+ `resolve()` with provenance and warnings), `Modules`, `Parameters`, `Stats`.
- [x] All modules implemented from the spec: samplers (own Sobol with Joe & Kuo numbers, own Halton, LHS, truncated Gaussian), (quasi-)oppositional initialisation, oversampling, mutation, crossover (incl. eigen), all bound corrections, selection + success record, archive, SHADE / SHADE_MODIFIED / LSHADE / jDE / jSO caps, linear population reduction, oppositional generation, statistics, exact budget, termination.
- [x] Split step API exposed to Python (`modde._core.ModularDE`): `initialize`, `sample_parameters`, `mutate`, `crossover`, `correct_bounds`, `evaluate`, `select`, `oppositional_generation`, `adapt`, `step`, `run`, plus state views (n × d).
- [x] Operator-level tests against the spec (`tests/test_operators.py`, `test_samplers.py`, `test_settings_core.py`) and run-level tests (`test_runs.py`: exact budget, evaluation offset, LPSR schedule, tracking mode, determinism, all presets, invariants over sampled configurations). 418 new tests; 1300 in total.
- [x] First speed check (details in [01_cpp_backend.md](01_cpp_backend.md)): core overhead 0.5–3 µs per evaluation; end-to-end on ioh f1 vs 0.0.4: L-SHADE 15–33× faster, classic DE 1.5–3×.
- **M2:** `ModularDE(settings).run(f)` in C++ reproduces the spec on all modules. ✅ (Python front end: Phase 3.)

### Phase 3: Python API and integrations (WS1)
- [ ] nanobind bindings + generated `.pyi` stubs.
- [ ] High-level `fmin`, `settings_from_dict`, named presets (L-SHADE, …).
- [ ] ioh integration (bounds/target/budget from the problem, logging of internal state).
- [ ] ConfigSpace generation from the registry, `settings_from_config`.
- [ ] Ask-tell interface (scope per D-A12).
- [ ] Docs skeleton and API reference wired up (WS2).
- **M3:** the full public API is usable. Alpha release on TestPyPI.

### Phase 4: Validation campaign (WS3, part 2)
- [ ] Property-based invariant tests over random configurations.
- [ ] Pairwise interaction matrix, results in `docs/` (generated).
- [ ] Reference-variant validation (DE/rand/1/bin, JADE, SHADE, L-SHADE, jDE, jSO-style) against the literature or reference implementations.
- [ ] Performance benchmarks against 0.0.4 (and other DE libraries if wanted).
- **M4:** the validation report is published and every interaction is classified.

### Phase 4b: CEC-suite validation (separate stage)
- [ ] Wrap CEC2014/CEC2017 implementations with `ioh.wrap_problem` (no integration into ioh). Validate the wrapped functions themselves first.
- [ ] Run L-SHADE / jSO-style presets under the competition protocol. Compare with published results.
- See [03_module_validation.md §5.6](03_module_validation.md#56-cec-suite-validation-phase-4b). Can run in parallel with Phase 5. Not in CI.
- **M4b:** CEC comparison added to the validation report.

### Phase 5: Documentation and release (WS2)
- [ ] Getting-started guide, tutorials, module reference, migration guide.
- [ ] Release 1.0 (or whatever version D-E4 decides) on PyPI.
- **M5:** release.

### Phase 6: Scope extension (Workstream 4, after 1.0)
- [ ] Survey DE variants not covered by the module set (restarts, current-to-pbest-w, non-linear population size reduction, other adaptation schemes, …) and prioritise them.
- [ ] Each new module follows the full pipeline: spec → registry → C++ → tests → interaction matrix → docs.
- [ ] Re-evaluate [FEATURE_REQUESTS.md](FEATURE_REQUESTS.md).

## Conventions for these documents
- Decisions are referenced as `D-<area><n>` (e.g. `D-A1`). Status values: `OPEN`, `PROPOSED`, `DECIDED (date)`.
- Known issues are referenced as `KI-<n>`.
- When a decision is made, update its status in [DECISIONS.md](DECISIONS.md) and the affected plan document in the same commit.
