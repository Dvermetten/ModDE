# Feature Requests (post-1.0 backlog)

Ideas that are explicitly **out of scope for 1.0** but should be reconsidered
afterwards. Each entry links to the decision that deferred it. Design choices
for 1.0 should avoid making these harder where that costs little.

| ID | Feature | Origin | Notes for 1.0 design |
|----|---------|--------|----------------------|
| FR-1 | **Customisable stats object.** Users choose which quantities are tracked and logged during a run (per generation / per evaluation), instead of a fixed set. | D-A20 | 1.0 ships the 0.0.4 tracked set in a stats object. Keep the tracking code in one place (`stats.hpp`) so it can later become a registry of trackers. |
| FR-2 | **Broader Python extensibility.** Every strategy overridable from Python, beyond the selected hooks (sampler, bound correction, mutation). | D-A8 | Keep strategy base-class signatures binding-friendly (Eigen refs, no raw pointers), so more trampolines can be added without API breaks. |
| FR-3 | **Parallel evaluation** inside the core (thread pool for C++ objectives, or parallel batch dispatch). | D-A16 | Batch evaluation already lets users parallelise. Release the GIL around pure C++ work. |
| FR-4 | **Bit-reproducibility across platforms** (own distribution implementations instead of `std::*_distribution`). | D-A13 | Route every random draw through one RNG wrapper, so the distributions can be swapped later. |
| FR-5 | **New DE modules and variants** (restarts, current-to-pbest-w as in jSO, non-linear population reduction, …). | D-A17 | Covered by Workstream 4 / Phase 6 "Scope extension" in [README.md](README.md). |
| FR-6 | **Linux aarch64 wheels / CI.** (macOS arm64 + x86_64 were moved into 1.0, D-A18 amendment.) | D-A18 | 1.0 builds Windows x64, Linux x86_64, macOS arm64/x86_64. Keep build scripts platform-agnostic. |
| FR-7 | **Dedicated logging / analysis integration** (e.g. a ready-made ioh logging setup, richer tracked state). | D-B8 (postponed) | For now, `modde.ModularDE` exposes its statistics as plain attributes that ioh loggers can watch. |
| FR-8 | **C++ benchmark target linked against IOHexperimenter** (profiling and benchmarking without Python overhead). | D-B5 | The core never depends on ioh, so it can be linked from a separate C++ target. |
| FR-9 | **Integer and mixed-integer variables.** | D-B6 | Keep `Settings` extensible. |
| FR-10 | **Faithful jSO.** The modules jSO needs beyond the current `jSO-like` preset: a memory slot fixed at 0.9, a linearly decreasing p (0.25 → 0.125), averaging new memory values with the old (`(Lehmer + old)/2`), and pbest ≠ target during the first half. | Phase 4b (03_module_validation.md §7.4) | Phase 6 scope extension; the published jSO raw data and the CEC2017 wrapper in `dev/studies/cec/` can validate it. |
