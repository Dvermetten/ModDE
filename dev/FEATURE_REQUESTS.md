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
| FR-6 | **macOS and aarch64 wheels / CI.** | D-A18 | 1.0 tests and builds Windows x64 + Linux x86_64. Keep build scripts platform-agnostic. |
