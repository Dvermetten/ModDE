# modde benchmarks

Windows-10-10.0.26300-SP0, Intel64 Family 6 Model 186 Stepping 3, GenuineIntel, Python 3.11.9, MSVC 1939, modde 1.0.0a1. Best of 3 repetitions unless stated.

## Core overhead per evaluation

| Configuration | d | µs / evaluation | ns / coordinate |
|---|---|---|---|
| classic DE | 2 | 0.47 | 236.1 |
| classic DE | 10 | 0.71 | 71.0 |
| classic DE | 50 | 1.41 | 28.3 |
| classic DE | 100 | 2.30 | 23.0 |
| L-SHADE | 2 | 0.48 | 242.5 |
| L-SHADE | 10 | 0.51 | 51.2 |
| L-SHADE | 50 | 1.62 | 32.5 |
| L-SHADE | 100 | 3.33 | 33.3 |
| jDE + eigen crossover | 2 | 0.49 | 245.2 |
| jDE + eigen crossover | 10 | 0.66 | 65.9 |
| jDE + eigen crossover | 50 | 4.36 | 87.1 |
| jDE + eigen crossover | 100 | 13.65 | 136.5 |

## Repeated full runs: new vs 0.0.4

| Configuration | Problem | d | 0.0.4 (25 runs) | new (25 runs) | speed-up |
|---|---|---|---|---|---|
| classic DE | BBOB f1 | 10 | 2.26 s | 0.59 s | 3.8× |
| classic DE | BBOB f1 | 20 | 4.93 s | 1.58 s | 3.1× |
| classic DE | BBOB f15 | 10 | 3.59 s | 1.72 s | 2.1× |
| classic DE | BBOB f15 | 20 | 8.80 s | 5.77 s | 1.5× |
| L-SHADE | BBOB f1 | 10 | 15.96 s | 1.18 s | 13.5× |
| L-SHADE | BBOB f1 | 20 | 49.89 s | 1.95 s | 25.6× |
| L-SHADE | BBOB f15 | 10 | 18.57 s | 1.81 s | 10.3× |
| L-SHADE | BBOB f15 | 20 | 55.77 s | 6.35 s | 8.8× |

## Batch vs one call per point

| Objective | d | time (budget 2000·d) |
|---|---|---|
| one call per point | 10 | 0.116 s |
| vectorised (batch) | 10 | 0.013 s (8.8× faster) |
| one call per point | 50 | 0.381 s |
| vectorised (batch) | 50 | 0.149 s (2.6× faster) |

## Cost of the ioh wrapper

| Objective | d | time (budget 2000·d) |
|---|---|---|
| plain vectorised NumPy | 10 | 0.017 s |
| wrapped as ioh problem (default) | 10 | 0.089 s |
| plain vectorised NumPy | 50 | 0.252 s |
| wrapped as ioh problem (default) | 50 | 0.442 s |

## Exact budget capping

| Budget | time |
|---|---|
| 200 000 (multiple of λ = 100) | 0.184 s |
| 200 037 (last generation capped) | 0.121 s |

## Reading the numbers (2026-11-19)

- **D-A19 target (≈ 20× vs 0.0.4):** met for adaptive configurations (L-SHADE 9–26× over 25 full runs), not for plain DE (1.5–4×). 0.0.4's plain DE was already mostly vectorised NumPy, and on BBOB f15 the ioh evaluation (two d × d rotations per point) dominates both versions. As decided in D-A19, no complexity was added to chase the last factor.
- **Core overhead** stays at 0.5–3.3 µs per evaluation for classic DE and L-SHADE. Eigenvector crossover is the expensive module at high d (one eigendecomposition per generation): 14 µs per evaluation at d = 100.
- **Batch evaluation** is 2.6–8.8× faster than one Python call per point (D-A11).
- **ioh wrapping** (the default for plain functions, D-B3) costs ≈ 3.6 µs per evaluation, because ioh calls the Python function once per point. For cheap, vectorisable objectives, `wrap=False, vectorized=True` is the fast path; the docs say so.
- **Exact budget capping** (D-A21) has no measurable cost (the difference is noise), so the budget is always exact.

Regenerate with `python benchmarks/benchmark.py` on an otherwise idle machine.
