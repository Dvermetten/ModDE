"""Speed benchmarks (D-A19). Run on an otherwise idle machine:

    python benchmarks/benchmark.py            # prints Markdown tables

Measures
1. core overhead per evaluation (objective returns zeros), by configuration and d;
2. total wall-clock time for repeated full runs on ioh BBOB problems, new vs 0.0.4;
3. batch vs one Python call per point (D-A11);
4. cost of the ioh wrapper vs a plain NumPy function (D-B3);
5. cost of exact budget capping (D-A21): budget a multiple of lambda vs not.
"""
from __future__ import annotations

import platform
import time
import warnings

import numpy as np

import modde

warnings.filterwarnings("ignore")

CONFIGS = {
    "classic DE": {},
    "L-SHADE": {"preset": "L-SHADE"},
    "jDE + eigen crossover": {"adaptation_method_F": "JDE", "adaptation_method_CR": "JDE",
                              "eigenvalue_crossover": True},
}


def timed(fn, repeat=3):
    best = np.inf
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def overhead():
    rows = ["| Configuration | d | µs / evaluation | ns / coordinate |", "|---|---|---|---|"]
    null = lambda X: np.zeros(len(X))  # noqa: E731
    for name, cfg in CONFIGS.items():
        for d in (2, 10, 50, 100):
            budget = 2000 * d
            t = timed(lambda: modde.ModularDE(null, budget=budget, dim=d, lb=-5, ub=5, wrap=False,
                                              vectorized=True, seed=1, **cfg).run())
            rows.append(f"| {name} | {d} | {1e6 * t / budget:.2f} | {1e9 * t / budget / d:.1f} |")
    return rows


def repeated_runs(n_runs=25):
    import ioh
    from modde.legacy import ModularDE as Legacy

    legacy_cfg = {
        "classic DE": {},
        "L-SHADE": dict(base_sampler="uniform", mutation_base="target", mutation_reference="pbest",
                        bound_correction="expc_center", crossover="bin", lpsr=True, memory_size=6,
                        use_archive=True, adaptation_method_F="shade", adaptation_method_CR="shade"),
    }
    rows = [f"| Configuration | Problem | d | 0.0.4 ({n_runs} runs) | new ({n_runs} runs) | speed-up |",
            "|---|---|---|---|---|---|"]
    for name in ("classic DE", "L-SHADE"):
        for fid in (1, 15):
            for d in (10, 20):
                budget = 2000 * d

                def old():
                    for s in range(n_runs):
                        p = ioh.get_problem(fid, 1, d)
                        p.set_final_target(-np.inf)
                        kw = dict(legacy_cfg[name])
                        if name == "L-SHADE":
                            kw["lambda_"] = 18 * d
                        Legacy(p, budget=budget, seed=s, **kw).run()

                def new():
                    for s in range(n_runs):
                        p = ioh.get_problem(fid, 1, d)
                        p.set_final_target(-np.inf)
                        modde.ModularDE(p, budget=budget, seed=s, **CONFIGS[name]).run()

                t_old, t_new = timed(old, 1), timed(new, 1)
                rows.append(f"| {name} | BBOB f{fid} | {d} | {t_old:.2f} s | {t_new:.2f} s | {t_old / t_new:.1f}× |")
    return rows


def batch_vs_point():
    rows = ["| Objective | d | time (budget 2000·d) |", "|---|---|---|"]
    point = lambda x: float(np.dot(x, x))  # noqa: E731
    batch = lambda X: np.einsum("ij,ij->i", X, X)  # noqa: E731
    for d in (10, 50):
        budget = 2000 * d
        t_p = timed(lambda: modde.ModularDE(point, budget=budget, dim=d, lb=-5, ub=5, wrap=False, seed=1).run())
        t_b = timed(lambda: modde.ModularDE(batch, budget=budget, dim=d, lb=-5, ub=5, wrap=False,
                                            vectorized=True, seed=1).run())
        rows.append(f"| one call per point | {d} | {t_p:.3f} s |")
        rows.append(f"| vectorised (batch) | {d} | {t_b:.3f} s ({t_p / t_b:.1f}× faster) |")
    return rows


def ioh_wrapper_cost():
    rows = ["| Objective | d | time (budget 2000·d) |", "|---|---|---|"]
    batch = lambda X: np.einsum("ij,ij->i", X, X)  # noqa: E731
    point = lambda x: float(np.dot(x, x))  # noqa: E731
    for d in (10, 50):
        budget = 2000 * d
        t_plain = timed(lambda: modde.ModularDE(batch, budget=budget, dim=d, lb=-5, ub=5, wrap=False,
                                                vectorized=True, seed=1).run())
        t_wrapped = timed(lambda: modde.ModularDE(point, budget=budget, dim=d, lb=-5, ub=5, seed=1).run())
        rows.append(f"| plain vectorised NumPy | {d} | {t_plain:.3f} s |")
        rows.append(f"| wrapped as ioh problem (default) | {d} | {t_wrapped:.3f} s |")
    return rows


def exact_budget_cost():
    null = lambda X: np.zeros(len(X))  # noqa: E731
    d, lam = 10, 100
    t_mult = timed(lambda: modde.ModularDE(null, budget=200_000, dim=d, lb=-5, ub=5, wrap=False,
                                           vectorized=True, lambda0=lam, seed=1).run(), 5)
    t_odd = timed(lambda: modde.ModularDE(null, budget=200_037, dim=d, lb=-5, ub=5, wrap=False,
                                          vectorized=True, lambda0=lam, seed=1).run(), 5)
    return ["| Budget | time |", "|---|---|",
            f"| 200 000 (multiple of λ = 100) | {t_mult:.3f} s |",
            f"| 200 037 (last generation capped) | {t_odd:.3f} s |"]


def main():
    info = modde._core.build_info()
    print(f"# modde benchmarks\n\n{platform.platform()}, {platform.processor()}, Python "
          f"{platform.python_version()}, {info['compiler']}, modde {modde.__version__}. "
          "Best of 3 repetitions unless stated.\n")
    for title, fn in (("Core overhead per evaluation", overhead),
                      ("Repeated full runs: new vs 0.0.4", repeated_runs),
                      ("Batch vs one call per point", batch_vs_point),
                      ("Cost of the ioh wrapper", ioh_wrapper_cost),
                      ("Exact budget capping", exact_budget_cost)):
        print(f"## {title}\n")
        print("\n".join(fn()) + "\n", flush=True)


if __name__ == "__main__":
    main()
