"""Cases of the regression snapshot (see tools/make_regression_snapshot.py)."""
from __future__ import annotations

import sys

import ioh

import modde
from modde import _core
from modde import _registry as R

DIM, BUDGET, SEED = 2, 200, 42


def _single_options():
    out = {}
    for p in R.PARAMS:
        if p.kind == "categorical" and p.group == "modules":
            for option in p.options[1:]:
                out[f"{p.name}={option}"] = {p.name: option}
        elif p.kind == "bool" and p.group == "modules":
            out[f"{p.name}=True"] = {p.name: True}
    out["mutation_n_comps=2"] = {"mutation_n_comps": 2}
    out["mutation_reference=PBEST,pbest_sampling=FIXED"] = {"mutation_reference": "PBEST",
                                                            "pbest_sampling": "FIXED"}
    return out


CASES = {**{f"preset:{name}": R.PRESETS[name].settings(DIM) for name in R.PRESETS}, **_single_options()}


def run_case(settings: dict, fid: int) -> float:
    problem = ioh.get_problem(fid, 1, DIM)
    kwargs = dict(settings)
    kwargs.setdefault("lambda0", 20)
    if kwargs.get("lambda0", 20) > 20:
        kwargs["lambda0"] = 20  # keep generations comparable within the small budget
    opt = modde.ModularDE(problem, budget=BUDGET, seed=SEED, **kwargs)
    opt.run()
    return float(opt.best_f)


def platform_key() -> str:
    compiler = _core.build_info()["compiler"].split()[0].lower()
    return f"{sys.platform}-{compiler}"
