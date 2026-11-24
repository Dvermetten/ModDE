"""Regression snapshot per platform (D-A13). See tools/make_regression_snapshot.py.

Bit-identical results are only guaranteed for the same build on the same
machine: another compiler version, or the CPU-dependent code paths of the C
runtime's math functions, can change the last bits of a result. On plateaus
(BBOB f7, the step ellipsoid) such a difference can break a tie in selection
differently, and the run diverges. A behaviour change in modde moves almost
every value of a case, so the comparison tolerates both:

* values agree within a relative tolerance of 1e-6, and
* at most ``MAX_MISMATCH`` of the 24 functions of a case may differ beyond that.

Measured with 1-4 ulp noise on every evaluation: at most 4 mismatches per case.
A different seed (a stand-in for a real behaviour change): at least 20 per case.
"""
import json
import math
import os
import warnings
from pathlib import Path

import pytest

from regression_cases import CASES, platform_key, run_case

SNAPSHOT = Path(__file__).parent / "regression" / f"snapshot_{platform_key()}.json"
RTOL = 1e-6
MAX_MISMATCH = 6


@pytest.fixture(scope="module")
def snapshot():
    if not SNAPSHOT.exists():
        if os.environ.get("MODDE_REQUIRE_SNAPSHOT"):
            pytest.fail(f"no regression snapshot for platform {platform_key()} (required in CI); generate "
                        "it with tools/make_regression_snapshot.py on this platform")
        pytest.skip(f"no regression snapshot for platform {platform_key()}")
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def test_snapshot_covers_all_cases(snapshot):
    assert set(snapshot) == set(CASES), "regenerate the snapshot after adding cases"


def _mismatches(got, expected):
    """Functions whose result differs; a non-finite result (NaN, inf) always counts."""
    return [(fid, e, g) for fid, (e, g) in enumerate(zip(expected, got), start=1)
            if not math.isfinite(g) or abs(g - e) > RTOL * max(1.0, abs(e))]


@pytest.mark.parametrize("name", list(CASES))
def test_results_unchanged(name, snapshot):
    warnings.filterwarnings("ignore")
    got = [run_case(CASES[name], fid) for fid in range(1, 25)]
    bad = _mismatches(got, snapshot[name])
    detail = ", ".join(f"f{fid}: {e:.10g} -> {g:.10g}" for fid, e, g in bad)
    assert len(bad) <= MAX_MISMATCH, (
        f"{name}: {len(bad)} of 24 functions changed ({detail}). "
        "If intended, regenerate with tools/make_regression_snapshot.py")
