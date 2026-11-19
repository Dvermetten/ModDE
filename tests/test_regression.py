"""Regression snapshot: exact results per platform (D-A13). See tools/make_regression_snapshot.py."""
import json
import warnings
from pathlib import Path

import pytest

from regression_cases import CASES, platform_key, run_case

SNAPSHOT = Path(__file__).parent / "regression" / f"snapshot_{platform_key()}.json"


@pytest.fixture(scope="module")
def snapshot():
    if not SNAPSHOT.exists():
        pytest.skip(f"no regression snapshot for platform {platform_key()}")
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def test_snapshot_covers_all_cases(snapshot):
    assert set(snapshot) == set(CASES), "regenerate the snapshot after adding cases"


@pytest.mark.parametrize("name", list(CASES))
def test_results_unchanged(name, snapshot):
    warnings.filterwarnings("ignore")
    got = [run_case(CASES[name], fid) for fid in range(1, 25)]
    assert got == snapshot[name], (
        f"{name}: results changed. If intended, regenerate with tools/make_regression_snapshot.py")
