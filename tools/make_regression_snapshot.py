"""(Re)generate the regression snapshot for the current platform.

The snapshot stores the final best value of every preset and every single
non-default module option on BBOB f1-f24 (d = 2, budget 200, seed 42). The test
`tests/test_regression.py` compares against it with a small tolerance (relative
1e-6, at most 6 of 24 functions per case may differ: last-bit differences between
compiler versions and CPUs can break ties on plateaus, D-A13), so any behaviour
change is noticed.

Regenerate deliberately, after an intended behaviour change, and mention it in
the changelog:  python tools/make_regression_snapshot.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from regression_cases import CASES, platform_key, run_case  # noqa: E402


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="directory to write to (default: tests/regression)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    data = {name: [run_case(settings, fid) for fid in range(1, 25)] for name, settings in CASES.items()}
    folder = Path(args.out) if args.out else ROOT / "tests" / "regression"
    out = folder / f"snapshot_{platform_key()}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(data)} cases x 24 functions)")


if __name__ == "__main__":
    main()
