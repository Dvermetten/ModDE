import numpy as np
import pytest

from modde import _core
from modde._settings import make_settings


def sphere(x):
    return float(np.sum(np.asarray(x) ** 2))


def sphere_batch(X):
    return np.sum(np.asarray(X) ** 2, axis=1)


class Recorder:
    """Objective that counts calls and remembers every evaluated point."""

    def __init__(self, f=sphere):
        self.f = f
        self.points = []

    def __call__(self, x):
        self.points.append(np.array(x, copy=True))
        return self.f(x)

    @property
    def calls(self):
        return len(self.points)


def make_de(dim=5, lb=-5.0, ub=5.0, seed=1, **kwargs):
    return _core.ModularDE(make_settings(dim, lb=lb, ub=ub, seed=seed, **kwargs))


@pytest.fixture
def de_factory():
    return make_de


# ------------------------------------------------------------------ CI annotations
# On GitHub Actions, report every failing test as an error annotation. Unlike the
# job log, annotations can be read without admin rights on the repository.

def _escape(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(text: str) -> str:
    return _escape(text).replace(":", "%3A").replace(",", "%2C")


def pytest_terminal_summary(terminalreporter):
    import os

    if not os.environ.get("GITHUB_ACTIONS"):
        return
    failed = terminalreporter.stats.get("failed", []) + terminalreporter.stats.get("error", [])
    for report in failed[:9]:
        lines = [line for line in report.longreprtext.splitlines() if line.startswith("E ")]
        message = "\n".join(lines[-15:]) or report.longreprtext[-1500:]
        path = report.location[0] if report.location else ""
        terminalreporter.write_line(
            f"::error file=tests/{os.path.basename(path)},title={_escape_property(report.nodeid)}::{_escape(message[:3000])}")
    if failed:
        ids = "\n".join(r.nodeid for r in failed)
        terminalreporter.write_line(f"::error title={len(failed)} failed tests::{_escape(ids[:4000])}")
