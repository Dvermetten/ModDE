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
