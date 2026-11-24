"""Samplers: spec/initialization.md §2."""
import numpy as np
import pytest
from scipy.stats import qmc

from modde import _core

SAMPLERS = ("UNIFORM", "GAUSSIAN", "SOBOL", "HALTON", "LHS")


@pytest.mark.parametrize("name", SAMPLERS)
def test_shape_range_and_determinism(name):
    Z = _core._sample(name, 7, 50, seed=3)
    assert Z.shape == (50, 7)
    assert np.all((Z >= 0) & (Z <= 1))
    np.testing.assert_array_equal(Z, _core._sample(name, 7, 50, seed=3))
    assert not np.array_equal(Z, _core._sample(name, 7, 50, seed=4))


@pytest.mark.parametrize("d", (1, 2, 7, 40))
def test_sobol_unscrambled_matches_scipy(d):
    """Same Joe & Kuo direction numbers as scipy, Gray-code order."""
    ours = _core._sample("SOBOL", d, 64, scramble=False)
    ref = qmc.Sobol(d, scramble=False).random(64)
    np.testing.assert_allclose(ours, ref, atol=1e-12)


@pytest.mark.parametrize("d", (1, 3, 10))
def test_halton_unscrambled_matches_scipy(d):
    ours = _core._sample("HALTON", d, 50, scramble=False)
    ref = qmc.Halton(d, scramble=False).random(51)[1:]  # we skip the all-zero point
    np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_sobol_digital_shift_keeps_stratification():
    """A digital shift preserves the (0, m, s)-net property in each dimension."""
    Z = _core._sample("SOBOL", 4, 32, seed=1)
    for j in range(4):
        assert sorted(np.floor(Z[:, j] * 32).astype(int)) == list(range(32))


def test_lhs_is_stratified():
    Z = _core._sample("LHS", 6, 40, seed=2)
    for j in range(6):
        assert sorted(np.floor(Z[:, j] * 40).astype(int)) == list(range(40))


def test_gaussian_is_truncated_normal():
    Z = _core._sample("GAUSSIAN", 2, 20000, seed=5)
    assert abs(Z.mean() - 0.5) < 0.01
    assert 0.15 < Z.std() < 0.17  # sd 1/6, slightly reduced by truncation
