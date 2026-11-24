"""Whole-run behaviour: budget, termination, schedules, determinism, presets,
and invariants over random configurations (WS3 §5.2, early version)."""
import numpy as np
import pytest

from conftest import Recorder, make_de, sphere, sphere_batch
from modde import _core
from modde import _registry as R
from modde._settings import make_settings


@pytest.mark.parametrize("vectorized", (False, True))
def test_exact_budget(vectorized):
    rec = Recorder(sphere_batch if vectorized else sphere)
    de = make_de(lambda0=7, budget=1000, dim=3)
    de.run(rec, vectorized=vectorized)
    expected_calls = 1000 if not vectorized else None
    assert de.stats.evaluations == 1000  # KI-8: 1000 is not a multiple of 7
    if expected_calls:
        assert rec.calls == 1000
    else:
        assert sum(len(p) for p in rec.points) == 1000


def test_evaluation_offset_counts_towards_budget():
    """D-B3: a problem with earlier evaluations; the budget applies to the total."""
    rec = Recorder()
    de = make_de(lambda0=10, budget=500, dim=2)
    de.stats.evaluations = 200
    de.run(rec)
    assert rec.calls == 300 and de.stats.evaluations == 500


def test_target_and_max_generations():
    de = make_de(lambda0=20, budget=100000, dim=2, target=1e-6)
    de.run(sphere)
    assert de.stats.best_f <= 1e-6 and de.stats.evaluations < 100000
    de = make_de(lambda0=20, budget=100000, dim=2, max_generations=7)
    de.run(sphere)
    assert de.stats.generation == 7


def test_linear_population_reduction():
    de = make_de(dim=4, lambda0=40, budget=4000, population_reduction="LINEAR", use_archive=True,
                 mutation_base="TARGET", mutation_reference="PBEST")
    sizes, ok_archive = [], True
    while de.step(sphere):
        sizes.append(de.lambda_)
        ok_archive &= de.archive.shape[0] <= de.archive_capacity
    sizes.append(de.lambda_)
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[-1] == de.settings.lambda_min == 4
    assert ok_archive


def test_tracking_mode_reports_per_evaluation_values():
    de = make_de(dim=3, lambda0=10, budget=300, track_evaluations=True, adaptation_method_F="JDE")
    seen = []

    def f(x):
        if de.initialized:
            seen.append((de.stats.curr_F, de.stats.corrected))
        return sphere(x)

    de.run(f, vectorized=True)  # tracking forces per-point calls even if vectorised
    assert len(seen) == 300 - 10
    assert de.stats.corr_so_far == sum(c for _, c in seen)


def test_determinism():
    a, b, c = make_de(seed=11), make_de(seed=11), make_de(seed=12)
    for de in (a, b, c):
        de.run(sphere)
    np.testing.assert_array_equal(a.population, b.population)
    assert not np.array_equal(a.population, c.population)


def test_classic_de_converges_on_sphere():
    de = make_de(dim=5, budget=20000)
    de.run(sphere)
    assert de.stats.best_f < 1e-6


@pytest.mark.parametrize("name", list(R.PRESETS))
def test_presets_run(name):
    dim = 5
    kw = R.PRESETS[name].settings(dim)
    de = _core.ModularDE(make_settings(dim, lb=-5.0, ub=5.0, seed=1, budget=3000, **kw))
    de.run(sphere)
    assert de.stats.evaluations == 3000 or de.break_conditions()
    assert np.isfinite(de.stats.best_f) and de.stats.best_f < 1.0


@pytest.mark.parametrize("dim", (1, 3, 10))
def test_invariants_over_sampled_configurations(dim):
    """Every configuration of the ConfigSpace runs; invariants hold every generation."""
    cs = R._build_configspace(dim)
    cs.seed(dim)
    for config in cs.sample_configuration(25):
        kw = dict(config)
        budget = 60 * dim + kw["lambda0"] * 3
        de = _core.ModularDE(make_settings(dim, lb=-5.0, ub=5.0, seed=0, budget=budget, **kw))
        lam_min = de.settings.lambda_min
        while True:
            more = de.step(sphere)
            X, f = de.population, de.fitness
            assert X.shape == (de.lambda_, dim) and np.all(np.isfinite(f)), kw
            assert de.lambda_ >= (lam_min if kw["population_reduction"] == "LINEAR" else kw["lambda0"])
            if kw["bound_correction"] != "NONE":
                assert np.all((X >= -5) & (X <= 5)), kw
            assert de.stats.evaluations <= budget
            if not more:
                break
        assert de.stats.evaluations == budget, kw
