"""Operator-level tests against dev/spec (WS3 §5.1)."""
import numpy as np
import pytest

from conftest import Recorder, make_de, sphere


def initialized(**kw):
    de = make_de(**kw)
    de.initialize(sphere)
    return de


# ---------------------------------------------------------------- mutation


def donors_rand_n(X, F, parents, n_comps):
    V = np.empty_like(X)
    for i, par in enumerate(parents):
        v = X[par[0]].copy()
        for c in range(n_comps):
            v += F[i] * (X[par[1 + 2 * c]] - X[par[2 + 2 * c]])
        V[i] = v
    return V


@pytest.mark.parametrize("n_comps", (1, 2, 3))
def test_rand_n_formula_and_distinct_parents(n_comps):
    de = initialized(mutation_n_comps=n_comps, lambda0=30)
    de.sample_parameters()
    de.mutate()
    parents = de.last_parents
    for i, par in enumerate(parents):
        assert len(par) == 1 + 2 * n_comps
        assert len(set(par)) == len(par) and i not in par  # KI-23
    np.testing.assert_allclose(de.donors, donors_rand_n(de.population, de.F_used, parents, n_comps))  # KI-1


def test_current_to_best():
    de = initialized(mutation_base="TARGET", mutation_reference="BEST", lambda0=20)
    de.sample_parameters()
    de.mutate()
    X, F = de.population, de.F_used
    best = X[np.argmin(de.fitness)]
    expected = [X[i] + F[i] * (best - X[i]) + F[i] * (X[p[0]] - X[p[1]]) for i, p in enumerate(de.last_parents)]
    np.testing.assert_allclose(de.donors, expected)


@pytest.mark.parametrize("sampling, rate, weighted", [("FIXED", 0.2, False), ("RANDOM", 0.3, False),
                                                      ("FIXED", 0.2, True)])
def test_pbest_vector_is_in_the_top_k(sampling, rate, weighted):
    kw = {"pbest_rate": rate} if sampling == "FIXED" else {"pbest_rate_max": rate}
    de = initialized(mutation_base="TARGET", mutation_reference="PBEST", pbest_sampling=sampling,
                     mutation_use_weighted_F=weighted, lambda0=40, **kw)
    de.sample_parameters()
    de.mutate()
    X, F = de.population, de.F_used
    w = 0.7 if weighted else 1.0  # progress t < 0.2 at the start (KI-24: only on the reference term)
    k = max(1, round(rate * 40))
    top = X[np.argsort(de.fitness, kind="stable")[:k]]
    for i, p in enumerate(de.last_parents):
        rest = de.donors[i] - X[i] - F[i] * (X[p[0]] - X[p[1]])
        x_pbest = X[i] + rest / (w * F[i])
        assert np.min(np.linalg.norm(top - x_pbest, axis=1)) < 1e-8


def test_archive_term_draws_from_population_when_archive_empty():
    de = initialized(mutation_base="TARGET", mutation_reference="PBEST", use_archive=True, lambda0=20)
    assert de.archive.shape[0] == 0  # KI-26: starts empty
    de.sample_parameters()
    de.mutate()
    X, F = de.population, de.F_used
    for i, p in enumerate(de.last_parents):
        assert len(p) == 1  # r1 only; the second vector comes from P u A
        # donor - x_i - F(x_pbest - x_i) = F (x_r1 - x~): x~ must be a population member != i, r1
        # (checked indirectly: some population member solves the equation)
        found = False
        for r in range(20):
            if r in (i, p[0]):
                continue
            rest = de.donors[i] - X[i] - F[i] * (X[p[0]] - X[r])
            # rest = F (x_pbest - x_i), x_pbest must be a population member
            x_pbest = X[i] + rest / F[i]
            if np.min(np.linalg.norm(X - x_pbest, axis=1)) < 1e-8:
                found = True
                break
        assert found


# ---------------------------------------------------------------- crossover


def prepared_for_crossover(crossover="BIN", eigen=False, CR=0.5, lambda0=20, dim=6):
    de = initialized(crossover=crossover, eigenvalue_crossover=eigen, lambda0=lambda0, dim=dim)
    de.sample_parameters()
    de.mutate()
    de.donors = de.population + 1000.0  # every coordinate distinguishable
    de.CR_used = np.full(lambda0, CR)
    de.crossover()
    return de


@pytest.mark.parametrize("CR, expected", [(0.0, 1), (1.0, 6)])
def test_binomial_extremes(CR, expected):
    de = prepared_for_crossover("BIN", CR=CR)
    from_donor = np.isclose(de.trials, de.donors).sum(axis=1)
    assert np.all(from_donor == expected)


def test_binomial_takes_at_least_one_coordinate():
    de = prepared_for_crossover("BIN", CR=0.3, lambda0=200)
    from_donor = np.isclose(de.trials, de.donors)
    assert np.all(from_donor.sum(axis=1) >= 1)
    assert abs(from_donor.mean() - (0.3 + 0.7 / 6)) < 0.03


@pytest.mark.parametrize("CR", (0.0, 0.5, 0.9))
def test_exponential_block_is_contiguous(CR):
    de = prepared_for_crossover("EXP", CR=CR, lambda0=100)
    mask = np.isclose(de.trials, de.donors)
    for row in mask:
        L = row.sum()
        assert L >= 1
        if L < len(row):  # exactly one cyclic run of True values
            changes = np.sum(row != np.roll(row, 1))
            assert changes == 2
    if CR == 0.0:
        assert np.all(mask.sum(axis=1) == 1)


def test_eigen_crossover_works_in_the_eigenbasis():
    """KI-16: with CR = 0, trial and target differ in exactly one eigen-coordinate."""
    de = prepared_for_crossover("BIN", eigen=True, CR=0.0, lambda0=30, dim=4)
    X = de.population
    _, B = np.linalg.eigh(np.cov(X, rowvar=False))
    Z = (de.trials - X) @ B
    assert np.all((np.abs(Z) > 1e-8).sum(axis=1) == 1)


def test_eigen_crossover_cr1_returns_donor():
    de = prepared_for_crossover("BIN", eigen=True, CR=1.0, lambda0=30, dim=4)
    np.testing.assert_allclose(de.trials, de.donors, atol=1e-8)


# ---------------------------------------------------------------- bound correction


def corrected(method, alpha=0.5, n=200, dim=3):
    de = initialized(bound_correction=method, hvb_alpha=alpha, lambda0=n, dim=dim)
    rng = np.random.default_rng(0)
    U = rng.uniform(-15, 15, size=(n, dim))  # many coordinates outside [-5, 5]
    de.trials = U
    de.correct_bounds()
    return de, U


ALL = ("SATURATE", "UNIF_RESAMPLE", "COTN", "TOROIDAL", "MIRROR", "HVB", "EXPC_TARGET", "EXPC_CENTER", "EXPS")


@pytest.mark.parametrize("method", ALL)
def test_corrections_stay_in_bounds_and_leave_feasible_coordinates(method):
    de, U = corrected(method)
    Uc = de.trials_corrected
    assert np.all((Uc >= -5) & (Uc <= 5))
    inside = (U >= -5) & (U <= 5)
    np.testing.assert_array_equal(Uc[inside], U[inside])
    assert de.corrected == list(np.any(~inside, axis=1))


def test_none_leaves_everything():
    de, U = corrected("NONE")
    np.testing.assert_array_equal(de.trials_corrected, U)
    assert not any(de.corrected)


def _normalised(x):
    return (x + 5) / 10


@pytest.mark.parametrize("method", ("SATURATE", "TOROIDAL", "MIRROR", "HVB"))
def test_deterministic_corrections_match_closed_forms(method):
    de, U = corrected(method, alpha=0.3)
    y, tau = _normalised(U), _normalised(de.population)
    b = (y > 1).astype(float)
    if method == "SATURATE":
        expected = b
    elif method == "TOROIDAL":
        expected = y - np.floor(y)
    elif method == "MIRROR":
        expected = np.abs(y - np.floor(y) - np.mod(np.floor(y), 2))
    else:
        expected = 0.3 * b + 0.7 * tau
    out = (y < 0) | (y > 1)
    np.testing.assert_allclose(_normalised(de.trials_corrected)[out], expected[out], atol=1e-12)


@pytest.mark.parametrize("method, ref", [("EXPC_TARGET", "parent"), ("EXPC_CENTER", "centre")])
def test_exponential_confined_lies_between_bound_and_reference(method, ref):
    de, U = corrected(method)
    y, yc = _normalised(U), _normalised(de.trials_corrected)
    r = _normalised(de.population) if ref == "parent" else np.full_like(y, 0.5)
    up, low = y > 1, y < 0
    assert np.all(yc[up] >= r[up] - 1e-12) and np.all(yc[up] <= 1)
    assert np.all(yc[low] <= r[low] + 1e-12) and np.all(yc[low] >= 0)


def test_cotn_never_leaves_the_domain():
    """KI-17: redrawn until inside."""
    de, _ = corrected("COTN", n=2000, dim=5)
    Uc = de.trials_corrected
    assert np.all((Uc >= -5) & (Uc <= 5))


# ---------------------------------------------------------------- selection / archive / adaptation


def selection_setup(**kw):
    de = initialized(lambda0=10, **kw)
    de.sample_parameters()
    de.mutate()
    de.crossover()
    de.trials = de.population.copy()  # in bounds: unchanged by correction
    de.correct_bounds()
    return de


def test_selection_ties_to_trial_and_success_record():
    de = selection_setup()
    old_X, old_f = de.population, de.fitness
    f_trial = old_f.copy()
    f_trial[:3] -= 1.0  # strict improvements
    f_trial[3] = old_f[3]  # tie -> trial wins, not a success
    f_trial[4:] += 1.0
    de.trial_fitness = f_trial
    de.select()
    rec = de.success
    assert rec["S"] == [0, 1, 2]
    assert rec["W"] == [True] * 4 + [False] * 6
    np.testing.assert_allclose(rec["delta_f"], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(rec["replaced"], old_X[:3])
    np.testing.assert_allclose(de.fitness, np.minimum(old_f, f_trial))


@pytest.mark.parametrize("cr_method", ("SHADE", "LSHADE"))
def test_shade_memory_update(cr_method):
    de = selection_setup(adaptation_method_F="SHADE", adaptation_method_CR=cr_method, memory_size=4)
    F = np.linspace(0.2, 0.9, 10)
    CR = np.linspace(0.1, 0.8, 10)
    de.F_used, de.CR_used = F, CR
    f_trial = de.fitness.copy()
    gains = np.array([1.0, 2.0, 3.0])
    f_trial[:3] -= gains
    de.trial_fitness = f_trial
    de.select()
    de.adapt()
    w = gains / gains.sum()
    assert de.memory_F[0] == pytest.approx(np.sum(w * F[:3] ** 2) / np.sum(w * F[:3]))  # KI-20
    cr = (np.sum(w * CR[:3] ** 2) / np.sum(w * CR[:3])) if cr_method == "LSHADE" else np.sum(w * CR[:3])
    assert de.memory_CR[0] == pytest.approx(cr)
    assert de.memory_index == 1


def test_lshade_terminal_value():
    de = selection_setup(adaptation_method_F="SHADE", adaptation_method_CR="LSHADE", memory_size=2)
    de.CR_used = np.zeros(10)
    f_trial = de.fitness.copy()
    f_trial[0] -= 1.0
    de.trial_fitness = f_trial
    de.select()
    de.adapt()
    assert np.isnan(de.memory_CR[0])
    # a terminal slot yields CR = 0 when sampled
    de.memory_CR = np.array([np.nan, np.nan])
    de.sample_parameters()
    assert np.all(de.CR_used == 0)


def test_jde_regeneration_rate_and_commit():
    de = initialized(adaptation_method_F="JDE", adaptation_method_CR="JDE", lambda0=4000, dim=2)
    de.sample_parameters()
    changed = np.mean(de.F_used != de.F)
    assert abs(changed - 0.1) < 0.02  # KI-2: probability tau, not 1 - tau
    F_before = de.F.copy()
    de.mutate()
    de.crossover()
    de.correct_bounds()
    f_trial = de.fitness.copy()
    f_trial[:2000] -= 1.0  # first half selected
    f_trial[2000:] += 1.0
    de.trial_fitness = f_trial
    de.select()
    np.testing.assert_array_equal(de.F[:2000], de.F_used[:2000])  # KI-22: kept on success
    np.testing.assert_array_equal(de.F[2000:], F_before[2000:])  # restored otherwise


def test_jso_caps_at_start():
    de = initialized(use_jso_caps=True, CR0=0.2, F0=0.9)
    de.sample_parameters()
    assert np.all(de.CR_used == 0.7) and np.all(de.F_used == 0.7)  # KI-21: CR floored


def test_archive_insertion_and_capacity():
    de = selection_setup(use_archive=True, archive_rate=0.5)
    f_trial = de.fitness - 1.0  # all succeed
    de.trial_fitness = f_trial
    de.select()
    de.adapt()
    assert de.archive_capacity == 5
    assert de.archive.shape == (5, 5)


# ---------------------------------------------------------------- oppositional generation


def test_oppositional_generation_keeps_best_of_union():
    de = initialized(oppositional_generation=True, oppositional_jump_rate=1.0, lambda0=12)
    X, f, F = de.population, de.fitness, de.F
    O = X.min(axis=0) + X.max(axis=0) - X  # KI-6: per-dimension bounds
    fO = np.array([sphere(o) for o in O])
    assert de.oppositional_generation(sphere)
    allf = np.concatenate([f, fO])
    keep = np.argsort(allf, kind="stable")[:12]
    np.testing.assert_allclose(de.population, np.vstack([X, O])[keep])
    np.testing.assert_allclose(de.F, np.concatenate([F, F])[keep])


# ---------------------------------------------------------------- initialisation


@pytest.mark.parametrize("factor", (0.5, 1.0))
def test_oversampling_evaluates_more_and_keeps_best(factor):
    rec = Recorder()
    de = make_de(lambda0=10, oversampling_factor=factor)
    de.initialize(rec)
    assert rec.calls == int(np.ceil(10 * (1 + factor)))  # KI-4
    best = np.sort([sphere(x) for x in rec.points])[:10]
    np.testing.assert_allclose(np.sort(de.fitness), best)


def test_opposite_initialisation():
    rec = Recorder()
    de = make_de(lambda0=10, oppositional_initialization="OPPOSITE")
    de.initialize(rec)
    pts = np.array(rec.points)
    np.testing.assert_allclose(pts[5:], -pts[:5])  # lb + ub - x with lb = -ub


@pytest.mark.parametrize("phi", (0.0, 0.5, 1.0))
def test_quasi_opposite_initialisation(phi):
    rec = Recorder()
    de = make_de(lambda0=40, oppositional_initialization="QUASI_OPPOSITE",
                 oppositional_initialization_factor=phi)
    de.initialize(rec)
    pts = np.array(rec.points)
    base, quasi = pts[:20], pts[20:]
    target = -phi * base  # c + phi (opposite - c), c = 0
    lo, hi = np.minimum(0, target), np.maximum(0, target)
    assert np.all((quasi >= lo - 1e-12) & (quasi <= hi + 1e-12))


def test_initial_population_with_and_without_fitness():
    P = np.random.default_rng(1).uniform(-5, 5, size=(15, 5))
    rec = Recorder()
    de = make_de(lambda0=10, initial_population=P)
    de.initialize(rec)
    assert rec.calls == 15
    rec2 = Recorder()
    de2 = make_de(lambda0=10, initial_population=P, initial_fitness=[sphere(p) for p in P])
    de2.initialize(rec2)
    assert rec2.calls == 0
    np.testing.assert_allclose(np.sort(de.fitness), np.sort(de2.fitness))
