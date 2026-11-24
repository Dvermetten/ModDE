"""Edge cases and API contracts found in the pre-release review (2026-11-24)."""
import warnings

import ioh
import numpy as np
import pytest

import modde
from modde import _core
from modde._settings import settings_from_dict, settings_to_dict


def sphere(x):
    return float(np.sum(np.asarray(x) ** 2))


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ---------------------------------------------------------------- non-finite objective values


def test_nan_is_treated_as_inf_and_warned_once():
    def f(x):
        return float("nan") if x[0] > 0 else sphere(x)

    with pytest.warns(RuntimeWarning, match="NaN") as record:
        res = modde.fmin(f, budget=3000, dim=3, lb=-5, ub=5, seed=1)
    assert sum("NaN" in str(w.message) for w in record) == 1
    assert not np.isnan(res.optimizer.fitness).any()
    assert np.all(res.optimizer.population[np.isfinite(res.optimizer.fitness), 0] <= 0)
    assert res.optimizer.core.stats.nan_evaluations > 0


def test_nan_first_individual_is_not_the_best():
    calls = [0]

    def f(x):
        calls[0] += 1
        return float("nan") if calls[0] == 1 else sphere(x)

    opt = quiet(modde.ModularDE, f, budget=500, dim=2, lb=-5, ub=5, seed=1, mutation_base="BEST")
    quiet(opt.run)
    assert np.isfinite(opt.best_f)


def test_infinite_gain_keeps_shade_memories_finite():
    calls = [0]

    def f(x):
        calls[0] += 1
        return float("inf") if calls[0] <= 3 else sphere(x)

    opt = modde.ModularDE(f, budget=2000, dim=5, lb=-5, ub=5, preset="SHADE", lambda0=20, seed=1)
    opt.run()
    assert np.isfinite(opt.memory_F).all() and np.isfinite(opt.memory_CR).all()


def test_unevaluated_trials_never_win_against_infinite_parents():
    de = _core.ModularDE(settings_from_dict(2, lb=-5, ub=5, seed=1, lambda0=6, budget=7))
    de.initialize(lambda x: float("inf"))
    before = de.population
    de.sample_parameters(); de.mutate(); de.crossover(); de.correct_bounds()
    de.evaluate(lambda x: float("inf"), False)  # only 1 of 6 trials fits the budget
    de.select()
    np.testing.assert_array_equal(de.population[1:], before[1:])


def test_infinite_bounds_rejected():
    with pytest.raises(ValueError, match="finite"):
        modde.ModularDE(sphere, budget=100, dim=2, lb=-np.inf, ub=1)


# ---------------------------------------------------------------- stopping


@pytest.mark.parametrize("seed", range(4))
def test_ioh_target_ends_the_generation(seed):
    count, hit = [0], [None]

    def f(x):
        count[0] += 1
        value = sphere(x)
        if value <= 1e-8 and hit[0] is None:
            hit[0] = count[0]
        return value

    problem = ioh.wrap_problem(f, f"sphere_target_{seed}", ioh.ProblemClass.REAL, dimension=2, lb=-5, ub=5,
                               calculate_objective=lambda iid, dim: ioh.RealSolution([0.0] * dim, 0.0))
    modde.ModularDE(problem, budget=20_000, preset="ODE", lambda0=20, oppositional_jump_rate=1.0,
                    seed=seed).run()
    assert problem.state.final_target_found and hit[0] is not None
    # only the rest of the batch that hit the target is evaluated (no jump, no next generation)
    assert problem.state.evaluations - hit[0] < 20


def test_reused_problem_that_reached_its_target_stops_immediately():
    problem = ioh.get_problem(1, 1, 2)
    modde.ModularDE(problem, budget=5000, seed=1).run()
    used = problem.state.evaluations
    with pytest.warns(UserWarning, match="already reached its final target"):
        opt = modde.ModularDE(problem, budget=5000, seed=2)
        opt.run()
    assert problem.state.evaluations == used


# ---------------------------------------------------------------- maximisation


def max_problem():
    return ioh.wrap_problem(lambda x: -float(np.sum(np.asarray(x) ** 2)), "neg_sphere", ioh.ProblemClass.REAL,
                            dimension=2, lb=-5, ub=5, optimization_type=ioh.OptimizationType.MAX)


def test_maximisation_target_in_problem_sign():
    p = max_problem()
    opt = modde.ModularDE(p, budget=5000, seed=1, target=-1e-3).run()
    assert opt.best_f >= -1e-3
    assert p.state.evaluations < 5000


def test_maximisation_fitness_in_problem_sign():
    p = max_problem()
    opt = modde.ModularDE(p, budget=300, seed=1).run()
    assert np.all(opt.fitness <= 0) and opt.best_f == pytest.approx(opt.fitness.max())


def test_maximisation_initial_fitness_in_problem_sign():
    p = max_problem()
    X = np.array([[1.0, 1.0], [0.1, 0.1], [2.0, 2.0], [3.0, 0.0]])
    f = np.array([-2.0, -0.02, -8.0, -9.0])
    opt = modde.ModularDE(p, budget=10, lambda0=4, initial_population=X, initial_fitness=f, seed=1)
    opt.initialize()
    assert opt.best_f == pytest.approx(-0.02)


# ---------------------------------------------------------------- state access


def test_assigning_state_reaches_the_core():
    opt = modde.ModularDE(sphere, budget=300, dim=3, lb=-5, ub=5, seed=1)
    opt.initialize()
    opt.population = np.zeros((opt.lambda_, 3))
    np.testing.assert_array_equal(opt.core.population, 0)
    opt.sample_parameters()
    opt.F_used = np.full(opt.lambda_, 0.123)
    np.testing.assert_allclose(opt.core.F_used, 0.123)


def test_read_only_state_cannot_be_assigned():
    opt = modde.ModularDE(sphere, budget=300, dim=3, lb=-5, ub=5, seed=1)
    with pytest.raises(AttributeError, match="read-only"):
        opt.archive = np.zeros((1, 3))
    with pytest.raises(AttributeError, match="read-only"):
        opt.best_f = 1.0


@pytest.mark.parametrize("name, value", [("fitness", np.ones(2)), ("F", np.ones(2)), ("CR_used", np.ones(3)),
                                         ("trial_fitness", np.ones(2)), ("memory_F", np.ones(99)),
                                         ("donors", np.ones((2, 3)))])
def test_wrong_shapes_are_rejected(name, value):
    opt = modde.ModularDE(sphere, budget=300, dim=3, lb=-5, ub=5, seed=1, preset="SHADE", lambda0=10)
    opt.initialize()
    with pytest.raises(ValueError, match=name):
        setattr(opt.core, name, value)


def test_unknown_attribute_message_names_the_class():
    at = modde.AskTellDE(dim=2, lb=-5, ub=5, budget=100)
    with pytest.raises(AttributeError, match="'AskTellDE' object has no attribute 'nonsense'"):
        at.nonsense


def test_asktell_has_the_tracked_attributes():
    at = modde.AskTellDE(dim=2, lb=-5, ub=5, budget=100, seed=1)
    at.tell([sphere(x) for x in at.ask()])
    assert at.generation == 0 and isinstance(at.corrected, float) and at.popstd > 0


# ---------------------------------------------------------------- conflicting inputs


def test_settings_cannot_be_combined():
    s = settings_from_dict(3, lb=-5, ub=5, budget=100)
    for extra in ({"budget": 999}, {"lb": -1}, {"preset": "SHADE"}, {"seed": 3}):
        with pytest.raises(ValueError, match="cannot be combined"):
            modde.ModularDE(sphere, settings=s, **extra)
        with pytest.raises(ValueError, match="cannot be combined"):
            modde.AskTellDE(settings=s, **extra)


def test_settings_dimension_must_match_the_problem():
    with pytest.raises(ValueError, match="dimension"):
        modde.ModularDE(ioh.get_problem(1, 1, 3), settings=settings_from_dict(5, lb=-5, ub=5))


def test_settings_with_a_plain_callable():
    s = settings_from_dict(3, lb=-5, ub=5, budget=200, seed=1)
    assert modde.ModularDE(sphere, settings=s).run().evaluations == 200


def test_preset_and_config_together_rejected():
    cfg = modde.get_presets(3)["DE"]
    with pytest.raises(ValueError, match="either `preset` or `config`"):
        modde.ModularDE(sphere, budget=100, dim=3, lb=-5, ub=5, preset="SHADE", config=cfg)


def test_vectorized_needs_wrap_false_for_plain_callables():
    with pytest.raises(ValueError, match="wrap=False"):
        modde.ModularDE(sphere, budget=100, dim=3, lb=-5, ub=5, vectorized=True)


def test_bounds_differing_from_the_problem_warn():
    with pytest.warns(UserWarning, match="differs from the problem's bounds"):
        modde.ModularDE(ioh.get_problem(1, 1, 2), budget=100, lb=-100)


def test_integer_arguments():
    assert modde.ModularDE(sphere, budget=2.7e2, dim=2, lb=-5, ub=5).core.settings.budget == 270
    with pytest.raises(ValueError, match="budget must be an integer"):
        modde.ModularDE(sphere, budget=10.5, dim=2, lb=-5, ub=5)
    with pytest.raises(ValueError, match="dim must be an integer"):
        modde.ModularDE(sphere, budget=10, dim=2.5, lb=-5, ub=5)
    with pytest.raises(ValueError, match="needs a budget"):
        modde.fmin(sphere, budget=None, dim=2, lb=-5, ub=5)


# ---------------------------------------------------------------- settings conversion


def test_settings_round_trip_with_user_only():
    opt = modde.ModularDE(sphere, budget=300, dim=3, lb=-5, ub=5, preset="L-SHADE")
    values = settings_to_dict(opt.settings, user_only=True)
    assert "dim" not in values and values["seed"] == opt.settings.seed
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        again = modde.ModularDE(sphere, dim=3, **values).run()
    assert again.best_f == opt.run().best_f


def test_full_settings_dict_can_be_passed_back():
    s = settings_from_dict(3, lb=-5, ub=5).resolve()
    settings_from_dict(3, **settings_to_dict(s))  # no "multiple values for dim"


@pytest.mark.parametrize("kw, message", [({"lambda_": 5}, "now use 'lambda0'"),
                                         ({"lambda0": 10.5}, "lambda0: expected a non-negative integer"),
                                         ({"use_archive": "maybe"}, "use_archive: expected True or False"),
                                         ({"F0": "x"}, "F0: expected a number")])
def test_readable_setting_errors(kw, message):
    with pytest.raises(ValueError, match=message):
        settings_from_dict(3, lb=-5, ub=5, **kw)


def test_bool_strings_and_whole_floats_accepted():
    s = settings_from_dict(3, lb=-5, ub=5, use_archive="true", lambda0=12.0)
    assert s.modules.use_archive is True and s.lambda0 == 12


def test_preset_names_case_insensitive():
    assert modde.preset_settings("l-shade", 10) == modde.preset_settings("L-SHADE", 10)


def test_archive_content_exported():
    assert modde.ArchiveContent.TRIALS.name == "TRIALS"
    assert "annotations" not in dir(modde)


# ---------------------------------------------------------------- warnings


def test_inactive_module_option_warns():
    with pytest.warns(UserWarning, match="archive_content has no effect"):
        modde.ModularDE(sphere, budget=100, dim=3, lb=-5, ub=5, archive_content="TRIALS")


def test_preset_values_do_not_cause_no_effect_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        modde.ModularDE(sphere, budget=100, dim=3, lb=-5, ub=5, preset="SHADE",
                        adaptation_method_F="NONE", adaptation_method_CR="NONE")


def test_derived_lambda_min_never_exceeds_lambda0():
    s = settings_from_dict(3, lb=-5, ub=5, lambda0=3, population_reduction="LINEAR",
                           mutation_base="TARGET", mutation_reference="BEST").resolve()
    assert s.lambda_min == 3


# ---------------------------------------------------------------- ask-tell


def test_tell_one_with_nan_completes_the_batch():
    at = modde.AskTellDE(dim=2, lb=-5, ub=5, budget=100, lambda0=4, seed=1)
    idx = [at.ask_one()[0] for _ in range(4)]
    with pytest.warns(RuntimeWarning, match="NaN"):
        for i in idx:
            at.tell_one(i, float("nan") if i == 0 else 1.0)
    assert at.evaluations == 4


def test_ask_one_state_is_cleared_by_tell():
    at = modde.AskTellDE(dim=2, lb=-5, ub=5, budget=100, lambda0=4, seed=1)
    at.ask_one()
    X = at.ask()
    at.tell([sphere(x) for x in X])
    i, x = at.ask_one()
    assert i == 0 and not any(np.allclose(x, y) for y in X)
