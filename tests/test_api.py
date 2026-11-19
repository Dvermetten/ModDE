"""Public Python API (Phase 3): fmin, ModularDE with ioh, ConfigSpace, presets, ask-tell."""
import warnings

import ioh
import numpy as np
import pytest

import modde
from conftest import sphere, sphere_batch


# ---------------------------------------------------------------- fmin / ModularDE


def test_fmin_plain_callable_is_wrapped_and_counted():
    x, f, evals, opt = modde.fmin(sphere, budget=3000, dim=4, lb=-5, ub=5, seed=1)
    assert evals == 3000 == opt.problem.state.evaluations  # D-B3: the problem counts
    assert f == pytest.approx(sphere(x)) and f < 1e-2
    assert type(opt.problem).__name__.endswith("WrappedProblem")


def test_fmin_without_wrapping_vectorized():
    res = modde.fmin(sphere_batch, budget=2000, dim=3, lb=-5, ub=5, seed=1, wrap=False, vectorized=True)
    assert res.evaluations == 2000 and res.f < 1e-2


def test_plain_callable_requires_bounds():
    with pytest.raises(ValueError, match="dim, lb and ub"):
        modde.fmin(sphere, budget=100)


def test_ioh_problem_bounds_target_and_provenance():
    p = ioh.get_problem(1, 1, 5)
    opt = modde.ModularDE(p, budget=200_000, preset="L-SHADE", seed=2).run()
    assert p.state.final_target_found and p.state.evaluations < 200_000
    assert opt.settings.provenance["lb"] == modde.Provenance.PROBLEM
    np.testing.assert_array_equal(opt.settings.lb, p.bounds.lb)


def test_budget_is_exact_on_ioh_problem():
    p = ioh.get_problem(15, 1, 3)
    modde.ModularDE(p, budget=777, lambda0=10, seed=1).run()
    assert p.state.evaluations == 777


def test_reused_problem_warns_and_total_respects_budget():
    p = ioh.get_problem(2, 1, 3)
    for _ in range(5):
        p(np.zeros(3))
    with pytest.warns(UserWarning, match="already has 5 evaluations"):
        modde.ModularDE(p, budget=300, seed=1).run()
    assert p.state.evaluations == 300


def test_maximisation_problem():
    p = ioh.wrap_problem(lambda x: 3.0 - float(np.sum(np.asarray(x) ** 2)), "modde_test_max",
                         ioh.ProblemClass.REAL, dimension=3, lb=-5, ub=5,
                         optimization_type=ioh.OptimizationType.MAX)
    opt = modde.ModularDE(p, budget=4000, seed=1).run()
    assert opt.best_f == pytest.approx(p.state.current_best.y) and opt.best_f > 2.99


def test_resolution_warnings_are_python_warnings():
    with pytest.warns(UserWarning, match="memory_size has no effect"):
        modde.ModularDE(sphere, budget=100, dim=2, lb=-1, ub=1, memory_size=10)


def test_logger_can_watch_tracked_attributes(tmp_path):
    p = ioh.get_problem(3, 1, 4)
    logger = ioh.logger.Analyzer(root=str(tmp_path), folder_name="run", algorithm_name="modde")
    opt = modde.ModularDE(p, budget=300, seed=1, track_evaluations=True)
    logger.watch(opt, ["curr_F", "corrected", "popstd"])
    p.attach_logger(logger)
    opt.run()
    p.detach_logger()
    logger.close()
    header = next(tmp_path.rglob("*.dat")).read_text().splitlines()[0]
    assert header.split() == ["evaluations", "raw_y", "curr_F", "corrected", "popstd"]


def test_split_step_api_through_wrapper():
    opt = modde.ModularDE(sphere, budget=500, dim=3, lb=-5, ub=5, seed=1)
    opt.initialize()
    opt.sample_parameters()
    opt.mutate()
    opt.crossover()
    opt.correct_bounds()
    opt.evaluate()
    opt.select()
    opt.adapt()
    assert opt.generation == 1 and opt.evaluations == 2 * opt.lambda_


def test_removed_names_point_to_legacy():
    with pytest.raises(AttributeError, match="modde.legacy.Parameters"):
        modde.Parameters  # noqa: B018
    assert modde.ModularDE.__module__ == "modde.optimizer"


# ---------------------------------------------------------------- ConfigSpace / presets


def test_configspace_groups():
    full = modde.get_configspace(5)
    modules_only = modde.get_configspace(5, add_popsize=False, add_initial_FCR=False, add_module_params=False)
    names = {hp.name for hp in modules_only.values()}
    assert "lambda0" not in names and "F0" not in names and "archive_rate" not in names
    assert "mutation_base" in names
    assert full["lambda0"].upper == max(200, 250)


def test_settings_config_roundtrip():
    cs = modde.get_configspace(6)
    cs.seed(1)
    for config in cs.sample_configuration(20):
        settings = modde.settings_from_config(6, config, lb=-5, ub=5)
        assert dict(modde.config_from_settings(settings, cs)) == dict(config)


def test_presets_are_configurations_and_run():
    presets = modde.get_presets(4)
    assert set(presets) == set(modde.preset_names())
    for name, config in presets.items():
        res = modde.fmin(sphere, budget=1500, dim=4, lb=-5, ub=5, seed=1, config=config)
        assert np.isfinite(res.f), name


def test_preset_and_keyword_override():
    opt = modde.ModularDE(sphere, budget=100, dim=3, lb=-5, ub=5, preset="L-SHADE", memory_size=9)
    assert opt.settings.memory_size == 9 and opt.settings.lambda0 == 54


def test_unknown_preset():
    with pytest.raises(KeyError, match="unknown preset"):
        modde.preset_settings("NOPE", 3)


# ---------------------------------------------------------------- ask-tell


PARITY_CONFIGS = [
    {},
    {"preset": "L-SHADE"},
    {"preset": "ODE"},
    {"preset": "jDE", "oversampling_factor": 0.5},
    {"mutation_n_comps": 2, "crossover": "EXP", "bound_correction": "EXPC_TARGET",
     "oppositional_initialization": "QUASI_OPPOSITE", "oppositional_generation": True,
     "oppositional_jump_rate": 0.5, "use_archive": True},
]


@pytest.mark.parametrize("cfg", PARITY_CONFIGS, ids=str)
@pytest.mark.parametrize("one_by_one", (False, True))
def test_ask_tell_matches_run(cfg, one_by_one):
    """D-D12: same seed -> identical trajectory as run()."""
    common = dict(dim=4, lb=-5.0, ub=5.0, budget=1500, seed=7, **cfg)
    ref = modde.ModularDE(sphere_batch, wrap=False, vectorized=True, **common).run()

    at = modde.AskTellDE(**common)
    while not at.stop:
        if one_by_one:
            i, x = at.ask_one()
            at.tell_one(i, sphere(x))
        else:
            X = at.ask()
            at.tell(sphere_batch(X))
    np.testing.assert_array_equal(at.population, ref.population)
    assert at.evaluations == ref.evaluations == 1500
    assert at.best_f == ref.best_f


def test_ask_is_idempotent_and_tell_checks_length():
    at = modde.AskTellDE(3, -1, 1, budget=100, seed=1)
    X1, X2 = at.ask(), at.ask()
    np.testing.assert_array_equal(X1, X2)
    with pytest.raises(ValueError, match="expected"):
        at.tell([1.0])
    with pytest.raises(RuntimeError):
        modde.AskTellDE(3, -1, 1).tell([1.0])


def test_ask_after_finish_raises_stopiteration():
    at = modde.AskTellDE(2, -1, 1, budget=40, lambda0=10, seed=1)
    while not at.stop:
        at.tell(sphere_batch(at.ask()))
    with pytest.raises(StopIteration):
        at.ask()


def test_package_ships_type_information():
    from importlib import resources
    files = {p.name for p in resources.files("modde").iterdir()}
    assert {"py.typed", "_core.pyi"} <= files
