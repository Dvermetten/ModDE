"""Logging validation (dev/03_module_validation.md §5.3a, D-B4).

Every value reported for an evaluation must belong to the individual being
evaluated, in every evaluation mode, including the evaluations outside the
normal generation order (initial candidates, oppositional jumps).
"""
import ioh
import numpy as np
import pytest

import modde

CONFIGS = {
    "jDE + mirror": dict(adaptation_method_F="JDE", adaptation_method_CR="JDE", bound_correction="MIRROR"),
    "opposition everywhere": dict(oppositional_initialization="QUASI_OPPOSITE", oppositional_generation=True,
                                  oppositional_jump_rate=1.0, bound_correction="EXPC_TARGET"),
    "L-SHADE": dict(preset="L-SHADE"),
}


def rastrigin(x):
    x = np.asarray(x, dtype=float)
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


class Checker:
    """Objective that checks the reported statistics against the core state."""

    def __init__(self):
        self.opt = None
        self.calls = 0
        self.kinds = {"init": 0, "trial": 0, "opposite": 0}
        self.reported = []

    def __call__(self, x):
        x = np.asarray(x, dtype=float).ravel()
        core = self.opt.core
        st = core.stats
        if not core.initialized:
            settings = core.settings
            assert st.curr_F == settings.F0 and st.curr_CR == settings.CR0 and not st.corrected
            self.kinds["init"] += 1
        else:
            T = core.trials_corrected
            hit = np.flatnonzero(np.all(T == x, axis=1))
            if len(hit):
                i = hit[0]
                assert st.curr_F == core.F_used[i]
                assert st.curr_CR == core.CR_used[i]
                assert st.corrected == core.corrected[i]
                self.kinds["trial"] += 1
            else:
                O = core.opposites
                i = np.flatnonzero(np.all(O == x, axis=1))[0]  # must be an opposite point
                assert st.curr_F == core.F[i] and st.curr_CR == core.CR[i] and not st.corrected
                self.kinds["opposite"] += 1
        self.calls += 1
        self.reported.append((st.curr_F, float(st.corrected)))
        return rastrigin(x)


@pytest.mark.parametrize("name", list(CONFIGS))
@pytest.mark.parametrize("vectorized", (False, True))
def test_reported_values_belong_to_the_evaluated_individual(name, vectorized):
    checker = Checker()
    f = (lambda X: np.array([checker(x) for x in np.atleast_2d(X)])) if vectorized else checker
    opt = modde.ModularDE(f, budget=1200, dim=4, lb=-5, ub=5, seed=1, wrap=False, vectorized=vectorized,
                          track_evaluations=True, **CONFIGS[name])
    checker.opt = opt
    opt.run()
    assert checker.calls == opt.evaluations == 1200
    assert checker.kinds["trial"] > 0
    if "opposition" in name:
        assert checker.kinds["opposite"] > 0


def test_ioh_log_matches_the_evaluations(tmp_path):
    """Logging every evaluation: one row per evaluation, watched values as seen by the objective."""
    checker = Checker()
    problem = modde.wrap_problem(checker, dim=3, lb=-5, ub=5, name="rastrigin")
    logger = ioh.logger.Analyzer(triggers=[ioh.logger.trigger.ALWAYS], root=str(tmp_path),
                                 folder_name="run", algorithm_name="modde")
    opt = modde.ModularDE(problem, budget=600, seed=2, track_evaluations=True,
                          **CONFIGS["opposition everywhere"])
    checker.opt = opt
    logger.watch(opt, ["curr_F", "corrected"])
    problem.attach_logger(logger)
    opt.run()
    problem.detach_logger()
    logger.close()

    rows = [line.split() for line in next(tmp_path.rglob("*.dat")).read_text().splitlines()]
    header, data = rows[0], [r for r in rows[1:] if r and r[0] != "evaluations"]
    assert header == ["evaluations", "raw_y", "curr_F", "corrected"]
    assert len(data) == 600 == problem.state.evaluations == checker.calls
    logged = np.array([[float(r[2]), float(r[3])] for r in data])
    np.testing.assert_allclose(logged, np.array(checker.reported), atol=1e-9)


def test_batch_mode_counts_match():
    """Without tracking, a vectorised objective is called in batches; counts still match exactly."""
    calls = []
    f = lambda X: (calls.append(len(X)), np.array([rastrigin(x) for x in X]))[1]  # noqa: E731
    opt = modde.ModularDE(f, budget=1001, dim=3, lb=-5, ub=5, seed=3, wrap=False, vectorized=True,
                          oppositional_generation=True, oppositional_jump_rate=0.5).run()
    assert sum(calls) == opt.evaluations == 1001
    assert max(calls) > 1  # really batched


def test_ask_tell_counts_match():
    at = modde.AskTellDE(3, -5, 5, budget=777, seed=4, oppositional_initialization="OPPOSITE",
                         oppositional_generation=True, oppositional_jump_rate=0.5)
    told = 0
    while not at.stop:
        X = at.ask()
        at.tell([rastrigin(x) for x in X])
        told += len(X)
    assert told == at.evaluations == 777
