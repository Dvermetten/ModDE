"""C++ settings resolution against the registry and spec/settings.md."""
import numpy as np
import pytest

from modde import _core
from modde import _registry as R
from modde._settings import make_settings

P = _core.Provenance


def resolved(dim=5, **kw):
    kw.setdefault("lb", -5.0)
    kw.setdefault("ub", 5.0)
    return make_settings(dim, **kw).resolve()


@pytest.mark.parametrize("p", [p for p in R.PARAMS if p.name != "dim"], ids=lambda p: p.name)
def test_every_registry_param_exists_in_core(p):
    target = _core.Modules() if (p.is_module and p.name != "track_evaluations") or p.name == "mutation_n_comps" \
        else _core.Settings(1)
    assert hasattr(target, p.name), p.name


@pytest.mark.parametrize("dim", (1, 5, 20))
def test_core_defaults_match_registry(dim):
    s = resolved(dim)
    for p in R.PARAMS:
        if p.name in ("dim", "seed"):
            continue
        if p.kind == "categorical":
            assert getattr(s.modules, p.name).name == p.options[0], p.name
        elif p.name == "mutation_n_comps":
            assert s.modules.mutation_n_comps == p.default
        elif p.kind == "bool":
            container = s if p.name == "track_evaluations" else s.modules
            assert getattr(container, p.name) == p.default, p.name
        elif p.default is not None:
            assert getattr(s, p.name) == pytest.approx(p.default_for(dim)), p.name
    assert s.lambda_min == R.derive_defaults({})["lambda_min"]


@pytest.mark.parametrize("kw", [
    {}, {"mutation_n_comps": 2}, {"mutation_reference": "RAND"},
    {"mutation_base": "TARGET", "mutation_reference": "PBEST"},
    {"mutation_base": "BEST", "mutation_n_comps": 0},
])
def test_derived_rules_match_registry(kw):
    s = resolved(10, **kw)
    assert s.n_parents_needed() == R.n_parents_needed(kw)
    assert s.lambda_min == R.derive_defaults(kw)["lambda_min"]


def test_provenance():
    s = make_settings(5, lb=-5.0, ub=5.0, lambda0=30)
    s.problem_fields = {"lb", "ub"}
    r = s.resolve()
    assert r.provenance["lambda0"] == P.USER
    assert r.provenance["budget"] == P.DEFAULT
    assert r.provenance["lambda_min"] == P.DERIVED
    assert r.provenance["lb"] == P.PROBLEM
    assert r.resolved and r.seed is not None


@pytest.mark.parametrize("kw, match", [
    ({"lb": None}, "required"),
    ({"lb": 5.0, "ub": 5.0}, "strictly smaller"),
    ({"mutation_n_comps": 2, "lambda0": 5}, "too small"),
    ({"population_reduction": "LINEAR", "lambda0": 10, "lambda_min": 20}, "larger than lambda0"),
    ({"budget": 10, "lambda0": 50}, "budget"),
    ({"adaptation_method_F": "SHADE", "F0": 0.0}, "F0 > 0"),
    ({"CR0": 1.5}, "valid range"),
    ({"initial_population": np.zeros((3, 5)), "lambda0": 10}, "fewer rows"),
    ({"initial_population": np.full((10, 5), 9.0), "lambda0": 10}, "outside the bounds"),
])
def test_resolution_errors(kw, match):
    kw = {"lb": -5.0, "ub": 5.0, **kw}
    with pytest.raises(ValueError, match=match):
        make_settings(5, **kw).resolve()


def test_lambda0_boundary_is_valid():
    assert resolved(5, mutation_n_comps=2, lambda0=6).lambda0 == 6


@pytest.mark.parametrize("kw, fragment", [
    ({"memory_size": 10}, "memory_size has no effect"),
    ({"pbest_rate": 0.2}, "pbest_rate has no effect"),
    ({"use_archive": True, "mutation_n_comps": 0, "mutation_base": "TARGET"}, "use_archive has no effect"),
    ({"mutation_use_weighted_F": True}, "mutation_use_weighted_F has no effect"),
])
def test_warnings_for_settings_without_effect(kw, fragment):
    assert any(fragment in w for w in resolved(5, **kw).warnings)


def test_no_warnings_by_default():
    assert list(resolved(5).warnings) == []


def test_unknown_option_and_setting():
    with pytest.raises(ValueError, match="unknown option"):
        make_settings(2, crossover="TRINOMIAL")
    with pytest.raises(ValueError, match="unknown setting"):
        make_settings(2, popsize=10)


def test_user_values_are_never_changed():
    s = resolved(5, lambda0=37, archive_rate=2.6, memory_size=11, use_archive=True,
                 adaptation_method_F="SHADE", population_reduction="LINEAR")
    assert (s.lambda0, s.archive_rate, s.memory_size) == (37, 2.6, 11)
