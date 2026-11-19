"""Consistency checks for the parameter registry (Phase 1, D-C3)."""
import pytest

from modde import _registry as R

DIMS = (1, 2, 5, 20, 100)


def test_names_unique():
    names = [p.name for p in R.PARAMS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("p", [p for p in R.PARAMS if p.kind == "categorical"], ids=lambda p: p.name)
def test_categorical_options_uppercase_and_unique(p):
    assert p.options and len(set(p.options)) == len(p.options)
    assert all(o == o.upper() for o in p.options)


@pytest.mark.parametrize("p", R.PARAMS, ids=lambda p: p.name)
def test_conditions_reference_existing_values(p):
    for c in p.active_when:
        parent = R.BY_NAME[c.parent]
        allowed = parent.options if parent.kind == "categorical" else (False, True)
        assert set(c.values) <= set(allowed), (p.name, c)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("p", [p for p in R.PARAMS if p.default is not None], ids=lambda p: p.name)
def test_defaults_valid_and_inside_tuning_range(p, dim):
    value = p.default_for(dim)
    valid = p.valid_for(dim)
    if valid is not None:
        lo, hi = valid
        assert lo is None or value >= lo
        assert hi is None or value <= hi
    tune = p.tune_for(dim)
    if tune is not None:
        assert tune[0] <= value <= tune[1], (p.name, dim, value, tune)


@pytest.mark.parametrize("p", [p for p in R.PARAMS if p.tune is not None], ids=lambda p: p.name)
def test_tunable_params_have_a_group(p):
    assert p.group in R.GROUPS


@pytest.mark.parametrize("dim", DIMS)
def test_configspace_builds_with_classic_de_default(dim):
    cs = R._build_configspace(dim)
    default = dict(cs.get_default_configuration())
    expected = R.module_defaults(dim)
    for name, value in default.items():
        assert value == expected[name], name
    assert default["mutation_base"] == "RAND" and default["crossover"] == "BIN"
    assert default["F0"] == 0.5 and default["CR0"] == 0.9 and default["lambda0"] == 10 * dim


@pytest.mark.parametrize("dim", (1, 5, 20))
def test_every_sampled_configuration_is_valid(dim):
    """D-C5: all configurations in the space must be valid (no forbidden clauses needed)."""
    cs = R._build_configspace(dim)
    cs.seed(dim)
    for config in cs.sample_configuration(300):
        c = dict(config)
        needed = R.n_parents_needed(c)
        assert c["lambda0"] >= needed + 1, c
        if c["population_reduction"] == "LINEAR":
            assert R.derive_defaults(c)["lambda_min"] <= c["lambda0"]


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("name", list(R.PRESETS))
def test_presets_are_valid_configurations(name, dim):
    cs = R._build_configspace(dim)
    preset = R.PRESETS[name].settings(dim)
    configuration = R._preset_configuration(name, dim, cs)
    configuration.check_valid_configuration()
    config = dict(configuration)
    for key, value in preset.items():
        assert key in R.BY_NAME, key
        if key in config:
            assert config[key] == value, (key, config[key], value)
        else:
            # Not part of the ConfigSpace: must equal what resolution would derive anyway.
            assert key in R.derive_defaults({**config}), key
            assert R.derive_defaults({**config})[key] == value, key
    # Every preset value that is in the space must be active (not silently dropped)
    for key in preset:
        if any(hp.name == key for hp in cs.values()):
            assert key in config, f"{name}: {key} is inactive in the configuration"


def test_parents_needed():
    assert R.n_parents_needed({}) == 3  # DE/rand/1
    assert R.n_parents_needed({"mutation_base": "TARGET", "mutation_reference": "PBEST"}) == 2
    assert R.n_parents_needed({"mutation_base": "RAND", "mutation_reference": "RAND",
                               "mutation_n_comps": 2}) == 6
    assert R.derive_defaults({})["lambda_min"] == 4
