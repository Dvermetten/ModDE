"""irace export (D-C10)."""
import re
import shutil
import subprocess

import pytest

import modde
from modde import irace


@pytest.fixture
def cs():
    return modde.get_configspace(5)


def test_parameter_file_lists_every_hyperparameter(cs):
    text = irace.irace_parameters(cs)
    lines = [l for l in text.splitlines() if not l.startswith("#")]
    assert len(lines) == len(list(cs.values()))
    by_name = {l.split()[0]: l for l in lines}
    assert re.search(r'"--lambda0 "\s+i,log\s+\(7, 250\)$', by_name["lambda0"])
    assert re.search(r'"--F0 "\s+r\s+\(0.1, 1.5\)$', by_name["F0"])
    assert "c      (RAND, BEST, TARGET)" in by_name["mutation_base"]
    assert by_name["archive_rate"].endswith('| use_archive == "True"')
    assert by_name["pbest_rate"].endswith('| pbest_sampling == "FIXED"')
    assert '|| (adaptation_method_CR %in% c("SHADE", "LSHADE"))' in by_name["memory_size"]


def test_configurations_file(cs):
    configs = {"default": cs.get_default_configuration(), **modde.get_presets(5, cs)}
    text = irace.irace_configurations(configs, cs)
    rows = [l for l in text.splitlines() if not l.startswith("#")]
    header = rows[0].split()
    assert header == [hp.name for hp in cs.values()]
    assert len(rows) == 1 + len(configs)
    default = dict(zip(header, rows[1].split()))
    assert default["mutation_base"] == "RAND" and default["archive_rate"] == "NA"


def test_round_trip_through_irace_arguments(cs):
    cs.seed(3)
    for config in cs.sample_configuration(50):
        args = []
        for name, value in dict(config).items():
            args += [f"--{name}", str(value)]
        assert dict(irace.config_from_irace(args, cs)) == dict(config)


def test_argument_string_and_equals_syntax(cs):
    config = irace.config_from_irace("--mutation_base=TARGET --use_archive True --archive_rate 2.5", cs)
    assert config["mutation_base"] == "TARGET" and config["archive_rate"] == 2.5
    with pytest.raises(ValueError, match="inactive"):
        irace.config_from_irace("--archive_rate 2.5", cs)  # use_archive defaults to False


def test_write_files(tmp_path):
    paths = irace.write_irace_files(tmp_path, dim=3)
    assert paths["parameters"].exists() and paths["configurations"].exists()
    assert "L-SHADE" in paths["configurations"].read_text()


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R / irace not installed")
def test_irace_reads_the_files(tmp_path):
    paths = irace.write_irace_files(tmp_path, dim=4)
    code = (f'library(irace); p <- readParameters("{paths["parameters"].as_posix()}"); '
            f'c <- readConfigurationsFile("{paths["configurations"].as_posix()}", p); cat(nrow(c))')
    out = subprocess.run(["Rscript", "-e", code], capture_output=True, text=True)
    if "there is no package called" in out.stderr:
        pytest.skip("irace R package not installed")
    assert out.returncode == 0, out.stderr
