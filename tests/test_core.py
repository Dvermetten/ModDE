"""Phase 0 checks for the compiled extension and the package layout."""
import importlib
import sys
import warnings

import numpy as np
import pytest

import modde
from modde import _core


def test_version_is_set():
    assert isinstance(modde.__version__, str) and modde.__version__


def test_build_info():
    info = _core.build_info()
    assert info["eigen"].startswith("3.")
    assert info["compiler"] != "unknown"
    # C++17 or newer (MSVC reports the language level in _MSVC_LANG)
    assert max(info["cplusplus"], info.get("msvc_lang", 0)) >= 201703


def test_individual_norms_n_by_d():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, 3))  # n = 7 individuals, d = 3
    np.testing.assert_allclose(_core.individual_norms(X), np.linalg.norm(X, axis=1))


def test_individual_norms_accepts_non_contiguous():
    X = np.arange(24, dtype=float).reshape(6, 4)
    view = X[::2, ::2]  # non-contiguous view
    np.testing.assert_allclose(_core.individual_norms(view), np.linalg.norm(view, axis=1))


def test_legacy_import_warns():
    sys.modules.pop("modde.legacy", None)
    with pytest.warns(DeprecationWarning, match="modde.legacy"):
        importlib.import_module("modde.legacy")


def test_top_level_legacy_name_warns_and_forwards():
    with pytest.warns(DeprecationWarning, match="modde.legacy.ModularDE"):
        cls = modde.ModularDE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from modde.legacy import ModularDE
    assert cls is ModularDE


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        modde.does_not_exist  # noqa: B018
