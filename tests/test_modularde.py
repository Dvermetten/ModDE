"""Module containing tests for ModularDE."""

import os
import shutil
import io
import json
import unittest
import unittest.mock

import numpy as np
import ioh

from modde import parameters, utils, ModularDE


class TestModularDEMeta(type):
    """Metaclass for generating test-cases."""

    def __new__(cls, name, bases, clsdict):
        """Method for generating new classes."""

        def make_test_fid(module, value, fid):
            return {
                f"test_{module}_{value}_f{fid}": lambda self: self.run_bbob_function(
                    module, value, fid
                )
            }

        def make_test_option(module, value):
            return {
                f"test_{module}_{value}": lambda self: self.run_module(module, value)
            }

        for module in parameters.Parameters.__modules__:
            m = getattr(parameters.Parameters, module)
            if type(m) == utils.AnyOf:
                for o in filter(None, m.options):
                    for fid in range(1, 25):
                        clsdict.update(make_test_fid(module, o, fid))
                    clsdict.update(make_test_option(module, o))

            elif type(m) == utils.InstanceOf:
                for fid in range(1, 25):
                    clsdict.update(make_test_fid(module, True, fid))

                clsdict.update(make_test_option(module, True))

        return super().__new__(cls, name, bases, clsdict)


class TestModularDE(unittest.TestCase, metaclass=TestModularDEMeta):
    """Test case for ModularDE Object. Gets applied for all Parameters.__modules__."""

    _dim = 2
    _budget = int(1e2 * _dim)

    def __init__(self, args, **kwargs):
        """Initializes the expected function value dictionary."""
        with open("tests/expected.json", "r") as f:
            self.bbob2d_per_module = json.load(f)
        super().__init__(args, **kwargs)

    def run_module(self, module, value):
        """Test a single run of the mechanism with a given module active."""
        self.p = parameters.Parameters(
            self._dim, budget=self._budget, seed=42, **{module: value}
        )
        self.c = ModularDE(ioh.get_problem(1, 1, self._dim), parameters=self.p).run()

    def run_bbob_function(self, module, value, fid):
        """Expects the output to be consistent with BBOB_2D_PER_MODULE_20_ITER."""
        np.random.seed(42)
        f = ioh.get_problem(fid, dimension=self._dim, instance=1)
        self.p = parameters.Parameters(
            self._dim, budget=self._budget, seed=42, **{module: value}
        )
        self.c = ModularDE(f, parameters=self.p).run()
        expected = self.bbob2d_per_module[f"{module}_{value}"][fid - 1]

        self.assertAlmostEqual(f.state.current_best_internal.y, expected)


if __name__ == "__main__":
    unittest.main()
