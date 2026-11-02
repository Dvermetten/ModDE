"""Tests for bound handling and oppositional generation."""

import unittest

import numpy as np
import ioh

from modde import ModularDE, Parameters


class TestBounds(unittest.TestCase):
    """Bounds are taken from the problem unless given explicitly."""

    _dim = 3

    def make_problem(self, name, lb, ub):
        self.evaluated = []

        def f(x):
            self.evaluated.append(np.array(x))
            return float(np.sum(np.asarray(x) ** 2))

        ioh.wrap_problem(f, name, ioh.ProblemClass.REAL, self._dim, lb=lb, ub=ub)
        return ioh.get_problem(name, dimension=self._dim)

    def test_bounds_from_problem(self):
        problem = self.make_problem("test_bounds_from_problem", 0, 10)
        de = ModularDE(problem, budget=200, seed=1)
        de.run()
        self.assertTrue(np.all(de.parameters.lb == 0))
        self.assertTrue(np.all(de.parameters.ub == 10))
        x = np.array(self.evaluated)
        self.assertTrue(np.all((x >= 0) & (x <= 10)))

    def test_bounds_from_problem_with_parameters(self):
        problem = self.make_problem("test_bounds_from_problem_params", 0, 10)
        de = ModularDE(problem, parameters=Parameters(self._dim, budget=200, seed=1))
        self.assertTrue(np.all(de.parameters.lb == 0))
        self.assertTrue(np.all(de.parameters.ub == 10))

    def test_explicit_bounds_take_precedence(self):
        problem = self.make_problem("test_explicit_bounds", 0, 10)
        de = ModularDE(
            problem, budget=200, seed=1, lb=np.full(self._dim, 1.0), ub=np.full(self._dim, 2.0)
        )
        de.run()
        self.assertEqual(de.parameters.lb.shape, (self._dim, 1))
        x = np.array(self.evaluated)
        self.assertTrue(np.all((x >= 1) & (x <= 2)))

    def test_default_bounds_without_problem_bounds(self):
        p = Parameters(self._dim)
        self.assertTrue(np.all(p.lb == -5))
        self.assertTrue(np.all(p.ub == 5))

    def test_invalid_bound_size(self):
        with self.assertRaises(ValueError):
            Parameters(self._dim, lb=np.zeros(self._dim + 1))


class TestOppositionalGeneration(unittest.TestCase):
    """Oppositional generation mirrors per dimension and keeps F/CR aligned."""

    _dim = 5

    def test_opposites_within_population_range(self):
        de = ModularDE(ioh.get_problem(1, 1, self._dim), budget=1000, seed=1)
        x = de.parameters.population.x.copy()
        lb, ub = x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True)
        evaluated = []
        de.fitness_func = lambda x_: evaluated.append(x_) or 0.0
        de.oppositional_generation()
        opposites = np.array(evaluated).T
        self.assertTrue(np.all((opposites >= lb) & (opposites <= ub)))

    def test_lshade_with_opposition(self):
        def run(seed):
            problem = ioh.get_problem(1, 1, self._dim)
            de = ModularDE(
                problem, budget=2000, seed=seed, mutation_base="target",
                mutation_reference="pbest", lpsr=True, use_archive=True,
                adaptation_method_F="shade", adaptation_method_CR="shade",
                oppositional_generation_probability=0.5,
            )
            while de.step():
                n = de.parameters.population.n
                self.assertEqual(len(de.parameters.F), n)
                self.assertEqual(len(de.parameters.CR), n)
            return problem.state.evaluations, problem.state.current_best.y

        self.assertEqual(run(7), run(7))


if __name__ == "__main__":
    unittest.main()
