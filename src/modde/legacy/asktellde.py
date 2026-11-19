"""Ask and tell interface to the Modular DE."""
import warnings
import typing
from collections import deque
from functools import wraps
import numpy as np
from .modularde import ModularDE, Parameters, Population
from itertools import islice
from typing import List, Callable


def check_break_conditions(f: typing.Callable) -> typing.Callable:
    """Decorator function, checks for break conditions for the ~AskTellDE.

    Raises a StopIteration if break_conditions are met for ~AskTellDE.

    Parameters
    ----------
    f: callable
        A method on ~AskTellDE

    Raises
    ------
    StopIteration
        When any(~AskTellDE.break_conditions) == True

    """

    @wraps(f)
    def inner(self, *args, **kwargs) -> typing.Any:
        if any(self.break_conditions):
            raise StopIteration(
                "Break conditions reached, ignoring call to: " + f.__qualname__
            )
        return f(self, *args, **kwargs)

    return inner


class AskTellDE(ModularDE):
    """Ask tell interface for the ModularCMAES."""

    def __init__(self, dim, *args, **kwargs) -> None:
        """Override the fitness_function argument with an empty callable."""
        p = Parameters(dim, *args, **kwargs)
        if not hasattr(self, "ask_queue"):
            self.ask_queue = deque()
        super().__init__(lambda: None, parameters=p, *args, **kwargs)

    def fitness_func(self, x: np.ndarray) -> None:
        """Overwrite function call for fitness_func, calls register_individual."""
        self.register_individual(x)

    def step(self):
        """Method is disabled on this interface.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError("Step is undefined in this interface")

    def run(self):
        """Method is disabled on this interface.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError("Run is undefined in this interface")

    def register_individual(self, x: np.ndarray) -> None:
        """Add new individuals to self.ask_queue.

        Parameters
        ----------
        x: np.ndarray
            The vector to be added to the ask_queue

        """
        self.ask_queue.append(x.reshape(-1, 1))

    @property
    def break_conditions(self) -> List[bool]:
        return [self.parameters.used_budget >= self.parameters.budget]

    def initialize_population(self) -> None:
        n_individuals = self.parameters.lambda_
        if self.parameters.oversampling_factor > 0:
            n_individuals = int(
                n_individuals * (1 + self.parameters.oversampling_factor)
            )
        # x = np.hstack(tuple(islice(self.parameters.sampler, n_individuals)))
        # x = self.parameters.lb + x * (self.parameters.ub - self.parameters.lb)
        if self.parameters.oppositional_initialization:
            x1 = self.parameters.sampler(int(np.ceil(n_individuals / 2)))
            x2 = self.parameters.lb.reshape(-1) + (self.parameters.ub.reshape(-1) - x1)
            x = np.hstack([x1, x2])[:n_individuals]
        else:
            x = self.parameters.sampler(n_individuals)
        x = np.transpose(x)
        f = np.empty(self.parameters.lambda_, object)
        for i in range(self.parameters.lambda_):
            self.register_individual(x[:, i])
        self.parameters.population = Population(
            x, np.array([None for _ in range(n_individuals)])
        )
        self.pop_initialized = False

    def finalize_initialization(self) -> None:
        idxs_best = np.argsort(self.parameters.population.f)[: self.parameters.lambda_]

        self.parameters.population = Population(
            self.parameters.population.x[:, idxs_best],
            self.parameters.population.f[idxs_best],
        )

        if self.parameters.use_archive:
            self.parameters.archive = self.parameters.population
        if self.parameters.init_stats:
            self.track_stats()
        self.pop_initialized = True

    def bound_correction(self):
        new_x = self.correct_bounds()
        for i in range(new_x.shape[1]):
            self.register_individual(new_x[:, i])
        #         print(new_x)
        # f = np.empty(new_x.shape[1], object)
        # for i in range(new_x.shape[1]):
        #    f[i] = self.fitness_func(new_x[:, i])
        self.parameters.offspring = Population(
            new_x, np.array([None for _ in range(new_x.shape[1])])
        )

    @check_break_conditions
    def ask(self) -> np.ndarray:
        """Retrieve the next indivual from the ask_queue.

        If the ask_queue is not defined yet, it is defined and mutate is
        called in order to fill it.

        Returns
        -------
        np.ndarray

        """
        if not hasattr(self, "ask_queue"):
            self.ask_queue = deque()
            self.initialize_population()
        if len(self.ask_queue) == 0:
            warnings.warn("Asking from empty queue, returning None")
            return
        return self.ask_queue.popleft().reshape(-1)

    @check_break_conditions
    def tell(self, xi: np.ndarray, fi: float) -> None:
        """Process a provided fitness value fi for a given individual xi.

        Parameters
        ----------
        xi: np.ndarray
            An individual previously returned by ask()
        fi: float
            The fitness value for xi
        Raises
        ------
        RuntimeError
            When ask() is not called before tell()
        ValueError
            When an unknown xi is provided to the method

        Warns
        -----
        UserWarning
            When the same xi is provided more than once

        """
        # pylint: disable=singleton-comparison
        # if (not self.parameters.population) and (not self.parameters.offspring):
        #    raise RuntimeError("Call to tell without calling ask first is prohibited")
        xi = xi.reshape((-1, 1))
        if self.pop_initialized:
            indices, *_ = np.where((self.parameters.offspring.x == xi).all(axis=0))
            if len(indices) == 0:
                raise ValueError("Unkown xi provided")

            for index in indices:
                if self.parameters.offspring.f[index] == None:  # noqa
                    self.parameters.offspring.f[index] = fi
                    break
            else:
                warnings.warn("Repeated call to tell with same xi", UserWarning)
                self.parameters.offspring.f[index] = fi

        else:
            indices, *_ = np.where((self.parameters.population.x == xi).all(axis=0))
            if len(indices) == 0:
                raise ValueError("Unkown xi provided")

            for index in indices:
                if self.parameters.population.f[index] == None:  # noqa
                    self.parameters.population.f[index] = fi
                    break
            else:
                warnings.warn("Repeated call to tell with same xi", UserWarning)
                self.parameters.population.f[index] = fi

        self.parameters.used_budget += 1

        if len(self.ask_queue) == 0 and not self.pop_initialized:
            self.finalize_initialization()
            self.mutate()
            self.crossover()
            self.bound_correction()

        if (
            len(self.ask_queue) == 0 and (self.parameters.population.f != None).all()
        ):  # noqa
            self.select()
            if self.parameters.init_stats:
                self.track_stats()
            self.parameters.adapt()
            self.mutate()
            self.crossover()
            self.bound_correction()
