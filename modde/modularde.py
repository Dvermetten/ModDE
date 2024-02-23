"""Main implementation of Modular DE."""
import os
from itertools import islice
from typing import List, Callable

import numpy as np
import ioh
from scipy import spatial
from copy import copy

from .parameters import Parameters
from .population import Population


class ModularDE:
    r"""The main class of the configurable DE continous optimizer.

    Attributes
    ----------
    _fitness_func: callable
        The objective function to be optimized, should be from the 'ioh' package
    parameters: Parameters
        All the parameters of the DE algorithm are stored in
        the parameters object. Note if a parameters object is not
        explicitly passed, all \*args and \**kwargs passed into the
        constructor of a ModularDE are directly passed into
        the constructor of a Parameters object.

    See Also
    --------
    modde.parameters.Parameters

    """

    parameters: "Parameters"
    _fitness_func: ioh.iohcpp.problem

    def __init__(
        self, fitness_func: ioh.iohcpp.problem, *args, parameters=None, **kwargs
    ) -> None:
        """Set _fitness_func and forwards all other parameters to Parameters object."""
        self._fitness_func = fitness_func
        self.parameters = (
            parameters
            if isinstance(parameters, Parameters)
            else Parameters(fitness_func.meta_data.n_variables, *args, **kwargs)
        )
        if not self.parameters.inialize_custom_pop:
            self.initialize_population()

    def initialize_population(self) -> None:
        n_individuals = self.parameters.lambda_
        if self.parameters.oversampling_factor > 0:
            n_individuals = int(
                n_individuals * (1 + self.parameters.oversampling_factor)
            )
        if self.parameters.oppositional_initialization:
            x1 = self.parameters.sampler(int(np.ceil(n_individuals / 2)))
            x2 = self.parameters.lb.reshape(-1) + (self.parameters.ub.reshape(-1) - x1)
            x = np.vstack([x1, x2])[:n_individuals]
        else:
            x = self.parameters.sampler(n_individuals)
        x = np.transpose(x)
        # x = np.hstack(tuple(islice(self.parameters.sampler, n_individuals)))
        # x = self.parameters.lb + x * (self.parameters.ub - self.parameters.lb)
        f = np.empty(self.parameters.lambda_, object)
        for i in range(self.parameters.lambda_):
            f[i] = self._fitness_func(x[:, i])
        idxs_best = np.argsort(f)[: self.parameters.lambda_]

        self.parameters.population = Population(x[:, idxs_best], f[idxs_best])

        if self.parameters.use_archive:
            self.parameters.archive = self.parameters.population

        if self.parameters.init_stats:
            self.track_stats()

    def initialize_custom_population(self, x, f=None) -> None:
        if f is None:
            f = np.empty(self.parameters.lambda_, object)
            for i in range(self.parameters.lambda_):
                f[i] = self._fitness_func(x[:, i])
        idxs_best = np.argsort(f)[: self.parameters.lambda_]

        self.parameters.population = Population(x[:, idxs_best], f[idxs_best])

        if self.parameters.use_archive:
            self.parameters.archive = self.parameters.population
        if self.parameters.init_stats:
            self.track_stats()

    def mutate(self) -> None:
        """Apply mutation operation."""
        if self.parameters.population is None:
            raise UserError("Population is not yet initialized")
        curr_parent_idx = 0
        parent_idxs = get_parent_idxs(
            self.parameters.population.n,
            self.parameters.min_lambda,
            self.parameters.rng,
        )
        mutated = np.zeros(self.parameters.population.x.shape)
        if self.parameters.mutation_base == "rand":
            mutated += self.parameters.population[
                parent_idxs[:, curr_parent_idx].tolist()
            ].x
            curr_parent_idx += 1
        elif (
            self.parameters.mutation_base == "target"
        ):  # current and target are equivalent
            mutated += self.parameters.population.x
        elif self.parameters.mutation_base == "best":
            mutated += self.parameters.population[
                int(np.argmin(self.parameters.population.f))
            ].x
        n_comps_to_add = self.parameters.mutation_n_comps

        F = self.parameters.F
        if self.parameters.mutation_use_weighted_F:
            if self.parameters.used_budget < 0.2 * self.parameters.budget:
                F = F * 0.7
            elif self.parameters.used_budget < 0.4 * self.parameters.budget:
                F = F * 0.8
            else:
                F = F * 1.2

        # if self.parameters.mutation_reference is not None:

        if self.parameters.mutation_reference == "pbest":
            if self.parameters.pbest_value:
                pbest = self.parameters.rng.uniform(
                    0, self.parameters.pbest_value, self.parameters.population.n
                )
            else:
                pbest = self.parameters.rng.uniform(
                    2 / self.parameters.population.n,
                    0.2,
                    size=self.parameters.population.n,
                )
            idxs_pbest = np.argsort(self.parameters.population.f)[
                self.parameters.rng.randint(
                    np.clip(
                        self.parameters.population.n * pbest,
                        1,
                        self.parameters.population.n,
                    )
                )
            ]
            mutated += F * (
                self.parameters.population[idxs_pbest.tolist()].x
                - self.parameters.population.x
            )
        elif self.parameters.mutation_reference == "best":
            mutated += F * (
                self.parameters.population[
                    int(np.argmin(self.parameters.population.f))
                ].x
                - self.parameters.population.x
            )
        elif self.parameters.mutation_reference == "rand":
            mutated += F * (
                self.parameters.population[parent_idxs[:, curr_parent_idx].tolist()].x
                - self.parameters.population.x
            )
            curr_parent_idx += 1

        if self.parameters.use_archive and self.parameters.archive is not None:
            archive_idxs = self.parameters.rng.randint(
                self.parameters.archive.n, size=self.parameters.population.n
            )
            mutated += F * (
                self.parameters.population[parent_idxs[:, curr_parent_idx].tolist()].x
                - self.parameters.archive[archive_idxs.tolist()].x
            )
            curr_parent_idx += 1
            n_comps_to_add -= 1

        for mut_idx in range(n_comps_to_add):
            mutated += F * (
                self.parameters.population[parent_idxs[:, curr_parent_idx].tolist()].x
                - self.parameters.population[
                    parent_idxs[:, curr_parent_idx + 1].tolist()
                ].x
            )
        self.parameters.mutated = mutated
        curr_parent_idx += 2

    def select(self) -> None:
        """Selection of best individuals in the population."""
        x_sel = np.where(
            self.parameters.population.f < self.parameters.offspring.f,
            self.parameters.population.x,
            self.parameters.offspring.x,
        )
        f_sel = np.where(
            self.parameters.population.f < self.parameters.offspring.f,
            self.parameters.population.f,
            self.parameters.offspring.f,
        )
        self.parameters.old_population = self.parameters.population
        self.parameters.population = Population(x_sel, f_sel)
        self.parameters.improved_individuals_idx = np.where(
            self.parameters.population.f < self.parameters.old_population.f
        )[0]

    def track_stats(self) -> None:
        self.parameters.stats.popmean = np.mean(self.parameters.population.x)
        self.parameters.stats.popstd = np.std(self.parameters.population.x)

    def crossover(self) -> None:
        """ """
        mutated = self.parameters.mutated
        parent_x = self.parameters.population.x
        if self.parameters.eigenvalue_crossover:
            C = np.cov(self.parameters.population.x)
            _, B = np.linalg.eigh(C)
            mutated = np.dot(B, mutated)
            parent_x = np.dot(B, parent_x)

        if self.parameters.crossover == "bin":
            chosen = self.parameters.rng.rand(*self.parameters.population.x.shape)
            j_rand = self.parameters.rng.randint(
                0,
                self.parameters.population.x.shape[0],
                size=self.parameters.population.x.shape[1],
            )
            chosen[
                j_rand.reshape(-1, 1),
                np.arange(self.parameters.population.x.shape[1])[:, None],
            ] = 0
            crossed = np.where(chosen <= self.parameters.CR, mutated, parent_x)
        elif self.parameters.crossover == "exp":
            crossed = copy(parent_x)
            for ind_idx in range(self.parameters.population.n):
                k = self.parameters.rng.randint(self.parameters.population.d)
                offset = 0
                while offset < self.parameters.population.d:
                    # print(crossed.shape)
                    # print(self.parameters.population.d)
                    # print((k+offset) % self.parameters.population.d)
                    crossed[
                        (k + offset) % self.parameters.population.d, ind_idx
                    ] = mutated[(k + offset) % self.parameters.population.d, ind_idx]
                    offset += 1
                    if self.parameters.rng.uniform() > self.parameters.CR[ind_idx]:
                        break  # offset += self.parameters.population.d
        # print('done')
        if self.parameters.eigenvalue_crossover:
            crossed = np.dot(B.T, crossed)
        self.parameters.crossed = crossed

    def step(self) -> bool:
        """The step method runs one iteration of the optimization process.

        The method is called within the self.run loop. There, a while loop runs
        until this step function returns a Falsy value.

        Returns
        -------
        bool
            Denoting whether to keep running this step function.

        """
        # print('mut')
        self.mutate()
        # print('cros')
        self.crossover()
        # print('bcor')
        self.bound_correction()
        # print('sel')
        self.select()
        if self.parameters.init_stats:
            self.track_stats()
        if (
            self.parameters.rng.uniform()
            < self.parameters.oppositional_generation_probability
        ):
            # print('opp')
            self.oppositional_generation()
        # print('ada')
        self.parameters.adapt()
        # print(self._fitness_func.state.evaluations)
        return not any(self.break_conditions)

    def oppositional_generation(self):
        lb = np.min(self.parameters.population.x, axis=0)
        ub = np.max(self.parameters.population.x, axis=0)
        opposition = lb + ub - self.parameters.population.x
        f = np.empty(opposition.shape[1], object)
        for i in range(opposition.shape[1]):
            f[i] = self.fitness_func(opposition[:, i])
        x_merged = np.append(self.parameters.population.x, opposition, axis=1)
        f_merged = np.append(self.parameters.population.f, f)
        idxs_keep = np.argsort(f_merged)[: self.parameters.lambda_]
        self.parameters.population = Population(
            x_merged[:, idxs_keep], f_merged[idxs_keep]
        )

    def run(self):
        """Run the step method until step method retuns a falsy value.

        Returns
        -------
        ModularDE

        """
        while self.step():
            pass
        return self

    @property
    def break_conditions(self) -> List[bool]:
        """A list with break conditions based on the state of the objective function.

        Returns
        -------
        [bool, bool]

        """
        return [
            self._fitness_func.state.evaluations >= self.parameters.budget,
            self._fitness_func.state.optimum_found,
        ]

    def fitness_func(self, x: np.ndarray) -> float:
        """Wrapper function for calling self._fitness_func.

        Adds 1 to self.parameters.used_budget for each fitnes function
        call.

        TODO: remove this function and have parameters keep track internally (pass in state of function in adapt function?)

        Parameters
        ----------
        x: np.ndarray
            array on which to call the objective/fitness function

        Returns
        -------
        float

        """
        self.parameters.used_budget += 1
        if self.parameters.init_stats:
            idx = self.parameters.stats.curr_idx
            self.parameters.stats.corrected = np.max(
                self.parameters.out_of_bounds[:, idx]
            )
            self.parameters.stats.curr_F = self.parameters.F[idx]
            self.parameters.stats.curr_CR = self.parameters.CR[idx]
            if self.parameters.stats.corrected:
                self.parameters.stats.corr_so_far += 1
                x_transformed = (
                    (x - self.parameters.lb) / (self.parameters.ub - self.parameters.lb)
                ).flatten()
                x_pre = (
                    (self.parameters.crossed[:, idx] - self.parameters.lb)
                    / (self.parameters.ub - self.parameters.lb)
                ).flatten()
                x_target = (
                    (self.parameters.population.x[:, idx] - self.parameters.lb)
                    / (self.parameters.ub - self.parameters.lb)
                ).flatten()
                self.parameters.stats.CS = float(
                    1
                    - spatial.distance.cosine(
                        (x_transformed - x_target), (x_pre - x_target)
                    )
                )
                self.parameters.stats.ED = float(np.linalg.norm(x_transformed - x_pre))
            else:
                self.parameters.stats.CS = 0.0
                self.parameters.stats.ED = 0.0
            self.parameters.stats.curr_idx += 1
            if self.parameters.stats.curr_idx >= self.parameters.lambda_:
                self.parameters.stats.curr_idx = 0
        return self._fitness_func(x.flatten())

    def __repr__(self):
        """Representation of ModularDE."""
        return f"<{self.__class__.__qualname__}: {self._fitness_func}>"

    def __str__(self):
        """String representation of ModularDE."""
        return repr(self)

    def bound_correction(self):
        new_x = self.correct_bounds()
        #         print(new_x)
        f = np.empty(new_x.shape[1], object)
        for i in range(new_x.shape[1]):
            f[i] = self.fitness_func(new_x[:, i])
        self.parameters.offspring = Population(new_x, f)

    def correct_bounds(self) -> np.ndarray:
        """Bound correction function.

        Rescales x to fall within the lower lb and upper
        bounds ub specified. Available strategies are:
        - None: Don't perform any boundary correction
        - unif_resample: Resample each coordinate out of bounds uniformly within bounds
        - mirror: Mirror each coordinate around the boundary
        - COTN: Resample each coordinate out of bounds using the one-sided normal
        distribution with variance 1/3 (bounds scaled to [0,1])
        - saturate: Set each out-of-bounds coordinate to the boundary
        - toroidal: Reflect the out-of-bounds coordinates to the oposite bound inwards

        Parameters
        ----------
        x: np.ndarray
            vector of which the bounds should be corrected
        ub: float
            upper bound
        lb: float
            lower bound
        correction_method: string
            type of correction to perform

        Returns
        -------
        np.ndarray
            bound corrected version of x
        bool
            whether the population was out of bounds

        Raises
        ------
        ValueError
            When an unkown value for correction_method is provided

        """
        x = self.parameters.crossed

        self.parameters.out_of_bounds = np.logical_or(
            x > self.parameters.ub, x < self.parameters.lb
        )
        n_out_of_bounds = self.parameters.out_of_bounds.max(axis=0).sum()
        if n_out_of_bounds == 0 or self.parameters.bound_correction is None:
            return x

        try:
            _, n = x.shape
        except ValueError:
            n = 1

        if self.parameters.bound_correction in ["hvb", "expc_target", "vector_target"]:
            base_x = (self.parameters.population.x - self.parameters.lb) / (
                self.parameters.ub - self.parameters.lb
            )
        elif self.parameters.bound_correction in ["vector_best"]:
            base_x = (
                self.parameters.population.x[np.argmin(self.parameters.population.f)]
                - self.parameters.lb
            ) / (self.parameters.ub - self.parameters.lb)
        else:
            base_x = None
        x_corr = perform_correction(
            (x - self.parameters.lb) / (self.parameters.ub - self.parameters.lb),
            self.parameters.out_of_bounds,
            self.parameters.bound_correction,
            self.parameters.rng,
            base_x,
        )

        return self.parameters.lb + (self.parameters.ub - self.parameters.lb) * x_corr


def perform_correction(x_transformed, oob_idx, method, rng, base_vector=None):
    x_pre = copy(x_transformed)
    y = x_transformed[oob_idx]
    if method == "mirror":
        x_transformed[oob_idx] = np.abs(y - np.floor(y) - np.mod(np.floor(y), 2))
    elif method == "COTN":
        x_transformed[oob_idx] = np.abs(
            (y > 0) - np.abs(rng.normal(0, 1 / 3, size=y.shape))
        )
    elif method == "unif_resample":
        x_transformed[oob_idx] = rng.uniform(size=y.shape)
    elif method == "saturate":
        x_transformed[oob_idx] = y > 0
    elif method == "toroidal":
        x_transformed[oob_idx] = np.abs(y - np.floor(y))
    elif method == "hvb":
        alpha = 0.5
        x_transformed[oob_idx] = alpha * (y > 0) + (1 - alpha) * base_vector[oob_idx]
    elif method == "expc_target":
        x_transformed[oob_idx] = np.abs(
            2 * (y > 0)
            - (
                (y > 0)
                - np.log(
                    1
                    + rng.uniform(size=np.sum(oob_idx))
                    * (np.exp(-1 * np.abs((y > 0) - base_vector[oob_idx])) - 1)
                )
            )
        )
    elif method == "expc_center":
        # base_vector = np.array(len(x_transformed)*[0.5])
        base_vector = np.ones(x_transformed.shape) * 0.5
        x_transformed[oob_idx] = np.abs(
            2 * (y > 0)
            - (
                (y > 0)
                - np.log(
                    1
                    + rng.uniform(size=np.sum(oob_idx))
                    * (np.exp(-1 * np.abs((y > 0) - base_vector[oob_idx])) - 1)
                )
            )
        )
    elif method == "exps":
        x_transformed[oob_idx] = np.abs(
            2 * (y > 0)
            - (
                (y > 0)
                - np.log(1 + rng.uniform(size=np.sum(oob_idx)) * (np.exp(-1) - 1))
            )
        )
    # elif method == "vector_target":
    #     alpha = y>0 ? base_vector / base_vector - y else -1*base_vector / y - base_vector
    #     x_transformed[oob_idx] = alpha*(y>0) + (1-alpha)*base_vector[oob_idx]
    else:
        raise ValueError(f"Unknown argument: {method} for correction_method")
    return np.array(x_transformed)


def get_parent_idxs(popsize, n_parents, rng):
    temp = rng.randint(low=1, high=popsize, size=(popsize, n_parents))
    temp += np.repeat(range(popsize), n_parents).reshape(temp.shape)
    return np.mod(temp, popsize)
