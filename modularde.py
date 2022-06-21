"""Main implementation of Modular CMA-ES."""
import os
from itertools import islice
from typing import List, Callable

import numpy as np
import ioh
from scipy import spatial
from copy import copy

from parameters import Parameters
from population import Population
# from .utils import timeit, ert


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
        self.initialize_population()

    def initialize_population(self) -> None:
        x = np.hstack(tuple(islice(self.parameters.sampler, self.parameters.lambda_)))
        x = self.parameters.lb + x * (self.parameters.ub - self.parameters.lb )
        f = np.empty(self.parameters.lambda_, object)
        for i in range(self.parameters.lambda_):
            f[i] = self._fitness_func(x[:, i])
        self.parameters.population = Population(x, f)
        if self.parameters.use_archive:
            self.parameters.archive = self.parameters.population
    
    def mutate(self) -> None:
        """Apply mutation operation.

        
        """

            
        if self.parameters.mutation == 'rand/1':
            parent_idxs = get_parent_idxs(self.parameters.population.n, 3)
            mutated = self.parameters.F * (self.parameters.population[parent_idxs[:, 1].tolist()].x - self.parameters.population[parent_idxs[:, 2].tolist()].x)
            self.parameters.mutated = mutated + self.parameters.population[parent_idxs[:, 0].tolist()].x
        elif self.parameters.mutation == 'rand/2':
            parent_idxs = get_parent_idxs(self.parameters.population.n, 5)
            mutated = self.parameters.F * (self.parameters.population[parent_idxs[:, 1].tolist()].x - self.parameters.population[parent_idxs[:, 2].tolist()].x)
            mutated += self.parameters.F * (self.parameters.population[parent_idxs[:, 3].tolist()].x - self.parameters.population[parent_idxs[:, 4].tolist()].x)
            self.parameters.mutated = mutated + self.parameters.population[parent_idxs[:, 0].tolist()].x
        elif self.parameters.mutation == 'best/1':
            parent_idxs = get_parent_idxs(self.parameters.population.n, 2)
            mutated = self.parameters.F * (self.parameters.population[parent_idxs[:, 0].tolist()].x - self.parameters.population[parent_idxs[:, 1].tolist()].x)
            self.parameters.mutated = mutated + self.parameters.population[int(np.argmin(self.parameters.population.f))].x
        elif self.parameters.mutation == 'target_pbest/1':
            parent_idxs = get_parent_idxs(self.parameters.population.n, 2)
            if self.parameters.use_archive and self.parameters.archive is not None:
                archive_idxs = np.random.randint(self.parameters.archive.n, size= self.parameters.population.n) 
                mutated = self.parameters.F * (self.parameters.population[parent_idxs[:, 0].tolist()].x - self.parameters.archive[archive_idxs.tolist()].x)
            else:
                mutated = self.parameters.F * (self.parameters.population[parent_idxs[:, 0].tolist()].x - self.parameters.population[parent_idxs[:, 1].tolist()].x)
            idxs_pbest = np.argsort(self.parameters.population.f)[np.random.randint(np.clip(self.parameters.population.n*np.random.uniform(2/self.parameters.population.n, 0.2, size = self.parameters.population.n), 1, self.parameters.population.n))]
            mutated += self.parameters.F * (self.parameters.population[idxs_pbest.tolist()].x - self.parameters.population.x)
            self.parameters.mutated = mutated + self.parameters.population.x
        elif self.parameters.mutation == 'target_best/2':
            parent_idxs = get_parent_idxs(self.parameters.population.n, 4)
            mutated = self.parameters.F * (self.parameters.population[parent_idxs[:, 0].tolist()].x - self.parameters.population[parent_idxs[:, 1].tolist()].x)
            mutated += self.parameters.F * (self.parameters.population[parent_idxs[:, 2].tolist()].x - self.parameters.population[parent_idxs[:, 3].tolist()].x)
            mutated += mutated + self.parameters.F * (self.parameters.population[int(np.argmin(self.parameters.population.f))].x - self.parameters.population.x)
            self.parameters.mutated = mutated + self.parameters.population.x
        elif self.parameters.mutation == 'target_rand/1':
            parent_idxs = get_parent_idxs(self.parameters.population.n, 3)
            mutated = self.parameters.F * (self.parameters.population[parent_idxs[:, 0].tolist()].x - self.parameters.population[parent_idxs[:, 1].tolist()].x)
            mutated += mutated + self.parameters.F * (self.parameters.population[parent_idxs[:, 2].tolist()].x - self.parameters.population.x)
            self.parameters.mutated = mutated + self.parameters.population.x     
        elif self.parameters.mutation == '2_opt/1':
            raise NotImplemented
            
    def select(self) -> None:
        """Selection of best individuals in the population.

        
        """
        x_sel = np.where(self.parameters.population.f < self.parameters.offspring.f, self.parameters.population.x, self.parameters.offspring.x)
        f_sel = np.where(self.parameters.population.f < self.parameters.offspring.f, self.parameters.population.f, self.parameters.offspring.f)
        self.parameters.population = Population(x_sel, f_sel)
        self.parameters.improved_individuals_idx = np.where(self.parameters.population.f < self.parameters.offspring.f)[0]

    def crossover(self) -> None:
        """
        """
        if self.parameters.crossover == "bin":
            chosen = np.random.rand(*self.parameters.population.x.shape)
            j_rand = np.random.randint(0, self.parameters.population.x.shape[0], size=self.parameters.population.x.shape[1])
            chosen[j_rand.reshape(-1,1),np.arange(self.parameters.population.x.shape[1])[:,None]] = 0
            self.parameters.crossed = np.where(chosen <= self.parameters.CR, self.parameters.mutated, self.parameters.population.x)
        elif self.parameters.crossover == "exp":
            crossed = copy(self.parameters.population.x)
            for ind_idx in range(self.parameters.population.n):
                k = np.random.randint(self.parameters.population.d)
                offset = 0
                while offset < self.parameters.population.d:
                    crossed[k+offset % self.parameters.population.d, ind_idx] = self.parameters.mutated[k+offset % self.parameters.population.d, ind_idx]
                    if np.random.uniform() > self.parameters.CR[ind_idx]:
                        break
            self.parameters.crossed = crossed
            
#         f = np.empty(self.parameters.lambda_, object)
#         for i in range(self.parameters.lambda_):
#             f[i] = self._fitness_func(crossed[:, i])
#         self.parameters.offspring = Population(crossed, f)

    def step(self) -> bool:
        """The step method runs one iteration of the optimization process.

        The method is called within the self.run loop. There, a while loop runs
        until this step function returns a Falsy value.

        Returns
        -------
        bool
            Denoting whether to keep running this step function.

        """
        self.mutate()
        self.crossover()
        self.bound_correction()
        self.select()
        self.parameters.adapt()
        return not any(self.break_conditions)

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
            # self.parameters.target >= self.parameters.fopt,
            self._fitness_func.state.evaluations >= self.parameters.budget,
            self._fitness_func.state.optimum_found
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
        return self._fitness_func(x.flatten())

    def __repr__(self):
        """Representation of ModularCMA-ES."""
        return f"<{self.__class__.__qualname__}: {self._fitness_func}>"

    def __str__(self):
        """String representation of ModularCMA-ES."""
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
        
        out_of_bounds = np.logical_or(x > self.parameters.ub, x < self.parameters.lb)
        n_out_of_bounds = out_of_bounds.max(axis=0).sum()
        if n_out_of_bounds == 0 or self.parameters.bound_correction is None:
            return x

        try:
            _, n = x.shape
        except ValueError:
            n = 1
            
#         ub, lb = np.tile(self.parameters.ub, n)[out_of_bounds], np.tile(self.parameters.lb, n)[out_of_bounds]
        #TODO: fix base vector
        if self.parameters.bound_correction in ['hvb', 'expc_target']:
            base_x = (self.parameters.population.x - self.parameters.lb) / (self.parameters.ub - self.parameters.lb)
        else:
            base_x = None
        x_corr = perform_correction((x - self.parameters.lb) / (self.parameters.ub - self.parameters.lb), out_of_bounds, self.parameters.bound_correction, base_x)
        return self.parameters.lb + (self.parameters.ub - self.parameters.lb) * x_corr
    
    
def perform_correction(x_transformed, oob_idx, method, base_vector=None):
#     print(oob_idx)
    x_pre = copy(x_transformed)
    y = x_transformed[oob_idx]
    if method == "mirror":
        x_transformed[oob_idx] = np.abs(
            y - np.floor(y) - np.mod(np.floor(y), 2)
        )
    elif method == "COTN":
        x_transformed[oob_idx] = np.abs(
            (y > 0) - np.abs(np.random.normal(0, 1 / 3, size=y.shape))
        )
    elif method == "unif_resample":
        x_transformed[oob_idx] = np.random.uniform(size=y.shape)
    elif method == "saturate":
        x_transformed[oob_idx] = (y > 0)
    elif method == "toroidal":
        x_transformed[oob_idx] = np.abs(y - np.floor(y))
    elif method == "hvb":
        alpha = 0.5
        x_transformed[oob_idx] = alpha*(y>0) + (1-alpha)*base_vector[oob_idx]
    elif method == "expc_target":
        x_transformed[oob_idx] = np.abs(2*(y>0) - ((y>0)-np.log(1+np.random.uniform(size=np.sum(oob_idx))*(np.exp(-1*np.abs((y>0) - base_vector[oob_idx]))-1))))
    elif method == "expc_center":
        base_vector = np.array(len(x_transformed)*[0.5])
        x_transformed[oob_idx] = np.abs(2*(y>0) - ((y>0)-np.log(1+np.random.uniform(size=np.sum(oob_idx))*(np.exp(-1*np.abs((y>0) - base_vector[oob_idx]))-1))))
    elif method == "exps":
        x_transformed[oob_idx] = np.abs(2*(y>0) - ((y>0)-np.log(1+np.random.uniform(size=np.sum(oob_idx))*(np.exp(-1)-1))))
    else:
        raise ValueError(f"Unknown argument: {method} for correction_method")
    return np.array(x_transformed)

def get_parent_idxs(popsize, n_parents):
    temp = np.random.randint(low=1,high=popsize, size=(popsize,n_parents))
    temp += np.repeat(range(popsize), n_parents).reshape(temp.shape)
    return np.mod(temp, popsize)