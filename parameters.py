"""Definition of Parameters objects, which are used by ModularCMA-ES."""
import os
import pickle
import warnings
from collections import deque
from typing import Generator, TypeVar, Union
from itertools import islice
from copy import copy

import numpy as np
from scipy import linalg, stats

from utils import AnnotatedStruct
from sampling import (
    gaussian_sampling,
    # orthogonal_sampling,
    # mirrored_sampling,
    sobol_sampling,
    halton_sampling,
    uniform_sampling,
)


class Parameters(AnnotatedStruct):
    """AnnotatedStruct object for holding the parameters for the ModularCMAES.

    Attributes
    ----------
    d: int 
        Dimensionality of the search-space
    budget: int = None 
        Total budget of the optimzation procedure
    initial_lambda_: int = None
        Initial population size, used only when `lspr` is active
    lambda_: int = None
        Population size at the current moment. When `lspr` is active, use `initial_lambda_` instead
    F: np.ndarray = None
        The F-values to use. Should be the same size as the population size (lambda_)
    CR: np.ndarray = None
        The crossover rates to use. Should be the same size as the population size (lambda_)
    bound_correction = str (
        None, "saturate", "unif_resample", "COTN", "toroidal", "mirror", 
        "hvb", "expc_target", "expc_center", "exps") = None
        How to deal with the box-constraints
    base_sampler = str (
        'gaussian', 'sobol', 'halton', 'uniform') = 'gaussian'
        Sampling method used for initialization. 
    mutation = str (
        'rand/1', 'rand/2', 'best/1', 'target_pbest/1', 'target_best/2', 'target_rand/1', '2_opt/1') = 'rand/1'
        Mutation strategy to use. Note: target_pbest is the same as curr_pbest
    crossover: (
        'bin', 'exp') = 'bin'   
        Crossover operator to use
    shade: bool = False
        Whether to use adaptive versions of F and CR as done in the SHADE algorithm
    lpsr: bool = False
        Whether to use linear population size reduction
    memory_size: int = None
        Size of the memory when shade-adaptation is used
    use_archive: bool = False
        Whether to incorporate an archive in the mutation step. Size of the archive is
        set using `archive_size`. Note: archive use is only supported in `target_pbest/1` mutation.
    archive_size: int = None
        Size of the population archive when `use_archive` is True
    """

    d: int
    budget: int = None
    initial_lambda_: int = None
    lambda_: int = None
    F: np.ndarray = None
    CR: np.ndarray = None
    bound_correction: (
        None, "saturate", "unif_resample", "COTN", "toroidal", "mirror", 
        "hvb", "expc_target", "expc_center", "exps") = None
    base_sampler: (
        'gaussian', 'sobol', 'halton', 'uniform') = 'gaussian'
    mutation: (
        'rand/1', 'rand/2', 'best/1', 'target_pbest/1', 'target_best/2', 'target_rand/1', '2_opt/1') = 'rand/1'
    crossover: (
        'bin', 'exp') = 'bin'   
    shade: bool = False
    lpsr: bool = False
    memory_size: int = None
    archive_size: int = None
    population: TypeVar("Population") = None
    offspring: TypeVar("Population") = None
    use_archive: bool = False
    __modules__ = (
        "bound_correction",
        "base_sampler",
        "mutation",
        "crossover",
        "Shade",
        "lpsr",
    )

    def __init__(self, *args, **kwargs) -> None:
        """Intialize parameters. Calls sub constructors for different parameter types."""
        super().__init__(*args, **kwargs)
        self.init_selection_parameters()
        self.init_fixed_parameters()
        self.init_dynamic_parameters()
        if self.shade:
            self.init_memory()
#         self.init_population()

    def get_sampler(self) -> Generator:
        """Function to return a sampler generator based on the values of other parameters.

        Returns
        -------
        generator
            a sampler

        """
        sampler = {
            "gaussian": gaussian_sampling,
            "sobol": sobol_sampling,
            "halton": halton_sampling,
            "uniform": uniform_sampling,
        }.get(self.base_sampler, gaussian_sampling)(self.d)

        return sampler

    def init_fixed_parameters(self) -> None:
        """Initialization function for parameters that are not restarted during a run."""
        self.used_budget = 0
        self.generation_counter = 0
        self.n_out_of_bounds = 0
        self.budget = self.budget or int(1e4) * self.d
        self.fopt = float("inf")
        self.xopt = None

    def init_selection_parameters(self) -> None:
        """Initialization function for parameters that influence in selection."""
        if self.lpsr:
            self.lambda_ = self.initial_lambda_ or 100 #TODO: set defaults
        else:
            self.lambda_ = self.lambda_ or (4 + np.floor(3 * np.log(self.d))).astype(int)
        self.initial_lambda_ = self.lambda_
        
        if self.F is None:
            self.F = np.array([0.5] * self.lambda_)
        elif len(self.F) == 1:
            self.F = np.array(self.F * self.lambda_)
        if self.CR is None:
            self.CR = np.array([0.5] * self.lambda_)
        elif len(self.CR) == 1:
            self.CR = np.array(self.CR * self.lambda_)
        
        self.sampler = self.get_sampler()
        self.set_default("ub", np.ones((self.d, 1)) * 5)
        self.set_default("lb", np.ones((self.d, 1)) * -5)
        
        if self.shade:
            self.memory_size = self.memory_size or 100
        if self.use_archive:
            self.archive_size = self.lambda_ * 2

#     def init_population(self) -> None:
#         """Initialization function for parameters for self-adaptive processes.

#         Examples are recombination weights and learning rates for the covariance
#         matrix adapation.
#         """
#         x = np.hstack(tuple(islice(self.sampler, self.lambda_)))
#         f = np.empty(self.lambda_, object)
#         for i in range(self.lambda_):
#             f[i] = self.fitness_func(x[:, i])
#         self.population = Population(x, f)

    def init_dynamic_parameters(self) -> None:
        """Initialization function of parameters that represent the dynamic state of the DE.
        """
        self.mutated = None
        self.offspring = None
        self.crossed = None
        self.improved_individuals_idx = []
        self.archive = None
            
        
    def init_memory(self) -> None:
        """ Initialize the memory when using SHADE-adaptation"""
        self.CR_memory = np.array([np.mean(self.CR)] * self.memory_size)
        self.F_memory = np.array([np.mean(self.F)] * self.memory_size)
        self.memory_idx = 0
        
        


    def adapt(self) -> None:
        """Method for adapting the internal state parameters.
        This takes care of updating archive and memory, and adapting strategy parameters if required.
        """
        self.generation_counter += 1
        
        if self.use_archive:
            if len(self.improved_individuals_idx) > 0:
                # if self.archive is None:
                #     self.archive = self.population[self.improved_individuals_idx.tolist()]
                # else:
                self.archive += self.population[self.improved_individuals_idx.tolist()]
                if self.archive.n > self.archive_size:
                    #TODO: replace all np.random with a generator object for better reproducibility
                    idxs = np.random.choice(self.archive.n, self.archive_size, False)
                    self.archive = self.archive[idxs.tolist()]
                    
        if self.shade:
            if len(self.improved_individuals_idx) > 0:
                weights = np.abs(self.population[self.improved_individuals_idx.tolist()].f - self.offspring[self.improved_individuals_idx.tolist()].f)
                weights /= np.sum(weights)
                self.CR_memory[self.memory_idx] = np.sum(weights * self.CR[self.improved_individuals_idx.tolist()])
                # if max(self.CR) != 0:
                #     self.CR_memory[self.memory_idx] = np.sum(weights * self.CR[self.improved_individuals_idx.tolist()])
                # else:
                #     self.CR_memory[self.memory_idx] = 1

                self.F_memory[self.memory_idx] = np.sum(weights * self.F[self.improved_individuals_idx.tolist()])
                # self.F_memory[self.memory_idx] = np.sum(self.population[self.improved_individuals_idx.tolist()].f**2)/np.sum(self.population[self.improved_individuals_idx.tolist()].f)

                self.memory_idx += 1
                if self.memory_idx == self.memory_size:
                    self.memory_idx = 0
            
            r = np.random.choice(self.memory_size, self.lambda_, replace=True)
            cr = np.random.normal(self.CR_memory[r], 0.1, self.lambda_)
            cr = np.clip(cr, 0, 1)
            # cr[cr == 1] = 0 #check what is the problem of cr 0
            # f = stats.cauchy.rvs(loc=self.F_memory[r], scale=0.1, size=self.lambda_)
            f = np.random.standard_cauchy(size=self.lambda_)*0.1+self.F_memory[r] #Faster than equivalent scipy code
            # f[f > 1] = 1

            #TODO: check if oversampling cauchy would save some time over this while loop
            while sum(f <= 0) != 0:
                r = np.random.choice(self.memory_size, sum(f <= 0), replace=True)
                # f[f <= 0] = stats.cauchy.rvs(loc=self.F_memory[r], scale=0.1, size=sum(f <= 0))
                f[f <= 0] = np.random.standard_cauchy(size=sum(f <= 0))*0.1+self.F_memory[r] #Faster than equivalent scipy code

            f[f > 1] = 1

            self.CR = np.array(cr)
            self.F = np.array(f)
        
        if self.lpsr:
            lambda_pre = self.lambda_
            self.lambda_ = int(np.round((4 - self.initial_lambda_)/self.budget * self.used_budget + self.initial_lambda_))
            if self.lambda_ < lambda_pre:
                # arg_remove = np.argmax(self.population.f)
                idxs_keep = [i for i in range(self.population.n) if i!=np.argmax(self.population.f)]
                self.population = self.population[idxs_keep]
                self.CR = self.CR[idxs_keep]
                self.F = self.F[idxs_keep]
                
                if self.use_archive:
                    self.archive_size = self.lambda_ * 2
                    if self.archive.n > self.archive_size:
                        idxs = np.random.choice(self.archive.n, self.archive_size, False)
                        self.archive = self.archive[idxs.tolist()]


#     @staticmethod
#     def from_config_array(d: int, config_array: list) -> "Parameters":
#         """Instantiate a Parameters object from a configuration array.

#         Parameters
#         ----------
#         d: int
#             The dimensionality of the problem

#         config_array: list
#             A list of length len(Parameters.__modules__),
#                 containing ints from 0 to 2

#         Returns
#         -------
#         A new Parameters instance

#         """
#         if not len(config_array) == len(Parameters.__modules__):
#             raise AttributeError(
#                 "config_array must be of length " + str(len(Parameters.__modules__))
#             )
#         parameters = dict()
#         for name, cidx in zip(Parameters.__modules__, config_array):
#             options = getattr(getattr(Parameters, name), "options", [False, True])
#             if not len(options) > cidx:
#                 raise AttributeError(
#                     f"id: {cidx} is invalid for {name} "
#                     f"with options {', '.join(map(str, options))}"
#                 )
#             parameters[name] = options[cidx]
#         return Parameters(d, **parameters)

    @staticmethod
    def load(filename: str) -> "Parameters":
        """Load stored  parameter objects from pickle.

        Parameters
        ----------
        filename: str
            A file path

        Returns
        -------
        A Parameters object

        """
        if not os.path.isfile(filename):
            raise OSError(f"{filename} does not exist")

        with open(filename, "rb") as f:
            parameters = pickle.load(f)
            if not isinstance(parameters, Parameters):
                raise AttributeError(
                    f"{filename} does not contain " "a Parameters object"
                )
        parameters.sampler = parameters.get_sampler()
        return parameters

    def save(self, filename: str = "parameters.pkl") -> None:
        """Save a parameters object to pickle.

        Parameters
        ----------
        filename: str
            The name of the file to save to.

        """
        with open(filename, "wb") as f:
            self.sampler = None
            pickle.dump(self, f)

    def record_statistics(self) -> None:
        """Method for recording metadata."""
        

#     def update(self, parameters: dict, reset_default_modules=False):
#         """Method to update the values of self based on a given dict of new parameters.

#         Note that some updated parameters might be overridden by:
#             self.init_selection_parameters()
#             self.init_adaptation_parameters()
#             self.init_local_restart_parameters()
#         which are called at the end of this function. Use with caution.


#         Parameters
#         ----------
#         parameters: dict
#             A dict with new parameter values

#         reset_default_modules: bool = False
#             Whether to reset the modules back to their default values.

#         """
#         if reset_default_modules:
#             for name in Parameters.__modules__:
#                 default_option, *_ = getattr(
#                     getattr(Parameters, name), "options", [False, True]
#                 )
#                 setattr(self, name, default_option)

#         for name, value in parameters.items():
#             if not hasattr(self, name):
#                 raise ValueError(f"The parameter {name} doesn't exist")
#             setattr(self, name, value)

#         self.init_selection_parameters()
#         self.init_adaptation_parameters()
#         self.init_local_restart_parameters()