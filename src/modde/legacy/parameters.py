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
from scipy.stats import qmc

from .utils import AnnotatedStruct
from .sampling import GaussianEngine, UniformEngine


class TrackedStats(AnnotatedStruct):
    """Object for keeping track of parameters we might want to track during the optimization process

    Attributes
    ----------
    curr_idx: int = 0
        Index of the current individual (for internal consistency)
    corrected: bool = False
        Wheter bound correction was applied to the current individual
    CS: float = 0
        If corrected, the cosine similarity between corrected and target points
    ED: float = 0
        If corrected, the euclidian distance between corrected and target points
    corr_so_far: int = 0
        Cumulative amount of corrections applied so far
    curr_F: float = 0
        F value used to get the current individual
    curr_CR: float = 0
        CR value used to get the current individual
    """

    curr_idx: int = 0
    corrected: bool = False
    CS: float = 0.0
    ED: float = 0.0
    corr_so_far: int = 0
    curr_F: float = 0.0
    curr_CR: float = 0.0
    F_Memory_mean: float = 0.0
    popmean: float = 0.0
    popstd: float = 0.0

    def __init__(self, *args, **kwargs) -> None:
        """Intialize parameters."""
        super().__init__(*args, **kwargs)


class Parameters(AnnotatedStruct):
    """AnnotatedStruct object for holding the parameters for the ModularDE.

    Attributes
    ----------
    d: int
        Dimensionality of the search-space
    budget: int = None
        Total budget of the optimzation procedure
    lambda_: int = None
        Population size at the current moment. When `lspr` is active, this corresponds to the initial population size and is adapted during the search
    F: np.ndarray = None
        The F-values to use. Should be the same size as the population size (lambda_)
    CR: np.ndarray = None
        The crossover rates to use. Should be the same size as the population size (lambda_)
    bound_correction: str (
        None, "saturate", "unif_resample", "COTN", "toroidal", "mirror",
        "hvb", "expc_target", "expc_center", "exps") = None
        How to deal with the box-constraints
    base_sampler: str (
        'gaussian', 'sobol', 'halton', 'uniform') = 'gaussian'
        Sampling method used for initialization.
    mutation_base: str ('rand', 'best', 'target') = 'rand'
        Which vector to use as the base element for the mutation. Best is best individual in the population, target
        is also often refered to as current (so each element is used exaclty once as the base for mutation)
    mutation_reference: str (None, 'pbest', 'best', 'rand') = None
        This corresponds to the classical -to-X DE variants. The current X is subtracted from the selected reference as an initial
        difference. Note that this is not counted as a difference component in the mutation_n_comps parameter.
    mutation_n_comps: int (1, 2) = 1
        The number of difference components to use.
    mutation_use_weighted_F: bool = False
        Decreases the effecive F values at the start of the search, increases them afterwards. Doesn't impact the F value adaptation
    crossover: (
        'bin', 'exp') = 'bin'
        Crossover operator to use
    eigenvalue_crossover: bool = False
        When enabled, crossover occurs based on the eigenvectors of the individuals rather than their current representation
    adaptation_method: (
        None, 'shade', 'jDE') = None
        Whether to use adaptive versions of F and CR as done in the SHADE or jDE algorithms
    use_jso_caps: bool = False
        Whether to cap F and CR in the begining parts of the search (based on provided budget)
    lpsr: bool = False
        Whether to use linear population size reduction
    use_archive: bool = False
        Whether to incorporate an archive in the mutation step. Size of the archive is
        set using `archive_size`. Using the archive is counted as a difference vector for mutation_n_comps parameter
    oppositional_initialization: bool = False
        Whether to use oppositional sampling (mirroring) in the initialization
    oppositional_initialization_factor: float = 1.0
        Factor of which the oppositional initialization should be closer to the center, use in combination with oppositional_initialization to achieve quasi-opposite initilization
    oversampling_factor: float = 0.0
        Fraction of additional individuals to sample during initialization. When this is e.g. 1.0, we would sample 2*lambda individuals and select the best lambda as the initial population
    memory_size: int = None
        Size of the memory when shade-adaptation is used
    archive_size: int = None
        Size of the population archive when `use_archive` is True
    oppositional_generation_probability: float = 0.0
        The probability to add a step of generating individuals based on the oppositional (mirroring) method. Occurs between selection and adaptation.
    init_stats: bool = False
        Wheter to initialize per-individual stats (to be tracked with IOHexperimenter)
    inialize_custom_pop: bool = False
        If True, population is not initialized by default, but can instead be initialized using `initialize_custom_population`
    seed: {None, int, `numpy.random.Generator`}, optional
    """

    d: int
    budget: int = None
    lambda_: int = None
    F: np.ndarray = None
    CR: np.ndarray = None
    bound_correction: (
        None,
        "saturate",
        "unif_resample",
        "COTN",
        "toroidal",
        "mirror",
        "hvb",
        "expc_target",
        "expc_center",
        "exps",
    ) = "saturate"
    base_sampler: ("gaussian", "sobol", "halton", "uniform", "LHC") = "uniform"
    mutation_base: ("rand", "best", "target") = "rand"
    mutation_reference: (None, "pbest", "best", "rand") = None
    pbest_value: float = None
    mutation_n_comps: (1, 2) = 1
    mutation_use_weighted_F: bool = False
    crossover: ("bin", "exp") = "bin"
    eigenvalue_crossover: bool = False
    adaptation_method_F: (None, "shade", "shade_modified", "jDE") = None
    adaptation_method_CR: (None, "shade", "jDE") = None
    use_jso_caps: bool = False
    lpsr: bool = False
    use_archive: bool = False
    oppositional_initialization: bool = False
    oppositional_initialization_factor: float = 1.0
    oversampling_factor: float = 0.0
    memory_size: int = None
    archive_size: int = None
    oppositional_generation_probability: float = 0.0
    population: TypeVar("Population") = None
    offspring: TypeVar("Population") = None
    __modules__ = (
        "bound_correction",
        "base_sampler",
        "mutation_base",
        "mutation_reference",
        "mutation_n_comps",
        "mutation_use_weighted_F",
        "use_archive",
        "crossover",
        "eigenvalue_crossover",
        "adaptation_method_F",
        "adaptation_method_CR",
        "use_jso_caps",
        "oppositional_initialization",
        "lpsr",
    )
    lb: np.ndarray = None
    ub: np.ndarray = None

    init_stats: bool = False
    inialize_custom_pop: bool = False

    seed: int = None

    def __init__(self, *args, **kwargs) -> None:
        """Intialize parameters. Calls sub constructors for different parameter types."""
        super().__init__(*args, **kwargs)
        self.rng = np.random.RandomState(seed=self.seed)
        self.init_selection_parameters()
        self.init_fixed_parameters()
        self.init_dynamic_parameters()
        if (
            self.adaptation_method_F is not None
            or self.adaptation_method_CR is not None
        ):
            self.init_memory()
        if self.init_stats:
            self.stats = TrackedStats()

    def get_sampler(self) -> Generator:
        """Function to return a sampler generator based on the values of other parameters.

        Returns
        -------
        generator
            a sampler

        """
        base = {
            "gaussian": GaussianEngine,
            "sobol": qmc.Sobol,
            "LHC": qmc.LatinHypercube,
            "halton": qmc.Halton,
            "uniform": UniformEngine,
        }.get(self.base_sampler, qmc.Halton)(self.d, seed=self.seed, scramble=True)

        # if self.oppositional_initialization:
        #     sampler = mirrored_sampling(sampler)

        sampler = lambda x: self.lb.reshape(-1) + (
            base.random(x) * (self.ub.reshape(-1) - self.lb.reshape(-1))
        )
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
            self.lambda_ = self.lambda_ or 18 * self.d  # defaults based on LSHADE
        else:
            self.lambda_ = self.lambda_ or 10 * self.d
        self.initial_lambda_ = self.lambda_
        if self.F is None:
            self.F = np.array([0.5] * self.lambda_)
        elif len(self.F) == 1:
            self.F = np.array(list(self.F) * self.lambda_)
        if self.CR is None:
            self.CR = np.array([0.5] * self.lambda_)
        elif len(self.CR) == 1:
            self.CR = np.array(list(self.CR) * self.lambda_)
        self.sampler = self.get_sampler()
        self.set_default("ub", np.ones((self.d, 1)) * 5)
        self.set_default("lb", np.ones((self.d, 1)) * -5)

        self.min_lambda = (
            2 * self.mutation_n_comps
            + int(self.mutation_base == "rand")
            + int(self.mutation_reference == "rand")
            + int(self.use_archive)
        )

        if (
            self.adaptation_method_F in ["shade", "shade_modified"]
            or self.adaptation_method_CR == "shade"
        ):
            self.memory_size = self.memory_size or 100
        if self.use_archive:
            # TODO: make archive size ratio a parameter
            self.archive_size = self.lambda_ * 2

    def init_dynamic_parameters(self) -> None:
        """Initialization function of parameters that represent the dynamic state of the DE."""
        self.mutated = None
        self.offspring = None
        self.crossed = None
        self.improved_individuals_idx = []
        self.archive = None

    def init_memory(self) -> None:
        """Initialize the memory when using SHADE-adaptation"""
        if self.adaptation_method_CR == "shade":
            self.CR_memory = np.array([np.mean(self.CR)] * self.memory_size)
            self.memory_idx = 0
        if self.adaptation_method_F in ["shade", "shade_modified"]:
            self.F_memory = np.array([np.mean(self.F)] * self.memory_size)
            self.memory_idx = 0
        if self.adaptation_method_CR == "jDE":
            self.tau_CR = 0.1
        if self.adaptation_method_F == "jDE":
            self.F_base = 0.1  # np.mean(self.F)
            self.F_update_strenght = 0.9
            self.tau_F = 0.1

    def adapt(self) -> None:
        """Method for adapting the internal state parameters.
        This takes care of updating archive and memory, and adapting strategy parameters if required.
        """
        self.generation_counter += 1

        if self.use_archive:
            if len(self.improved_individuals_idx) > 0:
                # Msetting the elements which have been replaced to archive
                self.archive += self.old_population[
                    self.improved_individuals_idx.tolist()
                ]
                if self.archive.n > self.archive_size:
                    # TODO: replace all np.random with a generator object for better reproducibility
                    idxs = self.rng.choice(self.archive.n, self.archive_size, False)
                    self.archive = self.archive[idxs.tolist()]

        if (
            self.adaptation_method_F in ["shade", "shade_modified"]
            or self.adaptation_method_CR == "shade"
        ):
            if len(self.improved_individuals_idx) > 0:
                weights = np.abs(
                    self.old_population[self.improved_individuals_idx.tolist()].f
                    - self.population[self.improved_individuals_idx.tolist()].f
                )
                weights /= np.sum(weights)
                if self.adaptation_method_CR == "shade":
                    self.CR_memory[self.memory_idx] = np.sum(
                        weights * self.CR[self.improved_individuals_idx.tolist()]
                    )
                if self.adaptation_method_F in ["shade", "shade_modified"]:
                    self.F_memory[self.memory_idx] = np.sum(
                        weights * self.F[self.improved_individuals_idx.tolist()]
                    )

                self.memory_idx += 1
                if self.memory_idx == self.memory_size:
                    self.memory_idx = 0

            r = self.rng.choice(self.memory_size, self.lambda_, replace=True)
        if self.adaptation_method_CR == "shade":
            cr = self.rng.normal(self.CR_memory[r], 0.1, self.lambda_)
            cr = np.clip(cr, 0, 1)
            self.CR = np.array(cr)
        if self.adaptation_method_F == "shade":
            # Faster than equivalent scipy code
            f = self.rng.standard_cauchy(size=self.lambda_) * 0.1 + self.F_memory[r]

            # TODO: check if oversampling cauchy would save some time over this while loop
            n_missing = np.sum(f <= 0)
            while (
                n_missing > 0
            ):  # can do this nicer with walrus operator, but that is python 3.8 specific, so won't go for that here
                r = self.rng.choice(self.memory_size, n_missing, replace=True)
                # Faster than equivalent scipy code
                f[f <= 0] = (
                    self.rng.standard_cauchy(size=n_missing) * 0.1 + self.F_memory[r]
                )
                n_missing = np.sum(f <= 0)

            f[f > 1] = 1

            self.F = np.array(f)

        if self.adaptation_method_F == "shade_modified":
            F_Memory_mean = np.mean(self.F_memory)
            if self.init_stats:
                self.stats.F_Memory_mean = F_Memory_mean
            # Faster than equivalent scipy code
            f = self.rng.standard_cauchy(size=self.lambda_) * 0.1 + F_Memory_mean

            # TODO: check if oversampling cauchy would save some time over this while loop
            n_missing = np.sum(f <= 0)
            while (
                n_missing > 0
            ):  # can do this nicer with walrus operator, but that is python 3.8 specific, so won't go for that here
                r = self.rng.choice(self.memory_size, n_missing, replace=True)
                # Faster than equivalent scipy code
                f[f <= 0] = (
                    self.rng.standard_cauchy(size=n_missing) * 0.1 + F_Memory_mean
                )
                n_missing = np.sum(f <= 0)

            f[f > 1] = 1

            self.F = np.array(f)

        if self.adaptation_method_F == "jDE":
            f_rand = self.rng.uniform(size=self.F.shape)
            self.F[
                f_rand > self.tau_F
            ] = self.F_base + self.F_update_strenght * self.rng.uniform(
                size=self.F[f_rand > self.tau_F].shape
            )

        if self.adaptation_method_CR == "jDE":
            cr_rand = self.rng.uniform(size=self.F.shape)
            self.CR[cr_rand > self.tau_CR] = self.rng.uniform(
                size=self.CR[cr_rand > self.tau_CR].shape
            )

        if self.lpsr:
            lambda_pre = self.lambda_
            self.lambda_ = int(
                np.round(
                    (self.min_lambda - self.initial_lambda_)
                    / self.budget
                    * self.used_budget
                    + self.initial_lambda_
                )
            )
            if self.lambda_ < lambda_pre:
                # arg_remove = np.argmax(self.population.f)
                idxs_keep = [
                    i
                    for i in range(self.population.n)
                    if i
                    not in np.argsort(self.population.f)[(self.lambda_ - lambda_pre) :]
                ]
                # idxs_keep = [i for i in range(self.population.n) if i!=np.argmax(self.population.f)]
                self.population = self.population[idxs_keep]
                self.CR = self.CR[idxs_keep]
                self.F = self.F[idxs_keep]

                if self.use_archive:
                    self.archive_size = self.lambda_ * 2
                    if self.archive.n > self.archive_size:
                        idxs = self.rng.choice(self.archive.n, self.archive_size, False)
                        self.archive = self.archive[idxs.tolist()]

        if self.use_jso_caps:
            if self.used_budget < 0.25 * self.budget:
                self.CR[self.CR > 0.7] = 0.7
            if self.used_budget < 0.5 * self.budget:
                self.CR[self.CR > 0.6] = 0.6
            if self.used_budget < 0.6 * self.budget:
                self.F[self.F > 0.7] = 0.7

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
