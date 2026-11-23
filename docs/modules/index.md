# Module reference

modde's DE is one algorithm assembled from modules. Every module has a default
that gives **classic DE/rand/1/bin**. Each page below explains what one group of
modules does (with the exact formulas), lists its options and parameters, and
summarises how its options combine with the rest, based on the
[interaction study](../interactions.md).

| Step of a generation | Page | Settings |
|---|---|---|
| (once) create the initial population | [Initialisation](initialization.md) | `base_sampler`, `oppositional_initialization`, `oversampling_factor`, `initial_population` |
| 0. draw F and CR for this generation | [F and CR control](adaptation.md) | `adaptation_method_F`, `adaptation_method_CR`, `use_jso_caps` |
| 1. mutation: donor vectors | [Mutation](mutation.md) | `mutation_base`, `mutation_reference`, `mutation_n_comps`, `pbest_sampling`, `mutation_use_weighted_F` |
| 2. crossover: trial vectors | [Crossover](crossover.md) | `crossover`, `eigenvalue_crossover` |
| 3. repair coordinates outside the bounds | [Bound correction](bound_correction.md) | `bound_correction` |
| 4–5. evaluate, one-to-one selection | [Selection and archive](selection_archive.md) | `use_archive`, `archive_content` |
| 6. optional opposition-based jump | [Oppositional generation](oppositional_generation.md) | `oppositional_generation` |
| 7. update archive, memories and population size | [Population size](population_size.md), [F and CR control](adaptation.md) | `population_reduction` |

## Notation

| Symbol | Meaning |
|---|---|
| $d$ | dimension |
| $\lambda$ | current population size (`lambda0` at the start) |
| $x_i$, $f_i$ | individual $i$ of the population and its objective value |
| $v_i$, $u_i$ | donor and trial vector of individual $i$ |
| $F_i$, $CR_i$ | scale factor and crossover rate used for individual $i$ in this generation |
| $x_{r}$ | a population member drawn uniformly at random; all random parents of one donor are distinct and differ from $i$ |
| $t \in [0, 1]$ | progress: evaluations used divided by the budget |
| $\mathrm{lb}, \mathrm{ub}$ | lower and upper bounds |

## Paper-faithfulness

modde implements the modules as described in the papers. Where 0.0.4 deviated
from a paper (or had a bug), the new version follows the paper; the
[migration guide](../migration.md) lists the changes. Where the published code of
an algorithm deviates from its paper, modde follows the paper and, if the
difference matters, offers the code's behaviour as an option (for example
[`archive_content`](selection_archive.md#archive_content) for L-SHADE).
