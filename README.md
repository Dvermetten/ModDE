# Modular DE 

This work-in-progress repository contains the code used to create a modular version of differential evolution. 

## Basic use-case: L-SHADE

To instantiate L-SHADE using modDE and optimize a function (using iohexperimenter), the following code can be used:

```python
from modularde import ModularDE
import ioh
import numpy as np

f = ioh.get_problem(23, 1, 5)
lshade = ModularDE(f, base_sampler='uniform', mutation_base='target', mutation_reference='pbest', bound_correction='expc_center', crossover='bin', lpsr=True, lambda_ = 18*5, memory_size = 6, use_archive=True, init_stats=True, adaptation_method_F='shade', adaptation_method_CR='shade')
lshade.run()
```

To perform a larger benchmark experiment which includes tracking of internal parameters, the following can be used (note that running the full experiment with detailed tracking will use a significant amount of storage):

```python
class LSHADE_interface():
    def __init__(self, bound_corr):
        self.bound_corr = bound_corr
        self.lshade = None
        
    def __call__(self, f):
        self.lshade = ModularDE(f, base_sampler='uniform', mutation_base='target', mutation_reference='pbest', bound_correction = self.bound_corr, crossover='bin', lpsr=True, lambda_ = 18*f.meta_data.n_variables, memory_size = 6, use_archive=True, init_stats = True, adaptation_method_F='shade', adaptation_method_CR='shade')
        self.lshade.run()
        
    @property
    def F(self):
        if self.lshade is None:
            return 0
        return self.lshade.parameters.stats.curr_F
    
    @property
    def CR(self):
        if self.lshade is None:
            return 0
        return self.lshade.parameters.stats.curr_CR

    @property
    def CS(self):
        if self.lshade is None:
            return 0
        return self.lshade.parameters.stats.CS
    
    @property
    def ED(self):
        if self.lshade is None:
            return 0
        return self.lshade.parameters.stats.ED
    
    @property
    def cumulative_corrected(self):
        if self.lshade is None:
            return 0
        return self.lshade.parameters.stats.corr_so_far
    
    @property
    def corrected(self):
        if self.lshade is None:
            return 0
        return self.lshade.parameters.stats.corrected
        
obj = LSHADE_interface('saturate')

exp = ioh.Experiment(algorithm = obj, #Set the optimization algorithm
  fids = range(1,25), iids = [1,2,3,4,5], dims = [5,30], reps = 5, problem_type = 'Real', #Problem definitions
  njobs = 12, logger_triggers = [ioh.logger.trigger.ALWAYS],#Enable paralellization
  logged = True, folder_name = f'L-SHADE_sat', algorithm_name = f'L-SHADE', store_positions = True, #Logging specifications
  experiment_attributes = {'SDIS' : 'Saturate'}, logged_attributes = ['corrected', 'cumulative_corrected', 'F', 'CR', 'CS', 'ED'], #Attribute tracking
  merge_output = True, zip_output = True, remove_data = True #Only keep data as a single zip-file
)

exp()
```

The design of this package is heavily based on the Modular CMA-ES package: https://github.com/IOHprofiler/ModularCMAES
