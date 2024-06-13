import torch
import gpytorch 
from typing import Any, Optional, Union
from gpytorch.priors import Prior
from gpytorch.constraints import Interval 
from torch import Tensor 

class MultitaskGaussianLikelihood_with_pot_and_force_regulation(gpytorch.likelihoods.MultitaskGaussianLikelihood):
    '''
    Define my own Multi-task Gaussian likelihood class so we can add required prior to potential and force parameters.
    Add two new choices of prior: task_pot_noise_prior : add prior to the potential task noise.
                                task_force_noise_prior: add prior to the force task noise. 
    '''
    def __init__(
        self,
        num_tasks: int,
        rank: int = 0,
        batch_shape: torch.Size = torch.Size(),
        task_prior: Optional[Prior] = None,
        noise_prior: Optional[Prior] = None,
        noise_constraint: Optional[Interval] = None,
        has_global_noise: bool = True,
        has_task_noise: bool = True,
        task_pot_noise_prior : Optional[Prior] = None,
        task_force_noise_prior : Optional[Prior] = None  
    ) -> None:
        
        super(MultitaskGaussianLikelihood_with_pot_and_force_regulation, self).__init__(num_tasks, rank, batch_shape,
                                                                                     task_prior, noise_prior, noise_constraint, has_global_noise,
                                                                                     has_task_noise)
        
        if rank != 0:
            if task_pot_noise_prior is not None:
                self.register_prior("TaskPotCovariancePrior", task_pot_noise_prior, lambda m: m.task_pot_noise_covar)
            
            if task_force_noise_prior is not None:
                self.register_prior("TaskForceCovariancePrior", task_force_noise_prior, lambda m: m.task_force_noise_covar)
        
    @property
    def task_pot_noise_covar(self) -> Tensor:
        if self.rank > 0:
            pot_noise_covar = self.task_noise_covar[0,0]
            return pot_noise_covar 
        else:
            raise AttributeError("Can not retrieve potential task noise when covariance is diagonal.")

    @property
    def task_force_noise_covar(self) -> Tensor:
        if self.rank > 0:
            force_noise_covar = self.task_noise_covar[1:,1:]
            return force_noise_covar 
        
        else:
            raise AttributeError("Can not retrieve force task noise when covariance is diagonal.")