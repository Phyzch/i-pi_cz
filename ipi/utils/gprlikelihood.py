import torch
import gpytorch 
from typing import Any, Optional, Union
from gpytorch.priors import Prior
from gpytorch.constraints import Interval 
from torch import Tensor 

class MultitaskGaussianLikelihood_covar_factor_regularization(gpytorch.likelihoods.MultitaskGaussianLikelihood):
    '''
    Define my own Multi-task Gaussian likelihood class so we can add required prior to potential and force parameters.
    Add one option for prior: task_covar_factor_noise_prior:  prior for the covariance factor of MultitaskGaussian distribution
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
        task_covar_factor_noise_prior : Optional[Prior] = None,
    ) -> None:
        
        super(MultitaskGaussianLikelihood_covar_factor_regularization, self).__init__(num_tasks, rank, batch_shape,
                                                                                     task_prior, noise_prior, noise_constraint, has_global_noise,
                                                                                     has_task_noise)
        
        if rank != 0:
            if task_covar_factor_noise_prior is not None:
                self.register_prior("TaskCovarianceFactorPrior", task_covar_factor_noise_prior, lambda m: m.task_noise_covar_factor)
            

    