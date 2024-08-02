
from typing import Any, Optional, Union 

import torch 
from  torch import Tensor 
from torch.distributions import Normal 
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _GaussianLikelihoodBase, Likelihood 
from gpytorch.priors import Prior 
from gpytorch.constraints import Interval, GreaterThan
from gpytorch.distributions import base_distributions
from gpytorch.lazy import LazyEvaluatedKernelTensor

from linear_operator.operators import (
    LinearOperator,
    DiagLinearOperator 
)

class RBFHessianGaussianLikelihood(_GaussianLikelihoodBase):
    r'''
    Base class for Gaussian Likelihoods for data with hessian information. 

    :param: ndof: number of degrees of freedom
    :param: hessian_triu_size: number of upper triangle part of hessian matrix.
    :param: batch_shape: shape of batch 
    :param: pot_noise_prior: prior for potential noise
    :param: force_noise_prior: prior for force noise 
    :param: hessian_noise_prior: prior for hessian noise.
    :param: pot_noise_constraint, force_noise_constraint, hessian_noise_constraint: constraint for pot, force and hessian data.
    '''
    def __init__(
        self, 
        ndof: int,
        hessian_triu_size: int,
        batch_shape: torch.Size= torch.Size(), 
        pot_noise_prior: Optional[Prior] = None,
        pot_noise_constraint: Optional[Interval] = None,
        force_noise_prior: Optional[Prior] = None, 
        force_noise_constraint: Optional[Interval] = None, 
        hessian_noise_prior: Optional[Prior] = None,
        hessian_noise_constraint: Optional[Interval] = None 
    ):
        super(Likelihood, self).__init__()

        if pot_noise_constraint is None:
            pot_noise_constraint = GreaterThan(1e-8)
        
        if force_noise_constraint is None:
            force_noise_constraint = GreaterThan(1e-6)
        
        if hessian_noise_constraint is None:
            hessian_noise_constraint = GreaterThan(1e-4)

        self.batch_shape = batch_shape 
        self.ndof = ndof 
        self.hessian_triu_size = hessian_triu_size 

        # register potential noises, the constraint for the potential noise & the prior for the potential noise
        # follwoing the convention in gpytorch, here pot_noises are variances of noise 
        self.register_parameter(
            name= "raw_pot_noises", parameter= torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        )
        self.register_constraint("raw_pot_noises", pot_noise_constraint)
        if pot_noise_prior is not None:
            self.register_prior("raw_pot_noises_prior", pot_noise_prior, lambda m: m.pot_noises)
        
        # register force noise, the constraint for the force noise & the prior for the force noise 
        # following the convention in gpytorch, here force_noises are variances of noise 
        self.register_parameter(
            name= "raw_force_noises", parameter= torch.nn.Parameter(torch.zeros(*batch_shape, ndof))
        )
        self.register_constraint("raw_force_noises", force_noise_constraint)
        if force_noise_prior is not None:
            self.register_prior("raw_force_noises_prior", force_noise_prior, lambda m: m.force_noises)

        # register hessian noises, the constraint for the hessian noise and the prior for the hessian noise 
        # following the convention in gpytorch, here hessian_noises are variances of noise 
        self.register_parameter(
            name= "raw_hessian_noises", parameter= torch.nn.Parameter(torch.zeros(*batch_shape, hessian_triu_size))
        )
        self.register_constraint("raw_hessian_noises", hessian_noise_constraint)
        if hessian_noise_prior is not None:
            self.register_prior("raw_hessian_noises_prior", hessian_noise_prior, lambda m: m.hessian_noises)


    @property
    def pot_noises(self) -> Optional[Tensor]:
        # variance of potential noise.
        return self.raw_pot_noises_constraint.transform(self.raw_pot_noises)
    
    @pot_noises.setter
    def pot_noises(self, value: Union[float, Tensor]) -> None:
        self.initialize(raw_pot_noises= self.raw_pot_noises_constraint.inverse_transform(value))

    @property
    def force_noises(self) -> Optional[Tensor]:
        # variance of force noises.
        return self.raw_force_noises_constraint.transform(self.raw_force_noises)
    
    @force_noises.setter
    def force_noises(self, value: Union[float, Tensor]) -> None:
        self.initialize(raw_force_noises= self.raw_force_noises_constraint.inverse_transform(value))

    @property
    def hessian_noises(self) -> Optional[Tensor]:
        # variance of hessian noises.
        return self.raw_hessian_noises_constraint.transform(self.raw_hessian_noises)
    
    @hessian_noises.setter
    def hessian_noises(self, value: Union[float, Tensor]) -> None:
        self.initialize(raw_hessian_noises= self.raw_hessian_noises_constraint.inverse_transform(value))
    
    def _shaped_noise_covar(
            self,
            M: int, M_H: int, 
            *params: Any, **kwargs: Any
    ) -> LinearOperator:
        '''
        :param: M: total number of input data points
        :param: M_H: number of input data points contain hessian information.
        '''
        n_batch_dim = len(self.batch_shape)
        pot_noises_var = self.pot_noises.repeat([ *([1] * n_batch_dim), M])  # shape: [M]
        force_noises_var = self.force_noises.repeat([ *([1] * n_batch_dim), M])  # shape: [M * d]
        hessian_noises_var = self.hessian_noises.repeat([ *([1] * n_batch_dim), M_H])  # shape: [M_H * hessian_triu_size]

        noises_var = torch.concat( (pot_noises_var, force_noises_var, hessian_noises_var), dim= -1)

        noises_var_operator = DiagLinearOperator(noises_var)

        return noises_var_operator 


    def forward(self, function_samples: Tensor, *params, **kwargs: Any):
        '''
        compute the conditional probability p(y|f(x)) of the tensor sample.
        :param: M: number of data points in the 1d data.
        :param: M_H: number of data points that have hessian information.
        '''
        if len(params) != 2:
            print("Must pass M & M_H into likelihood function.")

        M = params[0]
        M_H = params[1]
        
        noise = self._shaped_noise_covar(M, M_H).diagonal(dim1= -1, dim2= -2)  # take diagonal part of the matrix.
        return base_distributions.Independent(base_distributions.Normal( function_samples, noise.sqrt()), 1)

    def marginal(self, function_dist: MultivariateNormal, *params : Any, **kwargs: Any) -> MultivariateNormal:
        r"""
        compute the marginal probability p(y|X), marginalize over f(X).
        :param: function_dist: latent function distribution f(X). MultivariateNormal distribution.
        :param: M: number of data points in the 1d data.
        :param: M_H: number of data points that have hessian information.
        """
        if len(params) != 2:
            print("Must pass M & M_H into likelihood function.")

        M = params[0]
        M_H = params[1]

        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix

                # ensure that sumKroneckerLT is actually called
        if isinstance(covar, LazyEvaluatedKernelTensor):
            covar = covar.evaluate_kernel()
        
        noise_covar = self._shaped_noise_covar(
            M, M_H 
        )
        
        full_covar = covar + noise_covar 

        return function_dist.__class__(mean, full_covar)