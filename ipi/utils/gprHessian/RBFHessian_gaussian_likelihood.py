'''
The class that compute marginalized probability distribution after consider noise in the data. 
In the code, we transform noise from Cartesian coordinate into internal coordinate.
Written by Chenghao Zhang, Pacific Northwest National Laboratory (chenghao.zhang@pnnl.gov), 2024.
'''
from typing import Any, Optional, Union 

import numpy as np 
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
    :param: has_covar_factor: The noise covariance matrix is not diagonal. The covariance matrix is expressed as covar_factor * covar_factor.T.
    :param: noise_covar_factor_pot_grad_array: An array of covar factor for all data points. The covar factor here perform transformation of gradient noise from Cartesian coordinate into internal coordinate.
    :param: noise_covar_factor_with_hessian_array: An array of covar factor for data points with hessians. The covar factor here perform transformation of gradient & hessian noise from Cartesian coordinate into internal coordinate.
    :param: grad_covar_factor_rank: the rank of covariance factor for gradient component. (this is number of dimensions in Cartesian coordinate.)
    :param: hessian_covar_factor_rank: the rank of covariance factor for hessian component. (this is number of upper triangle components in Cartesian coordinate.)
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
        hessian_noise_constraint: Optional[Interval] = None, 
        has_covar_factor: bool= False,
        noise_covar_factor_pot_grad_array: torch.Tensor= torch.tensor([]),
        noise_covar_factor_with_hessian_array: torch.Tensor= torch.tensor([]),
        grad_covar_factor_rank: int= 0,
        hessian_covar_factor_rank: int= 0
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
        self.has_covar_factor = has_covar_factor
        
        if not has_covar_factor:
            # The case that the noise of potential, gradient & hessian are (sigma_V)^2 I, (sigma_g)^2 I, (sigma_H)^2 I
            pass
        else:
            # the case that the covariance matrix of the noise of gradient & hessian is the product of covariance factor:  covar_factor @ covar_factor ^T
            if grad_covar_factor_rank == 0 or hessian_covar_factor_rank == 0:
                raise("Must provides the rank of covariance factor for gradient noise & hessian noise if the noise have covariant factor.")
            self.grad_covar_factor_rank = grad_covar_factor_rank
            self.hessian_covar_factor_rank = hessian_covar_factor_rank
            self.rank = 1 + grad_covar_factor_rank + hessian_covar_factor_rank  # total rank (number of columns) for noise matrix of pot + grad + hessian. 

            # check the shape of covar_factor_matrix
            self.noise_covar_factor_pot_grad_array = noise_covar_factor_pot_grad_array
            self.noise_covar_factor_with_hessian_array = noise_covar_factor_with_hessian_array 
            if len(noise_covar_factor_pot_grad_array) != 0:
                row_size = noise_covar_factor_pot_grad_array[0].shape[0]
                column_size = noise_covar_factor_pot_grad_array[0].shape[1]
                assert row_size == 1 + ndof, "the number of rows for covar_factor_pot_grad should be the same as the size of gradient in gpr model"
                assert column_size == 1 + grad_covar_factor_rank, "the number of columns for covar_factor_pot_grad should be the same as the rank of gradient."

            if len(noise_covar_factor_with_hessian_array) != 0:
                row_size = noise_covar_factor_with_hessian_array[0].shape[0]
                column_size = noise_covar_factor_with_hessian_array[0].shape[1]
                assert row_size == 1 + ndof + hessian_triu_size,  "the number of rows for covar_factor_with_hessian should be the same as the size of gradient & hessian in gpr model"
                assert column_size == 1 + grad_covar_factor_rank + hessian_covar_factor_rank, "the number of columns for covar_factor_with_hessian should be the same as the rank of gradient & hessian."

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
            name= "raw_force_noises", parameter= torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        )
        self.register_constraint("raw_force_noises", force_noise_constraint)
        if force_noise_prior is not None:
            self.register_prior("raw_force_noises_prior", force_noise_prior, lambda m: m.force_noises)

        # register hessian noises, the constraint for the hessian noise and the prior for the hessian noise 
        # following the convention in gpytorch, here hessian_noises are variances of noise 
        self.register_parameter(
            name= "raw_hessian_noises", parameter= torch.nn.Parameter(torch.zeros(*batch_shape, 1))
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
    
    def update_noise_covar_factor_array(self,
                                        new_noise_covar_factor_pot_grad_array: torch.Tensor,
                                        new_noise_covar_factor_with_hessian_array: torch.Tensor):
        '''
        update the noise_covar_factor_array, which transform the noise from Cartesian coordinate into internal coordinate.
        Note here: for each data point, it has its own noise_covar_factor_pot_grad & noise_covar_factor_with_hessian (if the data point contains hessian information).
        Therefore, when we add new data into Gaussian Process Regression model, we need to update the noise_covar_factor array.
        :param: new_noise_covar_factor_pot_grad_array: the covar factor matrix for transformation of gradient noise for the new data point.
        :param: new_noise_covar_factor_with_hessian_array: the covar factor matrix for transformation of gradient + hessian noise for the new data point.
        '''
        if len(new_noise_covar_factor_pot_grad_array) != 0:
                row_size = new_noise_covar_factor_pot_grad_array[0].shape[0]
                column_size = new_noise_covar_factor_pot_grad_array[0].shape[1]
                assert row_size == 1 + self.ndof, "the number of rows for covar_factor_pot_grad should fit the size of gradient in gpr model"
                assert column_size == 1 + self.grad_covar_factor_rank, "the number of columns for covar_factor_pot_grad should fit the rank of gradient."

        if len(new_noise_covar_factor_with_hessian_array) != 0:
                row_size = new_noise_covar_factor_with_hessian_array[0].shape[0]
                column_size = new_noise_covar_factor_with_hessian_array[0].shape[1]
                assert row_size == 1 + self.ndof + self.hessian_triu_size,  "the number of rows for covar_factor_with_hessian should fit the size of gradient & hessian in gpr model"
                assert column_size == 1 + self.grad_covar_factor_rank + self.hessian_covar_factor_rank, "the number of columns for covar_factor_with_hessian should fit the rank of gradient & hessian."

        self.noise_covar_factor_pot_grad_array = torch.concat([self.noise_covar_factor_pot_grad_array, new_noise_covar_factor_pot_grad_array])
        self.noise_covar_factor_with_hessian_array = torch.concat([self.noise_covar_factor_with_hessian_array, new_noise_covar_factor_with_hessian_array])


    def _shaped_noise_covar(
            self,
            M: int, hessian_data_point_index_array: torch.Tensor, 
            *params: Any, **kwargs: Any
    ):
        '''
        :param: M: total number of input data points
        :param: hessian_data_point_index_array: the index for data point contains hessian information. 
        '''
        n_batch_dim = len(self.batch_shape)
        M_H = len(hessian_data_point_index_array)

        assert len(self.noise_covar_factor_pot_grad_array) == M, "the size of noise_covar_factor_pot_grad_array is not right. Do you forget to update it when adding new data?"
        assert len(self.noise_covar_factor_with_hessian_array) == M_H, "the size of noise_covar_factor_with_hessian_array is not right. Do you forget to update it when adding new data?"

        if not self.has_covar_factor:
            # The noise matrix is diagonal.
            pot_noises_var = self.pot_noises.repeat([ *([1] * n_batch_dim), M])  # shape: [M]
            force_noises_var = self.force_noises.repeat([ *([1] * n_batch_dim), M * self.ndof ])  # shape: [M * d]
            hessian_noises_var = self.hessian_noises.repeat([ *([1] * n_batch_dim), M_H * self.hessian_triu_size ])  # shape: [M_H * hessian_triu_size]
            noises_var = torch.concat( (pot_noises_var, force_noises_var, hessian_noises_var), dim= -1)
            matrix_size = M + M * self.ndof + M_H * self.hessian_triu_size 
            noise_covar_matrix = torch.zeros([matrix_size, matrix_size])
            diag_index = np.arange(matrix_size)
            noise_covar_matrix[diag_index, diag_index] = noises_var
        else:
            # Covariance matrix of the noise : covar_factor * Diag(pot_noise_var, force_noise_var, hessian_noise_var) * covar_factor
            pot_noises_std = torch.sqrt(self.pot_noises).type(self.noise_covar_factor_pot_grad_array.dtype)  # shape [1]
            force_noises_std = torch.sqrt(self.force_noises).repeat([self.grad_covar_factor_rank]).type(self.noise_covar_factor_pot_grad_array.dtype)  # shape: [grad_rank]
            hessian_noises_std = torch.sqrt(self.hessian_noises).repeat([self.hessian_covar_factor_rank]).type(self.noise_covar_factor_pot_grad_array.dtype)  # shape: [hessian_rank]

            matrix_size = M + M * self.ndof + M_H * self.hessian_triu_size  # the size of covariance matrix.  It is also number of rows for covar_factor
            covar_factor_rank_size = M + M * self.grad_covar_factor_rank + M_H * self.hessian_covar_factor_rank  # the column size of covar_factor
            noise_covar_matrix = torch.zeros([matrix_size, matrix_size], dtype= self.noise_covar_factor_pot_grad_array.dtype)  # the covariance matrix of noise. 
            noise_covar_factor_all_data = torch.zeros([matrix_size, covar_factor_rank_size], dtype= self.noise_covar_factor_pot_grad_array.dtype) # the covariance factor of noise. 

            # weight of covariance factor of the noise covariance matrix. This is noise in Cartesian coordinate. 
            noise_covar_factor_weight = torch.zeros([self.rank, self.rank], dtype= self.noise_covar_factor_pot_grad_array.dtype)  # the standard deviation of potential, gradient & hessian noise in Cartesian coordinate.
            noise_covar_factor_weight_diag = torch.concatenate([pot_noises_std, force_noises_std, hessian_noises_std])
            noise_covar_factor_weight[torch.arange(self.rank), torch.arange(self.rank)] = noise_covar_factor_weight_diag 
            
            # the noise for potential and gradient in Cartesian coordinate
            noise_covar_factor_weight_pot_grad = noise_covar_factor_weight[: 1 + self.grad_covar_factor_rank, : 1 + self.grad_covar_factor_rank]
            
            hessian_data_point_index = -1
            for data_point_index in range(M):
                pot_row_index = np.array([data_point_index])
                grad_row_index = np.arange(0, self.ndof) + (M + data_point_index * self.ndof)
                
                pot_column_index = np.array([data_point_index])
                grad_column_index = np.arange(0, self.grad_covar_factor_rank) + (M + data_point_index * self.grad_covar_factor_rank)                

                if data_point_index in hessian_data_point_index_array:
                    # the data point index in the hessian_data_point_index_array.
                    hessian_data_point_index = int(torch.argwhere(data_point_index == hessian_data_point_index_array)[0][0])
                    # the row and column index in noise_covar_factor matrix that corresponds to single data point.
                    hessian_row_index = np.arange(0, self.hessian_triu_size) + (M * (self.ndof + 1) + hessian_data_point_index * self.hessian_triu_size)
                    row_index = np.concatenate([ pot_row_index, grad_row_index, hessian_row_index])

                    hessian_column_index = np.arange(0, self.hessian_covar_factor_rank) + (M * (1 + self.grad_covar_factor_rank) + hessian_data_point_index * self.hessian_covar_factor_rank)
                    column_index = np.concatenate([pot_column_index, grad_column_index, hessian_column_index])

                    two_dimensional_index = np.meshgrid(row_index, column_index, indexing= 'ij')
                    noise_covar_factor = self.noise_covar_factor_with_hessian_array[hessian_data_point_index]  # covariance factor with hessian info.
                    noise_covar_factor_all_data[two_dimensional_index[0], two_dimensional_index[1]] = torch.matmul(noise_covar_factor, noise_covar_factor_weight)
                else:
                    # the row and column index in noise_covar_factor matrix that corresponds to single data point.
                    row_index = np.concatenate([ pot_row_index, grad_row_index ])
                    column_index = np.concatenate([ pot_column_index, grad_column_index])

                    two_dimensional_index = np.meshgrid(row_index, column_index, indexing= 'ij')
                    
                    noise_covar_factor_pot_grad = self.noise_covar_factor_pot_grad_array[data_point_index]
                    noise_covar_factor_all_data[two_dimensional_index[0], two_dimensional_index[1]] = torch.matmul(noise_covar_factor_pot_grad, noise_covar_factor_weight_pot_grad)

            # the covariance matrix of noise is covar_factor @ (covar_factor)^T
            noise_covar_matrix = torch.matmul(noise_covar_factor_all_data, noise_covar_factor_all_data.transpose(-1, -2))    
        
        return noise_covar_matrix 


    def forward(self, function_samples: Tensor, *params, **kwargs: Any):
        '''
        compute the conditional probability p(y|f(x)) of the tensor sample.
        :param: M: number of data points in the 1d data.
        :param: hessian_data_point_index_array: The index of data point that contains hessian information.
        '''
        if len(params) != 2:
            print("Must pass M & M_H into likelihood function.")

        M = params[0]
        hessian_data_point_index_array = params[1]
        
        noise = self._shaped_noise_covar(M, hessian_data_point_index_array).diagonal(dim1= -1, dim2= -2)  # take diagonal part of the matrix.
        return base_distributions.Independent(base_distributions.Normal( function_samples, noise.sqrt()), 1)

    def marginal(self, function_dist: MultivariateNormal, *params : Any, **kwargs: Any) -> MultivariateNormal:
        r"""
        compute the marginal probability p(y|X), marginalize over f(X).
        the covariance matrix will be K + noise_covar_matrix.
        :param: function_dist: latent function distribution f(X). MultivariateNormal distribution.
        :param: M: number of data points in the 1d data.
        :param: hessian_data_point_index_array: The index of data point that contains hessian information.
        """
        if len(params) != 2:
            print("Must pass M & M_H into likelihood function.")

        M = params[0]
        hessian_data_point_index_array = params[1]
        # covar: covariance matrix of the latent function.
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix

        # ensure that sumKroneckerLT is actually called
        if isinstance(covar, LazyEvaluatedKernelTensor):
            covar = covar.evaluate_kernel()
        
        # compute the covariance matrix of the noise.
        noise_covar = self._shaped_noise_covar(
            M, hessian_data_point_index_array
        )
        
        full_covar = covar + noise_covar 

        return function_dist.__class__(mean, full_covar)