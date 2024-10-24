"""
Change the mean_cache and covar_cache in prediction strategy. Make the inverse of covariance matirx as pseudo-inverse.
The GP regression becomes unstable when covariance matrix is invertible, which is the case when data points get close to each other.
"""

import torch
from gpytorch.models.exact_prediction_strategies import DefaultPredictionStrategy
from gpytorch.utils.memoize import (
    cached,
    clear_cache_hook,
)
from linear_operator import to_dense
from gpytorch import settings 
import functools

class RBFGradPredictionStrategies(DefaultPredictionStrategy):
    def __init__(self, train_inputs, train_prior_dist, train_labels, likelihood, root=None, inv_root=None):
        super(RBFGradPredictionStrategies, self).__init__(train_inputs, train_prior_dist, train_labels, likelihood, root= root, inv_root= inv_root)
    
    @property 
    @cached(name="mean_cache")
    def mean_cache(self, singular_value_cutoff = pow(10.0, -2)):
        """
        compute cache for the prediction of the mean value. 
        use pseudo-inverse (Moore-Penrose inverse) when the covariance matrix becomes ill-defined. (smallest eigenvalue close to 0)
        """
        mvn = self.likelihood(self.train_prior_dist, self.train_inputs)
        train_mean, train_train_covar = mvn.loc, mvn.lazy_covariance_matrix  # covariance matrix of y(x) (likelihood) : K(X,X) + sigma^2 I

        train_labels_offset = (self.train_labels - train_mean).unsqueeze(-1)   # y
        
        # pseudo-inverse the covariance matrix.
        train_train_covar_tensor = train_train_covar.to_dense()
        # compute eigenvalues using lobpcg: faster for sparse matrix.
        # covar_eigval, _ = torch.lobpcg(train_train_covar_tensor, k=1, largest= False)
        covar_eigval = torch.linalg.eigvals(train_train_covar_tensor)
        covar_eigval_min = torch.min(torch.real(covar_eigval))
        covar_eigval_max = torch.max(torch.real(covar_eigval))

        if abs(covar_eigval_min / covar_eigval_max) < singular_value_cutoff:
            # the covariance matrix is ill-conditioned. We should perform the pseudo-inverse 
            covar_inverse = torch.linalg.pinv(train_train_covar_tensor, rtol= singular_value_cutoff) # (K(X,X) + sigma^2 I)^-1 * y
            mean_cache = (covar_inverse @ train_labels_offset).squeeze(-1) 
        else:
            mean_cache = train_train_covar.evaluate_kernel().solve(train_labels_offset).squeeze(-1)  # (K(X,X) + sigma^2 I)^-1 * y
        
        if settings.detach_test_caches.on():
            mean_cache = mean_cache.detach()

        if mean_cache.grad_fn is not None:
            wrapper = functools.partial(clear_cache_hook, self)
            functools.update_wrapper(wrapper, clear_cache_hook)
            mean_cache.grad_fn.register_hook(wrapper)

        return mean_cache 

    @property
    @cached(name="covar_cache")
    def covar_cache(self, singular_value_cutoff = pow(10.0, -2)):
        """
        compute cache for the prediction of the covariance matrix. Which is (K(x,x) + sigma^2 I)^{-1/2}
        use pseudo-inverse (Moore-Penrose inverse) when the covariance matrix becomes ill-conditioned. (smallest eigenvalue close to 0)
        """
        train_train_covar = self.lik_train_train_covar 

        # pseudo-inverse the covariance matrix
        train_train_covar_tensor = train_train_covar.to_dense()
        # covar_eigval, _ = torch.lobpcg(train_train_covar_tensor, k= 1, largest= False)
        covar_eigval = torch.linalg.eigvals(train_train_covar_tensor)
        covar_eigval_min = torch.min(torch.real(covar_eigval))
        covar_eigval_max = torch.max(torch.real(covar_eigval))

        if abs(covar_eigval_min / covar_eigval_max) < singular_value_cutoff:
            # the covariance matrix is ill-conditioned. We should perform the pseudo-inverse 
            U, S ,Vh = torch.linalg.svd(train_train_covar_tensor)
            nonzero_indices = (S > singular_value_cutoff * covar_eigval_max).nonzero().squeeze(-1)
            nonzero_s = S[nonzero_indices]
            nonzero_u = torch.index_select(U, dim= 1, index= nonzero_indices)
            nonzero_vh = torch.index_select(Vh, dim= 0, index= nonzero_indices)
            nonzero_s_inv_root = 1/ torch.sqrt(nonzero_s)
            nonzero_s_inv_root_matrix = torch.diag(nonzero_s_inv_root)
            train_train_covar_inv_root = nonzero_u @ nonzero_s_inv_root_matrix @ nonzero_vh # (K(x,x) + sigma^2 I)^(-1/2)
        else:
            train_train_covar_inv_root = to_dense(train_train_covar.root_inv_decomposition().root)   # (K(x,x) + sigma^2 I)^(-1/2)

        return train_train_covar_inv_root