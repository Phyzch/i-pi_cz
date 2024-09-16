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

class RBFGradPredictionStrategies(DefaultPredictionStrategy):
    def __init__(self, train_inputs, train_prior_dist, train_labels, likelihood, root=None, inv_root=None):
        super(RBFGradPredictionStrategies, self).__init__(train_inputs, train_prior_dist, train_labels, likelihood, root= root, inv_root= inv_root)
    
    @property 
    @cached(name="mean_cache")
    def mean_cache(self):
        """
        compute cache for prediction of the mean value. 
        use pseudo-inverse (Moore-Penrose inverse) when the covariance matrix becomes ill-defined.
        """
        mvn = self.likelihood(self.train_prior_dist, self.train_inputs)
        train_mean, train_train_covar = mvn.loc, mvn.lazy_covariance_matrix  # covariance matrix of y(x) (likelihood) : K(X,X) + sigma^2 I

        train_label_offset = (self.train_labels - train_mean).unsqueeze(-1)   # y