import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
import torch
from torch import Tensor
import math

class CustomMarginalLogLikelihood(ExactMarginalLogLikelihood):
    """
    simple extension of ExactMarginalLogLikelihood class.
    We change the __forward__() function since num_data is different for our case (not function_dist.event_shape.numel() in ExactMarginalLogLikelihood)
    """

    def __init__(self, likelihood, model):
        super(CustomMarginalLogLikelihood, self).__init__(likelihood, model)

    def forward(self, function_dist, target, *params):
        """
        adopted from forward function in ExactMarginalLogLikelihood
        """
        if not isinstance(function_dist, MultivariateNormal):
            raise RuntimeError(
                "ExactMarginalLogLikelihood can only operate on Gaussian random variables"
            )

        # Get the log prob of the marginal distribution
        output = self.likelihood(
            function_dist, *params
        )  # output is the multitaskmultivariate normal distribution with noise var add to variance.
        
        res = self.log_prob_likelihood(output, target) # compute the log probability of training data in Gaussian Process Regression model.

        res = self._add_other_terms(
            res, params
        )  # add penality term from prior distribution of parameters. (See gpytorch.mlls.exact_marginal_log_likelihood.py)

        # Scale by the amount of data we have
        num_data = params[0]
        res = res.div(num_data)

        return res

    def log_prob_likelihood(self, normal_dist: MultivariateNormal, value: Tensor, singular_value_cutoff = pow(10.0, -8)):
        """
        compute the log probability of observable (target).
        Perform the pseudo-inverse when the covariance matrix is ill-conditioned.
        See log_prob function in multitask_multivariate_normal.py & multivariate_normal.py in gpytorch.distributions
        log(p) = -1/2 ( y^t (K + sigma^2 I)^-1 y + log|K + sigma^2 I| + n * log(2* pi) )
        """
        mean, covar = normal_dist.loc, normal_dist.lazy_covariance_matrix
        diff = value - mean 

        # get log determinant and first part of quadratic form
        covar_tensor = covar.to_dense() 
        logdet = torch.logdet(covar_tensor) # log(|K + sigma^2 I|)
        
        pseudo_inverse_covar = torch.linalg.pinv(covar_tensor, rtol= singular_value_cutoff) 
        inv_quad = diff @ pseudo_inverse_covar @ diff # y^t (K+ sigma^2 I)^-1 y

        res = -0.5 * sum([inv_quad, logdet, diff.size(-1) * math.log(2 * math.pi) ])
        return res 