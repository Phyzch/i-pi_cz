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
        
        res = output.log_prob(target)  # compute the log probability of training data in Gaussian Process Regression model.

        res = self._add_other_terms(
            res, params
        )  # add penality term from prior distribution of parameters. (See gpytorch.mlls.exact_marginal_log_likelihood.py)

        # Scale by the amount of data we have
        num_data = params[0]
        res = res.div(num_data)

        return res

