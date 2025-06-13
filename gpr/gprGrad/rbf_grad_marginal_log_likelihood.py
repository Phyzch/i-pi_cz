import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
import torch
from torch import Tensor
import math 

class RBFGradMarginalLogLikelihood(ExactMarginalLogLikelihood):
    """
    simple extension of ExactMarginalLogLikelihood class.
    We change the forward() function since we want to change the way we compute logarithm of probability.
    """

    def __init__(self, likelihood, model):
        super(RBFGradMarginalLogLikelihood, self).__init__(likelihood, model)


    def forward(self, function_dist, target, *params):
        r"""
        Computes the MLL given :math:`p(\mathbf f)` and :math:`\mathbf y`.

        :param ~gpytorch.distributions.MultivariateNormal function_dist: :math:`p(\mathbf f)`
            the outputs of the latent function (the :obj:`gpytorch.models.ExactGP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :rtype: torch.Tensor
        :return: Exact MLL. Output shape corresponds to batch shape of the model/input data.
        """
        if not isinstance(function_dist, MultivariateNormal):
            raise RuntimeError("ExactMarginalLogLikelihood can only operate on Gaussian random variables")

        # Get the log prob of the marginal distribution
        output = self.likelihood(function_dist, *params)  # output is the multitaskmultivariate normal distribution with noise var add to variance.
        
        res = output.log_prob(target)   
        res = self._add_other_terms(res, params)

        # Scale by the amount of data we have
        num_data = function_dist.event_shape.numel()
        return res.div_(num_data)
    
