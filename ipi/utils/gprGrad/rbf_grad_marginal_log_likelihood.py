import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
import torch
from torch import Tensor
import math 

class RBFGradMarginalLogLikelihood(ExactMarginalLogLikelihood):
    """
    simple extension of ExactMarginalLogLikelihood class.
    We change the __forward__() function since num_data is different for our case (not function_dist.event_shape.numel() in ExactMarginalLogLikelihood)
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
        res = self.log_prob_likelihood(output, target)
        # res1 = output.log_prob(target)   # old code
        res = self._add_other_terms(res, params)

        # Scale by the amount of data we have
        num_data = function_dist.event_shape.numel()
        return res.div_(num_data)
    
    def log_prob_likelihood(self, normal_dist: MultitaskMultivariateNormal, value: Tensor, singular_value_cutoff = pow(10.0, -4)):
        """
        compute the log probability of observable (target).
        Perform the pseudo-inverse when the covariance matrix is ill-conditioned.
        See log_prob function in multitask_multivariate_normal.py & multivariate_normal.py in gpytorch.distributions
        log(p) = -1/2 ( y^t (K + sigma^2 I)^-1 y + log|K + sigma^2 I| + n * log(2* pi) )
        """
        if not normal_dist._interleaved:
            new_shape = value.shape[:-2] + value.shape[:-3:-1]
            value = value.view(new_shape).transpose(-1, -2).contiguous() 
        value = value.reshape(*value.shape[:-2], -1)

        mean, covar = normal_dist.loc, normal_dist.lazy_covariance_matrix
        diff = value - mean 

        # get log determinant and first part of quadratic form
        covar_tensor = covar.to_dense() 
        logdet = torch.logdet(covar_tensor) # log(|K + sigma^2 I|)
        
        pseudo_inverse_covar = torch.linalg.pinv(covar_tensor, atol= singular_value_cutoff) 
        inv_quad = diff @ pseudo_inverse_covar @ diff # y^t (K+ sigma^2 I)^-1 y

        res = -0.5 * sum([inv_quad, logdet, diff.size(-1) * math.log(2 * math.pi) ])
        return res 
