import gpytorch 
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
import torch 

class CustomMarginalLogLikelihood(ExactMarginalLogLikelihood):
    '''
    simple extension of ExactMarginalLogLikelihood class.
    '''
    def __init__(self, likelihood, model):
        super(CustomMarginalLogLikelihood, self).__init__(likelihood, model)
    
    def forward(self, function_dist, target, *params):
        '''
        adopted from forward function in ExactMarginalLogLikelihood
        '''
        if not isinstance(function_dist, MultivariateNormal):
            raise RuntimeError("ExactMarginalLogLikelihood can only operate on Gaussian random variables")

        # Get the log prob of the marginal distribution
        output = self.likelihood(function_dist, *params)  # output is the multitaskmultivariate normal distribution with noise var add to variance.
        res = output.log_prob(target) 
        res_log_prob = res.clone() 
        res = self._add_other_terms(res, params)
        res_added_term = res - res_log_prob

        # Scale by the amount of data we have
        M = params[0]
        num_data = M
        res = res.div(num_data)
        
        return res