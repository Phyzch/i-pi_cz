import torch
from gpytorch.means.mean import Mean 
from typing import Any, Optional
from gpytorch.priors import Prior 
from gpytorch.constraints import Interval 

class ConstantMeanHessian(Mean):
    '''
    module that represents the mean function for data with Hessian information
    the mean function is a constant value.
    '''
    def __init__(self, constant_prior: Optional[Prior] = None,
                  constant_constraint: Optional[Interval] = None, 
                  batch_shape= torch.Size(), **kwargs):
        super(ConstantMeanHessian, self).__init__()
        self.batch_shape = batch_shape 
        self.register_parameter(name="raw_constant", parameter= torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        
        if constant_prior is not None:
            self.register_prior("mean_prior", constant_prior, "constant")
        if constant_constraint is not None:
            self.register_constraint("raw_constant", constant_constraint)
    
    @property 
    def constant(self):
        if hasattr(self, "raw_constant_constraint"):
            return self.raw_constant_constraint.transform(self.raw_constant)
        return self.raw_constant
    
    @constant.setter 
    def constant(self, value: torch.Tensor):
        if hasattr(self, "raw_constant_constraint"):
            self.initialize(raw_constant= self.raw_constant_constraint.inverse_transform(value))
        else:
            self.initialize(raw_constant= value)
    
    def forward(self, input, M_H= None, nactive= None):
        '''
        input shape: [M, d]
        mean function shape: [V1, .., V^(N), dV^(1)/dx, .., dV^(N)/dx, dV^(h_1)/dx^2, .., dV^(h_MH)/dx^2 ]
        :param: M_H: number of data points with Hessian information.
        :param: nactive: active dimensions for computing hessian.
        '''
        batch_shape = torch.broadcast_shapes(self.batch_shape, input.shape[:-2]) 

        M = input.size(-2)
        d = input.size(-1)

        # size of data points for function, gradient and hessian information.
        func_size = M 
        grad_size = M * d
        hessian_triu_size = int(M_H * nactive * (nactive + 1) / 2)

        total_size = int(func_size + grad_size + hessian_triu_size)

        mean = self.constant.expand(*batch_shape, total_size).contiguous() 
        mean[..., func_size: ] = 0 

        return mean 
    
    def __call__(self, x: torch.Tensor, **kwargs):
        # overwrite the __call__() function in gpytorch.mean.mean.py 
        if x.ndimension() == 1:
            x = x.unsqueeze(1)
        
        res = super(Mean, self).__call__(x, **kwargs)

        return res 
