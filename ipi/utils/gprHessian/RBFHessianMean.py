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

class MeanWithPotGradHessian(Mean):
    '''
    module that represents the mean function for data with Hessian information.
    The mean function will have a constant potential value, gradient value and hessian value.
    '''
    def __init__(self, constant_prior: Optional[Prior] = None, 
                        constant_constraint: Optional[Interval] = None,
                        grad_prior: Optional[Prior] = None, 
                        grad_constraint: Optional[Interval] = None,
                        hessian_prior: Optional[Prior] = None,
                        hessian_constraint: Optional[Interval] = None,
                        batch_shape= torch.Size(),
                        grad_size: int= 0,
                        hessian_size: int= 0,
                        **kwargs):
        super(MeanWithPotGradHessian, self).__init__()
        self.batch_shape = batch_shape 
        # set mean value constant
        self.register_parameter(name= "raw_constant", parameter= torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        if constant_prior is not None:
            self.register_prior("mean_prior", constant_prior, "constant")
        if constant_constraint is not None:
            self.register_constraint("raw_constant", constant_constraint)

        assert grad_size != 0 and hessian_size != 0, "Please provide the size of gradient and hessian data."
        
        self.grad_size = grad_size 
        self.hessian_size = hessian_size

        # set gradient value
        self.register_parameter(name= "raw_grad", parameter= torch.nn.Parameter(torch.zeros(*batch_shape, grad_size)))
        if grad_prior is not None:
            self.register_prior("grad_prior", grad_prior, "grad")
        if grad_constraint is not None:
            self.register_constraint("raw_grad", grad_constraint)

        # set hessian value
        self.register_parameter(name= "raw_hessian", parameter= torch.nn.Parameter(torch.zeros(*batch_shape, hessian_size)))
        if hessian_prior is not None:
            self.register_prior("hessian_prior", hessian_prior, "hessian")
        if hessian_constraint is not None:
            self.register_constraint("raw_hessian", hessian_constraint)

    @property 
    def constant(self):
        if hasattr(self, "raw_constant_constraint"):
            return self.raw_constant_constraint.transform(self.raw_constant)
        else:
            return self.raw_constant 
    
    @constant.setter
    def constant(self, value: torch.Tensor):
        if hasattr(self, "raw_constant_constraint"):
            self.initialize(raw_constant= self.raw_constant_constraint.inverse_transform(value))
        else:
            self.initialize(raw_constant= value)
    
    @property 
    def grad(self):
        if hasattr(self, "raw_grad_constraint"):
            return self.raw_grad_constraint.transform(self.raw_grad)
        else:
            return self.raw_grad 
    
    @grad.setter
    def grad(self, value: torch.Tensor):
        assert value.shape == self.raw_grad.shape 
        if hasattr(self, "raw_grad_constraint"):
            self.initialize(raw_grad= self.raw_grad_constraint.inverse_transform(value))
        else:
            self.initialize(raw_grad= value)
    
    @property 
    def hessian(self):
        if hasattr(self, "raw_hessian_constraint"):
            return self.raw_hessian_constraint.transform(self.raw_hessian)
        else:
            return self.raw_hessian 
    
    @hessian.setter 
    def hessian(self, value):
        assert value.shape == self.raw_hessian.shape 
        if hasattr(self, "raw_hessian_constraint"):
            self.initialize(raw_hessian= self.raw_hessian_constraint.inverse_transform(value))
        else:
            self.initialize(raw_hessian= value)
    
    def forward(self, input, M_H, nactive):
        '''
        function that return the mean function for 1d data.
        :param: M_H: number of data points with hessian information.
        :param: nactive: number of active dimensions for hessians.
        '''
        batch_shape = torch.broadcast_shapes(self.batch_shape, input.shape[:-2])

        M = input.shape[-2]
        d = input.shape[-1]

        hessian_triu_size = int(nactive * (nactive + 1)) / 2
        assert hessian_triu_size == self.hessian_size 
        assert d == self.grad_size 

        # size of data points for function, gradient and hessian
        func_size = M 
        grad_size = M * d 
        hessian_size = M_H * hessian_triu_size 

        func_mean = self.constant.repeat([*batch_shape, M])
        grad_mean = self.grad.repeat([*batch_shape, M])
        hessian_mean = self.hessian.repeat([*batch_shape, M_H])

        mean = torch.concatenate([func_mean, grad_mean, hessian_mean], dim= -1)

        return mean         
    
    def __call__(self, x: torch.Tensor, **kwargs):
        # overwrite the __call__() function in gpytorch.mean.mean.py
        if x.ndimension() == 1:
            x = x.unsqueeze(1)

        res = super(Mean, self).__call__(x, **kwargs)

        return res

    def set_mean_value(self, func: torch.Tensor , grad: torch.Tensor, hessian: torch.Tensor):
        '''
        set the function and gradient value for mean function
        '''
        assert func.shape == self.constant.shape 
        assert grad.shape == self.grad.shape 
        assert hessian.shape == self.hessian.shape 
                
        self.constant = func 
        self.grad = grad 

        self.hessian = hessian


