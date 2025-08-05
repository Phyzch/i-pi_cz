"""
Provide mean function for Gaussian Process Regression model.
Written by Chenghao Zhang & Niri Govind, Pacific Northwest National Laboratory (chenghao.zhang@pnnl.gov), 2024.
"""

import torch
from gpytorch.means.mean import Mean
from typing import Any, Optional
from gpytorch.priors import Prior
from gpytorch.constraints import Interval
from .RBFHessian_utils import take_upper_triangular_part

class ConstantMeanHessian(Mean):
    """
    module that represents the mean function for data with Hessian information.
    the mean function is a constant value.
    """

    def __init__(
        self,
        constant_prior: Optional[Prior] = None,
        constant_constraint: Optional[Interval] = None,
        batch_shape=torch.Size(),
        **kwargs
    ):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        super(ConstantMeanHessian, self).__init__()
        self.batch_shape = batch_shape
        self.register_parameter(
            name="raw_constant",
            parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1, device= self.device)),
        )
        # register prior and constraint for constant.
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
            self.initialize(
                raw_constant=self.raw_constant_constraint.inverse_transform(value)
            )
        else:
            self.initialize(raw_constant=value)

    def forward(self, input, hessian_data_point_index, nactive=None):
        """
        input shape: [M, d]
        mean function shape: [V1, .., V^(N), dV^(1)/dx, .., dV^(N)/dx, dV^(h_1)/dx^2, .., dV^(h_MH)/dx^2 ]
        :param: M_H: number of data points with Hessian information.
        :param: nactive: active dimensions for computing hessian.
        """
        batch_shape = torch.broadcast_shapes(self.batch_shape, input.shape[:-2])
        M_H = len(hessian_data_point_index)  # number of data ponits with hessian

        M = input.size(-2)  # number of data points
        d = input.size(-1)  # number of degrees of freedom

        # size of data points for function, gradient and hessian information.
        func_size = M
        grad_size = M * d
        hessian_triu_size = int(M_H * nactive * (nactive + 1) / 2)

        total_size = int(func_size + grad_size + hessian_triu_size)

        mean = self.constant.expand(*batch_shape, total_size).contiguous()
        mean[..., func_size:] = 0  # the gradient & hessians are 0.

        return mean

    def __call__(self, x: torch.Tensor, **kwargs):
        # overwrite the __call__() function in gpytorch.mean.mean.py
        if x.ndimension() == 1:
            x = x.unsqueeze(1)

        res = super(Mean, self).__call__(x, **kwargs)

        return res

# compute hessian from upper triangle components of hessian.
def convert_hessian_triu_to_hessian(batch_shape, grad_size, ref_hessian_upper_triangle: torch.Tensor):
    """
    convert the upper triangle part of the hessian matrix to the hessian matrix.
    """
    ref_hessian_triu = torch.zeros(*batch_shape, grad_size * grad_size).type(
        ref_hessian_upper_triangle.dtype
    ).to(device= ref_hessian_upper_triangle.device)
    triu_indices = torch.triu_indices(grad_size, grad_size).to(device= ref_hessian_upper_triangle.device)
    triu_1d_indices = triu_indices[0] * grad_size + triu_indices[1]
    ref_hessian_triu[..., triu_1d_indices] = ref_hessian_upper_triangle
    ref_hessian_triu = ref_hessian_triu.reshape(*batch_shape, grad_size, grad_size)
    if ref_hessian_upper_triangle.dim() == 2:
        ref_hessian = (
            ref_hessian_triu
            + torch.transpose(ref_hessian_triu, -1, -2)
            - torch.diag(ref_hessian_triu.diag())
        )
    elif ref_hessian_upper_triangle.dim() == 3:
        mask = torch.eye(grad_size, device= ref_hessian_upper_triangle.device).unsqueeze(0).expand(*batch_shape, -1, -1)
        ref_hessian_diagonal_term = mask * ref_hessian_triu 
        ref_hessian = (
            ref_hessian_triu 
            + torch.transpose(ref_hessian_triu, -1, -2)
            - ref_hessian_diagonal_term
        )
    else:
        raise ValueError(f"Incorrect dimension of upper triangle matrix. Expect 2 or 3. Get {ref_hessian_upper_triangle.dim()}")    
    return ref_hessian 

class MeanTaylorExpansion(Mean):
    """
    module that represents the mean function for data with Hessian information.
    The mean function will be Taylor expansion around the reference point.
    V(x) = V(x0) + V'(x0) (x-x0) + 1/2 * V''(x0) (x-x0)^2.
    V'(x) = V'(x0) + V''(x0) (x-x0)
    V''(x) = V''(x0)

    :param: ref_coordinate:  x0.
    :param: ref_grad:  V'(x0)
    :param: ref_hessian_upper_triangle: upper triangle component of V''(x0)
    :param: batch_shape: shape of batch. In the current implementation, this is [].
    :param: grad_size: the size of gradient
    :param: hessian_triu_size: the size of upper triangle component of hessian.
    """

    def __init__(
        self,
        ref_coordinate: torch.Tensor,
        ref_pot: torch.Tensor,
        ref_grad: torch.Tensor,
        ref_hessian_upper_triangle: torch.Tensor,
        grad_size: int = 0,
        hessian_triu_size: int = 0,
        **kwargs
    ):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        super(MeanTaylorExpansion, self).__init__()
        self.grad_size = grad_size
        self.hessian_triu_size = hessian_triu_size

        assert ref_pot.shape[0] == 1
        assert ref_grad.shape[0] == grad_size
        assert ref_hessian_upper_triangle.shape[0] == hessian_triu_size

        self.ref_coordinate = ref_coordinate
        self.ref_pot = ref_pot
        self.ref_grad = ref_grad

        self.ref_hessian = convert_hessian_triu_to_hessian(batch_shape, grad_size, ref_hessian_upper_triangle)

        self.ref_hessian_upper_triangle = ref_hessian_upper_triangle

    def forward(self, input, hessian_data_point_index, nactive):
        """
        function that return the mean function for 1d data.
        :param: hessian_data_point_index: the index for data point with hessian information.
        :param: nactive: number of active dimensions for hessians.
        """
        M_H = len(hessian_data_point_index)
        M = input.shape[-2]
        d = input.shape[-1]

        hessian_triu_size = int(nactive * (nactive + 1)) / 2
        assert hessian_triu_size == self.hessian_triu_size
        assert d == self.grad_size

        # size of data points for function, gradient and hessian.
        func_size = M
        grad_size = M * d
        hessian_size = M_H * hessian_triu_size

        # the Tylor expansion around the reference point.
        displacement = input - self.ref_coordinate
        # V(x) = V(x0) + V'(x0) (x-x0) + 0.5 * V''(x0) * (x-x0)^2
        func_mean = (
            self.ref_pot.repeat([M])
            + torch.sum(self.ref_grad * displacement, dim=-1)
            + 0.5
            * torch.sum(
                torch.matmul(displacement, self.ref_hessian) * displacement, axis=-1
            )
        )
        # V'(x) = V'(x0) + V''(x0) (x-x0)
        grad_mean = self.ref_grad.repeat([M, 1]) + torch.matmul(
            displacement, self.ref_hessian
        )
        grad_mean = grad_mean.reshape([M * d])
        # V''(x) = V''(x0)
        hessian_mean = self.ref_hessian_upper_triangle.repeat([M_H])

        mean = torch.concatenate([func_mean, grad_mean, hessian_mean], dim=-1)

        return mean

    def __call__(self, x: torch.Tensor, **kwargs):
        # overwrite the __call__() function in gpytorch.mean.mean.py
        # this is to pass parameter to forward() function. 
        if x.ndimension() == 1:
            x = x.unsqueeze(1)

        res = super(Mean, self).__call__(x, **kwargs)

        return res

# --- Code belows are for the mean function that interpolate between Taylor expansion around reactant and potential maximal. --- 
class StepSmoothFunc():
    """
    step smooth function to interpolate smoothly between 0 & 1.
    f(r) = 6 r^5 - 15 r^4 + 10 * r^3.
    This function satisfy: f(r) = 0, f'(r) = 0, f''(r) = 0 at r = 0, 1.
    """
    def __init__(self):
        pass 

    def __call__(self, r):
        """
        compute the function f(r)
        :return: torch.Tensor() 
        """
        if r < 0 or r > 1:
            raise ValueError(f"r should be in the range [0,1]. Get value: {r}")
        func = 6 * torch.pow(r, 5) - 15 * torch.pow(r, 4) + 10 * torch.pow(r, 3)
        return func 
    
    def deriv(self, r):
        """
        compute the first derivative f'(r)
        :return: torch.Tensor() 
        """
        if r < 0 or r > 1:
            raise ValueError(f"r should be in the range [0,1]. Get value: {r}")
        deriv =  30 * torch.pow(r, 4) - 60 * torch.pow(r ,3) + 30 * torch.pow(r, 2)
        return deriv
    
    def second_deriv(self, r):
        """
        compute the second derivative f''(r).
        :return: torch.Tensor() 
        """
        if r < 0 or r > 1:
            raise ValueError(f"r should be in the range [0,1]. Get value: {r}")
        second_deriv = 120 * torch.pow(r, 3) - 180 * torch.pow(r, 2) + 60 * r 
        return second_deriv

class ReactionCoordProj():
    """
    Project internal coordinate q to the 1d reaction coordinate.
    We do interpolation between reactant 1 (r1), instanton path pot maximum (b) and reactant 2 (r2)
    """
    def __init__(self, q_r1, q_b, q_r2):
        """
        :param: q_r1, q_b, q_r2: internal coordinate of reactant 1 (r1), pot maximum (b) and reactant 2 (r2). 
        """
        self.q_r1 = q_r1
        self.q_r2 = q_r2 
        self.q_b = q_b 
    
    def r(self, q):
        """
        compute the projection along the reaction coordinate (r) for internal coordinate q. 
        """
        sign = torch.dot(q - self.q_b, self.q_r1 - self.q_b) 
        if sign > 0:
            # reactant 1 side.
            r = torch.dot(q - self.q_b, self.q_r1 - self.q_b) / torch.pow(torch.linalg.norm(self.q_r1 - self.q_b), 2)
            if r > 1:
                raise ValueError(f"r should be in the range [0,1]. Get value: {r}")

        else:
            r = torch.dot(q - self.q_b, self.q_r2 - self.q_b) / torch.pow(torch.linalg.norm(self.q_r2 - self.q_b), 2) 
            if r > 1:
                raise ValueError(f"r should be in the range [0,1]. Get value: {r}")

        return r 
    
    def grad_r(self, q):
        """
        compute the gradient of r for internal coordinate q. 
        """
        sign = torch.dot(q - self.q_b, self.q_r1 - self.q_b) 
        if sign > 0:
            # reactant 1 side.
            dr = (self.q_r1 - self.q_b) / torch.pow(torch.linalg.norm(self.q_r1 - self.q_b), 2)
        else:
            # reactant 2 side.
            dr = (self.q_r2 - self.q_b) / torch.pow(torch.linalg.norm(self.q_r1 - self.q_b), 2)
        
        return dr 

class InterpolatedPotential():
    """
    interpolate potential between reactant 1, instanton path pot maximum and reactant 2.
    """
    def __init__(self,
                 ref_q_list,
                 ref_pot_list,
                 ref_grad_list,
                 ref_hessian_list):
        """
        ref_q_list: [q_r1, q_b, q_r2]
        ref_pot_list = [V(q_r1), V(q_b), V(q_r2)].
        similar data structure for ref_grad_list, ref_hessian_list.
        """
        [q_r1, q_b, q_r2] = ref_q_list 
        
        self.q_r1 = q_r1
        self.q_b = q_b 
        self.q_r2 = q_r2 

        self.reaction_coord_proj = ReactionCoordProj(q_r1, q_b, q_r2)
        self.step_smooth_func = StepSmoothFunc()

        self.coord_dict = {"q_r1": q_r1,
                      "q_b": q_b,
                      "q_r2": q_r2}
        
        self.pot_dict = {
            "q_r1": ref_pot_list[0],
            "q_b": ref_pot_list[1],
            "q_r2": ref_pot_list[2]
        }

        self.grad_dict = {
            "q_r1" : ref_grad_list[0],
            "q_b": ref_grad_list[1],
            "q_r2": ref_grad_list[2]
        }

        self.hess_dict = {
            "q_r1": ref_hessian_list[0],
            "q_b": ref_hessian_list[1],
            "q_r2": ref_hessian_list[2]
        }
    
    def pot(self, q, partial_term= False):
        """
        linearly interpolated potential between reactant and potential maximum.
        """
        sign = torch.dot(q - self.q_b, self.q_r1 - self.q_b) 
        if sign > 0:
            # interpolate between reactant 1 and potential maximum.
            q_r_symbol = "q_r1"
        else:
            q_r_symbol = "q_r2"
        q_b_symbol = "q_b"

        q_r = self.coord_dict[q_r_symbol]

        r = self.reaction_coord_proj.r(q)
        # step smooth function f: f(0) = 0, f(1) = 1.
        weight = self.step_smooth_func(r)

        barrier_contribution =  (self.pot_dict[q_b_symbol] 
                                 + torch.matmul(self.grad_dict[q_b_symbol], q - self.q_b)
                                 + torch.sum(torch.matmul(q - self.q_b, self.hess_dict[q_b_symbol])
                                               * (q - self.q_b),
                                               axis= -1  
                                            )
                                 ) 

        reactant_contribution = (self.pot_dict[q_r_symbol]
                                          + torch.matmul(self.grad_dict[q_r_symbol], q - q_r)
                                          + torch.sum(
                                              torch.matmul(q - q_r, self.hess_dict[q_r_symbol]) 
                                              * (q - q_r),
                                              axis= -1                                              
                                            )
                                          )
        
        result = (1 - weight) * barrier_contribution + weight * reactant_contribution

        if not partial_term:
            return result 
        else:
            return barrier_contribution, reactant_contribution

    def gradient(self, q, partial_term= False):
        """
        linearly interpolate gradient between reactant and the potential maximum.
        """
        sign = torch.dot(q - self.q_b, self.q_r1 - self.q_b) 
        if sign > 0:
            # interpolate between reactant 1 and potential maximum.
            q_r_symbol = "q_r1"
        else:
            q_r_symbol = "q_r2"
        q_b_symbol = "q_b"

        q_r = self.coord_dict[q_r_symbol]
        r = self.reaction_coord_proj.r(q)
        # step smooth function f: f(0) = 0, f(1) = 1.
        weight = self.step_smooth_func(r)
        # gradient of step smooth interpolation function: \nabla f(r)
        weight_gradient = self.reaction_coord_proj.grad_r(q) * self.step_smooth_func.deriv(r)

        barrier_contribution =  (self.grad_dict[q_b_symbol] + 
                                               torch.matmul(q - self.q_b, self.hess_dict[q_b_symbol]))
        
        reactant_contribution =  (self.grad_dict[q_r_symbol] + 
                                          torch.matmul(q - q_r, self.hess_dict[q_r_symbol]))
        if partial_term:
            return barrier_contribution, reactant_contribution
        else:
            barrier_pot_func, reactant_pot_func = self.pot(q, partial_term= True)
            weight_gradient_term = weight_gradient * (reactant_pot_func - barrier_pot_func)

            result = (1 - weight) * barrier_contribution + weight * reactant_contribution + weight_gradient_term
            return result 

    def hessian(self, q):
        """
        linearly interpolate hessian between reactant and potential maximum.
        """
        sign = torch.dot(q - self.q_b, self.q_r1 - self.q_b) 
        if sign > 0:
            # interpolate between reactant 1 and potential maximum.
            q_r_symbol = "q_r1"
        else:
            q_r_symbol = "q_r2"
        q_b_symbol = "q_b"

        q_r = self.coord_dict[q_r_symbol]
        r = self.reaction_coord_proj.r(q)
        # step smooth function f: f(0) = 0, f(1) = 1.
        weight = self.step_smooth_func(r)

        barrier_contribution = self.hess_dict[q_b_symbol]
        reactant_contribution = self.hess_dict[q_r_symbol]

        # gradient of step smooth interpolation function: \nabla f(r)
        weight_gradient = self.reaction_coord_proj.grad_r(q) * self.step_smooth_func.deriv(r)
        barrier_grad_func , reactant_grad_func = self.gradient(q, partial_term= True)
        weight_gradient_term = 2 * torch.outer(weight_gradient, reactant_grad_func - barrier_grad_func) 
        
        # hessian of step smooth interpolation function: \nabla^2 f(r)
        grad_r = self.reaction_coord_proj.grad_r(q)
        weight_hessian = self.step_smooth_func.second_deriv(r) * torch.outer(grad_r, grad_r)
        barrier_pot_func, reactant_pot_func = self.pot(q, partial_term= True)
        weight_hessian_term = weight_hessian * (reactant_pot_func - barrier_pot_func)

        result = (1 - weight) * barrier_contribution + weight * reactant_contribution + weight_gradient_term + weight_hessian_term

        return result 

class MeanInterpolatedHessian(Mean):
    """
    module that represents the mean function for data with Hessian information.
    The mean function will be interpolation between Taylor expansion around reactant1, potential maximum & reactant2.
    V(x) = (1- f(r)) * (V(x_b) + V'(x_b) (x-x_b) + 1/2 * V''(x_b) (x-x_b)^2) + f(r) * (V(x_r) + V'(x_r) (x-x_r) + V''(x_r) (x - x_r)^2).
    here r is the projection of coordinate along the reaction coordinate. 
    V'(x), V''(x) will be the gradient & Hessian terms. 

    :param: ref_coordinate_list: [x_r1, x_b, x_r2]
    :param: ref_pot_list: [V(x_r1), V(x_b), V(x_r2)]
    :param: ref_grad_list: [V'(x_r1), V'(x_b), V'(x_r2)]
    :param: ref_hessian_upper_triangle_list: List of upper triangle parts of the Hessian matrix.
    :param: grad_size: size of gradient vector.
    :param: hessian_triu_size: the size of upper triangle part of the hessian.
    """
    def __init__(
        self,
        ref_coordinate_list: torch.Tensor,
        ref_pot_list: torch.Tensor,
        ref_grad_list: torch.Tensor,
        ref_hessian_upper_triangle_list: torch.Tensor,
        grad_size: int = 0,
        hessian_triu_size: int = 0,
        **kwargs 
    ):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        super(MeanInterpolatedHessian, self).__init__()
        
        self.grad_size = grad_size
        self.hessian_triu_size = hessian_triu_size

        assert ref_pot_list.shape[0] == 3
        assert ref_grad_list.shape[0] == 3
        assert ref_grad_list.shape[1] == grad_size 
        assert ref_hessian_upper_triangle_list.shape[0] == 3
        assert ref_hessian_upper_triangle_list.shape[1] == hessian_triu_size

        self.ref_coordinate_list = ref_coordinate_list
        self.ref_pot_list = ref_pot_list
        self.ref_grad_list = ref_grad_list 
        
        ref_batch_shape = torch.Size([3])
        self.ref_hessian_list = convert_hessian_triu_to_hessian(ref_batch_shape,
                                                                grad_size,
                                                                ref_hessian_upper_triangle_list)
        
        self.ref_hessian_upper_triangle_list = ref_hessian_upper_triangle_list

        self.interpolated_potential = InterpolatedPotential(ref_coordinate_list,
                                                            ref_pot_list,
                                                            ref_grad_list,
                                                            self.ref_hessian_list)
    
    def forward(self, input, hessian_data_point_index, nactive):
        """
        function that returns the mean function for the 1d data.
        :param: hessian_data_point_index: the index for data point with hessian information.
        :param: nactive: number of active dimensions for hessians.
        """
        M_H = len(hessian_data_point_index)
        M = input.shape[-2]
        d = input.shape[-1]

        hessian_triu_size = int(nactive * (nactive + 1)) / 2
        assert hessian_triu_size == self.hessian_triu_size
        assert d == self.grad_size

        # size of data points for function, gradient and hessians.
        func_data_size = M 
        grad_data_size = M * d 
        hessian_data_size = M_H * hessian_triu_size

        # V(x) = interpolated potential between reactant minimum and potential maxima.
        func_mean_list = torch.zeros(M, device= self.device)
        for i in range(M):
            q = input[i]
            func = self.interpolated_potential.pot(q)
            func_mean_list[i] = func 
        
        grad_mean_list = torch.zeros(grad_data_size, device= self.device)
        for i in range(M):
            q = input[i]
            grad = self.interpolated_potential.gradient(q)
            grad_mean_list[i * d : (i + 1) * d] = grad 

        hessian_mean_list = torch.zeros(hessian_data_size, device= self.device)
        for index, data_index in enumerate(hessian_data_point_index):
            q = input[data_index]
            hessian = self.interpolated_potential.hessian(q)
            # take upper triangle part of hessian matrix.
            hessian_triu =  take_upper_triangular_part(hessian)
            hessian_mean_list[index * hessian_triu_size: (index + 1) * hessian_triu_size] = hessian_triu 
        
        mean = torch.concatenate([func_mean_list, grad_mean_list, hessian_mean_list], dim= -1)

        return mean
    
    def __call__(self, x: torch.Tensor, **kwargs):
        # overwrite the __Call__() function in gpytorch.mean.mean.py 
        # this is to pass parameter to forward() function.
        if x.ndimension() == 1:
            x = x.unsqueeze(1)

        res = super(Mean, self).__call__(x, **kwargs)

        return res