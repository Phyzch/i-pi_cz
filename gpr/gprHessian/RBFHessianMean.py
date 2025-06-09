"""
Provide mean function for Gaussian Process Regression model.
Written by Chenghao Zhang & Niri Govind, Pacific Northwest National Laboratory (chenghao.zhang@pnnl.gov), 2024.
"""

import torch
from gpytorch.means.mean import Mean
from typing import Any, Optional
from gpytorch.priors import Prior
from gpytorch.constraints import Interval


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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


class MeanWithPotGradHessian(Mean):
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
        batch_shape=torch.Size(),
        grad_size: int = 0,
        hessian_triu_size: int = 0,
        **kwargs
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super(MeanWithPotGradHessian, self).__init__()
        self.batch_shape = batch_shape
        self.grad_size = grad_size
        self.hessian_triu_size = hessian_triu_size

        assert ref_pot.shape[0] == 1
        assert ref_grad.shape[0] == grad_size
        assert ref_hessian_upper_triangle.shape[0] == hessian_triu_size

        self.ref_coordinate = ref_coordinate
        self.ref_pot = ref_pot
        self.ref_grad = ref_grad

        # compute hessian from upper triangle components of hessian.
        ref_hessian_triu = torch.zeros(*batch_shape, grad_size * grad_size).type(
            ref_hessian_upper_triangle.dtype
        ).to(device= ref_hessian_upper_triangle.device)
        triu_indices = torch.triu_indices(grad_size, grad_size).to(device= ref_coordinate.device)
        triu_1d_indices = triu_indices[0] * grad_size + triu_indices[1]
        ref_hessian_triu[..., triu_1d_indices] = ref_hessian_upper_triangle
        ref_hessian_triu = ref_hessian_triu.reshape(*batch_shape, grad_size, grad_size)
        ref_hessian = (
            ref_hessian_triu
            + torch.transpose(ref_hessian_triu, -1, -2)
            - torch.diag(ref_hessian_triu.diag())
        )

        self.ref_hessian = ref_hessian

        self.ref_hessian_upper_triangle = ref_hessian_upper_triangle

    def forward(self, input, hessian_data_point_index, nactive):
        """
        function that return the mean function for 1d data.
        :param: hessian_data_point_index: the index for data point with hessian information.
        :param: nactive: number of active dimensions for hessians.
        """
        batch_shape = torch.broadcast_shapes(self.batch_shape, input.shape[:-2])

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
        if x.ndimension() == 1:
            x = x.unsqueeze(1)

        res = super(Mean, self).__call__(x, **kwargs)

        return res
