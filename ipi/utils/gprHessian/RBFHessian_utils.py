"""
Utility function for Gaussian Process Regression model (with capability of predicting hessians.)
Written by Chenghao Zhang & Niri Govind, Pacific Northwest National Laboratory (chenghao.zhang@pnnl.gov), 2024.
"""

import numpy as np
import torch


def transform_1d_train_targets_into_pots_grads_hessians(
    train_targets: torch.Tensor, M: int, ndofs: int, fixdofs: np.ndarray, M_H: int
):
    """
    Transform the 1d training targets into the potential, gradient and hessian data.
    For the structure of 1d targets, see (eq.9) in J.Chem. Theory Comput. 2024, 20,3766-3778
    :param: M: total number of input data points.
    :param: ndofs: number of degrees of freedom in input data points.
    :param: fixdofs: numpy.ndarray. the dof that keep fixed in hessian calculation.
    :param: M_H: number of data points that contain hessian information.
    """
    nactive = ndofs - len(fixdofs)  # number of active dofs
    hessian_triu_size = int(
        nactive * (nactive + 1) / 2
    )  # the number of elements in upper triangle part of hessian matrix.
    targets_size = (
        M * (ndofs + 1) + hessian_triu_size * M_H
    )  # size of target data: pot (M) + gradient (M * ndofs) + hessian: (hessian_triu_size * M_H)

    batch_shape = train_targets.shape[:-2]
    assert (
        train_targets.shape[-1] == targets_size
    ), "the size of training target does not match the required data. Current shape: {}, required shape: {}".foramt(
        train_targets.shape[-1], targets_size
    )

    pot_data = train_targets[..., :M]
    gradient_data = train_targets[..., M : M * (ndofs + 1)]

    pots = pot_data
    gradients = gradient_data.reshape([*batch_shape, M, ndofs])

    if M_H > 0:
        hessian_data = train_targets[
            ..., M * (ndofs + 1) : M * (ndofs + 1) + hessian_triu_size * M_H
        ]
        hessian_triu = hessian_data.reshape([*batch_shape, M_H, hessian_triu_size])

        # generate 2d hessian matrix from upper triangle part of hessians.
        triu_indices = torch.triu_indices(nactive, nactive)
        triu_1d_indices = triu_indices[0] * nactive + triu_indices[1]

        upper_triangular_hessians = torch.zeros(
            [*batch_shape, M_H, nactive * nactive], dtype=hessian_triu.dtype
        )
        upper_triangular_hessians[..., triu_1d_indices] = hessian_triu
        upper_triangular_hessians = torch.reshape(
            upper_triangular_hessians, [*batch_shape, M_H, nactive, nactive]
        )

        hessians = upper_triangular_hessians.clone()
        for i in range(M_H):
            upper_triangular_hessian_slice = upper_triangular_hessians[i]
            hessians[i] = (
                upper_triangular_hessian_slice
                + upper_triangular_hessian_slice.t()
                - torch.diag(upper_triangular_hessian_slice.diag())
            )
    else:
        hessians = torch.Tensor([])

    return pots, gradients, hessians


def take_upper_triangular_part(tensor):
    """
    Take the upper triangular part of tensor. (here if the dimension of tensor > 2, we treat it as a stack of matrices. 
    The last two dimensions are the dimension for matrix.)
    :param: tensor: tensor to take the upper triangular part. the data type can be numpy.ndarray or torch.Tensor
    """
    if len(tensor) == 0:
        # in case the tensor is empty.
        return tensor

    batch_shape = tensor.shape[:-2]
    size1 = tensor.shape[-2]
    size2 = tensor.shape[-1]

    if size1 != size2:
        raise RuntimeError(
            "The matrix we try to take upper triangular part is not a square matrix.  size1: {},  size2: {}".format(
                size1, size2
            )
        )

    if type(tensor) == torch.Tensor:
        # torch.Tensor
        triu_indices = torch.triu_indices(size1, size2)
        triu_1d_indices = triu_indices[0] * size2 + triu_indices[1]
        # take upper triangular part of last two dimensions of Hessian.
        upper_triangle_tensor = torch.index_select(
            torch.reshape(tensor, (*batch_shape, size1 * size2)), -1, triu_1d_indices
        )
    elif type(tensor) == np.ndarray:
        # numpy.array
        triu_indices = np.triu_indices(size1)
        triu_1d_indices = triu_indices[0] * size2 + triu_indices[1]
        # take upper triangular part of last two dimensions of hessian.
        upper_triangle_tensor = np.reshape(tensor, (*batch_shape, size1 * size2))[
            ..., triu_1d_indices
        ]
    else:
        raise (
            "Do not support extracting upper triangle part of matrix whose data type is not torch.Tensor or numpy.ndarray"
        )

    return upper_triangle_tensor
