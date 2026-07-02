"""
Compute the log determinant of the hessian
using the linear_operator library. 
"""
import subprocess
import sys 
from gpytorch.settings import cg_tolerance
import numpy as np
import scipy 
import torch 
import time 

try:
    import linear_operator
    from linear_operator.operators import LowRankRootLinearOperator 
except ImportError:
    print("linear_operator is not found. Install it.")
    subprocess.check_all([
        sys.executable,
        "-m",
        "pip",
        "install",
        "linear_operator",
    ])
    import linear_operator
    from linear_operator.operators import LowRankRootLinearOperator 

positive_freq = 300 #shift negative frequency and zero frequency to this frequency value.
factor = 2 * np.pi * 3e10 * 2.4188843e-17 # convert to hartree
positive_eigval = (factor * positive_freq) ** 2 # convert to hartree

def solve_negative_and_zero_eigenpairs(hessian):
    """
    solve the negative and zero eigenpairs of ring polymer hessian matrix.
    There will be 1 negative eigenmode, 1 zero eigenmode, and 6 extra zero modes corresponding to translation and rotation.
    """
    negative_mode_number = 1
    zero_mode_number = 1 
    trans_rot_zero_mode_number = 6
    total_mode_number = negative_mode_number + zero_mode_number + trans_rot_zero_mode_number 

    #d, v = scipy.sparse.linalg.eigsh(hessian, k= total_mode_number, which='SA', return_eigenvectors=True)
    d, v = scipy.linalg.eigh(hessian, subset_by_index=[0, total_mode_number - 1])

    dd = (
        np.sign(d) * np.absolute(d) ** 0.5 / (2 * np.pi * 3e10 * 2.4188843e-17)
    ) # convert to cm^{-1}
    
    # Zeros
    cut0 = 0.01  # Note that dd[] units are cm^1
    condition = np.abs(dd) < cut0
    nzero = np.extract(condition, dd)
    print(f"Number of zero eigenvalues: {len(nzero)}")

    # shift_values for eigenvalues
    shift = positive_eigval - d

    return d, v, shift  

def create_shifted_linear_operator(hessian: np.ndarray, v, shift) -> linear_operator.LinearOperator:
    """
    Create a linear operator from the hessian matrix.
    shift hessian matrix to make it positive definite.
    :param: v: negative and zero eigenvectors.
    :param: shift: shift values for negative and zero eigenvalues.
    """
    dtype = torch.float32
    C = v @ np.sqrt(np.diag(shift)) 
    C = torch.tensor(C, dtype=dtype)
    shift_operator = LowRankRootLinearOperator(C)
    hessian_operator = linear_operator.to_linear_operator(torch.tensor(hessian, dtype=dtype))

    pd_hessian_operator = hessian_operator + shift_operator # positive definite hessian in linear operator form. s
    
    return pd_hessian_operator

def compute_logdet(pd_matrix: linear_operator.LinearOperator, 
                   random_vector_number= 10, 
                   max_tridiag_iter= 50, 
                   cg_tolerance = 1e-2) -> float:
    """
    Compute the log determinant of a positive definite matrix using linear operator.
    :param: random_vector_number: number of random vectors used to estimate the trace.
    :param: max_tridiag_iter: the maximum size of tridiagonalization matrix. 
    :param: cg_tolerance: the tolerance for the batched conjugate gradient solver, this is used to compute Lanczos tridiagonalization matrix.
    See https://arxiv.org/abs/1809.11165
    """
    with (linear_operator.settings.num_trace_samples(random_vector_number),
          linear_operator.settings.max_lanczos_quadrature_iterations(max_tridiag_iter),
          linear_operator.settings.cg_tolerance(cg_tolerance)):
        logdet = pd_matrix.logdet().item() 
    
    return logdet

def compute_hessian_logdet(hessian: np.ndarray,
                           random_vector_number= 1000,
                           max_tridiag_iter= 50,
                           cg_tolerance = 1e-2) -> float:
    """
    Compute the log determinant of the hessian matrix.
    Remove zero eigenvalue, use the absolute value of negative eigenvalue.
    """
    start_time = time.perf_counter()
    d, v, shift = solve_negative_and_zero_eigenpairs(hessian)
    elapsed_time = (time.perf_counter() - start_time) / 60
    print(f"Time to solve negative and zero eigenpairs: {elapsed_time:.2f} minutes")

    pd_hessian_operator = create_shifted_linear_operator(hessian, v, shift)

    shifted_positive_eigenvalues = positive_eigval  
    total_shifted_mode_number = d.shape[0] 
    shift_value = - total_shifted_mode_number * np.log(shifted_positive_eigenvalues) + np.log(np.abs(d[0]))

    start_time = time.perf_counter()
    logdet = compute_logdet(pd_hessian_operator, random_vector_number, max_tridiag_iter, cg_tolerance)
    elapsed_time = (time.perf_counter() - start_time) / 60
    print(f"Time to compute logdet: {elapsed_time:.2f} minutes")
 
    # remove log(shifted eigval). Add log(d[0]) which is negative eigenvalue.
    hess_logdet = logdet + shift_value

    nonzero_eigval_number = hessian.shape[0] - 1 - 6 # remove 1 zero eigenvalue and 6 translational and rotational zero eigenvalues.
    return hess_logdet