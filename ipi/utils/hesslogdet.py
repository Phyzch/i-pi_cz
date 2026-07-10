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
    from linear_operator import LinearOperator
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

class SparseLinearOperator(LinearOperator):
    """
    A linear operator that wraps a sparse matrix.
    sparse matrix need to be the torch sparse tensor, ideally created with torch.sparse_coo_tensor() method.
    """
    def __init__(self, sparse_matrix):
        self.sparse_matrix = sparse_matrix.coalesce() # Ensure the sparse matrix is in coalesced format
        super().__init__(sparse_matrix) 

    def _matmul(self, rhs):
        if rhs.ndimension() == 1:
            new_rhs = rhs.unsqueeze(1)  # Convert to a column vector
            result = torch.sparse.mm(self.sparse_matrix, new_rhs)
            return result.squeeze(1)  # Convert back to a 1D tensor
        else:  
            return torch.sparse.mm(self.sparse_matrix, rhs)
        
    def _size(self) -> torch.Size:
        return self.sparse_matrix.size()

    def _transpose_nonbatch(self):
        indices = self.sparse_matrix.indices()
        values = self.sparse_matrix.values()
        size = self.sparse_matrix.size()
        transposed_indices = torch.stack([indices[1], indices[0]], dim= 0)
        transposed_sparse_matrix = torch.sparse_coo_tensor(transposed_indices, values, size)
        return SparseLinearOperator(transposed_sparse_matrix)

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

def create_shifted_linear_operator(hessian_operator: linear_operator.LinearOperator, v, shift) -> linear_operator.LinearOperator:
    """
    Create a linear operator from the hessian matrix.
    shift hessian matrix to make it positive definite.
    :param: v: negative and zero eigenvectors.
    :param: shift: shift values for negative and zero eigenvalues.
    """
    dtype = hessian_operator.dtype
    C = v @ np.sqrt(np.diag(shift)) 
    C = torch.tensor(C, dtype=dtype)
    shift_operator = LowRankRootLinearOperator(C)

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

def create_block_diag_linear_operator(bead_hessian: np.ndarray) -> linear_operator.LinearOperator:
    """
    Create a block diagonal linear operator from the bead hessian matrix.
    :param: bead_hessian: the bead hessian matrix, which is a 3N x 3N matrix.
    :param: nbeads: number of beads in the ring polymer.
    :param: natoms: number of atoms in the system.
    """
    dtype = torch.from_numpy(np.empty(0, bead_hessian.dtype)).dtype
    bead_hessian = (bead_hessian + bead_hessian.transpose(0, 2, 1)) / 2
    bead_hessian_tensor = torch.tensor(bead_hessian, dtype=dtype)
    block_diag_operator = linear_operator.operators.BlockDiagLinearOperator(
        bead_hessian_tensor,
        block_dim = -3
    )
    
    return block_diag_operator

def create_spring_term_linear_operator(spring_term_param) -> linear_operator.LinearOperator:
    """
    Create a linear operator which represents the spring term in rp hessian. this is sparse tensor.
    """
    # m3_one_bead: mass for one bead, dimension (3 * natoms)
    # omega2 = (1/(betaN * hbar))^2.
    nbeads, natoms, omega2, m3_one_bead = spring_term_param
    
    # h_sp: spring term for one bead. size: (3 * natoms) 
    h_sp = list(np.array(m3_one_bead) * omega2)
    h_sp_diagonal = list(np.array(h_sp) * 2)
    h_sp_ndiag = list(-np.array(h_sp))

    row_indices = []
    col_indices = []
    val_list = []
    # diagonal term
    ii  = 3 * natoms 
    for i in range(0, nbeads):
        diag_indices = list(range(i * ii, (i + 1) * ii))
        row_indices = row_indices + diag_indices
        col_indices = col_indices + diag_indices 
        val_list = val_list + h_sp_diagonal 
    
    # off diagonal terms
    for i in range(0, nbeads - 1):
        row_index = list(range(i * ii, (i + 1) * ii))
        col_index = list(range((i + 1) * ii, (i + 2) * ii))
        
        row_indices = row_indices + row_index
        col_indices = col_indices + col_index 
        val_list = val_list + h_sp_ndiag

        row_index = list(range((i + 1) * ii, (i + 2) * ii))
        col_index = list(range(i * ii, (i + 1) * ii))
        
        row_indices = row_indices + row_index
        col_indices = col_indices + col_index
        val_list = val_list + h_sp_ndiag
    
    # corner off diagonal terms
    row_index = list(range(0, ii))
    col_index = list(range((nbeads - 1) * ii, nbeads * ii))

    row_indices = row_indices + row_index
    col_indices = col_indices + col_index 
    val_list = val_list + h_sp_ndiag 

    row_index = list(range((nbeads - 1) * ii, nbeads * ii))
    col_index = list(range(0, ii))

    row_indices = row_indices + row_index
    col_indices = col_indices + col_index 
    val_list = val_list + h_sp_ndiag 

    value = torch.tensor(np.array(val_list))
    row_indices = torch.tensor(np.array(row_indices))
    col_indices = torch.tensor(np.array(col_indices))
    indices = torch.stack([row_indices, col_indices], axis= 0)

    size= nbeads * 3 * natoms
    rp_sparse_tensor = torch.sparse_coo_tensor(indices, value, size= (size, size))
    rp_sparse_linear_operator = SparseLinearOperator(rp_sparse_tensor)

    return rp_sparse_linear_operator


def proj_hessian_operator(bead_hessian_operator,
                          rp_sparse_linear_operator,
                          proj_info):
    """
    transform hessian into dynmat.
    also project out translation and rotation dof.
    """
    ism, proj_vector = proj_info 
    hessian_operator = bead_hessian_operator + rp_sparse_linear_operator
    
    matrix_size = hessian_operator.size()[0]
    # mass weighted
    ism_tensor = torch.tensor(ism)
    ism_diag = linear_operator.operators.DiagLinearOperator(ism_tensor)
    dynmat = ism_diag.matmul(hessian_operator).matmul(ism_diag)

    # project out trans & rotation mode.
    proj_vector_tensor = torch.tensor(proj_vector, dtype= ism_tensor.dtype)
    proj_linear_operator = linear_operator.operators.IdentityLinearOperator(matrix_size) - LowRankRootLinearOperator(proj_vector_tensor.T) 

    projected_dynmat = proj_linear_operator.T.matmul(dynmat).matmul(proj_linear_operator)

    return projected_dynmat

def compute_block_hessian_logdet(nbeads, hessian):
    """
    The submatrix of PSD is also PSD. 
    We compute logdet for each block corresponds to hessian of each bead. (After shift & adding spring terms)
    """
    block_size = int(hessian.shape[0] / (nbeads))
    logdet_block_matrix = 0
    for i in range(nbeads):
        block_indices = range(i * block_size, (i+1) * block_size)
        block_hessian = hessian[:, block_indices][block_indices, :]
        eigvals = np.linalg.eigvalsh(block_hessian)
        logdet_block_matrix = logdet_block_matrix + np.sum(np.log(eigvals))

    return logdet_block_matrix

def compute_block_diagonal_lcholesky(nbeads, pd_hessian):
    """
    compute lower triangular cholesky linear operator for block diagonal principle matrix of pd_hessian.
    :param: pd_hessian: positive defintie hessian. numpy.array.
    :return: lcholesky: lower triangular cholesky decomposition.
    """
    block_size = int(pd_hessian.shape[0] / (nbeads))
    block_matrix = np.zeros((nbeads, block_size, block_size))
    for i in range(nbeads):
        block_indices = range(i * block_size, (i+1) * block_size)
        block = pd_hessian[:, block_indices][block_indices, :]
        block_matrix[i] = block 
    block_tensor_operator = linear_operator.operators.BlockDiagLinearOperator(torch.tensor(block_matrix))
    lcholesky = block_tensor_operator.cholesky(upper= False)
    return lcholesky

def estimate_variance_reduction(nbeads, pd_hessian):
    """
    assume hessian matrix is H.
    compute the frobenius norm of H.
    compute the frobenius norm of L^{-1} H R^{-1}.  H = LR where R = L^{T}
    here B is block diagonalized part of H. 
    """
    h_eigvals = np.linalg.eigvalsh(pd_hessian)
    frobenius_norm_log_h = np.sqrt(np.sum(np.power(np.log(h_eigvals) , 2)))
    lcholesky = compute_block_diagonal_lcholesky(nbeads, pd_hessian)
    rcholesky = lcholesky.transpose(0, 1)

    h1 = lcholesky.solve(right_tensor= torch.tensor(pd_hessian)) # L^{-1} H
    h2 = h1.matmul(rcholesky.solve(right_tensor= torch.eye(pd_hessian.shape[0]))) # yR = h1 ->  y = h1 R^{-1} I= L^{-1} H R{-1}

    h2_eigvals = np.linalg.eigvalsh(h2.to_dense().numpy())
    frobenius_norm_log_h2 = np.sqrt(np.sum(np.power(np.log(h2_eigvals) , 2)))

    print(f"frobenius norm of log(hessian) {frobenius_norm_log_h}")
    print(f"frobenius norm of log(B^(-1/2) H B^(-1/2)) {frobenius_norm_log_h2}")

def compute_residue_operator(nbeads, pd_hessian, sparse_pd_hessian_operator):
    """
    compute B^{-1/2} H B^{-1/2} of pd matrix pd_hessian.
    Here B is block diagonalized part of pd_hessian.
    """
    lcholesky = compute_block_diagonal_lcholesky(nbeads, pd_hessian)
    rcholesky = lcholesky.transpose(0, 1)
    
    r1 = lcholesky.inverse().matmul(sparse_pd_hessian_operator)
    r2 = r1.matmul(rcholesky.inverse())

    return r2 

def compute_trace_estimator_std(operator, random_vector_number, max_tridiag_iter, cg_tolerance,
                                 avg_num = 10):
    logdet_list = []
    for _ in range(avg_num):
        logdet = compute_logdet(operator, random_vector_number, max_tridiag_iter, cg_tolerance)
        logdet_list.append(logdet)
    
    std = np.std(logdet_list)
    return std 

def compute_hessian_logdet(hessian: np.ndarray,
                           bead_hessian: np.ndarray,
                           spring_term_param: tuple,
                           proj_info: tuple,
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

    bead_hessian_operator = create_block_diag_linear_operator(bead_hessian)
    rp_sparse_linear_operator = create_spring_term_linear_operator(spring_term_param)

    projected_hessian_operator = proj_hessian_operator(bead_hessian_operator,
                                              rp_sparse_linear_operator,
                                              proj_info)
    hessian_operator = linear_operator.to_linear_operator(torch.tensor(hessian))

    pd_hessian_operator = create_shifted_linear_operator(hessian_operator, v, shift)
    sparse_pd_hessian_operator = create_shifted_linear_operator (projected_hessian_operator,
                                                                       v,
                                                                       shift)

    shifted_positive_eigenvalues = positive_eigval  
    total_shifted_mode_number = d.shape[0] 
    shift_value = - total_shifted_mode_number * np.log(shifted_positive_eigenvalues) + np.log(np.abs(d[0]))

    nbeads = spring_term_param[0]
    block_logdet = compute_block_hessian_logdet(nbeads, hessian)
    print("log det for block diagonal term of shifted psd hessian.")

    # get positive definite hessian matrix.
    C = v @ np.sqrt(np.diag(shift))
    shift_matrix = C @ C.T 
    pd_hessian = hessian + shift_matrix
    
    # compute residue operator of sparse_pd_hessian_operator
    # estimate_variance_reduction(nbeads, pd_hessian)
    residue_operator = compute_residue_operator(nbeads, pd_hessian, sparse_pd_hessian_operator)

    # dense linear operator. 
    # start_time = time.perf_counter()
    # logdet = compute_logdet(pd_hessian_operator, random_vector_number, max_tridiag_iter, cg_tolerance)
    # elapsed_time = (time.perf_counter() - start_time) / 60
    # print(f"Time to compute logdet: {elapsed_time:.2f} minutes")

    start_time = time.perf_counter()
    logdet = compute_logdet(sparse_pd_hessian_operator, random_vector_number, max_tridiag_iter, cg_tolerance)
    elapsed_time = (time.perf_counter() - start_time) / 60
    print(f"Time to compute logdet in sparse form: {elapsed_time:.2f} minutes")
 
    start_time = time.perf_counter()
    residue_logdet = compute_logdet(residue_operator, random_vector_number, max_tridiag_iter, cg_tolerance)
    elapsed_time = (time.perf_counter() - start_time) / 60
    print(f"Time to compute logdet of residue in sparse form: {elapsed_time:.2f} minutes")
    logdet_from_residue = residue_logdet + block_logdet

    logdet_std = compute_trace_estimator_std(sparse_pd_hessian_operator, random_vector_number, max_tridiag_iter, cg_tolerance, avg_num= 10)
    residue_logdet_std = compute_trace_estimator_std(residue_operator, random_vector_number, max_tridiag_iter, cg_tolerance, avg_num= 10)
    print(f"std from 10 samples for logdet: {logdet_std}")
    print(f"std from 10 samples for residue_logdet: {residue_logdet_std}")

    # remove log(shifted eigval). Add log(d[0]) which is negative eigenvalue.
    hess_logdet = logdet + shift_value

    nonzero_eigval_number = hessian.shape[0] - 1 - 6 # remove 1 zero eigenvalue and 6 translational and rotational zero eigenvalues.
    return hess_logdet