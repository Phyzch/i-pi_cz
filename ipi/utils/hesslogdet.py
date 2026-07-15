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
import copy 

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
        transposed_size = torch.Size([*size[:-2] ,size[-1], size[-2]])
        transposed_sparse_matrix = torch.sparse_coo_tensor(transposed_indices, values, transposed_size)
        return SparseLinearOperator(transposed_sparse_matrix)
    
    def get_sparse_matrix(self):
        return copy.deepcopy(self.sparse_matrix)

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

    mscaled_sp_term = ism_diag.matmul(rp_sparse_linear_operator).matmul(ism_diag)
    projected_sp_op = proj_linear_operator.T.matmul(mscaled_sp_term).matmul(proj_linear_operator)
    return projected_dynmat, projected_sp_op

class ControlVariateLogDetTraceEstimator():
    """
    Use control variate method to decrease the variance of log determinant.
    """
    def __init__(self, base_linear_op):
        self.base_linear_op = base_linear_op
        self.control_variate_op = None
        self.residue_op = None 
    
    def build_control_variate_decomposition(self):
        self.construct_control_variate()
        self.construct_residue_op()

    def construct_control_variate(self, control_varaite_op= None) -> LinearOperator:
        """
        create self.control_variate_op (B).
        """
        NotImplementedError(f"compute_control_variate({self.__class__.__name__}) is not implemented")
    
    def compute_control_variate_logdet(self):
        """
        compute logdet of control variate operator.
        This term needs to be computed exactly & explicitly. 
        """
        NotImplementedError(f"compute_control_variate_logdet({self.__class__.__name__}) is not implemented")

    def compute_control_variate_lcholesky(self):
        """
        compute the lower triangular cholesky of control variate matrix.
        """
        lcholesky = self.control_variate_op.cholesky(upper= False)
        return lcholesky
    
    def construct_residue_op(self):
        """
        compute B^{-1/2} H B^{-1/2} of positive definite matrix pd_hessian.
        Here B is the control variate operator.
        """
        lcholesky = self.compute_control_variate_lcholesky()
        rcholesky = lcholesky.transpose(0,1)

        r1 = lcholesky.inverse().matmul(self.base_linear_op)
        r2 = r1.matmul(rcholesky.inverse())

        self.residue_op = r2 
        return r2 
    
    def compute_logdet(self, random_vector_number, max_tridiag_iter, cg_tolerance):
        """
        compute the log determinant of base linear operator.
        """
        # logdet(B)
        control_variate_logdet = self.compute_control_variate_logdet()

        # logdet(B^{-1/2} A B^{-1/2})
        residue_logdet = compute_logdet(self._residue_op, random_vector_number, max_tridiag_iter, cg_tolerance)
        logdet = residue_logdet + control_variate_logdet
        return logdet
    
    @property
    def residue_op(self):
        return copy.deepcopy(self._residue_op) 

    @residue_op.setter
    def residue_op(self, op):
        self._residue_op = op

class BlockDiagControlVariateLogDetTraceEstimator(ControlVariateLogDetTraceEstimator):
    """
    Use block diagonal part of base_linear_op as the control variate.
    """
    def __init__(self, base_linear_op, nbeads):
        self.nbeads = nbeads
        super().__init__(base_linear_op)
        
    def construct_control_variate(self):
        nbeads = self.nbeads

        pd_matrix = self.base_linear_op.to_dense()        
        block_size = int(pd_matrix.shape[0] / nbeads)
        block_matrix = np.zeros((nbeads, block_size, block_size))
        for i in range(nbeads):
            block_indices = range(i * block_size, (i + 1) * block_size)
            block = pd_matrix[:, block_indices][block_indices, :]
            block_matrix[i] = block 
        
        # use BlockDiagLinearOperator to construct block matrix.
        block_tensor_op = linear_operator.operators.BlockDiagLinearOperator(torch.tensor(block_matrix))

        self.control_variate_op = block_tensor_op
    
    def compute_control_variate_logdet(self):
        block_tensor = self.control_variate_op.base_linear_op.to_dense()
        block_number = self.control_variate_op.num_blocks
        block_matrix = block_tensor.numpy()
        logdet = 0
        for i in range(block_number):
            block = block_matrix[i]
            eigvals = np.linalg.eigvalsh(block)
            logdet = logdet + np.sum(np.log(eigvals))

        return logdet  

class SpringTermControlVariateLogDetTraceEstimator(ControlVariateLogDetTraceEstimator):
    """
    We use the spring term as the control variate.
    """
    def __init__(self, base_linear_op, nbeads):
        self.nbeads = nbeads
        super().__init__(base_linear_op)
    def construct_control_variate(self, spring_term_op):
        """
        set the spring term operator as the control covariate operator.
        """
        self.spring_term_op = spring_term_op 
        base_linear_op = self.base_linear_op

        # zero mode of spring term (coupled harmonic oscillator)
        size = base_linear_op.size()[0]
        block_size = int(size / self.nbeads)

        # zero mode for spring term
        sp_zero_modes = np.zeros((block_size, size))
        for i in range(block_size):
            zero_mode = np.zeros(size)
            indices = range(i, size, block_size)
            zero_mode[indices] = 1
            zero_mode = zero_mode / np.linalg.norm(zero_mode)
            sp_zero_modes[i] = zero_mode
        
        sp_zero_modes_tensor = torch.tensor(sp_zero_modes)
        zero_modes_proj_op = linear_operator.operators.LowRankRootLinearOperator(sp_zero_modes_tensor.T)

        comp = zero_modes_proj_op.matmul(self.base_linear_op).matmul(zero_modes_proj_op)

        # D^T A D, here D^T is zero modes projection operator.
        self.sp_zero_modes_proj_tensor = sp_zero_modes_tensor.matmul(self.base_linear_op.matmul(sp_zero_modes_tensor.T))
        self.control_variate_op = self.spring_term_op + comp 
        

    def build_control_variate_decomposition(self, spring_term_op):
        self.construct_control_variate(spring_term_op)
        self.construct_residue_op()

    def compute_control_variate_logdet(self):
        """
        use the fact that spring terms are block diagonal in each physical dimension space.
        """
        # zero modes.
        zero_modes_eigvals = torch.linalg.eigvalsh(self.sp_zero_modes_proj_tensor)
        logdet1 = np.sum(np.log(zero_modes_eigvals.numpy()))

        # spring terms
        sp_term_tensor = self.spring_term_op.to_dense()
        size = self.base_linear_op.size()[0]
        block_size = int(size / self.nbeads)

        logdet = 0
        for i in range(block_size):
            indices = range(i, size, block_size)
            sub_tensor = sp_term_tensor[indices, :][:, indices]
            eigvals = torch.linalg.eigvalsh(sub_tensor)
            nonzero_eigvals = eigvals[1:]
            logdet = logdet + np.sum(np.log(nonzero_eigvals.numpy()))
        
        logdet = logdet + logdet1

        return logdet
    
    def compute_sp_eigenvecs(self):
        """
        get eigenvectors of spring term tensor.
        """
        # spring terms
        sp_term_tensor = self.spring_term_op.to_dense()
        size = self.base_linear_op.size()[0]
        nbeads = self.nbeads
        block_size = int(size / self.nbeads)

        sp_eigvec_lists = torch.zeros(size, size) # columns are eigenvectors.
        # sparse form.
        row_indices = []
        col_indices = []
        val_list = []
        eigval_list = []
        for i in range(block_size):
            indices = range(i, size, block_size)
            sub_tensor = sp_term_tensor[indices, :][:, indices]
            eigvals, eigvecs = torch.linalg.eigh(sub_tensor)
            # sp_eigvec_lists[:, i * self.nbeads: (i + 1) * self.nbeads][indices, :] = eigvecs 
            eigval_list.append(eigvals)
            for j in range(nbeads): # indices for eigenvec.
                for k in range(nbeads):  # indices for element of eigenvec.
                    col_indices.append(i*nbeads + j)
                    row_indices.append(indices[k])
                    val_list.append(eigvecs[k, j])

        # create sparse tensor.
        value = torch.tensor(np.array(val_list))
        row_indices = torch.tensor(np.array(row_indices))
        col_indices = torch.tensor(np.array(col_indices))
        indices = torch.stack([row_indices, col_indices], axis= 0)
        sp_eigvec_sparse_tensor = torch.sparse_coo_tensor(indices, value, size= (size, size))
        sp_eigevec_sparse_linear_operator = SparseLinearOperator(sp_eigvec_sparse_tensor)

        eigval_list = torch.tensor(np.array(eigval_list).flatten())
        # dense tensor and linear operator.
        # sp_eigvec_tensor = torch.tensor(sp_eigvec_lists)
        # sp_eigvec_op = linear_operator.to_linear_operator(sp_eigvec_tensor)

        return eigval_list, sp_eigevec_sparse_linear_operator 
    
    def inverse_sqrt_control_variate(self, eigval_list, sp_eigvec_sparse_linear_operator: SparseLinearOperator):
        """
        compute U^{T} B^{-1/2}
        :param: eigval_list: a list of eigenvalues.
        :param: sp_eigvec_sparse_linear_operator: sparse linear operator, each column is eigenvector.
        """
        nbeads = self.nbeads
        size = self.base_linear_op.size()[0] 
        block_size = int(size / nbeads) # physical dimension f.
        zero_mode_index = range(0, size, nbeads)
        zero_mode_num = block_size

        # pseudo-inverse of eigenvalue.
        inv_eigval_list = torch.zeros_like(eigval_list)
        mask = torch.ones(size, dtype= torch.bool)
        mask[zero_mode_index] = False
        inv_eigval_list[mask] = 1.0 / eigval_list[mask]
        inv_sqrt_eigval_list = torch.sqrt(inv_eigval_list)
        inv_sqrt_eigval_linear_operator = linear_operator.operators.DiagLinearOperator(inv_sqrt_eigval_list)
        
        # U^T U0, here U0 is eigenstate of zero mode.
        row_indices = torch.tensor(np.array(zero_mode_index))
        col_indices = torch.tensor(np.array(range(0, zero_mode_num)))
        val_list = torch.ones(zero_mode_num)
        indices = torch.stack([row_indices, col_indices], axis= 0)
        sparse_proj_matrix = torch.sparse_coo_tensor(indices, val_list, size= (size, zero_mode_num))
        zero_mode_proj_operator = SparseLinearOperator(sparse_proj_matrix)

        # U0: zero mode eigenvec.
        sp_eigvec_sparse = sp_eigvec_sparse_linear_operator.get_sparse_matrix()
        zero_mode_eigvec_sparse = torch.index_select(sp_eigvec_sparse, dim= 1, index= torch.tensor(zero_mode_index))
        zero_mode_eigvec_linear_operator = SparseLinearOperator(zero_mode_eigvec_sparse)

        # (U0^{T} A U0)^{-1/2}
        sp_zero_modes_proj_tensor = self.sp_zero_modes_proj_tensor
        # SVD decomposition.
        u1, s1, v1 = torch.svd(sp_zero_modes_proj_tensor)
        inv_sqrt_s1 = 1/torch.sqrt(s1)
        inv_sqrt_zero_modes_proj_tensor = torch.matmul(torch.matmul(u1, torch.diag(inv_sqrt_s1)), v1.T)
        # Cholesky decomposition.
        lcholesky = torch.cholesky(sp_zero_modes_proj_tensor)
        lcholesky_inverse = lcholesky.inverse()
        inv_sqrt_zero_modes_proj_linear_op = linear_operator.to_linear_operator(lcholesky_inverse)

        # compute U^{T} B^{-1/2}
        comp1 = inv_sqrt_eigval_linear_operator.matmul(sp_eigvec_sparse_linear_operator.T)
        comp2 = zero_mode_proj_operator.matmul(inv_sqrt_zero_modes_proj_linear_op).matmul(zero_mode_eigvec_linear_operator.T)
        inv_sqrt_control_variate = comp1 + comp2 

        return inv_sqrt_control_variate

    def construct_residue_op(self):
        sp_eigvals, sp_eigvec_op = self.compute_sp_eigenvecs()
        # U^{T} B^{-1/2}
        inv_sqrt_control_variate = self.inverse_sqrt_control_variate(sp_eigvals, sp_eigvec_op)

        r1 = inv_sqrt_control_variate.matmul(self.base_linear_op)
        r2 = r1.matmul(inv_sqrt_control_variate.T)

        self.residue_op = r2

        # # do it brute force.
        # lcholesky = self.compute_control_variate_lcholesky()
        # rcholesky = lcholesky.transpose(0,1)

        # r11 = lcholesky.inverse().matmul(self.base_linear_op)
        # r21 = r11.matmul(rcholesky.inverse())
        # transformed_residue = sp_eigvec_op.T.matmul(r21).matmul(sp_eigvec_op)

        pass 

def compute_trace_estimator_std(operator, random_vector_number, max_tridiag_iter, cg_tolerance,
                                 avg_num = 10):
    logdet_list = []
    for _ in range(avg_num):
        logdet = compute_logdet(operator, random_vector_number, max_tridiag_iter, cg_tolerance)
        logdet_list.append(logdet)
    
    std = np.std(logdet_list)
    avg = np.mean(logdet_list)
    return std, avg 

def trace_estimate_original_matrix(sparse_pd_hessian_operator,
                                   random_vector_number,
                                   max_tridiag_iter, 
                                   cg_tolerance,
                                   estimate_logdet_std= False):
    # do the trace estimator on the matrix itself.
    start_time = time.perf_counter()
    logdet = compute_logdet(sparse_pd_hessian_operator, random_vector_number, max_tridiag_iter, cg_tolerance)
    elapsed_time = (time.perf_counter() - start_time) / 60
    print(f"logdet computed directly {logdet}")
    print(f"Time to compute logdet in sparse form: {elapsed_time:.2f} minutes")

    if estimate_logdet_std:
        # compute the standard deviation of trace estimator with 10 samples. 
        avg_num = 10
        logdet_std, logdet = compute_trace_estimator_std(sparse_pd_hessian_operator, random_vector_number, max_tridiag_iter, cg_tolerance, avg_num= avg_num)
        print(f"std from {avg_num} samples for logdet: {logdet_std}")

def blockdiagonal_control_variate_trace_estimate(sparse_pd_hessian_operator,
                                                  nbeads,
                                                  random_vector_number, 
                                                  max_tridiag_iter, 
                                                  cg_tolerance,
                                                  estimate_logdet_std= False):
    """
    evaluate the performance of using block diagonal term as control variate.
    """
    block_diag_trace_estimator = BlockDiagControlVariateLogDetTraceEstimator(sparse_pd_hessian_operator,
                                                                          nbeads)
    block_diag_trace_estimator.build_control_variate_decomposition()
    blocklogdet = block_diag_trace_estimator.compute_control_variate_logdet()

     # compute residue operator of sparse_pd_hessian_operator & do the trace estimator.
    start_time = time.perf_counter()
    logdet_from_residue = block_diag_trace_estimator.compute_logdet(random_vector_number,
                                                                    max_tridiag_iter,
                                                                    cg_tolerance)
    
    elapsed_time = (time.perf_counter() - start_time) / 60
    print(f"Time to compute logdet of residue in sparse form: {elapsed_time:.2f} minutes")
    print(f"logdet of block matrix {blocklogdet}, logdet of pd matrix use control variate: {logdet_from_residue}")

    if estimate_logdet_std:
        residue_operator = block_diag_trace_estimator.residue_op 
        avg_num = 10
        residue_logdet_std, logdet_from_residue = compute_trace_estimator_std(residue_operator, random_vector_number, max_tridiag_iter, cg_tolerance, avg_num= avg_num)
        print(f"std from {avg_num} samples for residue_logdet: {residue_logdet_std}")

    return logdet_from_residue

def spring_term_control_variate_trace_estimate(sparse_pd_hessian_operator,
                                               spring_term_operator,
                                                nbeads,
                                                random_vector_number, 
                                                max_tridiag_iter, 
                                                cg_tolerance,
                                                estimate_logdet_std= False):
    """
    evaluate the performance of using spring term as control variate.
    """    
    spring_term_cv_trace_estimator = SpringTermControlVariateLogDetTraceEstimator(sparse_pd_hessian_operator,
                                                                                  nbeads)
    
    spring_term_cv_trace_estimator.build_control_variate_decomposition(spring_term_operator)

    control_variate_logdet = spring_term_cv_trace_estimator.compute_control_variate_logdet()

    # compute residue operator of sparse_pd_hessian_operator & do the trace estimator.
    start_time = time.perf_counter()
    logdet_from_residue = spring_term_cv_trace_estimator.compute_logdet(random_vector_number,
                                                                    max_tridiag_iter,
                                                                    cg_tolerance)
    
    elapsed_time = (time.perf_counter() - start_time) / 60
    print(f"Time to compute logdet of residue in sparse form: {elapsed_time:.2f} minutes")
    print(f"logdet of control variate matrix {control_variate_logdet}, logdet of pd matrix use control variate: {logdet_from_residue}")

    if estimate_logdet_std:
        residue_operator = spring_term_cv_trace_estimator.residue_op 
        avg_num = 10
        residue_logdet_std, logdet_from_residue = compute_trace_estimator_std(residue_operator, random_vector_number, max_tridiag_iter, cg_tolerance, avg_num= avg_num)
        print(f"std from {avg_num} samples for residue_logdet: {residue_logdet_std}")

    return logdet_from_residue

def compute_hessian_logdet(hessian: np.ndarray,
                           bead_hessian: np.ndarray,
                           spring_term_param: tuple,
                           proj_info: tuple,
                           random_vector_number= 1000,
                           max_tridiag_iter= 50,
                           cg_tolerance = 1e-2,
                           estimate_logdet_std= False) -> float:
    """
    Compute the log determinant of the hessian matrix.
    Remove zero eigenvalue, use the absolute value of negative eigenvalue.
    """
    print(f"random vector number for trace estimation {random_vector_number}")
    
    start_time = time.perf_counter()
    d, v, shift = solve_negative_and_zero_eigenpairs(hessian)
    elapsed_time = (time.perf_counter() - start_time) / 60
    print(f"Time to solve negative and zero eigenpairs: {elapsed_time:.2f} minutes")

    # construct the positive defintie hessian matrix.
    # bead + spring term. Then mass weighted & project out zero mode.
    # finally shift negative eigenvalues to positive. 
    bead_hessian_operator = create_block_diag_linear_operator(bead_hessian)
    rp_sparse_linear_operator = create_spring_term_linear_operator(spring_term_param)

    projected_hessian_operator, projected_sp_op = proj_hessian_operator(bead_hessian_operator,
                                              rp_sparse_linear_operator,
                                              proj_info)
    
    sparse_pd_hessian_operator = create_shifted_linear_operator(projected_hessian_operator,
                                                                v,
                                                                shift)

    # compute the logdet for block diagonalized hessian part.
    nbeads = spring_term_param[0]
    
    # # use block diagonal component as control variate matrix.
    # logdet_bd = blockdiagonal_control_variate_trace_estimate(sparse_pd_hessian_operator,
    #                                               nbeads,
    #                                               random_vector_number,
    #                                               max_tridiag_iter,
    #                                               cg_tolerance,
    #                                               estimate_logdet_std= estimate_logdet_std)

    # use spring term as control variate.
    logdet_sp = spring_term_control_variate_trace_estimate(sparse_pd_hessian_operator,
                                               projected_sp_op,
                                               nbeads,
                                               random_vector_number,
                                               max_tridiag_iter,
                                               cg_tolerance,
                                               estimate_logdet_std= estimate_logdet_std)

    # logdet_origin = trace_estimate_original_matrix(sparse_pd_hessian_operator,
    #                                                random_vector_number, 
    #                                                max_tridiag_iter, 
    #                                                cg_tolerance, 
    #                                                estimate_logdet_std= estimate_logdet_std)

    # use the logdet from spring term.
    logdet = logdet_sp

    # remove log(shifted eigval). Add log(d[0]) which is negative eigenvalue.
    shifted_positive_eigenvalues = positive_eigval  
    total_shifted_mode_number = d.shape[0] 
    shift_value = - total_shifted_mode_number * np.log(shifted_positive_eigenvalues) + np.log(np.abs(d[0]))
    hess_logdet = logdet + shift_value

    return hess_logdet