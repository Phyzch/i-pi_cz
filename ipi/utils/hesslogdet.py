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

from linear_operator import LinearOperator
from linear_operator.operators import LowRankRootLinearOperator 
from linear_operator.utils.stochastic_lq import StochasticLQ

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


class TraceEstimation():
    """
    Perform the stochastic trace estimation of the operator.
    """
    def __init__(self, linear_op):
        self.linear_op = linear_op 

    @staticmethod
    def _compute_logdet(op, random_vector_number, max_tridiag_iter, cg_tolerance):
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
            logdet = op.logdet().item() 
        
        return logdet

    def _compute_trace_estimate_std(self, operator, random_vector_number, max_tridiag_iter, cg_tolerance,
                                 avg_num = 10):
        logdet_list = []
        for _ in range(avg_num):
            logdet = self._compute_logdet(operator, random_vector_number, max_tridiag_iter, cg_tolerance)
            logdet_list.append(logdet)
        
        std = np.std(logdet_list)
        avg = np.mean(logdet_list)
        return std, avg 

    def compute_logdet_estimate(self, random_vector_number, max_tridiag_iter, cg_tolerance):
        """
        Compute the log determinant of a positive definite matrix using linear operator.
        :param: random_vector_number: number of random vectors used to estimate the trace.
        :param: max_tridiag_iter: the maximum size of tridiagonalization matrix. 
        :param: cg_tolerance: the tolerance for the batched conjugate gradient solver, this is used to compute Lanczos tridiagonalization matrix.
        See https://arxiv.org/abs/1809.11165
        """
        logdet = self._compute_logdet(self.linear_op, random_vector_number, max_tridiag_iter, cg_tolerance)
        return logdet

    def compute_logdet_estimate_std(self, random_vector, max_tridiag_iter, cg_tolerance, avg_num):
        """
        estimate the std of trace estimator.
        """
        return self._compute_trace_estimate_std(self.linear_op, random_vector, max_tridiag_iter, cg_tolerance, avg_num)

    def proj_operator_logdet_calculation(self, proj_operator, max_tridiag_iter, cg_tolerance) -> float:
        """
        compute the trace estimate of projected linear operator exactly.
        """
        matrix_size = self.linear_op.size()[0]
        probe_vectors = proj_operator.to_dense()
        probe_vector_nums = probe_vectors.shape[-1]
        # to use batched cg to get the Lanczos tri-diagonalization matrix.
        max_cg_num = 1000
        with (linear_operator.settings.max_lanczos_quadrature_iterations(max_tridiag_iter),
              linear_operator.settings.cg_tolerance(cg_tolerance),
              linear_operator.settings.max_cg_iterations(max_cg_num)):
            _, t_mat = self.linear_op._solve(probe_vectors, None, num_tridiag= probe_vector_nums)
            eigenvalues, eigenvectors = linear_operator.utils.lanczos.lanczos_tridiag_to_diag(t_mat)
            slq = linear_operator.utils.stochastic_lq.StochasticLQ()
            (logdet_term, ) = slq.to_dense(self.linear_op.size(), eigenvalues, eigenvectors, [lambda x: x.log()])

            # the result here has been multiplied matrix_size / probe_vector_nums (in slq.to_dense())  
            # to consider the fact that random vector used in trace estimate has norm sqrt(N) & average over probe_vector_nums.
            # but here probe vectors has norm 1. We need to scale back.
            logdet_term = logdet_term / matrix_size * probe_vector_nums

        logdet = logdet_term.item()

        return logdet

    def complement_space_logdet_trace_estimate(self, 
                                               complement_proj_linear_op,
                                               transformation_operator,
                                               proj_op_num,
                                               random_vector_number,
                                               max_tridiag_iter,
                                               cg_tolerance):
        """
        compute the trace estimate in the subspace. 
        adopted from the InvQuadLogdet() function in _inv_quad_logdet.py in linear_operator pacakge.
        """
        # create random vectors
        size = self.linear_op.size()[0]
        precond_lt = linear_operator.operators.IdentityLinearOperator(size)
        random_vectors = precond_lt.zero_mean_mvn_samples(random_vector_number)
        random_vectors = random_vectors.unsqueeze(-2).transpose(0, -2).squeeze(0).mT.contiguous()
        random_vectors = random_vectors.to(complement_proj_linear_op.dtype)
        # (I - QQ^{T}) U v, here U is transformation operator.
        probe_vectors = complement_proj_linear_op.matmul(transformation_operator.matmul(random_vectors))
        # normalize the vector.
        probe_vector_norms = torch.norm(probe_vectors, p=2, dim= -2, keepdim= True)
        probe_vectors = probe_vectors.div(probe_vector_norms)


        # factor: in slq.to_dense(). the result * matrix.shape[-1], assuming the probe vector has norm sqrt{N} (N is matrix_size)
        # here the probe vector actually has norm sqrt(trace( I - QQ^T))  = sqrt(N - proj_op_num)
        # factor should be (N- proj_op_num) / N
        factor = (size - proj_op_num) / size

        max_cg_num = 1000
        # to use batched cg to get the Lanczos tri-diagonalization matrix. 
        with (linear_operator.settings.max_lanczos_quadrature_iterations(max_tridiag_iter),
            linear_operator.settings.cg_tolerance(cg_tolerance),
            linear_operator.settings.max_cg_iterations(max_cg_num)):
            _, t_mat = self.linear_op._solve(probe_vectors, None, num_tridiag= random_vector_number)
            eigenvalues, eigenvectors = linear_operator.utils.lanczos.lanczos_tridiag_to_diag(t_mat)
            slq = linear_operator.utils.stochastic_lq.StochasticLQ()
            (logdet_term, ) = slq.to_dense(self.linear_op.size(), eigenvalues, eigenvectors, [lambda x: x.log()])
            
            # rescale it since the expected norm of vector is now sqrt(N - proj_op_num)
            logdet_term = logdet_term * factor

        logdet = logdet_term.item()
        return logdet 


    def subspace_projection_logdet_estimate(self,
                                            proj_operator, 
                                            transformation_operator,
                                            random_vector_number, 
                                            max_tridiag_iter, 
                                            cg_tolerance):
        """
        compute the logdet of a given subspace vectors exactly.
        compute the logdet of the operator in the complement space use trace estimate.
        :param: proj_operator: each column is an eigenvector. 
        :param: transformation_operator: do the transformation of random vectors before use it for trace estimation.
        """
        # compute the exact trace estimate in projected space.
        proj_op_logdet = self.proj_operator_logdet_calculation(proj_operator, max_tridiag_iter, cg_tolerance)

        # compute logdet in the complement space.  (I- QQ^T) A (I-QQ^T)
        matrix_size = self.linear_op.size()[0]
        complement_proj_linear_op = linear_operator.operators.IdentityLinearOperator(matrix_size) - LowRankRootLinearOperator(proj_operator)
        
        # we need separate code to implement this 
        proj_op_num = proj_operator.size()[-1]
        complement_op_logdet = self.complement_space_logdet_trace_estimate(complement_proj_linear_op,
                                                                           transformation_operator,
                                                                           proj_op_num,
                                                                           random_vector_number,
                                                                           max_tridiag_iter,
                                                                           cg_tolerance)

        logdet = proj_op_logdet + complement_op_logdet

        probe_vec_total_num = random_vector_number + proj_op_num
        print(f"total number of probe vector used: {probe_vec_total_num}")

        return logdet
    
    def subspace_projection_logdet_estimate_std(self,
                                            proj_operator, 
                                            transformation_operator,
                                            random_vector_number, 
                                            max_tridiag_iter, 
                                            cg_tolerance,
                                            avg_num= 10):
        """
        compute the logdet of a given subspace vectors exactly.
        compute the logdet of the operator in the complement space use trace estimate.
        :param: proj_operator: each column is an eigenvector. 
        :param: transformation_operator: do the transformation of random vectors before use it for trace estimation.
        Return: std of trace estimate & value of. trace estimate.
        """
        # compute the exact trace estimate in the projected space:
        proj_op_logdet = self.proj_operator_logdet_calculation(proj_operator, max_tridiag_iter, cg_tolerance)

        # compute logdet in the complement space. (I- QQ^T) A (I - QQ^T)
        matrix_size = self.linear_op.size()[0]
        complement_proj_linear_op = linear_operator.operators.IdentityLinearOperator(matrix_size) - LowRankRootLinearOperator(proj_operator)
        
        # we need separate code to implement this 
        proj_op_num = proj_operator.size()[-1]

        complement_op_logdet_list = []
        for _ in range(avg_num):
            complement_op_logdet = self.complement_space_logdet_trace_estimate(complement_proj_linear_op,
                                                                           transformation_operator,
                                                                           proj_op_num,
                                                                           random_vector_number,
                                                                           max_tridiag_iter,
                                                                           cg_tolerance)
            complement_op_logdet_list.append(complement_op_logdet)
        
        complement_op_logdet_avg = np.mean(complement_op_logdet_list)
        complement_op_logdet_std = np.std(complement_op_logdet_list)

        logdet = proj_op_logdet + complement_op_logdet_avg
        logdet_std = complement_op_logdet_std

        return logdet_std, logdet

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
        trace_estimator = TraceEstimation(self.residue_op)
        residue_logdet = trace_estimator.compute_logdet_estimate(random_vector_number, max_tridiag_iter, cg_tolerance)

        logdet = residue_logdet + control_variate_logdet
        return logdet
    
    def compute_logdet_std_estimate(self, random_vector_number, max_tridiag_iter, cg_tolerance, avg_num):
        """
        estimate the std of trace estimation.
        """
        # logdet(B)
        control_variate_logdet = self.compute_control_variate_logdet()

        # logdet(B^{-1/2} A B^{-1/2})
        trace_estimator = TraceEstimation(self.residue_op)
        residue_logdet_std, residue_logdet = trace_estimator.compute_logdet_estimate_std(random_vector_number,
                                                                                 max_tridiag_iter,
                                                                                 cg_tolerance,
                                                                                 avg_num)
        
        logdet = residue_logdet + control_variate_logdet
        # control_variate_logdet has 0 std.
        logdet_std = residue_logdet_std

        return logdet_std, logdet

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
    
    def build_control_variate_decomposition(self, spring_term_op, spring_low_freq_index):
        self.construct_control_variate(spring_term_op, spring_low_freq_index)
        self.construct_residue_op()

    def construct_control_variate(self, spring_term_op, spring_low_freq_index):
        """
        set the spring term operator as the control covariate operator.
        """
        self.spring_term_op = spring_term_op 
        base_linear_op = self.base_linear_op

        # low frequency modes of spring term (coupled harmonic oscillator)
        self.spring_low_freq_index = spring_low_freq_index

        spring_low_freq_mode_num = (self.spring_low_freq_index + 1)
        size = base_linear_op.size()[0]
        block_size = int(size / self.nbeads)

        # low frequency modes for spring terms
        # we include the projection of low frequency modes of physical hessian in the control variate matrix.
        sp_term_tensor = self.spring_term_op.to_dense()
        sp_low_freq_modes = np.zeros((block_size * spring_low_freq_mode_num, size))
        
        for i in range(block_size):  # loop through physical dimension.
            indices = range(i, size, block_size)
            spring_tensor = sp_term_tensor[indices, :][:, indices]
            eigvals, eigvecs = torch.linalg.eigh(spring_tensor)
            low_freq_modes = np.zeros([spring_low_freq_mode_num, size])
            low_freq_modes[:, indices] = (eigvecs.T)[:spring_low_freq_mode_num, :]
            sp_low_freq_modes[i * spring_low_freq_mode_num: (i + 1) * spring_low_freq_mode_num] = low_freq_modes

        sp_low_freq_modes_tensor = torch.tensor(sp_low_freq_modes)
        low_freq_modes_proj_op = linear_operator.operators.LowRankRootLinearOperator(sp_low_freq_modes_tensor.T)
        
         ## U^T A U, here U^T is low frequency modes projection operator.
        self.sp_low_freq_modes_proj_tensor = sp_low_freq_modes_tensor.matmul(self.base_linear_op.matmul(sp_low_freq_modes_tensor.T))

        phys_hess_operator = self.base_linear_op - spring_term_op 
        comp = low_freq_modes_proj_op.matmul(phys_hess_operator).matmul(low_freq_modes_proj_op)
        self.control_variate_op = self.spring_term_op + comp 
        
    def compute_control_variate_logdet(self):
        """
        use the fact that spring terms are block diagonal in each physical dimension space.
        """
        # # zero modes.
        # zero_modes_eigvals = torch.linalg.eigvalsh(self.sp_zero_modes_proj_tensor)
        # logdet1 = np.sum(np.log(zero_modes_eigvals.numpy()))

        # low frequency proj of (physical + spring term)
        low_freq_modes_eigvals = torch.linalg.eigvalsh(self.sp_low_freq_modes_proj_tensor)
        logdet1 = np.sum(np.log(low_freq_modes_eigvals.numpy()))

        # spring terms
        sp_term_tensor = self.spring_term_op.to_dense()
        size = self.base_linear_op.size()[0]
        block_size = int(size / self.nbeads)

        logdet = 0
        for i in range(block_size):
            indices = range(i, size, block_size)
            sub_tensor = sp_term_tensor[indices, :][:, indices]
            eigvals = torch.linalg.eigvalsh(sub_tensor)
            nonmixed_eigvals = eigvals[self.spring_low_freq_index + 1: ]
            logdet = logdet + np.sum(np.log(nonmixed_eigvals.numpy()))
        
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
            # TODO: ideally this should be evaluated in the closed form. 
            # The O(P^{3}) scaling of eigendecomposition, where P is bead number is undesirable 
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

        self.sp_eigvec_linear_op = sp_eigevec_sparse_linear_operator
        self.sp_eigvec_sparse_tensor = sp_eigvec_sparse_tensor

        return eigval_list, sp_eigevec_sparse_linear_operator 
    
    def inverse_sqrt_control_variate(self, 
                                     eigval_list, 
                                     sp_eigvec_sparse_linear_operator: SparseLinearOperator,
                                     sp_basis_set= False):
        """
        compute  U^{T} B^{-1/2} or B^{-1/2}. here U is the eigenvector of the spring term. 
        :param: eigval_list: a list of eigenvalues.
        :param: sp_eigvec_sparse_linear_operator: sparse linear operator, each column is eigenvector.
        """
        nbeads = self.nbeads
        size = self.base_linear_op.size()[0] 
        block_size = int(size / nbeads) # physical dimension f.
    
        # low frequency mode index
        low_freq_mode_index = []
        spring_low_freq_mode_num = self.spring_low_freq_index + 1
        for i in range(block_size):
            mode_index = list(range(i * nbeads, i * nbeads + spring_low_freq_mode_num)) # low frequency spring eigenstate for each physical dimension.
            low_freq_mode_index = low_freq_mode_index + mode_index
        low_freq_mode_index = np.array(low_freq_mode_index)
        low_freq_mode_num = len(low_freq_mode_index)

        # pseudo-inverse of eigenvalue.
        inv_eigval_list = torch.zeros_like(eigval_list)
        mask = torch.ones(size, dtype= torch.bool)
        mask[low_freq_mode_index] = False
        inv_eigval_list[mask] = 1.0 / eigval_list[mask]
        inv_sqrt_eigval_list = torch.sqrt(inv_eigval_list)
        inv_sqrt_eigval_linear_operator = linear_operator.operators.DiagLinearOperator(inv_sqrt_eigval_list)
        
        # U^T U0, here U0 is eigenstate of low frequency mode.
        row_indices = torch.tensor(np.array(low_freq_mode_index))
        col_indices = torch.tensor(np.array(range(0, low_freq_mode_num)))
        val_list = torch.ones(low_freq_mode_num)
        indices = torch.stack([row_indices, col_indices], axis= 0)
        sparse_proj_matrix = torch.sparse_coo_tensor(indices, val_list, size= (size, low_freq_mode_num))
        low_freq_mode_proj_operator = SparseLinearOperator(sparse_proj_matrix)

        # U0: low freq mode eigenvec.
        sp_eigvec_sparse = sp_eigvec_sparse_linear_operator.get_sparse_matrix()
        low_freq_mode_eigvec_sparse = torch.index_select(sp_eigvec_sparse, dim= 1, index= torch.tensor(low_freq_mode_index))
        low_freq_mode_eigvec_linear_operator = SparseLinearOperator(low_freq_mode_eigvec_sparse)

        # (U0^{T} A U0)^{-1/2}
        sp_low_freq_modes_proj_tensor = self.sp_low_freq_modes_proj_tensor
        # Cholesky decomposition.
        lcholesky = torch.cholesky(sp_low_freq_modes_proj_tensor)
        lcholesky_inverse = lcholesky.inverse()
        inv_sqrt_low_freq_modes_proj_linear_op = linear_operator.to_linear_operator(lcholesky_inverse)

        if sp_basis_set:
            # compute U^{T} B^{-1/2}
            # S^{-1/2} U^{T}
            comp1 = inv_sqrt_eigval_linear_operator.matmul(sp_eigvec_sparse_linear_operator.T)
            # (U^{T} U0)  (U0^T H U0)^{-1/2} U0^{T} 
            comp2 = low_freq_mode_proj_operator.matmul(inv_sqrt_low_freq_modes_proj_linear_op).matmul(low_freq_mode_eigvec_linear_operator.T)
            inv_sqrt_control_variate = comp1 + comp2 
        else:
            # compute B^{-1/2}
            # U S^{-1/2} U^{T}
            comp1 = sp_eigvec_sparse_linear_operator.matmul(inv_sqrt_eigval_linear_operator).matmul(sp_eigvec_sparse_linear_operator.T)
            # U0 (U0^T H U0)^{-1/2} U0^{T} 
            comp2 = low_freq_mode_eigvec_linear_operator.matmul(inv_sqrt_low_freq_modes_proj_linear_op).matmul(low_freq_mode_eigvec_linear_operator.T)
            inv_sqrt_control_variate = comp1 + comp2 

        return inv_sqrt_control_variate

    def construct_residue_op(self):
        sp_eigvals, sp_eigvec_op = self.compute_sp_eigenvecs()

        # B^{-1/2}
        inv_sqrt_control_variate = self.inverse_sqrt_control_variate(sp_eigvals, sp_eigvec_op)
        r1 = inv_sqrt_control_variate.matmul(self.base_linear_op)
        r2 = r1.matmul(inv_sqrt_control_variate.T)
        # B^{-1/2} A B^{-1/2}
        # used for compute trace estimation with subspace projection.
        self._residue_op_for_subspace_proj = r2

        # U^{T} B^{-1/2}
        sp_basis_inv_sqrt_control_variate = self.inverse_sqrt_control_variate(sp_eigvals, sp_eigvec_op, sp_basis_set= True)
        r1 = sp_basis_inv_sqrt_control_variate.matmul(self.base_linear_op)
        r2 = r1.matmul(sp_basis_inv_sqrt_control_variate.T)
        # U^{T} B^{-1/2} A B^{-1/2} U
        self.residue_op = r2 

        pass

    def construct_projection_vector(self, projection_index):
        """
        construct the projection vector for the subspace method.
        :param: projection_index: We will use eigenvector (normal mode) of spring term [1, projection_index] as projection vector.
        """
        # we project out the subspace that is not included in control variate construction but still overlaps with physical modes.
        sp_eig_index_for_proj = np.array(range(self.spring_low_freq_index + 1, projection_index + 1))
        size = self.base_linear_op.size()[0]
        nbeads = self.nbeads
        block_size = int(size / self.nbeads)

        # indices in eigenvector space for spring term across all physical dimension.
        sp_eig_index_for_proj_all = []
        for i in range(block_size):
            indices = (i * nbeads) + sp_eig_index_for_proj
            indices= indices.tolist()
            sp_eig_index_for_proj_all = sp_eig_index_for_proj_all + indices
        
        sp_eig_index_for_proj_all = np.array(sp_eig_index_for_proj_all)

        sp_eigvec_for_proj = torch.index_select(self.sp_eigvec_sparse_tensor, dim= 1, index= torch.tensor(sp_eig_index_for_proj_all))

        self.sp_eigvec_for_proj_linear_op = SparseLinearOperator(sp_eigvec_for_proj)

    # The code below delegate the trace estimation with subspace projection to the TraceEstimator class.
    def compute_logdet_subspace_projection(self, projection_index, random_vector_number, max_tridiag_iter, cg_tolerance):
        """
        Use the subspace projection method to compute the trace of large eigenvector exactly
        and compute the orthogonal compliment space use stochastic trace estimator.
        The subspace projection version of trace estimtor is delegated to the TraceEstimation class.
        """
        # logdet(B)
        control_variate_logdet = self.compute_control_variate_logdet()
        
        # logdet(B^{-1/2} A B^{-1/2}) 
        self.construct_projection_vector(projection_index)
        trace_estimator = TraceEstimation(self._residue_op_for_subspace_proj)

        # here we use the subspace projection method to compute the logdet of residue operator B^{-1/2} A B^{-1/2}.
        # We also need self.sp_eigvec_linear_op to transform the basis set into the spring vector subspace when doing trace estimation.
        residue_logdet = trace_estimator.subspace_projection_logdet_estimate(self.sp_eigvec_for_proj_linear_op,
                                                                             self.sp_eigvec_linear_op,
                                                                             random_vector_number,
                                                                             max_tridiag_iter,
                                                                             cg_tolerance)
        
        logdet = residue_logdet + control_variate_logdet
        return logdet

    def compute_logdet_subspace_projection_std_estimate(self, projection_index, random_vector_number, max_tridiag_iter, cg_tolerance, avg_num):
        """
        estimate the standard deviation of the trace estimate after we do:
        (1) control variate
        (2) subspace projection on residue operator.
        """
        # logdet(B)
        control_variate_logdet = self.compute_control_variate_logdet()

        # logdet(B^{-1/2} A B^{-1/2}) 
        self.construct_projection_vector(projection_index)
        trace_estimator = TraceEstimation(self._residue_op_for_subspace_proj)

        # compute std of trace estimate and the average of the trace estimate.
        residue_logdet_std, residue_logdet = trace_estimator.subspace_projection_logdet_estimate_std(
            self.sp_eigvec_for_proj_linear_op,
            self.sp_eigvec_linear_op,
            random_vector_number,
            max_tridiag_iter,
            cg_tolerance,
            avg_num
        )

        logdet = residue_logdet + control_variate_logdet
        logdet_std = residue_logdet_std
        
        return logdet_std, logdet


def solve_negative_and_zero_eigenpairs(hessian):
    """
    solve the negative and zero eigenpairs of ring polymer hessian matrix.
    There will be 1 negative eigenmode, 1 zero eigenmode, and 6 extra zero modes corresponding to translation and rotation.
    """
    negative_mode_number = 1
    zero_mode_number = 1 
    trans_rot_zero_mode_number = 6
    total_mode_number = negative_mode_number + zero_mode_number + trans_rot_zero_mode_number 

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
    bead_hessian_linear_op = linear_operator.to_linear_operator(bead_hessian_tensor)
    block_diag_operator = linear_operator.operators.BlockDiagLinearOperator(
        bead_hessian_linear_op,
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
    complement_proj_linear_operator = linear_operator.operators.IdentityLinearOperator(matrix_size) - LowRankRootLinearOperator(proj_vector_tensor.T) 

    projected_dynmat = complement_proj_linear_operator.T.matmul(dynmat).matmul(complement_proj_linear_operator)

    mscaled_sp_term = ism_diag.matmul(rp_sparse_linear_operator).matmul(ism_diag)
    projected_sp_op = complement_proj_linear_operator.T.matmul(mscaled_sp_term).matmul(complement_proj_linear_operator)
    return projected_dynmat, projected_sp_op



def trace_estimate_original_matrix(sparse_pd_hessian_operator,
                                   random_vector_number,
                                   max_tridiag_iter, 
                                   cg_tolerance,
                                   estimate_logdet_std= False):
    # do the trace estimator on the matrix itself.
    start_time = time.perf_counter()
    trace_estimator = TraceEstimation(sparse_pd_hessian_operator)
    logdet = trace_estimator.compute_logdet_estimate(random_vector_number, max_tridiag_iter, cg_tolerance)
    elapsed_time = (time.perf_counter() - start_time) / 60

    if estimate_logdet_std:
        # compute the standard deviation of trace estimator with 10 samples. 
        avg_num = 10
        logdet_std, logdet = trace_estimator.compute_logdet_estimate_std(random_vector_number, 
                                                                         max_tridiag_iter, 
                                                                         cg_tolerance, 
                                                                         avg_num= avg_num)
        print(f"std from {avg_num} samples for logdet: {logdet_std}")
    
    print(f"logdet computed directly {logdet}")
    print(f"Time to compute logdet in sparse form: {elapsed_time:.2f} minutes")

    return logdet


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

    if estimate_logdet_std: 
        avg_num = 10
        residue_logdet_std, logdet_from_residue = block_diag_trace_estimator.compute_logdet_std_estimate(random_vector_number, max_tridiag_iter, cg_tolerance, avg_num= avg_num)
        print(f"std from {avg_num} samples for residue_logdet: {residue_logdet_std}")

    print(f"Time to compute logdet of residue in sparse form: {elapsed_time:.2f} minutes")
    print(f"logdet of block matrix {blocklogdet}, logdet of pd matrix use control variate: {logdet_from_residue}")

    return logdet_from_residue

def spring_term_control_variate_trace_estimate(sparse_pd_hessian_operator,
                                               spring_term_operator,
                                                nbeads,
                                                random_vector_number, 
                                                max_tridiag_iter, 
                                                cg_tolerance,
                                                estimate_logdet_std= False,
                                                subspace_projection= False,
                                                projection_index= 2):
    """
    evaluate the performance of using spring term as control variate.
    """    
    spring_term_cv_trace_estimator = SpringTermControlVariateLogDetTraceEstimator(sparse_pd_hessian_operator,
                                                                                  nbeads)
    
    spring_low_freq_index= 1
    spring_term_cv_trace_estimator.build_control_variate_decomposition(spring_term_operator, spring_low_freq_index)

    control_variate_logdet = spring_term_cv_trace_estimator.compute_control_variate_logdet()

    # compute residue operator and perform subspace projection on the residue operator.
    if subspace_projection:
        start_time = time.perf_counter()
        logdet_from_residue = spring_term_cv_trace_estimator.compute_logdet_subspace_projection(
            projection_index,
            random_vector_number,
            max_tridiag_iter,
            cg_tolerance
        )
        elapsed_time = (time.perf_counter() - start_time) / 60

        if estimate_logdet_std:
            # if estimate logdet std, then we use the avg logdet to replace the result of the single run.
            avg_num = 50
            logdet_std, logdet_from_residue = spring_term_cv_trace_estimator.compute_logdet_subspace_projection_std_estimate(
                projection_index,
                random_vector_number,
                max_tridiag_iter,
                cg_tolerance,
                avg_num
            )
            print(f"std from {avg_num} samples for residue logdet use subspace projection: {logdet_std}. projection until eigenmode index {projection_index}")

        print(f"Time to compute logdet of residue in sparse form: {elapsed_time:.2f} minutes")
        print(f"logdet of control variate matrix {control_variate_logdet}, logdet of pd matrix use control variate: {logdet_from_residue}")
    else:
        # compute residue operator of sparse_pd_hessian_operator & do the trace estimator.
        start_time = time.perf_counter()
        logdet_from_residue = spring_term_cv_trace_estimator.compute_logdet(random_vector_number,
                                                                        max_tridiag_iter,
                                                                        cg_tolerance)
        
        elapsed_time = (time.perf_counter() - start_time) / 60

        if estimate_logdet_std:
            # if estiamte logdet std, then we use avg logdet to replace the result of the single run.
            avg_num = 10
            logdet_std, logdet_from_residue =  spring_term_cv_trace_estimator.compute_logdet_std_estimate(random_vector_number, max_tridiag_iter, cg_tolerance, avg_num= avg_num)
            print(f"std from {avg_num} samples for residue_logdet: {logdet_std}")

        print(f"Time to compute logdet of residue in sparse form: {elapsed_time:.2f} minutes")
        print(f"logdet of control variate matrix {control_variate_logdet}, logdet of pd matrix use control variate: {logdet_from_residue}")

    return logdet_from_residue

def compute_hessian_logdet(hessian: np.ndarray,
                           bead_hessian: np.ndarray,
                           spring_term_param: tuple,
                           proj_info: tuple,
                           random_vector_number= 1000,
                           max_tridiag_iter= 50,
                           cg_tolerance = 1e-3,
                           estimate_logdet_std= False,
                           control_varaite= True,
                           subspace_proj= False,
                           proj_index= 2) -> float:
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
    if control_varaite:
        logdet_sp = spring_term_control_variate_trace_estimate(sparse_pd_hessian_operator,
                                                   projected_sp_op,
                                                   nbeads,
                                                   random_vector_number,
                                                   max_tridiag_iter,
                                                   cg_tolerance,
                                                   estimate_logdet_std= estimate_logdet_std,
                                                   subspace_projection= subspace_proj,
                                                   projection_index= proj_index)
        logdet = logdet_sp
    else:
        logdet_origin = trace_estimate_original_matrix(sparse_pd_hessian_operator,
                                                    random_vector_number, 
                                                    max_tridiag_iter, 
                                                    cg_tolerance, 
                                                    estimate_logdet_std= estimate_logdet_std)

        logdet = logdet_origin

    # remove log(shifted eigval). Add log(d[0]) which is negative eigenvalue.
    shifted_positive_eigenvalues = positive_eigval  
    total_shifted_mode_number = d.shape[0] 
    shift_value = - total_shifted_mode_number * np.log(shifted_positive_eigenvalues) + np.log(np.abs(d[0]))
    hess_logdet = logdet + shift_value

    return hess_logdet