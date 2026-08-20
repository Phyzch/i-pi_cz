"""
Compute the log determinant of the hessian
using the linear_operator library. 
"""
import subprocess
import sys 
import numpy as np
import scipy 
import torch 
import time 
import copy 
import pickle 
from contextlib import contextmanager
import scipy.sparse.linalg as sl
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

positive_freq = 3000 #shift negative frequency and zero frequency to this frequency value.
factor = 2 * np.pi * 3e10 * 2.4188843e-17 # convert to hartree
positive_eigval = (factor * positive_freq) ** 2 # convert to hartree

@contextmanager
def timer(name="Code"):
    start = time.perf_counter()
    
    yield
    
    elapsed = time.perf_counter() - start
    print(f"{name} took {elapsed / 60:.4f} min ({elapsed:.2f} s)")

class BaseCoupledOscillator(LinearOperator):
    """
    A linear operator that computes a given function of the coupled harmonic oscillator matrix.
    The coupled harmonic oscillator matrix is a block diagonal matrix with each block being a circulant matrix.
    The func of the coupled harmonic oscillator matrix can be computed using the fast Fourier transform (FFT).
    Specific function used is defined in the func(). 
    """
    def __init__(self, nbeads, transposed= False, scale_factor= 1):
        self._nbeads = nbeads
        self.transposed = transposed
        self._scale_factor = scale_factor
        P = self._nbeads
        k = torch.arange(P, dtype= torch.float32)
        eigval_tensor = 4 * torch.square(torch.sin(torch.pi * k / P )).to(dtype= torch.float32)
        eigval_tensor = eigval_tensor * torch.tensor(scale_factor)
        self._eigval_tensor = eigval_tensor
        super().__init__(nbeads= nbeads, transposed= transposed, scaled_factor= scale_factor)

    def func(self, nonzero_eigvals, *args):
        NotImplementedError("Need to oveerwrite functions used in the CoupledOscillator.")

    def matrix_func_v(self, v, *args):
        """
        compute B^{-1/2} v, use the fast fourier transform.
        Here B is the hessian of coupled harmonic oscillator.

        v: shape: [nbeads, 1]
        """
        # number of beads
        P = self._nbeads
        assert P == v.shape[0], f"v should have shape ({P},), but got {v.shape}"

        # for all eigenvalues for fft.
        eigvals = self._eigval_tensor
        v = v.to(eigvals.dtype)

        # forward fourier transform. 
        v_fft = torch.fft.fft(v, dim= 0)

        func_eigvals = torch.zeros(P, dtype= v_fft.dtype)
        func_eigvals[1:] = self.func(eigvals[1:], *args) # apply func to non-zero eigenvalues.

        # compute func_eigvals * v_fft
        w_fft = func_eigvals.unsqueeze(-1) * v_fft

        # back fourier transform.
        result = torch.fft.ifft(w_fft, dim= 0).real

        result = result.to(dtype = v.dtype)
        return result 
    
    def _matmul(self, rhs, *args):
        return self.matrix_func_v(rhs, *args)

    def _size(self) -> torch.Size:
        return torch.Size([self._nbeads, self._nbeads])

    def _transpose_nonbatch(self):
        op = type(self)(self._nbeads, transposed= (not self.transposed), scale_factor= self._scale_factor)
        return op 

class SqrtInvCoupledOscillator(BaseCoupledOscillator):
    """
    A linear operator that computes the inverse square root of the coupled harmonic oscillator matrix.
    The coupled harmonic oscillator matrix is a block diagonal matrix with each block being a circulant matrix.
    The inverse square root of the coupled harmonic oscillator matrix can be computed using the fast Fourier transform (FFT).
    """
    def __init__(self, nbeads, transposed= False, scale_factor= 1):
        super().__init__(nbeads= nbeads, transposed= transposed, scale_factor= scale_factor)

    def func(self, nonzero_eigvals, *args):
        return 1.0 / torch.sqrt(nonzero_eigvals)

class InvShiftedCoupledOscillator(BaseCoupledOscillator):
    """
     A linear operator that computes the (A - theta I)^{-1} of the coupled harmonic oscillator matrix.
     scale_factor is used to scale the eigenvalue.
    """
    def __init__(self, nbeads, transposed= False, scale_factor= 1):
        super().__init__(nbeads= nbeads, transposed= transposed, scale_factor= scale_factor)

    def func(self, nonzero_eigvals, *args):
        theta = args[0]
        if type(theta) is np.ndarray:
            theta = torch.tensor(theta)
        return 1.0 / (nonzero_eigvals - theta)

class BaseCoupledOscillatorLinearOperator(LinearOperator):
    """
    A linear operator class that computes the operation of the coupled harmonic oscillator Hessian matrix.
    the specific operation depends on the func defined in BaseCoupledOscillator() class assigned to self.coupled_oscillator.
    The matrix is in block diagonalized form with size [physical_dim * nbeads], 
    where each block has shape [physical_dim * physical_dim].
    Along bead dimension, the matrix is a circulant matrix. The inverse square root of the matrix can be computed using the fast Fourier transform (FFT).
    We need to scale it * scale_factor to ensure the correct scaling of the matrix. 
    """
    def __init__(self, zero_tensor, nbeads, physical_dim, scale_factor= 1.0, transposed= False):
        self._nbeads = nbeads
        self._physical_dim = physical_dim
        self._scale_factor = scale_factor # scale for eigenvalue.
        self.transposed = transposed 
        self._zero_tensor = zero_tensor
        self._set_coupled_oscillator()
        super().__init__(zero_tensor, nbeads= nbeads, physical_dim= physical_dim, scale_factor= scale_factor, transposed= transposed)

    def _set_coupled_oscillator(self):
        self.coupled_oscillator = BaseCoupledOscillator(self._nbeads, scale_factor= self._scale_factor)

    def _size(self) -> torch.Size:
        return torch.Size([self._nbeads * self._physical_dim, self._nbeads * self._physical_dim])

    def _matmul(self, rhs, *args):
        """
        Compute the matrix-vector product of the inverse square root of the coupled harmonic oscillator Hessian matrix with a vector.
        The input vector should have shape [physical_dim * nbeads].
        The output vector will have the same shape as the input vector.
        """
        assert rhs.shape[0] == self._size()[1], f"rhs should have shape ({self._nbeads * self._physical_dim},), but got {rhs.shape}"
        if rhs.ndimension() == 1:
            rhs = rhs.unsqueeze(-1)
        # Reshape rhs to [physical_dim, nbeads]
        batch_dim = rhs.shape[1]
        rhs_reshaped = rhs.view(self._nbeads, self._physical_dim, batch_dim).transpose(0, 1).contiguous()

        # Apply B_inv_sqrt_v to each physical dimension
        result_reshaped = torch.zeros_like(rhs_reshaped)
        for i in range(self._physical_dim):
            result_reshaped[i] = self.coupled_oscillator._matmul(rhs_reshaped[i], *args)
        
        # Reshape back to [physical_dim * nbeads]
        result = result_reshaped.transpose(0, 1).contiguous().view(rhs.shape)

        return result

    def _transpose_nonbatch(self):
        op = type(self)(
                        self._zero_tensor,
                        self._nbeads,
                        self._physical_dim,
                        self._scale_factor,
                        transposed= (not self.transposed)
                        )
        
        op.coupled_oscillator = self.coupled_oscillator.T 
        return op 
    

class SqrtInvCoupledOscillatorLinearOperator(BaseCoupledOscillatorLinearOperator):
    """
    A linear operator class that computes the inverse square root of coupled harmonic oscillator Hessian matrix.
    The matrix is in block diagonalized form with size [physical_dim * nbeads], 
    where each block has shape [physical_dim * physical_dim].
    Along bead dimension, the matrix is a circulant matrix. The inverse square root of the matrix can be computed using the fast Fourier transform (FFT).
    We need to scale it with 1/sqrt(scale_factor) to ensure the correct scaling of the matrix. 
    """
    def __init__(self, zero_tensor, nbeads, physical_dim, scale_factor= 1.0, transposed= False):
        super().__init__(zero_tensor, nbeads= nbeads, physical_dim= physical_dim, scale_factor= scale_factor, transposed= transposed)

    def _set_coupled_oscillator(self):
        self.coupled_oscillator = SqrtInvCoupledOscillator(self._nbeads, scale_factor= self._scale_factor)

class InvShiftedCoupledOscillatorLinearOperator(BaseCoupledOscillatorLinearOperator):
    """
    A linear operator class that computes (A - theta I)^{-1} of coupled harmonic oscillator Hessian matrix.
    """
    def __init__(self, zero_tensor,  nbeads, physical_dim, scale_factor= 1.0, transposed= False):
        super().__init__(zero_tensor, nbeads= nbeads, physical_dim= physical_dim, scale_factor= scale_factor, transposed= transposed)

    def _set_coupled_oscillator(self):
        self.coupled_oscillator = InvShiftedCoupledOscillator(self._nbeads, scale_factor= self._scale_factor)

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

class BaseTraceEstimator():
    """
    """
    def compute_logdet_estimate(self, random_vector_number, max_tridiag_iter, cg_tolerance):
        NotImplementedError("logdet estimate not implemented")

    def compute_logdet_estimate_std(self, random_vector, max_tridiag_iter, cg_tolerance, avg_num):
        NotImplementedError("std of logdet estimate not implemented")

    def info(self):
        NotImplementedError("Info func not implemented")

class TraceEstimator(BaseTraceEstimator):
    """
    Perform the stochastic trace estimation of the operator.
    """
    def __init__(self, linear_op):
        self.linear_op = linear_op 
        # self.linear_op = self.linear_op.to(dtype= torch.float32)

    def info(self):
        """
        print information about the trace estimator we use.
        """
        print("Basic Trace Estimator on original matrix.")

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
            linear_operator.settings.max_cg_iterations(max_tridiag_iter),
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

class SubspaceProjTraceEstimator(TraceEstimator):
    """
    Perform subspace projection and then apply trace estimate on the complement subspace. 
    """
    def __init__(self, linear_op):
        super().__init__(linear_op)
    
    def info(self):
        print("Use subspace projection with trace estimate.")

    def proj_operator_logdet_calculation(self, proj_operator, max_tridiag_iter, cg_tolerance) -> float:
        """
        compute the trace estimate of projected linear operator exactly.
        """
        matrix_size = self.linear_op.size()[0]
        probe_vectors = proj_operator.to_dense().to(dtype= proj_operator.dtype)
        probe_vector_nums = probe_vectors.shape[-1]
        # to use batched cg to get the Lanczos tri-diagonalization matrix.
        with (linear_operator.settings.max_lanczos_quadrature_iterations(max_tridiag_iter),
              linear_operator.settings.max_cg_iterations(max_tridiag_iter),
              linear_operator.settings.cg_tolerance(cg_tolerance)):
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
        probe_vectors = complement_proj_linear_op.matmul(random_vectors)
        # normalize the vector.
        probe_vector_norms = torch.norm(probe_vectors, p=2, dim= -2, keepdim= True)
        probe_vectors = probe_vectors.div(probe_vector_norms).to(dtype=complement_proj_linear_op.dtype)


        # factor: in slq.to_dense(). the result * matrix.shape[-1], assuming the probe vector has norm sqrt{N} (N is matrix_size)
        # here the probe vector actually has norm sqrt(trace( I - QQ^T))  = sqrt(N - proj_op_num)
        # factor should be (N- proj_op_num) / N
        factor = (size - proj_op_num) / size

        # to use batched cg to get the Lanczos tri-diagonalization matrix. 
        with (linear_operator.settings.max_lanczos_quadrature_iterations(max_tridiag_iter),
              linear_operator.settings.max_cg_iterations(max_tridiag_iter),
            linear_operator.settings.cg_tolerance(cg_tolerance)):
            _, t_mat = self.linear_op._solve(probe_vectors, None, num_tridiag= random_vector_number)
            eigenvalues, eigenvectors = linear_operator.utils.lanczos.lanczos_tridiag_to_diag(t_mat)
            slq = linear_operator.utils.stochastic_lq.StochasticLQ()
            (logdet_term, ) = slq.to_dense(self.linear_op.size(), eigenvalues, eigenvectors, [lambda x: x.log()])
            
            # rescale it since the expected norm of vector is now sqrt(N - proj_op_num)
            logdet_term = logdet_term * factor

        logdet = logdet_term.item()
        return logdet 


    def compute_logdet_estimate(self,
                                proj_operator, 
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
                                                                        proj_op_num,
                                                                        random_vector_number,
                                                                        max_tridiag_iter,
                                                                        cg_tolerance)

        logdet = proj_op_logdet + complement_op_logdet

        probe_vec_total_num = random_vector_number + proj_op_num
        print(f"total number of probe vector used: {probe_vec_total_num}")

        return logdet
    

    def compute_logdet_estimate_std(self,
                                    proj_operator, 
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

class ControlVariateLogDetEstimator(BaseTraceEstimator):
    """
    Use control variate method to decrease the variance of log determinant.
    """
    def __init__(self, base_linear_op):
        self.base_linear_op = base_linear_op
        self.control_variate_op = None
        self.residue_op = None 
    
    def info(self):
        print("use control variate method to compute the trace estimate.")

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
    
    def compute_logdet_estimate(self, random_vector_number, max_tridiag_iter, cg_tolerance):
        """
        compute the log determinant of base linear operator.
        """
        # logdet(B)
        control_variate_logdet = self.compute_control_variate_logdet()

        # logdet(B^{-1/2} A B^{-1/2})
        trace_estimator = TraceEstimator(self.residue_op)
        residue_logdet = trace_estimator.compute_logdet_estimate(random_vector_number, max_tridiag_iter, cg_tolerance)

        logdet = residue_logdet + control_variate_logdet
        return logdet
    
    def compute_logdet_estimate_std(self, random_vector_number, max_tridiag_iter, cg_tolerance, avg_num):
        """
        estimate the std of trace estimation.
        """
        # logdet(B)
        control_variate_logdet = self.compute_control_variate_logdet()

        # logdet(B^{-1/2} A B^{-1/2})
        trace_estimator = TraceEstimator(self.residue_op)
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

class BlockDiagCVLogDetEstimator(ControlVariateLogDetEstimator):
    """
    Use block diagonal part of base_linear_op as the control variate.
    """
    def __init__(self, base_linear_op, nbeads):
        super().__init__(base_linear_op)
        self.nbeads = nbeads
        self.build_control_variate_decomposition()

    def info(self):
        print("compute log det use block diagonal matrix as control variate")

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

class SpringCVLogDetEstimator(ControlVariateLogDetEstimator):
    """
    We use the spring term as the control variate.
    """
    def __init__(self, base_linear_op, nbeads, spring_term_op, spring_term_param):
        self.spring_term_param = spring_term_param
        if not hasattr(self,"projection_index"):
            self.projection_index = 0
        super().__init__(base_linear_op)
        self.nbeads = nbeads
        self.build_control_variate_decomposition(spring_term_op)
    
    def info(self):
        print("use spring term as control variate to estimate logdet.")
        
    def build_control_variate_decomposition(self, spring_term_op):
        # construct control variate B.
        self.construct_control_variate(spring_term_op)
        # construct residue operator B^{-1/2} A B^{-1/2}
        self.construct_residue_op()

    def construct_control_variate(self, spring_term_op):
        """
        set the spring term operator as the control covariate operator.
        """
        self.spring_term_op = spring_term_op 
        base_linear_op = self.base_linear_op

        # low frequency modes of spring term (coupled harmonic oscillator)
        self.zero_mode_index = 0

        zero_mode_num = 1
        size = base_linear_op.size()[0]
        block_size = int(size / self.nbeads)

        # low frequency modes for spring terms
        # we include the projection of low frequency modes of physical hessian in the control variate matrix.
        # use ifft to get eigenvectors.
        sp_zero_modes = np.zeros((block_size * zero_mode_num, size))
        selected_eigvecs = torch.zeros([self.nbeads, zero_mode_num])
        k = torch.arange(self.nbeads, dtype= torch.float32)
        nbeads_t = torch.tensor(self.nbeads).to(torch.float32) # nbeads in tensor format.
        for eig_index in range(zero_mode_num):
            if eig_index == 0:
                index = 0
                selected_eigvecs[:, eig_index] = torch.ones([self.nbeads]) / torch.sqrt(nbeads_t)
            elif eig_index % 2 == 0:
                index = int(eig_index / 2)
                selected_eigvecs[:, eig_index] = torch.sqrt(2 / nbeads_t) * torch.cos(2 * torch.pi * index * k / nbeads_t)
            else:
                index = int((eig_index + 1) / 2)
                selected_eigvecs[:, eig_index] = torch.sqrt(2 / nbeads_t) * torch.sin(2 * torch.pi * index * k / nbeads_t)

        for i in range(block_size):  # loop through physical dimension.
            indices = range(i, size, block_size)
            zero_modes = np.zeros([zero_mode_num, size])
            zero_modes[:, indices] = (selected_eigvecs.T)[:zero_mode_num, :]
            sp_zero_modes[i * zero_mode_num: (i + 1) * zero_mode_num] = zero_modes

        sp_zero_modes_tensor = torch.tensor(sp_zero_modes)
        zero_modes_proj_op = linear_operator.operators.LowRankRootLinearOperator(sp_zero_modes_tensor.T)


        ## U^T A U, here U^T is low frequency modes projection operator.
        self.zero_modes_proj_tensor = sp_zero_modes_tensor.matmul(self.base_linear_op.matmul(sp_zero_modes_tensor.T))

        phys_hess_operator = self.base_linear_op - spring_term_op 
        comp = zero_modes_proj_op.matmul(phys_hess_operator).matmul(zero_modes_proj_op)
        self.control_variate_op = self.spring_term_op + comp 
        
    def compute_control_variate_logdet(self):
        """
        use the fact that spring terms are block diagonal in each physical dimension space.
        """
        # low frequency proj of (physical + spring term)
        zero_modes_eigvals = torch.linalg.eigvalsh(self.zero_modes_proj_tensor)
        logdet1 = np.sum(np.log(zero_modes_eigvals.numpy()))

        size = self.base_linear_op.size()[0]
        block_size = int(size / self.nbeads)

        logdet = 0

        # use analytical result for eigvals of coupled oscillators.
        omega = self.spring_term_param[2] # (1/(beta_N * hbar))^2 
        scale = omega

        P = self.nbeads 
        k = torch.arange(P, dtype= torch.float32)
        eigvals = 4 * torch.square(torch.sin(torch.pi * k / P )).to(dtype= torch.float32)
        eigvals = eigvals * scale 

        nonmixed_eigvals = eigvals[1: ] # delete zero mode.
        logdet = logdet + np.sum(np.log(nonmixed_eigvals.numpy())) * block_size
        
        logdet = logdet + logdet1

        return logdet

    def compute_low_lying_sp_eigenpairs(self):
        """
        get eigenvectors of spring term tensor.
        Only compute the low lying eigevectors that we need.
        """
        # spring terms
        size = self.base_linear_op.size()[0]
        nbeads = self.nbeads
        block_size = int(size / self.nbeads)

        sp_eigvec_lists = torch.zeros(size, size) # columns are eigenvectors.
        # sparse form.
        row_indices = []
        col_indices = []
        val_list = []
        sp_eigvals = []

        # analytical form of eigvals.
        low_lying_vec_num = self.projection_index + 1
        eigvals = torch.zeros([low_lying_vec_num], dtype= torch.float32)
        eigvecs = torch.zeros([nbeads, low_lying_vec_num], dtype= torch.float32)

        k = torch.arange(nbeads, dtype= torch.float32)
        nbeads_t = torch.tensor(nbeads).to(torch.float32) # nbeads in tensor format.
        for eig_index in range(low_lying_vec_num):
            if eig_index == 0:
                index = 0
                eigvecs[:, eig_index] = torch.ones([nbeads]) / torch.sqrt(nbeads_t)
            elif eig_index % 2 == 0:
                index = int(eig_index / 2)
                eigvecs[:, eig_index] = torch.sqrt(2 / nbeads_t) * torch.cos(2 * torch.pi * index * k / nbeads_t)
            else:
                index = int((eig_index + 1) / 2)
                eigvecs[:, eig_index] = torch.sqrt(2 / nbeads_t) * torch.sin(2 * torch.pi * index * k / nbeads_t)

            eigvals[eig_index] = 4 * torch.square(torch.sin(torch.pi * torch.tensor(index) / nbeads_t)).to(dtype= torch.float32)

        omega = self.spring_term_param[2] # omega = (1/(beta_N * hbar)^2)
        scale = omega 
        eigvals = eigvals * scale 

        for i in range(block_size):
            indices = range(i, size, block_size)
            sp_eigvals.append(eigvals[:low_lying_vec_num])
            for j in range(low_lying_vec_num): # indices for eigenvec.
                for k in range(nbeads):  # indices for element of eigenvec.
                    col_indices.append(i* low_lying_vec_num + j)
                    row_indices.append(indices[k])
                    val_list.append(eigvecs[k, j])

        # create sparse tensor.
        value = torch.tensor(np.array(val_list))
        row_indices = torch.tensor(np.array(row_indices))
        col_indices = torch.tensor(np.array(col_indices))
        indices = torch.stack([row_indices, col_indices], axis= 0)
        sp_eigvec_sparse_tensor = torch.sparse_coo_tensor(indices, value, size= (size, low_lying_vec_num * block_size), dtype= self.base_linear_op.dtype)
        sp_eigevec_sparse_linear_operator = SparseLinearOperator(sp_eigvec_sparse_tensor)

        sp_eigvals = torch.tensor(np.array(sp_eigvals).flatten()).to(dtype= self.base_linear_op.dtype)

        self.sp_eigvec_sparse_tensor = sp_eigvec_sparse_tensor
        self.sp_eigvec_linear_op = sp_eigevec_sparse_linear_operator

        return sp_eigvals, sp_eigevec_sparse_linear_operator 
    
    def inverse_sqrt_control_variate(self, 
                                     sp_eigvec_sparse_linear_operator: SparseLinearOperator):
        """
        compute  U^{T} B^{-1/2} or B^{-1/2}. here U is the eigenvector of the spring term. 
        :param: eigval_list: a list of eigenvalues.
        :param: sp_eigvec_sparse_linear_operator: sparse linear operator, each column is eigenvector.
        """
        nbeads = self.nbeads
        size = self.base_linear_op.size()[0] 
        block_size = int(size / nbeads) # physical dimension f.

        # use FFT to replace the psuedo-inverse of coupled oscillator hessian matrix. 
        omega2 = self.spring_term_param[2]
        scale_factor = omega2  # (1/ beta_P * hbar)^2. This is scaling factor for spring term with respect to the standard coupled harmonic oscillator.

        # eigenvalue tensor for fft.
        zero_tensor = torch.zeros(size=(nbeads, 1), dtype= self.base_linear_op.dtype)
        sqrt_inverse_control_variate = SqrtInvCoupledOscillatorLinearOperator(zero_tensor, nbeads, block_size, scale_factor)
    
        # U0: zero mode eigenvec.
        zero_mode_index = []
        zero_mode_num = self.zero_mode_index + 1
        sp_eigvec_num = int(sp_eigvec_sparse_linear_operator.shape[1] / block_size) # number of low lying eigevectors computed
        for i in range(block_size):
            mode_index = list(range(i * sp_eigvec_num, i * sp_eigvec_num + zero_mode_num)) # low frequency spring eigenstate for each physical dimension.
            zero_mode_index = zero_mode_index + mode_index
        zero_mode_index = np.array(zero_mode_index)

        sp_eigvec_sparse = sp_eigvec_sparse_linear_operator.get_sparse_matrix()
        zero_mode_eigvec_sparse = torch.index_select(sp_eigvec_sparse, dim= 1, index= torch.tensor(zero_mode_index))
        zero_mode_eigvec_linear_operator = SparseLinearOperator(zero_mode_eigvec_sparse)

        # (U0^{T} A U0)^{-1/2}
        zero_modes_proj_tensor = self.zero_modes_proj_tensor
        # Cholesky decomposition.
        lcholesky = torch.cholesky(zero_modes_proj_tensor)
        lcholesky_inverse = lcholesky.inverse()
        inv_sqrt_zero_modes_proj_linear_op = linear_operator.to_linear_operator(lcholesky_inverse)

        # compute B^{-1/2}
        # U S^{-1/2} U^{T}
        # comp1 = sp_eigvec_sparse_linear_operator.matmul(inv_sqrt_eigval_linear_operator).matmul(sp_eigvec_sparse_linear_operator.T)
        # use the fft form operator as B^{-1/2}
        comp1 = sqrt_inverse_control_variate

        # U0 (U0^T H U0)^{-1/2} U0^{T} 
        comp2 = zero_mode_eigvec_linear_operator.matmul(inv_sqrt_zero_modes_proj_linear_op).matmul(zero_mode_eigvec_linear_operator.T)
        inv_sqrt_control_variate = comp1 + comp2 

        return inv_sqrt_control_variate

    def construct_residue_op(self):
        sp_eigvals, sp_eigvec_op = self.compute_low_lying_sp_eigenpairs()
        # B^{-1/2}
        inv_sqrt_control_variate = self.inverse_sqrt_control_variate(sp_eigvec_op)
        self.base_linear_op = self.base_linear_op.to(dtype= inv_sqrt_control_variate.dtype)
        r1 = inv_sqrt_control_variate.matmul(self.base_linear_op)
        r2 = r1.matmul(inv_sqrt_control_variate.T)
        # B^{-1/2} A B^{-1/2}
        # used for compute trace estimation with subspace projection.
        self.residue_op = r2

class SpringCVSubspaceLogDetEstimator(SpringCVLogDetEstimator):
    def __init__(self, base_linear_op, nbeads, spring_term_op, spring_term_param, projection_index):
        self.projection_index = projection_index
        super().__init__(base_linear_op, nbeads, spring_term_op, spring_term_param)

    def info(self):
        print("use spring term as control variate to estimate logdet.")
        print(f"The projection index to project out the subspace is {self.projection_index}")


    def construct_projection_vector(self, projection_index):
        """
        construct the projection vector for the subspace method.
        :param: projection_index: We will use eigenvector (normal mode) of spring term [1, projection_index] as projection vector.
        """
        # we project out the subspace that is not included in control variate construction but still overlaps with physical modes.
        sp_eig_index_for_proj = np.array(range(self.zero_mode_index + 1, projection_index + 1))
        proj_num = projection_index + 1
        size = self.base_linear_op.size()[0]
        nbeads = self.nbeads
        block_size = int(size / self.nbeads)

        # indices in eigenvector space for spring term across all physical dimension.
        sp_eig_index_for_proj_all = []
        eigvec_num = int(self.sp_eigvec_sparse_tensor.shape[1] / block_size)
        for i in range(block_size):
            indices = (i * eigvec_num) + sp_eig_index_for_proj
            indices= indices.tolist()
            sp_eig_index_for_proj_all = sp_eig_index_for_proj_all + indices
        
        sp_eig_index_for_proj_all = np.array(sp_eig_index_for_proj_all)

        sp_eigvec_for_proj = torch.index_select(self.sp_eigvec_sparse_tensor, dim= 1, index= torch.tensor(sp_eig_index_for_proj_all))

        self.sp_eigvec_for_proj_linear_op = SparseLinearOperator(sp_eigvec_for_proj)

   # The code below delegate the trace estimation with subspace projection to the TraceEstimator class.
    def compute_logdet_estimate(self, random_vector_number, max_tridiag_iter, cg_tolerance):
        """
        Use the subspace projection method to compute the trace of large eigenvector exactly
        and compute the orthogonal compliment space use stochastic trace estimator.
        The subspace projection version of trace estimtor is delegated to the TraceEstimation class.
        """
        # logdet(B)
        control_variate_logdet = self.compute_control_variate_logdet()
        
        projection_index = self.projection_index
        # logdet(B^{-1/2} A B^{-1/2}) 
        self.construct_projection_vector(projection_index)
        trace_estimator = SubspaceProjTraceEstimator(self._residue_op)

        # TODO: Test scaling of matrix vector multiplication.
        test_matmul_scaling(self._residue_op, "residue operator")

        # here we use the subspace projection method to compute the logdet of residue operator B^{-1/2} A B^{-1/2}.
        residue_logdet = trace_estimator.compute_logdet_estimate(self.sp_eigvec_for_proj_linear_op,
                                                                            random_vector_number,
                                                                            max_tridiag_iter,
                                                                            cg_tolerance)


        logdet = residue_logdet + control_variate_logdet
        return logdet

    def compute_logdet_estimate_std(self, random_vector_number, max_tridiag_iter, cg_tolerance, avg_num):
        """
        estimate the standard deviation of the trace estimate after we do:
        (1) control variate
        (2) subspace projection on residue operator.
        """
        # logdet(B)
        control_variate_logdet = self.compute_control_variate_logdet()

        # logdet(B^{-1/2} A B^{-1/2}) 
        projection_index = self.projection_index
        self.construct_projection_vector(projection_index)
        trace_estimator = SubspaceProjTraceEstimator(self._residue_op)

        # compute std of trace estimate and the average of the trace estimate.
        residue_logdet_std, residue_logdet = trace_estimator.compute_logdet_estimate_std(
            self.sp_eigvec_for_proj_linear_op,
            random_vector_number,
            max_tridiag_iter,
            cg_tolerance,
            avg_num
        )

        logdet = residue_logdet + control_variate_logdet
        logdet_std = residue_logdet_std
        
        return logdet_std, logdet


def compute_instanton_zero_mode(ism, pos):
    """
    uses mass scaled velocity to approximate zero mode.
    omega = 1/(beta_N * hbar) = 1/(imag time between beads)
    """
    sm = 1/ism
    nbeads, phys_dim = pos.shape
    sm = sm.reshape((nbeads, phys_dim))
    mscaled_pos = pos * sm 
    mscaled_vel = np.zeros_like(mscaled_pos)
    # first order method.
    mscaled_vel[0] = (mscaled_pos[1] - mscaled_pos[-1]) / 2
    mscaled_vel[-1] = (mscaled_pos[0] - mscaled_pos[-2] )/ 2
    mscaled_vel[1:-1] = (mscaled_pos[2:] - mscaled_pos[:-2]) / 2 

    zero_mode = mscaled_vel.flatten()
    zero_mode = zero_mode / np.linalg.norm(zero_mode)

    return zero_mode

class DavidsonPreconditioner():
    def __init__(self, nbeads, physical_dim, A: LinearOperator, spring_scale_factor= 1., precond_scaling_factor = 1.0):
        """
        Preconditioner for Davidson's algorithm.
        We use (A_sp + P_0 A P_0)^{-1} as preconditioner. 
        """
        self.base_operator = A 
        self.size = A.shape[0]
        self.nbeads= nbeads
        self.physical_dim = physical_dim
        self.precond_scaling_factor = precond_scaling_factor
        self.spring_scale_factor= spring_scale_factor * self.precond_scaling_factor
        self.compute_zero_mode_proj()
        zero_tensor = torch.zeros(size=(nbeads, 1))
        self.inv_shifted_coupled_oscillator = InvShiftedCoupledOscillatorLinearOperator(
                                                                                        zero_tensor, 
                                                                                        self.nbeads, 
                                                                                        self.physical_dim, 
                                                                                        scale_factor= self.spring_scale_factor
                                                                                        )
        super().__init__()

    def compute_zero_mode_proj(self):
        """
        construct translation zero modes along each physical dimension.
        """
        trans_modes = torch.zeros([self.physical_dim, self.size])
        for i in range(self.physical_dim):
            indices = range(i, self.size, self.physical_dim)
            trans_modes[i, indices] = 1
            trans_modes[i] = trans_modes[i] / np.linalg.norm(trans_modes[i])

        self.trans_modes = trans_modes

        zero_mode_proj = torch.matmul(self.trans_modes, self.base_operator.matmul(self.trans_modes.T))
        # U0^{T} A U0
        self.zero_mode_proj = zero_mode_proj 
        # Identity matrix I
        self.zero_mode_identity = torch.eye(self.physical_dim)

    def zero_mode_inverse(self, rhs, theta):
        """
        inverse of the zero mode projection component. 
        """
        a = self.zero_mode_proj - self.zero_mode_identity * theta 
        a_inverse = torch.inverse(a) * self.precond_scaling_factor
        result = (self.trans_modes.T).matmul(
            a_inverse.matmul(
                self.trans_modes.matmul(
                    rhs
                    )
                )
        )
        return result

    def precond_inverse(self, rhs, theta):
        """
        rhs: right hand side vectors.
        theta: approximate eigenvalue solved. 
        """
        result1 = self.inv_shifted_coupled_oscillator._matmul(rhs, (theta))
        result1 = result1.squeeze(-1)
        result2 = self.zero_mode_inverse(rhs, theta) 

        result = result1 + result2 
        return result 

def davidson(A:LinearOperator, precond, rtol= 0.05, atol=1e-8):
    """
    davidson algorithm. Solve the lowest eigenvalue and eigenvector.
    Acknowledgement: Joshua Goings.
    https://joshuagoings.com/2013/08/23/davidsons-method/
    rtol: relative error of the eigenvalue.
    """
    n = A.shape[0]					# Dimension of matrix
    mmax = int(max([n//2, 1000]))				# Maximum number of iterations
    neigs = 1
    k = 8				# number of initial guess vectors 

    # t = torch.eye(n,k)			# set of k unit vectors as guess
    t = torch.normal(0, 1, size= (n, k)) # set of k unit vectors as guess
    V = torch.zeros((n,mmax + k))		# array of zeros to hold guess vec
    I = linear_operator.operators.IdentityLinearOperator(n) # identity matrix same dimen as A
    # I = torch.eye(n)			# identity matrix same dimen as A
    # Begin block Davidson routine
    lowest_eigval_list = []
    for m in range(k,mmax,k):
        if m <= k:
            for j in range(0,k):
                V[:,j] = t[:,j]/torch.linalg.norm(t[:,j])
            theta_old = -1 
        elif m > k:
            theta_old = theta[:neigs]
        V[:,:m], _ = torch.linalg.qr(V[:,:m])
        T = torch.matmul(V[:,:m].T, 
                      A.matmul(V[:, :m])
                      )
        THETA,S = torch.linalg.eigh(T)
        THETA = THETA.real 
        S = S.real 
        idx = torch.argsort(THETA)
        theta = THETA[idx]
        s = S[:,idx]
        for j in range(0,k):
            w = (A - theta[j]*I).matmul( 
                          torch.matmul(V[:,:m],s[:,j])
                          )  # w is the residue.
            # need a good conditioner 
            # q = w/(theta[j]-A_diag)
            q = precond.precond_inverse(w, theta[j])

            V[:,(m+j)] = q
        rnorm = torch.linalg.norm(theta[:neigs] - theta_old) / np.linalg.norm(theta_old)
        norm = torch.linalg.norm(theta[:neigs] - theta_old) 
        lowest_eigval_list.append(theta[0].item())
        if rnorm < rtol and norm < atol and theta[0] < 0:
            break

        if m > 800:
            pass
    print("found lowest eigenvalues in each iteration:")
    print(lowest_eigval_list)
    print(f"Davidson info: matrix dimension {n}, subspace dim that reach the convergence: {m}, relative error tolerance {rtol}, absolute error tolerance {atol}")
    # print(lowest_eigval_list)
    eigvals = theta[:neigs]
    assert eigvals[0] < 0, "the lowest eigenvalue is not negative."
    eigvecs = torch.matmul(V[:, :m], s[:,:neigs])
    eigvecs = eigvecs / torch.linalg.norm(eigvecs, axis= 0)

    return eigvals, eigvecs

def solve_negative_and_zero_eigenpairs_davidson(hessian_operator, spring_term_param, trans_rot_vec, instanton_zero_mode):
    """
    Use Davidson method to approximately solve the few lowest eigenvalue and eigenvector.
    """
    negative_mode_number = 1
    instanton_zero_mode_number = 1 
    trans_rot_zero_mode_number = 6


    # translation and rotation mode is known.  
    # # shift the zero modes.
    shifted_hessian_operator = (hessian_operator 
                                + LowRankRootLinearOperator(torch.tensor(trans_rot_vec).T) * positive_eigval
                                + LowRankRootLinearOperator(torch.tensor(instanton_zero_mode[:, np.newaxis])) * positive_eigval) 
    

    nbeads, natoms, omega2, _ = spring_term_param
    phys_dim = natoms * 3 
    spring_eigval_scale_factor = omega2 
    precond_scaling_factor = 100.0
    precond = DavidsonPreconditioner(nbeads, phys_dim, shifted_hessian_operator, 
                                     spring_scale_factor= spring_eigval_scale_factor,
                                     precond_scaling_factor= precond_scaling_factor)

    rtol = 0.05
    atol = 1e-8
    d, v = davidson(shifted_hessian_operator, precond, rtol, atol)
    d_freq = np.sign(d[0]) * np.sqrt(np.abs(d[0])) / factor
    print(f"negative eigenvalue solved: {d_freq} cm^{-1}")
    # # lobpcg method:
    # hessian = hessian_operator.to_dense()
    # d, v = torch.lobpcg(hessian,k=1, largest= False)

    # lanczos method:
    d = np.concatenate([d, np.array([0] * (trans_rot_zero_mode_number + instanton_zero_mode_number))], axis= 0)
    v = np.concatenate([v, instanton_zero_mode[:, np.newaxis], trans_rot_vec.T], axis= 1)

    # shift_values for eigenvalues
    shift = positive_eigval - d

    return d, v, shift  

def solve_negative_and_zero_eigenpairs(hessian_operator, trans_rot_vec, zero_mode):
    """
    solve the negative and zero eigenpairs of ring polymer hessian matrix.
    There will be 1 negative eigenmode, 1 zero eigenmode, and 6 extra zero modes corresponding to translation and rotation.
    proj_vec: shape: [6, ndim]
    Scaling is O(N^3) with scipy.linalg.eigh. 
    """
    negative_mode_number = 1
    instanton_zero_mode_number = 1 
    trans_rot_zero_mode_number = 6

    # translation and rotation mode is known.  
    # shift these modes beforehand. 
    hessian = hessian_operator.to_dense()

    # shift the zero modes.
    shifted_hessian = hessian + positive_eigval * trans_rot_vec.T @ trans_rot_vec 

    with timer("scipy"):
        d, v = scipy.linalg.eigh(shifted_hessian, subset_by_index=[0, 0])

    d = np.concatenate([d, np.array([0] * (trans_rot_zero_mode_number + instanton_zero_mode_number))], axis= 0)
    v = np.concatenate([v, zero_mode[:, np.newaxis], trans_rot_vec.T], axis= 1)

    # with timer("scipy"):
    #     d, v = scipy.linalg.eigh(shifted_hessian, subset_by_index=[0, 1])

    # d = np.concatenate([d, np.array([0] * trans_rot_zero_mode_number )], axis= 0)
    # v = np.concatenate([v, trans_rot_vec.T], axis= 1)

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

def construct_trace_estimator(control_variate,
                              subspace_proj,
                              sparse_pd_hessian_operator,
                              nbeads,
                              spring_term_operator,
                              spring_term_param, 
                              projection_index):
    """
    construct the trace estimator.
    """
    if not control_variate:
        # trace estiamte on the original positive definite matrix.
        trace_estimator = TraceEstimator(sparse_pd_hessian_operator)
    else:
        if not subspace_proj:
            # construct the trace estimator that use spring term as control variate.
            trace_estimator = SpringCVLogDetEstimator(sparse_pd_hessian_operator,
                                                        nbeads,
                                                        spring_term_operator,
                                                        spring_term_param)
        else:
            # construct the trace estimator that use spring term as control variate and do subspace projection.
            trace_estimator = SpringCVSubspaceLogDetEstimator(
                sparse_pd_hessian_operator,
                nbeads,
                spring_term_operator,
                spring_term_param,
                projection_index
            )
    
    return trace_estimator

def estimate_logdet(trace_estimator: BaseTraceEstimator, 
                          random_vector_number,
                          max_tridiag_iter,
                          cg_tolerance,
                          estimate_logdet_std= False):
    """
    compute trace estimate using the trace estimator. 
    Also evaluate the std of trace estimate if required
    """
    # print out the information of the trace estimator.
    trace_estimator.info()

    logdet = trace_estimator.compute_logdet_estimate(
        random_vector_number,
        max_tridiag_iter,
        cg_tolerance
    )

    if estimate_logdet_std:
        # if estimate logdet std, then we use the avg logdet to replace the result of the single run.
        avg_num = 20
        logdet_std, logdet = trace_estimator.compute_logdet_estimate_std(
            random_vector_number,
            max_tridiag_iter,
            cg_tolerance,
            avg_num
        )
        print(f"std from {avg_num} samples for logdet: {logdet_std}.")

    print(f"logdet of pd matrix: {logdet}")

    return logdet

def test_solve_scaling(operator, max_tridiag_iter, cg_tolerance, operator_name= "operator"):
    vector_num = 100
    torch.manual_seed(42)
    with timer(f"test matmul scaling with {vector_num} random vectors for {operator_name}"):
        with (linear_operator.settings.max_lanczos_quadrature_iterations(max_tridiag_iter),
               linear_operator.settings.max_cg_iterations(max_tridiag_iter),
              linear_operator.settings.cg_tolerance(cg_tolerance),
              linear_operator.settings.num_trace_samples(vector_num)):
            size = operator.size()[1]
            x = torch.rand(size, vector_num)
            _, t_mat = operator._solve(x, None, num_tridiag= vector_num)
            # logdet = operator.logdet().item()

            pass

def test_matmul_scaling(operator, operator_name= "operator"):
    vector_num = 100
    torch.manual_seed(42)
    with timer(f"test matmul scaling with {vector_num} random vectors for {operator_name}"):
            size = operator.size()[1]
            for _ in range(vector_num):
                x = torch.rand(size, dtype= operator.dtype)
                y = operator.matmul(x)

            pass

def test_two_operator_matmul(operator1, operator2):
    """
    test the matmul of two operators.
    """
    assert operator1.size() == operator2.size(), "The two operators must have the same size."

    size = operator1.size()[1]
    torch.manual_seed(42)
    x = torch.rand(size)
    y1 = operator1.matmul(x) 
    y2 = operator2.matmul(x)
    diff = y1 - y2 
    print("relative error: \n")
    print(torch.sum(torch.abs(diff)) / torch.sum(torch.abs(y1)))

def compute_hessian_logdet(bead_hessian: np.ndarray,
                           spring_term_param: tuple,
                           proj_info: tuple,
                           pos: np.ndarray,
                           random_vector_number= 1000,
                           max_tridiag_iter= 50,
                           cg_tolerance = 1e-3,
                           estimate_logdet_std_bool= False,
                           control_varaite= True,
                           subspace_proj= False,
                           proj_index= 2) -> float:
    """
    Compute the log determinant of the hessian matrix.
    Remove zero eigenvalue, use the absolute value of negative eigenvalue.
    """
    print(f"random vector number for trace estimation {random_vector_number}")

    # construct the positive defintie hessian matrix.
    # bead + spring term. Then mass weighted & project out zero mode.
    # finally shift negative eigenvalues to positive. 
    bead_hessian_operator = create_block_diag_linear_operator(bead_hessian)
    rp_sparse_linear_operator = create_spring_term_linear_operator(spring_term_param)

    projected_hessian_operator, projected_sp_op = proj_hessian_operator(
                                            bead_hessian_operator,
                                            rp_sparse_linear_operator,
                                            proj_info
                                            )

    ism , proj_vec = proj_info 

    zero_mode = compute_instanton_zero_mode(ism, pos)

    with timer("solving negative and zero eigenpairs"):
        # d, v, shift = solve_negative_and_zero_eigenpairs(projected_hessian_operator,
        #                                                  proj_vec,
        #                                                  zero_mode)

        d, v, shift = solve_negative_and_zero_eigenpairs_davidson(projected_hessian_operator,
                                                                  spring_term_param,
                                                                  proj_vec,
                                                                  zero_mode)


    sparse_pd_hessian_operator = create_shifted_linear_operator(projected_hessian_operator,
                                                                v,
                                                                shift)

    nbeads = spring_term_param[0]

    with timer("total time for trace estimate:"):
        trace_estimator = construct_trace_estimator(
            control_varaite,
            subspace_proj,
            sparse_pd_hessian_operator,
            nbeads,
            projected_sp_op,
            spring_term_param,
            proj_index
        )

        logdet = estimate_logdet(trace_estimator,
                                random_vector_number,
                                max_tridiag_iter,
                                cg_tolerance,
                                estimate_logdet_std_bool)

    # remove log(shifted eigval). Add log(d[0]) which is negative eigenvalue.
    shifted_positive_eigenvalues = positive_eigval  
    total_shifted_mode_number = d.shape[0] 
    shift_value = - total_shifted_mode_number * np.log(shifted_positive_eigenvalues) + np.log(np.abs(d[0]))
    hess_logdet = logdet + shift_value

    print(f"logdet for the hessian matrix (after reverting the shift) is {hess_logdet}")
    return hess_logdet

if __name__ == "__main__":
    with open("hess.pkl", "rb") as f:
        hess_args = pickle.load(f)

    # number of random vectors to use.
    random_vector_number= 100
    # maximum number of lanczos steps
    max_tridiag_iter= 50
    # tolerance of batched conjugate gradient method. We use result of cg for lanczos. Check BBMM paper for detail
    # BBMM: GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration
    cg_tolerance = 1e-3
    # whether to estimate the standard deviation of our estimate.
    estimate_logdet_std_bool= True
    # whether to use control variate. Currently only spirng term as control variate is used.
    control_varaite= True
    # whether to do subspace projection
    subspace_proj= False
    # projection index for low frequency modes when doing subspace projection.
    proj_index= 2
    
    hess_logdet = compute_hessian_logdet(*hess_args,
                                         random_vector_number= random_vector_number,
                                         max_tridiag_iter= max_tridiag_iter,
                                         cg_tolerance= cg_tolerance,
                                         estimate_logdet_std_bool= estimate_logdet_std_bool,
                                         control_varaite= control_varaite,
                                         subspace_proj= subspace_proj,
                                         proj_index= proj_index
                                         )