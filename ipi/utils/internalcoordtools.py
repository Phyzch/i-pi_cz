'''
packages for transformation between Cartesian coordinate (x) and internal coordinate q.
See section 2.4 of Faraday Discuss., 2018, 212, 237 and J. Chem. Phys. 105, 192–212 (1996)
'''
import numpy as np 
from numpy.linalg import norm as npnorm

class non_redundant_coordinate_transformer():
    '''
    perform transformation between Cartesian coordinate and internal coordinate
    TODO: Vectorize codes.
    '''
    def __init__(self, natom, ref_x):
        '''
        :param: natom: number of atoms for the molecule
        :param: ref_x: Cartesian coordinate of the reference point 
               the reference point is used to compute transformation matrix U: transformation between non-redundant coordinate q and redundant coordinate d.
               dtype: numpy array.  size [3 * natom] (consistent with i-pi beads.q)
        '''
        self.natom = natom
        self.ref_x = ref_x 
        
        if np.size(ref_x) != 3 * natom:
            raise("The size of reference point for initializing non redundant coordinate is not 3 * natom: size of ref_x: {} , natom: {}".format(np.size(ref_x), natom))
        
        # compute transformation matrix U between non_redundant_coordinate q and redundant_coordinate d for reference point
        # we can access U as self.ref_U 
        self.compute_transformation_matrix_U_for_ref_point()

    # for reference point, compute transformation matrix U.
    def compute_transformation_matrix_U_for_ref_point(self):
        '''
        compute transformation matrix U for transformation between nonredundant coordinate q and redundant coordinate d.
        result is computed for reference point: ref_x.
        '''
        ref_B = self._compute_redundant_gradient_matrix_B(self.ref_x)
        ref_U = self._SVD_matrix_B(ref_B)
        
        self.ref_U = ref_U 
        self.ref_UT = np.transpose(self.ref_U)
    
    # x - > B
    def _compute_redundant_gradient_matrix_B(self, x):
        '''
        compute how changes in Cartesian coordinate x will affect redundant coordinates d.
        redundant gradient matrix B == \partial d / \partial x. 
        :param: x: Cartesian coordinate.  size: [3 * natom]
        
        :return: matrix B: redundant gradient matrix. size: [natom^2, 3 * natom]
        '''
        cartesian_x = np.reshape(x, (self.natom, 3))
        B = np.zeros([np.power(self.natom, 2) , 3 * self.natom])
        
        for i in range(self.natom):
            for j in range(i):
                # if i<=j, then dij = 0
                row_index = i * self.natom + j 

                column_index_1 = i * 3   # k == i
                B[row_index, column_index_1 : column_index_1 + 3] = - (cartesian_x[i] - cartesian_x[j]) / np.power(npnorm(cartesian_x[i] - cartesian_x[j]), 3)

                column_index_2 = j * 3 # k == j
                B[row_index, column_index_2, column_index_2 + 3] = - (cartesian_x[j] - cartesian_x[i]) / np.power(npnorm(cartesian_x[i] - cartesian_x[j]), 3)

        return B 

    # SVD decomposition of B to obtain U.
    def _SVD_matrix_B(self, B):
        '''
        This code should only be called once for the reference point.
        compute the transformation matrix U: which is the eigenvector of B B^T with nonzero eigenvalues. 
        This can be computed by doing SVD decomposition of B.
        left eigenvector is the eigenvector U we are trying to find. 
        The total number of non-redundant eigenvector is 3n-6. 

        :param: B: the transformation matrix between redundant coordinate d and Cartesian coordinate x for the reference point.
        :return: U: eigenvector of B B^T.  shape: [natom^2, 3 * natom - 6]
        '''
        U, S, Vh = np.linalg.svd(B)

        assert np.size(S) >= 3 * self.natom - 6, "number of nonzero singular value of B is smaller than 3n-6."

        # sort singular value according to their absolute values. descending order
        s_index = np.argsort(- np.abs(S)) 
        nonzero_s_index = s_index[: 3 * self.natom - 6]
        nonzero_s = S[nonzero_s_index]

        # sanity check 
        zero_s_index = s_index[3 * self.natom - 6 :]
        zero_s = S[zero_s_index]
        if np.size(zero_s) != 0:
            zero_s_max = np.max(np.abs(zero_s))
            if zero_s_max > np.power(1.0, -2) * np.min(nonzero_s):
                # nonzero value is too large
                raise("zero singular value of matrix B is too large. zero_s_max: {}  min(nonzero_s): {}".format(zero_s_max, np.min(nonzero_s)))

        U = U[:, nonzero_s_index]
        Vh = Vh[nonzero_s_index, :]
        
        pass 

        return U

    
    # x -> d
    def _compute_redundant_coordinate_d(self, x):
        '''
        compute redundant coordinates d from Cartessian coordinate x.
        
        :param: x: Cartesian coordinate. shape [natom, 3]
        return: d: redundant coordinate
        '''
        cartesian_x = np.reshape(x, (self.natom, 3))
        d = np.zeros([self.natom * self.natom])
        for i in range(self.natom):
            for j in range(i):  
                # if i<=j, Dij = 0.
                # element Dij
                index = i * self.natom + j 
                Dij = 1 / npnorm(cartesian_x[i] - cartesian_x[j])
                d[index] = Dij 
        
        return d 
    
    # d -> q 
    def _transform_redundant_d_to_nonredundant_q(self, d):
        '''
        transformation from redundant coordinate d to non-redundant coordinate q using matrix U.
        q = U^T * d
        
        :param: d. redundant coordinate. shape:[natom^2]
        
        return q: non-redundant coordinate. shape:[3 * natom - 6]
        '''
        
        q = np.matmul(self.ref_UT ,d)

        return q 

    # x-> q
    def get_internal_coordinate_q(self, x):
        '''
        transform Cartesian coordinate x to internal coordinate q. 
        :param: x: Cartesian coordinate. Shape: [batch, 3 * natom]
        
        :return: q: internal coordinate. Shape: [batch, 3 * natom -6]
        '''
        # redundant internal coordinate
        d = self._compute_redundant_coordinate_d(x)

        # non-redundant internal coordinate
        q = self._transform_redundant_d_to_nonredundant_q(d)
        
        return q 
    
    def _compute_hessian_d(self, x):
        '''
        compute hessian for redundant coordinate d. 
        Result will be a rank-3 tensor of shape: [natom^2, 3 * natom, 3 * natom]
        \partial^2 Dij / \partial r_{k alpha} \partial r_{l beta} = (-1)^m *  1/|r_i - r_j|^3 * ( 3 (r_{i alpha} - r_{j alpha} * (r_{i beta} - r_{j beta}) / |r_i - r_j|^2 - delta_{alpha, beta})  ) 
        m = 0 if k = l. m = 1 if k != l.
        '''
        cartesian_x = np.reshape(x, [self.natom, 3])
        natom = self.natom 

        hessian_d = np.zeros([ np.power(natom, 2), 3 * natom, 3 * natom ])
        
        for i in range(natom):
            for j in range(i):
                tensor_index1 = i * natom + j 
                rij = npnorm(cartesian_x[i] - cartesian_x[j])

                for case_k in range(2):
                    # atom index for first derivative r_{k alpha}.
                    if case_k == 0:
                        index_k = i 
                    else:
                        index_k = j 
                    
                    for case_l in range(2):
                        # atom index for second derivative r_{l beta}
                        if case_l == 0:
                            index_l = i 
                        else:
                            index_l = j 
                        
                        if index_l == index_k:
                            m = 0
                        else:
                            m = 1 
                        
                        hessian_submatrix = np.power(-1, m) * 1 / np.power(rij,3) * ( 3 * np.outer( x[i] - x[j] , x[i] - x[j] ) / np.power(rij, 2) - np.identity(3))

                        hessian_d[tensor_index1, 3 * index_k : 3 * index_k + 3, 3 * index_l : 3 * index_l + 3] = hessian_submatrix
        
        return hessian_d 


    # transformation between gradients and hessian. g_x <-> g_q.  h_x <-> h_q
    # for prediction: g_q - > g_x, h_q -> h_x
    def transform_internal_g_h_to_cartesian_g_h(self, x, g_q , hessian_bool = False, H_q = None):
        '''
        transform from internal gradient g & hessian H to external gradient g & hessian H.
        x: Cartesian coordinate. size: [3 * natom]
        g_q: gradient in nonredundant internal coordinate.
        H_q: Hessian in nonredundant internal coordinate.
        hessian_bool: if true. transform H_q to H_x. otherwise, only transform g_q -> g_x. default: False
        '''
        B = self._compute_redundant_gradient_matrix_B(x) # \partial d / \partial x.
        
        Bq = np.matmul(self.ref_UT, B) # \partial q / \partial x 

        Bq_T = np.transpose(Bq)  # transpose of Bq

        g_x = np.matmul(Bq, g_q)  # gradient in Cartesian coordinate.

        if hessian_bool == False:
            return g_x 
        else:
            # need to compute Hessian H_x:
            if H_q == None:
                raise("To also transform internal Hessian, please provide its value. It can not be None")
        
            H_x_part1 = np.matmul(np.matmul(Bq_T, H_q),Bq)

            # H_x_part2 = g_q^T * U^T * (partial^2 d / partial x partial x'). here (partial^2 d / partial x partial x') is a tensor.
            g_q_T = np.expand_dims(g_q, axis = 0)
            # g_q^T * U^T.  shape: [1, 3n]
            prefactor = np.matmul(g_q_T, self.ref_UT) 

            # compute hessian_d: rank-3 tensor. size [natom^2, 3 * natom, 3 * natom] 
            hessian_d = self._compute_hessian_d(x)

            H_x_part2 = np.tensordot(prefactor, hessian_d, axes = (1, 0))[0]

            H_x = H_x_part1 + H_x_part2 

            return g_x, H_x 
    

    # for training: g_x -> g_q, h_x -> h_q
    def transform_cartesian_g_h_to_internal_g_h(self, x, g_x, hessian_bool = False, H_x = None):
        '''
        transform from Cartesian coordinate system's gradient g_x and hessian h_x to 
        internal coordinate gradient g_q and hessian h_q.
        if hessian_bool = True, we also transform Hessian.
        :param: x: Cartesian coordinate. size: [3 * natom]
        :param: g_x: gradient in Cartesian coordinate.
        :param: H_x: hessian in Cartesian coordinate.
        :param: hessian_bool: if true, transform H_x to H_q. otherwise, only transform gradient g_x -> g_q. default: False.
        '''
        B = self._compute_redundant_gradient_matrix_B(x) # \partial d / \partial x.
        
        Bq = np.matmul(self.ref_UT, B) # \partial q / \partial x 

        Bq_T = np.transpose(Bq)  # transpose of Bq

        # TODO: check value of recond. We can check SVD matrix of B to get an idea of value of zero-eigenvalue in B.
        inverse_Bq_T = np.linalg.pinv(Bq_T, rcond = np.power(10.0 -8) )
        
        g_q = np.matmul(inverse_Bq_T, g_x)

        if hessian_bool == False:
            return g_q 
        else:
            if H_x == None:
                raise("To also transform Cartesian hessian, please provide its value. It can not be None")
            
            inverse_Bq = np.transpose(inverse_Bq_T)

            # Below we reverse the computation in transform_internal_g_h_to_cartesian_g_h
            # compute g_q^{T} \partial B_q / \partial x.
            g_q_T = np.expand_dims(g_q, axis = 0)
            # g_q^T * U^T.  shape: [1, 3n]
            prefactor = np.matmul(g_q_T, self.ref_UT) 

            # compute hessian_d: rank-3 tensor. size [natom^2, 3 * natom, 3 * natom] 
            hessian_d = self._compute_hessian_d(x)

            H_x_part2 = np.tensordot(prefactor, hessian_d, axes = (1, 0))[0]

            H_x_part1 = np.subtract(H_x , H_x_part2)

            H_q = np.matmul(np.matmul(inverse_Bq_T, H_x_part1), inverse_Bq)

            return g_q, H_q 

