"""
packages for transformation between Cartesian coordinate (x) and internal coordinate q.
See section 2.4 of Faraday Discuss., 2018, 212, 237 and J. Chem. Phys. 105, 192–212 (1996)
Code written by Chenghao Zhang & Niri Govind, Pacific Northwest National Laboratory, 2024.
"""

import numpy as np
from numpy.linalg import norm as npnorm


class non_redundant_coordinate_transformer:
    """
    perform transformation between Cartesian coordinate and internal coordinate
    """

    def __init__(self, natom, 
                 ref_x, 
                 singular_value_cutoff = np.power(10.0, -2)):
        """
        :param: natom: number of atoms for the molecule
        :param: ref_x: Cartesian coordinate of the reference point
               the reference point is used to compute transformation matrix U: transformation between non-redundant coordinate q and redundant coordinate d.
               dtype: numpy array.  size [3 * natom] (consistent with i-pi beads.q)
        """
        self.natom = natom
        self.ref_x = ref_x

        if np.size(ref_x) != 3 * natom:
            raise (
                "The size of reference point for initializing non redundant coordinate is not 3 * natom: size of ref_x: {} , natom: {}".format(
                    np.size(ref_x), natom
                )
            )

        # compute transformation matrix U between non_redundant_coordinate q and redundant_coordinate d for reference point
        # we can access U as self.ref_U
        self._compute_transformation_matrix_U_for_ref_point(singular_value_cutoff)

    # for reference point, compute transformation matrix U.
    def _compute_transformation_matrix_U_for_ref_point(self,
                                                      singular_value_cutoff):
        """
        compute transformation matrix U for transformation between nonredundant coordinate q and redundant coordinate d.
        result is computed for reference point: ref_x.
        """
        ref_x = np.expand_dims(self.ref_x, 0)  # batch size 1.  size: [1, 3 * natom]
        ref_B = self._compute_redundant_gradient_matrix_B(
            ref_x
        )  # this function takes in an array of coordinate x and return \partial d / \partial x: The B matrix.
        ref_B = ref_B[0]  # B: redundant Wilson's B matrix.  shape: [natom^2, 3 * natom]
        ref_U, ref_S, ref_Vh = self._SVD_matrix_B(
            ref_B,
            singular_value_cutoff
        )  # ref_U: shape [natom^2, internal_dof].   ref_S: eigenvalue matrix S, diagonal part is eigenvalue s_i.

        self.ref_U = ref_U  # left singular vector matrix. each column is one left singular vector.
        self.ref_UT = self.ref_U.T
        self.ref_S = ref_S
        self.ref_Vh = ref_Vh 
        self.nonzero_S_index_len = len(ref_S)
        print(f"Number of nonzero dofs: {self.nonzero_S_index_len}") 

    # transformation x - > B_d  (Wilson's B matrix for redundant coordinate d)
    def _compute_redundant_gradient_matrix_B(self, x):
        """
        compute how changes in Cartesian coordinate x will affect redundant coordinates d.
        redundant gradient matrix B == \\ partial d / \\ partial x.
        :param: x: Cartesian coordinate.  size: [nbatch, 3 * natom]

        :return: matrix B: redundant gradient matrix. size: [nbatch, natom^2, 3 * natom]
        """
        x_shape = np.shape(x)
        nbatch = x_shape[0]
        cartesian_x = np.reshape(x, (nbatch, self.natom, 3))
        B = np.zeros([nbatch, np.power(self.natom, 2), 3 * self.natom])

        for i in range(self.natom):
            for j in range(i):
                # if i<=j, then dij = 0
                row_index = i * self.natom + j

                column_index_1 = i * 3  # k == i
                B[:, row_index, column_index_1 : column_index_1 + 3] = -(
                    cartesian_x[:, i, :] - cartesian_x[:, j, :]
                ) / np.expand_dims(
                    np.power(
                        npnorm(cartesian_x[:, i, :] - cartesian_x[:, j, :], axis=1), 3
                    ),
                    axis=1,
                )

                column_index_2 = j * 3  # k == j
                B[:, row_index, column_index_2 : column_index_2 + 3] = -(
                    cartesian_x[:, j, :] - cartesian_x[:, i, :]
                ) / np.expand_dims(
                    np.power(
                        npnorm(cartesian_x[:, i, :] - cartesian_x[:, j, :], axis=1), 3
                    ),
                    axis=1,
                )

        return B

    def compute_delocalized_wilson_matrix_Bq(self, x):
        """
        compute how changes in Cartesian coordinate x will affect delocalized non-redundant coordinate q.
        Bq == \\ partial q / \\partial x.
        :param: x: Cartesian coordinate. size: [nbatch, 3 * natom]

        :return: matrix Bq: delocalized non-redundant Wilson B matrix.  size: [nbatch, 3 * natom - 6, 3 * natom]
        """
        B = self._compute_redundant_gradient_matrix_B(
            x
        )  # \partial d / \partial x. shape: [nbatch, n^2, 3n]

        # compute transformation matrix U for each point
        Bq = np.matmul(
            self.ref_UT, B
        )  # \partial q / \partial x. shape:[nbatch, 3n -6, 3n]
        
        return Bq 

    # SVD decomposition of B to obtain left singular vector matrix U.
    def _SVD_matrix_B(self, B, singular_value_cutoff):
        """
        This code should only be called once for the reference point.
        compute the transformation matrix U: which is the eigenvector of B B^T with nonzero eigenvalues.
        This can be computed by doing SVD decomposition of B.
        left eigenvector is the eigenvector U we are trying to find.
        The total number of non-redundant eigenvector is 3n-6.

        :param: B: the transformation matrix between redundant coordinate d and Cartesian coordinate x for the reference point.
        :return: U: eigenvector of B B^T.  shape: [natom^2, 3 * natom - 6]
        """
        U, S, Vh = np.linalg.svd(B)

        assert (
            np.size(S) >= 3 * self.natom - 6
        ), "number of nonzero singular value of B is smaller than 3n-6. Wrong"

        # sort singular value according to their absolute values. descending order
        s_index = np.array(range(len(S)))
        nonzero_S_index = s_index[: 3 * self.natom - 6]
        nonzero_S = S[nonzero_S_index]

        # sanity check in case we have zero sinuglar value number larger than 3n-6.
        zero_S_index = s_index[3 * self.natom - 6 :]
        zero_S = S[zero_S_index]
        if np.size(zero_S) != 0:
            zero_s_max = np.max(np.abs(zero_S))
            if zero_s_max > np.power(10.0, -2) * np.min(np.abs(nonzero_S)):
                # nonzero value is too large
                raise (
                    "zero singular value of matrix B is too large. zero_s_max: {}  min(nonzero_s): {}".format(
                        zero_s_max, np.min(np.abs(nonzero_S))
                    )
                )
            
        S_nonredundant = S[:-6]
        print(f"All non-redundant singular values: {S_nonredundant}")

        # check the case that non-zero singular value becomes 0 (could because of the extra symmetry).
        # In this case, we will have internal coordinate number < 3n - 6.
        singular_value_cutoff = np.max(nonzero_S) * singular_value_cutoff
        S_clip = np.array([s for s in S if s > singular_value_cutoff])
        nonzero_S_index_len = len(S_clip)

        U = U[:, :nonzero_S_index_len]
        Vh = Vh[:nonzero_S_index_len, :]
        S = S_clip

        print(f"The non-redundant singular values we include {S}")

        return U, S, Vh 

    # x -> d. here d is redundant coordinate.
    def _compute_redundant_coordinate_d(self, x):
        """
        compute redundant coordinates d from Cartessian coordinate x.

        :param: x: Cartesian coordinate. shape [nbatch, 3 * natom]
        return: d: redundant coordinate. shape: [nbatch, natom^2]
        """
        x_shape = np.shape(x)
        cartesian_x = np.reshape(x, (x_shape[0], self.natom, 3))

        d = np.zeros([x_shape[0], self.natom * self.natom])
        for i in range(self.natom):
            for j in range(i):
                # if i<=j, Dij = 0.
                # compute element Dij = d[i * natom + j]
                index = i * self.natom + j
                Dij = 1 / npnorm(cartesian_x[:, i, :] - cartesian_x[:, j, :], axis=1)
                d[:, index] = Dij

        return d

    # d -> q. transform the redundant coordinate d to non-redundant coordinate q.
    def _transform_redundant_d_to_nonredundant_q(self, d, x):
        """
        transformation from redundant coordinate d to non-redundant coordinate q using matrix U.
        q = U^T * d

        :param: d. redundant coordinate. shape:[nbatch, natom^2]

        return q: non-redundant coordinate. shape:[nbatch, internal_dof_#] (if with symmetry, internal_dof_# could be smaller than 3 * natom -6)
        """
        d_stack = np.expand_dims(d, axis=2)

        q = np.matmul(self.ref_UT, d_stack)

        q = np.squeeze(q, axis=2)

        return q

    # x-> q. combine x->d & d->q, transform Cartesian coordinate into non-redundant internal coordinate q.
    def get_internal_coordinate_q(self, x):
        """
        transform Cartesian coordinate x to internal coordinate q.
        :param: x: Cartesian coordinate. Shape: [nbatch, 3 * natom]

        :return: q: internal coordinate. Shape: [nbatch, internal_dof_#]
        """
        # redundant internal coordinate
        d = self._compute_redundant_coordinate_d(x)

        # non-redundant internal coordinate
        q = self._transform_redundant_d_to_nonredundant_q(d, x)

        return q

    def _compute_hessian_d(self, x):
        """
        compute hessian for redundant coordinate d.
        Result will be a rank-4 tensor of shape: [nbatch, natom^2, 3 * natom, 3 * natom]
        \\ partial^2 Dij / \\ partial r_{k alpha} \\ partial r_{l beta} = (-1)^m *  1/|r_i - r_j|^3 * ( 3 (r_{i alpha} - r_{j alpha}) * (r_{i beta} - r_{j beta}) / |r_i - r_j|^2 - delta_{alpha, beta})  )
        m = 0 if k = l. m = 1 if k != l.
        here the hessian is only nonzero when k=i or j, l = i or j.

        :param: x: [nbatch, 3 * natom]
        :return: hessian_d: [nbatch, natom^2, 3 * natom, 3 * natom]
        """
        x_shape = np.shape(x)
        nbatch = x_shape[0]
        cartesian_x = np.reshape(x, [nbatch, self.natom, 3])
        natom = self.natom

        hessian_d = np.zeros([nbatch, np.power(natom, 2), 3 * natom, 3 * natom])

        for i in range(natom):
            for j in range(i):
                tensor_index1 = i * natom + j
                rij = npnorm(
                    cartesian_x[:, i, :] - cartesian_x[:, j, :], axis=1
                )  # shape: [nbatch]. |r_i - r_j|
                rij_matrix = rij[:, np.newaxis, np.newaxis]  # shape:[nbatch , 1, 1]
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

                        xij_vector = cartesian_x[:, i, :] - cartesian_x[:, j, :]
                        xij_outer_product = (
                            xij_vector[:, :, np.newaxis] * xij_vector[:, np.newaxis, :]
                        )  # shape: [nbatch, 3, 3]
                        identity_matrix = np.tile(
                            np.expand_dims(np.identity(3), axis=0), (nbatch, 1, 1)
                        )  # shape: [nbatch, 3, 3]
                        hessian_submatrix = (
                            np.power(-1, m)
                            * 1
                            / np.power(rij_matrix, 3)
                            * (
                                3 * xij_outer_product / np.power(rij_matrix, 2)
                                - identity_matrix
                            )
                        )

                        hessian_d[
                            :,
                            tensor_index1,
                            3 * index_k : 3 * index_k + 3,
                            3 * index_l : 3 * index_l + 3,
                        ] = hessian_submatrix

        return hessian_d

    # transformation between gradients and hessian. g_x <-> g_q.  h_x <-> h_q
    # for prediction: g_q - > g_x, transform gradient.
    def transform_internal_gradient_to_cartesian_gradient(self, x, g_q):
        """
        transform from internal coordinate's gradient g to cartesian coordinate gradient g.
        :param: x: Cartesian coordinate. size: [nbatch, 3 * natom]
        :param: g_q: gradient in nonredundant internal coordinate. shape: [nbatch, internal_dof_#]

        :return: g_x: gradient in cartesian coordinate. shape: [nbatch, 3 * natom]
        """
        Bq = self.compute_delocalized_wilson_matrix_Bq(
            x
        ) # \partial q / \partial x. shape:[nbatch, 3n -6, 3n]

        Bq_T = np.transpose(
            Bq, axes=(0, 2, 1)
        )  # transpose of Bq. shape: [nbatch, 3n, 3n-6]

        # g_x = Bq_T * g_q
        g_x = np.squeeze(
            np.matmul(Bq_T, np.expand_dims(g_q, axis=2)), axis=2
        )  # gradient in Cartesian coordinate. [nbatch, 3n ]

        return g_x

    # for prediction: H_q -> H_x. transform hessian.
    def transform_internal_hessian_to_cartesian_hessian(self, x, g_q, H_q):
        """
        See eq.(18) in Faraday Discuss., 2018, 212, 237
        transform from internal coordinate's hessian H to cartesian coordinate hessian H.
        In this case, we need hessian in all dofs.
        :param: x: Cartesian coordinate. size: [nbatch, 3 * natom]
        :param: g_q: gradients in non-redundant internal coordinates. shape: [nbatch, internal_dof_#]
        :param: H_q: Hessians in non-redundant internal coordinates. shape: [nbatch, internal_dof_#, internal_dof_#]

        :return: H_x: Hessian in Cartesian dofs.
        """
        nbatch = np.shape(x)[0]
        assert (
            nbatch == np.shape(g_q)[0]
        ), "the number of data points with gradients for internal coordinate transform is wrong."
        assert (
            nbatch == np.shape(H_q)[0]
        ), "the number of data points with hessians for internal coordinate transform is wrong."

        Bq = self.compute_delocalized_wilson_matrix_Bq(
            x
        ) # \partial q / \partial x. shape:[nbatch, 3n -6, 3n]

        Bq_T = np.transpose(
            Bq, axes=(0, 2, 1)
        )  # transpose of Bq. shape: [nbatch, 3n, 3n-6]

        # shape: [nbatch, 3*natom, 3*natom]
        H_x_part1 = np.matmul(np.matmul(Bq_T, H_q), Bq)
        # H_x_part2 = g_q^T * U^T * (partial^2 d / partial x partial x'). here (partial^2 d / partial x partial x') is a tensor.
        g_q_T = np.expand_dims(g_q, axis=1)  # shape : [nbatch, 1, 3n -6]
        # g_q^T * U^T.  shape: [nbatch, natom^2]
        prefactor = np.squeeze(np.matmul(g_q_T, self.ref_UT), axis=1)

        # compute hessian_d: rank-3 tensor for each data point.. size [nbatch, natom^2, 3 * natom, 3 * natom]
        hessian_d = self._compute_hessian_d(x)

        # shape: [nbatch, 3 * natom, 3 * natom]
        H_x_part2 = np.sum(prefactor[:, :, np.newaxis, np.newaxis] * hessian_d, axis=1)

        H_x = H_x_part1 + H_x_part2

        return H_x

    # g_x -> g_q
    def transform_cartesian_gradient_to_internal_gradient(self, x, g_x):
        """
        transform from Cartesian coordinate system's gradient g_x to internal coordinate gradient g_q.
        :param: x: Cartesian coordinate. shape: [nbatch, 3 * natom]
        :param: g_x: gradient in Cartesian coordinate. shape:[nbatch, 3 * natom]

        :return g_q: gradient in internal coordinate. shape:[nbatch, 3 * natom - 6]
        """
        Bq = self.compute_delocalized_wilson_matrix_Bq(
            x
        ) # \partial q / \partial x. shape:[nbatch, 3n -6, 3n]

        Bq_T = np.transpose(
            Bq, axes=(0, 2, 1)
        )  # transpose of Bq. shape:[nbatch, 3n, 3n - 6]

        # inverse of Bq_T matrix.
        inverse_Bq_T = np.array(
            [
                np.linalg.pinv(Bq_T_element, rcond=np.power(10.0, -8))
                for Bq_T_element in Bq_T
            ]
        )

        # shape: [nbatch, 3n-6]
        g_q = np.squeeze(np.matmul(inverse_Bq_T, g_x[:, :, np.newaxis]), axis=2)

        return g_q

    # for training: H_x -> H_q
    def transform_cartesian_hessian_to_internal_hessian(self, x, g_x, H_x):
        """
        See eq.(21) in Faraday Discuss., 2018, 212, 237
        transform from Cartesian hessian H to internal dofs hessian H.
        In this case, we need hessian in all dofs.

        :param: x: Cartesian coordinate. shape: [nbatch, 3 * natom]
        :param: g_x: gradient in Cartesian coordinate. shape:[nbatch, 3 * natom]
        :param: H_x: hessian in Cartesian coordinate.  shape:[nbatch, 3 * natom, 3 * natom]

        :return: H_q: (only if hessian_bool = True). hessian in internal coordinate. shape:[nbatch, 3 * natom - 6, 3 * natom - 6]
        """
        nbatch = np.shape(x)[0]
        assert (
            nbatch == np.shape(g_x)[0]
        ), "the number of data points with gradients for internal coordinate transform is wrong."
        assert (
            nbatch == np.shape(H_x)[0]
        ), "the number of data points with hessians for internal coordinate transform is wrong."

        Bq = self.compute_delocalized_wilson_matrix_Bq(
            x
        ) # \partial q / \partial x. shape:[nbatch, 3n -6, 3n]

        Bq_T = np.transpose(
            Bq, axes=(0, 2, 1)
        )  # transpose of Bq. shape:[nbatch, 3n, 3n - 6]

        # inverse of Bq_T matrix.
        inverse_Bq_T = np.array(
            [
                np.linalg.pinv(Bq_T_element, rcond=np.power(10.0, -8))
                for Bq_T_element in Bq_T
            ]
        )

        # gradient in internal dofs
        g_q = np.squeeze(np.matmul(inverse_Bq_T, g_x[:, :, np.newaxis]), axis=2)

        H_x_shape = np.shape(H_x)
        assert (H_x_shape[1] == 3 * self.natom) and (
            H_x_shape[2] == 3 * self.natom
        ), "shape of Hessian matrix in cartesian coordinate H_x is wrong."

        inverse_Bq = np.transpose(inverse_Bq_T, axes=(0, 2, 1))

        # Below we reverse the computation in transform_internal_hessian_to_cartesian_hessian
        # compute g_q^{T} \partial B_q / \partial x.  shape [nbatch, 1, 3n-6]
        g_q_T = np.expand_dims(g_q, axis=1)
        # g_q^T * U^T.  shape: [nbatch, natom^2]
        prefactor = np.squeeze(np.matmul(g_q_T, self.ref_UT), axis=1)

        # compute hessian_d: rank-3 tensor. size [nbatch, natom^2, 3 * natom, 3 * natom]
        hessian_d = self._compute_hessian_d(x)
        # shape: [nbatch, 3 * natom, 3 * natom]
        H_x_part2 = np.sum(prefactor[:, :, np.newaxis, np.newaxis] * hessian_d, axis=1)

        H_x_part1 = np.subtract(H_x, H_x_part2)

        H_q = np.matmul(np.matmul(inverse_Bq_T, H_x_part1), inverse_Bq)

        return H_q

    def _compute_q_hessian_x(self, x):
        """
        compute the hessian of internal coordinate q for the reference point.
        d^2 q/ dx^2.
        """
        # shape: [n^2, 3n, 3n]
        hessian_d_xx = self._compute_hessian_d(np.array([x]))[0]
        # shape: [3n, 3n, 3n-6]
        hessian_q_xx = np.matmul(
            self.ref_UT[np.newaxis, np.newaxis, :, :],
            np.transpose(hessian_d_xx, (1, 2, 0))[..., np.newaxis],
        ).squeeze(-1)
        # shape: [3n-6, 3n, 3n]
        hessian_q_xx = np.transpose(hessian_q_xx, (2, 0, 1))
        return hessian_q_xx

    def compute_x_hessian_q(self, x_list):
        """
        compute the hessian of Cartesian coordinate x for the reference point with respect to internal coordinate q.
        d^2 x/ dq^2
        """
        hessian_x_qq_list = []
        for x in x_list:
            # d^2 q/ dx^2. shape: [3n-6, 3n, 3n]
            hessian_q_xx = self._compute_q_hessian_x(x)

            Bq = self.compute_delocalized_wilson_matrix_Bq(
                np.array([x])
            )[
                0
            ] # \partial q / \partial x. shape:[3n -6, 3n]

            # \partial x / partial q. shape: [3n, 3n-6]
            inverse_Bq = np.linalg.pinv(Bq, rcond=np.power(10.0, -8))

            # now compute d^2 x/ dq^2 = (-1) * (dx /dq) (d^2 q/ dx^2) * (dx/dq) * (dx/dq). shape:[3n-6, 3n-6, 3n-6]. The first index is the index of x in numerator.
            hessian_q_qq = (-1) * np.matmul(
                np.transpose(np.matmul(hessian_q_xx, inverse_Bq), (0, 2, 1)), inverse_Bq
            )
            # shape: [3n-6, 3n-6, 3n]
            hessian_x_qq = np.matmul(
                np.transpose(hessian_q_qq, (1, 2, 0)), np.transpose(inverse_Bq, (1, 0))
            )
            hessian_x_qq = np.transpose(hessian_x_qq, (2, 0, 1))
            hessian_x_qq_list.append(hessian_x_qq)
        
        hessian_x_qq_list = np.array(hessian_x_qq_list)

        return hessian_x_qq_list
