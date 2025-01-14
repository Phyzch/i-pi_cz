"""Functions that compute hessians efficiently in the internal coordinate.
We also treat rigid internal coordinate & flexible internal coordinate differently.
For rigid internal dofs, we only compute their hessian parts once.
For flexible internal dofs, we compute hessian for all beads provided. """

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

from ipi.utils.internal.internaltools import non_redundant_coordinate_transformer
import torch
import numpy as np
from ipi.utils.messages import verbosity, info

class SelectiveHessianCalculation:
    """
    perform hessian calculations in the internal coordinate.
    for rigid dofs, we only compute it for one bead and use the value for all other ring polymer beads.
    """
    def __init__(self,
                 train_x: np.ndarray,
                 coordinate_transformer: non_redundant_coordinate_transformer,
                 rigid_internal_dofs_cutoff: np.ndarray,
                 single_rp_bead,
                 single_rp_force):
        """
        decide the rigid_internal_dofs for the training data. 
        :param: train_x: all training data that we are interested in computing hessians about.
                        This can be Cartesian coordinate of all candidate ring polymer beads to compute hessians.
        :param: coordinate_transformer: The object (class) that transform between Cartesian coordinate 
                and non-redundant internal coordinate.
        :param: rigid_internal_dofs_cutoff: cutoff value to treat certain internal coordinate as rigid. 
        :param: single_rp_bead: Bead object. only 1 ring polymer bead, use this to compute hessian along rigid internal dofs.
        :param: single_rp_force: Force object for 1 ring polymer bead.
        """
        self.coordinate_transformer = coordinate_transformer
        assert single_rp_bead.nbeads == 1, "The bead number for single_rp_bead not equal to 1."
        self.single_rp_bead = single_rp_bead 
        self.single_rp_force = single_rp_force 

        # the change along the internal coordinate will be computed using Wilson's B matrix.
        # This is to treat planar molecules.
        Bq = coordinate_transformer.compute_delocalized_wilson_matrix_Bq(np.array([train_x[0]]))[0]
        (u, sq, vh) = np.linalg.svd(Bq, full_matrices= False)
        
        # compute the change of training data along the Cartesian coordinate.
        # This is for determining the rigid mode.
        train_x_change = np.max(train_x, axis= 0) - np.min(train_x, axis= 0)

        train_inputs = coordinate_transformer.get_internal_coordinate_q(train_x)
        input_dim = train_inputs.shape[1]
        self.input_dim = input_dim 

        # compute the change of training input. 
        # To be safe, we compute it using two approaches and choose the smaller one among two:
        train_inputs_change1 = np.abs(sq * (vh @ train_x_change))
        train_inputs_change2 = np.max(train_inputs, axis= 0) - np.min(train_inputs, axis= 0)
        train_inputs_change = np.min([train_inputs_change1, train_inputs_change2], axis= 0)

        self.train_inputs_change = train_inputs_change 

        print(f"@hessian calculation: determining rigid internal dofs for ring polymers, 
              for reference, train_inputs_change: {train_inputs_change}")
        
        print(f"The cutoff value for determining rigid internal dofs is: {rigid_internal_dofs_cutoff}")
        self.rigid_internal_dofs = np.array(
            [
                i for i in range(input_dim)
                if train_inputs_change[i] < rigid_internal_dofs_cutoff
            ]
        )

        print(f"@hessian calculation: rigid internal dofs: {self.rigid_internal_dofs}")

        self.flexible_internal_dofs = np.delete(np.arange(input_dim), self.rigid_internal_dofs)

        self.rigid_hessian_component_bool = False 
        self.rigid_hessian_component = np.zeros([input_dim, input_dim])
    
    def get_internal_coordinate_hessian_component(self, train_x, train_q, beads, forces, hess, index_q, d= 0.001):
        """
        compute the hessian component along internal coordinate index i: Hq[i, :].
        component of hessian matrix along dof (index_q) is computed and stored in hess. 
        :param: train_x: Cartesian coordinate x for training data.  shape: [nbeads, 3 * natoms]
        :param: train_q: delocalized internal coordinate q for training data. shape: [nbeads, 3 * natoms - 6]
        :param: beads: bead object. The shape of beads.q should be the same as train_x. beads.q shape: [nbeads, 3 * natoms]
        :param: forces: force object. Use this to call force driver to give force f. 
        :param: hess: hessian matrix, store the computed hessian component. Shape: [nbeads, 3 * natom - 6, 3 * natom - 6]
        :param: index_q: index in internal coordinate to compute hessian.
        :param: d: displacement. default value: 0.001

        """
        train_x1 = np.copy(train_x)
        train_q1 = np.copy(train_q)

        ndofs_q = np.shape(train_q)[1]
        ndofs_x = np.shape(train_x)[1]

        assert index_q < ndofs_q, "The index_q is larger than the number of dofs for internal coordinate q."
        dq = np.zeros([ndofs_q])
        dq[index_q] = d 

        Bq = self.coordinate_transformer.compute_delocalized_wilson_matrix_Bq(train_x)
        inverse_Bq = np.array(
            [
                np.linalg.pinv(Bq_element) for Bq_element in Bq 
            ]
        )

        dx = inverse_Bq @ dq[np.newaxis, np.newaxis, :]

        # compute force:
        # PLUS
        x = np.copy(train_x1)
        x = x + dx 
        beads.q[:] = x 
        g1 = - forces.f 

        # MINUS
        x = np.copy(train_x1)
        x = x - dx 
        beads.q[:] = x 
        g2 = - forces.f 

        # COMBINE 
        gx = (g1 - g2)/ (2 * d)
        gq = self.coordinate_transformer.transform_cartesian_gradient_to_internal_gradient(
            train_x1,
            gx)
        
        hess[:, index_q, :] = gq
        hess[:, :, index_q] = gq 
    
    def compute_rigid_dofs_hessian(self, train_x, train_q):
        """
        give hessians along rigid internal dofs.
        If self.rigid_hessian_component_bool = False, we compute hessians along rigid internal dofs.
           result stored in self.rigid_hessian_component.
        If self.rigid_hessian_component_bool = True. we do nothing.

        :param: train_x: data in cartesian coordinate. should be only 1 bead.
        :param: train_q: data in internal coordinate. should be only 1 bead.
        """
        if  not self.rigid_hessian_component_bool:
            self.rigid_hessian_component_bool = True 
            assert np.shape(train_x)[0] == 1, "The shape of train_x passed to compute hessians along rigid dofs is not 1."

            info(" @get hessian: rigid mode. Computing hessians.")
            rigid_ndofs = len(self.rigid_internal_dofs)
            for index, index_q in enumerate(self.rigid_internal_dofs):
                info(
                    "@get hessian: rigid mode. Computing hessian: %d of %d" %(index, rigid_ndofs),
                    verbosity= "low"
                )
                self.get_internal_coordinate_hessian_component(
                    train_x,
                    train_q,
                    self.single_rp_bead,
                    self.single_rp_force,
                    self.rigid_hessian_component,
                    index_q
                )
            
    def load_rigid_dofs_hessian(self, train_x, grad_x, hessian_x):
        """
        load the hessian components along rigid internal dofs.
        This function is used when we load hessian data from file, 
        so, we do not need to re-compute hessian for rigid components.
        :param: train_x: training data in Cartesian coordinate.
        :param: grad_x : gradient data in Cartesian coordinate. 
        :param: hessian_x: hessian data in Cartesian coordinate. 
        """
        if not self.rigid_hessian_component_bool:
            self.rigid_hessian_component_bool = True 
            hessian_q = self.coordinate_transformer.transform_cartesian_hessian_to_internal_hessian(
                train_x,
                grad_x,
                hessian_x
            )

            hessian_q_mean = np.mean(hessian_q, axis= 0)

            for index_q in self.rigid_internal_dofs:
                self.rigid_hessian_component[index_q, :] = hessian_q_mean[index_q, :]
                self.rigid_hessian_component[:, index_q] = hessian_q_mean[:, index_q]
        
    def get_hessian(self, rp_beads, rp_forces, x0, d= 0.001):
        """
        Compute hessian as finite difference of forces in the internal coordinate.
        Then transform hessian in internal coordinate back to Cartesian coordinate.

        For rigid modes, we compute it once for 1 bead or load it if it has been computed previously.
        For flexible modes, we compute their hessian components for all beads.
        """
        q = self.coordinate_transformer.get_internal_coordinate_q(x0)
        nbeads = rp_beads.nbeads 
        assert np.shape(x0)[0] == nbeads, "The shape of x0 does not match number of beads for Bead object."

        hess_q = np.zeros([nbeads, self.input_dim, self.input_dim])

        # compute hessian_q along rigid components if we haven't done so.
        if not self.rigid_hessian_component_bool:
            x_single_bead = np.array([x0[0]])
            q_single_bead = np.array([q[0]])
            self.compute_rigid_dofs_hessian(x_single_bead,q_single_bead)

        # load rigid hessian into dataset.
        hess_q_rigid = np.repeat(self.rigid_hessian_component[np.newaxis, :], nbeads, axis= 0)
        for index_q in self.rigid_internal_dofs:
            hess_q[:, index_q, :] = hess_q_rigid[:, index_q, :]
            hess_q[:,:, index_q]  = hess_q_rigid[:, :, index_q]
        
        # compute hessian along flexible modes. Results store in hess_q
        info(" @get hessian: Flexible modes: Computing hessian", verbosity.low)
        flexible_ndofs = len(self.flexible_internal_dofs)
        for index, index_q in enumerate(self.flexible_internal_dofs):
            info(
                " @get_hessian: flexible modes: Computing hessian: %d of %d" %(index, flexible_ndofs),
                verbosity.low
            )
            self.get_internal_coordinate_hessian_component(x0, q, rp_beads, rp_forces, hess_q, index_q, d= d)

        # transform hessian from internal coordinate q back to the Cartesian coordinate x.
        # compute g_x 
        rp_beads.q[:] = np.copy(x0)
        gx = - rp_forces.f 
        gq = self.coordinate_transformer.transform_cartesian_gradient_to_internal_gradient(np.copy(x0), gx)

        hess_x = self.coordinate_transformer.transform_internal_hessian_to_cartesian_hessian(
            x0, gq, hess_q
        )

        return hess_x 
    