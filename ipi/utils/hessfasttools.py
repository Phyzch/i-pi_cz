"""Functions that compute hessians efficiently in the internal coordinate.
We also treat rigid internal coordinate & flexible internal coordinate differently.
For rigid internal dofs, we only compute their hessian parts once.
For flexible internal dofs, we compute hessian for all beads provided. """

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

from ipi.utils.internal.ZmatrixInternal import non_redundant_coordinate_transformer
import torch
import h5py
import os
import numpy as np
from ipi.utils.messages import verbosity, info
from sklearn.linear_model import LinearRegression, Ridge 

class SelectiveHessianCalculation:
    """
    perform hessian calculations in the internal coordinate.
    for rigid dofs, we sample it with smaller number of beads
    and use linear regression to fit hessian of other beads.

    The class should store the training set of Hessian along rigid dofs for linear regression.
    """
    def __init__(self,
                 train_x: np.ndarray,
                 coordinate_transformer: non_redundant_coordinate_transformer,
                 rigid_internal_dofs_cutoff: np.ndarray,
                 cross_validation_bool
                 ):
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
        self.cross_validation_bool = cross_validation_bool

        self.nbeads = np.shape(train_x)[0]
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

        print(f"@hessian calculation: determining rigid internal dofs for ring polymers, \
              for reference, train_inputs_change: {train_inputs_change}")
        
        print(f"The cutoff value for determining rigid internal dofs is: {rigid_internal_dofs_cutoff}")
        
        
        self.rigid_internal_dofs = np.array(
            [
                i for i in range(input_dim)
                if train_inputs_change[i] < rigid_internal_dofs_cutoff
            ]
        )

        print(f"@hessian calculation: rigid internal dofs: {self.rigid_internal_dofs}")

        if len(self.rigid_internal_dofs) != 0:
            self.flexible_internal_dofs = np.delete(np.arange(input_dim), self.rigid_internal_dofs)
        else:
            self.flexible_internal_dofs = np.arange(input_dim)

        # for hessian matrix operation.
        if len(self.rigid_internal_dofs) != 0:
            self.rigid_internal_dofs_2d_index = np.meshgrid(self.rigid_internal_dofs,
                                                            self.rigid_internal_dofs,
                                                            indexing= 'ij')
            self.cross_term_2d_index = np.meshgrid(self.rigid_internal_dofs,
                                                   self.flexible_internal_dofs,
                                                   indexing= 'ij')

        print(f"@hessian calculation: flexible internal dofs: {self.flexible_internal_dofs}")

        self.rigid_mode_train_q_dataset = np.array([])
        self.rigid_mode_hessians_q_dataset = np.array([])
        self.rigid_mode_bead_index = np.array([])
        self.rigid_dofs_reg_model = None 
        self.cross_term_reg_model = None 

    def load_rigid_dofs_hessian(self, prefix):
        """
        load the hessian components along rigid internal dofs.
        :param: prefix: folder that will store hessian data along rigid modes
        """
        info("Load rigid hessian component", verbosity.low)
        file_name = "rigid_mode_hessian.h5"
        h5_file_path = os.path.join(prefix, file_name)
        
        if not os.path.exists(h5_file_path):
            print("no rigid mode hessian data. skip rigid hessian data loading step.")
            return

        with h5py.File(h5_file_path, "r") as h5f:
            self.rigid_mode_train_q_dataset = np.array(h5f["rigid_mode_train_q"])
            self.rigid_mode_hessians_q_dataset = np.array(h5f["rigid_mode_hessians_q"])
            self.rigid_mode_bead_index = np.array(h5f["rigid_mode_bead_index"])

            print(f"@bead index for hessian along rigid mode: {self.rigid_mode_bead_index}")

    def store_rigid_dofs_hessian(self, prefix):
        """
        store hessian components along rigid internal dofs.
        :param: prefix: folder that will store hessian data along rigid modes.
        """
        info("store rigid hessian component", verbosity.low)
        file_name = "rigid_mode_hessian.h5"
        h5_file_path = os.path.join(prefix, file_name)

        with h5py.File(h5_file_path, "w") as h5f:
            h5f.create_dataset("rigid_mode_train_q", 
                               data= self.rigid_mode_train_q_dataset, compression= 'gzip')
            
            h5f.create_dataset("rigid_mode_hessians_q", 
                               data= self.rigid_mode_hessians_q_dataset, compression= 'gzip')
            h5f.create_dataset("rigid_mode_bead_index",
                               data= self.rigid_mode_bead_index, compression= 'gzip')
        

    
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

        # dx = inverse_Bq @ dq[np.newaxis, np.newaxis, :]
        dx = np.einsum('pmn,n->pm', inverse_Bq, dq)

        # compute force:
        # PLUS
        x = np.copy(train_x1)
        x = x + dx 
        x_plus = np.copy(x)
        beads.q[:] = x 
        g1 = - forces.f 
        g1q = self.coordinate_transformer.transform_cartesian_gradient_to_internal_gradient(
            x_plus,
            g1
        )
        # MINUS
        x = np.copy(train_x1)
        x = x - dx 
        x_minus = np.copy(x)
        beads.q[:] = x 
        g2 = - forces.f 
        g2q = self.coordinate_transformer.transform_cartesian_gradient_to_internal_gradient(
            x_minus,
            g2
        )
        # COMBINE 
        # transform each gradient into internal coordinate. Then compute hessian component 
        gq = (g1q - g2q) / (2 * d)
        
        hess[:, index_q, :] = gq
        hess[:, :, index_q] = gq 
    
    def compute_rigid_dofs_hessian(self, 
                                   new_train_x,
                                   new_rp_bead,
                                   new_rp_force,
                                   new_rigid_mode_bead_index):
        """
        compute hessians along rigid internal dofs for new data point.
        This code is used to add new data point for hessian along rigid dofs.

        :param: new_train_x: new data point in cartesian coordinate to compute hessian in rigid dofs. 
        :param: new_rp_bead: ring polymer beads for new data point for rigid dofs sampling.
        :param: new_rp_force: force engine for new data point for rigid dofs sampling.
        :return: new_rigid_hessian_component: The computed hessian components along the rigid dofs.
        """
        new_rigid_bead_num = np.shape(new_train_x)[0]
        new_train_q = self.coordinate_transformer.get_internal_coordinate_q(new_train_x)
        assert new_rp_bead.nbeads == new_rigid_bead_num, "The bead number does not match for rigid dofs hessian calculation."
        info(" @get hessian: rigid mode. Computing hessians.")
        rigid_ndofs = len(self.rigid_internal_dofs)
        input_dim_q = np.shape(new_train_q)[1]
        new_rigid_hessian_component = np.zeros([new_rigid_bead_num, input_dim_q, input_dim_q])

        for new_index in new_rigid_mode_bead_index:
            if new_index in self.rigid_mode_bead_index:
                raise ValueError(f"repeated data points for rigid mode hessians. new_index: {new_index}." + 
                                 f"bead index that we have already computed: {self.rigid_mode_bead_index}")

        for index, index_q in enumerate(self.rigid_internal_dofs):
            info(
                "@get hessian: rigid mode. Computing hessian: %d of %d" %(index, rigid_ndofs),
                verbosity.low
            )
            self.get_internal_coordinate_hessian_component(
                new_train_x,
                new_train_q,
                new_rp_bead,
                new_rp_force,
                new_rigid_hessian_component,
                index_q
            )
        
        if len(self.rigid_mode_train_q_dataset) == 0:
            self.rigid_mode_train_q_dataset = new_train_q
            self.rigid_mode_bead_index = new_rigid_mode_bead_index 
        else:
            self.rigid_mode_train_q_dataset = np.concatenate(
                    [self.rigid_mode_train_q_dataset, new_train_q], 
                    axis= 0
                )
            self.rigid_mode_bead_index = np.concatenate([self.rigid_mode_bead_index, 
                                                         new_rigid_mode_bead_index])
        
        if len(self.rigid_mode_hessians_q_dataset) == 0:
            self.rigid_mode_hessians_q_dataset = new_rigid_hessian_component
        else:
            self.rigid_mode_hessians_q_dataset = np.concatenate(
                    [self.rigid_mode_hessians_q_dataset, new_rigid_hessian_component],
                    axis= 0
                )
        
    def linear_regression_fit_hessian(self, 
                                      ridge_regularization_alpha= 0.1):
        """
        fit the hessian along rigid modes using linear regression model.
        We construct linear regression model in this function.

        :param: ridge_regularization_alpha: the regularization strength for ridge regression.
        """
        data_num = len(self.rigid_mode_hessians_q_dataset)

        assert data_num != 0, "The hessian data point for rigid mode should be larger than 0."
        # use scikit learn linear regression fit.
        # x_shape: [n_samples, n_features]
        # y_shape: [n_samples, n_targets]
        # we need to flatten hessians into 1d array [n_targets] for each sample.
        rigid_dofs_hessians = self.rigid_mode_hessians_q_dataset[:, self.rigid_internal_dofs_2d_index[0], self.rigid_internal_dofs_2d_index[1]]
        y = rigid_dofs_hessians.reshape((data_num, -1))
        x = np.copy(self.rigid_mode_train_q_dataset)
        rigid_dofs_reg_model = Ridge(alpha= ridge_regularization_alpha).fit(x, y)

        cross_term_hessians = self.rigid_mode_hessians_q_dataset[:,
                                                                 self.cross_term_2d_index[0],
                                                                 self.cross_term_2d_index[1]]
        
        y1 = cross_term_hessians.reshape((data_num, -1))
        x1 = np.copy(self.rigid_mode_train_q_dataset)
        cross_term_reg_model = Ridge(alpha= ridge_regularization_alpha).fit(x1, y1)
        cross_term_reg_model.fit(x1, y1)

        self.rigid_dofs_reg_model = rigid_dofs_reg_model
        self.cross_term_reg_model = cross_term_reg_model

        # compute training data prediction error
        y_predicted = rigid_dofs_reg_model.predict(x)
        y_diff = y_predicted - y 
        rigid_dofs_hess_training_error = np.linalg.norm(y_diff, axis= 1) / np.linalg.norm(y, axis= 1)

        y1_predicted = cross_term_reg_model.predict(x1)
        y1_diff = y1_predicted - y1 
        cross_term_hess_training_error = np.linalg.norm(y1_diff, axis= 1) / np.linalg.norm(y1, axis= 1)
        print(f"@rigid_dofs fast_hess: training error for rigid dofs hessian (block diagonal): {rigid_dofs_hess_training_error}")
        print(f"@rigid_dofs fast_hess: training error for rigid dofs hessian (cross term): {cross_term_hess_training_error}")
    
    def linear_regression_cross_validation(self,
                                           ridge_regularization_alpha= 0.1):
        """
        use leave one out cross-validation method to test the performance of the linear regression model.

        :param: ridge_regularization_alpha: the regularization strength for ridge regression.
        """
        np.random.seed(seed= 41)
        data_num = len(self.rigid_mode_train_q_dataset)
        cv_index = np.random.choice(data_num, size= 1)

        # cross validation data
        cv_data_num = 1
        cv_train_q = self.rigid_mode_train_q_dataset[cv_index]
        cv_hessians_q = self.rigid_mode_hessians_q_dataset[cv_index]
        
        # training data after we leave one data point (cv) out
        training_data_num = data_num - cv_data_num
        training_q = np.delete(self.rigid_mode_train_q_dataset, cv_index, axis= 0)
        hessians_q = np.delete(self.rigid_mode_hessians_q_dataset, cv_index, axis= 0)

        # use linear regression fit 
        # for block diagonal terms along rigid dofs.
        rigid_mode_hessians = hessians_q[:, self.rigid_internal_dofs_2d_index[0], self.rigid_internal_dofs_2d_index[1]]
        y = rigid_mode_hessians.reshape((training_data_num, -1))
        x = np.copy(training_q)
        rigid_dofs_reg_model = Ridge(alpha= ridge_regularization_alpha).fit(x, y)

        # test the accuracy 
        predict_y = rigid_dofs_reg_model.predict(cv_train_q)
        cv_y = cv_hessians_q[:,self.rigid_internal_dofs_2d_index[0], self.rigid_internal_dofs_2d_index[1]].reshape((cv_data_num, -1))
        error = np.linalg.norm(predict_y - cv_y, axis= 1) / np.linalg.norm(cv_y, axis= 1)
        print(f"cross validation error for rigid mode linear regression: block diagonal term: {error}")

        # for cross term
        cross_term_hessians = hessians_q[:,
                                        self.cross_term_2d_index[0],
                                        self.cross_term_2d_index[1]]
        y1 = cross_term_hessians.reshape((training_data_num, -1))
        x1 = np.copy(training_q)
        cross_term_reg_model = Ridge(alpha= ridge_regularization_alpha).fit(x1, y1)

        # test the accuracy.
        predict_y1 = cross_term_reg_model.predict(cv_train_q)
        cv_y1 = cv_hessians_q[:, self.cross_term_2d_index[0], self.cross_term_2d_index[1]].reshape((cv_data_num, -1))
        error1 = np.linalg.norm(predict_y1 - cv_y1, axis= 1) / np.linalg.norm(cv_y1, axis= 1)
        print(f"cross validation error for rigid mode linear regression: cross term: {error1}")

        pass


    def linear_regression_predict_hessian(self, 
                                          predict_inputs: np.ndarray, 
                                          hessians_q: np.ndarray):
        """
        predict hessians along constrained dofs using Linear regression model.

        :param: predict_inputs: input data for linear regression model to predict hessians.
        :param: hessians_q: hessian data. The data along constrained dofs will be predicted by linear regression model.
        """
        data_num = predict_inputs.shape[0]
        num_rigid_dofs = len(self.rigid_internal_dofs)
        num_flexible_dofs = len(self.flexible_internal_dofs)

        # predict block diagonal component for rigid dofs
        predict_rigid_hessians = self.rigid_dofs_reg_model.predict(predict_inputs)
        predict_rigid_hessians = predict_rigid_hessians.reshape((data_num, num_rigid_dofs, num_rigid_dofs))
        hessians_q[:, self.rigid_internal_dofs_2d_index[0], self.rigid_internal_dofs_2d_index[1]] = (
            predict_rigid_hessians
        )

        # predict cross term between rigid dofs and flexible dofs.
        predict_cross_term_hessians = self.cross_term_reg_model.predict(predict_inputs)
        predict_cross_term_hessians = predict_cross_term_hessians.reshape((data_num, num_rigid_dofs, num_flexible_dofs))
        hessians_q[:, self.cross_term_2d_index[0], self.cross_term_2d_index[1]] = predict_cross_term_hessians
        hessians_q[:, self.cross_term_2d_index[1], self.cross_term_2d_index[0]] = predict_cross_term_hessians

    
    def rigid_modes_hessian_preprocess(self, 
                                       prefix,
                                       new_train_x= [],
                                       new_rp_bead= None,
                                       new_rp_force= None,
                                       new_rigid_mode_bead_index= [],
                                       ridge_regularization_alpha= 0.1):
        """
        prepare for linear regression prediction of rigid hessian components.
        (1) load existing hessian data along rigid dofs.
        (2) compute hessians along rigid dofs for new data point (new_train_x)
        (3) construct linear regression model.

        :param: prefix: folder that contains information about hessians along rigid modes.
        """
        self.load_rigid_dofs_hessian(prefix)
        
        # compute new data point for rigid dofs.
        if len(new_train_x) > 0:
            self.compute_rigid_dofs_hessian(new_train_x, new_rp_bead, new_rp_force, new_rigid_mode_bead_index)
        
        # construct the linear regression model.
        self.linear_regression_fit_hessian(ridge_regularization_alpha= ridge_regularization_alpha)

        # do cross validation for linear regression fit of hessians.
        # Only do this if we have hessian data point >= 3. (need at least 2 point for linear regression.)
        if self.cross_validation_bool and len(self.rigid_mode_train_q_dataset) >= 3:
            self.linear_regression_cross_validation(ridge_regularization_alpha= ridge_regularization_alpha)

    def get_hessian(self, rp_beads, rp_forces, x0, d= 0.001):
        """
        Compute hessian as finite difference of forces in the internal coordinate.
        Then transform hessian in internal coordinate back to Cartesian coordinate.

        For rigid modes, we use linear regression model to generate hessian components.
        For flexible modes, we compute their hessian components for all beads.
        """
        if self.rigid_dofs_reg_model is None:
            raise ValueError("Forget to preprocess the linear regression model for hessians along rigid modes?")
        
        q = self.coordinate_transformer.get_internal_coordinate_q(x0)
        nbeads = rp_beads.nbeads
        assert np.shape(x0)[0] == nbeads, "The shape of x0 does not match number of beads for Bead object."

        hess_q = np.zeros([nbeads, self.input_dim, self.input_dim])

        # use linear regression to fit hessian along rigid modes.
        self.linear_regression_predict_hessian(q, hess_q)

        # compute hessian along flexible mode
        info(" @get hessian: Flexible modes: Computing hessian", verbosity.low)
        flexible_ndofs = len(self.flexible_internal_dofs)
        for index, index_q in enumerate(self.flexible_internal_dofs):
            info(
                " @get_hessian: flexible modes: Computing hessian: %d of %d" %(index, flexible_ndofs),
                verbosity.low
            )
            self.get_internal_coordinate_hessian_component(x0, q, rp_beads, rp_forces, hess_q, index_q, d= d)

        # transform hessian (computed + fitted) from internal coordinate q back to the Cartesian coordinate x.
        # compute gradient: g_x
        rp_beads.q[:] = np.copy(x0)
        gx = - rp_forces.f
        gq = self.coordinate_transformer.transform_cartesian_gradient_to_internal_gradient(np.copy(x0), gx)

        hess_x = self.coordinate_transformer.transform_internal_hessian_to_cartesian_hessian(
            x0, gq, hess_q
        )

        return hess_x 
    
    def update_hessian_rigid_modes(self, x, f_x, hess_x):
        """
        update the hessian along rigid mode in internal dofs with prediction from linear regression model.
        
        :param: x: cartesian coordinate.
        :param: f_x: force
        :param: hess_x: hessian. 
        """
        g_x = - f_x 
        hess_q = self.coordinate_transformer.transform_cartesian_hessian_to_internal_hessian(x, g_x, hess_x)
        q = self.coordinate_transformer.get_internal_coordinate_q(x)

        # update rigid modes with linear regression.
        self.linear_regression_predict_hessian(q, hess_q)

        # transform it back to cartesian coordinate.
        g_q = self.coordinate_transformer.transform_cartesian_gradient_to_internal_gradient(x, g_x)
        hess_x = self.coordinate_transformer.transform_internal_hessian_to_cartesian_hessian(x, g_q, hess_q)

        return hess_x
    

