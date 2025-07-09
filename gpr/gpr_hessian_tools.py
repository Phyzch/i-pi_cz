"""
packages for predicting hessian of ring polymer beads using Gaussian Process Regression (GPR).
The GPR part use GPytorch framework (https://docs.gpytorch.ai/en/stable/)
Written by Chenghao Zhang & Niri Govind, Pacific Northwest National Laboratory (chenghao.zhang@pnnl.gov), 2024.
"""

import torch
torch.set_default_dtype(torch.float64)

import numpy as np
# from gpr.internalcoordtools import non_redundant_coordinate_transformer
from gpr.internal.ZmatrixInternal import non_redundant_coordinate_transformer # type: ignore
from .gprHessian.RBFHessian_gp import GPModelWithHessians, train_gpr_model
from .gprHessian.RBFHessian_utils import (
    take_upper_triangular_part,
    transform_1d_train_targets_into_pots_grads_hessians,
)
import gpr.gprHessian.RBFHessian_gp
import os 
import shutil
from ipi.utils.messages import  warning
from sklearn.linear_model import LinearRegression, Ridge 
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

class TransformTrainingTarget(object):
    """
    class that handles the transformation between [V, grad_V and hessian_V] & 1d traiining targets for GPR model.
    """

    def __init__(self, ndofs: int, fixdofs: np.ndarray):
        self.ndofs = ndofs
        self.fixdofs = fixdofs

    def transform_pots_grad_hessian_to_1d_data(
        self, pots: np.ndarray, grad_V: np.ndarray, hessians: np.ndarray
    ):
        """
        transform the potential V, gradient of V and hessians into 1d target data.

        :param: pots: potential V. 1d data.
                grad_V: gradient of potential. 2d data. [N,d]
                hessians: hessian of potential. 3d data [N, d', d']
        """
        grad_V_1d = grad_V.flatten()
        hess_triu_1d = take_upper_triangular_part(
            hessians
        ).flatten()  # upper triangular part of hessian.
        train_targets = np.concatenate([pots, grad_V_1d, hess_triu_1d], axis=0)

        return train_targets

    def transform_1d_data_to_pots_grad_hessian(
        self,
        train_targets: np.ndarray,
        train_inputs: np.ndarray,
        hessian_data_point_index: np.ndarray,
    ):
        """
        transform the 1d training targets to potential V, gradient dV/dx and hessian d^2 V/ dx^2
        :param: train_targets: 1d training targets.
        :param: train_inputs: training input [N, d]. here N is number of data points, d is dof.
        :param: hessian_data_point_index: data point index for hessians.
        """
        data_num = train_inputs.shape[0]
        ndofs = train_inputs.shape[1]
        assert ndofs == self.ndofs, "shape of training input is wrong."
        hessian_data_num = len(hessian_data_point_index)

        pots, gradients, hessians = transform_1d_train_targets_into_pots_grads_hessians(
            train_targets, data_num, ndofs, self.fixdofs, hessian_data_num
        )

        return pots, gradients, hessians


class NormalizeTrainingData(object):
    """
    normalize the potential, force and hessian of the training data.
    This will enforce the potential V is in the range of [0, 1]
    """ 
    def __init__(self, 
                 V: np.ndarray,
                 training_inputs: np.ndarray):
        """
        V_normalized = (V - V_mean)/V_range.

        :param: V: potential, 1d array
        """
        self.V_mean = np.mean(V)
        self.V_range = np.max(V) - np.min(V)

        # transform the coordinate. Do it for the initial data.
        # TODO: This code could cause potential trouble when we reload the training data.
        # Because q_mean and q_range depends on the dataset when the model is created.
        # it will change after we add new data when we re-initialize the model
        q_mean = np.mean(training_inputs, axis= 0) # <q>
        q_range = np.max(training_inputs, axis= 0) - np.min(training_inputs, axis= 0)

        self.q_mean = q_mean 
        self.q_range = q_range

        # the number of internal dofs.
        self.q_ndofs = np.shape(training_inputs)[1] 

    # re-scale the training inputs, gradients and hessians.
    def normalization_transform(
        self, 
        V: np.ndarray, 
        grad_V: np.ndarray, 
        hessian_V: np.ndarray,
        train_inputs: np.ndarray
    ):
        """
        normalize the potential, gradients and hessians.
        V_normalized = (V - V_mean) / V_range.
        grad_V_normalized = grad_V / V_range
        hessian_V_normalized = hessian_V / V_range

        This function performs the normalization procedure
        :param: V: potential. 1d array.[N]
                grad_V: gradient of potential, 2d array. [N,d]
                hessian_V: hessian of potential, 3d array. [N,d,d]
                here N is number of data, d is total dof of the system.
        """
        V_normalized = (V - self.V_mean) / self.V_range
        grad_V_normalized = grad_V / self.V_range

        # transform the training_inputs, gradients and hessians.
        train_inputs_normalized = (train_inputs - self.q_mean[np.newaxis, :]) / self.q_range[np.newaxis, :]
        grad_V_normalized = grad_V_normalized * self.q_range[np.newaxis, :]

        # normalization for hessian.
        hessian_V_normalized = self.normalization_transform_for_hessian(
            hessian_V
        )

        return V_normalized, grad_V_normalized, hessian_V_normalized, train_inputs_normalized
    
    # Perform normalization transformation for hessian.
    def normalization_transform_for_hessian(
            self,
            hessian_V: np.ndarray
    ):
        """
        normalize the hessian data.
        """  
        if len(hessian_V) > 0:
            hessian_V_normalized = hessian_V / self.V_range 
            # transform hessian because we re-scale the input.
            q_range_diag_matrix = np.diag(self.q_range, k= 0)
            hessian_V_normalized = np.matmul(np.matmul(q_range_diag_matrix, hessian_V_normalized), q_range_diag_matrix)
        else:
            hessian_V_normalized = np.array([])

        return hessian_V_normalized

    # New function. Perform normalization transformation for inputs.
    def normalization_transform_for_inputs(
            self,
            train_inputs: np.ndarray
    ):
        """
        normalize the training inputs.
        """
        train_inputs_normalized = (train_inputs - self.q_mean[np.newaxis, :]) / self.q_range[np.newaxis, :]
        return train_inputs_normalized

    def inverse_normalization_transform(
        self, 
        V_normalized, 
        grad_V_normalized: np.ndarray, 
        hessian_V_normalized: np.ndarray
    ):
        """
        inverse the normalization procedure for potential V, gradients and hessians.

        V = V_normalized * V_range + V_mean
        F = F_normalized * V_range
        hessian = hessian_normalized * V_range
        """
        V = V_normalized * self.V_range + self.V_mean
        grad_V = grad_V_normalized * self.V_range

        # inverse normalization of gradient and hessian 
        grad_V = grad_V / self.q_range[np.newaxis, :]

        # diagonal matrix with q range.
        inverse_q_range_diag_matrix = np.diag(1 / self.q_range, k= 0)

        if len(hessian_V_normalized) > 0:
            hessian = hessian_V_normalized * self.V_range
            hessian = np.matmul(
                np.matmul(
                    inverse_q_range_diag_matrix, 
                    hessian
                    ), 
                inverse_q_range_diag_matrix
                )
        else:
            hessian = np.array([])

        return V, grad_V, hessian

    def normalize_noise_var(self, 
                            pot_noise_var, 
                            force_noise_var: np.ndarray, 
                            hessian_noise_var: np.ndarray):
        """
        normalize the variance of noise by scaling it by self.V_range.
        Note the re-scaling due to the input rescaling is performed in noise_covar_factor matrix. 
        """
        normalized_pot_noise_var = pot_noise_var / np.power(self.V_range, 2)
        normalized_force_noise_var = force_noise_var / np.power(self.V_range, 2)
        normalized_hessian_noise_var = hessian_noise_var / np.power(self.V_range, 2)

        return (
            normalized_pot_noise_var,
            normalized_force_noise_var,
            normalized_hessian_noise_var,
        )
    

    def inverse_normalize_noise_var(
        self,
        normalized_pot_noise_var,
        normalized_force_noise_var: np.ndarray,
        normalized_hessian_noise_var: np.ndarray,
    ):
        """
        inverse the normalization procedure for the variance of the noise
        """
        pot_noise_var = normalized_pot_noise_var * np.power(self.V_range, 2)
        force_noise_var = normalized_force_noise_var * np.power(self.V_range, 2)

        # add code to inverse the normalization of the force and hessian.
        force_noise_var = force_noise_var / np.power(self.q_range, 2)
        inverse_square_q_range_diag_matrix = np.diag(1 / np.power(self.q_range, 2), k= 0)

        if len(normalized_hessian_noise_var) > 0:
            hessian_noise_var = normalized_hessian_noise_var * np.power(self.V_range, 2)
            hessian_noise_var = np.matmul(
                            np.matmul(
                                inverse_square_q_range_diag_matrix, 
                                hessian_noise_var
                                ), 
                            inverse_square_q_range_diag_matrix
                            )
        else:
            hessian_noise_var = np.array([])

        return pot_noise_var, force_noise_var, hessian_noise_var

    # Normalize the noise covariance factor because we re-scale the training inputs.
    def normalize_noise_covar_factor_array(self,
                                           noise_covar_factor_pot_grad_array: np.ndarray,
                                           noise_covar_factor_with_hessian_array: np.ndarray):
        """
        This function re-scale the noise transformation matrix when we re-scale the training inputs.
        """
        # transformation matrix for data points with only potential and gradient.
        q_size = self.q_ndofs 
        hessian_q_triu_size = int( (q_size + 1) * q_size / 2)

        size_no_hessian = 1 + q_size  # size for data point with only potential and gradient data.
        size_with_hessian = 1 + q_size + hessian_q_triu_size  # size for data point with pot, gradient and hessian data.

        # matrix that normalize the noise covariance factor for data point without hessian 
        matrix_no_hessian = np.zeros(shape= (size_no_hessian, size_no_hessian))
        matrix_no_hessian[0, 0] = 1 
        matrix_no_hessian[1:, 1:] = np.diag(self.q_range, k= 0)

        # matrix that normalize the noise covariance factor for data points with hessian.
        matrix_with_hessian = np.zeros(shape= (size_with_hessian, size_with_hessian))
        matrix_with_hessian[0, 0] = 1
        matrix_with_hessian[1: 1 + q_size, 1: 1 + q_size] = np.diag(self.q_range, k= 0)
        
        # matrix component that re-scale the upper triangle part of hessian.
        hessian_rescale_matrix = np.ones((q_size, q_size))
        q_range_diag_matrix = np.diag(self.q_range, k= 0)
        hessian_rescale_matrix = np.matmul(np.matmul(q_range_diag_matrix, hessian_rescale_matrix), q_range_diag_matrix)
        hessian_rescale_matrix_upper_triangle = take_upper_triangular_part(hessian_rescale_matrix)
        hessian_rescale_matrix = np.diag(hessian_rescale_matrix_upper_triangle)

        matrix_with_hessian[1 + q_size: size_with_hessian, 1 + q_size : size_with_hessian] = hessian_rescale_matrix

        # Now we re-scale the noise covar factor matrix.
        normalized_noise_covar_factor_pot_grad_array = np.matmul(matrix_no_hessian, 
                                                                 noise_covar_factor_pot_grad_array)

        if len(noise_covar_factor_with_hessian_array) > 0:
            normalized_noise_covar_factor_with_hessian_array = np.matmul(
                                                                        matrix_with_hessian, 
                                                                        noise_covar_factor_with_hessian_array
                                                                        )
        else:
            normalized_noise_covar_factor_with_hessian_array = np.array([])

        return normalized_noise_covar_factor_pot_grad_array, normalized_noise_covar_factor_with_hessian_array


class lin_model:
    """
    model that fit linear regression with polynomial degrees of input data.
    We perform feature engineer in the class. 
    """
    def __init__(self, degree, regularization = False, lambda_=0):
        if regularization:
            self.linear_model = Ridge(alpha=lambda_)
        else:
            self.linear_model = LinearRegression()
        self.poly = PolynomialFeatures(degree, include_bias=False)
        self.scaler = StandardScaler()
        
    def fit(self, X_train,y_train):
        ''' just fits the data. mapping and scaling are not repeated '''
        X_train_mapped = self.poly.fit_transform(X_train)
        X_train_mapped_scaled = self.scaler.fit_transform(X_train_mapped)
        self.linear_model.fit(X_train_mapped_scaled, y_train )

    def predict(self, X):
        X_mapped = self.poly.transform(X)
        X_mapped_scaled = self.scaler.transform(X_mapped)
        yhat = self.linear_model.predict(X_mapped_scaled)
        return(yhat)
    

class FixInternalDofs(object):
    """
    class that fix certain internal dofs in the training data before feeding data into the Gaussian Process Regression model.
    This is for the case that certain internal coordinates are identical across different data points.

    we also record the gradient and hessians along certain fixed dofs.
    These results will add back to the prediction of GPR model, then it will be transformed back to get gradients and hessians in Cartesian coordinate.
    
    :param: train_x: training data in Cartesian coordinate.
    :param: train_inputs: the transformed internal coordinate.
    :param: cartesian_fix_dofs: dofs in cartesian coordinate that will be fixed.
    :param: train_inputs: training data in internal coordinate.
    :param: grads: gradients in internal coordinate.
    :param: hessians: hessians in internal coordinate.
    :param: gpr_fix_internal_dofs_bool: bool variable. Fix internal dofs or not.
    :param: gpr_fix_internal_dofs_cutoff: if change of inputs along certain dof is small than certain value, it is fixed.
    """

    def __init__(
        self,
        train_x: np.ndarray,
        train_inputs: np.ndarray,
        hessian_data_point_index_array: np.ndarray,
        cartesian_fix_dofs: np.ndarray,
        coordinate_transformer: non_redundant_coordinate_transformer,
        grads: np.ndarray,
        hessians: np.ndarray,
        gpr_fix_internal_dofs_bool: bool,
        gpr_fix_internal_dofs_cutoff: float,
        rigid_internal_dofs_cutoff: float,
        gpr_fixed_internal_dofs= None,
        gpr_rigid_internal_dofs= None,
        force_ridge_regularization_alpha: float = 0.1,
        hessian_ridge_regularization_alpha: float = 0.1
    ):
        self.input_dim = grads.shape[1]
        self.fix_internal_dofs_cutoff = gpr_fix_internal_dofs_cutoff

        # check whether coordinate alng certain internal dofs need to be fixed.
        # the change along internal coordinate will be computed using Wilson's B matrix. 
        # This is to fix the problem for the planar molecule
        Bq = coordinate_transformer.compute_delocalized_wilson_matrix_Bq(np.array([train_x[0]]))[0]
        (u, sq, vh) = np.linalg.svd(Bq, full_matrices= False)
        # compute change of training data along Cartesian coordinate.
        train_x_change = np.max(train_x, axis= 0) - np.min(train_x, axis= 0)

        # check the case that we do not fix cartesian dofs. 
        # report error if these dofs are small

        train_x_cutoff = pow(10.0, -3)
        index = [i for i in range(len(train_x_change)) if train_x_change[i] < train_x_cutoff]
        if len(index) != 0:
            warning(f"@Warning: Planar molecules? The changes of cartesian coordinate are small.  dofs: {index}. cartesian coordinate change {train_x_change[index]}")
            print(f"Currently fixed cartesian dofs: {cartesian_fix_dofs}")

        # if cartesian dofs is fixed, we set its change to 0.
        train_x_change[cartesian_fix_dofs] = 0 

        train_inputs_change1 = np.abs(sq * (vh @ train_x_change))
        train_inputs_change2 = np.max(train_inputs, axis= 0) - np.min(train_inputs, axis= 0)
        # we use whichever is smaller : the change within internal coordinate or the change infered from cartesian coordinate
        # as the criterion to fix internal dofs.
        train_inputs_change = np.min([train_inputs_change1, train_inputs_change2], axis= 0)

        self.train_inputs_change = train_inputs_change 

        # output information for selecting fix internal dofs value.
        print(f"@For Fixing internal dofs: for reference, train_inputs_change: {train_inputs_change}.")
        print(f"Make sure cutoff value for fixing internal dofs is appropriate. Current value: {gpr_fix_internal_dofs_cutoff}")

        if np.min(train_inputs_change) < 1e-5 and (not gpr_fix_internal_dofs_bool):
            print(f"the minimum change of internal dofs inputs: {np.min(train_inputs_change)}")
            raise(RuntimeError, "Certain internal dofs of input data is fixed.\
                   Should turn gpr_fix_internal_dofs_bool on.")
    

        if gpr_fix_internal_dofs_bool:
            if gpr_fixed_internal_dofs is None:
                self.fixed_internal_dofs = np.array(
                    [
                        i
                        for i in range(self.input_dim)
                        if train_inputs_change[i] < gpr_fix_internal_dofs_cutoff
                    ]
                )
            else:
                self.fixed_internal_dofs = gpr_fixed_internal_dofs
                print("@gpr_hessian_model: load fixed internal dofs.")
            
            if gpr_rigid_internal_dofs is None:
                self.rigid_internal_dofs = np.array(
                    [
                        i for i in range(self.input_dim)
                        if (train_inputs_change[i] > self.fix_internal_dofs_cutoff) and
                        (train_inputs_change[i] < rigid_internal_dofs_cutoff)
                    ]
                ).astype(int)
            else:
                self.rigid_internal_dofs = gpr_rigid_internal_dofs
                print("@gpr_hessian_model: load rigid internal dofs")

            self.fixed_internal_dofs = np.array([i for i in self.fixed_internal_dofs if i not in self.rigid_internal_dofs]).astype(int)

            print(f"@gpr_hessian_model: For Fixing internal dofs: fixed_internal_dofs: {self.fixed_internal_dofs}")
            print(f"@gpr_hessian_model: rigid internal dofs {self.rigid_internal_dofs}")
        else:
            self.fixed_internal_dofs = np.array(
                []
            ).astype(int)

            self.rigid_internal_dofs = np.array(
                []
            ).astype(int)

        self.constrained_internal_dofs = np.concatenate(
            [self.fixed_internal_dofs,
             self.rigid_internal_dofs]
        ).astype(int)


        if len(self.constrained_internal_dofs) != 0:
            self.free_moving_dofs = np.delete(
                np.arange(self.input_dim), 
                self.constrained_internal_dofs
            )
            print(f"@gpr_hessian_model: free moving dofs {self.free_moving_dofs}")
            self.free_moving_dofs_2d_index = np.meshgrid(
                self.free_moving_dofs, 
                self.free_moving_dofs, 
                indexing="ij"
            )

            self.constrained_internal_dofs_2d_index = np.meshgrid(
                self.constrained_internal_dofs, 
                self.constrained_internal_dofs,
                indexing= 'ij'
            )

            self.cross_term_2d_index = np.meshgrid(
                self.constrained_internal_dofs,
                self.free_moving_dofs,
                indexing= 'ij'
            )

            # use linear regression to fit gradients. 
            self.grads_for_fixed_dofs = np.mean(grads, axis=0)[self.fixed_internal_dofs]
            
            # linear regression fit for gradient in rigid dof.
            if len(self.rigid_internal_dofs) != 0:
                self.grad_reg_model = self.linear_regression_fit_grad(
                    train_inputs,
                    grads,
                    force_ridge_regularization_alpha
                    )

            # use linear regression to fit hessians 
            if len(hessians) != 0:
                self.constrained_part_hessian_reg_model, self.cross_term_reg_model = self.linear_regression_fit_hessian(
                    train_inputs[hessian_data_point_index_array],
                    hessians,
                    hessian_ridge_regularization_alpha= hessian_ridge_regularization_alpha
                    )
            else:
                self.constrained_part_hessian_reg_model = None
                self.cross_term_reg_model = None  

            pass 

        else:
            self.free_moving_dofs = np.arange(self.input_dim)
            self.free_moving_dofs_2d_index = np.meshgrid(
                self.free_moving_dofs, self.free_moving_dofs, indexing="ij"
            )
            self.grads_for_fixed_dofs = np.array([])


    def linear_regression_fit_grad(
            self,
            train_inputs: np.ndarray,
            grads: np.ndarray,
            force_ridge_regularization_alpha
            ):
        """
        fit the gradient along the rigid internal dofs using linear regression model.
        """
        x = train_inputs
        y = grads[:, self.rigid_internal_dofs]
        # reg_model = LinearRegression().fit(x,y)

        # ridge regression.
        if force_ridge_regularization_alpha > 0:
            reg_model = lin_model(degree= 1, regularization= True, lambda_= force_ridge_regularization_alpha)
        else:
            reg_model = lin_model(degree=1, regularization= False)
    
        reg_model.fit(x, y)

        return reg_model

    def predict_rigid_grad(
            self,
            predict_inputs: np.ndarray
    ):
        """
        predict the gradient along the rigid internal dofs using Linear regression model.
        """
        predicted_grad = self.grad_reg_model.predict(predict_inputs)

        return predicted_grad 

    def linear_regression_fit_hessian(
            self,
            train_inputs: np.ndarray,
            hessians: np.ndarray,
            hessian_ridge_regularization_alpha = 0.1
    ):
        """
        fit the hessian along constrained dofs using linear regression model.
        :param: train_inputs: training inputs in internal coordinate.
        :param: hessians: hessians for training data.
                          The components correspond to constrained internal dofs will be fitted using linear regression model.
                          shape: [data_num, ndofs, ndofs]
        """
        constrained_dofs = self.constrained_internal_dofs
        num_constrained_dofs = len(constrained_dofs)
        data_num = len(hessians)
        
        constrained_dofs_hessians = hessians[:, self.constrained_internal_dofs_2d_index[0], self.constrained_internal_dofs_2d_index[1]]
        # to use scikit_learn Linear regression fit.
        # x shape: [n_samples, n_features]
        # y shape: [n_samples, n_targets]
        # we need to flatten hessians into 1d array [n_targets] for each sample
        y = constrained_dofs_hessians.reshape((data_num, -1))
        x = train_inputs
        # constrained_dofs_reg_model = Ridge(alpha= hessian_ridge_regularization_alpha).fit(x,y)
        constrained_dofs_reg_model = lin_model(1, regularization= True, lambda_ = hessian_ridge_regularization_alpha)
        constrained_dofs_reg_model.fit(x,y)

        constrained_free_moving_cross_term_hessians = hessians[:, 
                                                               self.cross_term_2d_index[0], 
                                                               self.cross_term_2d_index[1]]
                                                               
        y1 = constrained_free_moving_cross_term_hessians.reshape(data_num, -1)
        x1 = train_inputs 
        # cross_term_reg_model = Ridge(alpha= hessian_ridge_regularization_alpha).fit(x1, y1)
        cross_term_reg_model = lin_model(1, regularization= True, lambda_= hessian_ridge_regularization_alpha)
        cross_term_reg_model.fit(x1, y1)

        return constrained_dofs_reg_model, cross_term_reg_model

    def predict_constrained_hessian(
            self,
            predict_inputs: np.ndarray,
            hessians: np.ndarray
    ):
        """
        predict hessians along constrained dofs using Linear regression model.

        :param: predict_inputs: input data for linear regression model to predict hessians.
        :param: hessians: hessian data. The data along constrained dofs will be predicted by linear regression model.
        """
        data_num = predict_inputs.shape[0]
        num_constrained_dofs = len(self.constrained_internal_dofs)
        num_free_moving_dofs = len(self.free_moving_dofs)

        # predict block diagonal component for constrained dofs.
        predicted_constrained_hessians = self.constrained_part_hessian_reg_model.predict(predict_inputs)
        predicted_constrained_hessians = predicted_constrained_hessians.reshape((data_num, num_constrained_dofs, num_constrained_dofs))
        hessians[:, self.constrained_internal_dofs_2d_index[0], self.constrained_internal_dofs_2d_index[1]] = predicted_constrained_hessians

        # predict cross term between constrained dofs and free moving dofs
        predicted_cross_term_hessians = self.cross_term_reg_model.predict(predict_inputs)
        predicted_cross_term_hessians = predicted_cross_term_hessians.reshape((data_num, num_constrained_dofs, num_free_moving_dofs))

        hessians[:, self.cross_term_2d_index[0], self.cross_term_2d_index[1]] = predicted_cross_term_hessians
        hessians[:, self.cross_term_2d_index[1], self.cross_term_2d_index[0]] = predicted_cross_term_hessians

        return hessians

    def update_hessian_reg_model(self,
                                train_inputs,
                                hessian_data_point_index, 
                                hessians,
                                alpha):
        if (
            len(self.constrained_internal_dofs) != 0
            and len(hessians) > 0
        ):
            self.constrained_part_hessian_reg_model, self.cross_term_reg_model = self.linear_regression_fit_hessian(
                train_inputs[hessian_data_point_index],
                hessians,
                hessian_ridge_regularization_alpha= alpha 
            )
            

    def transform_training_inputs_to_free_moving_dofs(self, train_inputs: np.ndarray):
        """
        delete fixdofs from training inputs.
        :param: train_inputs: the training inputs in internal dofs.
        """
        moving_train_inputs = train_inputs[:, self.free_moving_dofs]
        return moving_train_inputs

    def transform_training_targets_to_free_moving_dofs(
        self, 
        grads: np.ndarray, 
        hessians: np.ndarray
    ):
        """
        delete fixdofs data from training gradients and hessians.
        """
        moving_grads = grads[:, self.free_moving_dofs]
        if len(hessians) > 0:
            index_2d = self.free_moving_dofs_2d_index
            moving_hessians = hessians[:, index_2d[0], index_2d[1]]
        else:
            moving_hessians = hessians

        return moving_grads, moving_hessians

    def transform_training_hessians_to_free_moving_dofs(
            self, 
            hessians: np.ndarray
    ):
        if len(hessians) > 0:
            index_2d = self.free_moving_dofs_2d_index
            moving_hessians = hessians[:, index_2d[0], index_2d[1]]
        else:
            moving_hessians = hessians 
        
        return moving_hessians

    def transform_noise_covar_factor_fixing_internal_dofs(
        self, noise_covar_factor, with_hessian_bool=False
    ):
        """
        delete rows of noise covariate factor matrix corresponding to gradient and hessians in internal coordinate of fixed dofs.
        """
        input_dim = self.input_dim
        hessian_triu_size = int((input_dim + 1) * input_dim / 2)
        if len(self.constrained_internal_dofs) != 0:
            row_to_delete_grad = 1 + np.array(
                self.constrained_internal_dofs
            )  # the row in noise_covar_factor matrix corresponds to gradient.

            if not with_hessian_bool:
                row_to_delete = row_to_delete_grad
                noise_covar_factor = np.delete(
                    noise_covar_factor, row_to_delete, axis=0
                )
                return noise_covar_factor
            else:
                # find the upper triangle index of hessians that we need to delete
                upper_triangle_index_matrix = np.zeros([input_dim, input_dim])
                for i in range(input_dim):
                    for j in range(i, input_dim):
                        upper_triangle_index_matrix[i, j] = (
                            i * (input_dim - (1 + i) / 2) + j
                        )
                        upper_triangle_index_matrix[j, i] = upper_triangle_index_matrix[
                            i, j
                        ]

                upper_triangle_index_matrix_free_moving = upper_triangle_index_matrix[
                    self.free_moving_dofs_2d_index[0], self.free_moving_dofs_2d_index[1]
                ]
                upper_triangle_index_matrix_free_moving = take_upper_triangular_part(
                    upper_triangle_index_matrix_free_moving
                )
                upper_triangle_index_matrix_free_moving = np.vectorize(int)(
                    upper_triangle_index_matrix_free_moving
                )
                fixed_hessian_triu_index = np.delete(
                    np.arange(hessian_triu_size),
                    upper_triangle_index_matrix_free_moving,
                )
                row_to_delete_hessian_triu_index = (
                    fixed_hessian_triu_index + 1 + input_dim
                )

                # delete rows that corresponds to gradient and hessian of fixed dofs.
                row_to_delete = np.concatenate(
                    [row_to_delete_grad, row_to_delete_hessian_triu_index]
                )
                noise_covar_factor = np.delete(
                    noise_covar_factor, row_to_delete, axis=0
                )
                return noise_covar_factor
        else:
            return noise_covar_factor

    def transform_noise_covar_factor_array_fixing_internal_dofs(
        self, noise_covar_factor_pot_grad_array, noise_covar_factor_with_hessian_array
    ):
        """
        delete rows of noise covar factor for different data points. (corresponds to fixed internal dofs)
        the noise covar factor matrix will transform the noise from Cartesian coordinate into internal coordinate.
        :param: noise_covar_factor_pot_grad_array: An array of noise covar factor matrix (transform gradients) for all data points.
        :param: noise_covar_factor_with_hessian_array: An array of noise covar factor matrix (transform gradients + hessians) for data points with hessian.
        """
        pot_grad_array_len = len(noise_covar_factor_pot_grad_array)
        hessian_len = len(noise_covar_factor_with_hessian_array)

        noise_covar_factor_pot_grad_array_new = []
        for i in range(pot_grad_array_len):
            new_covar_factor = self.transform_noise_covar_factor_fixing_internal_dofs(
                noise_covar_factor_pot_grad_array[i], with_hessian_bool=False
            )
            noise_covar_factor_pot_grad_array_new.append(new_covar_factor)
        noise_covar_factor_pot_grad_array_new = np.array(
            noise_covar_factor_pot_grad_array_new
        )

        noise_covar_factor_with_hessian_array_new = []
        for i in range(hessian_len):
            new_covar_factor = self.transform_noise_covar_factor_fixing_internal_dofs(
                noise_covar_factor_with_hessian_array[i], with_hessian_bool=True
            )
            noise_covar_factor_with_hessian_array_new.append(new_covar_factor)
        noise_covar_factor_with_hessian_array_new = np.array(
            noise_covar_factor_with_hessian_array_new
        )

        return (
            noise_covar_factor_pot_grad_array_new,
            noise_covar_factor_with_hessian_array_new,
        )

    def transform_from_free_moving_dofs_to_full_dofs(
        self,
        test_inputs,
        test_hessian_data_point_index, 
        test_moving_grads, 
        test_moving_hessians,
        zero_bool= False
    ):
        """
        Transform the prediction of the GPR model from free moving dofs into the full dofs.
        :param: test_inputs: the input in internal dofs.
        :param: test_hessian_data_point_index: the indexes for data points that have hessian data.
        :param: test_moving_grads: gradient along free moving dofs 
        :param: test_moving_hessians; hessians along free moving dofs.
        :param: zero_bool: whether to set gradient and hessian var along the fixed & rigid dof to 0.
        """
        test_data_num = test_moving_grads.shape[0]

        # the graidents in fixed dofs in testing data is the mean value of fixed dofs for gradients in training data
        test_grads_fixed_dofs = np.repeat(
            [self.grads_for_fixed_dofs], 
            test_data_num, 
            axis=0
        )

        # the prediction of the gradient data in all dofs
        test_grads = np.zeros([test_data_num, self.input_dim])
        test_grads[:, self.free_moving_dofs] = test_moving_grads
        if len(self.fixed_internal_dofs) != 0:
            # the grad along the fixed dof is the average value.
            test_grads[:, self.fixed_internal_dofs] = test_grads_fixed_dofs
        
        if len(self.rigid_internal_dofs) != 0:
            # the gradients in rigid internal dofs:
            test_grads_rigid_dofs = self.predict_rigid_grad(
                test_inputs
            )
            # the grad along the rigid dof is the linear regression fit value.
            test_grads[:, self.rigid_internal_dofs] = test_grads_rigid_dofs

        if len(test_moving_hessians) > 0:
            if len(self.constrained_internal_dofs) != 0:
                test_data_with_hessian_num = test_moving_hessians.shape[0]
                test_hessians = np.zeros(
                        [test_data_with_hessian_num,
                         self.input_dim,
                         self.input_dim]
                    )
                if zero_bool or self.constrained_part_hessian_reg_model is None:
                    pass 
                else:
                    # the prediction of hessian data in all dofs
                    # We use linear regression to predict hessians in constrained dofs.
                    test_inputs_with_hessian = test_inputs[test_hessian_data_point_index]
                    test_hessians = self.predict_constrained_hessian(
                        test_inputs_with_hessian,
                        test_hessians
                    )

                index_2d = self.free_moving_dofs_2d_index
                test_hessians[:, index_2d[0], index_2d[1]] = test_moving_hessians
            else:
                test_hessians = test_moving_hessians

        else:
            test_hessians = torch.Tensor([])

        return test_grads, test_hessians

class GPModelWithHessiansWrapper:
    """
    wrapper class for GPModelWithHessians.
    This class handles the transformation between internal coordinate and Cartesian coordinate + GPR training.
    This code will use training data with potentials, forces and hessians, then predict Hessians.
    I wrote addition codes (See utils/gprHessian) to extend the Gpytorch packages (https://docs.gpytorch.ai/en/stable/),
    so we can predict Hessians using Gaussian Process Regression model.
    See J. Chem. Theory Comput. 2024, 20, 9, 3766-3778 and J. Chem. Theory Comput. 2020, 16, 8, 5083-5089 for implementation.
    We put potential, gradients and hessian data into 1d array & use Gaussian Process Regression to predict hessians.
    """

    def __init__(
        self,
        train_x: np.ndarray,
        train_V: np.ndarray,
        train_grad_x: np.ndarray,
        train_hessian_x: np.ndarray,
        training_data_hessian_data_point_index_array: np.ndarray,
        natom: int,
        coordinate_transformer: non_redundant_coordinate_transformer,
        cartesian_fix_dofs: np.ndarray,
        gpr_SE_kernel_number: int,
        kernel_outputscale: np.ndarray,
        kernel_lengthscale_ratio: np.ndarray,
        noise_std,
        kernel_lengthscale_initio_value: np.ndarray = np.array([]),
        kernel_outputscale_initio_value: np.ndarray = np.array([]),
        constant_mean_func_bool=True,
        ref_mean_x: np.ndarray = np.array([]),
        ref_mean_V: np.ndarray = np.array([]),
        ref_mean_grad_x: np.ndarray = np.array([]),
        ref_mean_hessian_x: np.ndarray = np.array([]),
        train_bool= True,
        gpr_fix_internal_dofs_bool= False,
        gpr_fix_internal_dofs_cutoff= 1e-4,
        gpr_rigid_internal_dofs_cutoff= 5e-2,
        gpr_fixed_internal_dofs= None,
        gpr_rigid_internal_dofs= None,
        ridge_regularization_alpha= {
                "force": 0.1,
                "hessian": 0.1,
            },
        singular_value_cutoff = 1e-8
    ):
        """
        :param: train_x: [M, 3 * natom]. initial M training points x in Cartesian coordinate.
        :param: train_V: [M]. initial M training potentials V.
        :param: train_grad_x: [M, 3 * natom]. initial M training gradients in Cartesian coordinate.
        :param: train_hessians_x: [M_H, 3 * natom, 3 * natom].  hessians of initial M_H training data.
        :param: training_data_hessian_data_point_index: index of M_H data points that have hessians.
        :param: natom: number of atoms.
        :param: coordinate_transformer: an instance of class: non_redundnat_coordinate_transformer. 
                                        Responsible for transformation between Cartesian coordinates and internal coordinates.
        :param: cartesian_fix_dofs: cartesian dofs that will keep as fixed. 
                                    The internal coordinates correspond to these cartesian dofs will not be included 
                                    in the GPR model.
        :param: gpr_SE_kernel_number: number of squared exponential kernels that is used to construct the covariance function.
        :param: kernel_outputscale: output scale of each squared exponential kernels in Gaussian Process Regression model.
        :param: kernel_lengthscale_ratio: the ratio of length scale over the range of input data for each squared exponential kernels 
                                          in Gaussian Process Regression model.
        :param: noise_std: the noise of likelihood function p(y|f).  y = f + epsilon.  
                            Note the potential V, force f and hessian H have different noise.
                            The noise for force and hessian is defined in Cartesian coordinate. 
                            We need to transform it into the internal coordinate.
        :param: kernel_lengthscale_initio_value: If set, we will initialize the length scale of gpr kernel as this value.
        :param: kernel_outputscale_initio_value: If set, we will initialize the output scale of kernel as this value.
        :param: constant_mean_func_bool: If true, we will set the mean function of GPR model as function with constant value 
                                            & zero gradient / hessians. 
                                            Otherwise, it will be Taylor expansion around ref point to second order.
        :param: ref_mean_x, ref_mean_V, ref_mean_grad_x, ref_mean_hessian_x:  
        this is the coordinate / V / gradient / hessians of reference point which be used to set mean function of GPR model.
        
        :param: singular_value_cutoff: singular value cutoff for pesudo-inverse of covariance matrix.
        
        Several pre-processing steps:
        (1) transform data from Cartesian coordinate into internal coordinate.
        (2) compute noise in the internal coordinate.
        (3) normalize the data (rescale potential and inputs, then scale gradient and hessian correspondingly).
        (4) fix certain dofs that is not moving. only put free moving dofs into GPR modeling.
        (5) transform potential, gradient and hessian into 1d data array.
        """
        cuda_available = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda_available else 'cpu')
        print("GPytorch for force & hessian prediction.")
        if cuda_available:
            print("CUDA is available. GPU is enabled.")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs available: {torch.cuda.device_count()}")
            print(f"Current GPU device: {torch.cuda.current_device()}")
            print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            print("CUDA is not available. Running Gpytorch on CPU.")

        M_H = len(training_data_hessian_data_point_index_array)
        hessian_fixdofs = np.array([])
        assert (
            np.shape(train_x)[1] == 3 * natom
        ), "dim of coordinates for input data is not 3 * natom, this is wrong. train_x data shape: {} , 3 * natom: {}".format(
            np.reshape(train_x)[1], 3 * natom
        )
        assert (
            np.shape(train_grad_x)[1] == 3 * natom
        ), "dim of gradients for input data is not 3 * natom, this is wrong. train_grad shape:{}, 3 * natom: {}".format(
            np.shape(train_grad_x)[1], 3 * natom
        )
        assert (
            np.shape(train_hessian_x)[0] == M_H
        ), "number of data points (M_H) with hessian information is not consistent with training_data_hessian_data_point_index. M_H from train_hessians: {}, M_H from hessian_data_point_index: {}".format(
            np.shape(train_hessian_x)[0], M_H
        )

        self.natom = natom
        self.gpr_SE_kernel_number = gpr_SE_kernel_number
        self.coordinate_transformer = coordinate_transformer
        self.constant_mean_func_bool = constant_mean_func_bool
        self.force_ridge_regularization_alpha = ridge_regularization_alpha["force"]
        self.hessian_ridge_regularization_alpha = ridge_regularization_alpha["hessian"]

        # symmetrize the hessian
        if len(train_hessian_x) > 0:
            train_hessian_x_symmetrized = (
                np.transpose(train_hessian_x, (0, 2, 1)) + train_hessian_x
            ) / 2
        else:
            train_hessian_x_symmetrized = train_hessian_x

        # record the potential, gradient and hessians in Cartesian coordinate.
        self.train_V = np.copy(train_V)
        self.train_cartesian_gradient = np.copy(train_grad_x)
        self.train_cartesian_hessian = np.copy(train_hessian_x_symmetrized)
        self.training_data_hessian_data_point_index = np.copy(
            training_data_hessian_data_point_index_array
        )
        self.train_cartesian_input = np.copy(train_x)

        # ------ transform input, gradient, hessian from Cartesian coordinate into internal coordinate. ----- 
        train_inputs, train_grad_q, train_hessian_q = self.transform_data_into_internal_coordinate(
            train_x,
            train_grad_x,
            train_hessian_x_symmetrized,
            training_data_hessian_data_point_index_array
        ) 
        
        self.input_dim = np.shape(train_inputs)[1]  # the number of internal dofs.

        # record the training inputs and target in internal coordinate space.
        self.train_inputs = train_inputs
        self.train_grad_q = train_grad_q
        self.train_hessian_q = train_hessian_q

        # ---- compute noise in internal coordinate -----
        noise_covar_factor_pot_grad_array, noise_covar_factor_with_hessian_array,\
                                pot_noise_var, grad_noise_var, hessian_noise_var \
            = self.compute_noise_matrix(
                                        train_x,
                                        training_data_hessian_data_point_index_array,
                                        noise_std
                                        )

        # rank of force noise 1d array and hessian noise 1d array (upper triangle part of hessian.)
        force_noise_rank = 3 * natom
        hessian_noise_rank = int((3 * natom) * (3 * natom + 1) / 2)

        # Filter the fixed dofs in coordinate (q) and gradients & hessians.
        self.FixingDofs = FixInternalDofs(
            train_x,
            train_inputs,
            training_data_hessian_data_point_index_array,
            cartesian_fix_dofs,
            self.coordinate_transformer,  
            train_grad_q, 
            train_hessian_q,
            gpr_fix_internal_dofs_bool,
            gpr_fix_internal_dofs_cutoff,
            gpr_rigid_internal_dofs_cutoff,
            gpr_fixed_internal_dofs,
            gpr_rigid_internal_dofs,
            self.force_ridge_regularization_alpha,
            self.hessian_ridge_regularization_alpha 
        )

        moving_train_inputs, moving_train_grad_q, moving_train_hessian_q, \
            noise_covar_factor_pot_grad_array, noise_covar_factor_with_hessian_array = \
            self.filter_fixed_dof_from_data(
                train_inputs,
                train_grad_q,
                train_hessian_q,
                noise_covar_factor_pot_grad_array,
                noise_covar_factor_with_hessian_array
            )
        
        # normalize the potential, gradient and hessians.
        self.Normalizer = NormalizeTrainingData(
            train_V,
            moving_train_inputs
        )

        # -- normalize the input, potential, gradient and hessian. & noise of gradient & hessian -------
        (moving_normalized_train_inputs, normalized_train_V, moving_normalized_train_grad_q, moving_normalized_train_hessian_q, \
            noise_covar_factor_pot_grad_array, noise_covar_factor_with_hessian_array, \
            pot_noise_var, grad_noise_var, hessian_noise_var) = \
                self.normalize_data(
                    moving_train_inputs,
                    train_V,
                    moving_train_grad_q,
                    moving_train_hessian_q,
                    noise_covar_factor_pot_grad_array,
                    noise_covar_factor_with_hessian_array,
                    pot_noise_var,
                    grad_noise_var,
                    hessian_noise_var
                )

        # transform pots, gradients and hessisans in to 1d data.
        # After we have normalized the training data and excluded fixed dof in gradient and hessian data.
        free_moving_input_dims = len(self.FixingDofs.free_moving_dofs)

        self.TargetDataTransformer = TransformTrainingTarget(
            free_moving_input_dims, hessian_fixdofs
        )
        
        # transform potential, gradient and hessian from Cartesian coordinate into internal coordinate.
        train_targets = (
            self.TargetDataTransformer.transform_pots_grad_hessian_to_1d_data(
                normalized_train_V,
                moving_normalized_train_grad_q,
                moving_normalized_train_hessian_q,
            )
        )

        # Transform the numpy array to torch.Tensor. The Gpytorch need to deal with troch.Tensor instead of numpy.ndarray.
        hessian_fixdofs_tensor = torch.tensor([], device= self.device)

        (moving_normalized_train_inputs_tensor, train_targets_tensor, hessian_data_point_index_tensor,
         noise_covar_factor_pot_grad_array, noise_covar_factor_with_hessian_array) = map(
             lambda x: torch.from_numpy(x).to(device= self.device), 
             (
                 moving_normalized_train_inputs,
                 train_targets,
                 training_data_hessian_data_point_index_array,
                 noise_covar_factor_pot_grad_array,
                 noise_covar_factor_with_hessian_array
             )
         )

        # transform the gradients & hessians of reference point from Cartesian coordinate into internal coordinate.
        # The mean function m(x) of GPR model will be set as Taylor expansion around the potential of the reference point: V(x) = V(x0) + V'(x0) (x-x0) + 1/2 * V''(x0) (x-x0)^2
        (
            ref_mean_q_tensor,
            ref_mean_V_tensor,
            ref_mean_grad_q_tensor,
            ref_mean_hessian_q_tensor,
            ref_mean_hessian_q_upper_triag_tensor,
        ) = self.compute_mean_function_param(
            ref_mean_x, ref_mean_V, ref_mean_grad_x, ref_mean_hessian_x
        )

        # initialize the gaussian process regression model with input training data.
        # GPModelWithHessians are Gaussian Process Regression model that capable of using hessian as training data and also predicting hessians.
        # It transforms the potential, force & hessian into 1d data set. See eq.(4 - 9) in J. Chem. Theory Comput. 2024, 20, 3766−3778 for the set up.
        self.gpr_model = GPModelWithHessians(
            moving_normalized_train_inputs_tensor,
            train_targets_tensor,
            hessian_data_point_index_tensor,
            hessian_fixdofs_tensor,
            gpr_SE_kernel_number,
            kernel_outputscale,
            kernel_lengthscale_ratio,
            pot_noise_var,
            grad_noise_var,
            hessian_noise_var,
            force_noise_rank,
            hessian_noise_rank,
            noise_covar_factor_pot_grad_array,
            noise_covar_factor_with_hessian_array,
            kernel_lengthscale_initio_value,
            kernel_outputscale_initio_value,
            constant_mean_func_bool,
            ref_mean_q_tensor,
            ref_mean_V_tensor,
            ref_mean_grad_q_tensor,
            ref_mean_hessian_q_upper_triag_tensor,
            nugget= singular_value_cutoff
        )

        self.gpr_model = self.gpr_model.to(device= self.device)

        if train_bool:
            # train the gaussian process regression model.
            gpr.gprHessian.RBFHessian_gp.train_gpr_model(self.gpr_model)
        else:
            # print the condition number of the covariance matrix.
            gpr.gprHessian.RBFHessian_gp.check_cond_number(self.gpr_model)

    def transform_data_into_internal_coordinate(self, 
                                                train_x, 
                                                train_grad_x, 
                                                train_hessian_x_symmetrized, 
                                                training_data_hessian_data_point_index_array):
        """
        transform coordinate, gradients and hessians from Cartesian coordinate into the internal coordinate.
        """
        # transform the cartesian coordinate x to internal coordinate q
        train_inputs = self.coordinate_transformer.get_internal_coordinate_q(train_x)

        # transform the gradient of potential V into internal coordinate: dV/dx -> dV/dq
        train_grad_q = (
            self.coordinate_transformer.transform_cartesian_gradient_to_internal_gradient(
                train_x, train_grad_x
            )
        )
        # transform the hessian of potential V: d^2 V/ dx^2 -> d^2 V/ dq^2
        if len(training_data_hessian_data_point_index_array) > 0:
            train_hessian_q = (
                self.coordinate_transformer.transform_cartesian_hessian_to_internal_hessian(
                    train_x[training_data_hessian_data_point_index_array],
                    train_grad_x[training_data_hessian_data_point_index_array],
                    train_hessian_x_symmetrized,
                )
            )
        else:
            train_hessian_q = np.array([])
        
        return train_inputs, train_grad_q, train_hessian_q 

    def compute_noise_var(self, noise_std):
        """
        compute the variance of the noise.
        """
        pot_noise_std = noise_std["pot_noise_prior"]
        force_noise_std_cartesian = noise_std["force_noise_prior"]
        hessian_noise_std_cartesian = noise_std["hessian_noise_prior"]

        # variance of pot noise, force noise and hessian noise in Cartesian coordinate.
        pot_noise_var = np.array([np.power(pot_noise_std, 2)])
        force_noise_var = np.ones([1]) * np.power(force_noise_std_cartesian, 2)
        hessian_noise_var = np.ones([1]) * np.power(hessian_noise_std_cartesian, 2)

        self.Bmatrix_singular_value_square = np.power(
            self.coordinate_transformer.ref_S, 2
        )

        return pot_noise_var, force_noise_var, hessian_noise_var

    def compute_noise_covar_factor_for_each_data_point(self, x, with_hessian_bool):
        """
        compute the covariance factor for noise transformation for each data point x.
        below, we assume 0 represents blocks of potentials, 1 represents blocks of gradients and 2 represents blocks of hessians.
        """
        # covar_factor [1, 1] term.  inverse transpose of Wilson's B matrix
        Bq = self.coordinate_transformer.compute_delocalized_wilson_matrix_Bq(
            np.array([x])
        )[
            0
        ]

        # \partial x / \partial q.
        inverse_Bq_transpose = np.transpose(
            np.linalg.pinv(Bq), (1, 0)
        )

        q_size = Bq.shape[0]
        x_size = Bq.shape[1]
        hessian_q_triu_size = int((q_size * (q_size + 1)) / 2)
        hessian_x_triu_size = int((x_size * (x_size + 1)) / 2)

        # covar factor [2,1] d^2 x/ dq^2
        hessian_x_qq = self.coordinate_transformer.compute_x_hessian_q(
            np.array([x])
        )[0]  # d^2 x / dq^2. shape:[3n, 3n-6, 3n-6]
        
        hessian_x_qq_up_triangle = np.transpose(
            take_upper_triangular_part(hessian_x_qq), (1, 0)
        )  # shape: [(3n-6)(3n-5) / 2, 3n]

        # covar factor [2,2] term. d^2 x/ dq^2
        inverse_Bq_transpose_tensor = np.transpose(
            np.tensordot(inverse_Bq_transpose, inverse_Bq_transpose, axes=0),
            (0, 2, 1, 3),
        )
        inverse_Bq_transpose_tensor_diag = np.zeros(inverse_Bq_transpose_tensor.shape)
        inverse_Bq_transpose_tensor_diag[..., np.arange(x_size), np.arange(x_size)] = (
            np.diagonal(inverse_Bq_transpose_tensor, axis1=2, axis2=3)
        )
        covar_33 = (
            inverse_Bq_transpose_tensor
            + np.transpose(inverse_Bq_transpose_tensor, (0, 1, 3, 2))
            - inverse_Bq_transpose_tensor_diag
        )
        # take upper triangular part
        covar_33 = take_upper_triangular_part(covar_33)
        covar_33 = np.transpose(
            take_upper_triangular_part(np.transpose(covar_33, (2, 0, 1))), (1, 0)
        )

        row_size = 1 + q_size + hessian_q_triu_size
        col_size = 1 + x_size + hessian_x_triu_size

        # transformation matrix for covariance matrix of noise in internal coordinate (q) and Cartesian coordinate (x)
        noise_covar_factor = np.zeros([row_size, col_size])
        # potential part. covar_factor [0,0]
        noise_covar_factor[0, 0] = 1
        # grad part. covar_factor [1,1]
        grad_index_2d = np.meshgrid(
            1 + np.arange(q_size), 1 + np.arange(x_size), indexing="ij"
        )
        noise_covar_factor[grad_index_2d[0], grad_index_2d[1]] = inverse_Bq_transpose

        # grad- hessian covariance part. covar_factor[1,2] & covar_factor[2,1]
        grad_hessian_covar_index_2d = np.meshgrid(
            1 + q_size + np.arange(hessian_q_triu_size),
            1 + np.arange(x_size),
            indexing="ij",
        )
        noise_covar_factor[
            grad_hessian_covar_index_2d[0], grad_hessian_covar_index_2d[1]
        ] = hessian_x_qq_up_triangle

        # hessian covariance part. covar_factor[2,2].
        hessian_covar_index_2d = np.meshgrid(
            1 + q_size + np.arange(hessian_q_triu_size),
            1 + x_size + np.arange(hessian_x_triu_size),
            indexing="ij",
        )
        noise_covar_factor[hessian_covar_index_2d[0], hessian_covar_index_2d[1]] = (
            covar_33
        )

        if with_hessian_bool:
            return noise_covar_factor
        else:
            noise_covar_factor = noise_covar_factor[: 1 + q_size, : 1 + x_size]
            return noise_covar_factor

    def compute_noise_covar_factor_array(
        self, train_x, training_data_hessian_data_point_index_array
    ):
        """
        compute covariate factor for different training inputs data.

        See J. Chem. Theory Comput. 2024, 20, 3766-3778  eq.(13). We need back transform the noise matrix into internal coordinate.
        The noise matrix transform like covariance matrix K, see eq.(17).
        To correctly perform Gaussian Process regression in internal coordinate,
        you either transform covariance matrix from internal coordinate into Cartesian coordinate.
        Or transform potential, force, Hessian and noise matrix into internal coordinate.
        Here we adopt the second option: transform pot/ force/ hessian & noise matrix into internal coordinate.
        For certain applications, where descriptor of data is very long,
        our current approach is not efficient, we need to transform covariance matrix from internal coordinate into Cartesian coordinate.
        """
        training_data_num = train_x.shape[0]

        noise_covar_factor_pot_grad_array = (
            []
        )  # covariance factor for only potential and gradient
        noise_covar_factor_with_hessian_array = (
            []
        )  # covariance factor including hessian

        for data_point_index in range(training_data_num):
            noise_covar_factor = self.compute_noise_covar_factor_for_each_data_point(
                train_x[data_point_index], with_hessian_bool=False
            )
            noise_covar_factor_pot_grad_array.append(noise_covar_factor)

        noise_covar_factor_pot_grad_array = np.array(noise_covar_factor_pot_grad_array)

        for hessian_data_point_index in training_data_hessian_data_point_index_array:
            noise_covar_factor = self.compute_noise_covar_factor_for_each_data_point(
                train_x[hessian_data_point_index], with_hessian_bool=True
            )
            noise_covar_factor_with_hessian_array.append(noise_covar_factor)

        noise_covar_factor_with_hessian_array = np.array(
            noise_covar_factor_with_hessian_array
        )

        return noise_covar_factor_pot_grad_array, noise_covar_factor_with_hessian_array

    def compute_noise_matrix(self,
                             train_x,
                             training_data_hessian_data_point_index_array,
                             noise_std):
        """
        compute the transformation matrix for noise matrix and 
        noise for potential ,gradient and hessian.
        """
        # compute noise_covar_factor matrix for each data point. This matrix will transform noise from Cartesian coordinate into internal coordinate.
        # See eq.(13) in J. Chem. Theory Comput. 2024, 20, 9, 3766-3778 for transformation matrix L. The noise covar factor here is inverse of L (L^{-1}) for each data point.
        noise_covar_factor_pot_grad_array, noise_covar_factor_with_hessian_array = (
            self.compute_noise_covar_factor_array(
                train_x, training_data_hessian_data_point_index_array
            )
        )
        # set variance of noise.
        pot_noise_var, grad_noise_var, hessian_noise_var = self.compute_noise_var(
            noise_std
        )

        return noise_covar_factor_pot_grad_array, noise_covar_factor_with_hessian_array,\
               pot_noise_var, grad_noise_var, hessian_noise_var

    def normalize_data(self,
                       train_inputs,
                       train_V,
                       train_grad_q,
                       train_hessian_q,
                       noise_covar_factor_pot_grad_array,
                       noise_covar_factor_with_hessian_array,
                       pot_noise_var= None,
                       grad_noise_var= None,
                       hessian_noise_var= None):
        """
        normalize the potential, gradient and hessian data.
        Also normalize the noise and  transformation matrix for noise.
        """
        normalized_train_V, normalized_train_grad_q, normalized_train_hessians_q, normalized_train_inputs = (
            self.Normalizer.normalization_transform(
                train_V, 
                train_grad_q, 
                train_hessian_q,
                train_inputs
            )
        )

        noise_covar_factor_pot_grad_array, noise_covar_factor_with_hessian_array = (
            self.Normalizer.normalize_noise_covar_factor_array(
                noise_covar_factor_pot_grad_array,
                noise_covar_factor_with_hessian_array
            )
        )

        if pot_noise_var != None:
            pot_noise_var, grad_noise_var, hessian_noise_var = (
                self.Normalizer.normalize_noise_var(
                    pot_noise_var, grad_noise_var, hessian_noise_var
                )
            )

            return normalized_train_inputs, normalized_train_V, normalized_train_grad_q, normalized_train_hessians_q, \
                    noise_covar_factor_pot_grad_array, noise_covar_factor_with_hessian_array, \
                    pot_noise_var, grad_noise_var, hessian_noise_var

        else:
            return normalized_train_inputs, normalized_train_V, normalized_train_grad_q, normalized_train_hessians_q, \
                    noise_covar_factor_pot_grad_array, noise_covar_factor_with_hessian_array

    def filter_fixed_dof_from_data(self,
                                   train_inputs,
                                   train_grad_q,
                                   train_hessians_q,
                                   noise_covar_factor_pot_grad_array,
                                   noise_covar_factor_with_hessian_array):
        """
        Filter certain dofs that is fixed from internal dofs.
        The free moving dofs will be used in the Gaussian Process Regression modeling.
        :param: train_inputs: training inputs in internal coordinate.
                train_grad_q: training gradients in internal coordinate.
                train_hessians_q: training hessians in internal coordinate.
                noise_covar_factor_pot_grad_array: covariate factor that transform the gradient noise from 
                                                   Cartesian coordinate into internal coordinate.
                noise_covar_factor_with_hessian_array: covariate factor that transform the gradient & hessian noise 
                                                   from Cartesian coordinate into internal coordinate.
        
        """
        moving_train_inputs = (
            self.FixingDofs.transform_training_inputs_to_free_moving_dofs(
                train_inputs
                )
        )

        moving_train_grad_q, moving_train_hessian_q = (
            self.FixingDofs.transform_training_targets_to_free_moving_dofs(
                train_grad_q, 
                train_hessians_q
            )
        )

        # filter the fixed dofs from noise_covar_factor_array.
        noise_covar_factor_pot_grad_array, noise_covar_factor_with_hessian_array = (
            self.FixingDofs.transform_noise_covar_factor_array_fixing_internal_dofs(
                noise_covar_factor_pot_grad_array, noise_covar_factor_with_hessian_array
            )
        )

        return moving_train_inputs, moving_train_grad_q, moving_train_hessian_q,\
               noise_covar_factor_pot_grad_array, noise_covar_factor_with_hessian_array  

    def compute_mean_function_param(
        self, ref_mean_x, ref_mean_V, ref_mean_grad_x, ref_mean_hessian_x
    ):
        """
        compute the internal coordinate q of reference point & pot V, grad and hessian of reference point.
        """
        # set the mean function as the pot, grad and hessian at reference point in internal coordinate.
        if len(ref_mean_x) != 0:
            self.ref_mean_x = ref_mean_x
            self.ref_mean_V = ref_mean_V
            self.ref_mean_grad_x = ref_mean_grad_x
            # symmetrize hessian:
            ref_mean_hessian_x = (
                ref_mean_hessian_x + np.transpose(ref_mean_hessian_x, (1, 0))
            ) / 2
            self.ref_mean_hessian_x = ref_mean_hessian_x

            # transform coordinate (x), gradient (g_x) and hessian (h_x) from Cartesian coordinate into Internal coordinate.
            self.ref_mean_q = self.coordinate_transformer.get_internal_coordinate_q(
                np.array([ref_mean_x])
            )[0]

            ref_mean_q = np.copy(self.ref_mean_q)

            ref_mean_grad_q = self.coordinate_transformer.transform_cartesian_gradient_to_internal_gradient(
                np.array([ref_mean_x]), np.array([ref_mean_grad_x])
            )[
                0
            ]

            ref_mean_hessian_q = self.coordinate_transformer.transform_cartesian_hessian_to_internal_hessian(
                np.array([ref_mean_x]),
                np.array([ref_mean_grad_x]),
                np.array([ref_mean_hessian_x]),
            )[
                0
            ]

            self.ref_mean_grad_q = ref_mean_grad_q
            self.ref_mean_hessian_q = ref_mean_hessian_q

            # Filter the fixed dofs.
            ref_mean_q = self.FixingDofs.transform_training_inputs_to_free_moving_dofs(
                np.array([ref_mean_q])
            )[0]

            ref_mean_grad_q, ref_mean_hessian_q = (
                self.FixingDofs.transform_training_targets_to_free_moving_dofs(
                    np.array([ref_mean_grad_q]), np.array([ref_mean_hessian_q])
                )
            )

            ref_mean_grad_q = ref_mean_grad_q[0]
            ref_mean_hessian_q = ref_mean_hessian_q[0]

            # Normalize the data.
            ref_mean_V, ref_mean_grad_q, ref_mean_hessian_q, ref_mean_q = (
                self.Normalizer.normalization_transform(
                    ref_mean_V, 
                    np.array([ref_mean_grad_q]), 
                    np.array([ref_mean_hessian_q]),
                    np.array([ref_mean_q])
                )
            )

            ref_mean_q = ref_mean_q[0]
            ref_mean_grad_q = ref_mean_grad_q[0]
            ref_mean_hessian_q = ref_mean_hessian_q[0]
            # take upper triangle part of hessian (this is what we put in GPR model)
            ref_mean_hessian_q_upper_triag = take_upper_triangular_part(ref_mean_hessian_q)

            ref_mean_q_tensor = torch.tensor(ref_mean_q, device= self.device)
            ref_mean_V_tensor = torch.tensor(ref_mean_V, device= self.device)
            ref_mean_grad_q_tensor = torch.tensor(ref_mean_grad_q, device= self.device)
            ref_mean_hessian_q_tensor = torch.tensor(ref_mean_hessian_q, device= self.device)
            ref_mean_hessian_q_upper_triag_tensor = torch.tensor(ref_mean_hessian_q_upper_triag, device= self.device)

        else:
            ref_mean_q_tensor = torch.tensor([], device= self.device)
            ref_mean_V_tensor = torch.tensor([], device= self.device)
            ref_mean_grad_q_tensor = torch.tensor([], device= self.device)
            ref_mean_hessian_q_tensor = torch.tensor([], device= self.device)
            ref_mean_hessian_q_upper_triag_tensor = torch.tensor([], device= self.device)

        return (
            ref_mean_q_tensor,
            ref_mean_V_tensor,
            ref_mean_grad_q_tensor,
            ref_mean_hessian_q_tensor,
            ref_mean_hessian_q_upper_triag_tensor,
        )

    def predict_latent_function(
        self,
        test_x: np.ndarray,
        test_hessian_data_point_index: np.ndarray,
        internal_coordinate_bool=False,
    ):
        """
        compute the predicted potential V, gradients dV/dx and hessians d^2 V/dx^2 in Cartesian coordinate.
        We also compute the uncertainty of gradients and hessian predictions.
        The uncertainty is defined as sum of variance of gradients and hessian matrix elements.

        This function wraps predict_latent_function_GPHessian in ./gprHessian/RBFHessian_gp.py.
        This function serves as a wrapper function.
        This function handles 1. normalization  2. filter fixed dofs  3. transformation between Cartesian and internal dofs.

        :param: test_x: input Cartesian coordinate data [N, 3 * natom].
        :param: test_hessian_data_point_index: the index of data point in x that we need to predict the hessian information.
        :param: internal_coordinate_bool:  if internal_coordinate_bool = True, we output gradient and hessian in internal coordinate (Used for debugging).
                                               otherwise (default), we output the gradient and hessian in cartesian coordinate.
        """
        assert (
            np.shape(test_x)[1] == 3 * self.natom
        ), "dim of coordinate for input data is not 3 * natom"

        # transform the input data into internal coordinate.
        test_q = self.coordinate_transformer.get_internal_coordinate_q(test_x)

        moving_normalized_test_q_tensor = torch.from_numpy(
            self.get_free_moving_internal_coordinate(
                test_x
                )
        ).to(device= self.device, dtype=torch.float64)
        
        test_hessian_data_point_index_tensor = torch.from_numpy(
            test_hessian_data_point_index
            ).to(device= self.device)
        
        # use Gaussian process regression model to make prediction
        (
            pots,
            moving_grads_q,
            moving_hessians_q,
            pots_var,
            moving_grads_q_var,
            moving_hessians_q_var,
        ) = gpr.gprHessian.RBFHessian_gp.predict_latent_function_GPHessian(
            self.gpr_model, moving_normalized_test_q_tensor, test_hessian_data_point_index_tensor
        )

        # inverse the normalization procedure for mean value and variance.
        pots, moving_grads_q, moving_hessians_q = self.Normalizer.inverse_normalization_transform(
            pots, 
            moving_grads_q, 
            moving_hessians_q
        )
        
        pots_var, moving_grads_q_var, moving_hessians_q_var = (
            self.Normalizer.inverse_normalize_noise_var(
                pots_var, 
                moving_grads_q_var, 
                moving_hessians_q_var
            )
        )

        # back transform the mean value and variance from free moving dofs into full dofs 
        grads_q, hessians_q = (
            self.FixingDofs.transform_from_free_moving_dofs_to_full_dofs(
                test_q,
                test_hessian_data_point_index,
                moving_grads_q, 
                moving_hessians_q
            )
        )

        grads_q_var, hessians_q_var = (
            self.FixingDofs.transform_from_free_moving_dofs_to_full_dofs(
                test_q,
                test_hessian_data_point_index,
                moving_grads_q_var, 
                moving_hessians_q_var, 
                zero_bool=True
            )
        )

        # transform the gradient and hessian from internal coordinate to Cartesian coordinate.
        grads_x = self.coordinate_transformer.transform_internal_gradient_to_cartesian_gradient(
            test_x, grads_q
        )
        if len(test_hessian_data_point_index) > 0:
            hessians_x = self.coordinate_transformer.transform_internal_hessian_to_cartesian_hessian(
                test_x[test_hessian_data_point_index],
                grads_q[test_hessian_data_point_index],
                hessians_q,
            )
        else:
            hessians_x = torch.tensor([])

        # handle the variance of the gradient & hessians.
        # the trace of covariance matirx of gradients. This characterize the uncertainty of the force.
        grads_x_var_trace = np.sum(
            self.Bmatrix_singular_value_square * grads_q_var, axis=1
        )

        # compute the sum of variance of each 2d hessian elements. This will be our uncertainty of hessians.
        if len(hessians_q_var) > 0:
            hessian_var_sum = np.sum(
                np.sum(
                    hessians_q_var
                    * np.outer(
                        self.Bmatrix_singular_value_square,
                        self.Bmatrix_singular_value_square,
                    ),
                    axis=-1,
                ),
                axis=-1,
            )
        else:
            hessian_var_sum = np.array([])

        if internal_coordinate_bool:
            return (
                pots,
                grads_q,
                hessians_q,
                pots_var,
                grads_x_var_trace,
                hessian_var_sum,
            )
        else:
            return (
                pots,
                grads_x,
                hessians_x,
                pots_var,
                grads_x_var_trace,
                hessian_var_sum,
            )


    def update_variable_in_model(
        self,
        old_training_data_num,
        new_hessian_data_point_index,
        new_train_x,
        new_train_V,
        new_train_grad_x,
        new_train_hessian_x_symmetrized,
        new_train_inputs,
        new_train_grad_q,
        new_train_hessian_q
    ):
        """
        update training_inputs, gradients, potential and hessians with the new data.
        """
        self.train_cartesian_input = np.concatenate(
            [self.train_cartesian_input, new_train_x], axis=0
        )
        self.train_V = np.concatenate([self.train_V, new_train_V])
        self.train_cartesian_gradient = np.concatenate(
            [self.train_cartesian_gradient, new_train_grad_x], axis=0
        )
        if len(self.train_cartesian_hessian) > 0:
            if len(new_train_hessian_x_symmetrized) > 0:
                self.train_cartesian_hessian = np.concatenate(
                    [self.train_cartesian_hessian, new_train_hessian_x_symmetrized],
                    axis=0,
                )
        else:
            self.train_cartesian_hessian = new_train_hessian_x_symmetrized

        # update the index for data point that contains hessian information.
        new_hessian_data_point_index_in_full_data_set = (
            new_hessian_data_point_index + old_training_data_num
        )  # the hessian index in full data set after concatnate new data
        self.training_data_hessian_data_point_index = np.concatenate(
            [
                self.training_data_hessian_data_point_index,
                new_hessian_data_point_index_in_full_data_set,
            ],
            axis=0,
        ).astype(int)

        # update coordinate, gradient & hessian data.
        self.train_inputs = np.concatenate(
            [self.train_inputs, new_train_inputs], axis=0
        )
        self.train_grad_q = np.concatenate(
            [self.train_grad_q, new_train_grad_q], axis=0
        )
        if len(self.train_hessian_q) > 0:
            if len(new_train_hessian_q) > 0:
                self.train_hessian_q = np.concatenate(
                    [self.train_hessian_q, new_train_hessian_q], axis=0
                )
        else:
            self.train_hessian_q = new_train_hessian_q

    def update_model_with_new_data(
        self,
        new_train_x: np.ndarray,
        new_train_V: np.ndarray,
        new_train_grad_x: np.ndarray,
        new_train_hessian_x: np.ndarray,
        new_hessian_data_point_index: np.ndarray,
        retrain_bool=True,
    ):
        """
        add new training data into the GPR model.
        Then train the model to update the hyper-parameter
        This function wrpas the function: update_model_with_new_data_GPHessian in ./gprHessian/RBFHessian_gp.py

        :param: new_train_x: [M, 3 * natom]. Cartesian coordinate of the input data
        :param: new_train_V: [M] ab-initio potential data.
        :param: new_train_grad_x: [M, 3 * natom]: ab initio gradient data.
        :param: new_train_hessian_x: [M_H, 3 * natom, 3 * natom]: ab initio hessian data. Note not all data points contain hessian information.
        :param: new_hessian_data_point_index: the index of data points that contain hessian information.
        """
        assert (
            np.shape(new_train_x)[1] == 3 * self.natom
        ), "dim of coordinates for input data is not 3 * natom"
        assert (
            np.shape(new_train_grad_x)[1] == 3 * self.natom
        ), "dim of gradients for input data is not 3 * natom"

        if len(new_train_hessian_x) > 0:
            assert (
                np.shape(new_train_hessian_x)[1] == 3 * self.natom
                and np.shape(new_train_hessian_x)[2] == 3 * self.natom
            ), "the shape of hessian for input data is not 3 * natom"

        # symmetrize the hessian.
        if len(new_train_hessian_x) > 0:
            new_train_hessian_x_symmetrized = (
                np.transpose(new_train_hessian_x, (0, 2, 1)) + new_train_hessian_x
            ) / 2
        else:
            new_train_hessian_x_symmetrized = np.array([])

        # --------  transform input data into internal coordinate  --------------------------        
        new_train_inputs, new_train_grad_q, new_train_hessian_q = self.transform_data_into_internal_coordinate(
            new_train_x,
            new_train_grad_x,
            new_train_hessian_x_symmetrized,
            new_hessian_data_point_index
        )

        # number of training data before adding data into model.
        old_training_data_num = np.shape(self.train_cartesian_input)[0]

        # ------ update the recorded training inputs and targets  ------ 
        self.update_variable_in_model(
            old_training_data_num,
            new_hessian_data_point_index,
            new_train_x,
            new_train_V,
            new_train_grad_x,
            new_train_hessian_x_symmetrized,
            new_train_inputs,
            new_train_grad_q,
            new_train_hessian_q
        )

        # compute noise_covar_factor array for new training data. This new noise covar factor matrix will be added into Gaussian Process Regression model when we optimize hyper-parameters
        (
            new_noise_covar_factor_pot_grad_array,
            new_noise_covar_factor_with_hessian_array,
        ) = self.compute_noise_covar_factor_array(
            new_train_x, 
            new_hessian_data_point_index
        )

        # --- Filter the fixed dofs for input, gradient, hessian and transformation matrix for noise -------- 
        new_train_inputs, new_train_grad_q, new_train_hessian_q, \
        new_noise_covar_factor_pot_grad_array, new_noise_covar_factor_with_hessian_array = \
        self.filter_fixed_dof_from_data(
            new_train_inputs,
            new_train_grad_q,
            new_train_hessian_q,
            new_noise_covar_factor_pot_grad_array,
            new_noise_covar_factor_with_hessian_array
        )

        # update the linear regression model for hessian prediction
        if len(self.train_hessian_q) > 0:
            self.FixingDofs.update_hessian_reg_model(
                self.train_inputs,
                self.training_data_hessian_data_point_index,
                self.train_hessian_q,
                self.hessian_ridge_regularization_alpha
            )

        # --- normalize the potential, gradient and hessians and the noise covar factor matrix ---- 
        new_train_inputs, new_train_V, new_train_grad_q, new_train_hessian_q, \
        new_noise_covar_factor_pot_grad_array, new_noise_covar_factor_with_hessian_array = \
            self.normalize_data(
                new_train_inputs,
                new_train_V,
                new_train_grad_q,
                new_train_hessian_q,
                new_noise_covar_factor_pot_grad_array,
                new_noise_covar_factor_with_hessian_array
            )

        # Transform the potential, gradient, hessians into 1d target data
        new_train_targets = (
            self.TargetDataTransformer.transform_pots_grad_hessian_to_1d_data(
                new_train_V, new_train_grad_q, new_train_hessian_q
            )
        )

        # transform the training inputs, training targets into tensor.Torch
        (new_train_inputs_tensor, new_train_targets_tensor,\
         new_hessian_data_point_index_tensor, \
         new_noise_covar_factor_pot_grad_array, \
         new_noise_covar_factor_with_hessian_array) = map(
             lambda x: torch.from_numpy(x).to(device= self.device),
             (
                 new_train_inputs,
                 new_train_targets,
                 new_hessian_data_point_index,
                 new_noise_covar_factor_pot_grad_array,
                 new_noise_covar_factor_with_hessian_array
             )
         )

        # update the Gaussian Process Regression model with new data.
        gpr.gprHessian.RBFHessian_gp.update_model_with_new_data_GPHessian(
            self.gpr_model,
            new_train_inputs_tensor,
            new_train_targets_tensor,
            new_hessian_data_point_index_tensor,
            new_noise_covar_factor_pot_grad_array,
            new_noise_covar_factor_with_hessian_array,
            retrain_bool=retrain_bool,
        )
    
    def train_model(self):
        """
        function that trains the model
        """
        train_gpr_model(self.gpr_model)

    def get_free_moving_internal_coordinate(self, beads_x):
        """
        transform from Cartesian coordinate x to the free moving internal coordinates q.
        """
        beads_internal_coordinate = (
            self.coordinate_transformer.get_internal_coordinate_q(beads_x)
        )

        free_moving_beads_internal_coordinate = (
            self.FixingDofs.transform_training_inputs_to_free_moving_dofs(
                beads_internal_coordinate
            )
        )

        normalized_beads_internal_coordinate = self.Normalizer.normalization_transform_for_inputs(
            free_moving_beads_internal_coordinate
        )


        return normalized_beads_internal_coordinate

    # ------ output information -------
    def check_gpr_lengthscale(self):
        """
        check the length scale for Gaussian Process Regression model.
        """
        # the range of data in internal coordinate.
        moving_inputs = self.gpr_model.train_inputs[0].detach().cpu().numpy() 
        input_range = np.max(moving_inputs, axis= 0) - np.min(moving_inputs, axis= 0)

        gpr_kernel_number = self.gpr_SE_kernel_number

        # the output scale of gaussian process regression kernels
        gpr_hessian_kernel_outputscale = []
        for i in range(gpr_kernel_number):
            output_scale = np.copy(
                self.gpr_model.covar_module_component_list[i]
                .outputscale.detach().cpu().numpy()
            )
            gpr_hessian_kernel_outputscale.append(output_scale)
        gpr_hessian_kernel_outputscale = np.array(gpr_hessian_kernel_outputscale)

        # the length scale of squared exponential kernel of gaussian process regression model.
        gpr_hessian_lengthscale_ratio_list = []
        gpr_hessian_lengthscale_list = []
        for i in range(gpr_kernel_number):
            fitted_lengthscale = self.gpr_model.base_kernel_component_list[
                i
            ].lengthscale
            fitted_lengthscale = fitted_lengthscale.detach().cpu().numpy()[0]
            gpr_hessian_lengthscale_ratio = fitted_lengthscale / input_range
            gpr_hessian_lengthscale_ratio_list.append(gpr_hessian_lengthscale_ratio)
            gpr_hessian_lengthscale_list.append(fitted_lengthscale)
        gpr_hessian_lengthscale_ratio_list = np.array(
            gpr_hessian_lengthscale_ratio_list
        )
        gpr_hessian_lengthscale_list = np.array(gpr_hessian_lengthscale_list)

        return (
            gpr_hessian_kernel_outputscale,
            gpr_hessian_lengthscale_list,
            gpr_hessian_lengthscale_ratio_list,
        )

    # ---- save and load gaussian process regression model ------- 
    def save_model(self, file_path):
        """
        save the hyper-parameter of the gpr model.
        """
        state_dict = self.gpr_model.state_dict() 

        # save state dict.
        if os.path.exists(file_path + "#"):
            os.remove(file_path + "#")
        if os.path.exists(file_path):
            os.rename(file_path, file_path + "#")
        torch.save(state_dict, file_path)
    
    def load_model(self, file_path):
        """
        load the hyper-parameter of the gpr model
        """
        cuda_available = torch.cuda.is_available()
        if os.path.exists(file_path):
            if not cuda_available:
                state_dict = torch.load(file_path, map_location= torch.device('cpu'))
            else:
                state_dict = torch.load(file_path, map_location= torch.device('cuda:0'))
            self.gpr_model.load_state_dict(state_dict)
            print("successfully load the gpr model in gpr_hessian_tools.py")
        else:
            raise(FileExistsError, f"unable to load the gpr model in gpr_hessian_tools.py at file location: {file_path}")
    
    def output_fixed_internal_dofs(self):
        """
        output internal dofs that are fixed and excluded from GPR modeling.
        """
        fixed_internal_dofs = np.copy(self.FixingDofs.fixed_internal_dofs)
        return fixed_internal_dofs
    
    def output_rigid_internal_dofs(self):
        """
        output internal dofs that are rigid, which will be modeled by Linear regression.
        """
        rigid_internal_dofs = np.copy(self.FixingDofs.rigid_internal_dofs)
        return rigid_internal_dofs
