"""
packages for constructing gaussian process regression model
using gpytorch framework (See https://docs.gpytorch.ai/en/stable/).
Written by Chenghao Zhang & Niri Govind, Pacific Northwest National Laboratory (chenghao.zhang@pnnl.gov), 2024.
"""

import torch
import gpytorch
import numpy as np
# from gpr.internal.internal.CoulombInternal import non_redundant_coordinate_transformer
from gpr.internal.ZmatrixInternal import non_redundant_coordinate_transformer
import ipi.utils.depend
from ipi.utils.depend import dstrip
from .gprGrad.rbf_grad_gp import GPModelWithDerivatives, train_gpr
import os 
import shutil
from ipi.utils.messages import  warning

def predict_latent_function_gp_with_derivative(
    model: GPModelWithDerivatives, test_inputs, covar_bool=False
):
    """
    the function that predict the posterior distribution latent function f(test_inputs) of the test_inputs.

    :param: model: instance of GPModelWithDerivatives. Gaussian process regression model using derivative information.
    :param: test_inputs:  test data to compute posterior distribution. dtype: torch.tensor
    :param: covar_bool: bool variable, if true: output covariance matrix. If false: variance (diagonal of covariance matrix). Default: False

    suppose output_dim = m.
    return: mean: mean value of prediction for test_inputs. shape: [N, m]

            if covar_bool == True: return: test_covariance_list: [N ,m, m]. here we assume no correlation between predicted data points. 
            if covar_bool == False: return: test_var: [N, m]: each row is the variance of one data point.
    """
    # check the input dim of test data is correct
    test_input_dim = test_inputs.shape[-1]
    assert (
        test_input_dim == model.input_dim
    ), "dimension of input test_inputs data is incompatible with model"

    model.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        latent_func = model(
            test_inputs
        )  # MultitaskMultivariateNormal distribution object.

        data_num = test_inputs.shape[0]  # number of test_inputs data

        test_mean = latent_func.mean
        test_covariance = latent_func.covariance_matrix

        # diagonal component of covariance matrix is the variance of function and gradient
        test_var = torch.diag(test_covariance)
        test_var = test_var.reshape(
            [model.output_dim, data_num]
        )  # first row is variance for f,  second row is variance for df/dx1, third row: df/dx2, ..
        test_var = torch.transpose(test_var, 0, 1)  # now each row is one data piont.

        # return covariance matrix for each data set.
        # shape: [data_num, model.output_dim, model.output_dim]
        test_covariance_list = torch.zeros([data_num, model.output_dim, model.output_dim], device= model.device)
        for i in range(data_num):
            index = torch.arange(i, model.output_dim * data_num, data_num, device= model.device)
            index_2d = torch.meshgrid(index, index, indexing= 'ij')
            test_data_point_covariance = test_covariance[index_2d[0], index_2d[1]]
            test_covariance_list[i] = test_data_point_covariance
        
    if covar_bool:
        return test_mean, test_covariance_list
    else:
        return test_mean, test_var


def filter_new_training_data(
    model: GPModelWithDerivatives,
    new_train_inputs,
    existing_train_inputs,
    distance_cutoff,
):
    """
    we will reject adding new data when the new data point is too close to the existing data point. (given by distance_cutoff)
    Having two similar data points in the training set will make kernel matrix become ill-conditioned,
    when we try to make prediction, inverting kernel matrix K will be unstable, which causes GPR model to crash.
    :param: model: GPR model
    :param: new_train_inputs: numpy array. potential new training inputs to add to gpr model
    :param: existing_train_inputs: numpy array.  training inputs already in the GPR model.
    :param: distance_cutoff: cutoff of distance in the internal coordinate. The new training data should have distance from the existing training data larger than distance_cutoff.

    :return: filtered_new_train_inputs_index: numpy array. The index of the new training inputs after we deleting  the data which is too close to the existing data.
    """
    filtered_new_train_inputs_index = []

    # kernel output scale and kernel length scale of kernels
    kernel_output_scale = model.output_kernel_outputscale()
    kernel_length_scale = model.output_kernel_lengthscale()
    # normalize the output scale:
    output_scale_sum = np.sum(kernel_output_scale)
    kernel_output_scale_normalized = kernel_output_scale / output_scale_sum
    # effective kernel lengthscale for scaling internal coordinate. l_eff^{-2} = sum_{n} output_scale_n / (l_n)^2.
    effective_kernel_length_scale = np.power(
        np.sum(
            kernel_output_scale_normalized[:, np.newaxis]
            / np.power(kernel_length_scale, 2),
            axis=0,
        ),
        -0.5,
    )

    new_data_num = len(new_train_inputs)

    # record the distance of closet data to new data 
    internal_coordinate_closest_r_list = []

    for data_index in range(new_data_num):
        new_input = new_train_inputs[data_index]

        # distance from new input to the existing training data
        internal_coordinate_r = np.linalg.norm(
            (new_input - existing_train_inputs) / effective_kernel_length_scale, axis=1
        )

        # the data that is closest to the new input
        nearest_existing_training_inputs_index = np.argmin(internal_coordinate_r)

        internal_coordinate_closest_r = internal_coordinate_r[
            nearest_existing_training_inputs_index
        ]
        
        internal_coordinate_closest_r_list.append(internal_coordinate_closest_r)
        
        if internal_coordinate_closest_r > distance_cutoff:
            filtered_new_train_inputs_index.append(data_index)

    filtered_new_train_inputs_index = np.array(filtered_new_train_inputs_index)

    return filtered_new_train_inputs_index


def update_model_with_new_data(
    model: GPModelWithDerivatives, 
    new_train_inputs, 
    new_train_targets, 
    distance_cutoff,
    train_bool
):
    """
    add new training data into the model.
    Then train the model to update the hyper-parameter.

    :param: model: GPModelWithDerivative: GPR model that train with derivative information
    :param: new_train_inputs: new training input data. datatype: torch.tensor() or numpy array. Size [N, d]. here d is input_dim
    :param: new_train_targets: new training target data. datatype: torch.tensor() or numpy array. Size [N, m], here m is output dim (m = d + 1)
    """
    train_inputs = model.train_inputs[0]
    train_targets = model.train_targets

    # check the data type of input training data. If it's not torch.Tensor, convert it to torch.Tensor.
    if not isinstance(new_train_inputs, torch.Tensor):
        new_train_inputs_tensor = torch.from_numpy(np.array(new_train_inputs)).to(device= model.device, dtype=torch.float64)
    else:
        new_train_inputs_tensor = torch.clone(new_train_inputs)

    if not isinstance(new_train_targets, torch.Tensor):
        new_train_targets_tensor = torch.from_numpy(np.array(new_train_targets)).to(device= model.device, dtype=torch.float64)
    else:
        new_train_targets_tensor = torch.clone(new_train_targets)

    # check the input dimension of the new_train_inputs
    assert (
        new_train_inputs_tensor.shape[-1] == model.input_dim
    ), "the input dimension of new_train_inputs is wrong. new_train_input dim {}. required input dim {}".format(
        new_train_inputs.shape[-1], model.input_dim
    )
    # check the output dimension of the new_train_targets
    assert (
        new_train_targets_tensor.shape[-1] == model.output_dim
    ), "the output dimension of new_train_targets is wrong. new_train_targets dim {}, required output dim {}".format(
        new_train_targets.shape[-1], model.output_dim
    )

    # filter the new inputs which is too close to the existing data point.
    new_train_inputs_numpy = new_train_inputs_tensor.cpu().numpy()
    train_inputs_numpy = train_inputs.cpu().numpy()

    # filter the training input data to delete the one which is too close to the existing data.
    filtered_new_train_inputs_index = filter_new_training_data(
        model, new_train_inputs_numpy, train_inputs_numpy, distance_cutoff
    )
    filtered_new_train_inputs_tensor = new_train_inputs_tensor[
        filtered_new_train_inputs_index, :
    ]
    filtered_new_train_targets_tensor = new_train_targets_tensor[
        filtered_new_train_inputs_index, :
    ]

    if len(filtered_new_train_inputs_index) == 0:
        # all new data is too close to the existing data, we do not update the model.
        return filtered_new_train_inputs_index

    # create new training inputs and training targets
    full_train_inputs = torch.cat(
        [train_inputs, filtered_new_train_inputs_tensor], dim=0
    )
    full_train_targets = torch.cat(
        [train_targets, filtered_new_train_targets_tensor], dim=0
    )

    # set the training data for the model
    model.set_train_data(
        inputs=full_train_inputs, targets=full_train_targets, strict=False
    )

    # re-train the model to update the hyper-parameter
    if train_bool:
        train_gpr(model)

    return filtered_new_train_inputs_index


class FixInternalDofs(object):
    """
    class that fix certain internal dofs in the training data before feeding data into the Gaussian Process Regression model
    """

    def __init__(self,
                 train_x: np.ndarray, 
                 train_inputs: np.ndarray, 
                 train_targets: np.ndarray,
                 cartesian_fix_dofs: np.ndarray,
                 coordinate_transformer: non_redundant_coordinate_transformer,
                 gpr_fix_internal_dofs_bool: bool, 
                 gpr_fix_internal_dofs_cutoff: float,
                 internal_coord_type,
                 gpr_fixed_internal_dofs= None):
        
        self.input_dim = np.shape(train_inputs)[1]
        self.output_dim = np.shape(train_targets)[1]

        # code that use the change in inputs to choose relevant dofs in GPR model.
        """
        self.fix_internal_dofs_cutoff = gpr_fix_internal_dofs_cutoff
        # check whether coordinate along certain internal dofs need to be fixed.
        # the change along internal coordinate will be computed using Wilson's B matrix. (any coordinate should be fine.)
        # This is to fix the problem for the planar molecule
        sq = coordinate_transformer.ref_S 
        vh = coordinate_transformer.ref_Vh 
        # compute change of training data along Cartesian coordinate.
        train_x_change = np.max(train_x, axis= 0) - np.min(train_x, axis= 0)

        # check the case that we do not fix cartesian dofs. 
        # report error if these dofs are small
        train_x_cutoff = pow(10.0, -3)
        index = [i for i in range(len(train_x_change)) if train_x_change[i] < train_x_cutoff]
        if len(index) != 0:
            warning(f"@Warning: Planar molecules? The changes of cartesian coordinate are small.  dofs: {index}.Cartesian coordinate change {train_x_change[index]}")

        # if cartesian dofs is fixed, we set its change to 0.
        train_x_change[cartesian_fix_dofs] = 0 

        if internal_coord_type == "Coulomb":
            train_inputs_change = np.abs(sq * (vh @ train_x_change))
        elif internal_coord_type == "bond":
            train_inputs_change = np.max(train_inputs, axis= 0) - np.min(train_inputs, axis= 0)
        else:
            print("Warning: internal coord type unrecognized.")
            train_inputs_change = np.max(train_inputs, axis= 0) - np.min(train_inputs, axis= 0)
        
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
                        if train_inputs_change[i] < self.fix_internal_dofs_cutoff
                    ]
                )
            else:
                self.fixed_internal_dofs = gpr_fixed_internal_dofs
                print("@gpr_model: load fixed internal dofs.")
            
            print(f"@gpr_model: For Fixing internal dofs: fixed_internal_dofs: {self.fixed_internal_dofs}")
        else:
            self.fixed_internal_dofs =  np.array(
                []
            )

        if len(self.fixed_internal_dofs) != 0:
            self.free_moving_dofs = np.delete(
                np.arange(self.input_dim), self.fixed_internal_dofs
            )
            self.targets_fixed_dofs = np.mean(train_targets, axis=0)[
                self.fixed_internal_dofs + 1
            ]  # the first column of the target is the potential V.
        else:
            self.free_moving_dofs = np.arange(self.input_dim)
            self.targets_fixed_dofs = np.array([])
        """
        
        # code that use the change in forces as criterion to select dofs to include in GPR model.
        grad_q = train_targets[:, 1:]
        grad_q_change = np.max(grad_q, axis = 0) - np.min(grad_q, axis= 0)
        grad_q_change_cutoff = np.max(grad_q_change) / np.power(10.0, 3)

        self.free_moving_dofs = np.arange(self.input_dim)[ grad_q_change > grad_q_change_cutoff ]
        self.fixed_internal_dofs = np.arange(self.input_dim)[ grad_q_change <= grad_q_change_cutoff ]
        # load the fixed internal dofs from the folder.
        if gpr_fixed_internal_dofs is not None:
            self.fixed_internal_dofs = gpr_fixed_internal_dofs
            self.free_moving_dofs = np.delete(np.arange(self.input_dim), self.fixed_internal_dofs)

        print(f"@gpr_model: For Fixing internal dofs: fixed_internal_dofs: {self.fixed_internal_dofs}")
        print(f"@gpr_model: free moving dofs: {self.free_moving_dofs}")

        if len(self.fixed_internal_dofs) != 0:
            self.targets_fixed_dofs = np.mean(train_targets, axis= 0)[
                self.fixed_internal_dofs + 1
            ]
        else:
            self.targets_fixed_dofs = np.array([])

    def transform_training_data_to_free_moving_dofs(
        self, train_inputs: np.ndarray, train_targets: np.ndarray, noise_var=None
    ):
        """
        delete fixdofs from training data
        :param: train_inputs: the training inputs in internal dofs. Including all internal dofs.
        :param: train_targets: the training targets in internal dofs. Including all internal dofs.
        :param: noise_var: the variance of noise in internal dofs.
        """
        moving_train_inputs = self.transform_training_inputs_to_free_moving_dofs(
            train_inputs
        )
        moving_train_targets = self.transform_training_targets_to_free_moving_dofs(
            train_targets
        )  # the first column of target is potential V.

        if not (noise_var is None):
            if len(self.fixed_internal_dofs) != 0:
                moving_noise_var = np.delete(noise_var, self.fixed_internal_dofs + 1)
            else:
                moving_noise_var = noise_var

            return moving_train_inputs, moving_train_targets, moving_noise_var
        else:
            return moving_train_inputs, moving_train_targets

    def transform_training_inputs_to_free_moving_dofs(self, train_inputs: np.ndarray):
        """
        delete fixdofs from training inputs.
        :param: train_inputs: the training inputs in internal dofs.
        """
        if len(self.fixed_internal_dofs) != 0:
            moving_train_inputs = np.delete(
                train_inputs, self.fixed_internal_dofs, axis=1
            )
        else:
            moving_train_inputs = train_inputs

        return moving_train_inputs

    def transform_training_targets_to_free_moving_dofs(self, train_targets: np.ndarray):
        """
        delete fixdofs from training targets
        :param: train_targets: the training targets in internal dofs
        """
        if len(self.fixed_internal_dofs) != 0:
            moving_train_targets = np.delete(
                train_targets, self.fixed_internal_dofs + 1, axis=1
            )  # the first column of target is potential V.
        else:
            moving_train_targets = train_targets

        return moving_train_targets

    def transform_from_free_moving_dofs_to_full_dofs(
        self, moving_test_mean: np.ndarray, moving_test_covar_matrix: np.ndarray
    ):
        """
        Transform the prediction of GPR model from free moving dofs into full dofs.
        :param: moving_prediction_mean: the mean value of posterior prediction for moving dofs
        :param: moving_prediction_var: the variance of posterior prediction for moving dofs
        """
        test_data_num = moving_test_mean.shape[0]

        # the prediction of fixed dof for testing data.
        test_target_fixed_dofs = np.repeat(
            [self.targets_fixed_dofs], test_data_num, axis=0
        )

        # the mean value of the prediction of the test data with all dofs
        test_mean = np.zeros([test_data_num, self.output_dim])
        test_mean[:, 0] = moving_test_mean[:, 0]  # potential
        test_mean[:, self.free_moving_dofs + 1] = moving_test_mean[:, 1:]
        if len(self.fixed_internal_dofs) != 0:
            test_mean[:, self.fixed_internal_dofs + 1] = test_target_fixed_dofs

        # the covariance of the prediction of the test data with all dofs
        # the covar matrix component of fixed dofs is set to 0.
        test_covar_matrix = np.zeros([test_data_num, self.output_dim, self.output_dim])
        test_covar_matrix[:,0, 0] = moving_test_covar_matrix[:, 0, 0]
        free_moving_grad_index = self.free_moving_dofs + 1  # index 0 is for potential.
        free_moving_grad_index_2d = np.meshgrid(free_moving_grad_index,
                                                free_moving_grad_index,
                                                indexing= 'ij')
        test_covar_matrix[:, free_moving_grad_index_2d[0], free_moving_grad_index_2d[1]] = moving_test_covar_matrix[:, 1:, 1:]

        return test_mean, test_covar_matrix


class NormalizeTrainingData(object):
    """
    normalize the potential & force of training data.
    """
    def __init__(self, 
                 training_targets: np.ndarray,
                 training_inputs: np.ndarray):
        """
        V_normalized = (V - V_mean)/V_range.
        :param: training_targets : [V, dV/dx]. numpy array.
        """
        V = training_targets[:, 0]
        self.V_mean = np.mean(V)
        self.V_range = np.max(V) - np.min(V)

        # transform the coordinate. Do it for the initial data.
        # TODO: This code could cause trouble when we reload the training data.
        # Because the q_mean and q_range will change after we add new data.
        q_mean = np.mean(training_inputs, axis= 0)  # <q>
        q_range = np.max(training_inputs, axis= 0) - np.min(training_inputs, axis= 0) # q_max - q_min

        self.q_mean = q_mean 
        self.q_range = q_range

    def normalization_transform(self, 
                                training_targets,
                                training_inputs):
        """
        normalize the potential V & force F.
        V_normalized = (V - V_mean) / V_range.
        F_normalized = F / V_range.
        
        Then:
        F_normalized = F_normalized * q_range.
        q = (q - <q>) / q_range
        This function perform the normalize procedure.
        :param: training_targets : [V, dV/dq]. numpy array.
        """
        V = training_targets[:, 0]
        V_normalized = (V - self.V_mean) / self.V_range

        grad_V = training_targets[:, 1:]
        grad_V_normalized = grad_V / self.V_range

        # transform the coordinate and gradient.
        normalized_training_inputs = (training_inputs - self.q_mean[np.newaxis, :]) / self.q_range[np.newaxis, :]
        grad_V_normalized = self.q_range[np.newaxis, :] * grad_V_normalized

        normalized_training_targets = np.concatenate(
            [
                V_normalized[:, np.newaxis], 
                grad_V_normalized
            ], 
            axis=1
        )

        return normalized_training_targets, normalized_training_inputs

    def normalization_transform_for_inputs(self, 
                                           training_inputs):
        """
        normalize the input.
        q = (q - <q>) / q_range
        """
        normalized_training_inputs = (training_inputs - self.q_mean[np.newaxis, :]) / self.q_range[np.newaxis, :]
        return normalized_training_inputs

    def inverse_normalization_transform(self, normalized_training_targets):
        """
        inverse the normalization procedure of potential V & force F.

        V = V_normalized * V_range + V_mean
        F = F_normalized * V_range

        This function perform the inverse of normalization procedure.
        :param: normalized_training_targets: [V_normalized, d V_normalized /dx]. numpy array.
        """
        V_normalized = normalized_training_targets[:, 0]
        grad_V_normalized = normalized_training_targets[:, 1:]

        # scale the potential and gradient of potential
        V = V_normalized * self.V_range + self.V_mean
        grad_V = grad_V_normalized * self.V_range

        # new code: re-scale the grad_V.
        grad_V = grad_V / self.q_range[np.newaxis, :]

        training_targets = np.concatenate([V[:, np.newaxis], grad_V], axis=1)

        return training_targets

    def normalize_noise_var(self, noise_var):
        """
        normalize the variance of noise
        """
        normalized_noise_var = noise_var / np.power(self.V_range, 2)

        # rescale the variance of gradient noise due to the scaling of the input coordinate.
        normalized_grad_noise_var = normalized_noise_var[1:]
        normalized_grad_noise_var = normalized_grad_noise_var * np.power(self.q_range, 2)
        normalized_noise_var[1:] = normalized_grad_noise_var

        return normalized_noise_var

    def inverse_normalize_noise_covar_matrix(self, normalized_noise_covar_matrix):
        """
        inverse normalize the covariance matrix of the noise
        """
        noise_covar_matrix = normalized_noise_covar_matrix * np.power(self.V_range, 2)

        # inverse rescale the variance of gradient noise 
        grad_noise_covar_matrix = noise_covar_matrix[:,1:, 1:]
        grad_noise_covar_matrix = grad_noise_covar_matrix / np.outer(self.q_range, self.q_range)[np.newaxis, :]
        noise_covar_matrix[:,1:, 1:] = grad_noise_covar_matrix

        return noise_covar_matrix


class GPModelWithDerivativesWrapper:
    """
    wrapper class for GPModelWithDerivatives.
    handles the transformation between internal coordinate and cartesian coordinate + GPR training.
    """

    def __init__(
        self,
        train_x: np.ndarray,
        train_V: np.ndarray,
        train_grad_x: np.ndarray,
        natom: int,
        coordinate_transformer: non_redundant_coordinate_transformer,
        cartesian_fix_dofs: np.ndarray,
        gpr_SE_kernel_number: int,
        kernel_outputscale: np.ndarray,
        kernel_lengthscale_ratio: np.ndarray,
        noise_std,
        train_bool= True,
        gpr_fix_internal_dofs_bool= False,
        gpr_fix_internal_dofs_cutoff = 1e-4,
        gpr_fixed_internal_dofs= None,
        singular_value_cutoff = 1e-8
    ):
        """
        initialize the model.
        :param: train_x: [N, 3 * natom]. initial N training points x. (Cartesian coordinate)  numpy array.
        :param: train_V: [N]. initial N training potential V.    numpy array.
        :param: train_grad: [N, 3 * natom], initial N training data. gradient of potential V.  numpy array.
        :param: natom: number of atoms.
        :param: coordinate transformer: an instance of the class: non_redundant_coordinate_transformer. Responsible for transformation between external and internal coordinate.
        :param: gpr_SE_kernel_number: number of squared exponential kernels that is used to construct the covariance function.
        :param: kernel_outputscale: output scale of each squared exponential kernel used to construct covariance function. numpy array.
        :param: kernel_lengthscale_ratio: length scale ratio of each squared exponential kernel used to construct covariance function. numpy array.
        :param: noise_std:  the noise of likelihood function p(y|f).
                                                       y = f + epsilon.
                            Note the potential V and force f have different noise.
                            The noise for force is defined in the Cartesian coordinate, we need to transform it into the internal coordinate.
        :param: singular_value_cutoff: singular value cutoff for pesudo-inverse of covariance matrix.
        """
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

        self.natom = natom
        self.gpr_SE_kernel_number = gpr_SE_kernel_number
        self.coordinate_transformer = coordinate_transformer

        # record the input data about initial value of kernel and noise info.
        self.kernel_outputscale = kernel_outputscale
        self.kernel_lengthscale_ratio = kernel_lengthscale_ratio
        self.noise_std = noise_std

        # the training targets for the GPR with derivative is [V, dV/dx1, ..., dV/dxn]
        train_cartesian_targets = np.concatenate(
            [train_V[:, np.newaxis], train_grad_x], axis=1
        )

        self.train_cartesian_inputs = (
            train_x  # training inputs in cartesian coordinate x
        )
        self.train_cartesian_targets = train_cartesian_targets  # training targets in cartesian coordinate (V, dV/dx)

        # put tensor on gpu if cuda is available
        cuda_available = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda_available else 'cpu')
        print("GPytorch for force prediction:")
        if cuda_available:
            print("CUDA is available. GPU is enabled.")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs available: {torch.cuda.device_count()}")
            print(f"Current GPU device: {torch.cuda.current_device()}")
            print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            print("CUDA is not available. Running Gpytorch on CPU.")

        # --------- Transform coordinate and gradient from Cartesian coordinate into the internal coordinate.
        train_inputs, train_targets, likelihood_noise_variance = self.transform_data_to_internal_coordinate(
            train_x,
            train_V,
            train_grad_x,
            noise_std
        )
        self.train_inputs = (
            train_inputs  # training inputs in internal coordinate space q.
        )

        input_dim = np.shape(train_inputs)[
            1
        ]  # numbers of degree of freedom for the non-redundant internal coordinate q.
        output_dim = (
            input_dim + 1
        )  # output_dim = input_dim + 1 (train_targets = [V, dV/dx1, ..., dV/dxn])

        self.input_dim = input_dim
        self.output_dim = output_dim

        # ------- Normalize the training inputs, targets and noise -----------
        # decide normalization parameter. Here we normalize the potential as (V- <V>)/range(V). The force also needs to be scaled.
        self.Normalizer = NormalizeTrainingData(train_targets,
                                                train_inputs
                                                )

        normalized_train_inputs, normalized_train_targets, likelihood_noise_variance = self.normalize_data(
            train_inputs,
            train_targets,
            likelihood_noise_variance
        )
        # record normalized training input and normalized training targets. 
        self.normalized_train_inputs = (
            normalized_train_inputs  
        )

        # training outputs in internal coordinates q. (V, dV/dq)
        self.normalized_train_targets = normalized_train_targets  

        # -------- Fixing certain dofs ----------------
        # For the case we have to fix certain internal dofs. Apply a filter to fix some internal dofs
        # To filter internal dofs, we still use the initial train_inputs as criterion. (not the re-scaled one.)
        self.FixingDofs = FixInternalDofs(train_x,
                                          train_inputs, 
                                          normalized_train_targets,
                                          cartesian_fix_dofs,
                                          coordinate_transformer,
                                          gpr_fix_internal_dofs_bool,
                                          gpr_fix_internal_dofs_cutoff,
                                          self.coordinate_transformer.internal_coord_type,
                                          gpr_fixed_internal_dofs
                                          )
        
        moving_train_inputs, moving_train_targets, moving_likelihood_noise_variance = self.fix_internal_dofs(
            normalized_train_inputs, normalized_train_targets, likelihood_noise_variance
        )

        # ------- transform input from numpy array to torch.tensor -----------
        (moving_train_inputs, moving_train_targets) = map(
            lambda x: torch.from_numpy(x).to(device= self.device, dtype=torch.float64), (moving_train_inputs, moving_train_targets)
        )

        # -------- fixing certain dofs. -----------------

        # initialize the gaussian process regression model with input training data.
        self.gpr_model = GPModelWithDerivatives(
            moving_train_inputs,
            moving_train_targets,
            self.moving_input_dim,
            self.moving_output_dim,
            gpr_SE_kernel_number,
            kernel_outputscale,
            kernel_lengthscale_ratio,
            moving_likelihood_noise_variance,
            nugget= singular_value_cutoff 
        )
        self.gpr_model = self.gpr_model.to(device= self.device)

        if train_bool:   
            # train self.gpr_model() to get optimized hyperparameter
            self.train_gpr()

    def transform_data_to_internal_coordinate(self, train_x, train_V, train_grad_x, noise_std= None):
        """
        Transform the cartesian coordinate, and gradient into the internal coordinate.
        Transform the noise from Cartesian coordinate into the internal coordinate.
        """
        # transform cartesian coordinate x to internal coordinate q
        train_inputs = self.coordinate_transformer.get_internal_coordinate_q(train_x)

        # shape: [N, 3 * natom - 6]
        # transform the gradient of potential V: dV/dx -> dV/dq
        train_grad_q = (
            self.coordinate_transformer.transform_cartesian_gradient_to_internal_gradient(
                train_x, train_grad_x
            )
        )

        # target data: [V, dV/dq1, ..., dV/dqn]
        train_targets = np.concatenate([train_V[:, np.newaxis], train_grad_q], axis=1)

        if noise_std != None:
            # compute the estimated noise covariance factor for the force in the internal coordinate q. noise for Fq = dV/dq.
            likelihood_noise_variance = self.transform_cartesian_noise_to_gpr_model_noise(
                noise_std
            )

        if noise_std != None:
            return train_inputs, train_targets, likelihood_noise_variance
        else:
            return train_inputs, train_targets

    def normalize_data(self, train_inputs, train_targets, likelihood_noise_variance):
        """
        Normalize the training inputs, training targets and the noise.
        """
        # perform normalization on training targets.
        normalized_train_targets, normalized_train_inputs = (
            self.Normalizer.normalization_transform(
                train_targets,
                train_inputs
            )
        )
        
        likelihood_noise_variance = self.Normalizer.normalize_noise_var(
            likelihood_noise_variance
        )

        return normalized_train_inputs, normalized_train_targets, likelihood_noise_variance

    def fix_internal_dofs(self, train_inputs, train_targets, likelihood_noise_variance):
        """
        delete certain fixed dofs in training inputs and targets.
        """
        moving_train_inputs, moving_train_targets, moving_likelihood_noise_variance = (
            self.FixingDofs.transform_training_data_to_free_moving_dofs(
                train_inputs, train_targets, likelihood_noise_variance
            )
        )

        self.moving_input_dim = self.input_dim - len(self.FixingDofs.fixed_internal_dofs)
        self.moving_output_dim = self.output_dim - len(self.FixingDofs.fixed_internal_dofs)

        return moving_train_inputs, moving_train_targets, moving_likelihood_noise_variance


    def train_gpr(self):
        """
        train the gpr model.
        """
        train_gpr(self.gpr_model)


    def transform_cartesian_noise_to_gpr_model_noise(self, noise_std):
        """
        transform the noise in cartesian coordinate to noise in Gaussian Process Regressioin model.
        This is critical for the successful training of the GPR model, otherwise, we will treat the noise incorrectly.
        """
        pot_noise_std = noise_std["pot_noise_prior"]
        force_noise_std_cartesian = noise_std[
            "force_noise_prior"
        ]  # noise of force in the Cartesian coordinate. We assume the noise is isotropic.

        # force noise has to be scaled by the inverse of the singular value of Wilson's B matrix.
        self.Bmatrix_singular_value_square = np.power(
            self.coordinate_transformer.ref_S, 2
        )

        singular_value_square_inverse = 1 / self.Bmatrix_singular_value_square

        force_noise_var = singular_value_square_inverse * np.power(
            force_noise_std_cartesian, 2
        )

        pot_noise_var = np.power(pot_noise_std, 2)

        noise_variance = np.concatenate([[pot_noise_var], force_noise_var])
        return noise_variance

    def predict_latent_function(self, test_x, internal_coordinate_bool=False):
        """
        compute the predicted potential V and gradient dV/dx (mean value of latent prediction distribution) in Cartesian coordinate.
        Also compute the variance of potential & gradients dV/dq.
        This function wraps predict_latent_function_gp_with_derivative defined in GPModelWithDerivative.
        This function handles 1. normalization  2. filter fixed dofs  3. transformation between Cartesian and internal dofs.

        :param: test_x: input Cartesian coordinate data [N, 3 * natom].
        :param: internal_coordinate_bool: if internal coordinate bool = True, then we output gradient of potential in internal coordinate.
                                          otherwise, we output the gradient of potential in Cartesian coordinate.

        :return: V: predicted potential energy.
                grad_x: dV/dx, predicted gradient of potential energy. In Cartesian coordinate.
                Or grad_q: dV/dq, predicted gradient of potential energy, in internal coordinate.
                var_V: uncertainty (variance) of potential energy.
                var_grad_x_trace: trace of the covariance matrix in the Cartesian coordinate. This can be used as a measure of the force noise.
        """
        if type(test_x) == ipi.utils.depend.depend_array:
            test_x_array = dstrip(test_x).copy()
        else:
            test_x_array = test_x

        assert (
            np.shape(test_x_array)[1] == 3 * self.natom
        ), "dim of coordinates for input data is not 3 * natom"

        # transform to internal coordinate q. normalization + filter fixed dofs.
        moving_test_q = self.get_free_moving_internal_coordinate(test_x_array)
        moving_test_q = torch.from_numpy(moving_test_q).to(device= self.device, dtype=torch.float64)

        # use Gaussian process regression model to make prediction
        moving_normalized_test_mean, moving_normalized_test_covar_matrix = (
            predict_latent_function_gp_with_derivative(
                self.gpr_model, 
                test_inputs= moving_test_q,
                covar_bool= True
            )
        )

        moving_normalized_test_mean = (
            moving_normalized_test_mean.detach().cpu().numpy()
        )
        moving_normalized_test_covar_matrix = (
            moving_normalized_test_covar_matrix.detach().cpu().numpy()
        )

        # attach test_mean and test_var (0) of fixed dofs
        normalized_test_mean, normalized_test_covar_matrix = (
            self.FixingDofs.transform_from_free_moving_dofs_to_full_dofs(
                moving_normalized_test_mean, moving_normalized_test_covar_matrix
            )
        )

        # inverse the normalization procedure for mean value and variance.
        test_covar_matrix_q = self.Normalizer.inverse_normalize_noise_covar_matrix(
            normalized_test_covar_matrix
        )
        
        test_mean = self.Normalizer.inverse_normalization_transform(
            normalized_test_mean
        )

        V = test_mean[:, 0]
        grad_q = test_mean[:, 1:]  # gradient dV/dq.

        # transform gradient from internal coordinate back to cartesian coordinate.
        grad_x = self.coordinate_transformer.transform_internal_gradient_to_cartesian_gradient(
            test_x_array, grad_q
        )

        var_V = test_covar_matrix_q[:, 0, 0]
        var_grad_q = test_covar_matrix_q[:, np.arange(1, self.output_dim), np.arange(1, self.output_dim)]
        var_grad_q_trace = np.sum(var_grad_q, axis= 1)

        # transformation for covariance matrix from internal coordinate to Cartesian coordinate.
        test_data_num, output_dim_x = test_x_array.shape
        Bq = self.coordinate_transformer.compute_delocalized_wilson_matrix_Bq(test_x_array)
        Bq_T = np.transpose(Bq, (0, 2, 1))
        
        grad_covar_matrix_q = test_covar_matrix_q[:, 1:, 1:]
        grad_covar_matrix_x = Bq_T @ grad_covar_matrix_q @ Bq

        test_covar_matrix_x = np.zeros((test_data_num, output_dim_x + 1, output_dim_x + 1))
        test_covar_matrix_x[:, 0, 0] = test_covar_matrix_q[:, 0, 0]
        test_covar_matrix_x[:, 1:, 1:] = grad_covar_matrix_x 

        # covariance matrix for the noise in the Cartesian coordinate: Cov(noise_x, noise_x) = V S diag(var_grad_q) S V^T. Here S is singular value matrix, V is the right singular vector matrix.
        # the measure of the force noise can be defined as the trace of the covariance matrix of the force noise in Cartesian coordinate.
        # var_grad_x_trace = np.sum(
        #     self.Bmatrix_singular_value_square * var_grad_q, axis=1
        # )
        var_grad_x = np.diagonal(test_covar_matrix_x[:, 1:, 1:], axis1= 1, axis2= 2)
        var_grad_x_trace = np.sum(var_grad_x, axis= 1)

        if internal_coordinate_bool:
            return V, grad_q, var_V, var_grad_q_trace
        else:
            return V, grad_x, var_V, var_grad_x_trace  

    def update_model_with_new_data(
        self, 
        new_train_x: np.ndarray, 
        new_train_V: np.ndarray, 
        new_train_grad_x: np.ndarray, 
        distance_cutoff,
        train_bool= True
    ):
        """
        add new training data into the model.
        Then train the model to update the hyper-parameter.
        This function wraps the function: update_model_with_new_data(gpr_model, train_inputs, train_targets)
        TODO: Note, there is still room for improvement in this function, here each time we update new data, we have to re-compute the inverse of covariance matrix K.

        This function will update the self.gpr_model

        :param: new_train_x: [N, 3 * natom], input Cartesian coordinate data.  numpy array
                new_train_V: [N], ab-initio potential data.   numpy array
                new_train_grad_x: [N, 3 * natom], ab-initio gradient data.  numpy array.

        :return: None.
        """
        assert (
            np.shape(new_train_x)[1] == 3 * self.natom
        ), "dim of coordinates for input data is not 3 * natom"
        assert (
            np.shape(new_train_grad_x)[1] == 3 * self.natom
        ), "dim of gradients for input data is not 3 * natom"

        # input data in internal coordinate
        new_train_inputs, new_train_targets = self.transform_data_to_internal_coordinate(
            new_train_x,
            new_train_V,
            new_train_grad_x
        )

        # check the shape of gradient in internal coordinate.
        new_train_grad_q = new_train_targets[:,1:]
        assert (
            np.shape(new_train_grad_q)[1] == self.input_dim
        ), "train_grad_q for internal coordiante has wrong dimension"

        # normalize the new_train_targets
        normalized_new_train_targets, normalized_new_train_inputs = self.Normalizer.normalization_transform(
            new_train_targets,
            new_train_inputs
        )

        # fix certain dofs from input and targets, not including it in our gpr model.
        moving_new_train_inputs, moving_new_train_targets = (
            self.FixingDofs.transform_training_data_to_free_moving_dofs(
                normalized_new_train_inputs, normalized_new_train_targets
            )
        )

        # transform numpy array into tensor.
        (moving_new_train_inputs, moving_new_train_targets) = map(
            lambda x: torch.from_numpy(x).to(device= self.device, dtype=torch.float64), (moving_new_train_inputs, moving_new_train_targets)
        )

        # we only add new training data if they are not too close to each other.
        filtered_new_train_inputs_index = update_model_with_new_data(
            self.gpr_model,
            moving_new_train_inputs,
            moving_new_train_targets,
            distance_cutoff,
            train_bool
        )

        if len(filtered_new_train_inputs_index) != 0:
            self.update_training_variables(
                filtered_new_train_inputs_index,
                new_train_inputs,
                normalized_new_train_inputs,
                normalized_new_train_targets,
                new_train_x,
                new_train_V,
                new_train_grad_x
                )



    def update_training_variables(self, 
                                  filtered_new_train_inputs_index,
                                  new_train_inputs,
                                  normalized_new_train_inputs,
                                  normalized_new_train_targets,
                                  new_train_x,
                                  new_train_V,
                                  new_train_grad_x
                                  ):
        """
        update variables: 
        self.train_inputs, self.normalized_train_inputs, self.normalized_train_targets,
        self.train_cartesian_inputs, self.train_cartesian_targets
        """
        # update the training data and targets in internal coordinate q.
        self.train_inputs = np.concatenate(
            [
                self.train_inputs, 
                new_train_inputs[filtered_new_train_inputs_index]
            ],
            axis=0,
        )

        self.normalized_train_inputs = np.concatenate(
            [
                self.normalized_train_inputs, 
                normalized_new_train_inputs[filtered_new_train_inputs_index]
            ],
            axis= 0
        )

        self.normalized_train_targets = np.concatenate(
            [
                self.normalized_train_targets,
                normalized_new_train_targets[filtered_new_train_inputs_index],
            ],
            axis=0,
        )

        # update the training data and targets in cartesian coordinate x.
        new_train_cartesian_targets = np.concatenate(
            [
                new_train_V[:, np.newaxis],
                new_train_grad_x
            ],
            axis=1
        )

        self.train_cartesian_inputs = np.concatenate(
            [
                self.train_cartesian_inputs,
                new_train_x[filtered_new_train_inputs_index],
            ],
            axis=0,
        )
        self.train_cartesian_targets = np.concatenate(
            [
                self.train_cartesian_targets,
                new_train_cartesian_targets[filtered_new_train_inputs_index],
            ],
            axis=0,
        )

    # ------ functions below are auxiliary functions to output gpr model parameters ------------------------
    def output_kernel_lengthscale(self):
        """
        return the length scale of kernel for gpr model
        :return: length scale (numpy array)
        """
        lengthscale = self.gpr_model.output_kernel_lengthscale()

        return lengthscale

    def output_kernel_outputscale(self):
        """
        return the output scale of the kernel for gpr model.
        :return: output scale (numpy array)
        """
        output_scale = self.gpr_model.output_kernel_outputscale()

        return output_scale

    def output_training_cartesian_inputs(self):
        """
        output the training data set X (in cartesian coordinate) used to train the GPR model.
        """
        train_cartesian_X = np.copy(self.train_cartesian_inputs)

        return train_cartesian_X

    def output_free_moving_training_internal_inputs(self):
        """
        output the training data set Q (in non-redundant internal coordinate) used to train the GPR model
        """
        free_moving_train_inputs = (
            self.FixingDofs.transform_training_inputs_to_free_moving_dofs(
                np.copy(self.normalized_train_inputs)
            )
        )

        return free_moving_train_inputs

    def output_fitted_gpr_model_noises(self):
        """
        output the fitted noises for normalized internal coordinate potential V and forces f.
        """
        V_noises, force_noises = self.gpr_model.output_fitted_noise()

        return V_noises, force_noises

    def output_normalized_force_range(self):
        """
        output the range of normalized force
        """
        free_moving_normalized_targets = (
            self.FixingDofs.transform_training_targets_to_free_moving_dofs(
                self.normalized_train_targets
            )
        )
        free_moving_normalized_force = free_moving_normalized_targets[:, 1:]
        free_moving_normalized_force_range = (np.max(
            free_moving_normalized_force, 
            axis=0
        ) - 
        np.min(free_moving_normalized_force, 
                axis=0
              )
        )

        return free_moving_normalized_force_range

    def output_fixed_internal_dofs(self):
        """
        output the internal dofs we fix. These dofs will not be included in the GPR model.
        """
        fixed_internal_dofs = np.copy(self.FixingDofs.fixed_internal_dofs)
        return fixed_internal_dofs


    def get_free_moving_internal_coordinate(self, beads_x):
        """
        transform from Cartesian coordinate x to the free moving internal coordinates q.
        """
        beads_internal_coordinate = (
            self.coordinate_transformer.get_internal_coordinate_q(beads_x)
        )

        # add code to normalize the training inputs.
        normalized_beads_internal_coordinate = (
            self.Normalizer.normalization_transform_for_inputs(
                beads_internal_coordinate
            )
        )

        free_moving_beads_internal_coordinate = (
            self.FixingDofs.transform_training_inputs_to_free_moving_dofs(
                normalized_beads_internal_coordinate
            )
        )

        return free_moving_beads_internal_coordinate

    # ---- save and load gaussian process regression model -----
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
        if os.path.exists(file_path):
            cuda_available = torch.cuda.is_available()
            if not cuda_available:
                state_dict = torch.load(file_path, map_location= torch.device('cpu'))
            else:
                state_dict = torch.load(file_path, map_location= torch.device('cuda:0'))
            self.gpr_model.load_state_dict(state_dict)
            print("successfully load the gpr model in gprtools.py")
        else:
            raise(FileExistsError, f"unable to load the gpr model in gprtools.py at file location: {file_path}")
        