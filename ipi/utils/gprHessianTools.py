'''
packages for predicting hessian of ring polymer beads using Gaussian Process Regression (GPR).
The GPR part use GPytorch framework (https://docs.gpytorch.ai/en/stable/)
Written by Chenghao Zhang, Pacific Northwest National Laboratory (chenghao.zhang@pnnl.gov)
'''
import torch 
import numpy as np 
import gpytorch 
from ipi.utils.internalcoordtools import non_redundant_coordinate_transformer
from gprHessian.RBFHessian_gp import GPModelWithHessians, train_gpr_model, predict_latent_function_GPHessian
from gprHessian.RBFHessian_utils import take_upper_triangular_part, transform_1d_train_targets_into_pots_grads_hessians

class TransformTrainingTarget(object):
    '''
    class that handles the transformation between [V, grad_V and hessian_V] & 1d traiining targets for GPR model.
    '''
    def __init__(self, ndofs: int , fixdofs: np.ndarray):
        self.ndofs = ndofs 
        self.fixdofs = fixdofs 

    def transform_pots_grad_hessian_to_1d_data(self, pots: np.ndarray, grad_V: np.ndarray, hessians: np.ndarray):
        '''
        transform the potential V, gradient of V and hessians into 1d target data.

        :param: pots: potential V. 1d data.
                grad_V: gradient of potential. 2d data. [N,d]
                hessians: hessian of potential. 3d data [N, d', d']
        '''
        grad_V_1d = grad_V.flatten()
        hess_triu_1d = take_upper_triangular_part(hessians).flatten()  # upper triangular part of hessian.
        train_targets = np.concatenate([pots, grad_V_1d, hess_triu_1d], axis= 0)
        
        return train_targets 

    def transform_1d_data_to_pots_grad_hessian(self, train_targets: np.ndarray, train_inputs: np.ndarray, hessian_data_point_index: np.ndarray):
        '''
        transform the 1d training targets to potential V, gradient dV/dx and hessian d^2 V/ dx^2 
        :param: train_targets: 1d training targets.
        :param: train_inputs: training input [N, d]. here N is number of data points, d is dof.
        :param: hessian_data_point_index: data point index for hessians.
        '''
        data_num = train_inputs.shape[0]
        ndofs = train_inputs.shape[1]
        assert ndofs == self.ndofs, "shape of training input is wrong."
        hessian_data_num = len(hessian_data_point_index)

        pots, gradients, hessians = transform_1d_train_targets_into_pots_grads_hessians(train_targets, data_num, ndofs, self.fixdofs, hessian_data_num)

        return pots, gradients, hessians 

class NormalizeTrainingData(object):
    '''
    normalize the potential, force and hessian of the training data.
    '''
    def __init__(self, V: np.ndarray):
        '''
        V_normalized = (V - V_mean)/V_range.

        :param: V: potential, 1d array
        '''
        self.V_mean = np.mean(V)
        self.V_range = np.max(V) - np.min(V)

    def normalization_transform(self, V: np.ndarray, grad_V: np.ndarray, hessian_V: np.ndarray):
        '''
        normalize the potential, gradients and hessians.
        V_normalized = (V - V_mean) / V_range. 
        grad_V_normalized = grad_V / V_range
        hessian_V_normalized = hessian_V / V_range 

        This function performs the normalization procedure
        :param: V: potential. 1d array.[N]
                grad_V: gradient of potential, 2d array. [N,d]
                hessian_V: hessian of potential, 3d array. [N,d,d]
                here N is number of data, d is total dof of the system.
        '''
        V_normalized = (V - self.V_mean) / self.V_range 
        grad_V_normalized = grad_V / self.V_range 
        hessian_V_normalized = hessian_V / self.V_range 

        return V_normalized, grad_V_normalized, hessian_V_normalized 
    
    def inverse_normalization_transform(self, V_normalized, grad_V_normalized, hessian_V_normalized):
        '''
        inverse the normalization procedure for potential V, gradients and hessians.

        V = V_normalized * V_range + V_mean
        F = F_normalized * V_range
        hessian = hessian_normalized * V_range 
        '''
        V = V_normalized * self.V_range + self.V_mean 
        grad_V = grad_V_normalized * self.V_range 
        hessian = hessian_V_normalized * self.V_range 

        return V, grad_V, hessian 
    
    def normalize_noise_var(self, pot_noise_var, force_noise_var, hessian_noise_var):
        '''
        normalize the variance of noise 
        '''
        normalized_pot_noise_var = pot_noise_var / np.power(self.V_range, 2)
        normalized_force_noise_var = force_noise_var / np.power(self.V_range, 2)
        normalized_hessian_noise_var = hessian_noise_var/ np.power(self.V_range, 2)

        return normalized_pot_noise_var, normalized_force_noise_var, normalized_hessian_noise_var
    
    def inverse_normalize_noise_var(self, normalized_pot_noise_var, normalized_force_noise_var, normalized_hessian_noise_var):
        '''
        inverse the normalization procedure for the variance of the noise
        '''
        pot_noise_var = normalized_pot_noise_var * np.power(self.V_range, 2)
        force_noise_var = normalized_force_noise_var * np.power(self.V_range, 2)
        hessian_noise_var = normalized_hessian_noise_var * np.power(self.V_range, 2)

        return pot_noise_var, force_noise_var, hessian_noise_var
    

class FixInternalDofs(object):
    '''
    class that fix certain internal dofs in the training data before feeding data into the Gaussian Process Regression model.
    '''
    def __init__(self, train_inputs: np.ndarray,  grads: np.ndarray):
        self.input_dim = train_inputs.shape[1]
        self.fix_internal_dofs_cutoff = np.power(10.0, -4)

        # check whether coordinate alng certain internal dofs need to be fixed.
        train_inputs_change = np.max(train_inputs, axis= 0) - np.min(train_inputs, axis= 0)
        self.fixed_internal_dofs = np.array([i for i in range(self.input_dim) if train_inputs_change[i] < self.fix_internal_dofs_cutoff])
        if len(self.fixed_internal_dofs) !=  0:
            self.free_moving_dofs = np.delete(np.arange(self.input_dim), self.fixed_internal_dofs)
            self.grads_fixed_dofs = np.mean(grads, axis= 0)[self.fixed_internal_dofs]
        else:
            self.free_moving_dofs = np.arange(self.input_dim)
            self.grads_fixed_dofs = np.array([])
        
    def transform_training_inputs_to_free_moving_dofs(self, train_inputs: np.ndarray):
        '''
        delete fixdofs from training inputs.
        :param: train_inputs: the training inputs in internal dofs. 
        '''
        moving_train_inputs= train_inputs[:, self.free_moving_dofs]
        return moving_train_inputs 
    
    def transform_training_targets_to_free_moving_dofs(self, grads: np.ndarray, hessians: np.ndarray):
        '''
        delete fixdofs data from training gradients and hessians.
        '''
        moving_grads = grads[:, self.free_moving_dofs]
        moving_hessians = hessians[:, self.free_moving_dofs, self.free_moving_dofs]

        return moving_grads, moving_hessians 
    
    def transform_from_free_moving_dofs_to_full_dofs(self, test_moving_grads, test_moving_hessians):
        '''
        Transform the prediction of the GPR model from free moving dofs into the full dofs 
        '''
        test_data_num = test_moving_grads.shape[0]

        # the prediction of fixed dofs for graidents in testing data is the mean value of fixed dofs for gradients in training data
        test_grads_fixed_dofs = np.repeat([self.grads_fixed_dofs], test_data_num, axis= 0)

        # the prediction of the gradient data in all dofs
        test_grads = np.zeros([test_data_num, self.input_dim])
        test_grads[:, self.free_moving_dofs] = test_moving_grads
        if len(self.fixed_internal_dofs) != 0: 
            test_grads[:,self.fixed_internal_dofs]  = test_grads_fixed_dofs

        # the prediction of hessian data in all dofs
        test_hessians = np.zeros([test_data_num, self.input_dim, self.input_dim])
        test_hessians[:, self.free_moving_dofs, self.free_moving_dofs] = test_moving_hessians 

        return test_grads, test_hessians 

class GPModelWithHessiansWrapper():
    '''
    wrapper class for GPModelWithHessians
    handles the transformation between internal coordinate and Cartesian coordinate + GPR training.
    This code will use training data with potentials, forces and hessians, then predict Hessians.
    I wrote addition codes (See utils/gprHessian) to extend the gpytorch packages, so we can predict Hessians.
    '''
    def __init__(self, train_x: np.ndarray , train_V: np.ndarray, train_grad_x: np.ndarray, 
                 train_hessians_x: np.ndarray, training_data_hessian_data_point_index: np.ndarray,
                 natom: int, 
                 coordinate_transformer: non_redundant_coordinate_transformer,
                 gpr_SE_kernel_number: int, kernel_outputscale: np.ndarray, kernel_lengthscale_ratio: np.ndarray,
                 noise_std):
        '''
        :param: train_x: [M, 3 * natom]. initial M training points x in Cartesian coordinate.
        :param: train_V: [M]. initial N training potential V. 
        :param: train_grad_x: [M, 3 * natom]. initial M training gradients in Cartesian coordinate. 
        :param: train_hessians_x: [M_H, nactive, nactive].  hessians of initial M_H training data. (nactive is number of active dims.)
        :param: training_data_hessian_data_point_index: index of M_H data points that have hessians. 
        :param: natom: number of atoms.
        :param: coordinate_transformer: an instance of class: non_redundnat_coordinate_transformer. Responsible for transformation between external and internal dofs.
        :param: gpr_SE_kernel_number: number of squared exponential kernels that is used to construct the covariance function.
        :param: kernel_outputscale: output scale of each squared exponential kernel used to construct covariance function.
        :param: kernel_lengthscale_ratio: length scale ratio of each squared exponential kernel used to construct covariance function. numpy array.
        :param: noise_std: the noise of likelihood function p(y|f).  y = f + epsilon.  Note the potential V, force f and hessian H have different noise. 
                           The noise for force and hessian is defined in Cartesian coordinate. We need to transform it into the internal coordinate.
        '''
        M_H = len(training_data_hessian_data_point_index)
        hessian_fixdofs = np.array([])
        assert np.shape(train_x)[1] == 3 * natom, "dim of coordinates for input data is not 3 * natom, this is wrong. train_x data shape: {} , 3 * natom: {}".format(np.reshape(train_x)[1], 3 * natom)
        assert np.shape(train_grad_x)[1] == 3 * natom, "dim of gradients for input data is not 3 * natom, this is wrong. train_grad shape:{}, 3 * natom: {}".format(np.shape(train_grad_x)[1], 3 * natom)
        assert np.shape(train_hessians_x)[0] == M_H, "number of data points (M_H) with hessian information is not consistent with training_data_hessian_data_point_index. M_H from train_hessians: {}, M_H from hessian_data_point_index: {}".format(np.shape(train_hessians_x)[0], M_H)

        self.natom = natom
        self.gpr_SE_kernel_number = gpr_SE_kernel_number
        self.coordinate_transformer = coordinate_transformer

        # record the potential, gradient and hessians in Cartesian coordinate.
        self.train_V = np.copy(train_V)
        self.train_cartesian_gradient = np.copy(train_grad_x)
        self.train_cartesian_hessian = np.copy(train_hessians_x)
        self.training_data_hessian_data_point_index = np.copy(training_data_hessian_data_point_index)
        self.train_cartesian_input = np.copy(train_x)

        # transform the cartesian coordinate x to internal coordinate q 
        train_inputs = coordinate_transformer.get_internal_coordinate_q(train_x)

        input_dim = np.shape(train_inputs)[1]
        self.input_dim = input_dim 

        # transform the gradient of potential V: dV/dx -> dV/dq 
        train_grad_q = coordinate_transformer.transform_cartesian_gradient_to_internal_gradient(train_x, train_grad_x)
        # transform the hessian of potential V: d^2 V/ dx^2 -> d^2 V/ dq^2 
        train_hessians_q = coordinate_transformer.transform_cartesian_hessian_to_internal_hessian(train_x[training_data_hessian_data_point_index], train_grad_x[training_data_hessian_data_point_index], train_hessians_x)

        # record the training inputs and target in internal coordinate space.
        self.train_inputs = train_inputs 
        self.train_grad_q = train_grad_q
        self.train_hessian_q = train_hessians_q 

        # Transform the noise from Cartesian dofs into internal dofs.
        pot_noise_var, force_noise_var, hessian_noise_var = self.transform_cartesian_noise_to_gpr_model_noise(noise_std)
        
        # Normalize 
        self.Normalizer = NormalizeTrainingData(train_V)
        normalized_train_V, normalized_train_grad_q, normalized_train_hessians_q = self.Normalizer.normalization_transform(train_V, train_grad_q, train_hessians_q)
        pot_noise_var, force_noise_var, hessian_noise_var = self.Normalizer.normalize_noise_var(pot_noise_var, force_noise_var, hessian_noise_var)
        
        # Filter the fixed dofs.
        self.FixingDofs = FixInternalDofs(train_inputs, normalized_train_grad_q)
        moving_train_inputs = self.FixingDofs.transform_training_inputs_to_free_moving_dofs(train_inputs)
        moving_normalized_train_grad_q, moving_normalized_train_hessian_q = self.FixingDofs.transform_training_targets_to_free_moving_dofs(normalized_train_grad_q, normalized_train_hessians_q)
        force_noise_var, hessian_noise_var = self.FixingDofs.transform_training_targets_to_free_moving_dofs(force_noise_var, hessian_noise_var)  # filter the fixed dofs for force and hessian noise.

        # transform pots, gradients and hessisans in to 1d data. After normalize the training data and exclude fixed dof in gradient and hessian data.
        free_moving_input_dims = len(self.FixingDofs.free_moving_dofs)
        self.TargetDataTransformer = TransformTrainingTarget(free_moving_input_dims, hessian_fixdofs)
        train_targets = self.TargetDataTransformer.transform_pots_grad_hessian_to_1d_data(normalized_train_V, moving_normalized_train_grad_q, moving_normalized_train_hessian_q)
        hessian_noise_var = take_upper_triangular_part(hessian_noise_var)  # take upper triangular part of the hessian noise

        # Transform the numpy array to tensor 
        moving_train_inputs_tensor = torch.from_numpy(moving_train_inputs)
        hessian_data_point_index_tensor = torch.from_numpy(training_data_hessian_data_point_index)
        train_targets_tensor = torch.from_numpy(train_targets)
        hessian_fixdofs_tensor = torch.tensor([])

        # initialize the gaussian process regression model with input training data. 
        # GPModelWithHessians are Gaussian Process Regression model that capable of using hessian as training data and also predicting hessians.
        # It transforms the potential, force & hessian into 1d data set. See eq.(9) in J. Chem. Theory Comput. 2024, 20, 3766−3778
        self.gpr_model = GPModelWithHessians(moving_train_inputs_tensor, train_targets_tensor,
                                             hessian_data_point_index_tensor, hessian_fixdofs_tensor,
                                             gpr_SE_kernel_number,
                                             kernel_outputscale, kernel_lengthscale_ratio,
                                             pot_noise_var, force_noise_var, hessian_noise_var)
        
        # train the gaussian process regression model.
        train_gpr_model(self.gpr_model)
        

    def transform_cartesian_noise_to_gpr_model_noise(self, noise_std):
        '''
        transform the noise in Cartesian coordinate to noise in Gaussian Process Regression model.
        This is critical to the successful training of GPR model, otherwise, the model will treat the noise incorrectly & training will fail.
        '''
        pot_noise_std = noise_std["pot_noise_prior"]
        force_noise_std_cartesian = noise_std["force_noise_prior"]
        hessian_noise_std_cartesian = noise_std["hessian_noise_prior"]

        # force noise and hessian noise has to be scaled by the inverse of the singular value of Wilson's B matrix.
        self.Bmatrix_singular_value_square = np.power(self.coordinate_transformer.ref_S, 2)
        singular_value_square_inverse = 1 / self.Bmatrix_singular_value_square  # S^{-2}. inverse and square of singular value of Wilson's B matrix.

        # compute the noise in internal coordinate.
        pot_noise_var = np.power(pot_noise_std, 2)
        force_noise_var = np.power(force_noise_std_cartesian, 2) * singular_value_square_inverse
        # take upper triangular part of hessian as hessian noise.
        hessian_noise_var = np.power(hessian_noise_std_cartesian, 2) * np.outer(singular_value_square_inverse, singular_value_square_inverse)  


        return pot_noise_var, force_noise_var, hessian_noise_var 

    def predict_latent_function(self, test_x: np.ndarray, test_hessian_data_point_index: np.ndarray, internal_coordinate_bool = False):
        '''
        compute the predicted potential V, gradients dV/dx and hessians d^2 V/dx^2 in Cartesian coordinate.
        Also compute the variance.
        This function wraps predict_latent_function_GPHessian in ./gprHessian/RBFHessian_gp.py. 
        This function mainly serves as wrapper function.
        This function handles 1. normalization  2. filter fixed dofs  3. transformation between Cartesian and internal dofs.

        :param: test_x: input Cartesian coordinate data [N, 3 * natom].
        :param: test_hessian_data_point_index: the index of data point in x that we need to predict the hessian information.
        :param: internal_coordinate_bool:  if internal_coordinate_bool = True, we output gradient and hessian in internal coordinate (Used for debugging).
                                               otherwise (default), we output the gradient and hessian in cartesian coordinate.
        '''
        assert np.shape(test_x)[1] == 3 * self.natom, "dim of coordinate for input data is not 3 * natom"
        test_data_num = np.shape(test_x)[0]

        # transform the input data into internal coordinate.
        moving_test_q = self.get_free_moving_internal_coordinate(test_x)
        moving_test_q_tensor = torch.from_numpy(moving_test_q)

        # use Gaussian process regression model to make prediction
        pots, moving_grads_q, moving_hessians_q, pots_var, moving_grads_q_var, moving_hessians_q_var = predict_latent_function_GPHessian(self.gpr_model, moving_test_q_tensor,
                                                                                                                                               test_hessian_data_point_index)
        # back transform the mean value and variance from free moving dofs into full dofs 
        grads_q, hessians_q = self.FixingDofs.transform_from_free_moving_dofs_to_full_dofs(moving_grads_q, moving_hessians_q)
        grads_q_var, hessians_q_var = self.FixingDofs.transform_from_free_moving_dofs_to_full_dofs(moving_grads_q_var, moving_hessians_q_var)

        # inverse the normalization procedure for mean value and variance.
        pots, grads_q, hessians_q = self.Normalizer.inverse_normalization_transform(pots, grads_q, hessians_q)
        pots_var, grads_q_var, hessians_q_var = self.Normalizer.inverse_normalize_noise_var(pots_var, grads_q_var, hessians_q_var)

        # transform the gradient and hessian from internal coordinate to Cartesian coordinate.
        grads_x = self.coordinate_transformer.transform_internal_gradient_to_cartesian_gradient(test_x, grads_q)
        hessians_x = self.coordinate_transformer.transform_internal_hessian_to_cartesian_hessian(test_x, grads_q, hessians_q)

        # handle the variance of the gradient & hessians. 
        # the trace of covariance matirx of gradients. This characterize the uncertainty of the force.
        grads_x_var_sum = np.sum( self.Bmatrix_singular_value_square * grads_q_var, axis= 1)
        # compute the sum of variance of hessian element. This will serve as uncertainty of hessians.
        hessian_var_sum = np.sum(np.sum(hessians_q_var * np.outer(self.Bmatrix_singular_value_square, self.Bmatrix_singular_value_square),axis= -1), axis= -1)

        if internal_coordinate_bool:
            return pots, grads_q, hessians_q, pots_var, grads_x_var_sum, hessian_var_sum 
        else:
            return pots, grads_x, hessians_x, pots_var, grads_x_var_sum, hessian_var_sum 
        
    def update_model_with_new_data(self, new_train_x, new_train_V, new_train_grad_x, new_train_hessian_x, new_hessian_data_point_index):
        '''
        '''

    def get_free_moving_internal_coordinate(self, beads_x):
        '''
        transform from Cartesian coordinate x to the free moving internal coordinates q.
        '''
        beads_internal_coordinate = self.coordinate_transformer.get_internal_coordinate_q(beads_x)

        free_moving_beads_internal_coordinate = self.FixingDofs.transform_training_inputs_to_free_moving_dofs(beads_internal_coordinate)

        return free_moving_beads_internal_coordinate

        

