'''
packages for predicting hessian of ring polymer beads using Gaussian Process Regression (GPR).
The GPR part use GPytorch framework (https://docs.gpytorch.ai/en/stable/)
Written by Chenghao Zhang, Pacific Northwest National Laboratory (chenghao.zhang@pnnl.gov)
'''
import torch 
import numpy as np 
import gpytorch 
from ipi.utils.internalcoordtools import non_redundant_coordinate_transformer
from .gprHessian.RBFHessian_gp import GPModelWithHessians, train_gpr_model
from .gprHessian.RBFHessian_utils import take_upper_triangular_part, transform_1d_train_targets_into_pots_grads_hessians
import ipi.utils.gprHessian.RBFHessian_gp


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
    def __init__(self, train_inputs: np.ndarray,  grads: np.ndarray, hessians: np.ndarray):
        self.input_dim = train_inputs.shape[1]
        self.fix_internal_dofs_cutoff = np.power(10.0, -4)

        # check whether coordinate alng certain internal dofs need to be fixed.
        train_inputs_change = np.max(train_inputs, axis= 0) - np.min(train_inputs, axis= 0)
        self.fixed_internal_dofs = np.array([i for i in range(self.input_dim) if train_inputs_change[i] < self.fix_internal_dofs_cutoff])
        if len(self.fixed_internal_dofs) !=  0:
            self.free_moving_dofs = np.delete(np.arange(self.input_dim), self.fixed_internal_dofs)
            self.free_moving_dofs_2d_index = np.meshgrid(self.free_moving_dofs, self.free_moving_dofs, indexing= 'ij')
            self.grads_fixed_dofs = np.mean(grads, axis= 0)[self.fixed_internal_dofs]
            if len(hessians) != 0:
                self.hessians_fixed_dofs = np.mean(hessians, axis= 0)
                self.hessians_fixed_dofs[self.free_moving_dofs_2d_index[0], self.free_moving_dofs_2d_index[1]] = 0
            else:
                self.hessians_fixed_dofs = np.array([])
        else:
            self.free_moving_dofs = np.arange(self.input_dim)
            self.free_moving_dofs_2d_index = np.meshgrid(self.free_moving_dofs, self.free_moving_dofs, indexing= 'ij')
            self.grads_fixed_dofs = np.array([])
            self.hessians_fixed_dofs = np.array([])
        
    def update_hessians_fixed_dofs(self, new_hessian_data):
        if len(self.fixed_internal_dofs) != 0 and len(new_hessian_data) > 0:
            self.hessians_fixed_dofs = np.mean(new_hessian_data, axis= 0)
            self.hessians_fixed_dofs[self.free_moving_dofs_2d_index[0], 
                                    self.free_moving_dofs_2d_index[1]] = 0
        
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
        if len(hessians) > 0:
            index_2d = self.free_moving_dofs_2d_index
            moving_hessians = hessians[:, index_2d[0], index_2d[1]]
        else:
            moving_hessians = hessians 

        return moving_grads, moving_hessians 

    def transform_noise_covar_factor_fixing_internal_dofs(self, noise_covar_factor, with_hessian_bool= False):
        '''
        delete rows corresponding to gradient and hessians of fixed dofs.
        '''
        input_dim = self.input_dim
        hessian_triu_size = int((input_dim + 1) * input_dim / 2) 
        if len(self.fixed_internal_dofs) != 0:
            row_to_delete_grad = 1 + np.array(self.fixed_internal_dofs)

            if not with_hessian_bool:
                row_to_delete = row_to_delete_grad
                noise_covar_factor = np.delete(noise_covar_factor, row_to_delete, axis= 0)
                return noise_covar_factor
            else:
                # find the upper triangle index that we need to delete 
                upper_triangle_index_matrix = np.zeros([input_dim, input_dim])
                for i in range(input_dim):
                    for j in range(i, input_dim):
                        upper_triangle_index_matrix[i,j] = i * (input_dim - (1 + i) / 2) + j 
                        upper_triangle_index_matrix[j,i] = upper_triangle_index_matrix[i,j]
                
                upper_triangle_index_matrix_free_moving = upper_triangle_index_matrix[self.free_moving_dofs_2d_index[0], self.free_moving_dofs_2d_index[1]]
                upper_triangle_index_matrix_free_moving = take_upper_triangular_part(upper_triangle_index_matrix_free_moving)
                upper_triangle_index_matrix_free_moving = np.vectorize(int)(upper_triangle_index_matrix_free_moving)
                fixed_hessian_triu_index = np.delete(np.arange(hessian_triu_size), upper_triangle_index_matrix_free_moving)
                row_to_delete_hessian_triu_index = fixed_hessian_triu_index + 1 + input_dim 

                # delete rows that corresponds to gradient and hessian of fixed dofs.
                row_to_delete = np.concatenate([row_to_delete_grad, row_to_delete_hessian_triu_index])
                noise_covar_factor = np.delete(noise_covar_factor, row_to_delete, axis= 0)
                return noise_covar_factor
        else:
            return noise_covar_factor
        
    def transform_noise_covar_factor_array_fixing_internal_dofs(self, noise_covar_factor_pot_grad_array, noise_covar_factor_with_hessian_array):
        '''
        '''
        pot_grad_array_len = len(noise_covar_factor_pot_grad_array)
        hessian_len = len(noise_covar_factor_with_hessian_array)

        noise_covar_factor_pot_grad_array_new = []
        for i in range(pot_grad_array_len):
            new_covar_factor = self.transform_noise_covar_factor_fixing_internal_dofs(noise_covar_factor_pot_grad_array[i], with_hessian_bool= False)
            noise_covar_factor_pot_grad_array_new.append(new_covar_factor)
        noise_covar_factor_pot_grad_array_new = np.array(noise_covar_factor_pot_grad_array_new)

        noise_covar_factor_with_hessian_array_new = []
        for i in range(hessian_len):
            new_covar_factor = self.transform_noise_covar_factor_fixing_internal_dofs(noise_covar_factor_with_hessian_array[i], with_hessian_bool= True)
            noise_covar_factor_with_hessian_array_new.append(new_covar_factor)
        noise_covar_factor_with_hessian_array_new = np.array(noise_covar_factor_with_hessian_array_new)

        return noise_covar_factor_pot_grad_array_new, noise_covar_factor_with_hessian_array_new

    
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

        if len(test_moving_hessians) > 0:
            if len(self.fixed_internal_dofs) !=  0:
                test_data_with_hessian_num = test_moving_hessians.shape[0]
                # the prediction of hessian data in all dofs
                if len(self.hessians_fixed_dofs) != 0:
                    test_hessians = np.repeat([self.hessians_fixed_dofs], test_data_with_hessian_num, axis= 0)
                else:
                    test_hessians = np.zeros([test_data_with_hessian_num, self.input_dim, self.input_dim])
                index_2d = self.free_moving_dofs_2d_index
                test_hessians[:, index_2d[0], index_2d[1]] = test_moving_hessians 
            else: 
                test_hessians = test_moving_hessians
        else:
            test_hessians = torch.Tensor([])

        return test_grads, test_hessians 

class GPModelWithHessiansWrapper():
    '''
    wrapper class for GPModelWithHessians
    handles the transformation between internal coordinate and Cartesian coordinate + GPR training.
    This code will use training data with potentials, forces and hessians, then predict Hessians.
    I wrote addition codes (See utils/gprHessian) to extend the gpytorch packages, so we can predict Hessians.
    '''
    def __init__(self, train_x: np.ndarray , train_V: np.ndarray, train_grad_x: np.ndarray, 
                 train_hessian_x: np.ndarray, training_data_hessian_data_point_index_array: np.ndarray,
                 natom: int, 
                 coordinate_transformer: non_redundant_coordinate_transformer,
                 gpr_SE_kernel_number: int, kernel_outputscale: np.ndarray, kernel_lengthscale_ratio: np.ndarray,
                 noise_std, 
                 kernel_lengthscale_initio_value: np.ndarray= np.array([]),
                 kernel_outputscale_initio_value: np.ndarray= np.array([]),
                 constant_mean_bool= True,
                 ref_mean_x: np.ndarray= np.array([]), 
                 ref_mean_V: np.ndarray= np.array([]),
                 ref_mean_grad_x: np.ndarray= np.array([]),
                 ref_mean_hessian_x: np.ndarray= np.array([])):
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
        M_H = len(training_data_hessian_data_point_index_array)
        hessian_fixdofs = np.array([])
        assert np.shape(train_x)[1] == 3 * natom, "dim of coordinates for input data is not 3 * natom, this is wrong. train_x data shape: {} , 3 * natom: {}".format(np.reshape(train_x)[1], 3 * natom)
        assert np.shape(train_grad_x)[1] == 3 * natom, "dim of gradients for input data is not 3 * natom, this is wrong. train_grad shape:{}, 3 * natom: {}".format(np.shape(train_grad_x)[1], 3 * natom)
        assert np.shape(train_hessian_x)[0] == M_H, "number of data points (M_H) with hessian information is not consistent with training_data_hessian_data_point_index. M_H from train_hessians: {}, M_H from hessian_data_point_index: {}".format(np.shape(train_hessian_x)[0], M_H)

        self.natom = natom
        self.gpr_SE_kernel_number = gpr_SE_kernel_number
        self.coordinate_transformer = coordinate_transformer

        # symmetric the hessian
        if len(train_hessian_x) > 0:
            train_hessian_x_symmetrized = (np.transpose(train_hessian_x, (0, 2, 1)) + train_hessian_x) / 2
        else:
            train_hessian_x_symmetrized = train_hessian_x

        # record the potential, gradient and hessians in Cartesian coordinate.
        self.train_V = np.copy(train_V)
        self.train_cartesian_gradient = np.copy(train_grad_x)
        self.train_cartesian_hessian = np.copy(train_hessian_x_symmetrized)
        self.training_data_hessian_data_point_index = np.copy(training_data_hessian_data_point_index_array)
        self.train_cartesian_input = np.copy(train_x)

        # transform the cartesian coordinate x to internal coordinate q 
        train_inputs = coordinate_transformer.get_internal_coordinate_q(train_x)

        input_dim = np.shape(train_inputs)[1]
        self.input_dim = input_dim 

        # transform the gradient of potential V: dV/dx -> dV/dq 
        train_grad_q = coordinate_transformer.transform_cartesian_gradient_to_internal_gradient(train_x, train_grad_x)
        # transform the hessian of potential V: d^2 V/ dx^2 -> d^2 V/ dq^2 
        if len(training_data_hessian_data_point_index_array) > 0:
            train_hessian_q = coordinate_transformer.transform_cartesian_hessian_to_internal_hessian(train_x[training_data_hessian_data_point_index_array],
                                                                                                    train_grad_x[training_data_hessian_data_point_index_array],
                                                                                                    train_hessian_x_symmetrized)
        else:
            train_hessian_q = np.array([])

        # record the training inputs and target in internal coordinate space.
        self.train_inputs = train_inputs 
        self.train_grad_q = train_grad_q
        self.train_hessian_q = train_hessian_q 

        # Transform the noise from Cartesian dofs into internal dofs.
        noise_covar_factor_pot_grad_array, noise_covar_factor_with_hessian_array = self.compute_noise_covar_factor_array(train_x, train_inputs, training_data_hessian_data_point_index_array)
        force_noise_rank = 3 * natom 
        hessian_noise_rank = int((3 * natom) * (3 * natom + 1) / 2)

        # Normalize 
        self.Normalizer = NormalizeTrainingData(train_V)
        normalized_train_V, normalized_train_grad_q, normalized_train_hessians_q = self.Normalizer.normalization_transform(train_V, train_grad_q, train_hessian_q)

        # Filter the fixed dofs.
        self.FixingDofs = FixInternalDofs(train_inputs, normalized_train_grad_q, normalized_train_hessians_q)
        moving_train_inputs = self.FixingDofs.transform_training_inputs_to_free_moving_dofs(train_inputs)
        moving_normalized_train_grad_q, moving_normalized_train_hessian_q = self.FixingDofs.transform_training_targets_to_free_moving_dofs(normalized_train_grad_q, normalized_train_hessians_q)
        
        # fix noise_covar_factor_array.
        noise_covar_factor_pot_grad_array, noise_covar_factor_with_hessian_array = self.FixingDofs.transform_noise_covar_factor_array_fixing_internal_dofs(noise_covar_factor_pot_grad_array, noise_covar_factor_with_hessian_array)
        
        # set variance of noise.
        pot_noise_var, force_noise_var, hessian_noise_var = self.compute_noise_var(noise_std)
        pot_noise_var, force_noise_var, hessian_noise_var = self.Normalizer.normalize_noise_var(pot_noise_var, force_noise_var, hessian_noise_var)
        
        # transform pots, gradients and hessisans in to 1d data. After normalize the training data and exclude fixed dof in gradient and hessian data.
        free_moving_input_dims = len(self.FixingDofs.free_moving_dofs)
        self.TargetDataTransformer = TransformTrainingTarget(free_moving_input_dims, hessian_fixdofs)
        train_targets = self.TargetDataTransformer.transform_pots_grad_hessian_to_1d_data(normalized_train_V, moving_normalized_train_grad_q, moving_normalized_train_hessian_q)
        
        # Transform the numpy array to tensor 
        moving_train_inputs_tensor = torch.from_numpy(moving_train_inputs)
        hessian_data_point_index_tensor = torch.from_numpy(training_data_hessian_data_point_index_array)
        train_targets_tensor = torch.from_numpy(train_targets)
        hessian_fixdofs_tensor = torch.tensor([])
        noise_covar_factor_pot_grad_array = torch.from_numpy(noise_covar_factor_pot_grad_array)
        noise_covar_factor_with_hessian_array = torch.from_numpy(noise_covar_factor_with_hessian_array)

        # set the mean function as the pot, gradient and hessian at a given reference point 
        ref_mean_q_tensor, ref_mean_V_tensor, ref_mean_grad_q_tensor, ref_mean_hessian_q_tensor = self.compute_mean_function_param(ref_mean_x, ref_mean_V, ref_mean_grad_x, ref_mean_hessian_x)
 
        # initialize the gaussian process regression model with input training data. 
        # GPModelWithHessians are Gaussian Process Regression model that capable of using hessian as training data and also predicting hessians.
        # It transforms the potential, force & hessian into 1d data set. See eq.(9) in J. Chem. Theory Comput. 2024, 20, 3766−3778
        self.gpr_model = GPModelWithHessians(moving_train_inputs_tensor, train_targets_tensor,
                                             hessian_data_point_index_tensor, hessian_fixdofs_tensor,
                                             gpr_SE_kernel_number,
                                             kernel_outputscale, kernel_lengthscale_ratio,
                                             pot_noise_var, force_noise_var, hessian_noise_var,
                                             force_noise_rank, hessian_noise_rank, 
                                             noise_covar_factor_pot_grad_array, noise_covar_factor_with_hessian_array,
                                             kernel_lengthscale_initio_value,
                                             kernel_outputscale_initio_value,
                                             constant_mean_bool,
                                             ref_mean_q_tensor, ref_mean_V_tensor, ref_mean_grad_q_tensor, ref_mean_hessian_q_tensor)
        
        # train the gaussian process regression model.
        ipi.utils.gprHessian.RBFHessian_gp.train_gpr_model(self.gpr_model)
        

    def compute_noise_var(self, noise_std):
        '''
        Compute the covar factor for RBFHessian_gaussian_likelihood.
        The covariance factor will transform the noise in Cartesian coordinate to noise in internal coordinate.
        This is critical to the successful training of GPR model, otherwise, the model will treat the noise incorrectly & training will fail.
        '''
        pot_noise_std = noise_std["pot_noise_prior"]
        force_noise_std_cartesian = noise_std["force_noise_prior"]
        hessian_noise_std_cartesian = noise_std["hessian_noise_prior"]

        # variance of pot noise, force noise and hessian noise in Cartesian coordinate.
        x_size = 3 * self.natom
        hessian_x_triu_size = int(x_size * (x_size + 1) / 2)

        pot_noise_var = np.array([np.power(pot_noise_std, 2)])
        force_noise_var = np.ones([1]) * np.power(force_noise_std_cartesian, 2)
        hessian_noise_var = np.ones([1]) * np.power(hessian_noise_std_cartesian, 2)

        self.Bmatrix_singular_value_square = np.power(self.coordinate_transformer.ref_S, 2)
        # Need to consider the covariance between gradient and hessian & also covariance of hessian itself.
        # See J. Chem. Theory Comput. 2024, 20, 3766−3778  eq.(13). We need back transformation of noise matrix into internal coordinate. 
        # The noise matrix transform like covariance matrix K, see eq.(17). To correctly treat this problem, 
        # you either transform covariance matrix from internal coordinate into internal coordinate.
        # Or transform potential, force, Hessian and noise matrix into internal coordinate.
        # compute the noise_covar_factor 
        

        return pot_noise_var, force_noise_var, hessian_noise_var
        

    def compute_noise_covar_factor_for_each_data_point(self, x, with_hessian_bool):
        '''
        compute the covariance factor for noise transformation for each data point x. 
        '''    
        # covar_factor [2, 2] term.  inverse transpose of Wilson's B matrix
        B = self.coordinate_transformer._compute_redundant_gradient_matrix_B(np.array([x]))[0]
        # \partial q / \partial x. shape [3n - 6, 3n]
        Bq = np.matmul(self.coordinate_transformer.ref_UT, B)
        # \partial x / \partial q. 
        inverse_Bq_transpose = np.transpose(np.linalg.pinv(Bq, rcond= np.power(10.0, -8)), (1,0))

        q_size = Bq.shape[0]
        x_size = Bq.shape[1]
        hessian_q_triu_size = int((q_size * (q_size + 1)) / 2)
        hessian_x_triu_size = int((x_size * (x_size + 1)) / 2) 

        # covar factor [3,2] d^2 x/ dq^2 
        hessian_x_qq = self.coordinate_transformer.compute_x_hessian_q(x)  # d^2 x / dq^2. shape:[3n, 3n-6, 3n-6]
        hessian_x_qq_up_triangle = np.transpose(take_upper_triangular_part(hessian_x_qq), (1,0)) # shape: [(3n-6)(3n-5) / 2, 3n]

        # covar factor [3,3] term. d^2 x/ dq^2 
        inverse_Bq_transpose_tensor = np.transpose(np.tensordot(inverse_Bq_transpose,inverse_Bq_transpose, axes= 0), (0, 2, 1, 3))
        inverse_Bq_transpose_tensor_diag = np.zeros(inverse_Bq_transpose_tensor.shape)
        inverse_Bq_transpose_tensor_diag[..., np.arange(x_size), np.arange(x_size)] = np.diagonal(inverse_Bq_transpose_tensor, axis1= 2, axis2= 3)
        covar_33 = inverse_Bq_transpose_tensor + np.transpose(inverse_Bq_transpose_tensor, (0, 1, 3, 2)) - inverse_Bq_transpose_tensor_diag
        # take upper triangular part
        covar_33 = take_upper_triangular_part(covar_33)
        covar_33 = np.transpose(take_upper_triangular_part(np.transpose(covar_33, (2, 0, 1))), (1, 0)) 

        row_size = 1 + q_size + hessian_q_triu_size
        col_size = 1 + x_size + hessian_x_triu_size   

        # transformation matrix for covariance matrix of noise in internal coordinate (q) and Cartesian coordinate (x) 
        noise_covar_factor = np.zeros([row_size, col_size])
        # potential part
        noise_covar_factor[0, 0] = 1
        # grad part
        grad_index_2d = np.meshgrid(1 + np.arange(q_size), 1 + np.arange(x_size) , indexing= 'ij')
        noise_covar_factor[grad_index_2d[0], grad_index_2d[1]] = inverse_Bq_transpose

        # grad- hessian covariance part 
        grad_hessian_covar_index_2d = np.meshgrid(1 + q_size + np.arange(hessian_q_triu_size), 1 + np.arange(x_size), indexing= 'ij')
        noise_covar_factor[grad_hessian_covar_index_2d[0], grad_hessian_covar_index_2d[1]] = hessian_x_qq_up_triangle

        # hessian covariance part
        hessian_covar_index_2d = np.meshgrid(1 + q_size + np.arange(hessian_q_triu_size), 1 + x_size + np.arange(hessian_x_triu_size), indexing= 'ij')
        noise_covar_factor[hessian_covar_index_2d[0], hessian_covar_index_2d[1]] = covar_33 

        if with_hessian_bool:
            return noise_covar_factor
        else:
            noise_covar_factor = noise_covar_factor[: 1 + q_size, : 1 + x_size]
            return noise_covar_factor
         

    def compute_noise_covar_factor_array(self, train_x, train_inputs, training_data_hessian_data_point_index_array):
        '''
        compute covariate factor for different training inputs data.
        '''
        training_data_num = train_x.shape[0]
        x_size = train_x.shape[1]
        q_size = train_inputs.shape[1]

        noise_covar_factor_pot_grad_array = []  # covariance factor for only potential and gradient
        noise_covar_factor_with_hessian_array = []  # covariance factor including hessian

        for data_point_index in range(training_data_num):
            noise_covar_factor = self.compute_noise_covar_factor_for_each_data_point(train_x[data_point_index], with_hessian_bool= False)
            noise_covar_factor_pot_grad_array.append(noise_covar_factor)
        
        noise_covar_factor_pot_grad_array = np.array(noise_covar_factor_pot_grad_array)
        
        for hessian_data_point_index in training_data_hessian_data_point_index_array:
            noise_covar_factor = self.compute_noise_covar_factor_for_each_data_point(train_x[hessian_data_point_index], with_hessian_bool= True)
            noise_covar_factor_with_hessian_array.append(noise_covar_factor)

        noise_covar_factor_with_hessian_array = np.array(noise_covar_factor_with_hessian_array)

        return noise_covar_factor_pot_grad_array, noise_covar_factor_with_hessian_array

    def compute_mean_function_param(self, ref_mean_x, ref_mean_V, ref_mean_grad_x, ref_mean_hessian_x ):
        '''
        compute the internal coordinate q of reference point & pot V, grad and hessian of reference point.
        '''    
        # set the mean function as the pot, grad and hessian at reference point
        if len(ref_mean_x) != 0:
            self.ref_mean_x = ref_mean_x 
            self.ref_mean_V = ref_mean_V
            self.ref_mean_grad_x = ref_mean_grad_x
            # symmetrize hessian:
            ref_mean_hessian_x = (ref_mean_hessian_x + np.transpose(ref_mean_hessian_x, (1,0))) / 2
            self.ref_mean_hessian_x = ref_mean_hessian_x

            self.ref_mean_q = self.coordinate_transformer.get_internal_coordinate_q(np.array([ref_mean_x]))[0]
            
            ref_mean_grad_q = self.coordinate_transformer.transform_cartesian_gradient_to_internal_gradient(np.array([ref_mean_x]), np.array([ref_mean_grad_x]))[0]
            
            ref_mean_hessian_q = self.coordinate_transformer.transform_cartesian_hessian_to_internal_hessian(np.array([ref_mean_x]), np.array([ref_mean_grad_x]),
                                                                                                             np.array([ref_mean_hessian_x]))[0]
            
            self.ref_mean_grad_q = ref_mean_grad_q 
            self.ref_mean_hessian_q = ref_mean_hessian_q  

            # Normalize 
            ref_mean_V, ref_mean_grad_q, ref_mean_hessian_q = self.Normalizer.normalization_transform(ref_mean_V, ref_mean_grad_q, ref_mean_hessian_q)
            
            # Filter 
            ref_mean_q = self.FixingDofs.transform_training_inputs_to_free_moving_dofs(np.array([self.ref_mean_q]))[0]
            ref_mean_grad_q, ref_mean_hessian_q = self.FixingDofs.transform_training_targets_to_free_moving_dofs(np.array([ref_mean_grad_q]), np.array([ref_mean_hessian_q]))
            ref_mean_grad_q = ref_mean_grad_q[0]
            ref_mean_hessian_q = ref_mean_hessian_q[0]

            ref_mean_hessian_q = take_upper_triangular_part(ref_mean_hessian_q)

            ref_mean_q_tensor = torch.tensor(ref_mean_q)
            ref_mean_V_tensor = torch.tensor(ref_mean_V)
            ref_mean_grad_q_tensor = torch.tensor(ref_mean_grad_q)
            ref_mean_hessian_q_tensor = torch.tensor(ref_mean_hessian_q)

        else:
            ref_mean_q_tensor = torch.tensor([])
            ref_mean_V_tensor = torch.tensor([])
            ref_mean_grad_q_tensor = torch.tensor([])
            ref_mean_hessian_q_tensor = torch.tensor([])

        return ref_mean_q_tensor, ref_mean_V_tensor, ref_mean_grad_q_tensor, ref_mean_hessian_q_tensor 



    def predict_latent_function(self, test_x: np.ndarray, 
                                test_hessian_data_point_index: np.ndarray, internal_coordinate_bool = False):
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
        pots, moving_grads_q, moving_hessians_q, pots_var, moving_grads_q_var, moving_hessians_q_var = ipi.utils.gprHessian.RBFHessian_gp.predict_latent_function_GPHessian(self.gpr_model, moving_test_q_tensor,
                                                                                                                                               test_hessian_data_point_index)
        # back transform the mean value and variance from free moving dofs into full dofs 
        grads_q, hessians_q = self.FixingDofs.transform_from_free_moving_dofs_to_full_dofs(moving_grads_q, moving_hessians_q)
        grads_q_var, hessians_q_var = self.FixingDofs.transform_from_free_moving_dofs_to_full_dofs(moving_grads_q_var, moving_hessians_q_var)

        # inverse the normalization procedure for mean value and variance.
        pots, grads_q, hessians_q = self.Normalizer.inverse_normalization_transform(pots, grads_q, hessians_q)
        pots_var, grads_q_var, hessians_q_var = self.Normalizer.inverse_normalize_noise_var(pots_var, grads_q_var, hessians_q_var)

        # transform the gradient and hessian from internal coordinate to Cartesian coordinate.
        grads_x = self.coordinate_transformer.transform_internal_gradient_to_cartesian_gradient(test_x, grads_q)
        if len(test_hessian_data_point_index) > 0:
            hessians_x = self.coordinate_transformer.transform_internal_hessian_to_cartesian_hessian(test_x[test_hessian_data_point_index], grads_q[test_hessian_data_point_index], hessians_q)
        else:
            hessians_x = torch.tensor([])

        # handle the variance of the gradient & hessians. 
        # the trace of covariance matirx of gradients. This characterize the uncertainty of the force.
        grads_x_var_sum = np.sum( self.Bmatrix_singular_value_square * grads_q_var, axis= 1)
        if len(hessians_q_var) > 0:
            # compute the sum of variance of hessian element. This will serve as uncertainty of hessians.
            hessian_var_sum = np.sum(np.sum(hessians_q_var * np.outer(self.Bmatrix_singular_value_square, self.Bmatrix_singular_value_square),axis= -1), axis= -1)
        else:
            hessian_var_sum = np.array([])

        if internal_coordinate_bool:
            return pots, grads_q, hessians_q, pots_var, grads_x_var_sum, hessian_var_sum 
        else:
            return pots, grads_x, hessians_x, pots_var, grads_x_var_sum, hessian_var_sum 
        
    def update_model_with_new_data(self, new_train_x: np.ndarray, new_train_V: np.ndarray, new_train_grad_x: np.ndarray, 
                                   new_train_hessian_x: np.ndarray, new_hessian_data_point_index: np.ndarray, retrain_bool= True):
        '''
        add new training data into the GPR model.
        Then train the model to update the hyper-parameter 
        This function wrpas the function: update_model_with_new_data_GPHessian in ./gprHessian/RBFHessian_gp.py

        :param: new_train_x: [M, 3 * natom]. Cartesian coordinate of the input data 
        :param: new_train_V: [M] ab-initio potential data.
        :param: new_train_grad_x: [M, 3 * natom]: ab initio gradient data.
        :param: new_train_hessian_x: [M_H, 3 * natom, 3 * natom]: ab initio hessian data. Note not all data points contain hessian information.
        :param: new_hessian_data_point_index: the index of data points that contain hessian information. 
        '''
        assert np.shape(new_train_x)[1] == 3 * self.natom, "dim of coordinates for input data is not 3 * natom"
        assert np.shape(new_train_grad_x)[1] == 3 * self.natom, "dim of gradients for input data is not 3 * natom"
        assert (np.shape(new_train_hessian_x)[1] == 3 * self.natom and np.shape(new_train_hessian_x)[2] == 3 * self.natom), "the shape of hessian for input data is not 3 * natom"

        # transform input data into internal coordinate
        new_train_inputs = self.coordinate_transformer.get_internal_coordinate_q(new_train_x)
        # transform the gradient & hessian into internal coordinate 
        new_train_grad_q = self.coordinate_transformer.transform_cartesian_gradient_to_internal_gradient(new_train_x, new_train_grad_x)
        
        if len(new_train_hessian_x) > 0:
            new_train_hessian_x_symmetrized = (np.transpose(new_train_hessian_x, (0, 2, 1)) + new_train_hessian_x) / 2
            new_train_hessian_q = self.coordinate_transformer.transform_cartesian_hessian_to_internal_hessian(new_train_x[new_hessian_data_point_index], 
                                                                                                          new_train_grad_x[new_hessian_data_point_index],
                                                                                                          new_train_hessian_x_symmetrized)
            
        else:
            new_train_hessian_q = np.array([])
        
        # update the recorded training inputs and targets
        self.train_cartesian_input = np.concatenate([ self.train_cartesian_input, new_train_x ], axis= 0)
        self.train_V = np.concatenate([self.train_V, new_train_V])
        self.train_cartesian_gradient = np.concatenate([self.train_cartesian_gradient, new_train_grad_x], axis= 0)
        if len(self.train_cartesian_hessian) > 0:
            if len(new_train_hessian_x_symmetrized) > 0:
                self.train_cartesian_hessian = np.concatenate([self.train_cartesian_hessian, new_train_hessian_x_symmetrized], axis= 0)
        else:
            self.train_cartesian_hessian = new_train_hessian_x_symmetrized

        training_data_num = np.shape(self.train_cartesian_input)[0]
        new_hessian_data_point_index_in_full_data_set = new_hessian_data_point_index + training_data_num  # the hessian index in full data set after concatnate new data 
        self.training_data_hessian_data_point_index = np.concatenate([self.training_data_hessian_data_point_index, new_hessian_data_point_index_in_full_data_set], axis= 0)

        self.train_inputs = np.concatenate([self.train_inputs, new_train_inputs], axis= 0)
        self.train_grad_q = np.concatenate([self.train_grad_q, new_train_grad_q], axis= 0)
        if len(self.train_hessian_q) > 0:
            if len(new_train_hessian_q) > 0:
                self.train_hessian_q = np.concatenate([self.train_hessian_q, new_train_hessian_q], axis= 0)
        else:
            self.train_hessian_q = new_train_hessian_q
        
        # update the FixInternalDofs.hessians_fixed_dofs 
        if len(self.train_hessian_q) > 0:
            self.FixingDofs.update_hessians_fixed_dofs(self.train_hessian_q)

        # compute noise_covar_factor array for new training data.
        new_noise_covar_factor_pot_grad_array, new_noise_covar_factor_with_hessian_array = self.compute_noise_covar_factor_array(new_train_x, new_train_inputs, new_hessian_data_point_index)

        # Normalize the potential, gradient and hessians
        new_train_V, new_train_grad_q, new_train_hessian_q = self.Normalizer.normalization_transform(new_train_V, new_train_grad_q, new_train_hessian_q)

        # Filter the fixed dofs
        new_train_inputs = self.FixingDofs.transform_training_inputs_to_free_moving_dofs(new_train_inputs)
        new_train_grad_q, new_train_hessian_q = self.FixingDofs.transform_training_targets_to_free_moving_dofs(new_train_grad_q, new_train_hessian_q)
        # fix noise_covar_factor_array.
        new_noise_covar_factor_pot_grad_array, new_noise_covar_factor_with_hessian_array = self.FixingDofs.transform_noise_covar_factor_array_fixing_internal_dofs(new_noise_covar_factor_pot_grad_array, 
                                                                                                                                                                   new_noise_covar_factor_with_hessian_array)

        # Transform the potential, gradient, hessians into 1d target data
        new_train_targets = self.TargetDataTransformer.transform_pots_grad_hessian_to_1d_data(new_train_V, new_train_grad_q, new_train_hessian_q)

        # transform the training inputs, training targets into tensor.Torch
        new_train_inputs_tensor = torch.from_numpy(new_train_inputs)
        new_train_targets_tensor = torch.from_numpy(new_train_targets)
        new_hessian_data_point_index_tensor = torch.from_numpy(new_hessian_data_point_index)
        new_noise_covar_factor_pot_grad_array = torch.from_numpy(new_noise_covar_factor_pot_grad_array)
        new_noise_covar_factor_with_hessian_array = torch.from_numpy(new_noise_covar_factor_with_hessian_array)

        # update the Gaussian Process Regression model with new data.
        ipi.utils.gprHessian.RBFHessian_gp.update_model_with_new_data_GPHessian(self.gpr_model, new_train_inputs_tensor, 
                                                                      new_train_targets_tensor, 
                                                                      new_hessian_data_point_index_tensor,
                                                                      new_noise_covar_factor_pot_grad_array,
                                                                      new_noise_covar_factor_with_hessian_array,
                                                                      retrain_bool= retrain_bool)

    
    def train_model(self, output_training_info= False):
        '''
        function that trains the model
        '''
        train_gpr_model(self.gpr_model, output_training_info= output_training_info)

    def get_free_moving_internal_coordinate(self, beads_x):
        '''
        transform from Cartesian coordinate x to the free moving internal coordinates q.
        '''
        beads_internal_coordinate = self.coordinate_transformer.get_internal_coordinate_q(beads_x)

        free_moving_beads_internal_coordinate = self.FixingDofs.transform_training_inputs_to_free_moving_dofs(beads_internal_coordinate)

        return free_moving_beads_internal_coordinate

        

