'''
packages for constructing gaussian process regression model
using gpytorch framework (See https://docs.gpytorch.ai/en/stable/).
Written by Chenghao Zhang, Pacific Northwest National Laboratory (chenghao.zhang@pnnl.gov)
'''
import torch 
import gpytorch
import math
import numpy as np 
from ipi.utils.internalcoordtools import non_redundant_coordinate_transformer


class GPModelWithDerivatives(gpytorch.models.ExactGP):
    '''
    Gaussian Process model with multiple output (f(x), df/dx1, .. , df/dxn)
    '''
    def __init__(self, train_inputs: torch.Tensor, train_targets: torch.Tensor, 
                 ard_num_dims: int, output_dims: int,
                 gpr_SE_kernel_number: int,
                 kernel_outputscale, kernel_lengthscale_ratio,
                 likelihood_noise_variance):
        '''
        :param: train_inputs: training data.  torch.Tensor object. shape: [N, d]. N: number of data points. d: input data dimensions.
        :param: train_targets: training data.  torch.Tensor object. shape: [N, m]. N: number of data points. m: output data dimensions. (multiple output)
        :param: ard_num_dims: input data dimension (d). ard represents: automatic relevance determination. 
        :param: output_dims: output dims for multi-dimensional targets. 
                note: For training all gradient in d dimension, we should set num_tasks = ard_num_dims + 1  (f, df/dx1, .. , df/dx_d )
        :param: gpr_SE_kernel_number: number of squared exponential kernel used to construct the covariance function.
        :param: kernel_outputscale: output scale of each squared exponential kernel used to construct covariance function. numpy array. 
        :param: kernel_lengthscale_ratio: length scale of each squared exponential kernel used to construct covariance function. numpy array. 
                ratio of length scale parameter and the range of internal coordinate q along one dimension.
        :param: noise_variance :   mean and std to specify the prior of covariance factor for MultitaskGaussian distribution.
                                                       y = f + epsilon, where the variance of epsilon noise term is defined by likelihood noise variance.

        Note we can access train_inputs, train_targets, likelihood later as : self.train_inputs, self.train_targets, self.likelihood. This is defined in the gpytorch.models.ExactGP.
        '''
        self.input_dim = ard_num_dims
        self.output_dim = output_dims 
        
        # set the noise prior information and construct the likelihood class.
        likelihood = self._set_likelihood_noise_prior(output_dims, likelihood_noise_variance)

        super(GPModelWithDerivatives, self).__init__(train_inputs, train_targets, likelihood)

        # set the mean function for the Gaussian Processes
        self._set_mean_function(train_targets)

        # set covariance function (kernel) for Gaussian process regression:
        self._set_gpr_kernel(ard_num_dims, train_inputs,
                             gpr_SE_kernel_number, kernel_outputscale, kernel_lengthscale_ratio)
        
    def _set_likelihood_noise_prior(self, output_dims, likelihood_noise_variance):
        '''
        set the prior distribution for the variance of noise (for both potential V and force F)
        :param: output_dims: dimension of the output target.
        :param: likelihood_noise_variance: The mean value for the distribution of the variance of the noise. We set the prior of the noise variance as a Gaussian distribution.  (A gaussian distribution prior on the variance of the gaussian distribution of noise.)
        '''
        self.noise_rank = 0
        
        likelihood_noise_variance_mean = likelihood_noise_variance
        likelihood_noise_variance_std = likelihood_noise_variance / 10   # we set the std of the prior distribution as 1/10 of the mean value. 

        noise_mean_tensor = torch.from_numpy(likelihood_noise_variance_mean)
        noise_std_tensor = torch.from_numpy(likelihood_noise_variance_std)

        # set the prior of the noise as a normal distribution. 
        task_noise_prior = gpytorch.priors.NormalPrior(noise_mean_tensor, noise_std_tensor)

        # set the constraint of the variance of the noise to be 10 times larger or smaller than the prior mean value.
        noise_lower_bound_tensor = noise_mean_tensor.div(10)
        noise_upper_bound_tensor = noise_mean_tensor.mul(10)
        noise_constraint = gpytorch.constraints.Interval(noise_lower_bound_tensor, noise_upper_bound_tensor)

        # See documentation in https://docs.gpytorch.ai/en/stable/_modules/gpytorch/likelihoods/multitask_gaussian_likelihood.html#MultitaskGaussianLikelihood 
        # here task_noise: the noise for each output target dimension.  global_noise: noise for all targets (we turn it off)
        # rank= 0: represents the covariance matrix of the task noise will be diagonal, which means noise of force along each dimension is independent.
        # noise_prior: set the prior for the variance of the noise.   noise_constraint: add constraint to the variance of noise. Otherwise, the code will use default constraint, which is not appropriate.
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(output_dims, rank= 0, 
                                                                      noise_prior= task_noise_prior, noise_constraint= noise_constraint,
                                                                      has_task_noise= True, has_global_noise= False)

        # set initial value of task noises as the mean value of the prior. 
        likelihood.task_noises = noise_mean_tensor

        return likelihood 

    def _set_mean_function(self, train_targets):
        '''
        set the mean function for the Gaussian Process Regression.
        set the initial value of the mean function as the mean of target potential V.
        '''
        self.mean_module = gpytorch.means.ConstantMeanGrad()  # mean function for Gaussian Processes using gradient information
        # set initial value of mean constant
        train_target_func = train_targets[:,0].detach().numpy()  # function f in training data (other data are gradient df/dx)
        mean_constant_estimate = np.mean(train_target_func)
        self.mean_module.constant = torch.nn.Parameter(torch.ones(1) * mean_constant_estimate)

    def _set_gpr_kernel(self, ard_num_dims, train_inputs,
                        gpr_SE_kernel_number, kernel_outputscale, kernel_lengthscale_ratio):
        '''
        set the kernel for the Gaussian Process Regression.
        kernel = (sigma_m) ^2 exp(- sum_i (x1_i - x2_i)^2 / l_i^2 )
        :param: ard_num_dims: number of input dimensions (ard: automatic relevance determination)
        :param: train_inputs: internal coordinate q for the training data. 
        :param: gpr_SE_kernel_number:  We provide options to have multiple Squared Exponential kernel to sum together to form the kernel. Here gpr_SE_kernel_number is the number of Squared Exponential kernels.
        :param: kernel_outputscale: the output scale of the gpr kernel. Here kernel_outputscale = (sigma_m) ^2
        :param: kernel_lengthscale_ratio: l_i / |q_i^{max} - q_i^{min}|.  The ratio of the kernel length scale and the range of initial training data along dim i.
        '''
        self.gpr_SE_kernel_number = gpr_SE_kernel_number

        covar_module_component_list = []
        base_kernel_component_list = []
        
        # we choose Gamma distribution as prior distribution for output scale and length scale. 
        # See https://docs.gpytorch.ai/en/stable/priors.html  &  https://www.wikiwand.com/en/Gamma_distribution
        # alpha parameter (shape parameter) for the gamma distribution of the length and outputscale. 
        length_gamma_alpha = 3.0
        output_gamma_alpha = 3.0        

        for i in range(gpr_SE_kernel_number):
            # The prior distribution of the length scale of the parameter is decided by the initial training inputs. 
            # this is bad for cross-validation, but for simply training the model, it should be fine.
            train_inputs_range = torch.max(train_inputs, dim= 0).values - torch.min(train_inputs , dim= 0).values 
            length_scale = kernel_lengthscale_ratio[i] * train_inputs_range   # set it as a given ratio of the training input range.
            length_gamma_beta = torch.div(length_gamma_alpha, length_scale)  # value of beta: rate of the gamma distribution.

            output_scale = kernel_outputscale[i]

            # set prior for length scale and output scale
            lengthscale_prior = gpytorch.priors.GammaPrior(length_gamma_alpha, length_gamma_beta)
            outputscale_prior = gpytorch.priors.GammaPrior(output_gamma_alpha, output_gamma_alpha / output_scale)

            # also add length scale constraint: minimum cutoff to prevent over-fitting. 
            length_scale_ratio_cutoff = 0.1
            length_scale_cutoff = length_scale_ratio_cutoff * train_inputs_range
            lengthscale_constraint = gpytorch.constraints.GreaterThan(length_scale_cutoff)

            # set Squared Exponential kernel function
            base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims= ard_num_dims, 
                                                         lengthscale_prior= lengthscale_prior, lengthscale_constraint= lengthscale_constraint)
            
            covar_module = gpytorch.kernels.ScaleKernel(base_kernel, outputscale_prior = outputscale_prior)

            # Initialize lengthscale and output scale to the mean of priors
            covar_module.base_kernel.lengthscale = lengthscale_prior.mean 
            covar_module.outputscale = outputscale_prior.mean 

            base_kernel_component_list.append(base_kernel)
            covar_module_component_list.append(covar_module)

        self.base_kernel_component_list = base_kernel_component_list 
        self.covar_module_component_list = covar_module_component_list

        # sum of Squared Exponential Covariance function. 
        self.covar_module = self.covar_module_component_list[0]
        for i in range(1, gpr_SE_kernel_number):
            self.covar_module = self.covar_module + self.covar_module_component_list[i]

    def forward(self, x):
        '''
        forward function is used to define the model.  See https://docs.gpytorch.ai/en/stable/examples/00_Basic_Usage/Implementing_a_custom_Kernel.html
        forward function takes in some n*d input data (x) and returns a prior MultivariateNormal distribution with mean and covariance evaluated at x.
        '''
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

# ------ functions below are auxiliary functions to output gpr model parameters ------------------------
    def output_kernel_lengthscale(self):
        '''
        output length scale of the base kernel
        :return: length scale. shape: [SE_kernel_number, ard_num_dims]. numpy array.
        '''
        # output length scale of all kernel component
        lengthscale_list = []
        for i in range(self.gpr_SE_kernel_number):
            length_scale = np.copy(self.base_kernel_component_list[i].lengthscale[0].detach().numpy())
            lengthscale_list.append(length_scale)
        lengthscale_list = np.array(lengthscale_list)

        return lengthscale_list

    def output_kernel_outputscale(self):
        '''
        output the output scale of the covar module
        :return: output covariance scale list. shape: [SE_kernel_number, output_dims]. numpy array 
        '''
        # output the outputscale of all kernel component
        outputscale_list = []
        for i in range(self.gpr_SE_kernel_number):
            output_scale = np.copy(self.covar_module_component_list[i].outputscale.detach().numpy())
            outputscale_list.append(output_scale)
        outputscale_list = np.array(outputscale_list)

        return outputscale_list
    
    def output_fitted_noise(self):
        '''
        output fitted noise for potential V and force f.
        '''
        task_noises_var = self.likelihood.task_noises 

        task_noises_var = task_noises_var.detach().numpy()
        task_noises_std = np.sqrt(task_noises_var)

        V_noises = task_noises_std[0]
        force_noises = task_noises_std[1:]

        return V_noises, force_noises 


def train_gpr(model:GPModelWithDerivatives , training_error_cutoff = np.power(10.0, -3)):
    '''
    the function that trains the model.
    :param: model: GPytorch model 
    :param: training_error_cutoff: train until the change of loss function is smaller than the cutoff.
    
    :return: None
    '''
    # set model & likelihood to the training mode
    model.train()
    likelihood = model.likelihood 
    likelihood.train()

    # choose the optimizer for the training to train the parameter of models (raw_parameter)
    # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)  

    # define loss function for GPs. -- we choose the marginal log likelihood
    # because we need to maximise the marginal log likelihood, we should define the loss function as -mll
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    train_inputs = model.train_inputs[0]  # model.train_inputs is the tuple containing our training data.
    train_targets = model.train_targets

    # initialize loss_func_change and old_loss to enable while loop
    loss_func_change = 1000
    old_loss_value = 1000

    train_counts = 0 

    # for debug
    # print("Iter %d" %(train_counts))
    # print("mean_module constant: " + str(model.mean_module.constant))    
    # kernel_lengthscale = model.output_kernel_lengthscale()
    # print("kernel lengthscale: " + str(kernel_lengthscale) )
    # output_scale = model.output_kernel_outputscale()
    # print("outputscale: " + str(output_scale))
    # print("noise:" + str(likelihood.task_noises))
    # print("\n")

    loss_value_list = []
    loss_prior_list = []
    
    while loss_func_change > training_error_cutoff:
        # reset the gradients of all optimized torch.Tensor 
        optimizer.zero_grad()   
        # output from model training data
        output = model(train_inputs)
        # calculate the loss function. here the returned loss is a torch.tensor.
        loss = - mll(output, train_targets)
        loss_value = loss.item() 
        loss_value_list.append(loss_value)

        if loss_value > old_loss_value and abs((loss_value - old_loss_value) / old_loss_value) > 0.1:
            print("@WARNING: the training could be unstable. loss function increases:  {}   ->     {}".format(old_loss_value, loss_value))

        # calculate the change of loss function to decide whether we will stop the loop.
        loss_func_change = np.abs(loss_value - old_loss_value)
        old_loss_value = loss_value 

        # compute loss from prior
        loss_prior = torch.tensor(0.0)
        loss_prior = - mll._add_other_terms(loss_prior, [])
        loss_prior_list.append(loss_prior.item())

        # backpropagation the loss function to compute the gradient of each parameter
        loss.backward()
        # optimizer optimize the parameter using the gradient info.
        optimizer.step()

        train_counts = train_counts + 1

    # for debug:
    # print("Iter %d - Loss %.3f" %(train_counts, loss_value))
    # print("mean_module constant: " + str(model.mean_module.constant))
    # kernel_lengthscale = model.output_kernel_lengthscale()
    # print("kernel lengthscale: " + str(kernel_lengthscale) )
    # output_scale = model.output_kernel_outputscale()
    # print("outputscale: " + str(output_scale))
    # print("noise:" + str(likelihood.task_noises))
    # print("\n")

    pass 

def predict_latent_function_gp_with_derivative(model:GPModelWithDerivatives, test_inputs, covar_bool = False):
    '''
    the function that predict the posterior distribution latent function f(test_inputs) of the test_inputs.
    
    :param: model: instance of GPModelWithDerivatives. Gaussian process regression model using derivative information.
    :param: test_inputs:  test data to compute posterior distribution. dtype: torch.tensor
    :param: covar_bool: bool variable, if true: output covariance matrix. If false: variance (diagonal of covariance matrix). Default: False
    
    suppose output_dim = m.
    return: mean: mean value of prediction for test_inputs. shape: [N, m]

            if covar_bool == True: return: test_covariance: [N * m , N * m]
            if covar_bool == False: return: test_var: [N, m]: each row is the variance of one data point. 
    '''
    # check the input dim of test data is correct
    test_input_dim = test_inputs.shape[-1]
    assert test_input_dim == model.input_dim, "dimension of input test_inputs data is incompatible with model"

    model.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        latent_func = model(test_inputs)  # MultitaskMultivariateNormal distribution object.

        data_num = test_inputs.shape[0]  # number of test_inputs data 

        test_mean = latent_func.mean 
        test_covariance = latent_func.covariance_matrix
        
        # diagonal component of covariance matrix is the variance of function and gradient
        test_var =  torch.diag(test_covariance)

        test_var = test_var.reshape([model.output_dim, data_num])  # first row is variance for f,  second row is variance for df/dx1, third row: df/dx2, .. 
        test_var = torch.transpose(test_var, 0, 1)  # now each row is one data piont.

    if covar_bool:
        return test_mean, test_covariance 
    else:
        return test_mean, test_var 

def filter_new_training_data(model: GPModelWithDerivatives ,new_train_inputs, existing_train_inputs, distance_cutoff):
    '''
    we will reject adding new data when the new data point is too close to the existing data point. (given by distance_cutoff)
    Having two similar data points in the training set will make kernel matrix become ill-conditioned, 
    when we try to make prediction, inverting kernel matrix K will be unstable, which causes GPR model to crash. 
    :param: model: GPR model
    :param: new_train_inputs: numpy array. potential new training inputs to add to gpr model
    :param: existing_train_inputs: numpy array.  training inputs already in the GPR model.
    :param: distance_cutoff: cutoff of distance in the internal coordinate. The new training data should have distance from the existing training data larger than distance_cutoff.
    
    :return: filtered_new_train_inputs_index: numpy array. The index of the new training inputs after we deleting  the data which is too close to the existing data.
    '''
    filtered_new_train_inputs_index = []

    # kernel output scale and kernel length scale of kernels
    kernel_output_scale = model.output_kernel_outputscale()
    kernel_length_scale = model.output_kernel_lengthscale()
    # normalize the output scale:
    output_scale_sum = np.sum(kernel_output_scale)
    kernel_output_scale_normalized = kernel_output_scale / output_scale_sum
    # effective kernel lengthscale for scaling internal coordinate. l_eff^{-2} = sum_{n} output_scale_n / (l_n)^2.   
    effective_kernel_length_scale = np.power(np.sum(kernel_output_scale_normalized[:, np.newaxis] / np.power(kernel_length_scale, 2) , axis = 0), -0.5)
    
    new_data_num = len(new_train_inputs)

    for data_index in range(new_data_num):
        new_input = new_train_inputs[data_index]

        # distance from new input to the existing training data 
        internal_coordinate_r = np.linalg.norm((new_input - existing_train_inputs)/ effective_kernel_length_scale, axis= 1)

        # the data that is closest to the new input 
        nearest_existing_training_inputs_index = np.argmin(internal_coordinate_r)

        internal_coordinate_closest_r = internal_coordinate_r[nearest_existing_training_inputs_index]

        if internal_coordinate_closest_r > distance_cutoff:
            filtered_new_train_inputs_index.append(data_index)
    
    filtered_new_train_inputs_index = np.array(filtered_new_train_inputs_index)

    return filtered_new_train_inputs_index


def update_model_with_new_data(model : GPModelWithDerivatives, new_train_inputs, new_train_targets, distance_cutoff):
    '''
    add new training data into the model. 
    Then train the model to update the hyper-parameter.

    :param: model: GPModelWithDerivative: GPR model that train with derivative information
    :param: new_train_inputs: new training input data. datatype: torch.tensor() or numpy array. Size [N, d]. here d is input_dim
    :param: new_train_targets: new training target data. datatype: torch.tensor() or numpy array. Size [N, m], here m is output dim (m = d + 1)
    '''
    train_inputs = model.train_inputs[0]
    train_targets = model.train_targets

    # check the data type of input training data. If it's not torch.Tensor, convert it to torch.Tensor.
    if type(new_train_inputs) != torch.Tensor:
        new_train_inputs_tensor = torch.from_numpy(np.array(new_train_inputs))
    else:
        new_train_inputs_tensor = torch.clone(new_train_inputs)

    if type(new_train_targets) != torch.Tensor:
        new_train_targets_tensor = torch.from_numpy(np.array(new_train_targets))
    else:
        new_train_targets_tensor = torch.clone(new_train_targets)

    # check the input dimension of the new_train_inputs
    assert new_train_inputs_tensor.shape[-1] == model.input_dim, "the input dimension of new_train_inputs is wrong. new_train_input dim {}. required input dim {}".format(new_train_inputs.shape[-1], model.input_dim)
    # check the output dimension of the new_train_targets
    assert new_train_targets_tensor.shape[-1] == model.output_dim, "the output dimension of new_train_targets is wrong. new_train_targets dim {}, required output dim {}".format(new_train_targets.shape[-1], model.output_dim)

    # filter the new inputs which is too close to the existing data point.
    new_train_inputs_numpy = new_train_inputs_tensor.numpy() 
    train_inputs_numpy = train_inputs.numpy() 

    # filter the training input data to delete the one which is too close to the existing data.
    filtered_new_train_inputs_index = filter_new_training_data(model, new_train_inputs_numpy, train_inputs_numpy, distance_cutoff)
    filtered_new_train_inputs_tensor = new_train_inputs_tensor[filtered_new_train_inputs_index, :]
    filtered_new_train_targets_tensor = new_train_targets_tensor[filtered_new_train_inputs_index, :]

    if len(filtered_new_train_inputs_index) == 0:
        # all new data is too close to the existing data, we do not update the model.
        return filtered_new_train_inputs_index

    # create new training inputs and training targets
    full_train_inputs = torch.cat([train_inputs, filtered_new_train_inputs_tensor], dim = 0)
    full_train_targets = torch.cat([train_targets, filtered_new_train_targets_tensor], dim = 0)

    # set the training data for the model
    model.set_train_data(inputs= full_train_inputs, targets= full_train_targets, strict= False)

    # re-train the model to update the hyper-parameter 
    train_gpr(model)

    return filtered_new_train_inputs_index

class FixInternalDofs(object):
    """
    class that fix certain internal dofs in the training data before feeding data into the Gaussian Process Regression model 
    """
    def __init__(self, train_inputs: np.ndarray, train_targets: np.ndarray):
        self.input_dim = np.shape(train_inputs)[1]
        self.output_dim = np.shape(train_targets)[1]
        self.fix_internal_dofs_cutoff = np.power(10.0, -4)

        # check whether coordinate along certain internal dof is fixed
        train_inputs_change = np.max(train_inputs, axis= 0) - np.min(train_inputs, axis= 0)
        self.fixed_internal_dofs = np.array([i for i in range(self.input_dim) if train_inputs_change[i] < self.fix_internal_dofs_cutoff])
        if len(self.fixed_internal_dofs) != 0:
            self.free_moving_dofs = np.delete(np.arange(self.input_dim), self.fixed_internal_dofs)
            self.targets_fixed_dofs = np.mean(train_targets, axis= 0)[self.fixed_internal_dofs + 1]  # the first column of the target is the potential V.
        else:
            self.free_moving_dofs = np.arange(self.input_dim)
            self.targets_fixed_dofs = np.array([])

        
        

    def transform_training_data_to_free_moving_dofs(self, train_inputs: np.ndarray, train_targets: np.ndarray, noise_var= None):
        '''
        delete fixdofs from training data
        :param: train_inputs: the training inputs in internal dofs. Including all internal dofs.
        :param: train_targets: the training targets in internal dofs. Including all internal dofs.
        :param: noise_var: the variance of noise in internal dofs.
        '''
        moving_train_inputs = self.transform_training_inputs_to_free_moving_dofs(train_inputs)
        moving_train_targets = self.transform_training_targets_to_free_moving_dofs(train_targets) # the first column of target is potential V.

        if type(noise_var) != type(None):
            if len(self.fixed_internal_dofs) != 0:
                moving_noise_var = np.delete(noise_var, self.fixed_internal_dofs + 1)
            else:
                moving_noise_var = noise_var 

            return moving_train_inputs, moving_train_targets, moving_noise_var 
        else:
            return moving_train_inputs, moving_train_targets 
    
    def transform_training_inputs_to_free_moving_dofs(self, train_inputs: np.ndarray):
        '''
        delete fixdofs from training inputs.
        :param: train_inputs: the training inputs in internal dofs. 
        '''
        if len(self.fixed_internal_dofs) != 0:
            moving_train_inputs = np.delete(train_inputs, self.fixed_internal_dofs, axis = 1)
        else:
            moving_train_inputs = train_inputs

        return moving_train_inputs

    def transform_training_targets_to_free_moving_dofs(self, train_targets: np.ndarray):
        '''
        delete fixdofs from training targets
        :param: train_targets: the training targets in internal dofs
        '''
        if len(self.fixed_internal_dofs) != 0:
            moving_train_targets =  np.delete(train_targets, self.fixed_internal_dofs + 1, axis= 1) # the first column of target is potential V.
        else:
            moving_train_targets = train_targets

        return moving_train_targets

    def transform_from_free_moving_dofs_to_full_dofs(self, moving_test_mean: np.ndarray, moving_test_var: np.ndarray):
        '''
        Transform the prediction of GPR model from free moving dofs into full dofs.
        :param: moving_prediction_mean: the mean value of posterior prediction for moving dofs 
        :param: moving_prediction_var: the variance of posterior prediction for moving dofs 
        '''
        test_data_num = moving_test_mean.shape[0]

        # the prediction of fixed dof for testing data.
        test_target_fixed_dofs = np.repeat([self.targets_fixed_dofs], test_data_num, axis= 0)

        # the mean value of the prediction of the test data with all dofs
        test_mean = np.zeros([test_data_num, self.output_dim])
        test_mean[:,0] = moving_test_mean[:, 0]  # potential
        test_mean[:, self.free_moving_dofs + 1] = moving_test_mean[:, 1:]
        if len(self.fixed_internal_dofs) != 0:
            test_mean[:, self.fixed_internal_dofs + 1] = test_target_fixed_dofs 

        # the variance of the prediction of the test data with all dofs 
        test_var = np.zeros([test_data_num, self.output_dim])
        test_var[:,0] = moving_test_var[:, 0] # potential
        test_var[:, self.free_moving_dofs + 1] = moving_test_var[:, 1:]
        if len(self.fixed_internal_dofs) != 0:
            test_var[:, self.fixed_internal_dofs + 1] = 0 

        return test_mean, test_var 

class NormalizeTrainingData(object):
    '''
    normalize the potential & force of training data.
    '''
    def __init__(self, training_targets: np.ndarray):
        '''
        V_normalized = (V - V_mean)/V_range.
        :param: training_targets : [V, dV/dx]. numpy array.
        '''
        V = training_targets[:, 0]
        self.V_mean = np.mean(V)
        self.V_range = np.max(V) - np.min(V)

    def normalization_transform(self, training_targets):
        '''
        normalize the potential V & force F.
        V_normalized = (V - V_mean) / V_range.  
        F_normalized = F / V_range.

        This function perform the normalize procedure.  
        :param: training_targets : [V, dV/dq]. numpy array.
        '''
        V = training_targets[:, 0]
        V_normalized = (V - self.V_mean) / self.V_range 

        grad_V = training_targets[:,1:]
        grad_V_normalized = grad_V / self.V_range 

        normalized_training_targets = np.concatenate( [ V_normalized[:, np.newaxis], grad_V_normalized ] , axis = 1 )

        return normalized_training_targets

    def inverse_normalization_transform(self, normalized_training_targets):
        '''
        inverse the normalization procedure of potential V & force F.

        V = V_normalized * V_range + V_mean
        F = F_normalized * V_range 

        This function perform the inverse of normalization procedure.
        :param: normalized_training_targets: [V_normalized, d V_normalized /dx]. numpy array.
        '''
        V_normalized = normalized_training_targets[:, 0]
        grad_V_normalized = normalized_training_targets[:, 1:]

        # scale the potential and gradient of potential
        V = V_normalized * self.V_range + self.V_mean 
        grad_V = grad_V_normalized * self.V_range 

        training_targets = np.concatenate([ V[:, np.newaxis], grad_V ], axis = 1)

        return training_targets
    
    def normalize_noise_var(self, noise_var):
        '''
        normalize the variance of noise
        '''
        normalized_noise_var = noise_var / np.power(self.V_range, 2)

        return normalized_noise_var 
    
    def inverse_normalize_noise_var(self, normalized_noise_var):
        '''
        inverse normalize the variance of the noise 
        '''
        noise_var = normalized_noise_var * np.power(self.V_range ,2)

        return noise_var

class GPModelWithDerivativesWrapper():
    '''
    wrapper class for GPModelWithDerivatives. 
    handles the transformation between internal coordinate and cartesian coordinate + GPR training.
    '''
    def __init__(self, train_x: np.ndarray, train_V: np.ndarray, train_grad_x: np.ndarray, 
                 natom: int, coordinate_transformer : non_redundant_coordinate_transformer,
                 gpr_SE_kernel_number: int,
                 kernel_outputscale: np.ndarray, kernel_lengthscale_ratio: np.ndarray,
                 noise_std):
        '''
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
        '''
        assert np.shape(train_x)[1] == 3 * natom, "dim of coordinates for input data is not 3 * natom, this is wrong. train_x data shape: {} , 3 * natom: {}".format(np.reshape(train_x)[1], 3 * natom)
        assert np.shape(train_grad_x)[1] == 3 * natom, "dim of gradients for input data is not 3 * natom, this is wrong. train_grad shape:{}, 3 * natom: {}".format(np.shape(train_grad_x)[1], 3 * natom)
        
        self.natom = natom
        self.gpr_SE_kernel_number = gpr_SE_kernel_number

        self.coordinate_transformer = coordinate_transformer

        # the training targets for the GPR with derivative is [V, dV/dx1, ..., dV/dxn]
        train_cartesian_targets = np.concatenate([train_V[:, np.newaxis], train_grad_x ], axis = 1)

        # transform cartesian coordinate x to internal coordinate q
        train_inputs = coordinate_transformer.get_internal_coordinate_q(train_x)

        input_dim = np.shape(train_inputs)[1]   # numbers of degree of freedom for the non-redundant internal coordinate q.
        output_dim = input_dim + 1 # output_dim = input_dim + 1 (train_targets = [V, dV/dx1, ..., dV/dxn])
        self.input_dim = input_dim
        self.output_dim = output_dim 

        # shape: [N, 3 * natom - 6]
        # transform the gradient of potential V: dV/dx -> dV/dq  
        train_grad_q = coordinate_transformer.transform_cartesian_gradient_to_internal_gradient(train_x, train_grad_x)
        # target data: [V, dV/dx1, ..., dV/dxn]
        train_targets = np.concatenate( [ train_V[:, np.newaxis] , train_grad_q ], axis = 1 )

        # compute the estimated noise covariance factor for the force in the internal coordinate q. noise for Fq = dV/dq. 
        likelihood_noise_variance  = self.transform_cartesian_noise_to_gpr_model_noise(noise_std)

        # decide normalization parameter. Here we normalize the potential as (V- <V>)/range(V). The force also needs to be scaled. 
        self.Normalizer = NormalizeTrainingData(train_targets)
        # perform normalization on training targets.
        normalized_train_targets = self.Normalizer.normalization_transform(train_targets)
        likelihood_noise_variance = self.Normalizer.normalize_noise_var(likelihood_noise_variance)

        self.train_inputs = train_inputs  # training inputs in internal coordinate space q.
        self.normalized_train_targets = normalized_train_targets  # training outputs in internal coordinates q. (V, dV/dq)

        self.train_cartesian_inputs = train_x  # training inputs in cartesian coordinate x
        self.train_cartesian_targets = train_cartesian_targets  # training targets in cartesian coordinate (V, dV/dx)
        
        # For the case we have to fix certain internal dofs. Apply a filter to fix some internal dofs
        self.FixingDofs = FixInternalDofs(train_inputs, normalized_train_targets)
        moving_train_inputs, moving_train_targets, moving_likelihood_noise_variance = self.FixingDofs.transform_training_data_to_free_moving_dofs(train_inputs, 
                                                                                                                                            normalized_train_targets,
                                                                                                                                            likelihood_noise_variance)
        moving_input_dim = input_dim - len(self.FixingDofs.fixed_internal_dofs)
        moving_output_dim = output_dim - len(self.FixingDofs.fixed_internal_dofs)        

        # transform input from numpy array to torch.tensor
        moving_train_inputs_tensor = torch.from_numpy(moving_train_inputs)
        moving_train_targets_tensor = torch.from_numpy(moving_train_targets)
        
        # initialize the gaussian process regression model with input training data.
        self.gpr_model = GPModelWithDerivatives(moving_train_inputs_tensor, moving_train_targets_tensor, 
                                                moving_input_dim, moving_output_dim,
                                                gpr_SE_kernel_number,
                                                kernel_outputscale, kernel_lengthscale_ratio, 
                                                moving_likelihood_noise_variance)

        # train self.gpr_model() to get optimized hyperparameter
        train_gpr(self.gpr_model)

    def transform_cartesian_noise_to_gpr_model_noise(self, noise_std):
        '''
        transform the noise in cartesian coordinate to noise in Gaussian Process Regressioin model.
        This is critical for the successful training of the GPR model, otherwise, we will treat the noise incorrectly. 
        '''
        pot_noise_std = noise_std["pot_noise_prior"]
        force_noise_std_cartesian = noise_std["force_noise_prior"]  # noise of force in the Cartesian coordinate. We assume the noise is isotropic. 

        # force noise has to be scaled by the inverse of the singular value of Wilson's B matrix.
        self.Bmatrix_singular_value_square = np.power(self.coordinate_transformer.ref_S, 2)

        singular_value_square_inverse = 1 / self.Bmatrix_singular_value_square

        force_noise_var = singular_value_square_inverse * np.power(force_noise_std_cartesian, 2) 

        pot_noise_var = np.power(pot_noise_std, 2)

        noise_variance = np.concatenate([[pot_noise_var], force_noise_var])
        return noise_variance 

    def predict_latent_function(self, test_x, internal_coordinate_bool = False):
        '''
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
        '''
        assert np.shape(test_x)[1] == 3 * self.natom , "dim of coordinates for input data is not 3 * natom"
        
        test_data_num = np.shape(test_x)[0]
        # transform to internal coordinate q.
        moving_test_q = self.get_free_moving_internal_coordinate(test_x)
        moving_test_q_tensor = torch.from_numpy(moving_test_q)

        # use Gaussian process regression model to make prediction 
        moving_normalized_test_mean_tensor, moving_normalized_test_var_tensor = predict_latent_function_gp_with_derivative(self.gpr_model, test_inputs = moving_test_q_tensor, covar_bool = False)
        
        moving_normalized_test_mean = moving_normalized_test_mean_tensor.detach().cpu().numpy()
        moving_normalized_test_var = moving_normalized_test_var_tensor.detach().cpu().numpy()

        # attach test_mean and test_var (0) of fixed dofs
        normalized_test_mean, normalized_test_var = self.FixingDofs.transform_from_free_moving_dofs_to_full_dofs(moving_normalized_test_mean, moving_normalized_test_var)

        # inverse the normalization procedure for mean value and variance.
        test_var = self.Normalizer.inverse_normalize_noise_var(normalized_test_var)
        test_mean = self.Normalizer.inverse_normalization_transform(normalized_test_mean)

        V = test_mean[:, 0]
        grad_q = test_mean[:, 1:]  # gradient dV/dq.
        
        # transform gradient from internal coordinate back to cartesian coordinate. 
        grad_x = self.coordinate_transformer.transform_internal_gradient_to_cartesian_gradient(test_x, grad_q)

        var_V = test_var[:, 0]
        var_grad_q = test_var[:, 1:]

        # covariance matrix for the noise in the Cartesian coordinate: Cov(noise_x, noise_x) = V S diag(var_grad_q) S V^T. Here S is singular value matrix, V is the right singular vector matrix.
        # the measure of the force noise can be defined as the trace of the covariance matrix of the force noise in Cartesian coordinate. 
        var_grad_x_trace = np.sum( self.Bmatrix_singular_value_square * var_grad_q, axis = 1)

        if internal_coordinate_bool:
            return V, grad_q, var_V,  var_grad_x_trace 
        else:
            return V, grad_x, var_V,  var_grad_x_trace 
    
    def update_model_with_new_data(self, new_train_x, new_train_V, new_train_grad_x, distance_cutoff):
        '''
        add new training data into the model. 
        Then train the model to update the hyper-parameter.
        This function wraps the function: update_model_with_new_data(gpr_model, train_inputs, train_targets)
        TODO: Note, there is still room for improvement in this function, here each time we update new data, we have to re-compute the inverse of covariance matrix K.

        This function will update the self.gpr_model

        :param: new_train_x: [N, 3 * natom], input Cartesian coordinate data.  numpy array
                new_train_V: [N], ab-initio potential data.   numpy array
                new_train_grad_x: [N, 3 * natom], ab-initio gradient data.  numpy array.
        
        :return: None.
        '''
        assert np.shape(new_train_x)[1] == 3 * self.natom, "dim of coordinates for input data is not 3 * natom"
        assert np.shape(new_train_grad_x)[1] == 3 * self.natom, "dim of gradients for input data is not 3 * natom"

        # input data in internal coordinate
        new_train_inputs = self.coordinate_transformer.get_internal_coordinate_q(new_train_x)

        # gradient of potential in internal coordinate
        new_train_grad_q = self.coordinate_transformer.transform_cartesian_gradient_to_internal_gradient(new_train_x, new_train_grad_x)
        assert np.shape(new_train_grad_q)[1] == self.input_dim, "train_grad_q for internal coordiante has wrong dimension"
        new_train_targets = np.concatenate([ new_train_V[:,np.newaxis], new_train_grad_q ], axis = 1)

        # normalize the new_train_targets
        normalized_new_train_targets = self.Normalizer.normalization_transform(new_train_targets)

        # For the case we have to fix certain dofs
        moving_new_train_inputs, moving_new_train_targets = self.FixingDofs.transform_training_data_to_free_moving_dofs(new_train_inputs, normalized_new_train_targets)

        normalized_moving_new_train_targets_tensor = torch.from_numpy(moving_new_train_targets)
        moving_new_train_inputs_tensor = torch.from_numpy(moving_new_train_inputs)

        filtered_new_train_inputs_index = update_model_with_new_data(self.gpr_model, moving_new_train_inputs_tensor, 
                                                                     normalized_moving_new_train_targets_tensor, distance_cutoff)

        if len(filtered_new_train_inputs_index) != 0:
            # update the training data and targets in internal coordinate q.
            self.train_inputs = np.concatenate([self.train_inputs, new_train_inputs[filtered_new_train_inputs_index] ], axis = 0)
            self.normalized_train_targets = np.concatenate([self.normalized_train_targets, normalized_new_train_targets[filtered_new_train_inputs_index]  ], axis = 0)

            # update the training data and targets in cartesian coordinate x.
            new_train_cartesian_targets = np.concatenate([new_train_V[:,np.newaxis], new_train_grad_x] , axis = 1)
            self.train_cartesian_inputs = np.concatenate([self.train_cartesian_inputs, new_train_x[filtered_new_train_inputs_index]], axis = 0)
            self.train_cartesian_targets = np.concatenate([self.train_cartesian_targets, new_train_cartesian_targets[filtered_new_train_inputs_index]], axis = 0)


# ------ functions below are auxiliary functions to output gpr model parameters ------------------------
    def output_kernel_lengthscale(self):
        '''
        return the length scale of kernel for gpr model
        :return: length scale (numpy array)
        '''
        lengthscale = self.gpr_model.output_kernel_lengthscale()

        return lengthscale 

    def output_kernel_outputscale(self):
        '''
        return the output scale of the kernel for gpr model.
        :return: output scale (numpy array)
        '''
        output_scale = self.gpr_model.output_kernel_outputscale()
        
        return output_scale 

    def output_training_cartesian_inputs(self):
        '''
        output the training data set X (in cartesian coordinate) used to train the GPR model.
        '''
        train_cartesian_X = np.copy(self.train_cartesian_inputs)

        return train_cartesian_X
    
    def output_free_moving_training_internal_inputs(self):
        '''
        output the training data set Q (in non-redundant internal coordinate) used to train the GPR model
        '''
        free_moving_train_inputs = self.FixingDofs.transform_training_inputs_to_free_moving_dofs(np.copy(self.train_inputs))
        
        return free_moving_train_inputs
    
    def output_fitted_gpr_model_noises(self):
        '''
        output the fitted noises for normalized internal coordinate potential V and forces f.
        '''
        V_noises, force_noises = self.gpr_model.output_fitted_noise()

        return V_noises, force_noises 
    
    def output_normalized_force_range(self):
        '''
        output the range of normalized force
        '''
        free_moving_normalized_targets = self.FixingDofs.transform_training_targets_to_free_moving_dofs(self.normalized_train_targets)
        free_moving_normalized_force = free_moving_normalized_targets[:,1:]
        free_moving_normalized_force_range = np.max(free_moving_normalized_force, axis = 0) - np.min(free_moving_normalized_force, axis = 0)

        return free_moving_normalized_force_range
    
    def get_free_moving_internal_coordinate(self, beads_x):
        '''
        transform from Cartesian coordinate x to the free moving internal coordinates q.
        '''
        beads_internal_coordinate = self.coordinate_transformer.get_internal_coordinate_q(beads_x)
        
        free_moving_beads_internal_coordinate = self.FixingDofs.transform_training_inputs_to_free_moving_dofs(beads_internal_coordinate)

        return free_moving_beads_internal_coordinate