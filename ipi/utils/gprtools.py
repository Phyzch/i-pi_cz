'''
packages for constructing gaussian process regression model
using gpytorch framework (See https://docs.gpytorch.ai/en/stable/)
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
    def __init__(self, train_inputs, train_targets, ard_num_dims, output_dims,
                 gpr_SE_kernel_number,
                 kernel_initial_outputscale, kernel_prior_lengthscale_ratio,
                 likelihood_noise_std_constraint):
        '''
        :param: train_inputs: training data.  torch.Tensor object. shape: [N, d]. N: number of data points. d: input data dimensions.
        :param: train_targets: training data.  torch.Tensor object. shape: [N, m]. N: number of data points. m: output data dimensions. (multiple output)
        :param: ard_num_dims: input data dimension (d)
        :param: output_dims: output dims for multi-dimensional output. 
        note: For training all gradient in d dimension, we should set num_tasks = ard_num_dims + 1  (f, df/dx1, .. , df/dx_d )
        :param: gpr_SE_kernel_number: number of squared exponential kernel use to construct the covariance function.
        :param: kernel_initial_outputscale: output scale of each squared exponential kernel used to construct covariance function. numpy array. 
        :param: kernel_initial_lengthscale: length scale of each squared exponential kernel used to construct covariance function. numpy array.
        :param: likelihood_noise_std_constraint: lower bound and upper bound for the noise of likelihood function p(y|f). 
                                                       y = f + epsilon, where the variance of epsilon noise term is defined by likelihood noise variance.

        We can access train_x, train_y, likelihood later as : self.train_inputs, self.train_targets, self.likelihood.
        '''
        self.input_dim = ard_num_dims
        self.output_dim = output_dims 
        
        # rank of noise covariance matrix.
        if likelihood_noise_std_constraint["noise_rank"] == 0:
            noise_rank = 0
        else:
            noise_rank = output_dims
        self.noise_rank = noise_rank

        # # # set the noise constraint for the likelihood. The default noise variance = 10^{-4} is too large.
        pot_noise_variance_lower_bound = np.power(likelihood_noise_std_constraint["pot_lower_bound_ratio"],2)
        pot_noise_variance_upper_bound = np.power(likelihood_noise_std_constraint["pot_upper_bound_ratio"],2)

        # set the noise constraint for the likelihood of force.
        force_noise_std_ratio_lower_bound = likelihood_noise_std_constraint["force_lower_bound_ratio"]
        force_noise_std_ratio_upper_bound = likelihood_noise_std_constraint["force_upper_bound_ratio"]

        # set the range of force noise using the info from the data.
        train_targets_force_numpy = train_targets[:,1:].detach().numpy() 
        force_range = np.max(train_targets_force_numpy, axis = 0) - np.min(train_targets_force_numpy , axis = 0)
        force_noise_lower_bound_array = np.power(force_range * force_noise_std_ratio_lower_bound,2) # lower bound is 0.5%. the input here is the variance of the noise. 
        force_noise_upper_bound_array = np.power(force_range * force_noise_std_ratio_upper_bound,2)  # upper bound is 2%

        noise_lower_bound_tensor = torch.from_numpy( np.concatenate( [ [pot_noise_variance_lower_bound], force_noise_lower_bound_array ] ) )
        noise_upper_bound_tensor = torch.from_numpy( np.concatenate( [ [pot_noise_variance_upper_bound], force_noise_upper_bound_array ] ) )
        noise_constraint = gpytorch.constraints.Interval(noise_lower_bound_tensor  , noise_upper_bound_tensor)

        pot_noise_prior_std = likelihood_noise_std_constraint["pot_noise_prior_std"]
        force_noise_prior_std = likelihood_noise_std_constraint["force_noise_prior_std"]
        noise_mean = torch.from_numpy(np.zeros([output_dims]))
        noise_covar = np.zeros([output_dims, output_dims])
        noise_covar[0,0] = np.power(pot_noise_prior_std, 2)
        for i in range(1,output_dims):
            noise_covar[i,i] = np.power(force_noise_prior_std ,2)
        noise_covar = torch.from_numpy(noise_covar)
        noise_prior = gpytorch.priors.MultivariateNormalPrior(noise_mean, noise_covar)
        
        # easiest task prior. apply to all covariance matrix.
        task_noise_prior_mean = 0
        task_noise_prior_std = likelihood_noise_std_constraint["task_noise_prior_std"]
        task_noise_prior = gpytorch.priors.NormalPrior(task_noise_prior_mean, task_noise_prior_std)
        global_noise_constraint = gpytorch.constraints.Interval(np.power(10.0, -6), np.power(10.0, -4))

        if noise_rank == 0:
        # likelihood: gpytorch.likelihood object. likelihood of observable given prediction f(X):  P(y|f(X)). See:  https://docs.gpytorch.ai/en/stable/likelihoods.html
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(output_dims, rank= 0,  noise_prior = noise_prior, noise_constraint = noise_constraint,
                                                                        has_task_noise= True, has_global_noise= False)
        else:           
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(output_dims, rank= noise_rank,  task_prior= task_noise_prior, noise_constraint= global_noise_constraint,
                                                                       has_task_noise= True, has_global_noise= True)
        
        # TODO list: set initial value of task_covar_factor to agree with prior mean.
        task_noise_covar_factor = torch.normal(torch.zeros(output_dims, output_dims), torch.ones(output_dims, output_dims) * task_noise_prior_std * np.sqrt(2 / output_dims))
        likelihood.task_noise_covar_factor = torch.nn.Parameter(task_noise_covar_factor)

        super(GPModelWithDerivatives, self).__init__(train_inputs, train_targets, likelihood)

        # mean function for prior distribution of Gaussian Processes
        self.mean_module = gpytorch.means.ConstantMeanGrad()  # mean function for Gaussian Processes using gradient information
        # set initial value of mean constant
        train_target_func = train_targets[:,0].detach().numpy()  # function f in training data (other data are gradient df/dx)
        mean_constant_estimate = np.mean(train_target_func)
        self.mean_module.constant = torch.nn.Parameter(torch.ones(1) * mean_constant_estimate)

        # set covariance function for Gaussian process regression:
        self.gpr_SE_kernel_number = gpr_SE_kernel_number
        covar_module_component_list = []
        base_kernel_component_list = []
        # we choose Gamma distribution as prior distribution for output scale and length scale.
        length_gamma_alpha = 9.0
        output_gamma_alpha = 3.0        

        for i in range(gpr_SE_kernel_number):
            # use length scale decided by data.
            train_inputs_numpy_array = train_inputs.detach().numpy()
            train_inputs_range = np.max(train_inputs_numpy_array, axis = 0) - np.min(train_inputs_numpy_array , axis = 0)
            length_scale = kernel_prior_lengthscale_ratio[i] * train_inputs_range   # set it as 0.1 of the training input range.

            self.initial_train_inputs_lengthscale = length_scale   # record the length scale of initial training inputs.

            length_scale = torch.from_numpy(length_scale)
            length_gamma_beta = torch.div(length_gamma_alpha, length_scale)

            output_scale = kernel_initial_outputscale[i]

            # set prior for length scale and output scale
            lengthscale_prior = gpytorch.priors.GammaPrior(length_gamma_alpha, length_gamma_beta)
            outputscale_prior = gpytorch.priors.GammaPrior(output_gamma_alpha, output_gamma_alpha / output_scale)

            # set Squared Exponential kernel function
            base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims= ard_num_dims, lengthscale_prior= lengthscale_prior)
            covar_module = gpytorch.kernels.ScaleKernel(base_kernel, outputscale_prior = outputscale_prior)

            # Initialize lengthscale and output scale to the mean of priors
            covar_module.base_kernel.lengthscale = lengthscale_prior.mean 
            covar_module.outputscale = outputscale_prior.mean 

            base_kernel_component_list.append(base_kernel)
            covar_module_component_list.append(covar_module)

        self.covar_module_component_list = covar_module_component_list
        self.base_kernel_component_list = base_kernel_component_list 

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
            output_scale = np.copy( self.covar_module_component_list[i].outputscale.detach().numpy())
            outputscale_list.append(output_scale)
        outputscale_list = np.array(outputscale_list)

        return outputscale_list
    
    def output_fitted_noise(self):
        '''
        output fitted noise range
        '''
        if self.noise_rank == 0:
            task_noises_var = self.likelihood.task_noises 
        else:
            task_noise_covar = self.likelihood.task_noise_covar
            task_noises_var = torch.diagonal(task_noise_covar)

        task_noises_var = task_noises_var.detach().numpy()
        task_noises_std = np.sqrt(task_noises_var)

        V_noises = task_noises_std[0]
        force_noises = task_noises_std[1:]

        return V_noises, force_noises 


def train_gpr(model:GPModelWithDerivatives , training_error_cutoff = np.power(10.0, -4)):
    '''
    the function that trains the model.
    model: GPytorch model 
    training_error_cutoff: train until the change of loss function is smaller
    The function annotation of model here should also allow using subclass of ExactGP class.
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

    while loss_func_change > training_error_cutoff:
        # reset the gradients of all optimized torch.Tensor 
        optimizer.zero_grad()   
        # output from model training data
        output = model(train_inputs)
        # calculate the loss function. here the returned loss is a torch.tensor.
        loss = - mll(output, train_targets)

        # calculate the change of loss function to decide whether we will stop the loop.
        loss_value = loss.item() 
        loss_func_change = np.abs(loss_value - old_loss_value)
        old_loss_value = loss_value 

        # backpropagation the loss function to compute the gradient of each parameter
        loss.backward()
        # optimizer optimize the parameter using the gradient info.
        optimizer.step()

        train_counts = train_counts + 1

        # for debug.
        # if train_counts % 10 == 0:
        #     print("Iter %d - Loss %.3f" %(train_counts, loss_value))
        #     print("mean_module constant: " + str(model.mean_module.constant))
        #     kernel_lengthscale = model.output_kernel_lengthscale()
        #     print("kernel lengthscale: " + str(kernel_lengthscale) )
        #     output_scale = model.output_kernel_outputscale()
        #     print("outputscale: " + str(output_scale))
        #     print("\n")


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
    :param: covar_bool: bool variable, decide whether output covariance matrix or variance (diagonal of covariance matrix)
    
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


def predict_observable_gp_with_derivative(model: GPModelWithDerivatives, test_inputs, covar_bool = False):
        '''
        the function t hat predict the posterior distribution observable y = f(X) + epsilon of the test inputs.

        :param: model: instance of GPModelWithDerivative. Gaussian process regression model using derivative information.
        :param: test_inputs: test data to compute posterior distribution. dtype: torch.tensor
        :param: covar_bool: bool variable, decide whether output covariance matrix or variance.

        suppose output_dim = m
        return: mean: mean value of the prediction for test inputs. shape: [N, m]

                if covar_bool == True: return: test_covariance: [N * m , N * m]
                if covar_bool == False: return: test_var: [N, m]: each row is the variance of one data point.
        '''
        test_input_dim = test_inputs.shape[-1]
        assert test_input_dim == model.input_dim, "dimension of input test_inputs data is incompatible with model"

        model.eval()
        likelihood = model.likelihood
        likelihood.eval() 

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observable = likelihood(model(test_inputs))

            data_num = test_inputs.shape[0]

            test_mean = observable.mean 
            test_covariance = observable.covariance_matrix

            # diagonal component of covariance matrix is the variance of function and gradient
            test_var = torch.diag(test_covariance)

            test_var = test_var.reshape([model.output_dim, data_num])
            test_var = torch.transpose(test_var, 0, 1)
        
        if covar_bool:
            return test_mean, test_covariance
        else:
            return test_mean, test_var 



def update_model_with_new_data(model : GPModelWithDerivatives, new_train_inputs, new_train_targets):
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

    full_train_inputs = torch.cat([train_inputs, new_train_inputs_tensor], dim = 0)
    full_train_targets = torch.cat([train_targets, new_train_targets_tensor], dim = 0)

    # set the training data for the model
    model.set_train_data(inputs= full_train_inputs, targets= full_train_targets, strict= False)

    # re-train the model to update the hyper-parameter 
    train_gpr(model)


class GPModelWithDerivativesWrapper():
    '''
    wrapper class for GPModelWithDerivatives. 
    handles the transformation between internal coordinate and cartesian coordinate + GPR training.
    '''
    def __init__(self, train_x, train_V, train_grad, 
                 natom, coordinate_transformer : non_redundant_coordinate_transformer,
                 gpr_SE_kernel_number,
                 kernel_initial_outputscale, kernel_prior_lengthscale_ratio,
                 likelihood_noise_std_constraint):
        '''
        initialize the model.
        :param: train_x: [N, 3 * natom]. initial N training points x. (Cartesian coordinate)  numpy array.
        :param: train_V: [N]. initial N training potential V.    numpy array.
        :param: train_grad: [N, 3 * natom], initial N training data. gradient of potential V.  numpy array.
        :param: natom: number of atoms.
        :param: coordinate transformer: an instance of the class: non_redundant_coordinate_transformer. Responsible for transformation between external and internal coordinate. 
        :param: gpr_SE_kernel_number: number of squared exponential kernel use to construct the covariance function.
        :param: kernel_initial_outputscale: output scale of each squared exponential kernel used to construct covariance function. numpy array. 
        :param: kernel_prior_lengthscale_ratio: length scale ratio of each squared exponential kernel used to construct covariance function. numpy array.
        :param: likelihood_noise_std_constraint: lower bound and upper bound for the noise of likelihood function p(y|f). 
                                                       y = f + epsilon, where the variance of epsilon noise term is defined by likelihood noise variance.
        '''
        assert np.shape(train_x)[1] == 3 * natom, "dim of coordinates for input data is not 3 * natom, this is wrong. train_x data shape: {} , 3 * natom: {}".format(np.reshape(train_x)[1], 3 * natom)
        assert np.shape(train_grad)[1] == 3 * natom, "dim of gradients for input data is not 3 * natom, this is wrong. train_grad shape:{}, 3 * natom: {}".format(np.shape(train_grad)[1], 3 * natom)
        
        # input data for machine learning model
        input_dim = 3 * natom - 6   # degree of freedom for molecule - 3 (translational dof) - 3(rotational dof)
        output_dim = 3 * natom - 5 # input_dim + 1 (grad dV/dx + potential V)

        self.coordinate_transformer = coordinate_transformer

        train_cartesian_targets = np.concatenate([train_V[:, np.newaxis], train_grad ], axis = 1)
        # transform cartesian coordinate x to internal coordinate q
        train_inputs = coordinate_transformer.get_internal_coordinate_q(train_x)

        # shape: [N, 3 * natom - 6]
        train_grad_q = coordinate_transformer.transform_cartesian_g_h_to_internal_g_h(train_x, train_grad, hessian_bool = False)
        # target data: [V, dV/dx1, ..., dV/dxn]
        train_targets = np.concatenate( [ train_V[:, np.newaxis] , train_grad_q ], axis = 1 )

        # decide normalization parameter
        self.compute_potential_normalization_parameter(train_targets)

        # perform normalization on training targets.
        # when making prediction, we need back transform of normalization.
        normalized_train_targets, normalized_train_inputs = self.normalization_transform(train_targets, train_inputs)

        # transform input from numpy array to torch.tensor
        normalized_train_inputs_tensor = torch.from_numpy(normalized_train_inputs)
        normalized_train_targets_tensor = torch.from_numpy(normalized_train_targets)

        # initialize the gaussian process regression model with inpt training data.
        self.gpr_model = GPModelWithDerivatives(normalized_train_inputs_tensor, normalized_train_targets_tensor, input_dim, output_dim,
                                                gpr_SE_kernel_number,
                                                kernel_initial_outputscale, kernel_prior_lengthscale_ratio, 
                                                likelihood_noise_std_constraint)

        # train self.gpr_model() to get optimized hyperparameter
        train_gpr(self.gpr_model)

        self.normalized_train_inputs = normalized_train_inputs  # training inputs in internal coordinate space q.
        self.normalized_train_targets = normalized_train_targets  # training outputs in internal coordinates q. (V, dV/dq)

        self.train_cartesian_inputs = train_x  # training inputs in cartesian coordinate x
        self.train_cartesian_targets = train_cartesian_targets  # training targets in cartesian coordinate (V, dV/dx)

        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.natom = natom
        self.gpr_SE_kernel_number = gpr_SE_kernel_number

    def compute_potential_normalization_parameter(self, training_targets):
        '''
        normalize the potential V & force F.
        V_normalized = (V - V_mean)/V_range.

        This function decide normalize parameter from the initial training data. 

        :param: training_targets : [V, dV/dx]. numpy array.
        '''
        V = training_targets[:, 0]
        self.V_mean = np.mean(V)
        self.V_range = np.max(V) - np.min(V)

        # further normalize the force after scale the potential V.
        f = training_targets[:, 1:]
        f_normalized1 = f / self.V_range 

        self.grad_V_normalized1_mean = np.mean(f_normalized1, axis = 0)
        self.grad_V_normalized1_range = np.max(f_normalized1, axis = 0) - np.min(f_normalized1, axis = 0)

    def normalization_transform(self, training_targets, training_inputs):
        '''
        normalize the potential V & force F.
        V_normalized = (V - V_mean) / V_range.  
        F_normalized = F / V_range.

        Then we scale force again to make sure force is also in the range [0,1], also scale the input q.

        This function perform the normalize procedure.  
        :param: training_targets : [V, dV/dq]. numpy array.
        :param: training_inputs: internal coordinate q.
        '''    
        # normalize the potential
        V = training_targets[:,0]
        V_normalized = (V - self.V_mean) / self.V_range 

        grad_V = training_targets[:,1:]
        grad_V_normalized1 = grad_V / self.V_range 

        # further normalize the gradient of potential by scale the length scale x.
        grad_V_normalized = (grad_V_normalized1 - self.grad_V_normalized1_mean) / self.grad_V_normalized1_range 
        normalized_training_inputs = training_inputs * self.grad_V_normalized1_range

        normalized_training_targets = np.concatenate( [ V_normalized[:, np.newaxis], grad_V_normalized ] , axis = 1 )

        return normalized_training_targets, normalized_training_inputs

    def normalization_transform_training_inputs(self, training_inputs):
        '''
        scale the training inputs for GPR model.
        This is like normalization_transform, but only scale the inputs.

        '''
        normalized_training_inputs = training_inputs * self.grad_V_normalized1_range

        return normalized_training_inputs

    def inverse_normalization_transform(self, normalized_training_targets):
        '''
        inverse the normalization procedure of potential V & force F.
        First scale the force:
        F_normalized1 = F_normalized * F_scale + F_mean

        Then we further scale the potential. 
        V = V_normalized * V_range + V_mean
        F = F_normalized1 * V_range 

        This function perform the inverse of normalization procedure.
        :param: normalized_training_targets: [V_normalized, d V_normalized /dx]. numpy array.
        :param: 
        '''
        V_normalized = normalized_training_targets[:, 0]
        grad_V_normalized = normalized_training_targets[:, 1:]

        # scale back the gradient of potential and internal coordinate
        grad_V_normalized1 = grad_V_normalized * self.grad_V_normalized1_range + self.grad_V_normalized1_mean

        # scale the potential and gradient of potential
        V = V_normalized * self.V_range + self.V_mean 
        grad_V = grad_V_normalized1 * self.V_range 

        training_targets = np.concatenate([ V[:, np.newaxis], grad_V ], axis = 1)

        return training_targets


    def predict_latent_function(self, test_x, internal_coordinate_bool = False):
        '''
        compute the predicted potential V and gradient dV/dx (mean value of latent prediction distribution) in Cartesian coordinate.
        Also compute the variance of potential & gradients dV/dq. 
        This function wraps predict_latent_function_gp_with_derivative.

        :param: test_x: input Cartesian coordinate data [N, 3 * natom]. 
        :param: internal_coordinate_bool: if internal coordinate bool = True, then we output gradient of internal coordinate.
        :return: V: predicted potential energy.
                grad_x: dV/dx, predicted gradient of potential energy. In Cartesian coordinate.
                var_V: uncertainty (variance) of potential energy.
                var_grad: variance of gradients along different internal coordinate. We can postprocess to get uncertainty about force prediction.
        '''
        assert np.shape(test_x)[1] == 3 * self.natom , "dim of coordinates for input data is not 3 * natom"

        # transform to internal coordinate q.
        test_q = self.coordinate_transformer.get_internal_coordinate_q(test_x)
        test_q_normalized = self.normalization_transform_training_inputs(test_q)
        test_q_normalized_tensor = torch.from_numpy(test_q_normalized)

        # use Gaussian process regression to make prediction 
        normalized_test_mean_tensor, normalized_test_var_tensor = predict_latent_function_gp_with_derivative(self.gpr_model, test_inputs = test_q_normalized_tensor, covar_bool = False)
        
        normalized_test_mean = normalized_test_mean_tensor.detach().cpu().numpy()
        normalized_test_var = normalized_test_var_tensor.detach().cpu().numpy()

        test_var = normalized_test_var * np.power(self.V_range , 2) 
        test_var[:,1:] = test_var[:,1:] * np.power(self.grad_V_normalized1_range, 2)
        test_mean = self.inverse_normalization_transform(normalized_test_mean)

        V = test_mean[:, 0]
        grad_q = test_mean[:, 1:]  # gradient dV/dq.
        
        # transform gradient from internal coordinate back to cartesian coordinate. 
        grad_x = self.coordinate_transformer.transform_internal_g_h_to_cartesian_g_h(test_x, grad_q, hessian_bool = False)

        var_V = test_var[:, 0]
        var_grad_q = test_var[:, 1:]

        if internal_coordinate_bool:
            return V, grad_q, var_V, var_grad_q 
        else:
            return V, grad_x, var_V, var_grad_q
    


    def predict_observable(self, test_x, internal_coordinate_bool = False):
        '''
        similar to predict_latent_function. But instead of output f(X) = (V, dV/dx) in predict_latent_function, we compute the observable y = f(X) + epsilon. (with noise)
        
        :param: test_x: input Cartesian coordinate data [N, 3 * natom]. 
        
        :return: V: predicted potential energy.
                grad_x: dV/dx, predicted gradient of potential energy. In Cartesian coordinate.
                var_V: uncertainty (variance) of potential energy.
                var_grad: variance of gradients along different internal coordinate. We can postprocess to get uncertainty about force prediction.
        '''
        assert np.shape(test_x)[1] == 3 * self.natom 

        # transform to internal coordinate q.
        test_q = self.coordinate_transformer.get_internal_coordinate_q(test_x)
        test_q_normalized = self.normalization_transform_training_inputs(test_q)
        test_q_normalized_tensor = torch.from_numpy(test_q_normalized)


        # use Gaussian process regression to make prediction 
        normalized_test_observable_mean_tensor, normalized_test_observable_var_tensor = predict_observable_gp_with_derivative(self.gpr_model, test_inputs = test_q_normalized_tensor, covar_bool = False)
        normalized_test_observable_mean = normalized_test_observable_mean_tensor.detach().cpu().numpy()
        normalized_test_observable_var = normalized_test_observable_var_tensor.detach().cpu().numpy() 

        test_observable_mean = self.inverse_normalization_transform(normalized_test_observable_mean)
        test_observable_var = normalized_test_observable_var * np.power(self.V_range, 2) 
        test_observable_var[:,1:] = test_observable_var[:,1:] * np.power(self.grad_V_normalized1_range, 2)

        V = test_observable_mean[:, 0]
        grad_q = test_observable_mean[:, 1:]

        # transform the gradient from internal coordinate q back to cartesian coordinate
        grad_x = self.coordinate_transformer.transform_internal_g_h_to_cartesian_g_h(test_x, grad_q, hessian_bool = False)

        var_V = test_observable_var[:, 0]
        var_grad_q = test_observable_var[:, 1:]

        if internal_coordinate_bool:
            return V, grad_q, var_V, var_grad_q 
        else:
            return V, grad_x, var_V, var_grad_q

    def update_model_with_new_data(self, new_train_x, new_train_V, new_train_grad):
        '''
        add new training data into the model. 
        Then train the model to update the hyper-parameter.
        This function wraps the function: update_model_with_new_data(gpr_model, train_inputs, train_targets)
        
        This function will update the self.gpr_model

        :param: new_train_x: [N, 3 * natom], input Cartesian coordinate data.  numpy array
                new_train_V: [N], ab-initio potential data.   numpy array
                new_train_grad: [N, 3 * natom], ab-initio force data.  numpy array.
        
        :return: None.
        '''
        assert np.shape(new_train_x)[1] == 3 * self.natom, "dim of coordinates for input data is not 3 * natom"
        assert np.shape(new_train_grad)[1] == 3 * self.natom, "dim of gradients for input data is not 3 * natom"

        # input data for machine learning model
        # internal coordinate
        new_train_inputs = self.coordinate_transformer.get_internal_coordinate_q(new_train_x)

        # gradient of potential in internal coordinate
        new_train_grad_q = self.coordinate_transformer.transform_cartesian_g_h_to_internal_g_h(new_train_x, new_train_grad, hessian_bool = False)
        assert np.shape(new_train_grad_q)[1] == 3 * self.natom - 6, "train_grad_q for internal coordiante has wrong dimension"
        new_train_targets = np.concatenate([ new_train_V[:,np.newaxis], new_train_grad_q ], axis = 1)

        # normalize the new_train_targets
        normalized_new_train_targets, normalized_new_train_inputs = self.normalization_transform(new_train_targets, new_train_inputs)
        normalized_new_train_targets_tensor = torch.from_numpy(normalized_new_train_targets)
        normalized_new_train_inputs_tensor = torch.from_numpy(normalized_new_train_inputs)

        update_model_with_new_data(self.gpr_model, normalized_new_train_inputs_tensor, normalized_new_train_targets_tensor)

        # update the training data and targets in internal coordinate q.
        self.normalized_train_inputs = np.concatenate([self.normalized_train_inputs, normalized_new_train_inputs], axis = 0)
        self.normalized_train_targets = np.concatenate([self.normalized_train_targets, normalized_new_train_targets], axis = 0)

        # update the training data and targets in cartesian coordinate x.
        new_train_cartesian_targets = np.concatenate([new_train_V[:,np.newaxis], new_train_grad] , axis = 1)
        self.train_cartesian_inputs = np.concatenate([self.train_cartesian_inputs, new_train_x], axis = 0)
        self.train_cartesian_targets = np.concatenate([self.train_cartesian_targets, new_train_cartesian_targets], axis = 0)

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
    
    def output_normalized_training_internal_inputs(self):
        '''
        output the training data set Q (in non-redundant internal coordinate) used to train the GPR model
        '''
        train_internal_q = np.copy(self.normalized_train_inputs)
        
        return train_internal_q
    
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
        normalized_force = self.normalized_train_targets[:,1:]
        normalized_force_range = np.max(normalized_force, axis = 0) - np.min(normalized_force, axis = 0)

        return normalized_force_range
    
    def output_train_input_range(self):
        '''
        output the input range of training data
        '''
        return np.copy(self.gpr_model.initial_train_inputs_lengthscale)