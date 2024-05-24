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
    def __init__(self, train_inputs, train_targets, ard_num_dims, output_dims):
        '''
        :param: train_inputs: training data.  torch.Tensor object. shape: [N, d]. N: number of data points. d: input data dimensions.
        :param: train_targets: training data.  torch.Tensor object. shape: [N, m]. N: number of data points. m: output data dimensions. (multiple output)
        :param: ard_num_dims: input data dimension (d)
        :param: num_tasks: number of tasks for multi-dimensional output. 
        note: For training all gradient in d dimension, we should set num_tasks = ard_num_dims + 1  (f, df/dx1, .. , df/dx_d )

        We can access train_x, train_y, likelihood later as : self.train_inputs, self.train_targets, self.likelihood.
        '''
        # likelihood: gpytorch.likelihood object. likelihood of observable given prediction f(X):  P(y|f(X)). See:  https://docs.gpytorch.ai/en/stable/likelihoods.html
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(output_dims)

        super(GPModelWithDerivatives, self).__init__(train_inputs, train_targets, likelihood)

        self.input_dim = ard_num_dims
        self.output_dim = output_dims 

        # mean function for prior distribution of Gaussian Processes
        self.mean_module = gpytorch.means.ConstantMeanGrad()  # mean function for Gaussian Processes using gradient information
        
        # kernel function. base kernel before adding outputscaling 
        self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims = ard_num_dims)

        # kernel function. adding outputscale parameter to base_kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel) 
    
    def forward(self, x):
        '''
        forward function is used to define the model.  See https://docs.gpytorch.ai/en/stable/examples/00_Basic_Usage/Implementing_a_custom_Kernel.html
        forward function takes in some n*d input data (x) and returns a prior MultivariateNormal distribution with mean and covariance evaluated at x.
        '''
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def train_gpr(model:gpytorch.models.ExactGP , training_error_cutoff = 0.001):
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
    
    train_inputs = model.train_inputs
    train_targets = model.train_targets

    # initialize loss_func_change and old_loss to enable while loop
    loss_func_change = 1000
    old_loss_value = 1000

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
    

    pass 

def predict_latent_function_gp_with_derivative(model:GPModelWithDerivatives, test_inputs, covar_bool = False):
    '''
    the function that predict the posterior distribution latent function f(test_inputs) of the test_inputs.
    
    :param: model: instance of GPModelWithDerivatives. Gaussian process regression model using derivative information.
    :param: test_inputs:  test data to compute posterior distribution
    :param: covar_bool: bool variable to wether output covariance matrix.
    
    return: mean: mean value of prediction for test_inputs

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
        
        # diagonal component of variance information.
        # variance of function and gradient
        test_var =  torch.diag(test_covariance)

        test_var = test_var.reshape([model.output_dim, data_num])  # first row is variance for f,  second row is variance for df/dx1, third row: df/dx2, .. 
        test_var = torch.transpose(test_var, -2, -1)  # now each row is one data piont.

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
    train_inputs = model.train_inputs
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
    def __init__(self, train_x, train_V, train_grad, natom, coordinate_transformer : non_redundant_coordinate_transformer):
        '''
        initialize the model.
        :param: train_x: [N, 3 * natom]. initial N training points x. (Cartesian coordinate)
        :param: train_V: [N]. initial N training potential V.
        :param: train_grad: [N, 3 * natom], initial N training data. gradient of potential V. 
        :param: natom: number of atoms.
        :param: coordinate transformer: an instance of the class: non_redundant_coordinate_transformer. Responsible for transformation between external and internal coordinate. 
        '''
        assert np.shape(train_x)[1] == 3 * natom, "dim of coordinates for input data is not 3 * natom, this is wrong. train_x data shape: {} , 3 * natom: {}".format(np.reshape(train_x)[1], 3 * natom)
        assert np.shape(train_grad)[1] == 3 * natom, "dim of gradients for input data is not 3 * natom, this is wrong. train_grad shape:{}, 3 * natom: {}".format(np.shape(train_grad)[1], 3 * natom)
        
        # input data for machine learning model
        input_dim = 3 * natom - 6   # degree of freedom for molecule - 3 (translational dof) - 3(rotational dof)
        output_dim = 3 * natom - 5 # input_dim + 1 (grad dV/dx + potential V)

        self.coordinate_transformer = coordinate_transformer

        # transform cartesian coordinate x to internal coordinate q
        
 
        



