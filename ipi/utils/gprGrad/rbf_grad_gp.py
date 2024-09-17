import torch
import gpytorch
import numpy as np
from gpytorch import settings
from gpytorch.utils.generic import length_safe_zip
from gpytorch.distributions import MultivariateNormal
import warnings
from gpytorch.utils.warnings import GPInputWarning
from .rbf_grad_prediction_strategies import RBFGradPredictionStrategies
from .rbf_grad_marginal_log_likelihood import RBFGradMarginalLogLikelihood
from gpytorch.models.exact_prediction_strategies import prediction_strategy

class GPModelWithDerivatives(gpytorch.models.ExactGP):
    """
    Gaussian Process model with multiple output (f(x), df/dx1, .. , df/dxn)
    """

    def __init__(
        self,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        ard_num_dims: int,
        output_dims: int,
        gpr_SE_kernel_number: int,
        kernel_outputscale,
        kernel_lengthscale_ratio,
        likelihood_noise_variance,
    ):
        """
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
        """
        self.input_dim = ard_num_dims
        self.output_dim = output_dims

        # set the noise prior information and construct the likelihood class.
        likelihood = self._set_likelihood_noise_prior(
            output_dims, likelihood_noise_variance
        )

        super(GPModelWithDerivatives, self).__init__(
            train_inputs, train_targets, likelihood
        )

        # set the mean function for the Gaussian Processes
        self._set_mean_function(train_targets)

        # set covariance function (kernel) for Gaussian process regression:
        self._set_gpr_kernel(
            ard_num_dims,
            train_inputs,
            gpr_SE_kernel_number,
            kernel_outputscale,
            kernel_lengthscale_ratio,
        )

    def _set_likelihood_noise_prior(self, output_dims, likelihood_noise_variance):
        """
        set the prior distribution for the variance of noise (for both potential V and force F)
        :param: output_dims: dimension of the output target.
        :param: likelihood_noise_variance: The mean value for the distribution of the variance of the noise. We set the prior of the noise variance as a Gaussian distribution.  (A gaussian distribution prior on the variance of the gaussian distribution of noise.)
        """
        self.noise_rank = 0

        likelihood_noise_variance_mean = likelihood_noise_variance
        likelihood_noise_variance_std = (
            likelihood_noise_variance / 10
        )  # we set the std of the prior distribution as 1/10 of the mean value.

        noise_mean_tensor = torch.from_numpy(likelihood_noise_variance_mean)
        noise_std_tensor = torch.from_numpy(likelihood_noise_variance_std)

        # set the prior of the noise as a normal distribution.
        task_noise_prior = gpytorch.priors.NormalPrior(
            noise_mean_tensor, noise_std_tensor
        )

        # set the constraint of the variance of the noise to be 10 times larger or smaller than the prior mean value.
        noise_lower_bound_tensor = noise_mean_tensor.div(10)
        noise_upper_bound_tensor = noise_mean_tensor.mul(10)
        noise_constraint = gpytorch.constraints.Interval(
            noise_lower_bound_tensor, noise_upper_bound_tensor
        )

        # See documentation in https://docs.gpytorch.ai/en/stable/_modules/gpytorch/likelihoods/multitask_gaussian_likelihood.html#MultitaskGaussianLikelihood
        # here task_noise: the noise for each output target dimension.  global_noise: noise for all targets (we turn it off)
        # rank= 0: represents the covariance matrix of the task noise will be diagonal, which means noise of force along each dimension is independent.
        # noise_prior: set the prior for the variance of the noise.   noise_constraint: add constraint to the variance of noise. Otherwise, the code will use default constraint, which is not appropriate.
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            output_dims,
            rank=0,
            noise_prior=task_noise_prior,
            noise_constraint=noise_constraint,
            has_task_noise=True,
            has_global_noise=False,
        )

        # set initial value of task noises as the mean value of the prior.
        likelihood.task_noises = noise_mean_tensor

        return likelihood

    def _set_mean_function(self, train_targets):
        """
        set the mean function for the Gaussian Process Regression.
        set the initial value of the mean function as the mean of target potential V.
        """
        self.mean_module = (
            gpytorch.means.ConstantMeanGrad()
        )  # mean function for Gaussian Processes using gradient information
        # set initial value of mean constant
        train_target_func = (
            train_targets[:, 0].detach().numpy()
        )  # function f in training data (other data are gradient df/dx)
        mean_constant_estimate = np.mean(train_target_func)
        self.mean_module.constant = torch.nn.Parameter(
            torch.ones(1) * mean_constant_estimate
        )

    def _set_gpr_kernel(
        self,
        ard_num_dims,
        train_inputs,
        gpr_SE_kernel_number,
        kernel_outputscale,
        kernel_lengthscale_ratio,
    ):
        """
        set the kernel for the Gaussian Process Regression.
        kernel = (sigma_m) ^2 exp(- sum_i (x1_i - x2_i)^2 / l_i^2 )
        :param: ard_num_dims: number of input dimensions (ard: automatic relevance determination)
        :param: train_inputs: internal coordinate q for the training data.
        :param: gpr_SE_kernel_number:  We provide options to have multiple Squared Exponential kernel to sum together to form the kernel. Here gpr_SE_kernel_number is the number of Squared Exponential kernels.
        :param: kernel_outputscale: the output scale of the gpr kernel. Here kernel_outputscale = (sigma_m) ^2
        :param: kernel_lengthscale_ratio: l_i / |q_i^{max} - q_i^{min}|.  The ratio of the kernel length scale and the range of initial training data along dim i.
        """
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
            train_inputs_range = (
                torch.max(train_inputs, dim=0).values
                - torch.min(train_inputs, dim=0).values
            )
            length_scale = (
                kernel_lengthscale_ratio[i] * train_inputs_range
            )  # set it as a given ratio of the training input range.
            length_gamma_beta = torch.div(
                length_gamma_alpha, length_scale
            )  # value of beta: rate of the gamma distribution.

            output_scale = kernel_outputscale[i]

            # set prior for length scale and output scale
            lengthscale_prior = gpytorch.priors.GammaPrior(
                length_gamma_alpha, length_gamma_beta
            )
            outputscale_prior = gpytorch.priors.GammaPrior(
                output_gamma_alpha, output_gamma_alpha / output_scale
            )

            # also add length scale constraint: minimum cutoff to prevent over-fitting.
            length_scale_ratio_cutoff = 0.1
            length_scale_cutoff = length_scale_ratio_cutoff * train_inputs_range
            lengthscale_constraint = gpytorch.constraints.GreaterThan(
                length_scale_cutoff
            )

            # set Squared Exponential kernel function
            base_kernel = gpytorch.kernels.RBFKernelGrad(
                ard_num_dims=ard_num_dims,
                lengthscale_prior=lengthscale_prior,
                lengthscale_constraint=lengthscale_constraint,
            )

            covar_module = gpytorch.kernels.ScaleKernel(
                base_kernel, outputscale_prior=outputscale_prior
            )

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
        """
        forward function is used to define the model.  See https://docs.gpytorch.ai/en/stable/examples/00_Basic_Usage/Implementing_a_custom_Kernel.html
        forward function takes in some n*d input data (x) and returns a prior MultivariateNormal distribution with mean and covariance evaluated at x.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    # ------ functions below are auxiliary functions to output gpr model parameters ------------------------
    def output_kernel_lengthscale(self):
        """
        output length scale of the base kernel
        :return: length scale. shape: [SE_kernel_number, ard_num_dims]. numpy array.
        """
        # output length scale of all kernel component
        lengthscale_list = []
        for i in range(self.gpr_SE_kernel_number):
            length_scale = np.copy(
                self.base_kernel_component_list[i].lengthscale[0].detach().numpy()
            )
            lengthscale_list.append(length_scale)
        lengthscale_list = np.array(lengthscale_list)

        return lengthscale_list

    def output_kernel_outputscale(self):
        """
        output the output scale of the covar module
        :return: output covariance scale list. shape: [SE_kernel_number, output_dims]. numpy array
        """
        # output the outputscale of all kernel component
        outputscale_list = []
        for i in range(self.gpr_SE_kernel_number):
            output_scale = np.copy(
                self.covar_module_component_list[i].outputscale.detach().numpy()
            )
            outputscale_list.append(output_scale)
        outputscale_list = np.array(outputscale_list)

        return outputscale_list

    def output_fitted_noise(self):
        """
        output fitted noise for potential V and force f.
        """
        task_noises_var = self.likelihood.task_noises

        task_noises_var = task_noises_var.detach().numpy()
        task_noises_std = np.sqrt(task_noises_var)

        V_noises = task_noises_std[0]
        force_noises = task_noises_std[1:]

        return V_noises, force_noises


    # __call__ function in Gpytorch code. 
    # We need to change the prediction strategy to use pseudo-inverse when inverse the covariance matrix. 
    # This will help when the covariance matrix becomes ill-conditioned. 
    # The code is the same as in Gpytorch, the only changes we make is to use the new prediction strategy.
    # See: https://arxiv.org/abs/1602.00853
    def __call__(self, *args, **kwargs):
        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args]

        # Training mode: optimizing
        if self.training:
            if self.train_inputs is None:
                raise RuntimeError(
                    "train_inputs, train_targets cannot be None in training mode. "
                    "Call .eval() for prior predictions, or call .set_train_data() to add training data."
                )
            if settings.debug.on():
                if not all(
                    torch.equal(train_input, input) for train_input, input in length_safe_zip(train_inputs, inputs)
                ):
                    raise RuntimeError("You must train on the training inputs!")
            res = super().__call__(*inputs, **kwargs)
            return res

        # Prior mode
        elif settings.prior_mode.on() or self.train_inputs is None or self.train_targets is None:
            full_inputs = args
            full_output = super(GPModelWithDerivatives, self).__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            return full_output

        # Posterior mode
        else:
            if settings.debug.on():
                if all(torch.equal(train_input, input) for train_input, input in length_safe_zip(train_inputs, inputs)):
                    warnings.warn(
                        "The input matches the stored training data. Did you forget to call model.train()?",
                        GPInputWarning,
                    )

            # Get the terms that only depend on training data
            if self.prediction_strategy is None:
                train_output = super().__call__(*train_inputs, **kwargs)

                # Create the prediction strategy for
                # self.prediction_strategy = RBFGradPredictionStrategies(
                #     train_inputs=train_inputs,
                #     train_prior_dist=train_output,
                #     train_labels=self.train_targets,
                #     likelihood=self.likelihood,
                # )

                self.prediction_strategy = prediction_strategy(
                    train_inputs=train_inputs,
                    train_prior_dist=train_output,
                    train_labels=self.train_targets,
                    likelihood=self.likelihood,
                )

            # Concatenate the input to the training input
            full_inputs = []
            batch_shape = train_inputs[0].shape[:-2]
            for train_input, input in length_safe_zip(train_inputs, inputs):
                # Make sure the batch shapes agree for training/test data
                if batch_shape != train_input.shape[:-2]:
                    batch_shape = torch.broadcast_shapes(batch_shape, train_input.shape[:-2])
                    train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                if batch_shape != input.shape[:-2]:
                    batch_shape = torch.broadcast_shapes(batch_shape, input.shape[:-2])
                    train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                    input = input.expand(*batch_shape, *input.shape[-2:])
                full_inputs.append(torch.cat([train_input, input], dim=-2))

            # Get the joint distribution for training/test data
            full_output = super(GPModelWithDerivatives, self).__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix

            # Determine the shape of the joint distribution
            batch_shape = full_output.batch_shape
            joint_shape = full_output.event_shape
            tasks_shape = joint_shape[1:]  # For multitask learning
            test_shape = torch.Size([joint_shape[0] - self.prediction_strategy.train_shape[0], *tasks_shape])

            # Make the prediction
            with settings.cg_tolerance(settings.eval_cg_tolerance.value()):
                (
                    predictive_mean,
                    predictive_covar,
                ) = self.prediction_strategy.exact_prediction(full_mean, full_covar)

            # Reshape predictive mean to match the appropriate event shape
            predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()
            return full_output.__class__(predictive_mean, predictive_covar)


def train_gpr(model: GPModelWithDerivatives, training_error_cutoff=np.power(10.0, -3)):
    """
    the function that trains the model.
    :param: model: GPytorch model
    :param: training_error_cutoff: train until the change of loss function is smaller than the cutoff.

    :return: None
    """
    # set model & likelihood to the training mode
    model.train()
    likelihood = model.likelihood
    likelihood.train()

    # choose the optimizer for the training to train the parameter of models (raw_parameter)
    # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # define loss function for GPs. -- we choose the marginal log likelihood
    # because we need to maximise the marginal log likelihood, we should define the loss function as -mll
    
    # TODO: debug this one.
    # mll = RBFGradMarginalLogLikelihood(likelihood, model)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    train_inputs = model.train_inputs[
        0
    ]  # model.train_inputs is the tuple containing our training data.
    train_targets = model.train_targets

    # initialize loss_func_change and old_loss to enable while loop
    loss_func_change = 1000
    old_loss_value = 1000

    train_counts = 0

    loss_value_list = []
    loss_prior_list = []

    while loss_func_change > training_error_cutoff:
        # reset the gradients of all optimized torch.Tensor
        optimizer.zero_grad()
        # output from model training data
        output = model(train_inputs)
        # calculate the loss function. here the returned loss is a torch.tensor.
        loss = -mll(output, train_targets)
        loss_value = loss.item()
        loss_value_list.append(loss_value)

        if (
            loss_value > old_loss_value
            and abs((loss_value - old_loss_value) / old_loss_value) > 0.1
        ):
            print(
                "@WARNING: the training could be unstable. loss function increases:  {}   ->     {}".format(
                    old_loss_value, loss_value
                )
            )

        # calculate the change of loss function to decide whether we will stop the loop.
        loss_func_change = np.abs(loss_value - old_loss_value)
        old_loss_value = loss_value

        # compute loss from prior
        loss_prior = torch.tensor(0.0)
        loss_prior = -mll._add_other_terms(loss_prior, [])
        loss_prior_list.append(loss_prior.item())

        # backpropagation the loss function to compute the gradient of each parameter
        loss.backward()
        # optimizer optimize the parameter using the gradient info.
        optimizer.step()

        train_counts = train_counts + 1

    pass
