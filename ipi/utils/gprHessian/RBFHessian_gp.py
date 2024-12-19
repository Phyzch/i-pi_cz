"""
Contains classes for Gaussian Process Regression model, which is capable of predicting hessians.
The code is adapted from GPytorch package: https://gpytorch.ai/. This code is based on version v1.12.
Written by Chenghao Zhang & Niri Govind, Pacific Northwest National Laboratory (chenghao.zhang@pnnl.gov), 2024.
"""

from .RBFHessianKernel import RBFKernelHessian
from .RBFHessianMean import ConstantMeanHessian, MeanWithPotGradHessian
from .RBFHessian_prediction_strategy import RBFHessianPredictionStrategy
from .RBFHessian_gaussian_likelihood import RBFHessianGaussianLikelihood
from .RBFHessian_marginal_log_likelihood import CustomMarginalLogLikelihood
from .RBFHessian_utils import transform_1d_train_targets_into_pots_grads_hessians
import torch
import gpytorch
import numpy as np
from gpytorch import settings
from gpytorch.utils.generic import length_safe_zip
from gpytorch.distributions import MultivariateNormal
import warnings
from gpytorch.utils.warnings import GPInputWarning


class GPModelWithHessians(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        training_data_hessian_data_point_index: torch.Tensor,
        hessian_fixdofs: torch.Tensor,
        gpr_SE_kernel_number: int,
        kernel_outputscale: np.ndarray,
        kernel_lengthscale_ratio: np.ndarray,
        likelihood_pot_noise_var: np.ndarray,
        likelihood_force_noise_var: np.ndarray,
        likelihood_hessian_noise_var: np.ndarray,
        likelihood_force_noise_rank: int,
        likelihood_hessian_noise_rank: int,
        noise_covar_factor_pot_grad_array: torch.Tensor,
        noise_covar_factor_with_hessian_array: torch.Tensor,
        kernel_lengthscale_initio_value: np.ndarray = np.array([]),
        kernel_outputscale_initio_value: np.ndarray = np.array([]),
        constant_mean_func_bool=True,
        ref_mean_coordinate: torch.Tensor = torch.Tensor([]),
        ref_mean_pot: torch.Tensor = torch.Tensor([]),
        ref_mean_grad: torch.Tensor = torch.Tensor([]),
        ref_mean_hessian: torch.Tensor = torch.Tensor([]),
    ):
        """
        :param: train_inputs: input coordinate of training data.
                shape: [M, ard_num_dims]
        :param: train_targets: 1d targets of training data.
        :param: training_data_hessian_data_point_index:
                the indices of data points that contain hessian information.
        :param: hessian_fixdofs:
                the index of hessian dofs that need to be excluded from modeling.
                (In current implementation, it is empty)
        :param: gpr_SE_kernel_number: number of squared exponential kernel for GPR model.
        :param: kernel_outputscale:
                shape: [gpr_SE_kernel_number]   Estimation of the output scale of kernel
        :param: kernel_lengthscale_ratio:
                shape: [gpr_SE_kernel_number, ard_num_dims]
                Estimation of the ratio between length scale of kernel and range of input data.
        :param: likelihood_pot_noise_var:
                shape: [1]. Estimation of the variance of the potential noise.
        :param: likelihood_force_noise_var:
                shape: [ard_num_dims]: Estimation of the variance of the force noise.
        :param: likelihood_hessian_noise_var:
                shape: [hessian_triu_size].
                Estimation of the variance of the hessian noise.
        :param: likelihood_force_noise_rank:
                The rank of covar factor of force noises.
                This is equal to number of degrees of freedoms of force in Cartesian coordinate.
        :param: likelihood_hessian_noise_rank:
                The rank of covar factor of hessian noises.
                This is equal to the number of upper triangle components of hessian matrix in Cartesian coordinate.
        :param: noise_covar_factor_pot_grad_array:
                shape: [M, 1 + ard_num_dims, 1 + force_noise_rank]
                the covariance factor matrix that transform noise in Cartesian coordinate into internal coordiante.
                (only transform gradient noise).
        :param: noise_covar_factor_with_hessian_array:
                shape: [M_H, 1 + ard_num_dims + hessian_triu_size, 1 + force_noise_rank + hessian_noise_rank]
                the covariance factor matrix that transform noise in Cartesian coordinate into internal coordinate
                (include gradient and hessian noise).
        :param: kernel_lengthscale_initio_value:
                If set, we will initialize the length scale of kernel as this value.
        :param: kernel_outputscale_initio_value:
                If set, we will initialize the output scale of kernel as this value.
        :param: constant_mean_func_bool:
                If true, we will set the mean function of GPR model as function with constant value & zero gradient / hessians.
                Otherwise, it will be Taylor expansion around ref point to second order.
        :param: ref_mean_x, ref_mean_V, ref_mean_grad_x, ref_mean_hessian_x:
                this is the coordinate / V / gradient / hessians of reference point which be used to set mean function of GPR model.
        """
        # the data point index that contains the hessian information.
        self.training_data_hessian_data_point_index = (
            training_data_hessian_data_point_index
        )

        # dofs that we will not include in hessian calculations.
        self.hessian_fixdofs = hessian_fixdofs
        ard_num_dims = train_inputs.shape[-1]
        data_num = train_inputs.shape[-2]

        self.ard_num_dims = ard_num_dims

        # number of active degrees of freedom.
        nactive = ard_num_dims - len(hessian_fixdofs)
        # the number of upper triangle components in hessian matrices.
        hessian_triu_size = int(nactive * (nactive + 1) / 2)

        self.hessian_triu_size = hessian_triu_size
        # the length of target data.
        target_len = data_num * (ard_num_dims + 1) + hessian_triu_size * len(
            training_data_hessian_data_point_index
        )
        assert len(train_targets) == target_len, "the length of target data is wrong."

        # set the likelihood function for Gaussian Process Regression model. Likelihood function describe the noise in data.
        likelihood = self._set_likelihood_noise_prior(
            train_inputs,
            likelihood_pot_noise_var,
            likelihood_force_noise_var,
            likelihood_hessian_noise_var,
            likelihood_force_noise_rank,
            likelihood_hessian_noise_rank,
            noise_covar_factor_pot_grad_array,
            noise_covar_factor_with_hessian_array,
        )

        super(GPModelWithHessians, self).__init__(
            train_inputs, train_targets, likelihood
        )

        # set the mean function of GPR model. It will be either (1) constant value function (2) function as Taylor expansion of potential around a reference point.
        self._set_mean_function(
            train_inputs,
            train_targets,
            constant_mean_func_bool,
            ref_mean_coordinate,
            ref_mean_pot,
            ref_mean_grad,
            ref_mean_hessian,
        )

        # set the covariance function (kernel) for Gaussian Process regression.
        self._set_gpr_kernel(
            train_inputs,
            gpr_SE_kernel_number,
            kernel_outputscale,
            kernel_lengthscale_ratio,
            kernel_lengthscale_initio_value,
            kernel_outputscale_initio_value,
        )

    def _set_mean_function(
        self,
        train_inputs,
        train_targets,
        constant_mean_bool,
        ref_mean_coordinate,
        ref_mean_pot,
        ref_mean_grad,
        ref_mean_hessian,
    ):
        """
        set the mean function for the Gaussian Process Regression.
        If constant_mean_bool = True, we will set the mean function as constant potential function.
        If constant_mean_bool = False, we will set the mean function as Taylor expansion around the reference point to second order.
        """
        data_num = train_inputs.shape[-2]

        if constant_mean_bool:
            mean_constant_estimate = torch.mean(
                train_targets[..., :data_num], dim=-1
            )  # the mean value of the potential.
            self.mean_module = ConstantMeanHessian()
            self.mean_module.constant = mean_constant_estimate  # set the constant (size 1) as mean value of prior
        else:
            assert ref_mean_pot.shape[0] == 1
            assert ref_mean_grad.shape[0] == self.ard_num_dims
            assert ref_mean_hessian.shape[0] == self.hessian_triu_size
            self.ref_mean_coordinate = ref_mean_coordinate
            # set the mean function as Taylor expansion around the reference point.
            self.mean_module = MeanWithPotGradHessian(
                ref_mean_coordinate,
                ref_mean_pot,
                ref_mean_grad,
                ref_mean_hessian,
                grad_size=self.ard_num_dims,
                hessian_triu_size=self.hessian_triu_size,
            )

    def _set_gpr_kernel(
        self,
        train_inputs,
        gpr_SE_kernel_number,
        kernel_outputscale,
        kernel_lengthscale_ratio,
        kernel_lengthscale_initio_value,
        kernel_outputscale_initio_value,
    ):
        """
        set the kernel for the Gaussian Process Regression.
        We set constraint and prior for lengthscale and output scale parameter in the model.
        If kernel_lengthscale_initio_value is given (typically inherit from previous gpr model training), we will use this value.
        Otherwise, we use the length scale value computed from kernel_length_scale_ratio.
        """
        self.gpr_SE_kernel_number = gpr_SE_kernel_number

        ard_num_dims = train_inputs.shape[-1]

        covar_module_component_list = []
        base_kernel_component_list = []

        # we choose Gamma distribution as prior distribution for output scale and length scale.
        # See https://docs.gpytorch.ai/en/stable/priors.html  &  https://www.wikiwand.com/en/Gamma_distribution
        # alpha parameter (shape parameter) for the gamma distribution of the length and outputscale.
        length_gamma_alpha = 3.0
        output_gamma_alpha = 3.0

        for i in range(gpr_SE_kernel_number):
            # The prior distribution of the length scale of the parameter is decided by the initial training inputs (we only provides the ratio).
            # this is bad for cross validation, but for simply training model, it works fine.
            train_inputs_range = (
                torch.max(train_inputs, dim=0).values
                - torch.min(train_inputs, dim=0).values
            )
            length_scale = kernel_lengthscale_ratio[i] * train_inputs_range
            length_gamma_beta = torch.div(length_gamma_alpha, length_scale)

            output_scale = kernel_outputscale[i]

            # set prior for lengthscale and outputscale
            lengthscale_prior = gpytorch.priors.GammaPrior(
                length_gamma_alpha, length_gamma_beta
            )
            outputscale_prior = gpytorch.priors.GammaPrior(
                output_gamma_alpha, output_gamma_alpha / output_scale
            )

            # add lengthscale constraint
            length_scale_ratio_cutoff = 0.1
            length_scale_cutoff = length_scale_ratio_cutoff * train_inputs_range
            lengthscale_constraint = gpytorch.constraints.GreaterThan(
                length_scale_cutoff
            )

            # set Squared exponential kernel function which also includes hessian data.
            # This kernel assume data is 1d data, where we compress potential V, gradient g and hessian h into 1d.
            base_kernel = RBFKernelHessian(
                ard_num_dims=ard_num_dims,
                lengthscale_prior=lengthscale_prior,
                lengthscale_constraint=lengthscale_constraint,
                hessian_fixdofs=self.hessian_fixdofs,
            )

            covar_module = gpytorch.kernels.ScaleKernel(
                base_kernel, outputscale_prior=outputscale_prior
            )

            # Initialize lengthscale and outputscale to the mean of priors. Or use the value specified by users.
            if len(kernel_lengthscale_initio_value) == 0:
                covar_module.base_kernel.lengthscale = lengthscale_prior.mean
            else:
                if len(kernel_lengthscale_initio_value[i]) == len(
                    covar_module.base_kernel.lengthscale[0]
                ):
                    # the size of initio value chosen by users match the size of kernel in the model, we set the value
                    covar_module.base_kernel.lengthscale[0] = torch.tensor(
                        kernel_lengthscale_initio_value[i]
                    )
                else:
                    print(
                        "@Warning: GPRHessian model: the initio length scale value (from previous GPR model) does not match the shape of the model, \
                          we will still use the initio value set in the input file."
                    )
                    covar_module.base_kernel.lengthscale = lengthscale_prior.mean

            if len(kernel_outputscale_initio_value) == 0:
                covar_module.outputscale = outputscale_prior.mean
            else:
                covar_module.outputscale = torch.tensor(
                    kernel_outputscale_initio_value[i]
                )

            base_kernel_component_list.append(base_kernel)
            covar_module_component_list.append(covar_module)

        self.base_kernel_component_list = base_kernel_component_list
        self.covar_module_component_list = covar_module_component_list

        # sum of Squared Exponential Covariance function. This will be the covariance function.
        self.covar_module = self.covar_module_component_list[0]
        for i in range(1, gpr_SE_kernel_number):
            self.covar_module = self.covar_module + self.covar_module_component_list[i]

    def _set_likelihood_noise_prior(
        self,
        train_inputs,
        likelihood_pot_noise_var,
        likelihood_force_noise_var,
        likelihood_hessian_noise_var,
        likelihood_force_noise_rank,
        likelihood_hessian_noise_rank,
        noise_covar_factor_pot_grad_array,
        noise_covar_factor_with_hessian_array,
    ):
        """
        set the prior and constraint for the noise of GPR model.
        The information will be contained in likelihood class: RBFHessianGaussianLikelihood.
        """
        ard_num_dims = train_inputs.shape[-1]
        batch_shape = train_inputs.shape[:-2]

        nactive = ard_num_dims - len(self.hessian_fixdofs)
        hessian_triu_size = int((nactive + 1) * nactive / 2)

        # First: check the shape of the potential noise and force noise
        assert (
            likelihood_pot_noise_var.shape[0] == 1
        ), "the shape of potential noise in GPR model is wrong. The current shape is {}, the right shape is {}".format(
            likelihood_pot_noise_var.shape[0], 1
        )
        assert (
            likelihood_force_noise_var.shape[0] == 1
        ), "the shape of the force noise in GPR model is wrong. The current shape is {}, the right shape is {}".format(
            likelihood_force_noise_var.shape[0], 1
        )
        assert (
            likelihood_hessian_noise_var.shape[0] == 1
        ), "the shape of hessian noise in GPR model is wrong. The current shape is {}, the right shape is {}".format(
            likelihood_hessian_noise_var.shape[0], 1
        )

        if len(noise_covar_factor_with_hessian_array) != 0:
            assert (
                noise_covar_factor_with_hessian_array[0].shape[0]
                == 1 + ard_num_dims + hessian_triu_size
            ), "the row size of noise_covar_factor is wrong."
            assert (
                noise_covar_factor_with_hessian_array[0].shape[1]
                == 1 + likelihood_force_noise_rank + likelihood_hessian_noise_rank
            ), "the column size of noise_covar_factor is wrong."
        if len(noise_covar_factor_pot_grad_array) != 0:
            assert (
                noise_covar_factor_pot_grad_array[0].shape[0] == 1 + ard_num_dims
            ), "the row size of noise_covar_factor is wrong."
            assert (
                noise_covar_factor_pot_grad_array[0].shape[1]
                == 1 + likelihood_force_noise_rank
            ), "the column size of noise_covar_factor is wrong."

        self.likelihood_force_noise_rank = likelihood_force_noise_rank
        self.likelihood_hessian_noise_rank = likelihood_hessian_noise_rank

        # pot noise prior and pot noise constraint
        pot_noise_mean = torch.from_numpy(likelihood_pot_noise_var)
        pot_noise_std = torch.from_numpy(likelihood_pot_noise_var / 10)
        pot_noise_prior = gpytorch.priors.NormalPrior(pot_noise_mean, pot_noise_std)

        pot_noise_lower_bound = pot_noise_mean.div(10)
        pot_noise_upper_bound = pot_noise_mean.mul(10)
        pot_noise_constraint = gpytorch.constraints.Interval(
            pot_noise_lower_bound, pot_noise_upper_bound
        )

        # force noise prior and force noise constraint:
        force_noise_mean = torch.from_numpy(likelihood_force_noise_var)
        force_noise_std = torch.from_numpy(likelihood_force_noise_var / 10)
        force_noise_prior = gpytorch.priors.NormalPrior(
            force_noise_mean, force_noise_std
        )

        force_noise_lower_bound = force_noise_mean.div(10)
        force_noise_upper_bound = force_noise_mean.mul(10)
        force_noise_constraint = gpytorch.constraints.Interval(
            force_noise_lower_bound, force_noise_upper_bound
        )

        # hessian noise prior and noise constraint:
        hessian_noise_mean = torch.from_numpy(likelihood_hessian_noise_var)
        hessian_noise_std = torch.from_numpy(likelihood_hessian_noise_var / 10)
        hessian_noise_prior = gpytorch.priors.NormalPrior(
            hessian_noise_mean, hessian_noise_std
        )

        hessian_noise_lower_bound = hessian_noise_mean.div(10)
        hessian_noise_upper_bound = hessian_noise_mean.mul(10)
        hessian_noise_constraint = gpytorch.constraints.Interval(
            hessian_noise_lower_bound, hessian_noise_upper_bound
        )

        # likelihood function
        likelihood = RBFHessianGaussianLikelihood(
            ard_num_dims,
            hessian_triu_size,
            batch_shape,
            pot_noise_prior,
            pot_noise_constraint,
            force_noise_prior,
            force_noise_constraint,
            hessian_noise_prior,
            hessian_noise_constraint,
            has_covar_factor=True,
            noise_covar_factor_pot_grad_array=noise_covar_factor_pot_grad_array,
            noise_covar_factor_with_hessian_array=noise_covar_factor_with_hessian_array,
            grad_covar_factor_rank=likelihood_force_noise_rank,
            hessian_covar_factor_rank=likelihood_hessian_noise_rank,
        )

        # set the initial value of pot noise, force noise and hessian noise
        likelihood.pot_noises = pot_noise_mean
        likelihood.force_noises = force_noise_mean
        likelihood.hessian_noises = hessian_noise_mean

        return likelihood

    def forward(self, x, inputs_hessian_data_point_index=torch.tensor([]), **kwargs):
        """
        return the distribution of the training targets.
        The mean function will be given by mean module.
        The covariance of Gaussian distribution will be given by covar_module.
        Following the convention of pytorch (&Gpytorch), the __call__() function will call __forward__() function to get distribution of target data.

        :param: inputs_hessian_data_point_index: the index of data that contain hessian information in the input data.
        """
        nactive = self.ard_num_dims - len(self.hessian_fixdofs)
        mean_x = self.mean_module(
            x, hessian_data_point_index=inputs_hessian_data_point_index, nactive=nactive
        )
        with settings.lazily_evaluate_kernels(False):
            covar_x = self.covar_module(
                x,
                x,
                hessian_data_point_index_1=inputs_hessian_data_point_index,
                hessian_data_point_index_2=inputs_hessian_data_point_index,
            )

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, *args, **kwargs) -> MultivariateNormal:
        """
        Adapted from __call__() function in gpytorch/models/exact_gp.py/ExactGP class
        *args are new input data (either training inputs or test inputs)
        **kwargs: key word arguments.
        :param: inputs_hessian_data_point_index is used for covariance matrix (kernel) evaluation. (see forward() function).
        We need to provide inputs_hessian_data_point_index for the forward function when we call __call__().
        """
        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        inputs = [
            i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args
        ]  # make inputs data have 2 dimensions.

        # Training mode: optimizing
        if self.training:
            if self.train_inputs is None:
                raise RuntimeError(
                    "train_inputs, train_targets cannot be None in training mode. "
                    "Call .eval() for prior predictions, or call .set_train_data() to add training data."
                )

            if settings.debug.on():
                if not all(
                    torch.equal(train_input, input)
                    for train_input, input in length_safe_zip(train_inputs, inputs)
                ):
                    raise RuntimeError("You must train on the training inputs!")

            res = gpytorch.module.Module.__call__(
                self,
                *inputs,
                inputs_hessian_data_point_index=self.training_data_hessian_data_point_index
            )  # this will call the forward() function. hessian_data_point_index is in **kwargs.
            return res

        # Prior mode
        elif (
            settings.prior_mode.on()
            or self.train_inputs is None
            or self.train_targets is None
        ):
            full_inputs = args

            inputs_hessian_data_point_index = kwargs.get(
                "inputs_hessian_data_point_index"
            )
            if inputs_hessian_data_point_index is None:
                raise RuntimeError(
                    "Must provide inputs_hessian_data_point_index for computing kernel."
                )
            if not isinstance(inputs_hessian_data_point_index, torch.Tensor):
                raise RuntimeError(
                    "The inputs_hessian_data_point_index must be a tensor."
                )

            full_output = gpytorch.module.Module.__call__(self, *full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError(
                        "GPModelWithHessian.forward method must return a MultivariateNormal"
                    )

            return full_output

        # Posterior mode: Compute the posterior prediction of the GPR model.
        else:
            inputs_hessian_data_point_index = kwargs.get(
                "inputs_hessian_data_point_index"
            )
            if inputs_hessian_data_point_index is None:
                raise RuntimeError(
                    "Must provide inputs_hessian_data_point_index for computing kernel."
                )
            if not isinstance(inputs_hessian_data_point_index, torch.Tensor):
                raise RuntimeError(
                    "The inputs_hessian_data_point_index must be a tensor."
                )

            if all(
                torch.equal(train_input, input)
                for train_input, input in length_safe_zip(train_inputs, inputs)
            ):
                warnings.warn(
                    "The input matches the stored training data. Did you forget to call model.train()?",
                    GPInputWarning,
                )

            # make the prediction:
            # Get the terms that only depend on training data
            if self.prediction_strategy is None:
                train_outputs = gpytorch.module.Module.__call__(
                    self,
                    *train_inputs,
                    inputs_hessian_data_point_index=self.training_data_hessian_data_point_index
                )

                # Create the prediction strategy
                self.prediction_strategy = RBFHessianPredictionStrategy(
                    train_inputs=train_inputs[0],
                    train_prior_dist=train_outputs,
                    train_labels=self.train_targets,
                    likelihood=self.likelihood,
                    training_data_hessian_data_point_index=self.training_data_hessian_data_point_index,
                    hessian_fixdofs=self.hessian_fixdofs,
                )

            # Concatenate the training input and test input into one input for generating the joint distribution
            full_inputs = []
            batch_shape = train_inputs[0].shape[:-2]
            for train_input, input in length_safe_zip(train_inputs, inputs):
                # Make sure the batch shapes agree for training/test data
                if batch_shape != train_input.shape[:-2]:
                    batch_shape = torch.broadcast_shapes(
                        batch_shape, train_input.shape[:-2]
                    )
                    train_input = train_input.expand(
                        *batch_shape, *train_input.shape[-2:]
                    )
                if batch_shape != input.shape[:-2]:
                    batch_shape = torch.broadcast_shapes(batch_shape, input.shape[:-2])
                    train_input = train_input.expand(
                        *batch_shape, *train_input.shape[-2:]
                    )
                    input = input.expand(*batch_shape, *input.shape[-2:])
                full_inputs.append(torch.cat([train_input, input], dim=-2))

            # Get the joint distribution for training / test data
            # the hessian data point index in test data should be shifted by number of training data when we compute covariance matrix of full inputs (training input + test input).
            inputs_hessian_data_point_index_in_full_input = (
                inputs_hessian_data_point_index + train_inputs[0].shape[-2]
            )
            full_inputs_hessian_data_point_index = torch.cat(
                (
                    self.training_data_hessian_data_point_index,
                    inputs_hessian_data_point_index_in_full_input,
                )
            ).to(torch.int32)

            full_output = gpytorch.module.Module.__call__(
                self,
                *full_inputs,
                inputs_hessian_data_point_index=full_inputs_hessian_data_point_index
            )
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError(
                        "ExactGP.forward must return a MultivariateNormal"
                    )
            full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix

            # Make the prediction of test data.
            with settings.cg_tolerance(
                settings.eval_cg_tolerance.value()
            ) and settings.fast_pred_var(True):
                (
                    predictive_mean,
                    predictive_covar,
                ) = self.prediction_strategy.exact_prediction(
                    full_mean,
                    full_covar,
                    inputs_hessian_data_point_index,
                    inputs[0].shape[-2],
                )

            return full_output.__class__(predictive_mean, predictive_covar)


def train_gpr_model(
    model: GPModelWithHessians,
    training_error_cutoff=np.power(10.0, -1),
    output_training_info=True,
):
    """
    the function that train the GPR model.
    :param: model: GPR model with Hessian information.
    :param: training_error_cutoff: train until the change of loss function in one step is smaller than the cutoff.
    :return: None
    """
    # set model & likelihood to the training mode
    likelihood = model.likelihood
    model.train()
    likelihood.train()

    train_inputs = model.train_inputs[0]
    train_targets = model.train_targets

    # number of total training data points : M.
    # number of data points containing hessian information: M_H.
    M = train_inputs.shape[-2]

    # choose the optimizer for the training to train the parameter of models (raw_parameter)
    # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # define loss function for GPs. -- we choose the marginal log likelihood
    # because we need to maximise the marginal log likelihood, we should define the loss function as -mll
    # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    mll = CustomMarginalLogLikelihood(likelihood, model)

    # initialize loss_func_change and old_loss to enable while loop
    loss_func_change = 1000
    old_loss_value = 1000

    train_counts = 0
    train_counts_output = 20

    loss_value_list = []
    loss_prior_list = []
    loss_mll_list = []

    while loss_func_change > training_error_cutoff:
        # reset the gradients of all optimized torch.Tensor
        optimizer.zero_grad()
        # output from model training data
        output = model(train_inputs)
        # calculate the loss function. here the returned loss is a torch.tensor.
        loss = -mll(
            output, train_targets, M, model.training_data_hessian_data_point_index
        )
        loss_value = loss.item()

        loss_prior = torch.tensor(0.0)
        loss_prior = -mll._add_other_terms(loss_prior, []) / M
        loss_prior_list.append(loss_prior.item())

        # loss function from probability distribution. No contribution from prior. 
        loss_mll = loss_value - loss_prior
        loss_mll_list.append(loss_mll)

        # calculate the change of loss function to decide whether we will stop the loop.
        loss_func_change = np.abs(loss_value - old_loss_value)
        old_loss_value = loss_value

        loss_value_list.append(loss_value)
        # back propagation the loss function to compute the gradient of each parameter
        loss.backward()

        # optimizer optimize the parameter using the gradient info.
        optimizer.step()

        train_counts = train_counts + 1

        if output_training_info:
            if train_counts % train_counts_output == 0:
                print("Iter %d - Loss: %.3f" % (train_counts, loss.item()))

    if output_training_info:
        print("Iter %d - Loss: %.3f" % (train_counts, loss.item()))
        print(f"loss_value_list: {loss_value_list}")
        
    pass


def predict_latent_function_GPHessian(
    model: GPModelWithHessians,
    test_inputs: torch.Tensor,
    test_data_hessian_data_point_index: np.ndarray,
):
    """
    predict the latent function (Gaussian distribution) for test data.
    Extract mean value (prediction) of potential, gradient and hessian from distribution.
    Extract variance of potential, gradient and hessian from distribution.

    :param: model: Gaussian Process Regression model capable of predicting hessians.
    :param: test_inputs: test inputs data.
    :param: test_data_hessian_data_point_index: index in test data inputs that

    :return:  All return data are numpy array.
              pots: [test_data_num]. mean value of posterior prediction for potentials.
              grads: [test_data_num, ndofs]. mean value of posterior prediction for gradients.
              hessians: [test_data_with_hessian_number, ndofs, ndofs]. mean value of posterior prediction for hessians.
              pots_var: [test_data_num]. variance of posterior prediction for potentials.
              grads_var: [test_data_num, ndofs]. variance of posterior predictions for gradients.
              hessians_var: [test_data_with_hessian_number, ndofs, ndofs]: variance of posterior predictions for hessians.
    """
    test_data_hessian_data_point_index_tensor = torch.from_numpy(
        test_data_hessian_data_point_index
    )

    test_data_num = test_inputs.shape[0]
    test_data_with_hessian_number = len(test_data_hessian_data_point_index)
    ndofs = test_inputs.shape[1]
    fixdofs = np.array([])

    model.eval()

    with torch.no_grad():
        prediction_latent_function = model(
            test_inputs,
            inputs_hessian_data_point_index=test_data_hessian_data_point_index_tensor,
        )

        # mean value of multi-variate normal distribution.
        test_prediction_mean = prediction_latent_function.mean
        pots, grads, hessians = transform_1d_train_targets_into_pots_grads_hessians(
            test_prediction_mean,
            test_data_num,
            ndofs,
            fixdofs,
            test_data_with_hessian_number,
        )

        # the diagonal component of covariance matrix of multi-variate normal distribution gives the uncertainty of prediction.
        test_prediction_variance = torch.diag(
            prediction_latent_function.covariance_matrix
        )
        pots_var, grads_var, hessians_var = (
            transform_1d_train_targets_into_pots_grads_hessians(
                test_prediction_variance,
                test_data_num,
                ndofs,
                fixdofs,
                test_data_with_hessian_number,
            )
        )  # transform 1d data to pots, grads, hessian. It is the same for variance & mean value

        pots = pots.detach().cpu().numpy()
        grads = grads.detach().cpu().numpy()
        hessians = hessians.detach().cpu().numpy()
        pots_var = pots_var.detach().cpu().numpy()
        hessians_var = hessians_var.detach().cpu().numpy()

        return pots, grads, hessians, pots_var, grads_var, hessians_var


def update_model_with_new_data_GPHessian(
    model: GPModelWithHessians,
    new_train_inputs: torch.Tensor,
    new_train_targets: torch.Tensor,
    new_train_data_hessian_data_point_index: torch.Tensor,
    new_noise_covar_factor_pot_grad_array: torch.Tensor,
    new_noise_covar_factor_with_hessian_array: torch.Tensor,
    retrain_bool=True,
):
    """
    Add new training input data and training target data.
    """
    train_inputs = model.train_inputs[0]
    train_targets = model.train_targets
    train_data_hessian_data_point_index = (
        model.training_data_hessian_data_point_index.clone()
    )

    train_data_num = train_inputs.shape[-2]
    new_train_data_num = new_train_inputs.shape[-2]

    ard_num_dim = (
        model.ard_num_dims
    )  # number of dimensions for automatic resonance determination (number of degrees of freedom)
    hessian_triu_size = (
        model.hessian_triu_size
    )  # size of upper triangular part of hessian.

    M_H = len(train_data_hessian_data_point_index)
    new_M_H = len(new_train_data_hessian_data_point_index)

    assert isinstance(
        new_train_inputs, torch.Tensor
    ), "the data type of new_train_inputs need to be torch.Tensor"
    assert isinstance(
        new_train_inputs, torch.Tensor
    ), "the data type of new train targets need to be torch.Tensor"
    assert isinstance(
        new_train_data_hessian_data_point_index, torch.Tensor
    ), "the data type of new_train_data_hessian_data_point_index need to be torch.Tensor"

    # new hessian data point index
    new_train_data_hessian_data_point_index_in_full_data = (
        new_train_data_hessian_data_point_index + train_data_num
    )
    full_train_data_hessian_data_point_index = torch.concat(
        (
            train_data_hessian_data_point_index,
            new_train_data_hessian_data_point_index_in_full_data,
        ),
        dim=0,
    ).to(torch.int32)
    model.training_data_hessian_data_point_index = (
        full_train_data_hessian_data_point_index
    )

    # new training inputs
    full_train_inputs = torch.cat((train_inputs, new_train_inputs), dim=-2)

    # new training targets
    full_targets_pot = torch.cat(
        (
            train_targets[..., :train_data_num],
            new_train_targets[..., :new_train_data_num],
        ),
        dim=-1,
    )
    full_targets_grads = torch.cat(
        (
            train_targets[..., train_data_num : train_data_num * (ard_num_dim + 1)],
            new_train_targets[
                ..., new_train_data_num : new_train_data_num * (ard_num_dim + 1)
            ],
        ),
        dim=-1,
    )
    full_targets_hessian = torch.cat(
        (
            train_targets[
                ...,
                train_data_num * (ard_num_dim + 1) : train_data_num * (ard_num_dim + 1)
                + hessian_triu_size * M_H,
            ],
            new_train_targets[
                ...,
                new_train_data_num
                * (ard_num_dim + 1) : new_train_data_num
                * (ard_num_dim + 1)
                + hessian_triu_size * new_M_H,
            ],
        ),
        dim=-1,
    )

    full_train_targets = torch.cat(
        (full_targets_pot, full_targets_grads, full_targets_hessian), dim=-1
    )

    model.set_train_data(full_train_inputs, full_train_targets, strict=False)

    # update the noise_covar_factor_array (transform noise from Cartesian coordinate into internal coordinate) of model.likelihood
    model.likelihood.update_noise_covar_factor_array(
        new_noise_covar_factor_pot_grad_array, new_noise_covar_factor_with_hessian_array
    )

    # re-train the model to update the hyper parameter
    if retrain_bool:
        train_gpr_model(model)
