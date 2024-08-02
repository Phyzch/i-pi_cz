import torch
from gpytorch.models.exact_prediction_strategies import DefaultPredictionStrategy
from gpytorch import settings 
from gpytorch.utils.memoize import add_to_cache, cached, clear_cache_hook, pop_from_cache
import functools

from linear_operator import to_dense, to_linear_operator
from linear_operator.operators import (
    LinearOperator,
    MatmulLinearOperator,
    ZeroLinearOperator,
)

class RBFHessianPredictionStrategy(DefaultPredictionStrategy):
    '''
    prediction strategy for the kernel which is Radial Basis Function, and the case we have the Hessian data in the target data.
    '''
    def __init__(self, train_inputs, train_prior_dist, train_labels, likelihood, training_data_hessian_data_point_index, hessian_fixdofs):
        '''
        :param: train_inputs: training input data.
        :param: train_prior_dist: Prior distribution of the training targets (MultivariateNormal Distribution)
        :param: train_labels: training targets data (y)
        :param: likelihood: likelihood function, add noise to the GPR model.
        :param: training_data_hessian_data_point_index: the index in the training data that has the hessian information
        :param: hessian_fixdofs: the index of dofs we exclude from the hessian calculation.
        '''
        self.train_inputs = train_inputs  # input X  
        self.train_prior_dist = train_prior_dist  # prior distribution f(X)
        self.train_labels = train_labels # targets (train_y)
        self.likelihood = likelihood   # the likelihood here is the RBFHessianGaussianLikelihood instance.

        M_H = len(training_data_hessian_data_point_index)
        M = train_inputs.shape[-2]

        mvn = self.likelihood(train_prior_dist, M, M_H)  # probability distribution of trained_y 
        self.lik_train_train_covar = mvn.lazy_covariance_matrix  # K(X,X) + sigma^2 I 
        self.train_mean = mvn.loc # mean value of prediction of GPR 

        self.training_data_hessian_data_point_index = training_data_hessian_data_point_index 
        self.hessian_fixdofs = hessian_fixdofs 

        self.train_num = M # number of training points 
        self.training_data_hessian_data_point_number = len(self.training_data_hessian_data_point_index)
        self.d = train_inputs.shape[-1]   # dimension d of inputs 
        self.nactive = int(self.d - len(hessian_fixdofs))  # number of active dofs
        self.hessian_triu_size = int((self.nactive + 1) * self.nactive / 2)  # the size of upper triangle part of the hessian.

    @property
    @cached(name="mean_cache")
    def mean_cache(self):
        train_mean, train_train_covar = self.train_mean, self.lik_train_train_covar  # covariance matrix of y(x) (likelihood) : K(X,X) + sigma^2 I

        train_labels_offset = (self.train_labels - train_mean).unsqueeze(-1)   # y
        mean_cache = train_train_covar.evaluate_kernel().solve(train_labels_offset).squeeze(-1)  # (K(X,X) + sigma^2 I)^-1 * y

        if settings.detach_test_caches.on():
            mean_cache = mean_cache.detach()

        if mean_cache.grad_fn is not None:
            wrapper = functools.partial(clear_cache_hook, self)
            functools.update_wrapper(wrapper, clear_cache_hook)
            mean_cache.grad_fn.register_hook(wrapper)

        return mean_cache
    

    def exact_predictive_covar(
        self, test_test_covar: LinearOperator, test_train_covar: LinearOperator
    ) -> LinearOperator:
        """
        Computes the posterior predictive covariance of a GP

        :param ~linear_operator.operators.LinearOperator test_train_covar:
            Covariance matrix between test and train inputs  K(x*, X)
        :param ~linear_operator.operators.LinearOperator test_test_covar: Covariance matrix between test inputs  K(x*, x*)
        :return: A LinearOperator representing the predictive posterior covariance of the test points
        """
        if settings.fast_pred_var.on():
            self._last_test_train_covar = test_train_covar

        if settings.skip_posterior_variances.on():
            return ZeroLinearOperator(*test_test_covar.size())

        if settings.fast_pred_var.off():
            dist = self.train_prior_dist.__class__(
                torch.zeros_like(self.train_prior_dist.mean), self.train_prior_dist.lazy_covariance_matrix
            )
            if settings.detach_test_caches.on():
                train_train_covar = self.lik_train_train_covar.detach()  # (K(X,X) + sigma^2 I)
            else:
                train_train_covar = self.lik_train_train_covar

            test_train_covar = to_dense(test_train_covar)
            train_test_covar = test_train_covar.transpose(-1, -2)
            covar_correction_rhs = train_train_covar.solve(train_test_covar) # (K(X,X) + sigma^2 I)^-1 * K(X, x*)
            # For efficiency
            if torch.is_tensor(test_test_covar):
                # We can use addmm in the 2d case
                if test_test_covar.dim() == 2:
                    return to_linear_operator(
                        torch.addmm(test_test_covar, test_train_covar, covar_correction_rhs, beta=1, alpha=-1) # k(x*, x*) - K(x*, X) * covar_correction_rhs
                    )
                else:
                    return to_linear_operator(test_test_covar + test_train_covar @ covar_correction_rhs.mul(-1))
            # In other cases - we'll use the standard infrastructure
            else:
                return test_test_covar + MatmulLinearOperator(test_train_covar, covar_correction_rhs.mul(-1))

        precomputed_cache = self.covar_cache
        covar_inv_quad_form_root = self._exact_predictive_covar_inv_quad_form_root(precomputed_cache, test_train_covar)
        if torch.is_tensor(test_test_covar):
            return to_linear_operator(
                torch.add(
                    test_test_covar, covar_inv_quad_form_root @ covar_inv_quad_form_root.transpose(-1, -2), alpha=-1
                )
            )
        else:
            return test_test_covar + MatmulLinearOperator(
                covar_inv_quad_form_root, covar_inv_quad_form_root.transpose(-1, -2).mul(-1)
            )



    def exact_prediction(self, joint_mean, joint_covar, test_data_hessian_data_point_index, test_data_num):
        '''
        Compute the posterior predictive mean and covariance of a GP.
        :param: joint_mean:  mean value of joint gaussian distribution of training data and test data.
        :param: joint_covar: covariance matrix of joint gaussian distribution of training data and test data.
        :param: test_data_hessian_data_point_index: data index in the test data that has hessian information.
        :param: test_data_num: number of test data points 
        '''  
        MH_1 = len(self.training_data_hessian_data_point_index)  # number of training data that contains hessian information.
        MH_2 = len(test_data_hessian_data_point_index)  # number of test data that contains hessian information.

        M1 = self.train_num  # number of training data
        M2 = test_data_num   # number of test data 

        d = self.d # dimensions of the input data 
        hessian_triu_size = self.hessian_triu_size 

        training_target_pots_index = torch.arange(start= 0, end= M1)
        training_target_grads_index = torch.arange(start= (M1 + M2), end= (M1 + M2) + M1 * d)
        training_target_hessian_index = torch.arange(start= (M1 + M2) * (d + 1), end= (M1 + M2) * (d + 1) + MH_1 * hessian_triu_size)
        training_target_index = torch.concat((training_target_pots_index, training_target_grads_index, training_target_hessian_index),  dim= -1)

        test_target_pots_index = torch.arange(start= M1, end= M1 + M2)
        test_target_grads_index = torch.arange(start= (M1 + M2) + M1 * d, end= (M1+ M2) * (d + 1))
        test_target_hessian_index = torch.arange(start= (M1 + M2) * (d + 1) + MH_1 * hessian_triu_size, end= (M1 + M2) * (d + 1) + (MH_1 + MH_2) * hessian_triu_size)
        test_target_index = torch.concat( (test_target_pots_index, test_target_grads_index, test_target_hessian_index), dim= -1 )
        
        test_mean = torch.index_select(joint_mean, dim= -1, index= test_target_index)

        if joint_covar.size(-1) < settings.max_eager_kernel_size.value(): 
            test_covar = joint_covar[..., test_target_index, :].to_dense()
        else:
            test_covar = joint_covar[..., test_target_index, :]

        test_test_covar = test_covar[..., :, test_target_index]
        test_train_covar = test_covar[..., :, training_target_index]

        prediction_mean = self.exact_predictive_mean(test_mean, test_train_covar)
        prediction_var = self.exact_predictive_covar(test_test_covar, test_train_covar)
        
        return (
            prediction_mean,
            prediction_var
        )
    