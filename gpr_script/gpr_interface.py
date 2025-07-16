"""
Code to interface the i-pi module with gpr code.
"""
import gpr.gprtools 
import gpr.gpr_hessian_tools
import numpy as np 
import os 

from ipi.engine.motion.neb_instanton_gpr import MAPNEBGPRMover, SharedData
from ipi.engine.motion import Motion 
from ipi.utils.depend import dstrip
from ipi.utils.scripting import (
    InteractiveSimulation
)
import ipi.utils.nebinstool
import gpr.internal.ZmatrixInternal
import ipi.utils.nebinstool
import ipi.utils.hessfasttools
import internal_util
import gpr_util 
from ipi.engine.beads import Beads
from timeit import default_timer as timer

class ActiveLearning(object):
    def __init__(self,
                 sim: InteractiveSimulation,
                 motion: MAPNEBGPRMover):
        self.sim = sim
        self.motion = motion 
        self.gprForceMapper = GPRForceMapper(sim, motion)
        self.gprHessianMapper = GPRHessianMapper(sim, motion)
        self.total_steps = sim.tsteps


    def run_one_step(self, write_outputs= True):
        """
        run the simulation for steps.
        """
        # update the gpr model for the neb stage.
        if self.motion.options["stage"] == 'neb':
            self.sim.run(steps= 1, write_outputs= write_outputs)
            self.update_gpr_model()
        elif self.motion.options["stage"] == "instanton":
            if not self.motion.instanton_temperature_avail:
                # need to compute the temperature for the instanton path.
                if self.motion.options["test_gpr_model_along_instanton_path"]:
                    # optimize GPR model to make sure it will give accurate force for dynamics.
                    # Separate point as test set and training set, add training set until the generalization error is small.
                    self.optimize_GPR_model_for_dynamics_evolution()
            else:
                if self.motion.rp_map.final_hessian_bool:
                    # need to compute hessian for ring polymer beads. Either use GPR or do it ab-initio
                    self.bind_gpr_hessian_mapper()
                    # create gpr hessian model either reading data from input file or using training data from gpr model.
                    self.construct_gpr_hessian_model()
                    # bind the gpr hessian model.
                    self.motion.bind_gpr_hessian_model(self.gprHessianMapper.gpr_hessian_model)
                    # add new hessian data into GPR model.
                    # the location of new hessian data is given by self.new_hessian_data_point_index.
                    # candidate_hessian_point_x spaced with equal distance along the path.
                    self.add_new_hessian_and_grad_data()
                    pass 
            
            self.sim.run(steps= 1, write_outputs= write_outputs) 
        elif self.motion.options["stage"] == "converged":
            # converge stage.
            self.sim.run(steps= 1, write_outputs= write_outputs)
        else:
            raise ValueError("Invalid stage options. Must be: neb, instanton or converged.")

    def run(self, write_outputs= True):
        """
        run for total number of steps.
        """
        for step in range(self.sim.step, self.total_steps):
            self.run_one_step(write_outputs= write_outputs)

    def initialize_gpr_model(self):
        """
        initialize the coordinate transformer and gpr model.
        1. Initialize coordinate transformer to transform between internal coordinate and cartesian coordinate.
        2. initialize GPR_Wrapper, which combines coordinate transformer and GPR model.
        """
        self.gprForceMapper.initialize_gpr_model()

    def update_gpr_model(self):
        """
        update the Gaussian Process Regression model at the end of each simulation step.
        """
        self.gprForceMapper.update_gpr_model()

    def optimize_GPR_model_for_dynamics_evolution(self):
        """
        further optimize the GPR model to make sure GPR predicted force is accurate along the instanton path.
        This is to ensure the temperature we get from the dynamical evolution is accurate.
        """
        self.gprForceMapper.optimize_GPR_model_for_dynamics_evolution()

    def bind_gpr_hessian_mapper(self):
        self.gprHessianMapper.bind(self)

    def construct_gpr_hessian_model(self):
        """
        construct the gpr_hessian_model. 
        """
        self.gprHessianMapper.construct_gpr_hessian_model()

    def add_new_hessian_and_grad_data(self):
        """
        add new hessian and gradient data to the gpr hessian model.
        """
        self.gprHessianMapper.add_new_hessian_and_grad_data()

class GPRForceMapper(object):
    """
    Gaussian Process Regression (GPR) model to predict potential and force.
    """
    def __init__(self, 
                 sim: InteractiveSimulation,
                 motion: MAPNEBGPRMover):
        """
        bind the gpr model and coordinate_transformer.
        All gpr model operation will be performed in this class.
        """
        self.sim = sim 
        self.motion = motion 

        # bead and forces for cross validation.
        self.gpr_beads = Beads(motion.beads.natoms, 1)
        self.gpr_forces = motion.forces.copy(self.gpr_beads, motion.cell)
        self.energy_shift = self.motion.optarrays["energy_shift"]

    def initialize_internal_coord(self):
        """
        initialize the coordinate transformer between the internal coordinate and the Cartesian coordinate.
        """
        # initialize the non-redundant coordinate transformer.
        # call select ref points to generate ref points for coordinate transformer.
        ref_x_list = internal_util.select_reference_points(self.motion)

        coordinate_transformer = internal_util.create_coordinate_transformer(self.motion,
                                                               ref_x_list)
        return coordinate_transformer
    
    def _initialize_gpr_model(self, train_x, train_V, train_grad, coordinate_transformer):
        """
        Initialize the GPR model.
        """
        # load parameters for the gpr model initialization. 
        # TODO: need a cleaner interface for gpr input parameter.
        neb_final_gpr_folder = "neb_final_gpr_training"
        fix_dofs = self.motion.optarrays["fix_dofs"]
        natoms = self.motion.beads.natoms 
        gpr_SE_kernel_number = self.motion.options["gpr_SE_kernel_number"]
        kernel_output_scale = self.motion.optarrays["gpr_kernel_outputscale"]
        kernel_lengthscale_ratio = self.motion.optarrays["gpr_kernel_lengthscale_ratio"]
        gpr_noise_std = self.motion.optarrays["gpr_noise_std"]
        gpr_fix_internal_dofs_bool = self.motion.options["gpr_fix_internal_dofs_bool"]
        gpr_fix_internal_dofs_cutoff = self.motion.options["gpr_fix_internal_dofs_cutoff"]
        gpr_fixed_internal_dofs = gpr_util.read_fixed_internal_dofs(prefix= "neb_final_gpr_training")
        gpr_covar_inverse_nugget = self.motion.optarrays["gpr_covar_inverse_nugget"]

        gpr_model = gpr.gprtools.GPModelWithDerivativesWrapper(
            train_x,
            train_V,
            train_grad,
            natoms,
            coordinate_transformer,
            fix_dofs,
            gpr_SE_kernel_number,
            kernel_output_scale,
            kernel_lengthscale_ratio,
            gpr_noise_std,
            train_bool= False,
            gpr_fix_internal_dofs_bool= gpr_fix_internal_dofs_bool,
            gpr_fix_internal_dofs_cutoff= gpr_fix_internal_dofs_cutoff,
            gpr_fixed_internal_dofs= gpr_fixed_internal_dofs,
            singular_value_cutoff= gpr_covar_inverse_nugget
        )

        # option to load trained hyper-parameters for gpr model.
        read_gpr_training_data_bool = self.motion.options["read_initial_gpr_training_data"]
        if (read_gpr_training_data_bool or self.motion.options["stage"] == "instanton"):
            # see if there is option to read hyper-parameter without training the model
            neb_final_gpr_folder = "neb_final_gpr_training"
            model_hyperparameter_exists = gpr_util.load_training_hyperparameter_in_gpr_model(
                gpr_model, neb_final_gpr_folder
            )

            if not model_hyperparameter_exists:
                gpr_model.train_gpr()

        else:
            gpr_model.train_gpr()

        # store internal dofs in the gpr model        
        gpr_util.store_fixed_internal_dofs_gpr_model(
            gpr_model,
            prefix= neb_final_gpr_folder
        )

        # store training hyperparameters in the gpr model.
        gpr_util.store_training_hyperparameter_in_gpr_model(
            gpr_model,
            neb_final_gpr_folder
        )

        return gpr_model

    def check_training_result(self):
        """
        check whether the initial training of GPR model is successful.
        """
        # check the training error. potential underfitting.
        train_x = self.gpr_model.output_training_cartesian_inputs()
        predicted_V_shift, predicted_grad, _, var_grad_x_trace = (
            self.gpr_model.predict_latent_function(train_x)
        )

        predicted_forces = - predicted_grad

        initial_data_number = train_x.shape[0]
        # check the training error.  
        ab_initio_V_shift = np.copy(self.gpr_model.train_cartesian_targets[:,0])
        ab_initio_forces = np.copy(-self.gpr_model.train_cartesian_targets[:,1:])
        
        print("\n")
        print(
            "@initial gpr training info: check the overfitting and underfitting of kernel length scale"
        )

        # check the difference between ab-initio potential V and the predicted potential V:
        V_error = np.abs(ab_initio_V_shift - predicted_V_shift) / np.abs(
            ab_initio_V_shift
        )

        # check the difference between ab-initio force f and the predicted force f.
        df = np.linalg.norm(ab_initio_forces - predicted_forces, axis=1)
        ab_initio_force_amplitude = np.linalg.norm(ab_initio_forces, axis=1)
        df_error = df / ab_initio_force_amplitude

        print("\n")
        print("@initial Gaussian Process Regression fitting:")
        print("error of potential prediction: " + str(V_error))
        print("error of force prediction: " + str(df_error))
        print("\n")

        # check overfitting on the unseen test data to test over-fitting.
        print("@initial gpr training info: Test Overfitting of GPR model.")
        print("The error in test set can be large if we start the model with a small number of data.")
        print("In this case, the test data is out of trust region of the model.")

        # cross validation.
        try:
            energy_shift = self.motion.optarrays["energy_shift"]
        except:
            energy_shift = 0

        if initial_data_number >= 2:
            test_x = train_x[0] * 1 / 4 + train_x[1] * 3 / 4
            print("x[0] * 1/4 + x[1] * 3/4")
            (
                predicted_test_V_shift,
                predicted_test_force,
                ab_initio_test_pot,
                ab_initio_test_force,
            ) = gpr_util.check_gpr_fitting_error(
                self.gpr_beads,
                self.gpr_forces,
                self.gpr_model,
                energy_shift,
                test_x
            )

        if initial_data_number >= 4:
            test_x = train_x[3] * 1 / 4 + train_x[2] * 3 / 4
            print("x[3] * 1/4 + x[2] * 3/4")
            (
                predicted_test_V_shift,
                predicted_test_force,
                ab_initio_test_pot,
                ab_initio_test_force,
            ) = gpr_util.check_gpr_fitting_error(
                self.gpr_beads,
                self.gpr_forces,
                self.gpr_model,
                energy_shift,
                test_x
            )
    
    def initialize_gpr_model(self):
        """
        initialize the coordinate transformer and gpr model.
        1. Initialize coordinate transformer to transform between internal coordinate and cartesian coordinate.
        2. initialize GPR_Wrapper, which combines coordinate transformer and GPR model.
        """
        coordinate_transformer = self.initialize_internal_coord()

        # for the training data, we have the option to read it from .txt file or generate it using the current geometry.
        # this provides the flexibility for choosing the training data for the initial model.
        train_x, train_V, train_grad = self.motion._get_training_data()

        # time this function.
        start_time = timer()
        # initialize the gpr model use training data and internal coordinate transformer.
        gpr_model = self._initialize_gpr_model(train_x, train_V, train_grad, coordinate_transformer)
        
        end_time = timer() 
        time_elapsed = (end_time - start_time) / 60
        print(f"the time used for construct \
                gpr model to predict force along instanton path is: {time_elapsed} min")

        # bind the coordinate transformer and gpr model to the motion object.
        # this will enable the motion object to use gpr model to predict potential and force.
        self.coordinate_transformer = coordinate_transformer
        self.gpr_model = gpr_model 
        self.motion.bind_gpr_model(gpr_model, coordinate_transformer)

        # output the representation of internal coordinates.
        internal_util.output_internal_coord(coordinate_transformer)
        # check the training error and cross-validation error of the gpr model.
        self.check_training_result()

    # update the gpr model
    def update_gpr_model(self):
        """
        update the Gaussian Process Regression model at the end of each simulation step.
        """
        step = self.sim.step 
        early_stop_bool = self.motion.early_stop_bool
        outrange_bead_index_list = self.motion.outrange_bead_index_list

        gpr_trust_region = self.motion.optarrays["gpr_trust_region"]
        print("The trust region for the gpr model now: " + str(gpr_trust_region))

        new_training_x = self.select_new_GPR_data_point(early_stop_bool, outrange_bead_index_list)
        if(len(new_training_x) == 0):
            # surrogate potential energy surface is accurate. exit the neb stage.
            self.motion.neb_stage_exit_step(step)
            self.exit_neb_stage()
        else:
            # compute ab initio force & force error before update gpr model.
            new_ab_initio_pots, new_ab_initio_forces = self.before_gpr_update_force_error(new_training_x)
            # update the gpr model (subroutine).
            self._update_gpr_model(new_training_x, new_ab_initio_pots, new_ab_initio_forces)
            # check force error after update the gpr model.
            self.after_gpr_update_force_error(new_training_x, new_ab_initio_forces)
            # output info about gpr force error.
            self.output_force_error_info(step)

    def select_new_GPR_data_point(self, early_stop_bool, outrange_bead_index_list):
        """
        select the new data point to add to the GPR model.
        """
        motion = self.motion
        beads_q = motion.beads.q 
        nbeads = motion.beads.nbeads 
        if early_stop_bool:
            # in this case, select the bead that moves out of trust region with the largest force uncertainty.
            outrange_bead_x =  np.copy(beads_q[outrange_bead_index_list])
            _, _, _, var_grad_x_trace_list = self.gpr_model.predict_latent_function(outrange_bead_x)
            bead_index_for_update = outrange_bead_index_list[np.argmax(var_grad_x_trace_list)]
            new_training_x = np.array([beads_q[bead_index_for_update]])
        else:
            # select the data point with the force uncertainty estimate larger than the cutoff value 
            # to add to the training data set.
            force_uncertainty_cutoff = motion.optarrays["gpr_force_uncertainty_criterion"] + 1e-5 
            # compute gpr predicted force uncertainty 
            _, _, _, var_grad_x_trace_list = self.gpr_model.predict_latent_function(beads_q)

            force_uncertainty = np.sqrt(var_grad_x_trace_list)

            large_uncertainty_bool = (force_uncertainty > force_uncertainty_cutoff)
            large_uncertainty_bead_index = np.arange(nbeads)[large_uncertainty_bool]

            new_training_x = dstrip(beads_q[large_uncertainty_bead_index]).copy()
        
        return new_training_x

    def before_gpr_update_force_error(self, new_training_x):
        """
        compute the force error before update the gpr model.
        Also compute the ab initio forces for the new data.
        """
        # ML learned forces.
        _, beads_grads, _, _ = self.gpr_model.predict_latent_function(new_training_x)
        gpr_beads_forces = - beads_grads
        
        # ab initio forces.
        bead_number = new_training_x.shape[0]
        natoms = self.motion.beads.natoms
        new_ab_initio_forces = np.zeros([bead_number, 3 * natoms])
        new_ab_initio_pots = np.zeros([bead_number])
        for i in range(bead_number):
            self.gpr_beads.q[0] = new_training_x[i]
            new_ab_initio_forces[i] = dstrip(self.gpr_forces.f).copy()[0]
            new_ab_initio_pots[i] = dstrip(self.gpr_forces.pots).copy()[0]
        
        # count the number of ab initio calculations we have done.    
        SharedData.ab_initio_bead_calculation_number = (
            SharedData.ab_initio_bead_calculation_number + bead_number
        )

        # compute the |f|, |f_GPR|, |f - f_GPR|, |f-f_GPR|/|f|
        self.gpr_force_amplitude_list = np.linalg.norm(gpr_beads_forces, axis= 1)
        self.ab_initio_force_amplitude_list = np.linalg.norm(new_ab_initio_forces, axis= 1)
        force_diff_list = gpr_beads_forces - new_ab_initio_forces
        self.force_diff_amplitude_list = np.linalg.norm(force_diff_list, axis= 1)
        self.force_diff_ratio_list = (
            self.force_diff_amplitude_list / self.ab_initio_force_amplitude_list
        )

        return new_ab_initio_pots, new_ab_initio_forces
    
    def _update_gpr_model(self, new_training_x, new_ab_initio_pots, new_ab_initio_forces):
        """
        update the gpr model with new pot and force data.
        """
        distance_cutoff = self.motion.options["distance_cutoff_for_training_data"]
        train_grad_model_bool = self.motion.options["train_grad_model_bool"]

        new_shifted_pots = new_ab_initio_pots - self.energy_shift
        new_ab_initio_grads = - new_ab_initio_forces

        self.gpr_model.update_model_with_new_data(
            new_training_x,
            new_shifted_pots,
            new_ab_initio_grads,
            distance_cutoff,
            train_grad_model_bool
        )
    
    def after_gpr_update_force_error(self, new_training_x, new_ab_initio_forces):
        """
        compute the force error after update the gpr model.
        """
         # ML learned forces.
        _, beads_grads, _, _ = self.gpr_model.predict_latent_function(new_training_x)
        gpr_beads_forces = - beads_grads

        force_diff = gpr_beads_forces - new_ab_initio_forces
        self.force_diff_amplitude_after_update_list = np.linalg.norm(force_diff, axis= 1)
        self.force_diff_ratio_after_update_list = (
            self.force_diff_amplitude_after_update_list / self.ab_initio_force_amplitude_list
        )

        # check the uncertainty of force for the updated potential.
        # increase the gpr_force_uncertainty criterion if it is not met after we have updated the pot.
        beads_q = self.motion.beads.q 
        _, _, _, var_grad_x_uncertainty = self.gpr_model.predict_latent_function(beads_q)
        max_std_grad_x_uncertainty = np.max(np.sqrt(var_grad_x_uncertainty))
        if max_std_grad_x_uncertainty > self.motion.optarrays["gpr_force_uncertainty_criterion"]:
            print("@Warning: The uncertainty of gpr prediction is still higher than cutoff criterion after update the model.")
            print(f"max std force uncertainty: {max_std_grad_x_uncertainty}")
            print(f"The force uncertainty criterion will be increased to {max_std_grad_x_uncertainty}")
            self.motion.optarrays["gpr_force_uncertainty_criterion"] = max_std_grad_x_uncertainty

    def output_force_error_info(self, step):
        """
        output info about |f - f_GPR|.
        """
        new_data_num = len(self.ab_initio_force_amplitude_list)
        if new_data_num != 0:
            print(
                "@Outerloop Exit info: ab initio |f|: "
                + str(self.ab_initio_force_amplitude_list)
            )
            print(
                "@Outloop Exit info: GPR predicted |f_GPR|: "
                + str(self.gpr_force_amplitude_list)
            )
            print("@Outerloop Exit info: |f_GPR -f|/|f|:" + str(self.force_diff_ratio_list))
            print(
                "@Outerloop Exit info: max(|f_GPR - f|/|f|): "
                + str(np.max(self.force_diff_ratio_list))
            )
            print(
                "@Outerloop Exit info: |f_GPR -f| :" + str(self.force_diff_amplitude_list)
            )

            print("After update:")
            print("@Outerloop Exit info: |f_GPR -f|/|f|:" + str(self.force_diff_ratio_after_update_list))
            print(
                "@Outerloop Exit info: |f_GPR -f| :" + str(self.force_diff_amplitude_after_update_list)
            )

            print("Finish Outerloop: " + str(step))
            print("\n")
            print("\n")
        
        self.force_diff_ratio_list = []
        self.ab_initio_force_amplitude_list = []
        self.gpr_force_amplitude_list = []

        self.force_diff_ratio_after_update_list = []
        self.force_diff_amplitude_after_update_list = [] 
    
    def update_GPR_model_with_one_new_data_point(self, unused_train_x, train_beads, train_forces):
        """
        update the GPR model with 1 training data point with the highest uncertainty variance.
        """
        distance_cutoff = self.motion.options["distance_cutoff_for_training_data"]
        train_grad_model_bool = self.motion.options["train_grad_model_bool"]
        # select the point with the largest force variance
        _, _, _, var_grad_x_trace = self.gpr_model.predict_latent_function(
            unused_train_x
        )
        new_data_index = np.argmax(var_grad_x_trace)
        new_train_x = unused_train_x[new_data_index]
        new_train_x = np.array([new_train_x])
        unused_train_x = np.delete(
            unused_train_x, new_data_index, axis=0
        )  # delete new data point from the unused training set

        # compute ab initio potential and forces for the new training data set and add to the GPR model.
        train_beads.q = new_train_x
        new_train_V = dstrip(train_forces.pots).copy()
        new_train_f = dstrip(train_forces.f).copy()
        new_train_shifted_V = new_train_V - self.energy_shift
        new_train_grad_x = -new_train_f
        self.gpr_model.update_model_with_new_data(
            new_train_x,
            new_train_shifted_V,
            new_train_grad_x,
            distance_cutoff,
            train_grad_model_bool
        )

        return unused_train_x

    def optimize_GPR_model_for_dynamics_evolution(self):
        """
        further optimize the GPR model to make sure GPR predicted force is accurate along the instanton path.
        This is to ensure the temperature we get from the dynamical evolution is accurate.
        Add training data to GPR model and optimize hyper-parameters to make sure the GPR model 
        generate accurate force along LI-NEB path to have correct temperature
        This is achieved by cross-validation technique:
        (1) Generate 50 data points along the LI-NEB path using cubic interpolation
        (2) Randomly choose 5 data points from them as test data
        (3) Use GPR model to predict the force, compute the force error
        (4) Add more training data into the training set until the force error is small on testing data:
            either relative force error is small or absolute force error is small.
            The selection of the training data is based on the force uncertainty.
        """
        gpr_relative_force_error_criterion = self.motion.optarrays["gpr_relative_force_error_criterion"]
        gpr_absolute_force_error_criterion = self.motion.optarrays["gpr_absolute_force_error_criterion"]
        total_data_set_number = 50
        testing_data_number = 5
         # ---------------  generate training and test data for GPR model.  ---------
        # we interpolate the converged LI-NEB path with N + 2 data point, 
        # where 2 end data point is already in the training set of GPR model (as they are end points for LI-NEB path)
        LINEB_path_x, _ = ipi.utils.nebinstool.path_cubic_interpolation(
            self.motion.beads.q, total_data_set_number + 2
        )
        LINEB_path_x = LINEB_path_x[1:-1]

        index_list = np.arange(total_data_set_number)
        np.random.shuffle(
            index_list
        )  # shuffle the index to get test data and training data.

        test_x = LINEB_path_x[index_list[:testing_data_number]]
        unused_train_x = LINEB_path_x[index_list[testing_data_number:]]
        # ------------------------

        # create beads and forces object for testing and training data set
        natoms = self.motion.beads.natoms
        test_beads = Beads(
            natoms, 1
        )  
        test_forces = self.motion.forces.copy(test_beads, self.motion.cell)

        train_beads = Beads(natoms, 1)
        train_forces = self.motion.forces.copy(train_beads, self.motion.cell)

         # compute the ab-initio forces for the test beads:
        ab_initio_test_data_f = np.zeros([testing_data_number, 3 * natoms])
        for i in range(testing_data_number):
            test_beads.q[0] = test_x[i]
            ab_initio_test_data_f[i] = dstrip(test_forces.f).copy()[0]
        # magnitude of the ab initio force
        ab_initio_test_data_f_magnitude = np.linalg.norm(
            ab_initio_test_data_f, axis=1
        )  

        # --- make predictions of forces for test data set using GPR model. ---
        _, gpr_test_grad, _, _ = self.gpr_model.predict_latent_function(test_x)
        gpr_test_f = - gpr_test_grad
        # absolute error on the test force
        test_f_diff_magnitude = np.linalg.norm(
            gpr_test_f - ab_initio_test_data_f, axis=1
        )  
        # relative error on the test force.
        test_f_diff_magnitude_ratio = (
            test_f_diff_magnitude / ab_initio_test_data_f_magnitude
        )  
        # --------------------------------------------------------------------

        pass_test_bool= False 
        while not pass_test_bool:
            # ---- see if the prediction on the test data is satisfactory:
            pass_test_bool = True
            for i in range(testing_data_number):
                if (
                    test_f_diff_magnitude_ratio[i]
                    > gpr_relative_force_error_criterion
                    and test_f_diff_magnitude[i]
                    > gpr_absolute_force_error_criterion
                ):
                    pass_test_bool = False
                    break

            if pass_test_bool:
                break
            # -------------------

            # select the point with the largest force variance
            unused_train_x = self.update_GPR_model_with_one_new_data_point(unused_train_x, train_beads, train_forces)

            if len(unused_train_x) == 0:
                # all training data set has been used but no satisfactory result achieved on test data.
                print("\n")
                print(
                    "@WARNING: After adding all training data into the gpr model, " \
                    "the prediction of force on unseen test data is still unsatisfactory"
                )
                print("\n")
                break

            # --- make predictions of forces for test data set using GPR model. ---
            _, gpr_test_grad, _, _ = self.gpr_model.predict_latent_function(test_x)
            gpr_test_f = - gpr_test_grad

            test_f_diff_magnitude = np.linalg.norm(
                gpr_test_f - ab_initio_test_data_f, axis=1
            )  # absolute error on the test force
            test_f_diff_magnitude_ratio = (
                test_f_diff_magnitude / ab_initio_test_data_f_magnitude
            )  # relative error on the test force.
            # --------------------------------------------------------------------
        
        print("\n")
        print("@Update GPR model with more training data for dynamics along LINEB path")
        print(
            "absolute error for the force of the test data: "
            + str(test_f_diff_magnitude)
        )
        print(
            "relative error for the force of the test data: "
            + str(test_f_diff_magnitude_ratio)
        )
        print("The force error along LI-NEB path is small. Pass the test.")
        print("\n")

        # ---- update the data in the gpr training data folder.  ----- 
        train_x = self.gpr_model.train_cartesian_inputs
        train_V = self.gpr_model.train_cartesian_targets[:, 0]
        train_V_to_store = train_V + self.energy_shift
        train_grad = self.gpr_model.train_cartesian_targets[:, 1:]
        train_f_to_store = -train_grad

        neb_gpr_folder_path = "neb_final_gpr_training"
        gpr_util.store_training_data(
            train_x, train_V_to_store, train_f_to_store, prefix= neb_gpr_folder_path
        )

        gpr_util.store_training_hyperparameter_in_gpr_model(self.gpr_model, neb_gpr_folder_path)
        
        # store fixed dofs.
        gpr_util.store_fixed_internal_dofs_gpr_model(
            self.gpr_model,
            prefix = neb_gpr_folder_path
        )
        # -------------------------

    def exit_neb_stage(self):
        """
        store all training data and hyper-parameters at the end of neb stage.
        """
        beads_q = self.motion.beads.q 
        # store potential and forces for the final LI-NEB beads.
        beads_potential_shift, beads_potential_grad_x, _, _ = (
            self.gpr_model.predict_latent_function(np.copy(beads_q))
        )
        self.LINEB_pots = (
            beads_potential_shift + self.energy_shift
        )
        self.LINEB_forces = -beads_potential_grad_x

        gpr_util.store_training_data(
            beads_q, self.LINEB_pots, self.LINEB_forces, prefix="LINEB_beads"
        )

        # store all training data
        train_x = self.gpr_model.train_cartesian_inputs
        train_V = self.gpr_model.train_cartesian_targets[:, 0]
        train_V_to_store = train_V + self.energy_shift
        train_grad = self.gpr_model.train_cartesian_targets[:, 1:]
        train_f_to_store = -train_grad

        gpr_util.store_training_data(
            train_x, train_V_to_store, train_f_to_store, prefix="neb_final_gpr_training"
        )
        neb_gpr_folder_path = "neb_final_gpr_training"
        gpr_util.store_training_hyperparameter_in_gpr_model(self.gpr_model,
                                                            neb_gpr_folder_path)
        
        # store fixed dofs.
        gpr_util.store_fixed_internal_dofs_gpr_model(
            self.gpr_model,
            prefix = neb_gpr_folder_path
        )

class GPRHessianMapper(object):
    """
    Gaussian Process Regression to predict Hessians for rate calculation.
    """
    def __init__(self, 
                 sim: InteractiveSimulation,
                 motion: MAPNEBGPRMover):
        """
        bind the gpr model and coordinate_transformer.
        All gpr model operation will be performed in this class.
        """
        self.sim = sim 
        self.motion = motion 
        self.rp_map = self.motion.rp_map

    def bind(self,
             active_learning: ActiveLearning):
        """
        bind the gpr model.
        """
        self.gpr_model = active_learning.gprForceMapper.gpr_model 
        self.coordinate_transformer = active_learning.gprForceMapper.coordinate_transformer

        self.neb_beads = self.motion.beads
        nebmover = self.motion 
        # bind variables 
        self.energy_shift = nebmover.optarrays["energy_shift"]
        self.fix_dofs = np.array(nebmover.optarrays["fix_dofs"])
        # bind the gpr kernel condition
        self.gpr_SE_kernel_number = nebmover.options["gpr_SE_kernel_number"]
        self.gpr_kernel_outputscale = nebmover.optarrays["gpr_kernel_outputscale"]
        self.gpr_kernel_lengthscale_ratio = nebmover.optarrays[
            "gpr_kernel_lengthscale_ratio"
        ]
        self.gpr_noise_std = nebmover.optarrays["gpr_noise_std"]

        self.gpr_fix_internal_dofs_bool = nebmover.options["gpr_fix_internal_dofs_bool"]
        self.gpr_fix_internal_dofs_cutoff = nebmover.options["gpr_fix_internal_dofs_cutoff"]
        self.gpr_rigid_internal_dofs_cutoff = nebmover.options["gpr_rigid_internal_dofs_cutoff"]
        # bind the file that we use to read hessian data
        self.read_gpr_hessian_folder = nebmover.options["read_gpr_hessian_folder"]

        # options about which ab-initio data point we will add to existing training data
        self.add_new_hessian_data_bool = nebmover.options["add_new_hessian_data_bool"]
        self.candidate_hessian_data_number = nebmover.options[
            "candidate_hessian_data_number"
        ]
        self.new_hessian_data_index = nebmover.optarrays["new_hessian_data_index"]
        
        # options about which ab-initio potential and gradient data (along the path)
        # we will add to the existing training data
        self.add_new_grad_data_bool= nebmover.options["add_new_grad_data_bool"]
        self.candidate_grad_data_number = nebmover.options["candidate_grad_data_number"]
        self.new_grad_data_index = nebmover.optarrays["new_grad_data_index"]

        self.train_grad_model_bool = nebmover.options["train_grad_model_bool"]
        self.train_hessian_model_bool = nebmover.options["train_hessian_model_bool"]

        # options to use compute selective hessians in the internal coordinate.
        # we define the rigid mode in the internal coordinate and only compute hessians for 1 bead along rigid mode.
        self.selective_hessian_bool = nebmover.options["selective_hessian_bool"]
        self.new_hessian_data_index_rigid_mode = nebmover.optarrays["new_hessian_data_index_rigid_mode"]

        # options to do cross validation of gpr hessian model.
        self.cross_validation_bool = nebmover.options["cross_validation_bool"]
        # regularization for the linear regression and GPR.
        self.ridge_regularization_alpha = nebmover.optarrays["ridge_regularization_alpha"]
        self.gpr_covar_inverse_nugget = nebmover.optarrays["gpr_covar_inverse_nugget"]

# ------- construct gpr hessian model --------------------
    def construct_selective_hessian_calculator(self, candidate_hessian_point_x):
        """
        initialize SelectiveHessianCalculation class.
        Use this function to only compute a few number of beads along rigid mode 
        and perform linear regression.
        - load existing hessian computed along the rigid mode.
        - compute new hessian along rigid mode if required.
        - construct linear regression model. 
        """
        self.selective_hessian_calculator = ipi.utils.hessfasttools.SelectiveHessianCalculation(
            candidate_hessian_point_x,
            self.coordinate_transformer,
            self.gpr_rigid_internal_dofs_cutoff,
            self.cross_validation_bool
            )
        
        if len(self.new_hessian_data_index_rigid_mode) > 0:
            # for computing new data point for hessian along rigid modes.
            new_train_x_rigid_mode =  candidate_hessian_point_x[self.new_hessian_data_index_rigid_mode]
            new_rigid_mode_bead_number = len(new_train_x_rigid_mode)
            new_rigid_mode_rp_bead = Beads(self.neb_beads.natoms, new_rigid_mode_bead_number)
            new_rigid_mode_rp_force = self.motion.forces.copy(new_rigid_mode_rp_bead, self.motion.cell)

            self.selective_hessian_calculator.rigid_modes_hessian_preprocess(
                prefix= self.read_gpr_hessian_folder,
                new_train_x= new_train_x_rigid_mode,
                new_rp_bead = new_rigid_mode_rp_bead,
                new_rp_force = new_rigid_mode_rp_force,
                new_rigid_mode_bead_index= self.new_hessian_data_index_rigid_mode,
                ridge_regularization_alpha= self.ridge_regularization_alpha["hessian"]
            )
        else:
            self.selective_hessian_calculator.rigid_modes_hessian_preprocess(
                prefix= self.read_gpr_hessian_folder,
                new_rigid_mode_bead_index= self.new_hessian_data_index_rigid_mode,
                ridge_regularization_alpha= self.ridge_regularization_alpha["hessian"]
            )

    def train_gpr_hessian_model(self):
        """
        train the gpr hessian model:
        (1) train the model
        (2) time the model training.
        (3) check training error of the model.
        """
        if self.train_hessian_model_bool:
            print("We are going to train the gpr model with hessian data.\
                This can be expensive. To add data without training the model, set train_hessian_model_bool= False ")
            start_t = timer()

            self.gpr_hessian_model.train_model()

            end_t = timer()
            time_elapsed = (end_t - start_t) / 60
            print(f"the elapsed time for re-training the model is {time_elapsed} min.")

            pass
    
    def store_gpr_hessian_model_parameters(self, folder):
        """
        (1) store the hyper-parameter for gpr model
        (2) store the rigid dofs for gpr hessian model.
        (3) store the fix dofs for gpr hessian model.
        """ 
        gpr_util.store_training_hyperparameter_in_gpr_hessian_model(
            self.gpr_hessian_model, folder
        )

        # store fixed internal dofs.
        gpr_util.store_fixed_internal_dofs_gpr_hessian_model(
            self.gpr_hessian_model,
            folder
        )

        # store rigid internal dofs in the gpr model
        gpr_util.store_rigid_internal_dofs_gpr_hessian_model(
            self.gpr_hessian_model,
            folder
        )

    def load_gpr_hessian_training_data(self, 
                                       candidate_hessian_point_x):
        """
        load the training data for the gaussian process regression model with hessian data.
        """
        print(
                "read_gpr_hessian_folder provided. \
                Will read potential & gradients & hessians from folder and create gpr_hessian model."
        )
        # create gpr_hessian model using data read from read_gpr_hessian_folder
        (
            cartesian_coordinate_x,
            training_V,
            training_forces,
            hessian_index_list,
            hessian_data_list,
        ) = gpr_util.read_training_data_with_hessian(
            self.read_gpr_hessian_folder
        )
        # load fixed internal dofs and rigid internal dofs
        gpr_fixed_internal_dofs = gpr_util.read_fixed_internal_dofs(self.read_gpr_hessian_folder)
        gpr_rigid_internal_dofs = gpr_util.read_rigid_internal_dofs(self.read_gpr_hessian_folder)

        if self.selective_hessian_bool:
            # initialize the selective hessian calculator to compute hessian along rigid modes.
            self.rp_map.construct_selective_hessian_calculator(candidate_hessian_point_x)
            # update the hessian along the rigid mode.
            hessian_data_list = self.rp_map.selective_hessian_calculator.update_hessian_rigid_modes(
                cartesian_coordinate_x[hessian_index_list],
                training_forces[hessian_index_list],
                hessian_data_list
            )

        return (cartesian_coordinate_x, 
                training_V, training_forces, 
                hessian_index_list, hessian_data_list, 
                gpr_fixed_internal_dofs, gpr_rigid_internal_dofs)

    def _initialize_gpr_hessian_model(self, gpr_data):
        """
        create gpr hessian model using the data.
        """
        (train_x, 
        train_V, train_forces, 
        hessian_index_list, hessian_data_list, 
        gpr_fixed_internal_dofs, gpr_rigid_internal_dofs) = gpr_data 

        train_V_shifted = train_V - self.energy_shift
        train_grads = -train_forces

        # choose the first data point with hessian information as the reference point for mean function.
        ref_x = train_x[hessian_index_list[0]]
        ref_V_shifted = np.array([train_V_shifted[hessian_index_list[0]]])
        ref_grads = train_grads[hessian_index_list[0]]
        ref_hessians = hessian_data_list[0]
        
        # For testing the error induced by forward and backward transformation of gradient and hessian.
        gpr_util.analyze_transformation_between_cartesian_coord_and_internal_coord(
            np.array([ref_x]), np.array([ref_grads]), np.array([ref_hessians]), self.coordinate_transformer
        )

        self.gpr_hessian_model = (
            gpr.gpr_hessian_tools.GPModelWithHessiansWrapper(
                train_x,
                train_V_shifted,
                train_grads,
                hessian_data_list,
                hessian_index_list,
                self.motion.beads.natoms,
                self.coordinate_transformer,
                self.fix_dofs,
                self.gpr_SE_kernel_number,
                self.gpr_kernel_outputscale,
                self.gpr_kernel_lengthscale_ratio,
                self.gpr_noise_std,
                constant_mean_func_bool= False,
                ref_mean_x=ref_x,
                ref_mean_V=ref_V_shifted,
                ref_mean_grad_x=ref_grads,
                ref_mean_hessian_x=ref_hessians,
                train_bool= False,
                gpr_fix_internal_dofs_bool= self.gpr_fix_internal_dofs_bool,
                gpr_fix_internal_dofs_cutoff= self.gpr_fix_internal_dofs_cutoff,
                gpr_rigid_internal_dofs_cutoff = self.gpr_rigid_internal_dofs_cutoff,
                gpr_fixed_internal_dofs= gpr_fixed_internal_dofs,
                gpr_rigid_internal_dofs= gpr_rigid_internal_dofs,
                ridge_regularization_alpha= self.ridge_regularization_alpha,
                singular_value_cutoff= self.gpr_covar_inverse_nugget
            )
        )

        model_hyperparameter_exists = \
            gpr_util.load_training_hyperparameter_for_gpr_hessian_model(
                self.gpr_hessian_model,
                self.read_gpr_hessian_folder
        )

        if (not model_hyperparameter_exists) | self.train_hessian_model_bool:
            # the hyper-parameter of the gpr hessian model does not exist.
            # or we want to train the model by setting train_hessian_model as true.
            self.train_gpr_hessian_model()

        gpr_util.analyze_train_error(self.gpr_hessian_model)
            

    def construct_new_gpr_hessian_model(self,
                                        candidate_hessian_point_x):
        """
        """
        
        print(
            "read_gpr_hessian_folder not provided. Will create gpr_hessian model from training data in gpr model."
        )
        # the initial data for gpr_hessian model is the same as gpr_model.
        train_x = np.copy(self.gpr_model.train_cartesian_inputs)
        train_V_shifted = np.copy(self.gpr_model.train_cartesian_targets[:, 0])
        train_grads = np.copy(self.gpr_model.train_cartesian_targets[:, 1:])

        if not self.add_new_hessian_data_bool:
            raise (
            "Error. You must provide hessian data for gpr_hessian training. \
                Either add new hessian data (add_new_hessian_data_bool= True) or read hessian data \
                from read_gpr_hessian_folder"
        )

        if len(self.new_hessian_data_index) == 0:
            raise("Must provide the index of new hessian data point if add_new_hessian_data_bool = True")

        # use the first data point as the reference point for mean function 
        # when constructing gpr model with hessian
        ref_x = candidate_hessian_point_x[self.new_hessian_data_index[0]]
        new_beads = Beads(self.neb_beads.natoms, 1)
        new_forces = self.motion.forces.copy(new_beads, self.motion.cell)
        new_beads.q[0] = ref_x
        
        ref_V_shifted = dstrip(new_forces.pots).copy() - self.energy_shift
        ref_grads = -dstrip(new_forces.f).copy()[0] 

        if self.selective_hessian_bool:
            # initialize the selective_hessian_calculator.
            self.rp_map.construct_selective_hessian_calculator(candidate_hessian_point_x)

            ref_hessians = self.rp_map.selective_hessian_calculator.get_hessian(
                new_beads,
                new_forces,
                np.copy(new_beads.q)
            )
            ref_hessians = ref_hessians[0]

        else:
            # only 1 bead, so no need to transform the hessian.
            ref_hessians = ipi.utils.nebinstool.get_hessian(
                new_beads,
                new_forces,
                np.copy(new_beads.q),
                self.neb_beads.natoms, 
                1
            )

        # include the reference hessian data point into the training data.
        train_x = np.concatenate([train_x, [ref_x]], axis= 0)
        train_V_shifted = np.concatenate([train_V_shifted, ref_V_shifted], axis= 0)
        train_V = train_V_shifted + self.energy_shift 
        train_grads = np.concatenate([train_grads, [ref_grads]], axis= 0)
        train_forces = - train_grads 
        hessian_data_list = np.array([ref_hessians])
        hessian_index = (train_x.shape[0] - 1)
        hessian_index_list = np.array([hessian_index])

        gpr_data = (train_x, train_V, train_forces, 
                    hessian_index_list, hessian_data_list, 
                    None, None)

        self._initialize_gpr_hessian_model(gpr_data)

  
    def load_gpr_hessian_model(self,
                               candidate_hessian_point_x):
        """
        load gpr hessian model. The hessian are already computed.
        """
        gpr_data = self.load_gpr_hessian_training_data(candidate_hessian_point_x)

        self._initialize_gpr_hessian_model(gpr_data)

        self.store_gpr_hessian_model_parameters(self.read_gpr_hessian_folder)

    def cross_validate_gpr_hessian_model(self,
                                         candidate_hessian_point_x):
        """
        read training data (potential V, gradient, hessians) from folder. 
        split data into training set and cross validation set.
        Perform the cross validation. 
        """
        print("Cross validate the gpr hessian model.")
        print("\n")

        (cartesian_coordinate_x, 
        potential_data, force_data, 
        hessian_index_list, hessian_data_list, 
        gpr_fixed_internal_dofs, gpr_rigid_internal_dofs) = self.load_gpr_hessian_training_data(candidate_hessian_point_x)        

        train_set, cv_set, cv_index = gpr_util.split_train_cv_data(
            cartesian_coordinate_x,
            potential_data,
            force_data,
            hessian_index_list,
            hessian_data_list,
            training_ratio = 0.8
        )
        # training data
        train_x, training_V, training_forces, train_hessian_index_list, train_hessian_data_list = train_set 
        # cross validation data.
        cv_x, cv_V, cv_force, cv_hessian_index_list, cv_hessian_data = cv_set 
        
        gpr_data = (train_x, training_V, training_forces,
                    train_hessian_index_list, train_hessian_data_list,
                    gpr_fixed_internal_dofs, gpr_rigid_internal_dofs)

        self._initialize_gpr_hessian_model(gpr_data)

        self.store_gpr_hessian_model_parameters(self.read_gpr_hessian_folder)

        # cross validate the ML model.
        if len(cv_x) > 0:
            cv_V_shifted = cv_V - self.energy_shift
            cv_grads = - cv_force 
            
            print(f"index for cross validation data point: {cv_index}")
            gpr_util.analyze_cross_validation_error(
                self.gpr_hessian_model,
                cv_x,
                cv_V_shifted,
                cv_grads,
                cv_hessian_index_list,
                cv_hessian_data
            )

    def construct_gpr_hessian_model(self):
        """
        construct the gpr_hessian model, which will predict hessian information using Gaussian Process Regression.
        """
        candidate_hessian_point_x, _ = (
            ipi.utils.nebinstool.path_equal_distance_interpolation(
                np.copy(self.neb_beads.q), self.candidate_hessian_data_number
            )
        )
        
        if self.read_gpr_hessian_folder == "None":
            # create gpr_hessian model using data from gpr model
            self.construct_new_gpr_hessian_model(
                candidate_hessian_point_x
            )
            pass
        else:
            if not self.cross_validation_bool:
                self.load_gpr_hessian_model(candidate_hessian_point_x)
            else:
                self.cross_validate_gpr_hessian_model(candidate_hessian_point_x)

            pass    
# ------ construct / load gpr hessian model. -------------------

# ---- add new grad & hessian data to the gpr hessian model. ----
    def add_new_hessian_data(self):
        """
        (1) compute ab initio hessian at new hessian data index.
        (2) add new hessian data into gpr_hessian_model.
        """
        if os.path.exists(os.path.join(self.read_gpr_hessian_folder, "candidate_hessian_data_info.h5") ):
            ab_initio_hessian_file_exists = True 
        else:
            ab_initio_hessian_file_exists = False

        # get the coordinate for the data point that we want to compute the hessian info.
        if ab_initio_hessian_file_exists:
            # read candidate_hessian_point_x, hessian_index_in_candidate_list from self.read_gpr_hessian_folder.
            (candidate_hessian_point_x, self.hessian_index_in_candidate_list) = (
                gpr_util.read_candidate_hessian_data_coordinate(
                    self.read_gpr_hessian_folder
                )
            )
        else:
            candidate_hessian_point_x, _ = (
                ipi.utils.nebinstool.path_equal_distance_interpolation(
                    np.copy(self.neb_beads.q), self.candidate_hessian_data_number
                )
            )
            # index of hessian data that is already computed among candidate data point list.
            self.hessian_index_in_candidate_list = np.array([])

        if self.add_new_hessian_data_bool:
            # find the location of data point we can compute hessian & the index of data point that we have already computed hessians.
            if len(self.new_hessian_data_index) == 0:
                    raise("Must provide the index of new hessian data point if add_new_hessian_data_bool= True")
            
            # the first index of new hessian data index is already used when constructing the model.
            if not ab_initio_hessian_file_exists:
                self.hessian_index_in_candidate_list = np.array([self.new_hessian_data_index[0]])
                self.new_hessian_data_index = self.new_hessian_data_index[1:]

            # handling error when specify the hessian data point.
            assert (
                len(candidate_hessian_point_x) == self.candidate_hessian_data_number
            ), "the candidate hessian data point number read from file is not the same as the one in input.xml"

            if len(self.new_hessian_data_index) != 0:
                assert (
                    np.max(self.new_hessian_data_index) < self.candidate_hessian_data_number
                ), "the index of new hessian data point should not be larger than the number of candidate hessian data point"

            if len(self.new_hessian_data_index) != 0 and len(self.hessian_index_in_candidate_list) != 0: 
                common_index = np.intersect1d(
                    self.new_hessian_data_index, self.hessian_index_in_candidate_list
                )
                assert (
                    len(common_index) == 0
                ), "At least one data point in new_hessian_data_index coincide with the one point that we have already computed hessian.\
                    please double check new_hessian_data_index entry in input.xml"

            if len(self.new_hessian_data_index) > 0:
                # the new data point that we will compute hessian.
                new_hessian_point_x = candidate_hessian_point_x[self.new_hessian_data_index]
                new_hessian_data_num = len(new_hessian_point_x)
                # beads & forces object to call the server to compute hessians.
                natoms = self.neb_beads.natoms
                new_beads = Beads(natoms, new_hessian_data_num)
                new_forces = self.motion.forces.copy(new_beads, self.motion.cell)
                new_beads.q = new_hessian_point_x

                new_pots = new_forces.pots
                new_grads = -dstrip(new_forces.f).copy()

                # compute ab initio hessians of new data points.
                if self.selective_hessian_bool:
                    new_hessians = self.rp_map.selective_hessian_calculator.get_hessian(
                        new_beads,
                        new_forces,
                        np.copy(new_beads.q)
                    )
                else:
                    new_hessians = ipi.utils.nebinstool.get_hessian(
                        new_beads,
                        new_forces,
                        np.copy(new_beads.q),
                        natoms,
                        new_hessian_data_num
                    )

                    new_hessians = np.transpose(
                        np.reshape(
                            new_hessians, [3 * natoms, new_hessian_data_num, 3 * natoms]
                        ),
                        (1, 0, 2),
                    )


                gpr_util.add_hessian_data_to_model(
                    self.gpr_hessian_model,
                    new_hessian_point_x,
                    new_pots,
                    new_grads,
                    new_hessians,
                    self.energy_shift,
                    retrain_bool= False,
                )
        
        return candidate_hessian_point_x

    def add_new_grad_data(self):
        """
        """
        if os.path.exists( os.path.join(self.read_gpr_hessian_folder, "candidate_grad_data_info.h5") ):
            ab_initio_grad_file_exists = True 
        else:
            ab_initio_grad_file_exists = False

        if ab_initio_grad_file_exists:
            (candidate_grad_point_x, self.grad_index_in_candidate_list) = (
                gpr_util.read_candidate_grad_data_coordinate(
                    self.read_gpr_hessian_folder
                )
            )
        else:
            candidate_grad_point_x, _ = (
                ipi.utils.nebinstool.path_equal_distance_interpolation(
                    np.copy(self.neb_beads.q),
                    self.candidate_grad_data_number
                )
            )
            self.grad_index_in_candidate_list = np.array([])

            
        if self.add_new_grad_data_bool:
            # error handling.
            if len(self.new_grad_data_index) == 0:
                raise("Must provide the index of new gradient data point if add_new_grad_data_bool=True")
                
            assert (
                len(candidate_grad_point_x) == self.candidate_grad_data_number
            ), "the candidate gradient data point number read from the file is not the same as the one in input.xml"

            if len(self.new_grad_data_index) != 0:
                assert(
                    np.max(self.new_grad_data_index) < self.candidate_grad_data_number
                ), "the index of new gradient data point should not be larger than the number of candidate gradient data point"

            if len(self.new_grad_data_index) != 0 and len(self.grad_index_in_candidate_list) != 0:
                common_index = np.intersect1d(
                    self.new_grad_data_index, self.grad_index_in_candidate_list
                )

                assert (
                    len(common_index) == 0
                    ), "At least one data point in new_grad_data_index coincide with the one point that we have already computed grads.\
                        please double check new_grad_data_index entry in input.xml" 
            
            new_grad_point_x = candidate_grad_point_x[self.new_grad_data_index]
            new_grad_point_num = len(self.new_grad_data_index)

            natoms = self.neb_beads.natoms 
            new_beads = Beads(natoms, new_grad_point_num)
            new_forces = self.motion.forces.copy(new_beads, self.motion.cell)
            new_beads.q = new_grad_point_x 

            # compute ab initio potentials and forces.
            new_pots = new_forces.pots 
            new_grads = -dstrip(new_forces.f).copy() 

            # add potential and gradient data into the gpr model.
            gpr_util.add_potential_grad_data_to_model(
                self.gpr_hessian_model,
                new_grad_point_x,
                new_pots,
                new_grads,
                self.energy_shift,
                retrain_bool= False
            )
        
        return candidate_grad_point_x

    def store_ab_initio_hessian_and_grad_data(self,
                                              candidate_grad_point_x,
                                              candidate_hessian_point_x):
        """
        store the computed ab initio gradient and hessian data into data folder.
        """
        # if we have updated the data, we store the data set and training hyper-parameters to a given folder.
        if self.add_new_hessian_data_bool or self.add_new_grad_data_bool:
            # create a new data folder with up to date potential, gradient & hessian data.
            # the newly computed hessian will also be stored in this file.
            self.data_destination_folder = (
                gpr_util.store_training_data_in_gpr_hessian_model(
                    self.gpr_hessian_model, self.energy_shift
                )
            )

            # store the hyper-parameters & fix dofs & rigid dofs of the gpr model in the data folder.
            self.store_gpr_hessian_model_parameters(self.data_destination_folder)

            # update candidate hessian data info.
            if self.add_new_hessian_data_bool:
                # update the hessian index with newly computed data point.
                self.hessian_index_in_candidate_list = np.concatenate(
                    [self.hessian_index_in_candidate_list, self.new_hessian_data_index]
                )
            # store candidate_hessian_point_x, hessian_index_in_candidate_list in data destination folder.
            gpr_util.store_candidate_hessian_data_coordinate(
                candidate_hessian_point_x,
                self.hessian_index_in_candidate_list,
                self.data_destination_folder,
            )

            # update candidate gradient data info.
            if self.add_new_grad_data_bool:
                # update the grad index with newly computed data point.
                self.grad_index_in_candidate_list = np.concatenate(
                    [self.grad_index_in_candidate_list, self.new_grad_data_index]
                )
            # store candidate_grad_point_x, grad_index_in_candidate_list in data destination folder.
            gpr_util.store_candidate_grad_data_coordinate(
                candidate_grad_point_x,
                self.grad_index_in_candidate_list,
                self.data_destination_folder
            )

        # if we do selective hessian modeling.
        if self.selective_hessian_bool:
                if self.add_new_hessian_data_bool or self.add_new_grad_data_bool:
                    # store the information about hessian along rigid mode in new folder.
                    self.selective_hessian_calculator.store_rigid_dofs_hessian(self.data_destination_folder)                    
                elif len(self.new_hessian_data_index_rigid_mode) > 0:
                    # we have added new hessian data for rigid mode.
                    self.selective_hessian_calculator.store_rigid_dofs_hessian(self.read_gpr_hessian_folder)
    

    def add_new_hessian_and_grad_data(self):
        """
        (1) compute the new ab initio hessian at new_hessian_data_index.
        (2) Add new hessian data into gpr_hessian_model
        (3) store the updated data set into new folder.
        """
        # For the initial stage, must provide hessian data to add to the training data.
        if (
            not self.add_new_hessian_data_bool
        ) and self.read_gpr_hessian_folder == "None":
            raise (
                "Error. You must provide hessian data for gpr_hessian training. \
                  Either add new hessian data (add_new_hessian_data_bool= True) or read hessian data \
                  from read_gpr_hessian_folder"
            )

        # Now we add ab initio hessian data along the path into the gpr model.
        candidate_hessian_point_x = self.add_new_hessian_data()

        # Now we add ab initio grad data along the path into the gpr model.
        candidate_grad_point_x = self.add_new_grad_data() 

        # train the model.
        if (self.add_new_hessian_data_bool or self.add_new_grad_data_bool) and self.train_hessian_model_bool:
            self.train_gpr_hessian_model()

            gpr_util.analyze_train_error(self.gpr_hessian_model)
                
        # store the computed ab inito gradient and hessian data if we compute new data point. 
        self.store_ab_initio_hessian_and_grad_data(candidate_grad_point_x,
                                                candidate_hessian_point_x)

# ---- add new grad & hessian data to the gpr hessian model. ----

