"""
Code to interface the i-pi module with gpr code.
"""
import gpr.gprtools 
import numpy as np 

from ipi.engine.motion.neb_instanton_gpr import MAPNEBGPRMover
from ipi.engine.motion import Motion 
from ipi.utils.depend import dstrip
from ipi.utils.scripting import (
    InteractiveSimulation
)
import internal_util
import gpr_util 
from ipi.engine.beads import Beads

class ActiveLearning(object):
    def __init__(self,
                 sim: InteractiveSimulation,
                 motion: MAPNEBGPRMover):
        self.sim = sim
        self.motion = motion 

        self.total_steps = sim.tsteps
        # bead and forces for cross validation.
        self.gpr_beads = Beads(motion.beads.natoms, 1)
        self.gpr_forces = motion.forces.copy(self.gpr_beads, motion.cell)

        stage = motion.options["stage"]
        if stage == 'neb':
            # initialize the internal coordinate and gpr model.
            self.initialize_gpr_model()
            # check the training error and cross-validation error of the gpr model.
            self.check_training_result()
    
    def run_one_step(self, write_outputs= True):
        """
        run the simulation for steps.
        """
        self.sim.run(steps= 1, write_outputs= write_outputs)

        # update the gpr model.
        if self.motion.options["stage"] == 'neb':
            self.update_gpr_model()

    def run(self, write_outputs= True):
        """
        run for total number of steps.
        """
        for step in range(self.sim.step, self.total_steps):
            self.run_one_step(write_outputs= write_outputs)

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
        if read_gpr_training_data_bool:
            # see if there is option to read hyper-parameter without training the model
            neb_final_gpr_folder = "neb_final_gpr_training"
            model_hyperparameter_exists = gpr_util.load_training_hyperparameter_in_gpr_model(
                gpr_model, neb_final_gpr_folder
            )

            if not model_hyperparameter_exists:
                gpr_model.train_gpr()

        else:
            gpr_model.train_gpr()
        
        return gpr_model

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

        # initialize the gpr model use training data and internal coordinate transformer.
        gpr_model = self._initialize_gpr_model(train_x, train_V, train_grad, coordinate_transformer)
        
        # bind the coordinate transformer and gpr model to the motion object.
        # this will enable the motion object to use gpr model to predict potential and force.
        self.coordinate_transformer = coordinate_transformer
        self.gpr_model = gpr_model 

        self.motion.bind_gpr_model(gpr_model, coordinate_transformer)

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
    
    def update_gpr_model(self):
        """
        update the Gaussian Process Regression model at the end of each simulation step.
        """
        step = self.sim.step 
        early_stop_bool = self.motion.early_stop_bool
        outrange_bead_index_list = self.motion.outrange_bead_index_list

        gpr_trust_region = self.motion.optarrays["gpr_trust_region"]
        print("The trust region for the gpr model now: " + str(gpr_trust_region))
        