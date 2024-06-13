"""Holds the algorithms to perform nudged elastic band (NEB) calculations to find instanton path.
J. Chem. Phys. 148, 102334 (2018); https://doi.org/10.1063/1.5007180

The NEB calculation is accelerated by Gaussian Process Regression method. See: J. Chem. Phys. 147, 152720 (2017) and Faraday Discuss., 2018,212, 237-258 (https://doi.org/10.1039/C8FD00085A)

The algorithms are first implemented by Chenghao Zhang, 2023. Adapted from neb module & instanton module in i-pi package.
Algorithm for using li-neb to search instanton path.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2021 i-PI developers
# See the "licenses" directory for full license information.

import sys
import numpy as np
from numpy.linalg import norm as npnorm
import scipy 
import time
from ipi.utils import units
from ipi.engine.normalmodes import NormalModes
from ipi.engine.motion import Motion
from ipi.utils.depend import dstrip
from ipi.utils.softexit import softexit
from ipi.utils.mintools import Damped_BFGS, FIRE
from ipi.utils.messages import verbosity, info
from ipi.engine.beads import Beads
import ipi.utils.nebinstool
from ipi.utils.nebinstool import RK4, dydt_inverted_pot
from ipi.utils.internalcoordtools import non_redundant_coordinate_transformer
import ipi.utils.gprtools
import ipi.utils.nebinstgprtool
import ipi.utils.nebinstool
from timeit import default_timer as timer

np.set_printoptions(threshold=10000, linewidth=1000)  # Remove in cleanup

__all__ = ["LINEBGradientMapper", "MAPNEBGPRMover"]


class MAPNEBGPRMover(Motion):
    """Nudged elastic band routine. for minimum action path (MAP)
    See J. Chem. Phys. 148, 102334 (2018)
    Attributes:
    mode: optimization method to optimize the Nudged elastic band 
    prefix: prefix for output file
    tolerances: convergence tolerance.  "gradient" : gradient criteria for action gradient for interior beads and end beads.
    energy_shift: Set the zero energy (unit Hatree). For DFT, it's usually set as the energy of the ground state.
    time_step: time step for MD simulation used in Verlet algorithm of Nudged elastic band
    stage: stage of calculation.  default : "neb"
    instanton_bead_number: The number (P) of ring-polymer generated along Minimum action path (MAP).  Do not confuse it with nbeads, which is the NEB replica number (N). typicallly P > N.
    instanton_path_energy: the energy for the dynamics on the inverted PES. = the potential energy (V) of the end beads for minimum action Path.
    spring_k : spring constant between Nudged elastic band. (for force on mass-scaled coordinate)
    kappa: energy restraint term on end beads. (for force on mass-scaled coordinate)
    alt_out: step # to output neb beads & energy information.
    """

    def __init__(
        self,
        fixcom=False,
        fixatoms=None,
        mode = "verlet",
        prefix = "neb_instanton",
        tolerances = { "gradient": 5e-3},
        energy_shift = 0.00,
        time_step = 4.0,
        instanton_time_step = 4.0,
        stage = "neb",
        instanton_bead_number = 20,
        instanton_path_energy = 0.00,
        instanton_temperature = 1.0,
        instanton_bead_q = np.zeros(0, float),
        instanton_bead_pot = np.zeros(0, float),
        instanton_hessian =  np.eye(0,0,0, float),
        path_interpolation_bead_number = 20,
        spring_k = 0.1,
        kappa = { "left" : 50, "right": 50 },
        final_hessian_bool = False,
        alt_out = 5,
        gpr_force_criterion = 0.02,
        gpr_trust_region_ratio = 0.05,
        gpr_kernel_outputscale_prior_mean = np.zeros(0, float),
        gpr_kernel_lengthscale_prior_mean_ratio = np.zeros(0, float),
        gpr_likelihood_noise_std_constraint = {"pot_noise_prior": 1e-5, "force_noise_prior": 1e-5},
        gpr_SE_kernel_number = 1,
        read_initial_gpr_training_data = False
    ):
        """Initialises NEBMover.

        Args:
           fixcom: An optional boolean which decides whether the centre of mass
              motion will be constrained or not. Defaults to False.
        """
        super(MAPNEBGPRMover, self).__init__(fixcom=fixcom, fixatoms=fixatoms)

        # parameters to pass in from input.xml
        self.options = {}
        
        # mode for optimization
        self.options["mode"] = mode  

        self.options["stage"] = stage
        self.options["tolerances"] = tolerances
        self.options["alt_out_step"] = alt_out   # step to output geometry.
        self.options["prefix"] = prefix 
        self.options["final_hessian_bool"] = final_hessian_bool
        self.options["read_initial_gpr_training_data"] = read_initial_gpr_training_data

        # numerical values / arrays. option from input.xml
        self.optarrays = {}
        self.optarrays["energy_shift"] = energy_shift 
        
        self.optarrays["spring_k"] = spring_k
        self.optarrays["kappa"] = kappa

        self.optarrays["time_step"] = time_step
        self.optarrays["instanton_time_step"] = instanton_time_step

         # number of beads to interpolate MAP path to generate instanton beads.
        self.optarrays["path_interpolation_bead_number"] = path_interpolation_bead_number

        # input variable for instanton 
        self.optarrays["instanton_path_energy"] = instanton_path_energy 
        self.optarrays["instanton_bead_number"] = instanton_bead_number

        # for store the instanton result in RESTART file
        self.optarrays["instanton_temperature"] = instanton_temperature
        self.optarrays["instanton_bead_q"] = instanton_bead_q
        self.optarrays["instanton_bead_pot"] = instanton_bead_pot 
        self.optarrays["instanton_hessian"] = instanton_hessian 

        self.nebgm = LINEBGradientMapper()

        # variables for neb move
        self.velocity_mscaled = None 
        self.x = None 
        self.action = None 
        self.f_mscaled = None 

        # variable below is for Gaussian Process Regression.
        # index list indicating that the ab-initio forces is close to gpr predicted forces.
        # we use this list when we try to converge the calculation.
        if np.shape(gpr_kernel_outputscale_prior_mean) == (0,):
            raise("You must provide output scale for covariance function. This should be a numpy array, with size equal to number of Squared Exponential (SE) kernel you use.")
        if np.shape(gpr_kernel_lengthscale_prior_mean_ratio) == (0,):
            raise("You must provide length scale for covariance function. This should be a numpy array, with size equal to number of Squared Exponential (SE) kernel you use.")

        assert len(gpr_kernel_lengthscale_prior_mean_ratio) == gpr_SE_kernel_number, "The number of length scale of kernels should match the number of Squared Exponential kernel you use"
        assert len(gpr_kernel_outputscale_prior_mean) == gpr_SE_kernel_number, "The number of output scale of kernels should match the number of Squared Exponential kernel you use."

        self.optarrays["gpr_force_criterion"] = gpr_force_criterion # criterion to stop the full calculation.
        self.optarrays["gpr_trust_region_ratio"] = gpr_trust_region_ratio
        self.optarrays["gpr_kernel_outputscale_prior_mean"] = gpr_kernel_outputscale_prior_mean
        self.optarrays["gpr_kernel_lengthscale_prior_mean_ratio"] = gpr_kernel_lengthscale_prior_mean_ratio
        self.optarrays["gpr_likelihood_noise_std_constraint"] = gpr_likelihood_noise_std_constraint
        self.options["gpr_SE_kernel_number"]  = gpr_SE_kernel_number

        self.ab_initio_index_list = []
        self.force_diff_ratio_list = []

        self.coordinate_transformer = None 
        self.gpr_model = None 


        self.ab_initio_force_calculation_number = 0

        self.start_time = timer()

    def bind(self, ens, beads, nm, cell, bforce, prng, omaker):
        super(MAPNEBGPRMover, self).bind(ens, beads, nm, cell, bforce, prng, omaker)
        if len(self.fixatoms) == len(self.beads[0]):
            softexit.trigger(
                status="bad",
                message="WARNING: all atoms are fixed, geometry won't change. Exiting simulation.",
            )

        # Redefine normal modes
        self.nm = NormalModes(
            transform_method="matrix", open_paths=np.arange(self.beads.natoms)
        )

        self.nm.bind(self.ensemble, self, Beads(self.beads.natoms, self.beads.nbeads))

        # fixatoms mask.
        self.fixatoms_mask = np.ones(3 * self.beads.natoms, dtype=bool)
        if len(self.fixatoms) > 0:
            self.fixatoms_mask[3 * self.fixatoms] = 0
            self.fixatoms_mask[3 * self.fixatoms + 1] = 0
            self.fixatoms_mask[3 * self.fixatoms + 2] = 0

        # create bead object that is used to add training data to GPR model.
        self.gpr_beads = Beads(self.beads.natoms, 1)
        self.gpr_forces = self.forces.copy(self.gpr_beads, self.cell)

        # create bead object that is used to add more training data to GPR model during initial training.
        self.new_gpr_beads = Beads(self.beads.natoms, self.beads.nbeads)
        self.new_gpr_forces = self.forces.copy(self.new_gpr_beads, self.cell)

        self.nebgm.bind(self)
    
    def generate_initial_training_data(self):
        '''
        generate training data for Gaussian Process Regression model
        '''
        # choose all NEB beads as initial training data.
        # We will train the GPR model to optimize hyperparameter when we initialize it.
        train_x = np.copy(self.beads.q)
        # potential energy has to shift relative to the energy_shift for training.
        train_V = np.copy(self.forces.pots) - self.optarrays["energy_shift"]
        train_grad = - np.copy(dstrip(self.forces.f))
        # count the # of ab-initio calculation we have done.
        self.ab_initio_force_calculation_number = self.ab_initio_force_calculation_number + self.beads.nbeads

        # option to add more training data between data points.
        self.new_gpr_beads.q[:-1] = (self.beads.q[:-1] + self.beads.q[1:])/2
        self.new_gpr_beads.q[-1] = self.beads.q[-1] + (self.beads.q[-1] - self.beads.q[-2]) / 2

        new_train_x = np.copy(self.new_gpr_beads.q)
        new_train_V = np.copy(self.new_gpr_forces.pots)- self.optarrays["energy_shift"]
        new_train_grad = - np.copy(dstrip(self.new_gpr_forces.f))

        # concatenate training data.
        train_x = np.concatenate([train_x, new_train_x], axis = 0)
        train_V = np.concatenate([train_V, new_train_V], axis = 0)
        train_grad = np.concatenate([train_grad, new_train_grad], axis = 0)
        # count the # of ab-initio calculation we have done.
        self.ab_initio_force_calculation_number = self.ab_initio_force_calculation_number + self.new_gpr_beads.nbeads

        # add more training data between data points
        end_bead_index1 = 2
        end_bead_index2 = 5
        bead_path_for_interpolation = self.beads.q[end_bead_index1 : end_bead_index2 + 1, :]
        interpolation_bead_number = 12
        spline_x, _ = ipi.utils.nebinstool.path_cubic_interpolation(bead_path_for_interpolation, interpolation_bead_number)
        self.new_gpr_beads.q[:] = spline_x[1:-1]

        new_train_x2 = np.copy(self.new_gpr_beads.q)
        new_train_V2 = np.copy(self.new_gpr_forces.pots) - self.optarrays["energy_shift"]
        new_train_grad2 = - np.copy(dstrip(self.new_gpr_forces.f))
        # concatenate training data.
        train_x = np.concatenate([train_x, new_train_x2], axis = 0)
        train_V = np.concatenate([train_V, new_train_V2], axis = 0)
        train_grad = np.concatenate([train_grad, new_train_grad2], axis = 0)
        # count the # of ab-initio calculation we have done.
        self.ab_initio_force_calculation_number = self.ab_initio_force_calculation_number + self.new_gpr_beads.nbeads


        # store the initial training_data
        train_V_to_store = train_V + self.optarrays["energy_shift"]
        train_f_to_store = - train_grad 
        ipi.utils.nebinstgprtool.store_initial_training_data(train_x, train_V_to_store, train_f_to_store)

        return train_x, train_V, train_grad 
    
    def read_initial_training_data(self):
        '''
        read initial training data stored in files (previously computed)
        '''
        train_x, stored_train_V, stored_train_f = ipi.utils.nebinstgprtool.read_initial_training_data()
        # count the # of ab-initio calculation we have done
        ab_initio_calculation_number = np.shape(train_x)[0]
        self.ab_initio_force_calculation_number = self.ab_initio_force_calculation_number + ab_initio_calculation_number 

        train_V = stored_train_V - self.optarrays["energy_shift"]
        train_grad = - stored_train_f 

        return train_x, train_V, train_grad

    def initialialize_GPR_model(self):
        '''
        initialize the gaussian process regression model.
        1. Initialize coordinate transformer to transform between internal coordinate and cartesian coordinate.
        2. initialize GPR_Wrapper, which combines coordinate transformer and GPR model.
        '''
        # Initialize non redundant coordinate transformer.
        # choose the middle point of the initial instanton path as reference point.
        nbeads = self.beads.nbeads
        beads_pots = np.copy(self.forces.pots)
        bead_index_at_transition_state = np.argmax(beads_pots)

        ref_x = dstrip(self.beads.q[bead_index_at_transition_state]).copy()

        self.coordinate_transformer = non_redundant_coordinate_transformer(self.beads.natoms, ref_x)

        # attach ab_initio potential to self.nebgm.ab_initio_pot and self.nebgm.ab_initio_force
        self.nebgm.ab_initio_pot = np.copy(self.forces.pots)
        self.nebgm.ab_initio_force = np.copy(dstrip(self.forces.f))
        self.initial_force_amplitude = npnorm(dstrip(self.forces.f), axis = 1)

        read_gpr_training_data_bool = self.options["read_initial_gpr_training_data"]
        if not read_gpr_training_data_bool:
            train_x, train_V, train_grad = self.generate_initial_training_data()
        else:
            train_x, train_V, train_grad = self.read_initial_training_data()

        self.gpr_model = ipi.utils.gprtools.GPModelWithDerivativesWrapper(train_x, train_V, train_grad,
                                                                     self.beads.natoms, self.coordinate_transformer,
                                                                     gpr_SE_kernel_number= self.options["gpr_SE_kernel_number"],
                                                                    kernel_initial_outputscale= self.optarrays["gpr_kernel_outputscale_prior_mean"],
                                                                    kernel_prior_lengthscale_ratio= self.optarrays["gpr_kernel_lengthscale_prior_mean_ratio"],
                                                                    likelihood_noise_std_constraint= self.optarrays["gpr_likelihood_noise_std_constraint"])

    def check_initial_training_result(self):
        '''
        check whether the training of GPR model is successful. If not, stop the simulation and report error
        '''
        # self.output_recommended_hyper_parameter()

        predicted_V_shift, predicted_grad, _, _ = self.gpr_model.predict_latent_function(self.beads.q) 

        predicted_forces = - predicted_grad 

        ab_initio_V_shift = self.forces.pots - self.optarrays["energy_shift"]
        ab_initio_forces = self.forces.f 

        # check length scale for possible over fitting
        learned_kernel_length_scale = self.gpr_model.output_kernel_lengthscale()
        internal_input_range = np.max(self.gpr_model.normalized_train_inputs, axis=0) - np.min(self.gpr_model.normalized_train_inputs, axis = 0)

        scaled_kernel_lengthscale = learned_kernel_length_scale / internal_input_range 
        kernel_output_scale_var = self.gpr_model.output_kernel_outputscale()
        kernel_output_scale_std = np.sqrt(kernel_output_scale_var)

        print("\n")
        print("@check the overfitting and underfitting of kernel length scale")
        for i in range(self.gpr_model.gpr_SE_kernel_number):
            print("kernel {}: ".format(i))
            print("square root of kernel output scale (\u03C3): " + str(kernel_output_scale_std[i]))
            print("kernel_length_scale / input scale:   " + str(scaled_kernel_lengthscale[i])  )
        print("\n")

        force_range = self.gpr_model.output_normalized_force_range()
        V_noises, force_noises = self.gpr_model.output_fitted_gpr_model_noises()
        force_noises_ratio = force_noises / force_range
        print("force noise amplitude: " + str(force_noises_ratio))
        print("potential noise amplitude: " + str(V_noises))

        # check energy:
        V_error = np.abs(ab_initio_V_shift - predicted_V_shift) / np.abs(ab_initio_V_shift)

        # check force:
        df = np.linalg.norm( ab_initio_forces - predicted_forces , axis = 1)
        ab_initio_force_amplitude = np.linalg.norm(ab_initio_forces, axis = 1)
        df_error = df / ab_initio_force_amplitude
        
        print("\n")
        print("@initial Gaussian Process Regression fitting:")
        print("error of potential prediction: " + str(V_error))
        print("error of force prediction: " + str(df_error))
        print("\n")

        # check overfitting.
        print("@Test Overfitting of GPR model.")

        test_q = self.beads.q[0] * 1/4 + self.beads.q[1] * 3/ 4
        print("q[0] * 1/4 + q[1] * 3/4")
        predicted_test_V_shift, predicted_test_force, ab_initio_test_pot, ab_initio_test_force = ipi.utils.nebinstgprtool.check_gpr_fitting_error(self.gpr_beads, self.gpr_forces, self.gpr_model, self.optarrays["energy_shift"], test_q)

        test_q = self.beads.q[3] * 1/4 + self.beads.q[2] * 3/4
        print("q[3] * 1/4 + q[2] * 3/4")
        predicted_test_V_shift, predicted_test_force, ab_initio_test_pot, ab_initio_test_force = ipi.utils.nebinstgprtool.check_gpr_fitting_error(self.gpr_beads, self.gpr_forces, self.gpr_model, self.optarrays["energy_shift"], test_q)
        
        test_q = self.beads.q[4] * 1/4 + self.beads.q[5] * 3/ 4
        print("q[4] * 1/4 + q[5] * 3/4")
        predicted_test_V_shift, predicted_test_force, ab_initio_test_pot, ab_initio_test_force = ipi.utils.nebinstgprtool.check_gpr_fitting_error(self.gpr_beads, self.gpr_forces, self.gpr_model, self.optarrays["energy_shift"], test_q)
        
        test_q = self.beads.q[7] * 1/4 + self.beads.q[8] * 3/4
        print("q[7] * 1/4 + q[8] * 3/4")
        predicted_test_V_shift, predicted_test_force, ab_initio_test_pot, ab_initio_test_force = ipi.utils.nebinstgprtool.check_gpr_fitting_error(self.gpr_beads, self.gpr_forces, self.gpr_model, self.optarrays["energy_shift"], test_q)

        pass 



    def update_GPR_model(self, early_stop_bool, outrange_bead_index, step):
        '''
        update GPR model with new training data. Which new training data to be added will depend on stop criterion.
        evaluate potential and force of one bead. 
        Then update the Gassian Process Regression model.
        '''
        attempt_exit_bool = False 

        bead_index_for_update = -1
        if early_stop_bool:
            # in this case, one bead move out of trust region.
            bead_index_for_update = outrange_bead_index
            training_x = dstrip(self.beads.q[bead_index_for_update]).copy()
            training_x = np.array([training_x])
            training_V_shift, training_grad_x, _, _ = self.gpr_model.predict_observable(training_x)
            training_bead_forces = - training_grad_x[0]
        else:
            # in this case, NEB calculation converges on GPR fitted PES.
            # find the bead with the largest energy uncertainty.
            beads_V_shift, beads_grad_x, beads_var_V, beads_var_grad_q = self.gpr_model.predict_observable(self.beads.q)
            beads_forces = - beads_grad_x 
            bead_index_for_update = np.argmax(beads_var_V)
            training_bead_forces = beads_forces[bead_index_for_update]
            training_x = dstrip(self.beads.q[bead_index_for_update]).copy()
            training_x = np.array([training_x])

        # consistency check : the gpr_beads we claims to do the simulation should have same bead number of training data.
        assert self.gpr_beads.nbeads == len(training_x)
        self.gpr_beads.q[:] = training_x 

        # get energy and forces (in Cartesian coordinate) from force engine.
        ab_initio_beads_energy = dstrip(self.gpr_forces.pots).copy() 
        ab_initio_beads_forces = dstrip(self.gpr_forces.f).copy() 
        ab_initio_beads_grad = - ab_initio_beads_forces

        # set ab_initio pot and force in nebgm.
        self.nebgm.ab_initio_pot[bead_index_for_update] = ab_initio_beads_energy 
        self.nebgm.ab_initio_force[bead_index_for_update] = ab_initio_beads_forces

        # count the # of ab-initio calculation we have done.
        self.ab_initio_force_calculation_number = self.ab_initio_force_calculation_number + 1

        ab_initio_force_amplitude = np.linalg.norm(ab_initio_beads_forces[0])
        force_diff = training_bead_forces - ab_initio_beads_forces[0]
        force_diff_ratio = np.linalg.norm(force_diff) / ab_initio_force_amplitude
        self.force_diff_ratio_list.append(force_diff_ratio)
        self.ab_initio_force_amplitude_list = [ab_initio_force_amplitude]
        self.gpr_force_prediction_amplitude_list = [np.linalg.norm(training_bead_forces)]

        if not early_stop_bool:
            if force_diff_ratio < self.optarrays["gpr_force_criterion"]:
                # the ab-initio force is close to the force predicted by GPR. we check forces on other beads and try to exit.
                self.ab_initio_bead_index = [bead_index_for_update]
                attempt_exit_bool = True

        # update GPR model with coordinate (training_x), potential (beads_energy) and forces in cartesian coordiante (beads_forces)
        ab_initio_beads_energy_shift = ab_initio_beads_energy - self.optarrays["energy_shift"]
        self.gpr_model.update_model_with_new_data(training_x, ab_initio_beads_energy_shift, ab_initio_beads_grad)
        self.nebgm.gpr_model = self.gpr_model

        if attempt_exit_bool:
            self.neb_stage_exit_step(step)

        # output info about force diff ratio |f_GPR -f|/|f|
        print("@Outerloop Exit info: ab initio |f|: " + str(self.ab_initio_force_amplitude_list))
        print("@Outloop Exit info: GPR predicted |f_GPR|: " + str(self.gpr_force_prediction_amplitude_list))
        print("For reference: maximum |f| in initial training data: max: {:.4f},   min: {:.4f}".format(np.max(self.initial_force_amplitude), np.min(self.initial_force_amplitude))  )
        print("@Outerloop Exit info: |f_GPR -f|/|f|:" + str(self.force_diff_ratio_list))        
        print("Finish Outerloop: " + str(step))
        print("\n")
        print("\n")

        self.force_diff_ratio_list = []
        self.ab_initio_force_amplitude_list = []
        self.gpr_force_prediction_amplitude_list = []
        


    def step(self, step=None):
        """Does one simulation time step.
        """
        print(" @NEB Outerloop STEP %d, stage: %s" % (step, self.options["stage"]))
        
        # print initial geometry and energy of neb path.
        if step == 0:
            ipi.utils.nebinstool.print_neb_instanton_geo(
                self.options["prefix"] + "_initial_",
                step,
                self.beads.nbeads,
                self.beads.natoms,
                self.beads.names,
                self.beads.q,
                self.forces.pots,
                self.cell,
                self.optarrays["energy_shift"],
                self.output_maker
            )

            # convert unit for spring_k , kappa. only do it for STEP = 0, not for RESTART simulation. 
            self.optarrays["spring_k"] = self.optarrays["spring_k"] / np.power( units.unit_to_internal("length", "angstrom", 1) , 2)  # input unit: angstrom^{-2}
            self.optarrays["kappa"]["left"] = self.optarrays["kappa"]["left"] / ( units.unit_to_internal("length" , "angstrom", 1) * units.unit_to_internal("energy", "electronvolt", 1) )
            self.optarrays["kappa"]["right"] = self.optarrays["kappa"]["right"] / ( units.unit_to_internal("length" , "angstrom", 1) * units.unit_to_internal("energy", "electronvolt", 1) )
            self.nebgm.spring_k = (self.optarrays["spring_k"]).copy()
            self.nebgm.kappa = dict(self.optarrays["kappa"])

            # Only do it for initial calculation. Not for restart.
            self.optarrays["instanton_path_energy"] = self.optarrays["instanton_path_energy"] + self.optarrays["energy_shift"]  # shift the instanton path energy according to energy shift.
            self.nebgm.instanton_path_energy = self.optarrays["instanton_path_energy"]
            # TODO: assign instanton path energy also for RP_MAP object.
        
        if self.coordinate_transformer == None:
            # initialize Gaussian Process Regression(GPR) model and coordiante transformer
            self.initialialize_GPR_model()

            self.check_initial_training_result()

            self.compute_initial_neb_path_length_in_scaled_internal_coordinate()

            # bind the gpr model and coordinate_transformer to the LINEGradientMapper class
            self.nebgm.gpr_model = self.gpr_model 
            self.nebgm.coordinate_transformer = self.coordinate_transformer


        # Check if we restarted a converged calculation (by mistake)
        if self.options["stage"] == "converged":
            softexit.trigger(
                status="success",
                message="neb calculation converged. Instanton geometry calculation finishes. Exiting simulation",
            )

        if self.options["stage"] == "neb":
            # use nudged elastic band method to find minmum action path.
            # then we will switch to the stage "instanton"
            early_stop_bool, outrange_bead_index = self.neb_loop(step)

            # update Gaussian Process Regression model with new training data
            self.update_GPR_model(early_stop_bool, outrange_bead_index, step)


    def neb_stage_exit_step(self, step):
        '''
        check the ab-initio forces and compare it with forces predicted by GPR.
        We do not move NEB path during this process.
        If all beads pass the test: their ab-initio forces are close to GPR predicted forces,
        then we exit the NEB loop.
        '''
        while(len(self.ab_initio_bead_index) < self.beads.nbeads):
            # gpr bead index is the index list that we still need to verify the ab-initio forces.
            gpr_bead_index_list = np.array(range(self.beads.nbeads))
            gpr_bead_index_list = np.delete(gpr_bead_index_list, self.ab_initio_bead_index)

            beads_V, beads_grad_x, beads_var_V, beads_var_grad_q = self.gpr_model.predict_observable(self.beads.q)
            beads_forces = -beads_grad_x

            # find the bead that has the largest energy variance and we haven't evaluated its ab-initio potential.
            index_in_gpr_bead_index_list = np.argmax(beads_var_V[gpr_bead_index_list])
            bead_index_for_update = gpr_bead_index_list[index_in_gpr_bead_index_list]

            # compute the ab-initio force and potential for the given bead.
            training_x = dstrip(self.beads.q[bead_index_for_update]).copy()
            training_x = np.array([training_x])
            self.gpr_beads.q[:] = training_x

            # get energy and forces (in Cartesian coordinate) from force engine.
            ab_initio_beads_energy = dstrip(self.gpr_forces.pots).copy() 
            ab_initio_beads_energy_shift = ab_initio_beads_energy - self.optarrays["energy_shift"]
            ab_initio_beads_forces = dstrip(self.gpr_forces.f).copy() 
            ab_initio_beads_grad = - ab_initio_beads_forces

            # update it in nebgm.ab_initio_pot and nebgm.ab_initio_force
            self.nebgm.ab_initio_pot[bead_index_for_update] = ab_initio_beads_energy 
            self.nebgm.ab_initio_force[bead_index_for_update] = ab_initio_beads_forces

            # count the # of ab-initio calculation we have done.
            self.ab_initio_force_calculation_number = self.ab_initio_force_calculation_number + 1

            # update the model with ab-initio data.
            self.gpr_model.update_model_with_new_data(training_x, ab_initio_beads_energy_shift, ab_initio_beads_grad)

            # check whether the ab-inito force is close to the gpr predicted force
            force_diff = beads_forces[bead_index_for_update] - ab_initio_beads_forces[0]
            force_diff_ratio = np.linalg.norm(force_diff) / np.linalg.norm(ab_initio_beads_forces[0])
            
            self.force_diff_ratio_list.append(force_diff_ratio)
            self.ab_initio_force_amplitude_list.append( np.linalg.norm(ab_initio_beads_forces[0]) )
            self.gpr_force_prediction_amplitude_list.append( np.linalg.norm(beads_forces[bead_index_for_update]) )

            if force_diff_ratio < self.optarrays["gpr_force_criterion"]:
                self.ab_initio_bead_index.append(bead_index_for_update)
            else:
                # the current bead configuration has not converged yet. Need to do Nudged Elastic band on updated surface.
                self.ab_initio_bead_index = []

                break 
        
        beads_pots_shift, beads_forces, _, _ = self.gpr_model.predict_latent_function(self.beads.q)
        beads_pots = beads_pots_shift + self.optarrays["energy_shift"]

        # all beads pass the test. The simulation has converged
        if len(self.ab_initio_bead_index) == self.beads.nbeads:
            info( "@Exit step: NEB_instanton: path optimization converged. Step %i \n" % step, verbosity.low)

            # print neb beads geometry and energy.
            ipi.utils.nebinstool.print_neb_instanton_geo(
                self.options["prefix"] + "_neb_FINAL",
                step,
                self.beads.nbeads,
                self.beads.natoms,
                self.beads.names,
                self.beads.q,
                beads_pots,
                self.cell,
                self.optarrays["energy_shift"],
                self.output_maker
            )

            # output number of ab-initio calculation.
            ipi.utils.nebinstgprtool.print_ab_initio_calculation_number(self.ab_initio_force_calculation_number, self.output_maker)
            print("ab initio calculation number : " + str(self.ab_initio_force_calculation_number))

            # output the time for execuation 
            self.end_time = timer()
            time_elapsed = (self.end_time - self.start_time) / 60  # time elapsed in minutes 
            print("the running time for the program: " + str(time_elapsed) + " min.")

            self.options["stage"] = "converged"
        

    def neb_loop(self, outer_loop_step):
        '''
        Finish the inner loop of neb 
        '''
        grad_max = 1000
        tolerances = self.options["tolerances"]
        
        # inner loop doing neb
        neb_step = 0  # count the step number of neb move. (inner loop)
        
        early_stop_bool = False
        outrange_bead_index = -1 # index for beads that move out of trusted region that causes early stop.

        self.neb_initialize(outer_loop_step) # we have to re-initialize Nudged Elastic Band variable 
        
        self.print_geometry(outer_loop_step)

        print("\n")
        print("@Start outer loop: " + str(outer_loop_step) + "\n")
        while grad_max > tolerances["gradient"]:
            grad_max, early_stop_bool, outrange_bead_index = self.neb_step(outer_loop_step, neb_step)
            neb_step = neb_step + 1

            # beads move out of trust region.
            if early_stop_bool:
                break 
        
        return early_stop_bool, outrange_bead_index

        
    def neb_step(self, outer_loop_step, neb_step):
        '''
        doing neb move for one step.
        '''
        n_activedim = self.beads.q[0].size - len(self.fixatoms) * 3
        nbeads = self.beads.nbeads
        dt = self.optarrays["time_step"]

        # check if spring_k and kappa value is appropriate.
        self.check_spring_k_kappa()

        grad_max = 0
        # check early stop condition if there are beads out of trust region
        early_stop_bool, outrange_bead_index = ipi.utils.nebinstgprtool.check_neb_early_stop(self.beads.q,
                                                                                            self.optarrays["gpr_trust_region_ratio"],
                                                                                            self.gpr_model,
                                                                                            self.scaled_internal_coordinate_neb_path_length,
                                                                                            self.initial_effective_kernel_length_scale)
        
        # stop the step early if there are beads out of trust region.
        if early_stop_bool:
            return grad_max, early_stop_bool, outrange_bead_index

        # neb move using gradient of LINEBGradient
        if self.options["mode"] == "verlet":
            # Only initialize velocity for fresh start, not for RESTART
            dx_mscaled = dt * self.velocity_mscaled + 0.5 * self.f_mscaled * np.power(dt, 2)
            dx = dx_mscaled / np.sqrt(self.beads.m3[:, self.fixatoms_mask])

            # update position
            self.old_x = np.copy(self.x)
            self.x = self.x + dx
            self.beads.q[:, self.fixatoms_mask] = self.x

            self.old_f_mscaled = np.copy(self.f_mscaled) # record old force
            self.old_action = self.action
            self.f_mscaled, self.action = self.nebgm(self.x)  # evaluate the force & action using the updated position

            self.velocity_mscaled = self.velocity_mscaled + dt * (self.old_f_mscaled + self.f_mscaled) / 2

            # project velocity along the direction of the current force
            f_unit_vector = self.f_mscaled / np.linalg.norm(self.f_mscaled)

            v_f_inner_product = np.inner( f_unit_vector.flatten() , self.velocity_mscaled.flatten() )

            if v_f_inner_product < 0:
                self.velocity_mscaled = np.zeros([self.beads.nbeads, 3 * (self.beads.natoms - len(self.fixatoms))])
            else:
                self.velocity_mscaled = v_f_inner_product * f_unit_vector


        else:
            softexit.trigger(
                status="bad",
                message="Only projected velocity verlet is implemented. set mode == 'verlet' ",
            )

        grad_max = np.amax(npnorm(self.nebgm.neb_optimization_force, axis = 1))

        # output info about neb calculation.
        self.neb_instanton_step_info(outer_loop_step, neb_step, grad_max)
        
        return grad_max, early_stop_bool, outrange_bead_index   

    def neb_initialize(self, step):
        '''
        initialize action, force, velocity for nudged elastic band calculation. (inner loop calculation.)
        '''
        info(
            " @NEB: start inner loop neb for step {}".format(step),
            verbosity.debug,
        )

        self.velocity_mscaled = np.zeros([self.beads.nbeads, 3 * (self.beads.natoms - len(self.fixatoms))])  # velocity of free moving particles on mass scaled coordinate.
        self.old_f_mscaled = np.zeros([self.beads.nbeads, 3 * (self.beads.natoms - len(self.fixatoms))])  # forces (in the nudged elastic band algorithm) from previous step on mass scaled coordinate
        self.x = np.copy(self.beads.q[:, self.fixatoms_mask])  # coordinate of free moving atoms
        self.old_x = None
        self.action = None # current action
        self.old_action = None   # action at previous step
        self.f_mscaled, self.action = self.nebgm(self.x)  # forces at current step on mass scaled coordinate


    def neb_instanton_step_info(self, outer_loop_step, neb_step, grad_max):
        '''
        output the information about convergence check for each step of neb move
        '''
        tolerances = self.options["tolerances"]

        print("\n")
        info("@Inner step summary: Outer loop # {} , inner loop # {},  max force gradient {:4.2e} , (condition {:4.2e})".format(
                outer_loop_step, neb_step,
                grad_max, tolerances["gradient"]
            ),
            verbosity.low
            )

        print("old action: " + str(self.old_action) + "  new action: " + str(self.action))
        # check the optimization gradient for LI-NEB
        print("beads optimization gradient: " + str(npnorm(self.nebgm.neb_optimization_force, axis = 1)))
        # check potential of beads.
        print("beads potential relative to instanton path energy (eV): " + str( (self.nebgm.beads_energy - self.optarrays["instanton_path_energy"]) * units.unit_to_user("energy", "electronvolt", 1)  ))
        # check distance between beads (effect of spring_k)
        print("distance between beads in mass scaled coordinate: " + str( self.nebgm.beads_mscaled_distance))
        print("\n")
        print("@Inner loop Finish: outer loop step {}, finish inner loop neb step {}".format(outer_loop_step, neb_step))
        print("\n")
        print("\n")

    def compute_initial_neb_path_length_in_scaled_internal_coordinate(self):
        '''
        compute the initial neb path length in scaled internal coordinate and use it as criterion for early stop
        '''
        # kernel output scale and kernel length scale of kernels
        kernel_output_scale = self.gpr_model.output_kernel_outputscale()
        kernel_length_scale = self.gpr_model.output_kernel_lengthscale()
        kernel_number = self.gpr_model.gpr_SE_kernel_number

        # deal with numerical noise where 1 kernel is very small & overfits the model
        kernel_output_scale_max = np.max(kernel_output_scale)
        for i in range(kernel_number):
            # in case kernel output scale for 1 kernel is too small. Effective eliminate this kernel (this kernel probably overfits the noise.)
            if kernel_output_scale[i] < np.power(10.0, -4) * kernel_output_scale_max:
                kernel_output_scale[i] = 0

        # normalize the output scale:
        output_scale_sum = np.sum(kernel_output_scale)
        kernel_output_scale_normalized = kernel_output_scale / output_scale_sum

        # effective kernel lengthscale for scaling internal coordinate. l_eff^{-2} = sum_{n} output_scale_n / (l_n)^2.   
        effective_kernel_length_scale = np.power(np.sum(kernel_output_scale_normalized[:, np.newaxis] / np.power(kernel_length_scale, 2) , axis = 0), -0.5)

        beads_internal_coordinate = self.coordinate_transformer.get_internal_coordinate_q(np.copy(self.beads.q))

        distance_in_scaled_internal_coordinate = np.linalg.norm(  (beads_internal_coordinate[1:] - beads_internal_coordinate[:-1]) / effective_kernel_length_scale , axis = 1)

        scaled_internal_coordinate_neb_path_length = np.sum(distance_in_scaled_internal_coordinate)

        self.scaled_internal_coordinate_neb_path_length = scaled_internal_coordinate_neb_path_length
        self.distance_in_scaled_internal_coordinate = distance_in_scaled_internal_coordinate

        self.initial_effective_kernel_length_scale = effective_kernel_length_scale

    def check_spring_k_kappa(self):
        '''
        check the amplitude of spring k and kappa. to see if it is appropriate. If not, update it.
        '''
        dt = self.optarrays["time_step"]
        spring_k = self.optarrays["spring_k"]
        left_kappa = self.optarrays["kappa"]["left"]
        right_kappa = self.optarrays["kappa"]["right"]

        # check spring_k * (dt)^2. It should be smaller than 0.4 and larger than 0.1 (too small spring_k will make bead hard to reach equal distance)
        # ideal value is 0.25
        val1 = spring_k * np.power(dt, 2)
        # scale spring_k, left_kappa and right_kappa
        spring_k_scale = 0.1 / val1

        # check |dV/dx| * kappa / sqrt(m_H) * (dt)^2, it should be smaller than 1 and larger than 0.1
        # ideal value is 0.5
        # check the left end bead.
        m_H = 1837 # mass of hydrogen in atomic unit.
        
        max_force2 = np.max(np.abs(self.nebgm.rbf[0]))  # maximum gradient of left end bead.
        val2 = max_force2 * np.power(dt, 2) * left_kappa / np.sqrt(m_H)
        left_kappa_scale = 0.2 / val2

        # check the right end bead.
        max_force3 = np.max(np.abs(self.nebgm.rbf[-1]))  # maximum gradient of right end bead
        val3 = max_force3 * np.power(dt,2) * right_kappa / np.sqrt(m_H)
        right_kappa_scale = 0.2 / val3

        self.optarrays["spring_k"] = self.optarrays["spring_k"] * spring_k_scale
        self.nebgm.spring_k = self.nebgm.spring_k * spring_k_scale

        self.optarrays["kappa"]["left"] = self.optarrays["kappa"]["left"] * left_kappa_scale
        self.nebgm.kappa["left"] = self.nebgm.kappa["left"] * left_kappa_scale

        self.optarrays["kappa"]["right"] = self.optarrays["kappa"]["right"] * right_kappa_scale
        self.nebgm.kappa["right"] = self.nebgm.kappa["right"] * right_kappa_scale

    def print_geometry(self, step):
        '''
        print beads geometry and beads energy.
        '''
        pots = self.nebgm.beads_energy
        if (
            self.options["alt_out_step"] > 0 and np.mod(step, self.options["alt_out_step"]) == 0
        ):
            ipi.utils.nebinstool.print_neb_instanton_geo(
                self.options["prefix"],
                step,
                self.beads.nbeads,
                self.beads.natoms,
                self.beads.names,
                self.beads.q,
                pots,
                self.cell,
                self.optarrays["energy_shift"],
                self.output_maker
            )
          
class LINEBGradientMapper(object):
    """Creation of the multi-dimensional function that will be minimized.
        Functional analog of a GradientMapper in geop.py

        Fixed atoms are excluded via boolean mask. 1 = moving, 0 = fixed.

    Attributes:
        spring_k: spring constants
        tangent: plain or improved tangents
          "plain":    J. Chem. Phys. 113, 9901 (2000); https://doi.org/10.1063/1.1329672
          "improved": J. Chem. Phys. 113, 9978 (2000); https://doi.org/10.1063/1.1323224
    """

    def __init__(self):
        self.spring_k = None    # spring constants for internal beads  
        self.kappa = None   # spring constants for beads at two ends.  

        self.init_allpots = None   #  initial potential for all beads. This potential will not be updated.
        self.action_forces = None  # minus gradient of abbreviated action 
        self.action = None    # abbreviated action. 
        self.neb_optimization_force = None  # neb force for optimization of action with constraints at two ends.
        self.neb_transverse_force = None # neb force for interior beads along transverse direction 

        self.instanton_path_energy = None # energy E of instanton path in JWKB approximation. See: Section II. A in J. Chem. Phys. 148, 102334 (2018)


    def bind(self, ens : MAPNEBGPRMover):
        '''
        :param: ens: A NEBMover instance.
        Copy beads, cell, forces of NEB mover to itself.
        '''
        # In principle, there is no need in dforces within the Mapper,
        # BUT dbeads are needed to calculate tangents for the endpoints,
        # and dforces are needed outside the Mapper to construct the "main" forces.
        self.dbeads = ens.beads.copy()
        self.dcell = ens.cell.copy()
        self.dforces = ens.forces.copy(self.dbeads, self.dcell)
        self.fixatoms = ens.fixatoms.copy()

        self.instanton_path_energy = ens.optarrays["instanton_path_energy"]   # inherit instanton path energy from NEB mover. 

        # Mask to exclude fixed atoms from 3N-arrays
        self.fixatoms_mask = np.ones(3 * ens.beads.natoms, dtype=bool)
        if len(ens.fixatoms) > 0:
            self.fixatoms_mask[3 * ens.fixatoms] = 0
            self.fixatoms_mask[3 * ens.fixatoms + 1] = 0
            self.fixatoms_mask[3 * ens.fixatoms + 2] = 0

        # Create reduced bead and force object (excluding the fixed atoms. But including the beads at two ends that move)
        self.rbeads = Beads(ens.beads.natoms, ens.beads.nbeads)
        self.rbeads.q[:] = ens.beads.q[:]
        self.rforces = ens.forces.copy(self.rbeads, self.dcell)

        self.spring_k = ens.optarrays["spring_k"] # bind spring force spring_k from NEBMover.
        self.kappa = ens.optarrays["kappa"] # bind energy constraint force kappa from NEBMover. 

        self.energy_shift = ens.optarrays["energy_shift"]

        # bind the gpr model
        self.gpr_model = ens.gpr_model
        self.coordinate_transformer = ens.coordinate_transformer
        self.ab_initio_pot = np.zeros([self.dbeads.nbeads])
        self.ab_initio_force = np.zeros([self.dbeads.nbeads, 3 * self.dbeads.natoms])

    def compute_tangent_vector(self, nimage, natom, mscaled_q):
        '''
        :param: nimage: number of replica images
        :param: natom: number of atoms (free moving)
        :param: bq: beads coordinate (mass_scaled coordinate)

        :return: btau: unit director for tangent vector of all internal beads in mass_scaled coordinates. (We do not need tangent vector for beads at two ends.)
        '''
        beads_energy = self.beads_energy
        btau = np.zeros((nimage, 3 * natom), float)  # tangent direction.
        
        for ii in range(1, nimage - 1):
            d1 = mscaled_q[ii] - mscaled_q[ii - 1]  # tau minus
            d2 = mscaled_q[ii + 1] - mscaled_q[ii]  # tau plus

            # Improved tangent estimate
            # J. Chem. Phys. 113, 9978 (2000) https://doi.org/10.1063/1.1323224

            # Energy of images: (ii+1) < (ii) < (ii-1)
            if beads_energy[ii + 1] < beads_energy[ii] < beads_energy[ii - 1]:
                btau[ii] = d1
            # Energy of images (ii-1) < (ii) < (ii+1)
            elif beads_energy[ii - 1] <= beads_energy[ii] <= beads_energy[ii + 1]:
                btau[ii] = d2
            # Energy of image (ii) is a minimum or maximum
            else:
                maxpot = max(abs(beads_energy[ii + 1] - beads_energy[ii]), abs(beads_energy[ii - 1] - beads_energy[ii]))
                minpot = min(abs(beads_energy[ii + 1] - beads_energy[ii]), abs(beads_energy[ii - 1] - beads_energy[ii]))

                if beads_energy[ii + 1] >= beads_energy[ii - 1]:
                    btau[ii] = d2 * maxpot + d1 * minpot
                elif beads_energy[ii + 1] < beads_energy[ii - 1]:
                    btau[ii] = d2 * minpot + d1 * maxpot
            btau[ii] /= npnorm(btau[ii])

        return btau

    
    def compute_neb_action(self, nimage, mscaled_q):
        '''
        compute abbreviated action W. See eq.(10) in J. Chem. Phys. 148, 102334 (2018)
        Note in atomic unit, hbar = kb = 1. 
        
        :param: nimage: number of images (replicas)
        :param: mscaled_q: mass weighted coordinates for free moving atoms [nimag, 3 * natom]

        :return: action: abbreviated action of the ring polymer path
        '''
        beads_energy = self.beads_energy

        action = 0

        # sqrt(2 (V - E))
        action_each_bead = np.zeros([nimage])
        for i in range(nimage):
            if beads_energy[i] < self.instanton_path_energy:
                action_each_bead[i] = 0
            else:
                action_each_bead[i] = np.sqrt(2 * (beads_energy[i] - self.instanton_path_energy))

        for j in range(1 , nimage):
            rj = mscaled_q[j]
            rj_1 = mscaled_q[j - 1]
            r_dist = npnorm(rj - rj_1)  
            action = action + 1/2 * (action_each_bead[j] + action_each_bead[j-1]) * r_dist 
        
        return action 
    
    def compute_neb_action_force(self, nimage, natom, mscaled_q, mscaled_f):
        '''
        compute the negative gradient of abbreviated action W. (for scaled coordinates.) See eq. (11) in J. Chem. Phys. 148, 102334 (2018).
        Note I will use the same symbol as given in the eq.(11) in the paper.

        :param: nimag: number of images (replica). scalar 
        :param: natom: number of freely moving atoms. scalar 
        :param: mscaled_q: mass weighted coordinates for free moving atoms. size: [nimag, 3 * natom]
        :param: mscaled_f: mass scaled forces for all beads. size: [nimag, 3 * natom]

        :return: action_force:  the negative gradient of abbreviated action W. (for scaled coordinates) size: [nimag, 3 * natom].
        '''
        beads_energy = self.beads_energy

        bead_displs_vector = mscaled_q[1:] - mscaled_q[:-1]  # displacement vector of beads. [nbeads-1, 3 * natom]

        bead_distance = npnorm( bead_displs_vector , axis = 1)  # |r_j - r_{j-1}|  [nbeads -1]

        bead_displs_unit_vector = np.transpose(np.transpose(bead_displs_vector) / bead_distance) # unit vector for beads displacement vector [nbeads -1, 3* natom] 

        # sqrt(2 (V - E))
        action_each_bead = np.zeros([nimage])
        for i in range(nimage):
            if beads_energy[i] < self.instanton_path_energy:
                action_each_bead[i] = 0
            else:
                action_each_bead[i] = np.sqrt(2 * (beads_energy[i] - self.instanton_path_energy))
        
        action_force = np.zeros([nimage, 3 * natom])
        for j in range(1 , nimage-1):
            dj1 = bead_distance[j-1]  #|r_{j} - r_{j-1}|.  d_{j}
            dj2 = bead_distance[j] # |r_{j+1} - r_{j}|. d_{j+1}
            dj1_unit_vector = bead_displs_unit_vector[j-1] # \hat{d}_{j}
            dj2_unit_vector = bead_displs_unit_vector[j]  # \hat{d}_{j+1}
            fj = mscaled_f[j]
            
            # for pathological case that inner beads' energy is lower than the required end bead energy.
            if action_each_bead[j] > 0:
                gj_force_component = 0.5 * ( 1/action_each_bead[j] * (dj1 + dj2) * fj )
            else:
                gj_force_component = 0

            gj_curvature_component = 0.5 * (- (action_each_bead[j] + action_each_bead[j-1]) * dj1_unit_vector + (action_each_bead[j] + action_each_bead[j+1]) * dj2_unit_vector)

            gj = gj_force_component + gj_curvature_component

            action_force[j] = gj 

        return action_force 
    
    def compute_force_tangent_vector_inner_product(self, mscaled_f, btau, nimage):
        '''
        compute inner product of unit vector of f (force) and btau (tangent vector).
        For the converged calculation, two unit vector should be almost aligned with each other
        :param: mscaled_f: force in mass scaled coordinate
        :param: btau: tangent vector for inner beads
        :param: nimage: number of replica
        '''
        force_norm = npnorm(mscaled_f, axis = 1)
        mscaled_f_unit_vector = np.transpose(np.transpose(mscaled_f) / force_norm)

        f_tau_inner_product = np.zeros([nimage])
        for i in range(1, nimage - 1):
            inner_product = np.inner(mscaled_f_unit_vector[i], btau[i])
            f_tau_inner_product[i] = inner_product 
        
        return f_tau_inner_product

    def compute_neb_optimization_force(self, nimage, natom, btau, mscaled_q,  mscaled_f):
        '''
        compute the optimization forces for nudged elastic band beads. See eq.(15 - 22) in J. Chem. Phys. 148, 102334 (2018).

        :param: nimag: number of images (replica). scalar 
        :param: natom: number of freely moving atoms. scalar 
        :param: btau: tangent vector for internal beads.  size: [nimag, 3 * natoms]
        :param: mscaled_q: mass weighted coordinates for free moving atoms. size: [nimag, 3 * natom]
        :param: mscaled_f: mass scaled forces for all beads. size: [nimag, 3 * natom]

        :return: optimization_force: the optimization force for nudged elastic band. size: [nimag, 3 * natom]
        '''
        beads_energy = self.beads_energy

        left_kappa = self.kappa["left"]   # kappa: restraint force back to iso-energy contour. kappa on the left side
        right_kappa = self.kappa["right"] # kappa on the rigtht side
        spring_k = self.spring_k    # spring_k: spring force between beads.

        neb_optimization_force = np.zeros([nimage, 3 * natom])
        self.neb_transverse_force = np.zeros([nimage, 3* natom])
        # spring forces for beads. Note the spring force at two ends are different from spring forces for internal beads.
        spring_force = np.zeros([nimage, 3 * natom])
        # spring force for internal beads
        for ii in range(1, nimage - 1):
            spring_force[ii] = (npnorm(mscaled_q[ii+1] - mscaled_q[ii]) - npnorm(mscaled_q[ii] - mscaled_q[ii-1])) * spring_k * btau[ii]
        
        # spring force for end bead 0
        unit_vec_1 = (mscaled_q[1] - mscaled_q[0]) / npnorm(mscaled_q[1] - mscaled_q[0])  # unit vector for q[1] - q[0]
        spring_force_bead0 = spring_k * (npnorm(mscaled_q[1] - mscaled_q[0]) - npnorm(mscaled_q[2] - mscaled_q[1])) * unit_vec_1  
        f0 = mscaled_f[0] / npnorm(mscaled_f[0])   # unit vector along force at beads: 0
        spring_force[0] = spring_force_bead0 - np.dot(spring_force_bead0 , f0) * f0  # spring force transverse to gradient.

        # spring force for end bead nimag - 1
        unit_vec_2 = (mscaled_q[nimage - 2] - mscaled_q[nimage - 1]) / npnorm(mscaled_q[nimage - 2] - mscaled_q[nimage - 1])
        spring_force_bead1 = spring_k * ( npnorm(mscaled_q[nimage - 2] - mscaled_q[nimage - 1]) - npnorm(mscaled_q[nimage - 3] - mscaled_q[nimage -2]) ) * unit_vec_2 
        f1 = mscaled_f[nimage - 1] / npnorm(mscaled_f[nimage - 1])  # unit vector along force at beads: nimage - 1 
        spring_force[nimage - 1] = spring_force_bead1 - np.dot(spring_force_bead1 , f1) * f1  # spring force transverse to gradient.

        # end_beads_energy_constraint_force: force to draw end beads back to isoenergy contours.
        end_beads_energy_constraint_force = np.zeros([2, 3 * natom])
        end_beads_energy_constraint_force[0] = mscaled_f[0] / npnorm(mscaled_f[0]) * left_kappa * (beads_energy[0] - self.instanton_path_energy)  # kappa * (V(r) - E) * \hat{f}(r) for beads 0
        end_beads_energy_constraint_force[1] = mscaled_f[nimage - 1] / npnorm(mscaled_f[nimage - 1]) * right_kappa * (beads_energy[nimage -1] - self.instanton_path_energy)  # kappa * (V(r) - E) * \hat{f}(r) for beads n-1.

        self.spring_forces = spring_force   # store the spring force between beads
        self.end_bead_energy_constraint_forces = end_beads_energy_constraint_force  # store energy constraint force for end beads.

        # for internal beads, transverse force from negative gradient of action.
        for ii in range(1, nimage - 1):
            neb_optimization_force[ii] = self.action_forces[ii] - np.dot(self.action_forces[ii] , btau[ii]) * btau[ii]
        
        self.neb_transverse_force = neb_optimization_force  # transverse gradient for interior neb beads.

        neb_optimization_force[0] = end_beads_energy_constraint_force[0]
        neb_optimization_force[nimage - 1] = end_beads_energy_constraint_force[1]

        neb_optimization_force = neb_optimization_force + spring_force 

        return neb_optimization_force 
    

    def get_bead_potential_and_forces(self):
        '''
        Get potential and forces for all rbeads.
        Using Gaussian Process Regression model.
        return: potential for all beads.
        '''
        test_x = np.copy(self.rbeads.q)
        beads_potential_shift, beads_potential_grad_x, var_V, var_grad_q = self.gpr_model.predict_latent_function(test_x)

        beads_forces = - beads_potential_grad_x
        # the predicted potential is the one relative to the energy shift.
        beads_potential = beads_potential_shift + self.energy_shift

        # check if ab_initio potential and forces are available. If so, use it and then reset it to zero.
        for i in range(self.dbeads.nbeads):
            if self.ab_initio_pot[i] != 0:
                beads_potential[i] = self.ab_initio_pot[i]
                self.ab_initio_pot[i] = 0 
            if np.linalg.norm(self.ab_initio_force[i]) != 0:
                beads_forces[i] = self.ab_initio_force[i]
                self.ab_initio_force[i] = np.zeros([3 * self.dbeads.natoms])

        return beads_potential, beads_forces 
        

    def __call__(self, x):
        """Returns the potential for all beads and the gradient.
        update reduced bead coordinates (&dbeads coordinate) (sticly speaking the free-moving atom parts) with x.
        :param: x = q[:, self.fixatoms_mask] : new coordinates for updated freely moving particles.

        rbf: physical forces for reduced beads
        rbq: position for reduced beads
        btau: tangent vector directions.
        """

        # Bead positions
        # Touch positions only if they have changed (to avoid triggering forces)
        # I need both dbeads and rbeads because of the endpoint tangents.
        if (self.rbeads.q[:, self.fixatoms_mask] != x).any():
            self.rbeads.q[:, self.fixatoms_mask] = x
        rbq = np.copy(self.rbeads.q[:, self.fixatoms_mask])
        
        mscaled_q = rbq * np.sqrt( self.dbeads.m3[:, self.fixatoms_mask] )  # mass scaled coordinates.
        self.mscaled_q = mscaled_q

        # use Gaussian Process Regression to get the potential and forces for beads.
        beads_potential, beads_forces = self.get_bead_potential_and_forces()
        # energy for reudced beads. All potential energy of beads are needed.
        self.beads_energy = np.copy(beads_potential) # beads energy.  

        # Forces for reduced beads
        self.rbf = beads_forces.copy()[:, self.fixatoms_mask]

        # mass weighted force
        mscaled_f = self.rbf / np.sqrt( self.dbeads.m3[: , self.fixatoms_mask] )  # 1/sqrt(m) * f: mass scaled force.
        self.mscaled_f = mscaled_f

        # Number of images
        nimage = self.dbeads.nbeads
        # Number of atoms that is free to move.
        natom = self.dbeads.natoms - len(self.fixatoms)

        self.spring_forces = np.zeros([nimage, 3 * natom])
        self.end_bead_energy_constraint_forces = np.zeros([2, 3 * natom ])
        self.beads_mscaled_distance = npnorm(mscaled_q[1:] - mscaled_q[:-1] , axis = 1)

        # abbreviated action for the ring polymer instanton path.
        self.action = self.compute_neb_action(nimage, mscaled_q)
        # negative gradient of abbreviated action for each bead. We only compute it for the internal beads (excluding two ends)
        self.action_forces = self.compute_neb_action_force(nimage, natom, mscaled_q, mscaled_f)

        # compute direction of tangent vector, using either improved methods.
        btau = self.compute_tangent_vector(nimage, natom, mscaled_q)

        # compute inner product for mass scaled force and tangent vector btau
        self.f_tau_inner_product = self.compute_force_tangent_vector_inner_product( mscaled_f, btau, nimage)

        # evaluate the nudged elastic band forces for perpendicular action forces and the spring force. (on mass scaled coordinate for free moving atoms.)
        neb_optimization_force = self.compute_neb_optimization_force(nimage, natom, btau, mscaled_q, mscaled_f)

        self.neb_optimization_force = np.copy(neb_optimization_force)

        return neb_optimization_force, self.action 
    




    


