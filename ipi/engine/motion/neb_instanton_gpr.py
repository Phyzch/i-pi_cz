"""Holds the algorithms to perform the Line Integral Nudged Elastic Band (NEB) method to find the instanton path.
J. Chem. Phys. 148, 102334 (2018); https://doi.org/10.1063/1.5007180

The LI-NEB calculation is accelerated by Gaussian Process Regression method. See: J. Chem. Phys. 147, 152720 (2017) and Faraday Discuss., 2018,212, 237-258 (https://doi.org/10.1039/C8FD00085A)

The algorithm is first implemented by Chenghao Zhang, 2023. Adapted from neb module & instanton module in i-pi package.
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
    Accelerated by Gaussian Process Regression (GPR) J. Chem. Phys. 147, 152720 (2017) & Faraday Discuss., 2018,212, 237-258 (https://doi.org/10.1039/C8FD00085A)
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
        gpr_force_criterion = 0.05,
        gpr_trust_region_ratio = 0.03,
        gpr_kernel_outputscale = np.zeros(0, float),
        gpr_kernel_lengthscale_ratio = np.zeros(0, float),
        gpr_noise_std = {"pot_noise_prior": 1e-6, "force_noise_prior": 1e-4},
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
        if np.shape(gpr_kernel_outputscale) == (0,):
            raise("You must provide output scale for covariance function. This should be a numpy array, with size equal to number of Squared Exponential (SE) kernel you use.")
        if np.shape(gpr_kernel_lengthscale_ratio) == (0,):
            raise("You must provide length scale for covariance function. This should be a numpy array, with size equal to number of Squared Exponential (SE) kernel you use.")

        assert len(gpr_kernel_lengthscale_ratio) == gpr_SE_kernel_number, "The number of length scale of kernels should match the number of Squared Exponential kernel you use"
        assert len(gpr_kernel_outputscale) == gpr_SE_kernel_number, "The number of output scale of kernels should match the number of Squared Exponential kernel you use."

        self.optarrays["gpr_force_criterion"] = gpr_force_criterion  # criterion to stop the outer loop.
        self.optarrays["gpr_trust_region_ratio"] = gpr_trust_region_ratio  # criterion to early stop the LI-NEB on PES generated by GPR.
        self.optarrays["gpr_kernel_outputscale"] = gpr_kernel_outputscale  # output scale of the kernel
        self.optarrays["gpr_kernel_lengthscale_ratio"] = gpr_kernel_lengthscale_ratio  # outpu
        self.optarrays["gpr_noise_std"] = gpr_noise_std
        self.options["gpr_SE_kernel_number"]  = gpr_SE_kernel_number

        # index list storing the bead index whose ab-initio forces are close to their gpr predicted forces.
        self.ab_initio_index_list = []
        # |df|/|f_{ab initio}| for bead in index list.
        self.force_diff_ratio_list = []
        self.ab_initio_force_amplitude_list = []
        self.gpr_force_prediction_amplitude_list = []

        self.coordinate_transformer = None # coordinate transformer between the Cartesian coordinate and the internal coordinate 
        self.gpr_model = None  # Gaussian Process Regression model instance.

        self.ab_initio_bead_calculation_number = 0  # record the number of ab initio calculation on beads. 

        self.internal_coordinate_closest_r_list = [] # measure the distance of beads from the training data in internal coordinate.
        self.trust_region_distance_cutoff = 0  # the distance cutoff for the trust region in internal coordinate system.

        self.start_time = timer()  # used to record the time for the calculation.

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
        # We use One Image evaluation method, each time only update bead for one image.
        self.gpr_beads = Beads(self.beads.natoms, 1)
        self.gpr_forces = self.forces.copy(self.gpr_beads, self.cell)

        self.nebgm.bind(self)
    
    def generate_initial_training_data(self):
        '''
        generate training data for Gaussian Process Regression model
        '''
        # choose all NEB beads as initial training data.
        # We will train the GPR model to optimize hyperparameter using the initial data.
        train_x = np.copy(self.beads.q)
        # potential energy has to shift relative to the energy_shift for training.
        train_V = np.copy(self.forces.pots) - self.optarrays["energy_shift"]
        train_grad = - np.copy(dstrip(self.forces.f))
        # count the # of ab-initio calculation we have done.
        self.ab_initio_bead_calculation_number = self.ab_initio_bead_calculation_number + self.beads.nbeads
        
        # store the initial training_data: potential V and forces f.
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
        self.ab_initio_bead_calculation_number = self.ab_initio_bead_calculation_number + ab_initio_calculation_number 

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
        # choose the point with the highest potential in the initial instanton path as reference point.
        nbeads = self.beads.nbeads
        beads_pots = np.copy(self.forces.pots)
        bead_index_at_transition_state = np.argmax(beads_pots)
        ref_x = dstrip(self.beads.q[bead_index_at_transition_state]).copy()

        # create coordinate_transformer, which handles the transformation from the Cartesian coordinate to internal coordinate.
        self.coordinate_transformer = non_redundant_coordinate_transformer(self.beads.natoms, ref_x)

        # attach ab_initio potential to self.nebgm.ab_initio_pot and self.nebgm.ab_initio_force.
        # In the LI-NEB algorithm, when there is ab-initio potential & force data available, we will use that potential and force. 
        # If the ab-initio data point is not available, we use the potential and force generated by Gaussian Process Regression (GPR)
        self.nebgm.ab_initio_pot = np.copy(self.forces.pots)
        self.nebgm.ab_initio_force = np.copy(dstrip(self.forces.f))

        # for the training data, we have the option to read it from .txt file or generate it using the current geometry. 
        # this provides the flexibility for choosing the training data for the initial model. 
        read_gpr_training_data_bool = self.options["read_initial_gpr_training_data"]
        if not read_gpr_training_data_bool:
            train_x, train_V, train_grad = self.generate_initial_training_data()
        else:
            train_x, train_V, train_grad = self.read_initial_training_data()

        self.gpr_model = ipi.utils.gprtools.GPModelWithDerivativesWrapper(train_x, train_V, train_grad,
                                                                     self.beads.natoms, self.coordinate_transformer,
                                                                     gpr_SE_kernel_number= self.options["gpr_SE_kernel_number"],
                                                                     kernel_outputscale= self.optarrays["gpr_kernel_outputscale"],
                                                                     kernel_lengthscale_ratio= self.optarrays["gpr_kernel_lengthscale_ratio"],
                                                                     noise_std= self.optarrays["gpr_noise_std"])

    def check_initial_training_result(self):
        '''
        check whether the training of GPR model is successful. If not, stop the simulation and report error
        '''
        # first check the prediction of the training data. See if there is under-fitting.
        predicted_V_shift, predicted_grad, _, _ = self.gpr_model.predict_latent_function(self.beads.q) 

        predicted_forces = - predicted_grad 

        ab_initio_V_shift = self.forces.pots - self.optarrays["energy_shift"]
        ab_initio_forces = self.forces.f 

        # check length scale for possible over-fitting
        learned_kernel_length_scale = self.gpr_model.output_kernel_lengthscale()
        internal_input_range = np.max(self.gpr_model.output_training_internal_inputs(), axis=0) - np.min(self.gpr_model.output_training_internal_inputs(), axis = 0)
        scaled_kernel_lengthscale = learned_kernel_length_scale / internal_input_range 

        # check the size of covariance function (kernel). 
        kernel_output_scale_var = self.gpr_model.output_kernel_outputscale()
        kernel_output_scale_std = np.sqrt(kernel_output_scale_var)

        print("\n")
        print("@initial gpr training info: check the overfitting and underfitting of kernel length scale")
        for i in range(self.gpr_model.gpr_SE_kernel_number):
            print("kernel {}: ".format(i))
            print("square root of kernel output scale (\u03C3): " + str(kernel_output_scale_std[i]))
            print("kernel_length_scale / input scale:   " + str(scaled_kernel_lengthscale[i])  )
        print("\n")

        # check the force noise and potential noise. We can see for force noise of certain internal coordinate, it is quite large.
        force_range = self.gpr_model.output_normalized_force_range()
        V_noises, force_noises = self.gpr_model.output_fitted_gpr_model_noises()
        force_noises_ratio = force_noises / force_range
        print("potential noise amplitude: " + str(V_noises))
        print("force noise ratio  (amplitude / range): " + str(force_noises_ratio))
        print("internal coordinate force range: " + str(force_range))

        # check the difference between ab-initio potential V and the predicted potential V:
        V_error = np.abs(ab_initio_V_shift - predicted_V_shift) / np.abs(ab_initio_V_shift)

        # check the difference between ab-initio force f and the predicted force f.
        df = np.linalg.norm(ab_initio_forces - predicted_forces , axis = 1)
        ab_initio_force_amplitude = np.linalg.norm(ab_initio_forces, axis = 1)
        df_error = df / ab_initio_force_amplitude
        
        print("\n")
        print("@initial Gaussian Process Regression fitting:")
        print("error of potential prediction: " + str(V_error))
        print("error of force prediction: " + str(df_error))
        print("\n")

        # check overfitting on the unseen test data to test over-fitting. 
        print("@initial gpr training info: Test Overfitting of GPR model.")

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

        self.ab_initio_bead_calculation_number = self.ab_initio_bead_calculation_number + 4 
        pass 

    def step(self, step=None):
        """
        Does one simulation time step.
        if stage = 'neb', we will do LI-NEB with Gaussian Process Regression.
        if stage = 'instanton', we will evolve instanton beads along the path. 
        if stage = 'converged', we will stop the simulation. 
        """
        print(" @NEB Outerloop STEP %d, stage: %s" % (step, self.options["stage"]))
        
        if step == 0:
            # print initial geometry and energy of neb path.
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

            # The instanton path energy is defined relative to the energy shift.
            # We perform the transformation only when we start the initial calculation. Not for restarting the calculation.
            self.optarrays["instanton_path_energy"] = self.optarrays["instanton_path_energy"] + self.optarrays["energy_shift"]  # shift the instanton path energy according to energy shift.
            self.nebgm.instanton_path_energy = self.optarrays["instanton_path_energy"]
            
            # TODO: assign instanton path energy also for RP_MAP object.
        
        if self.coordinate_transformer == None:
            # initialize Gaussian Process Regression(GPR) model and coordiante transformer
            self.initialialize_GPR_model()
            # check the training result on the test data which is unseen by GPR.
            self.check_initial_training_result()

            # bind the gpr model and coordinate_transformer to the LINEGradientMapper class
            # the LINEBGradientMapper will perform LI-NEB using gpr generated potential and force.
            self.nebgm.gpr_model = self.gpr_model 
            self.nebgm.coordinate_transformer = self.coordinate_transformer

        # Check if we restarted a converged calculation or the calculation converged.
        if self.options["stage"] == "converged":
            # output number of ab-initio calculation.
            ipi.utils.nebinstgprtool.print_ab_initio_calculation_number(self.ab_initio_bead_calculation_number, self.output_maker, step)
            print("ab initio calculation number : " + str(self.ab_initio_bead_calculation_number))

            # output the time for execuation 
            self.end_time = timer()
            time_elapsed = (self.end_time - self.start_time) / 60  # time elapsed in minutes 
            print("the running time for the program: " + str(time_elapsed) + " min.")

            softexit.trigger(
                status="success",
                message="neb calculation converged. Instanton geometry calculation finishes. Exiting simulation",
            )

        if self.options["stage"] == "neb":
            # use nudged elastic band method to find minmum action path.
            # then we will switch to the stage "instanton"
            # perform LI-NEB algorithm on the surrogated PES generated by GPR. stop either LI-NEB converge or one bead move out of the trust region.
            early_stop_bool, outrange_bead_index_list = self.neb_loop(step)

            # update Gaussian Process Regression model with new training data
            self.update_GPR_model(early_stop_bool, outrange_bead_index_list, step)



  

    def neb_loop(self, outer_loop_step):
        '''
        the inner loop of Line Integral Nudged Elastic Band method.
        The loop will stop once one of the two criteria is met:
        (1) The LI-NEB algorithm converge on the surrogated PES generated by Gaussian Process Regression model. 
            This is the case when all the gradient of LI-NEB beads are smaller than the tolerance value.
        (2) One LI-NEB bead move out of the trust region. In this case, PES generated by GPR is not reliable any more, 
            we need to early stop the algorithm and compute the ab-initio V & F at that given bead & add to the training data.
            The trust region is defined in the internal coordinate, scaled by the length scale of the squared exponential kernel.
        '''
        grad_max = 1000
        tolerances = self.options["tolerances"]  # tolerances for converging the LI-NEB calculation.
        
        neb_step = 0  # count the step number of neb move. (inner loop)
        
        early_stop_bool = False
        outrange_bead_index_list = [] # index for beads that move out of trusted region that causes the early stop.

        self.neb_loop_initialize(outer_loop_step) 
        
        # print geometry when outer_loop_step % alt = 0. for record.
        self.print_geometry(outer_loop_step)

        print("\n")
        print("@Start outer loop: " + str(outer_loop_step) + "\n")
        while grad_max > tolerances["gradient"]:
            grad_max, early_stop_bool, outrange_bead_index_list = self.neb_step(outer_loop_step, neb_step)
            neb_step = neb_step + 1

            # beads move out of trust region.
            if early_stop_bool:
                break 
        
        if not early_stop_bool:
            print("@LI-NEB converge on GPR PES.")

        return early_stop_bool, outrange_bead_index_list


    def neb_loop_initialize(self, step):
        '''
        Initialize the action, force, velocity for nudged elastic band calculation. (inner loop calculation.)
        Each time we restart the neb-loop, the PES generated by GPR has changed. Therefore, we should restart the LI-NEB algorithm.
        '''
        info(
            " @NEB: start inner loop neb for step {}".format(step),
            verbosity.debug,
        )
        # velocity of free moving particles on mass scaled coordinate.
        self.velocity_mscaled = np.zeros([self.beads.nbeads, 3 * (self.beads.natoms - len(self.fixatoms))])  
        
        # coordinate of free moving atoms
        self.x = np.copy(self.beads.q[:, self.fixatoms_mask])  
        self.old_x = None

        # action of LI-NEB beads
        self.action = None 
        self.old_action = None  
        
        # negative gradient of LI-NEB action for each bead on mass scaled coordinate
        self.old_f_mscaled = np.zeros([self.beads.nbeads, 3 * (self.beads.natoms - len(self.fixatoms))])  

        self.f_mscaled, self.action = self.nebgm(self.x)  

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

        # print("old action: " + str(self.old_action) + "  new action: " + str(self.action))
        # check the optimization gradient for LI-NEB
        print("beads optimization gradient: " + str(npnorm(self.nebgm.neb_optimization_force, axis = 1)))
        # check the potential of beads.
        beads_energy_relative_to_instanton_energy= (self.nebgm.beads_energy - self.optarrays["instanton_path_energy"]) * units.unit_to_user("energy", "electronvolt", 1)
        print("beads potential relative to instanton path energy (eV): " + str(beads_energy_relative_to_instanton_energy ))
        # check distance between beads (effect of spring_k)
        print("distance between beads in mass scaled coordinate: " + str( self.nebgm.beads_mscaled_distance))
        print("\n")
        print("@Finish Inner loop: outer loop step {}, LI-NEB inner loop step {}".format(outer_loop_step, neb_step))
        print("\n")
        print("\n")


    def neb_step(self, outer_loop_step, neb_step):
        '''
        LI-NEB move for one step.
        '''
        n_activedim = self.beads.q[0].size - len(self.fixatoms) * 3
        nbeads = self.beads.nbeads
        dt = self.optarrays["time_step"]

        # scale the spring_k term in LI-NEB relative to time step.
        # scale the kappa (energy constraint term for two end beads) in LI-NEB relative to the force at end beads. Using stability criterion.
        self.check_spring_k_kappa()

        grad_max = 0
        # check early stop condition if there are beads out of trust region.
        # the trust region is defined in the internal coordinate, scaled by length scale of the kernel.
        early_stop_bool, outrange_bead_index_list, self.internal_coordinate_closest_r_list, self.trust_region_distance_cutoff = ipi.utils.nebinstgprtool.check_neb_early_stop(self.beads.q,
                                                                                            self.optarrays["gpr_trust_region_ratio"],
                                                                                            self.gpr_model, 
                                                                                            outer_loop_step, neb_step
                                                                                            )
        
        # stop the step early if there are beads out of trust region.
        if early_stop_bool:
            return grad_max, early_stop_bool, outrange_bead_index_list

        # neb move using gradient of LINEBGradient
        # use projected verlet algorithm.
        if self.options["mode"] == "verlet":
            dx_mscaled = dt * self.velocity_mscaled + 0.5 * self.f_mscaled * np.power(dt, 2)
            dx = dx_mscaled / np.sqrt(self.beads.m3[:, self.fixatoms_mask])

            # update position
            self.old_x = np.copy(self.x)
            self.x = self.x + dx
            self.beads.q[:, self.fixatoms_mask] = self.x

            self.old_f_mscaled = np.copy(self.f_mscaled) # record old force
            self.old_action = self.action
            # evaluate the force & action using the updated position. LI-NEB algorithm. 
            self.f_mscaled, self.action = self.nebgm(self.x)  

            self.velocity_mscaled = self.velocity_mscaled + dt * (self.old_f_mscaled + self.f_mscaled) / 2

            # project the velocity along the direction of the current force
            f_unit_vector = self.f_mscaled / np.linalg.norm(self.f_mscaled)

            v_f_inner_product = np.inner( f_unit_vector.flatten() , self.velocity_mscaled.flatten() )

            if v_f_inner_product < 0:
                self.velocity_mscaled = np.zeros([self.beads.nbeads, 3 * (self.beads.natoms - len(self.fixatoms))])
            else:
                self.velocity_mscaled = v_f_inner_product * f_unit_vector


        else:
            softexit.trigger(
                status="bad",
                message="Only projected velocity verlet is currently implemented. set mode == 'verlet' ",
            )

        # compute maximum LI-NEB gradient among all beads. used for monitoring the convergence of LI-NEB.
        grad_max = np.amax(npnorm(self.nebgm.neb_optimization_force, axis = 1))

        # output info about neb calculation.
        self.neb_instanton_step_info(outer_loop_step, neb_step, grad_max)
        
        return grad_max, early_stop_bool, outrange_bead_index_list   


    def update_GPR_model_one_bead_subroutine(self, training_x, bead_index_for_update, training_bead_forces):
        '''
        compute ab initio potential for the bead of interest, add it into the training data.
        Also compute difference between ab initio force and force predicted by GPR. Check the convergence of the GPR model.
        '''
        # consistency check : the gpr_beads we claims to do the simulation should have same bead number of training data.
        assert self.gpr_beads.nbeads == len(training_x)
        self.gpr_beads.q[:] = training_x 

        # get energy and forces (in Cartesian coordinate) from force engine. ab initio calculation.
        ab_initio_beads_energy = dstrip(self.gpr_forces.pots).copy() 
        ab_initio_beads_forces = dstrip(self.gpr_forces.f).copy() 
        ab_initio_beads_grad = - ab_initio_beads_forces

        # update GPR model with coordinate (training_x), potential (beads_energy) and forces in cartesian coordiante (beads_forces)
        ab_initio_beads_energy_shift = ab_initio_beads_energy - self.optarrays["energy_shift"]
        self.gpr_model.update_model_with_new_data(training_x, ab_initio_beads_energy_shift, ab_initio_beads_grad)
        self.nebgm.gpr_model = self.gpr_model

        # set ab_initio pot and force in nebgm.
        self.nebgm.ab_initio_pot[bead_index_for_update] = ab_initio_beads_energy 
        self.nebgm.ab_initio_force[bead_index_for_update] = ab_initio_beads_forces
        # count the # of ab-initio calculation we have done.
        self.ab_initio_bead_calculation_number = self.ab_initio_bead_calculation_number + 1

        # compute the difference between ab initio force and gpr force.
        ab_initio_force_amplitude = np.linalg.norm(ab_initio_beads_forces[0])
        force_diff = training_bead_forces - ab_initio_beads_forces[0]
        force_diff_ratio = np.linalg.norm(force_diff) / ab_initio_force_amplitude

        # add |df|/|f| , f and f^{GPR} into the list.
        self.force_diff_ratio_list.append(force_diff_ratio)
        self.ab_initio_force_amplitude_list.append(ab_initio_force_amplitude)
        self.gpr_force_prediction_amplitude_list.append(np.linalg.norm(training_bead_forces))

        return force_diff_ratio 

    def update_GPR_model(self, early_stop_bool, outrange_bead_index_list, step):
        '''
        update GPR model with new training data. Which new training data we will add depends on the stop criterion.
        evaluate potential and force of one bead. 
        Then update the Gassian Process Regression model.
        '''
        bead_index_for_update = -1
        if early_stop_bool:
            # in this case, several beads have moved out of trust region. We add this bead into the training data.
            for outrange_bead_index in outrange_bead_index_list:
                bead_index_for_update = outrange_bead_index
                training_x = np.array([dstrip(self.beads.q[bead_index_for_update]).copy()])
                
                # evaluate the gpr predicted V & f. For comparison with ab-initio V & F. 
                _, training_grad_x, _, _ = self.gpr_model.predict_latent_function(training_x)
                training_bead_forces = - training_grad_x[0]

                # compute ab initio force for the bead and add it to the GPR model. 
                # evaluate the difference between ab initio force and force predicted by GPR.
                force_diff_ratio = self.update_GPR_model_one_bead_subroutine(training_x, bead_index_for_update,
                                                                            training_bead_forces)
                

            # deal with the case that the trust region distance cutoff could be too large.
            # in this case, we have to decrease the trust region distance.
            outrange_bead_distance_list = self.internal_coordinate_closest_r_list[outrange_bead_index_list]
            
            distance_cutoff = self.trust_region_distance_cutoff * 1.5
            force_error_ratio_cutoff = 0.1

            for i in range(len(outrange_bead_index_list)):
                outrange_bead_distance = outrange_bead_distance_list[i]
                bead_force_diff_ratio = self.force_diff_ratio_list[i]
                # we have to make sure when the bead is close to the trust region, the error of gpr force prediction is small.
                if outrange_bead_distance < distance_cutoff and bead_force_diff_ratio > force_error_ratio_cutoff:
                    self.optarrays["gpr_trust_region_ratio"] = self.optarrays["gpr_trust_region_ratio"] / 2
                    print("@READJUST TRUST REGION: the trust region ratio now: " + str(self.optarrays["gpr_trust_region_ratio"]))
                    break 

        else:
            # in this case, NEB calculation converges on GPR fitted PES.
            # find the bead with the largest force uncertainty: sum of force variance along different dimensions.
            _, beads_grad_x, _, beads_var_x_trace = self.gpr_model.predict_latent_function(self.beads.q)
            beads_forces = - beads_grad_x 

            bead_index_for_update = np.argmax(beads_var_x_trace)
            training_x = np.array([dstrip(self.beads.q[bead_index_for_update]).copy()])
            training_bead_forces = beads_forces[bead_index_for_update]

            # compute ab initio force for the bead and add it to the GPR model. 
            # evaluate the difference between ab initio force and force predicted by GPR.
            force_diff_ratio = self.update_GPR_model_one_bead_subroutine(training_x, bead_index_for_update,
                                                                        training_bead_forces)

            if force_diff_ratio < self.optarrays["gpr_force_criterion"]:
                # the ab-initio force is close to the force predicted by GPR. we check forces on other beads and try to exit.
                self.bead_index_with_converged_gpr_force = [bead_index_for_update]
                self.neb_stage_exit_step(step)

  

        # output info about force diff ratio |f_GPR -f|/|f|
        print("@Outerloop Exit info: ab initio |f|: " + str(self.ab_initio_force_amplitude_list))
        print("@Outloop Exit info: GPR predicted |f_GPR|: " + str(self.gpr_force_prediction_amplitude_list))
        print("@Outerloop Exit info: |f_GPR -f|/|f|:" + str(self.force_diff_ratio_list))        
        print("Finish Outerloop: " + str(step))
        print("\n")
        print("\n")

        self.force_diff_ratio_list = []
        self.ab_initio_force_amplitude_list = []
        self.gpr_force_prediction_amplitude_list = []
        

    def neb_stage_exit_step(self, step):
        '''
        check the ab-initio forces and compare it with forces predicted by GPR.
        We do not move NEB path during this process.
        If all beads pass the test: their ab-initio forces are close to GPR predicted forces,
        then we exit the NEB loop.
        '''
        while(len(self.bead_index_with_converged_gpr_force) < self.beads.nbeads):
            # gpr bead index is the index list that we still need to verify the ab-initio forces.
            gpr_bead_index_list = np.array(range(self.beads.nbeads))
            gpr_bead_index_list = np.delete(gpr_bead_index_list, self.bead_index_with_converged_gpr_force)

            _, beads_grad_x, _ , var_grad_x_trace = self.gpr_model.predict_latent_function(self.beads.q)
            beads_forces = -beads_grad_x

            # find the bead that has the largest energy variance among beads that we haven't evaluated their ab-initio potential.
            index_in_gpr_bead_index_list = np.argmax(var_grad_x_trace[gpr_bead_index_list])
            bead_index_for_update = gpr_bead_index_list[index_in_gpr_bead_index_list]

            # compute the ab-initio force and potential for the given bead.
            training_x = np.array([dstrip(self.beads.q[bead_index_for_update]).copy()])
            training_bead_forces = beads_forces[bead_index_for_update]
            
            # compute ab initio force for the bead and add it to the GPR model. 
            # evaluate the difference between ab initio force and force predicted by GPR.
            force_diff_ratio = self.update_GPR_model_one_bead_subroutine(training_x, bead_index_for_update,
                                                                         training_bead_forces)

            if force_diff_ratio < self.optarrays["gpr_force_criterion"]:
                self.bead_index_with_converged_gpr_force.append(bead_index_for_update)
            else:
                # the current bead configuration has not converged yet. Need to do nudged elastic band on the updated surrogated PES.
                self.bead_index_with_converged_gpr_force = []
                break 

        # all beads pass the test. The simulation has converged
        if len(self.bead_index_with_converged_gpr_force) == self.beads.nbeads:
            beads_pots_shift, beads_forces, _, _ = self.gpr_model.predict_latent_function(self.beads.q)
            beads_pots = beads_pots_shift + self.optarrays["energy_shift"]

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

            self.options["stage"] = "converged"
      
# ------ code below is for auxiliary functions --------------
    def check_spring_k_kappa(self):
        '''
        check the amplitude of spring k and kappa. to see if it is appropriate. If not, update it.
        '''
        dt = self.optarrays["time_step"]
        spring_k = self.optarrays["spring_k"]
        left_kappa = self.optarrays["kappa"]["left"]  # energy constraint constant for left end bead
        right_kappa = self.optarrays["kappa"]["right"] # energy constraint constant for right end bead

        # check spring_k * (dt)^2. We use stability criterion by setting spring_k * dt^2 = 0.25.
        val1 = spring_k * np.power(dt, 2)
        spring_k_scale = 0.25 / val1
        self.optarrays["spring_k"] = self.optarrays["spring_k"] * spring_k_scale
        self.nebgm.spring_k = self.nebgm.spring_k * spring_k_scale

        # check |dV/dx| * kappa / sqrt(m_H) * (dt)^2. We use stability criterion to set it as 0.5 (empirical value).
        m_H = 1837 # mass of hydrogen in atomic unit.
        # check the left end bead.
        max_force2 = np.max(np.abs(self.nebgm.rbf[0]))  # maximum gradient of left end bead.
        val2 = max_force2 * np.power(dt, 2) * left_kappa / np.sqrt(m_H)
        left_kappa_scale = 0.5 / val2
        self.optarrays["kappa"]["left"] = self.optarrays["kappa"]["left"] * left_kappa_scale
        
        # check the right end bead.
        max_force3 = np.max(np.abs(self.nebgm.rbf[-1]))  # maximum gradient of right end bead
        val3 = max_force3 * np.power(dt,2) * right_kappa / np.sqrt(m_H)
        right_kappa_scale = 0.5 / val3
        self.optarrays["kappa"]["right"] = self.optarrays["kappa"]["right"] * right_kappa_scale

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
    """Creation of the Line Integral function that will be minimized.
        Functional analog of a GradientMapper in geop.py

        Fixed atoms are excluded via boolean mask. 1 = moving, 0 = fixed.
        Reference paper: J. Chem. Phys. 148, 102334 (2018)
    Attributes:
        spring_k: spring constants
        kappa: energy constraint constant at two end beads. 
        action: abbreviated action W for the LI-NEB path.
        action_forces: negative gradient of abbreviated action W: - \nabla W
        neb_optimization_force: the force for LI-NEB method. See eq.(18) and eq.(19), (21) in the paper.
        neb_transverse_force: projection of action_force along transverse direction of NEB beads.
    """

    def __init__(self):
        self.spring_k = None    # spring constants for internal beads  
        self.kappa = None   # spring constants for beads at two ends.  

        self.init_allpots = None   #  initial potential for all beads. This potential will not be updated.
        self.action = None    # abbreviated action. 
        self.action_forces = None  # minus gradient of abbreviated action 
        self.neb_optimization_force = None  # neb force for optimization of action with constraints at two ends.
        self.neb_transverse_force = None # neb force for interior beads along transverse direction 

        self.instanton_path_energy = None # energy E of instanton path in JWKB approximation. See: Section II. A in J. Chem. Phys. 148, 102334 (2018)
    
    def bind(self, ens : MAPNEBGPRMover):
        '''
        :param: ens: A NEBMover instance.
        Copy beads, cell, forces of NEB mover to itself.
        '''
        self.dbeads = ens.beads.copy()
        self.dcell = ens.cell.copy()
        self.fixatoms = ens.fixatoms.copy()

        self.instanton_path_energy = ens.optarrays["instanton_path_energy"]   # bind the instanton path energy from NEB mover. 

        # Mask to exclude fixed atoms from 3N-arrays
        self.fixatoms_mask = np.ones(3 * ens.beads.natoms, dtype=bool)
        if len(ens.fixatoms) > 0:
            self.fixatoms_mask[3 * ens.fixatoms] = 0
            self.fixatoms_mask[3 * ens.fixatoms + 1] = 0
            self.fixatoms_mask[3 * ens.fixatoms + 2] = 0

        # Create reduced bead and force object (excluding the fixed atoms. But including the beads at two ends that also moves)
        self.rbeads = Beads(ens.beads.natoms, ens.beads.nbeads)
        self.rbeads.q[:] = ens.beads.q[:]
        self.rforces = ens.forces.copy(self.rbeads, self.dcell) # this will bind rbeads with rforces.

        self.spring_k = ens.optarrays["spring_k"] # bind spring force spring_k from NEBMover.
        self.kappa = ens.optarrays["kappa"] # bind end beads energy constraint constant kappa from NEBMover. 

        self.energy_shift = ens.optarrays["energy_shift"]

        # bind the gpr model from NEBMover.
        self.gpr_model = ens.gpr_model
        self.coordinate_transformer = ens.coordinate_transformer
        self.ab_initio_pot = np.zeros([self.dbeads.nbeads])
        self.ab_initio_force = np.zeros([self.dbeads.nbeads, 3 * self.dbeads.natoms])


    def __call__(self, x):
        """Returns the LI-NEB gradient for all beads. 
        update reduced bead coordinates (&dbeads coordinate) (sticly speaking the free-moving atom parts) with x.
        :param: x = q[:, self.fixatoms_mask] : new coordinates for updated freely moving particles.

        rbf: physical forces for reduced beads
        rbq: position for reduced beads
        btau: tangent vector directions.
        """
        self.rbeads.q[:, self.fixatoms_mask] = x
        rbq = np.copy(x)
        
        mscaled_q = rbq * np.sqrt( self.dbeads.m3[:, self.fixatoms_mask] )  # mass scaled coordinates.
        self.mscaled_q = mscaled_q

        # use Gaussian Process Regression to get the potential and forces for beads.
        self.beads_energy , beads_forces = self.get_gpr_potential_and_forces()

        # Forces for free moving dofs.
        self.rbf = beads_forces.copy()[:, self.fixatoms_mask]

        # mass weighted force
        mscaled_f = self.rbf / np.sqrt( self.dbeads.m3[: , self.fixatoms_mask] )  # 1/sqrt(m) * f: mass scaled force.
        self.mscaled_f = mscaled_f

        # Number of images
        nimage = self.dbeads.nbeads
        # Number of atoms that is free to move.
        natom = self.dbeads.natoms - len(self.fixatoms)

        self.spring_forces = np.zeros([nimage, 3 * natom])
        self.end_bead_energy_constraint_forces = np.zeros([2, 3 * natom])
        self.beads_mscaled_distance = npnorm(mscaled_q[1:] - mscaled_q[:-1] , axis = 1)

        # abbreviated action for the ring polymer instanton path.
        self.action = self.compute_neb_action(nimage, mscaled_q)
        
        # negative gradient of abbreviated action for each bead. We only compute it for the internal beads (excluding two ends)
        self.action_forces = self.compute_neb_action_force(nimage, natom, mscaled_q, mscaled_f)

        # compute direction of tangent vector, using either improved methods.
        btau = self.compute_tangent_vector(nimage, natom, mscaled_q)

        # evaluate the nudged elastic band optimization forces for perpendicular action forces and the spring force. (on mass scaled coordinate for free moving atoms.)
        neb_optimization_force = self.compute_neb_optimization_force(nimage, natom, btau, mscaled_q, mscaled_f)

        self.neb_optimization_force = np.copy(neb_optimization_force)

        return neb_optimization_force, self.action 
   

    def compute_tangent_vector(self, nimage, natom, mscaled_q):
        '''
        we used the improved tangent direction:
        J. Chem. Phys. 113, 9978 (2000); https://doi.org/10.1063/1.1323224
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
            
            gj_force_component = 0.5 * ( 1/action_each_bead[j] * (dj1 + dj2) * fj )
            
            gj_curvature_component = 0.5 * (- (action_each_bead[j] + action_each_bead[j-1]) * dj1_unit_vector + (action_each_bead[j] + action_each_bead[j+1]) * dj2_unit_vector)
            gj = gj_force_component + gj_curvature_component

            action_force[j] = gj 

        return action_force 
    
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

        # kappa: restraint force back to iso-energy contour.
        left_kappa = self.kappa["left"]   #  kappa for the left end beads
        right_kappa = self.kappa["right"] # kappa for the right end beads. 
        spring_k = self.spring_k    # spring force between beads.

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
        spring_force[0] = spring_force_bead0 - np.dot(spring_force_bead0 , f0) * f0  # spring force component transverse to the gradient of potential.

        # spring force for end bead nimag - 1
        unit_vec_2 = (mscaled_q[nimage - 2] - mscaled_q[nimage - 1]) / npnorm(mscaled_q[nimage - 2] - mscaled_q[nimage - 1])
        spring_force_bead1 = spring_k * ( npnorm(mscaled_q[nimage - 2] - mscaled_q[nimage - 1]) - npnorm(mscaled_q[nimage - 3] - mscaled_q[nimage -2]) ) * unit_vec_2 
        f1 = mscaled_f[nimage - 1] / npnorm(mscaled_f[nimage - 1])  # unit vector along force at beads: nimage - 1 
        spring_force[nimage - 1] = spring_force_bead1 - np.dot(spring_force_bead1 , f1) * f1  # spring force component transverse to the gradient of potential.

        # end_beads_energy_constraint_force: force to draw end beads back to isoenergy contours.
        end_beads_energy_constraint_force = np.zeros([2, 3 * natom])
        end_beads_energy_constraint_force[0] = f0 * left_kappa * (beads_energy[0] - self.instanton_path_energy)  # kappa * (V(r) - E) * \hat{f}(r) for beads 0
        end_beads_energy_constraint_force[1] = f1 * right_kappa * (beads_energy[nimage -1] - self.instanton_path_energy)  # kappa * (V(r) - E) * \hat{f}(r) for beads n-1.

        self.spring_forces = spring_force   # store the spring force between beads
        self.end_bead_energy_constraint_forces = end_beads_energy_constraint_force  # store energy constraint force for end beads.

        # for internal beads, transverse force from negative gradient of action.
        for ii in range(1, nimage - 1):
            neb_optimization_force[ii] = self.action_forces[ii] - np.dot(self.action_forces[ii] , btau[ii]) * btau[ii]
        
        self.neb_transverse_force = neb_optimization_force  # transverse gradient for interior neb beads.

        # add energy constraint force for two end beads.
        neb_optimization_force[0] = end_beads_energy_constraint_force[0]
        neb_optimization_force[nimage - 1] = end_beads_energy_constraint_force[1]

        # add spring force for all beads.
        neb_optimization_force = neb_optimization_force + spring_force 

        return neb_optimization_force 
    

    def get_gpr_potential_and_forces(self):
        '''
        Get potential and forces for all beads using the Gaussian Process Regression model.
        When there is ab-initio potential and force available, we use the ab initio value. 
        return: potential for all beads.
        '''
        test_x = np.copy(self.rbeads.q)
        beads_potential_shift, beads_potential_grad_x, _, _ = self.gpr_model.predict_latent_function(test_x)

        beads_forces = - beads_potential_grad_x
        # the predicted potential is the one relative to the energy shift.
        beads_potential = beads_potential_shift + self.energy_shift

        # check if ab_initio potential and forces are available. 
        # If so, use it and then reset it to zero, so we do not re-use it after we move the bead.
        for i in range(self.dbeads.nbeads):
            if self.ab_initio_pot[i] != 0:
                beads_potential[i] = self.ab_initio_pot[i]
                self.ab_initio_pot[i] = 0 
            if np.linalg.norm(self.ab_initio_force[i]) != 0:
                beads_forces[i] = self.ab_initio_force[i]
                self.ab_initio_force[i] = np.zeros([3 * self.dbeads.natoms])

        return beads_potential, beads_forces 
        

 




    


