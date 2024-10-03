"""Holds the algorithms to perform the Line Integral Nudged Elastic Band (NEB) method to find the instanton path.
J. Chem. Phys. 148, 102334 (2018); https://doi.org/10.1063/1.5007180

The LI-NEB calculation is accelerated by Gaussian Process Regression method. See: J. Chem. Phys. 147, 152720 (2017) and Faraday Discuss., 2018,212, 237-258 (https://doi.org/10.1039/C8FD00085A)

The algorithm is first implemented by Chenghao Zhang, 2023. Adapted from neb module & instanton module in i-pi package.

Written by Chenghao Zhang & Niri Govind, Pacific Northwest National Laboratory (chenghao.zhang@pnnl.gov)
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2021 i-PI developers
# See the "licenses" directory for full license information.

import numpy as np
from numpy.linalg import norm as npnorm
from ipi.utils import units
from ipi.engine.normalmodes import NormalModes
from ipi.engine.motion import Motion
from ipi.utils.depend import dstrip
from ipi.utils.softexit import softexit
from ipi.utils.messages import verbosity, info, warning
from ipi.engine.beads import Beads
import ipi.utils.nebinstool
from ipi.utils.nebinstool import RK4
from ipi.utils.internalcoordtools import non_redundant_coordinate_transformer
import ipi.utils.gprtools
import ipi.utils.nebinstgprtool
import ipi.utils.nebinstool
import ipi.utils.gpr_hessian_tools
import ipi.utils.mintools
import os
from timeit import default_timer as timer
import threading 

np.set_printoptions(threshold=10000, linewidth=1000)  # Remove in cleanup

__all__ = ["LINEBGradientMapper", "MAPNEBGPRMover"]


class MAPNEBGPRMover(Motion):
    """Nudged elastic band routine. for minimum action path (MAP)
    See J. Chem. Phys. 148, 102334 (2018)
    Accelerated by Gaussian Process Regression (GPR).
    Ref: J. Chem. Phys. 147, 152720 (2017) & Faraday Discuss., 2018,212, 237-258
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
        mode="verlet",
        prefix="neb_instanton",
        tolerances={"gradient": 5e-3, "gradient_end_bead": 1e-2},
        energy_shift=0.00,
        FIRE={"tmax": 4.0, "tmin": 0.1, 
              "Ndelay": 5, "finc": 1.1, "fdec": 0.5, 
              "alpha0": 0.15, "alpha_shrink": 0.99, 
              "Nmax": 100, "maxstep": 100,
              "neb_step_update_kappa": 20},
        time_step=4.0,
        cg_big_step= 1.0,
        instanton_time_step=4.0,
        stage="neb",
        instanton_bead_number=20,
        instanton_path_energy=0.00,
        instanton_temperature=1.0,
        instanton_bead_q=np.zeros(0, float),
        instanton_bead_pot=np.zeros(0, float),
        instanton_hessian=np.eye(0, 0, 0, float),
        neb_inner_loop_step_max = 100,
        spring_k=0.1,
        kappa={"left": 50, "right": 50},
        dynamical_adjust_ratio= {"spring_k": 0.1,
                                 "kappa": 0.2},
        end_bead_energy_converge_value = 1e-4,
        variable_spring_constant=False,
        VSC_E_ref=0.00,
        VSC_spring_k_max_ratio=3.00,
        alt_out=5,
        gpr_relative_force_error_criterion=0.05,
        gpr_absolute_force_error_criterion=0.002,
        gpr_trust_region=0.1,
        minimum_trust_region= 0.05,
        distance_cutoff_for_training_data= 0.05,
        gpr_kernel_outputscale=np.zeros(0, float),
        gpr_kernel_lengthscale_ratio=np.zeros(0, float),
        gpr_noise_std={
            "pot_noise_prior": 1e-6,
            "force_noise_prior": 1e-4,
            "hessian_noise_prior": 1e-3,
        },
        gpr_SE_kernel_number=1,
        gpr_fix_internal_dofs_bool= False,
        gpr_fix_internal_dofs_cutoff= 1e-4,
        read_initial_gpr_training_data=False,
        test_gpr_model_along_instanton_path= False,
        final_hessian_bool=False,
        ab_initio_hessian_bool=False,
        read_gpr_hessian_folder="None",
        add_new_hessian_data_bool=False,
        train_hessian_model_bool= True, 
        candidate_hessian_data_number=20,
        new_hessian_data_index=np.zeros(0, int),
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
        self.options["alt_out_step"] = alt_out  # step to output geometry.
        self.options["prefix"] = prefix
        self.options["final_hessian_bool"] = final_hessian_bool
        self.options["ab_initio_hessian_bool"] = ab_initio_hessian_bool
        self.options["read_initial_gpr_training_data"] = read_initial_gpr_training_data
        # for testing gpr prediction along lineb path.
        self.options["test_gpr_model_along_instanton_path"] = test_gpr_model_along_instanton_path
        # for store ab initio hessians used for gpr hessian model.
        self.options["read_gpr_hessian_folder"] = read_gpr_hessian_folder
        self.options["add_new_hessian_data_bool"] = add_new_hessian_data_bool
        self.options["train_hessian_model_bool"] = train_hessian_model_bool
        self.options["candidate_hessian_data_number"] = candidate_hessian_data_number
        self.options["gpr_fix_internal_dofs_bool"] = gpr_fix_internal_dofs_bool 
        self.options["gpr_fix_internal_dofs_cutoff"] = gpr_fix_internal_dofs_cutoff
        # minimum value for allowed trust region ratio.
        # This is to prevent the algorithm making the trust region ratio too small.
        self.options["minimum_trust_region"] = minimum_trust_region

        # The cutoff for the scaled internal coordinate distnace for training data.
        # The training data is not allowed to be too close to each other, which will make the kernel matrix ill-conditioned.
        self.options["distance_cutoff_for_training_data"] = distance_cutoff_for_training_data

        # numerical values / arrays. option from input.xml
        self.optarrays = {}
        self.optarrays["energy_shift"] = energy_shift

        self.optarrays["neb_inner_loop_step_max"] = neb_inner_loop_step_max
        self.optarrays["spring_k"] = spring_k
        self.optarrays["kappa"] = kappa
        self.optarrays["dynamical_adjust_ratio"] = dynamical_adjust_ratio
        self.optarrays["end_bead_energy_converge_value"] = end_bead_energy_converge_value

        # option to vary the spring constant term
        self.optarrays["variable_spring_constant"] = variable_spring_constant
        self.optarrays["VSC_E_ref"] = VSC_E_ref
        self.optarrays["VSC_spring_k_max_ratio"] = VSC_spring_k_max_ratio

        self.optarrays["FIRE"] = FIRE   # parameters for FIRE optimization algorithm.
        self.optarrays["time_step"] = time_step
        self.optarrays["cg_big_step"] = cg_big_step
        self.optarrays["instanton_time_step"] = instanton_time_step

        # input variable for instanton
        self.optarrays["instanton_path_energy"] = instanton_path_energy
        self.optarrays["instanton_bead_number"] = instanton_bead_number

        # for store the instanton result in RESTART file
        self.optarrays["instanton_temperature"] = instanton_temperature
        self.optarrays["instanton_bead_q"] = instanton_bead_q
        self.optarrays["instanton_bead_pot"] = instanton_bead_pot
        self.optarrays["instanton_hessian"] = instanton_hessian

        # for store ab initio hessians used for gpr hessian model.
        self.optarrays["new_hessian_data_index"] = new_hessian_data_index

        self.nebgm = LINEBGradientMapper()
        self.rp_map = RP_MAP()

        # variables for neb move
        self.velocity_mscaled = None
        self.x = None
        self.action = None
        self.f_mscaled = None
        self.grad_mscaled = None

        # variable below is for Gaussian Process Regression.
        if np.shape(gpr_kernel_outputscale) == (0,):
            raise (
                "You must provide output scale for covariance function. This should be a numpy array, with size equal to number of Squared Exponential (SE) kernel you use."
            )
        if np.shape(gpr_kernel_lengthscale_ratio) == (0,):
            raise (
                "You must provide length scale for covariance function. This should be a numpy array, with size equal to number of Squared Exponential (SE) kernel you use."
            )

        assert (
            len(gpr_kernel_lengthscale_ratio) == gpr_SE_kernel_number
        ), "The number of length scale of kernels should match the number of Squared Exponential kernel you use"
        assert (
            len(gpr_kernel_outputscale) == gpr_SE_kernel_number
        ), "The number of output scale of kernels should match the number of Squared Exponential kernel you use."

        self.optarrays["gpr_relative_force_error_criterion"] = (
            gpr_relative_force_error_criterion  # criterion to stop the outer loop.
        )
        self.optarrays["gpr_absolute_force_error_criterion"] = (
            gpr_absolute_force_error_criterion  # criterion to stop the outer loop when absolute value of gpr force error is small enough
        )
        self.optarrays["gpr_trust_region"] = (
            gpr_trust_region  # criterion to early stop the LI-NEB on PES generated by GPR.
        )
        self.optarrays["gpr_kernel_outputscale"] = (
            gpr_kernel_outputscale  # output scale of the kernel
        )
        self.optarrays["gpr_kernel_lengthscale_ratio"] = (
            gpr_kernel_lengthscale_ratio  # lengthscale of the gpr kernel.
        )
        self.optarrays["gpr_noise_std"] = gpr_noise_std
        self.options["gpr_SE_kernel_number"] = gpr_SE_kernel_number

        # index list storing the bead index whose ab-initio forces are close to their gpr predicted forces.
        self.ab_initio_index_list = []
        # |df|/|f_{ab initio}| for bead in index list.
        self.force_diff_ratio_list = []
        self.ab_initio_force_amplitude_list = []
        self.gpr_force_prediction_amplitude_list = []
        self.force_diff_amplitude_list = []

        self.coordinate_transformer = None  # coordinate transformer between the Cartesian coordinate and the internal coordinate
        self.gpr_model = None  # Gaussian Process Regression model instance.

        self.ab_initio_bead_calculation_number = (
            0  # record the number of ab initio calculation on beads.
        )

        self.internal_coordinate_closest_r_list = (
            []
        )  # measure the distance of beads from the training data in internal coordinate.
        self.trust_region_distance_cutoff = (
            0  # the distance cutoff for the trust region in internal coordinate system.
        )

        self.start_time = timer()  # used to record the time for the calculation.

        self.neb_optimization_step = 0

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
        self.rp_map.bind(self)

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
                self.output_maker,
            )

            # check number of active threads in the current python process
            num_threads = threading.active_count() 
            print(f"Number of threads used by the python program: {num_threads}")

            # The instanton path energy is defined relative to the energy shift.
            # We perform the transformation only when we start the initial calculation. Not for restarting the calculation.
            self.optarrays["instanton_path_energy"] = (
                self.optarrays["instanton_path_energy"] + self.optarrays["energy_shift"]
            )  # shift the instanton path energy according to energy shift.
            self.nebgm.instanton_path_energy = self.optarrays["instanton_path_energy"]
            self.rp_map.instanton_path_energy = self.optarrays["instanton_path_energy"]

            self.optarrays["VSC_E_ref"] = (
                self.optarrays["VSC_E_ref"] + self.optarrays["energy_shift"]
            )
            self.nebgm.VSC_E_ref = self.nebgm.VSC_E_ref + self.optarrays["energy_shift"]

        if self.coordinate_transformer is None:
            # initialize Gaussian Process Regression(GPR) model and coordiante transformer
            self.initialialize_GPR_model()
            if not self.options["stage"] == "test_gpr_hessian":
                # check the training result on the test data which is unseen by GPR.
                self.check_initial_training_result()

            # bind the gpr model and coordinate_transformer to the LINEGradientMapper class
            # the LINEBGradientMapper will perform LI-NEB using gpr generated potential and force.
            self.nebgm.gpr_model = self.gpr_model
            self.nebgm.coordinate_transformer = self.coordinate_transformer

            self.rp_map.gpr_model = self.gpr_model
            self.rp_map.coordinate_transformer = self.coordinate_transformer

        # Check if we enter the program directly into "instanton" stage:
        if self.options["stage"] == "instanton" and step == 0:
            self.rp_map.skip_neb_mode_bool = True
            print("Skip neb stage. Go directly into instanton stage. \n")
        else:
            self.rp_map.skip_neb_mode_bool = False

        # the file that store the maximum optimization gradient of LI-NEB beads.
        if self.options["stage"] == "neb" and step == 0:
            # file stores information of optimization gradient for each step in the inner loop. 
            optimization_gradient_file_name = "optimization_gradient.txt"
            self.optimization_gradient_file = open(optimization_gradient_file_name, "w")
            # file stores information of optimization gradient for each step in the outer loop.
            optimization_gradient_outloop_name = "optimization_gradient_outloop.txt"
            self.optimization_gradient_outloop_file = open(optimization_gradient_outloop_name, "w")
            
            # file stores the number of ab initio calculations & number of optimization steps
            # each time we print out the geometry of the file.
            geometry_info_file_name = "geometry_info.txt"
            self.geometry_info_file = open(geometry_info_file_name, "w")
            self.geometry_info_file.write("step   optimization_step   ab_initio_calculation_number \n")


        # Check if we restarted a converged calculation or the calculation converged.
        if self.options["stage"] == "converged":
            # output number of ab-initio calculation.
            ipi.utils.nebinstgprtool.print_ab_initio_calculation_number(
                self.ab_initio_bead_calculation_number, self.output_maker, step
            )
            print(
                "ab initio calculation number : "
                + str(self.ab_initio_bead_calculation_number)
            )

            # output the time for execuation
            self.end_time = timer()
            time_elapsed = (
                self.end_time - self.start_time
            ) / 60  # time elapsed in minutes
            print("the running time for the program: " + str(time_elapsed) + " min.")

            softexit.trigger(
                status="success",
                message="neb calculation converged. Instanton geometry calculation finishes. Exiting simulation",
            )

        elif self.options["stage"] == "neb":
            # use nudged elastic band method to find minmum action path.
            # then we will switch to the stage "instanton"
            # perform LI-NEB algorithm on the surrogated PES generated by GPR. stop either LI-NEB converge or one bead move out of the trust region.
            self.neb_loop_initialize()
            
            early_stop_bool, outrange_bead_index_list, grad_max_inner_bead, grad_max_end_bead = self.neb_loop(step)

            # write optimization gradient for each time we update the GPR model
            self.optimization_gradient_outloop_file.write(
                str(step) + " "
                + str(grad_max_inner_bead) + " "
                + str(grad_max_end_bead) + "\n"
            )

            # update Gaussian Process Regression model with new training data
            self.update_GPR_model(early_stop_bool, outrange_bead_index_list, step)

            print("optimization step so far for neb stage: " + str(self.neb_optimization_step))

        elif self.options["stage"] == "instanton":
            # generate instanton ring polymer beads from minimum action path found by NEB.
            info(
                "Now generate instanton path from Minimum Action Path (MAP) found by NEB."
            )
            print("total optimization step for neb stage: " + str(self.neb_optimization_step))

            self.rp_map.generate_ring_polymer_beads(self.beads, step)

            # save the potential, q, temperature, hessian of instanton beads for RESTART.
            self.save_instanton_ring_polymer()

            # ! If we exit here, the RESTART file will not record the hessian and instanton geometry we just computed.
            # therefore, we set ["stage"] == "converged" and exit at next step.
            self.options["stage"] = "converged"

        elif self.options["stage"] == "test_gpr_hessian":
            # test predicting hessian using gaussian process regression.
            self.test_gpr_hessian()

            print(
                "finish testing gpr model which predicts hessians. Please check the output result for performance. Exiting now."
            )
            self.options["stage"] = "converged"

        else:
            raise (
                "unrecognized stage parameter. The stage has to be neb or instanton or converged"
            )

    def generate_initial_training_data(self):
        """
        generate training data for Gaussian Process Regression model
        """
        # choose all NEB beads as initial training data.
        # We will train the GPR model to optimize hyperparameter using the initial data.
        train_x = np.copy(self.beads.q)
        # potential energy has to shift relative to the energy_shift for training.
        train_V = np.copy(self.forces.pots) - self.optarrays["energy_shift"]
        train_grad = -np.copy(dstrip(self.forces.f))
        # count the # of ab-initio calculation we have done.
        self.ab_initio_bead_calculation_number = (
            self.ab_initio_bead_calculation_number + self.beads.nbeads
        )
        return train_x, train_V, train_grad

    def read_initial_training_data(self):
        """
        read initial training data stored in files (previously computed)
        """
        train_x, stored_train_V, stored_train_f = (
            ipi.utils.nebinstgprtool.read_training_data(prefix="neb_final_gpr_training")
        )
        # count the # of ab-initio calculation we have done
        ab_initio_calculation_number = np.shape(train_x)[0]
        self.ab_initio_bead_calculation_number = (
            self.ab_initio_bead_calculation_number + ab_initio_calculation_number
        )

        train_V = stored_train_V - self.optarrays["energy_shift"]
        train_grad = -stored_train_f

        return train_x, train_V, train_grad

    def initialialize_GPR_model(self):
        """
        initialize the gaussian process regression model.
        1. Initialize coordinate transformer to transform between internal coordinate and cartesian coordinate.
        2. initialize GPR_Wrapper, which combines coordinate transformer and GPR model.
        """
        # Initialize non redundant coordinate transformer.
        # choose the point with the highest potential in the initial instanton path as reference point.
        beads_pots = np.copy(self.forces.pots)
        bead_index_at_transition_state = np.argmax(beads_pots)
        ref_x = dstrip(self.beads.q[bead_index_at_transition_state]).copy()

        # create coordinate_transformer, which handles the transformation from the Cartesian coordinate to internal coordinate.
        self.coordinate_transformer = non_redundant_coordinate_transformer(
            self.beads.natoms, ref_x
        )

        # attach ab_initio potential to self.nebgm.ab_initio_pot and self.nebgm.ab_initio_force.
        # In the LI-NEB algorithm, when there is ab-initio potential & force data available, we will use that potential and force.
        # If the ab-initio data point is not available, we use the potential and force generated by Gaussian Process Regression (GPR)
        self.nebgm.ab_initio_pot = np.copy(self.forces.pots)
        self.nebgm.ab_initio_force = np.copy(dstrip(self.forces.f))
        self.initial_beads_force_amplitude = np.linalg.norm(
            dstrip(self.forces.f).copy(), axis=1
        )

        # for the training data, we have the option to read it from .txt file or generate it using the current geometry.
        # this provides the flexibility for choosing the training data for the initial model.
        read_gpr_training_data_bool = self.options["read_initial_gpr_training_data"]
        if not read_gpr_training_data_bool:
            train_x, train_V, train_grad = self.generate_initial_training_data()
        else:
            train_x, train_V, train_grad = self.read_initial_training_data()

        self.gpr_model = ipi.utils.gprtools.GPModelWithDerivativesWrapper(
            train_x,
            train_V,
            train_grad,
            self.beads.natoms,
            self.coordinate_transformer,
            gpr_SE_kernel_number=self.options["gpr_SE_kernel_number"],
            kernel_outputscale=self.optarrays["gpr_kernel_outputscale"],
            kernel_lengthscale_ratio=self.optarrays["gpr_kernel_lengthscale_ratio"],
            noise_std=self.optarrays["gpr_noise_std"],
            train_bool= False,
            gpr_fix_internal_dofs_bool= self.options["gpr_fix_internal_dofs_bool"],
            gpr_fix_internal_dofs_cutoff= self.options["gpr_fix_internal_dofs_cutoff"] 
        )
        
        if read_gpr_training_data_bool:
            # see if there is option to read hyper-parameter without training the model
            neb_final_gpr_folder = "neb_final_gpr_training"
            model_hyperparameter_exists = ipi.utils.nebinstgprtool.load_training_hyperparameter_in_gpr_model(
                self.gpr_model, neb_final_gpr_folder
            )

            if not model_hyperparameter_exists:
                self.gpr_model.train_gpr()

        else:
            self.gpr_model.train_gpr()

    def check_initial_training_result(self):
        """
        check whether the training of GPR model is successful. If not, stop the simulation and report error
        """
        # first check the prediction of the training data. See if there is under-fitting.
        predicted_V_shift, predicted_grad, _, _ = (
            self.gpr_model.predict_latent_function(self.beads.q)
        )

        predicted_forces = -predicted_grad

        ab_initio_V_shift = self.forces.pots - self.optarrays["energy_shift"]
        ab_initio_forces = self.forces.f

        # check length scale for possible over-fitting
        learned_kernel_length_scale = self.gpr_model.output_kernel_lengthscale()
        internal_input_range = np.max(
            self.gpr_model.output_free_moving_training_internal_inputs(), axis=0
        ) - np.min(self.gpr_model.output_free_moving_training_internal_inputs(), axis=0)
        scaled_kernel_lengthscale = learned_kernel_length_scale / internal_input_range

        # check the size of covariance function (kernel).
        kernel_output_scale_var = self.gpr_model.output_kernel_outputscale()
        kernel_output_scale_std = np.sqrt(kernel_output_scale_var)

        print("\n")
        print(
            "@initial gpr training info: check the overfitting and underfitting of kernel length scale"
        )
        for i in range(self.gpr_model.gpr_SE_kernel_number):
            print("kernel {}: ".format(i))
            print(
                "square root of kernel output scale (\u03C3): "
                + str(kernel_output_scale_std[i])
            )
            print(
                "kernel_length_scale / input scale:   "
                + str(scaled_kernel_lengthscale[i])
            )
        print("\n")

        # check the force noise and potential noise. We can see for force noise of certain internal coordinate, it is quite large.
        force_range = self.gpr_model.output_normalized_force_range()
        V_noises, force_noises = self.gpr_model.output_fitted_gpr_model_noises()
        force_noises_ratio = force_noises / force_range
        print("potential noise amplitude: " + str(V_noises))
        print("force noise ratio  (amplitude / range): " + str(force_noises_ratio))
        print("internal coordinate force range: " + str(force_range))

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
        nbeads = self.beads.nbeads

        if nbeads >= 2:
            test_q = self.beads.q[0] * 1 / 4 + self.beads.q[1] * 3 / 4
            print("q[0] * 1/4 + q[1] * 3/4")
            (
                predicted_test_V_shift,
                predicted_test_force,
                ab_initio_test_pot,
                ab_initio_test_force,
            ) = ipi.utils.nebinstgprtool.check_gpr_fitting_error(
                self.gpr_beads,
                self.gpr_forces,
                self.gpr_model,
                self.optarrays["energy_shift"],
                test_q,
            )

            self.ab_initio_bead_calculation_number = (
                self.ab_initio_bead_calculation_number + 1
            )   

        if nbeads >= 4:
            test_q = self.beads.q[3] * 1 / 4 + self.beads.q[2] * 3 / 4
            print("q[3] * 1/4 + q[2] * 3/4")
            (
                predicted_test_V_shift,
                predicted_test_force,
                ab_initio_test_pot,
                ab_initio_test_force,
            ) = ipi.utils.nebinstgprtool.check_gpr_fitting_error(
                self.gpr_beads,
                self.gpr_forces,
                self.gpr_model,
                self.optarrays["energy_shift"],
                test_q,
            )

            self.ab_initio_bead_calculation_number = (
            self.ab_initio_bead_calculation_number + 1
            )

        pass

    def neb_loop(self, outer_loop_step):
        """
        the inner loop of Line Integral Nudged Elastic Band method.
        The loop will stop once one of the two criteria is met:
        (1) The LI-NEB algorithm converge on the surrogated PES generated by Gaussian Process Regression model.
            This is the case when all the gradient of LI-NEB beads are smaller than the tolerance value.
        (2) One LI-NEB bead move out of the trust region. In this case, PES generated by GPR is not reliable any more,
            we need to early stop the algorithm and compute the ab-initio V & F at that given bead & add to the training data.
            The trust region is defined in the internal coordinate, scaled by the length scale of the squared exponential kernel.
        """
        info(
            " @NEB: start inner loop neb for step {}".format(outer_loop_step),
            verbosity.debug,
        )

        grad_max_inner_bead = 1000
        grad_max_end_bead = 1000
        tolerances = self.options[
            "tolerances"
        ]  # tolerances for converging the LI-NEB calculation.

        neb_step = 0  # count the step number of neb move. (inner loop)

        early_stop_bool = False
        outrange_bead_index_list = (
            []
        )  # index for beads that move out of trusted region that causes the early stop.

        # print geometry when outer_loop_step % alt = 0. for record.
        self.print_geometry(outer_loop_step)

        print("\n")
        print("@Start outer loop: " + str(outer_loop_step) + "\n")
        while (
            grad_max_inner_bead > tolerances["gradient"]
            or grad_max_end_bead > tolerances["gradient_end_bead"]
        ):
            (
                grad_max_inner_bead,
                grad_max_end_bead,
                early_stop_bool,
                outrange_bead_index_list,
            ) = self.neb_step(outer_loop_step, neb_step, grad_max_inner_bead, grad_max_end_bead)

            neb_step = neb_step + 1

            # beads move out of trust region.
            if early_stop_bool:
                break

        if not early_stop_bool:
            print("@LI-NEB converge on GPR PES.")


        return early_stop_bool, outrange_bead_index_list, grad_max_inner_bead, grad_max_end_bead

    def neb_loop_initialize(self):
        """
        Initialize the action, force, velocity for nudged elastic band calculation. (inner loop calculation.)
        Each time we restart the neb-loop, the PES generated by GPR has changed. Therefore, we should restart the LI-NEB algorithm.
        """
        # coordinate of free moving atoms
        self.x = np.copy(self.beads.q[:, self.fixatoms_mask])

        # action of LI-NEB beads
        self.action = None
        
        # adjust the spring constant k and energy constraint term.
        self.nebgm.initialize_force(self.x)
        self.update_spring_k_kappa()

        mscaled_x = self.x * np.sqrt(
            self.beads.m3[:, self.fixatoms_mask]
        )
        self.action, self.grad_mscaled = self.nebgm(
            mscaled_x
        )
        # negative gradient of LI-NEB action for each bead on mass scaled coordinate
        self.f_mscaled = -self.grad_mscaled

        # mass scaled velocity. Used in dynamics optimizatioin algorithm,
        # for example: projected velocity verlet or FIRE 
        self.velocity_mscaled = np.zeros(
            [self.beads.nbeads, 3 * (self.beads.natoms - len(self.fixatoms))]
        )

        self.time_step = self.optarrays["time_step"]
        
        if self.options["mode"] == "cg":
            # use conjugate gradient method. 
            # The search is performed in the mass scaled coordinate.
            # initialize the search direction as gradient direction.
            self.conjugate_search_direction = self.f_mscaled

        if self.options["mode"] == "FIRE":
            # initialize parameter for FIRE method.
            self.alpha0 = self.optarrays["FIRE"]["alpha0"]
            self.alpha = self.alpha0
            self.alpha_shrink = self.optarrays["FIRE"]["alpha_shrink"]

            self.dtmax = self.optarrays["time_step"] * self.optarrays["FIRE"]["tmax"]
            self.dtmin = self.optarrays["time_step"] * self.optarrays["FIRE"]["tmin"]
            
            self.Ndelay = self.optarrays["FIRE"]["Ndelay"]
            self.finc = self.optarrays["FIRE"]["finc"]
            self.fdec = self.optarrays["FIRE"]["fdec"]

            self.Nmax = self.optarrays["FIRE"]["Nmax"]
            self.maxstep = self.optarrays["FIRE"]["maxstep"]

            self.Ndn = 0  # number of steps going down hill
            self.Nup = 0  # number of steps going up hill.
            


    def neb_instanton_step_info(
        self, outer_loop_step, neb_step, grad_max_inner_bead, grad_max_end_bead
    ):
        """
        output the information about convergence check for each step of neb move
        """
        tolerances = self.options["tolerances"]

        print("\n")
        info(
            "@Inner step summary: Outer loop # {} , inner loop # {},  max force gradient for inner bead {:4.2e}, (condition {:4.2e}), max force gradient for end bead {:4.2e} (condition {:4.2e})".format(
                outer_loop_step,
                neb_step,
                grad_max_inner_bead,
                tolerances["gradient"],
                grad_max_end_bead,
                tolerances["gradient_end_bead"],
            ),
            verbosity.low,
        )

        # record total number of optimization step
        self.neb_optimization_step = self.neb_optimization_step + 1

        # store the optimization gradient info.
        self.optimization_gradient_file.write(
            str(self.neb_optimization_step) + "  "
            + str(grad_max_inner_bead) + "  "
            + str(grad_max_end_bead) + "\n"
        )

        # print("old action: " + str(self.old_action) + "  new action: " + str(self.action))
        # check the optimization gradient for LI-NEB
        print(
            "beads optimization gradient: "
            + str(npnorm(self.nebgm.neb_optimization_force, axis=1))
        )
        # check the potential of beads.
        beads_energy_relative_to_instanton_energy = (
            self.nebgm.beads_energy - self.optarrays["instanton_path_energy"]
        ) * units.unit_to_user("energy", "electronvolt", 1)
        print(
            "beads potential relative to instanton path energy (eV): "
            + str(beads_energy_relative_to_instanton_energy)
        )
        # check distance between beads (effect of spring_k)
        print(
            "distance between beads in mass scaled coordinate: "
            + str(self.nebgm.beads_mscaled_distance)
        )
        print("\n")
        print(
            "@Finish Inner loop: outer loop step {}, LI-NEB inner loop step {}".format(
                outer_loop_step, neb_step
            )
        )
        print("\n")
        print("\n")

    def neb_step(self, outer_loop_step, neb_step, grad_max_inner_bead, grad_max_end_bead):
        """
        LI-NEB move for one step.
        """
        nbeads = self.beads.nbeads

        neb_inner_loop_step_max = self.optarrays["neb_inner_loop_step_max"]


        if self.options["mode"] == "FIRE":
            if neb_step % self.optarrays["FIRE"]["neb_step_update_kappa"] == 0:
                self.update_spring_k_kappa()
        else:
            self.update_spring_k_kappa()
        

        if neb_step > neb_inner_loop_step_max:
            softexit.trigger(
                status= "bad",
                message= "The neb inner loop fails to converge after reaching the maximum optimization steps: " + str(neb_inner_loop_step_max),
            )

        # We have changed spring constant, so, we have to recompute neb optimization force.
        x_mscaled = self.x * np.sqrt(
            self.beads.m3[:, self.fixatoms_mask]
        )
        self.action, self.grad_mscaled = self.nebgm(x_mscaled)
        self.f_mscaled = -self.grad_mscaled

        # check early stop condition if there are beads out of trust region.
        # the trust region is defined in the internal coordinate, scaled by length scale of the kernel.
        (
            early_stop_bool,
            outrange_bead_index_list,
            self.internal_coordinate_closest_r_list,
        ) = ipi.utils.nebinstgprtool.check_neb_early_stop(
            self.beads.q,
            self.optarrays["gpr_trust_region"],
            self.gpr_model,
            outer_loop_step,
            neb_step,
        )

        # stop the step early if there are beads out of trust region.
        if early_stop_bool:
            return (
                grad_max_inner_bead,
                grad_max_end_bead,
                early_stop_bool,
                outrange_bead_index_list,
            )

        # neb move using gradient of LINEBGradient
        # See: J. Chem. Phys. 128, 134106 (2008) for the performance of different optimization algorithms.
        if self.options["mode"] == "verlet":
            # This is Quick-min algorithm in J. Chem. Phys. 128, 134106 (2008)
            self.neb_step_projected_verlet()
        elif self.options["mode"] == "cg":
            # move one step using conjugate gradient method 
            self.neb_step_cg()
        elif self.options["mode"] == "FIRE":
            # move one step using FIRE method.
            self.neb_step_FIRE()
        else:
            softexit.trigger(
                status="bad",
                message="Only projected velocity verlet (verlet), conjugate gradient (cg) and FIRE are currently implemented. set mode == 'verlet' ",
            )

        # compute maximum LI-NEB gradient among all beads. used for monitoring the convergence of LI-NEB.
        grad_norm = npnorm(self.nebgm.neb_optimization_force, axis=1)

        grad_max_inner_bead = np.amax(grad_norm[1 : nbeads - 1])
        grad_max_end_bead = np.amax(np.array([grad_norm[0], grad_norm[-1]]))
        # output info about neb calculation.
        self.neb_instanton_step_info(
            outer_loop_step, neb_step, grad_max_inner_bead, grad_max_end_bead
        )

        return (
            grad_max_inner_bead,
            grad_max_end_bead,
            early_stop_bool,
            outrange_bead_index_list,
        )

    def neb_step_projected_verlet(self):
        """
        use the projected velocity verlet algorithm to optimize the bead position 
        """
        x_mscaled = self.x * np.sqrt(
            self.beads.m3[:, self.fixatoms_mask]
        )

        x_mscaled, self.velocity_mscaled, self.action, self.grad_mscaled = \
            ipi.utils.nebinstool.projected_verlet(
            x_mscaled, 
            self.velocity_mscaled,
            (self.action, self.grad_mscaled),
            self.nebgm,
            self.time_step
        )
        self.f_mscaled = -self.grad_mscaled

        # update new position
        self.x = x_mscaled / np.sqrt(
            self.beads.m3[:, self.fixatoms_mask]
        )
        self.beads.q[:, self.fixatoms_mask] = self.x

    def neb_step_cg(self):
        """
        use the conjugate gradient algorithm to optimize the bead position.
        """
        x_mscaled = self.x * np.sqrt(
            self.beads.m3[:, self.fixatoms_mask]
        )

        x_mscaled, self.action, self.grad_mscaled, self.conjugate_search_direction= \
            ipi.utils.nebinstool.conjugate_gradient(
                x_mscaled,
                (self.action, self.grad_mscaled),
                self.nebgm,
                self.conjugate_search_direction,
                self.optarrays["cg_big_step"]
            )
        self.f_mscaled = -self.grad_mscaled 

        # update new position 
        self.x = x_mscaled / np.sqrt(
            self.beads.m3[:, self.fixatoms_mask]
        )
        self.beads.q[:, self.fixatoms_mask] = self.x

    def neb_step_FIRE(self):
        """
        use the FIRE (Fast Inertial Relaxation Engine) algorithm to optimize the bead position.
        """
        x_mscaled = self.x * np.sqrt(
            self.beads.m3[:, self.fixatoms_mask]
        )
        fdf0 = (self.action, self.grad_mscaled)
        # one step using FIRE. 
        # the x_mscaled will be updated in the mintools.FIRE() code.
        self.velocity_mscaled, self.alpha, self.Ndn, self.Nup, self.time_step  = \
              ipi.utils.mintools.FIRE(x_mscaled,
                                self.nebgm,
                                fdf0,
                                self.velocity_mscaled,
                                self.alpha,
                                self.Ndn,
                                self.Nup,
                                self.time_step,
                                self.maxstep,
                                self.dtmax,
                                self.dtmin,
                                self.Ndelay,
                                self.Nmax,
                                self.finc,
                                self.fdec,
                                self.alpha0,
                                self.alpha_shrink
                                )
        # update mass scaled gradient, force and action for LI-NEB 
        # FIRE() code has already called fdf(x) in the code.
        self.action = self.nebgm.action 
        self.grad_mscaled = self.nebgm.neb_optimization_gradient
        self.f_mscaled = -self.grad_mscaled
        # update new position 
        self.x = x_mscaled / np.sqrt(
            self.beads.m3[:, self.fixatoms_mask]
        )

        self.beads.q[:, self.fixatoms_mask] = self.x


    def update_GPR_model_one_bead_subroutine(
        self, training_x, bead_index_for_update, training_bead_forces
    ):
        """
        compute ab initio potential for the bead of interest, add it into the training data.
        Also compute difference between ab initio force and force predicted by GPR. Check the convergence of the GPR model.
        """
        # consistency check : the gpr_beads we claims to do the simulation should have same bead number of training data.
        assert self.gpr_beads.nbeads == len(training_x)
        self.gpr_beads.q[:] = training_x

        # get energy and forces (in Cartesian coordinate) from force engine. ab initio calculation.
        ab_initio_beads_energy = dstrip(self.gpr_forces.pots).copy()
        ab_initio_beads_forces = dstrip(self.gpr_forces.f).copy()
        ab_initio_beads_grad = -ab_initio_beads_forces

        # update GPR model with coordinate (training_x), potential (beads_energy) and forces in cartesian coordiante (beads_forces)
        ab_initio_beads_shifted_energy = (
            ab_initio_beads_energy - self.optarrays["energy_shift"]
        )
        self.gpr_model.update_model_with_new_data(
            training_x,
            ab_initio_beads_shifted_energy,
            ab_initio_beads_grad,
            self.options["distance_cutoff_for_training_data"],
        )
        self.nebgm.gpr_model = self.gpr_model

        # set ab_initio pot and force in nebgm.
        self.nebgm.ab_initio_pot[bead_index_for_update] = ab_initio_beads_energy
        self.nebgm.ab_initio_force[bead_index_for_update] = ab_initio_beads_forces
        # count the # of ab-initio calculation we have done.
        self.ab_initio_bead_calculation_number = (
            self.ab_initio_bead_calculation_number + 1
        )

        # compute the difference between ab initio force and gpr force.
        ab_initio_force_amplitude = np.linalg.norm(ab_initio_beads_forces[0])
        force_diff = training_bead_forces - ab_initio_beads_forces[0]
        force_diff_ratio = np.linalg.norm(force_diff) / ab_initio_force_amplitude

        # add |df|/|f| , f and f^{GPR} into the list.
        self.force_diff_ratio_list.append(force_diff_ratio)
        self.ab_initio_force_amplitude_list.append(ab_initio_force_amplitude)
        self.gpr_force_prediction_amplitude_list.append(
            np.linalg.norm(training_bead_forces)
        )

        return force_diff_ratio, force_diff

    def update_GPR_model_with_beads_cause_early_stop(self, outrange_bead_index_list):
        """
        compute the ab initio potential and forces for beads far away from the trust region that causes the early stop.
        """
        force_diff_list = []
        for outrange_bead_index in outrange_bead_index_list:
            bead_index_for_update = outrange_bead_index
            training_x = np.array([dstrip(self.beads.q[bead_index_for_update]).copy()])

            # evaluate the gpr predicted V & f. For comparison with ab-initio V & F.
            _, training_grad_x, _, _ = self.gpr_model.predict_latent_function(
                training_x
            )
            training_bead_forces = -training_grad_x[0]

            # compute ab initio force for the bead and add it to the GPR model.
            # evaluate the difference between ab initio force and force predicted by GPR.
            force_diff_ratio, force_diff = self.update_GPR_model_one_bead_subroutine(
                training_x, bead_index_for_update, training_bead_forces
            )
            force_diff_list.append(force_diff)

        force_diff_amplitude_list = np.linalg.norm(force_diff_list, axis=1)
        self.force_diff_amplitude_list = force_diff_amplitude_list
        # deal with the case that the trust region distance cutoff could be too large.
        # we can detect this situation when the gpr model predict the force for beads close to the trust region far away from the true force.
        # in this case, we have to decrease the trust region distance.
        outrange_bead_distance_list = self.internal_coordinate_closest_r_list[
            outrange_bead_index_list
        ]
        # we require beads within 1.5 * trust region should have error of force prediction < 0.1
        distance_cutoff = self.trust_region_distance_cutoff * 1.5
        force_error_ratio_cutoff = 0.1

        _, gpr_grad_x, _, _ = self.gpr_model.predict_latent_function(self.beads.q)

        for i in range(len(outrange_bead_index_list)):
            outrange_bead_distance = outrange_bead_distance_list[i]
            bead_force_diff_ratio = self.force_diff_ratio_list[i]

            # we do not consider trust region is too small when the absolute error of force is small but relative error of force is large
            if (
                force_diff_amplitude_list[i]
                > self.optarrays["gpr_absolute_force_error_criterion"]
            ):
                # we have to make sure when the bead is close to the trust region, the relative error of gpr force prediction is small.
                if (
                    outrange_bead_distance < distance_cutoff
                    and bead_force_diff_ratio > force_error_ratio_cutoff
                ):
                    if (
                        self.optarrays["gpr_trust_region"]
                        > self.options["minimum_trust_region"] * 2
                    ):
                        self.optarrays["gpr_trust_region"] = (
                            self.optarrays["gpr_trust_region"] / 2
                        )
                        print(
                            "@READJUST TRUST REGION: the trust region ratio now: "
                            + str(self.optarrays["gpr_trust_region"])
                        )
                        break

    def update_GPR_model_all_image_strategy(self, step):
        """
        we compute ab initio potential and forces for all beads.
        check the error of force prediction and try to exit.
        """
        info(
            "All Image strategy for updating GPR model when LI-NEB converges. \n",
            verbosity.low,
        )

        # compute gpr predicted forces
        _, beads_grad_x, _, _ = self.gpr_model.predict_latent_function(self.beads.q)
        beads_forces = -beads_grad_x

        # compute ab initio potential and forces
        training_x = np.copy(self.beads.q)
        ab_initio_beads_energy = dstrip(self.forces.pots).copy()
        ab_initio_shifted_energy = (
            ab_initio_beads_energy - self.optarrays["energy_shift"]
        )
        ab_initio_forces = dstrip(self.forces.f).copy()
        ab_initio_grad_x = -ab_initio_forces

        # update the gpr model with new data
        self.gpr_model.update_model_with_new_data(
            training_x,
            ab_initio_shifted_energy,
            ab_initio_grad_x,
            self.options["distance_cutoff_for_training_data"],
        )
        self.nebgm.gpr_model = self.gpr_model

        # set ab_initio pot and force in nebgm.
        self.nebgm.ab_initio_pot[:] = ab_initio_beads_energy
        self.nebgm.ab_initio_force[:] = ab_initio_forces
        # count the # of ab-initio calculation we have done.
        self.ab_initio_bead_calculation_number = (
            self.ab_initio_bead_calculation_number + self.beads.nbeads
        )

        # check whether the ab-initio forces are close to gpr predicted forces.
        force_diff_list = beads_forces - ab_initio_forces
        force_diff_amplitude_list = np.linalg.norm(force_diff_list, axis=1)
        self.force_diff_amplitude_list = force_diff_amplitude_list
        self.ab_initio_force_amplitude_list = np.linalg.norm(ab_initio_forces, axis=1)
        self.gpr_force_prediction_amplitude_list = np.linalg.norm(beads_forces, axis=1)
        self.force_diff_ratio_list = (
            np.linalg.norm(force_diff_list, axis=1)
            / self.ab_initio_force_amplitude_list
        )

        # when the error of force is smaller than a given value, we assume GPR fitting is successful.
        gpr_absolute_force_error_criterion = self.optarrays[
            "gpr_absolute_force_error_criterion"
        ]

        gpr_force_converge_bool = True
        for i in range(self.beads.nbeads):
            # when the error of force is smaller than a given value, we assume GPR fitting is successful.
            if force_diff_amplitude_list[i] < gpr_absolute_force_error_criterion:
                pass
            # when the bead force is large, we require relative error of force to be small.
            elif (
                self.force_diff_ratio_list[i]
                < self.optarrays["gpr_relative_force_error_criterion"]
            ):
                pass
            else:
                gpr_force_converge_bool = False

        if gpr_force_converge_bool:
            self.neb_stage_exit_step(step, ab_initio_beads_energy)

    def update_GPR_model(self, early_stop_bool, outrange_bead_index_list, step):
        """
        update GPR model with new training data. Which new training data we will add depends on the stop criterion.
        evaluate potential and force of one bead.
        Then update the Gassian Process Regression model.
        """
        print("The trust region now: " + str(self.optarrays["gpr_trust_region"]))
        if early_stop_bool:
            # in this case, several beads have moved out of trust region. We add this bead into the training data.
            self.update_GPR_model_with_beads_cause_early_stop(outrange_bead_index_list)
        else:
            # All image strategy
            # We compute the potential and force for all images and add them to the training data.
            # The convergence criterion is reached when gpr forces for all beads converge to the ab initio forces.
            self.update_GPR_model_all_image_strategy(step)

        # output info about force diff ratio |f_GPR -f|/|f|
        print(
            "@Outerloop Exit info: ab initio |f|: "
            + str(self.ab_initio_force_amplitude_list)
        )
        print(
            "@Outloop Exit info: GPR predicted |f_GPR|: "
            + str(self.gpr_force_prediction_amplitude_list)
        )
        print("@Outerloop Exit info: |f_GPR -f|/|f|:" + str(self.force_diff_ratio_list))
        print(
            "@Outerloop Exit info: max(|f_GPR - f|/|f|): "
            + str(np.max(self.force_diff_ratio_list))
        )
        print(
            "@Outerloop Exit info: |f_GPR -f| :" + str(self.force_diff_amplitude_list)
        )
        print("Finish Outerloop: " + str(step))
        print("\n")
        print("\n")

        self.force_diff_ratio_list = []
        self.ab_initio_force_amplitude_list = []
        self.gpr_force_prediction_amplitude_list = []

    def neb_stage_exit_step(self, step, beads_pots):
        """
        We exit neb stage and enter instanton stage.
        This function store gpr_model training data points & record LINEB path coordinate, force & potentials.
        """
        info(
            "@Exit step: NEB_instanton: path optimization converged. Step %i \n" % step,
            verbosity.low,
        )

        self.optimization_gradient_file.close()
        self.optimization_gradient_outloop_file.close()
        self.geometry_info_file.close()

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
            self.output_maker,
        )

        self.options["stage"] = "instanton"

        # store potential and forces for the final LI-NEB beads.
        self.LINEB_pots = (
            self.gpr_model.train_cartesian_targets[-self.beads.nbeads :, 0]
            + self.optarrays["energy_shift"]
        )
        self.LINEB_forces = -self.gpr_model.train_cartesian_targets[
            -self.beads.nbeads :, 1:
        ]

        ipi.utils.nebinstgprtool.store_training_data(
            self.beads.q, self.LINEB_pots, self.LINEB_forces, prefix="LINEB_beads"
        )

        # store all training data
        train_x = self.gpr_model.train_cartesian_inputs
        train_V = self.gpr_model.train_cartesian_targets[:, 0]
        train_V_to_store = train_V + self.optarrays["energy_shift"]
        train_grad = self.gpr_model.train_cartesian_targets[:, 1:]
        train_f_to_store = -train_grad

        ipi.utils.nebinstgprtool.store_training_data(
            train_x, train_V_to_store, train_f_to_store, prefix="neb_final_gpr_training"
        )
        neb_gpr_folder_path = "neb_final_gpr_training"
        ipi.utils.nebinstgprtool.store_training_hyperparameter_in_gpr_model(self.gpr_model,
                                                                            neb_gpr_folder_path)

    # ------ code below is for auxiliary functions --------------
    def update_spring_k_kappa(self):
        """
        check the amplitude of spring k and kappa. to see if it is appropriate. If not, update it.
        """
        dt = self.optarrays["time_step"]
        spring_k = self.optarrays["spring_k"]
        left_kappa = self.optarrays["kappa"][
            "left"
        ]  # energy constraint constant for left end bead
        right_kappa = self.optarrays["kappa"][
            "right"
        ]  # energy constraint constant for right end bead

        end_bead_energy_converge_value = self.optarrays["end_bead_energy_converge_value"]
        end_bead_gradient_tolerances = self.options["tolerances"]["gradient_end_bead"]

        # check spring_k * (dt)^2. We use stability criterion by setting spring_k * dt^2 = 0.25.
        val1 = spring_k * np.power(dt, 2)
        spring_k_ratio = self.optarrays["dynamical_adjust_ratio"]["spring_k"]
        spring_k_scale = spring_k_ratio / val1
        self.optarrays["spring_k"] = self.optarrays["spring_k"] * spring_k_scale
        self.nebgm.spring_k = self.nebgm.spring_k * spring_k_scale
        self.nebgm.VSC_k_max = self.nebgm.spring_k
        self.nebgm.VSC_k_ref = self.nebgm.VSC_k_max / self.nebgm.VSC_spring_k_max_ratio

        kappa_ratio = self.optarrays["dynamical_adjust_ratio"]["kappa"]
        
        min_max_force = np.power(10.0, -3)
        # check |dV/dx| * kappa / sqrt(m_H) * (dt)^2. We use stability criterion to set it as 0.5 (empirical value).
        m_H = 1837  # mass of hydrogen in atomic unit.
        # check the left end bead.
        max_force2 = np.max(
            np.abs(self.nebgm.rbf[0])
        )  # maximum gradient of left end bead.
        max_force2 = np.max([min_max_force, max_force2])
        val2 = max_force2 * np.power(dt, 2) * left_kappa / np.sqrt(m_H)
        left_kappa_scale = kappa_ratio / val2
        self.optarrays["kappa"]["left"] = (
            self.optarrays["kappa"]["left"] * left_kappa_scale
        )

        # make sure the kappa value is not too large for the convergence
        if abs(self.nebgm.beads_energy[0] - self.optarrays["instanton_path_energy"]) < end_bead_energy_converge_value:
            left_kappa_for_converge = 0.2 * end_bead_gradient_tolerances / end_bead_energy_converge_value
            self.optarrays["kappa"]["left"] = np.min([self.optarrays["kappa"]["left"] , left_kappa_for_converge])

        # check the right end bead.
        max_force3 = np.max(
            np.abs(self.nebgm.rbf[-1])
        )  # maximum gradient of right end bead
        max_force3 = np.max([min_max_force, max_force3])
        val3 = max_force3 * np.power(dt, 2) * right_kappa / np.sqrt(m_H)
        right_kappa_scale = kappa_ratio / val3
        self.optarrays["kappa"]["right"] = (
            self.optarrays["kappa"]["right"] * right_kappa_scale
        )

        # make sure kappa value is not too large for the converge.
        if abs(self.nebgm.beads_energy[-1] - self.optarrays["instanton_path_energy"]) < end_bead_energy_converge_value:
            right_kappa_for_converge = 0.2 * end_bead_gradient_tolerances / end_bead_energy_converge_value
            self.optarrays["kappa"]["right"] = np.min([self.optarrays["kappa"]["right"], right_kappa_for_converge])

    def scale_down_spring_constant_and_kappa(self):
        """
        When the inner loop code fails to converge after several steps, we keep scaling down the spring force and kappa term.
        """
        scale = 0.5
        self.optarrays["spring_k"] = self.optarrays["spring_k"] * scale
        self.nebgm.spring_k = self.nebgm.spring_k * scale 
        self.nebgm.VSC_k_max = self.nebgm.spring_k 
        self.nebgm.VSC_k_ref = self.nebgm.VSC_k_max / self.nebgm.VSC_spring_k_max_ratio

        self.optarrays["kappa"]["left"] = (
            self.optarrays["kappa"]["left"] * scale 
        )

        self.optarrays["kappa"]["right"] = (
            self.optarrays["kappa"]["right"] * scale
        )

    def print_geometry(self, step):
        """
        print beads geometry and beads energy.
        """
        pots = self.nebgm.beads_energy
        if (
            self.options["alt_out_step"] > 0
            and np.mod(step, self.options["alt_out_step"]) == 0
        ):
            # print energy, force and coordinate 
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
                self.output_maker,
            )

            # print number of ab initio calculations and number of optimization step for each file.
            self.geometry_info_file.write(f"{step} {self.neb_optimization_step} {self.ab_initio_bead_calculation_number} \n")

    def save_instanton_ring_polymer(self):
        """
        save the ring polymer instanton computed in RP_MAP class.
        Therefore, the result can be stored in RESTART file
        """
        self.optarrays["instanton_temperature"] = self.rp_map.instanton_temp
        self.optarrays["instanton_bead_q"] = self.rp_map.rp_beads.q
        self.optarrays["instanton_bead_pot"] = self.rp_map.rp_beads_pots
        self.optarrays["instanton_hessian"] = self.rp_map.rp_hessian

        # print hessian
        if self.options["final_hessian_bool"]:
            ipi.utils.nebinstool.print_instanton_hess(
                self.options["prefix"] + "_FINAL",
                self.optarrays["instanton_hessian"],
                self.output_maker,
            )

    # ------------------- code below used to test the ability that gaussian process regression (GPR) predicts hessian information ----------
    def test_gpr_hessian(self):
        """
        test the performance of gaussian process regression model that predicts hessians.
        We randomly sample data around the center coordinate & separate data into training data and test data.
        """
        train_data_number = 20
        test_data_number = 5
        train_data_with_hessian_number = 5
        test_data_with_hessian_number = test_data_number

        total_data_number = int(train_data_number + test_data_number)
        total_data_with_hessian_number = int(
            train_data_with_hessian_number + test_data_with_hessian_number
        )

        natoms = self.beads.natoms
        # beads for computing potential and force for all data points
        beads1 = Beads(natoms, total_data_number)
        forces1 = self.forces.copy(beads1, self.cell)

        # beads for computing hessian for selected data point.
        beads2 = Beads(natoms, total_data_with_hessian_number)
        forces2 = self.forces.copy(beads2, self.cell)

        # create cartesian coordinate of training data and test data
        center_coordinate = self.beads.q[0]
        sample_distance = 0.01
        data_coordinate_shape = np.concatenate(
            [[total_data_number], np.array(center_coordinate.shape)]
        )
        data_coordinate_displacement = np.random.normal(
            0, scale=sample_distance, size=data_coordinate_shape
        )
        data_coordinate = (
            center_coordinate[np.newaxis, ...] + data_coordinate_displacement
        )

        train_data_coordinate = data_coordinate[:train_data_number]
        test_data_coordinate = data_coordinate[train_data_number:]

        train_data_with_hessian_coordinate = train_data_coordinate[
            :train_data_with_hessian_number
        ]
        test_data_with_hessian_coordinate = test_data_coordinate

        # assign coordinate to beads.q
        beads1.q = data_coordinate
        beads2.q = np.concatenate(
            [train_data_with_hessian_coordinate, test_data_with_hessian_coordinate]
        )

        # compute potential and force and hessian.
        train_pots = forces1.pots[:train_data_number] - self.optarrays["energy_shift"]
        train_gradients = -np.copy(dstrip(forces1.f))[:train_data_number]

        test_gradients = -np.copy(dstrip(forces1.f))[train_data_number:]

        hessians = ipi.utils.nebinstool.get_hessian(
            beads2,
            forces2,
            np.copy(beads2.q),
            beads2.natoms,
            beads2.nbeads,
            self.fixatoms,
        )

        hessians = np.transpose(
            np.reshape(hessians, [3 * natoms, beads2.nbeads, 3 * natoms]), (1, 0, 2)
        )

        train_ab_initio_hessians = hessians[:train_data_with_hessian_number]
        test_ab_initio_hessians = hessians[train_data_with_hessian_number:]

        train_hessian_data_point_index_array = np.arange(train_data_with_hessian_number)
        test_hessian_data_point_index_array = np.arange(test_data_with_hessian_number)

        # create Gaussian Process regression model that can predict hessians
        ref_x = train_data_coordinate[0]
        ref_V = np.array([train_pots[0]])
        ref_grad_x = train_gradients[0]
        ref_hessian_x = train_ab_initio_hessians[0]
        self.gpr_hessian_model = ipi.utils.gpr_hessian_tools.GPModelWithHessiansWrapper(
            train_data_coordinate,
            train_pots,
            train_gradients,
            train_ab_initio_hessians,
            train_hessian_data_point_index_array,
            self.beads.natoms,
            self.coordinate_transformer,
            self.options["gpr_SE_kernel_number"],
            self.optarrays["gpr_kernel_outputscale"],
            self.optarrays["gpr_kernel_lengthscale_ratio"],
            self.optarrays["gpr_noise_std"],
            constant_mean_func_bool=False,
            ref_mean_x=ref_x,
            ref_mean_V=ref_V,
            ref_mean_grad_x=ref_grad_x,
            ref_mean_hessian_x=ref_hessian_x,
        )

        (
            gpr_hessian_kernel_outputscale,
            gpr_hessian_lengthscale_list,
            gpr_hessian_lengthscale_ratio_list,
        ) = ipi.utils.nebinstgprtool.check_gpr_hessian_model_lengthscale(
            self.gpr_hessian_model
        )

        # test the training data
        predicted_train_hessians_q, train_ab_initio_hessians_q = (
            ipi.utils.nebinstgprtool.compare_ab_initio_hessian_and_predicted_hessian(
                train_data_with_hessian_coordinate,
                train_gradients,
                train_ab_initio_hessians,
                train_hessian_data_point_index_array,
                self.gpr_hessian_model,
                internal_coordinate_bool=True,
                training_data_bool=True,
            )
        )

        predicted_train_hessians, train_ab_initio_hessians = (
            ipi.utils.nebinstgprtool.compare_ab_initio_hessian_and_predicted_hessian(
                train_data_with_hessian_coordinate,
                train_gradients,
                train_ab_initio_hessians,
                train_hessian_data_point_index_array,
                self.gpr_hessian_model,
                internal_coordinate_bool=False,
                training_data_bool=True,
            )
        )

        # test the test data
        predicted_test_hessians_q, test_ab_initio_hessians_q = (
            ipi.utils.nebinstgprtool.compare_ab_initio_hessian_and_predicted_hessian(
                test_data_with_hessian_coordinate,
                test_gradients,
                test_ab_initio_hessians,
                test_hessian_data_point_index_array,
                self.gpr_hessian_model,
                internal_coordinate_bool=True,
                training_data_bool=False,
            )
        )

        predicted_test_hessians, test_ab_initio_hessians = (
            ipi.utils.nebinstgprtool.compare_ab_initio_hessian_and_predicted_hessian(
                test_data_with_hessian_coordinate,
                test_gradients,
                test_ab_initio_hessians,
                test_hessian_data_point_index_array,
                self.gpr_hessian_model,
                internal_coordinate_bool=False,
                training_data_bool=False,
            )
        )

        pass


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
        self.spring_k = None  # spring constants for internal beads
        self.kappa = None  # spring constants for beads at two ends.

        self.init_allpots = (
            None  # initial potential for all beads. This potential will not be updated.
        )
        self.action = None  # abbreviated action.
        self.action_forces = None  # minus gradient of abbreviated action
        self.neb_optimization_force = (
            None  # neb force for optimization of action with constraints at two ends.
        )
        self.neb_transverse_force = (
            None  # neb force for interior beads along transverse direction
        )

        self.instanton_path_energy = None  # energy E of instanton path in JWKB approximation. See: Section II. A in J. Chem. Phys. 148, 102334 (2018)

    def bind(self, ens: MAPNEBGPRMover):
        """
        :param: ens: A NEBMover instance.
        Copy beads, cell, forces of NEB mover to itself.
        """
        self.dbeads = ens.beads.copy()
        self.dcell = ens.cell.copy()
        self.fixatoms = ens.fixatoms.copy()

        self.instanton_path_energy = ens.optarrays[
            "instanton_path_energy"
        ]  # bind the instanton path energy from NEB mover.

        # Mask to exclude fixed atoms from 3N-arrays
        self.fixatoms_mask = np.ones(3 * ens.beads.natoms, dtype=bool)
        if len(ens.fixatoms) > 0:
            self.fixatoms_mask[3 * ens.fixatoms] = 0
            self.fixatoms_mask[3 * ens.fixatoms + 1] = 0
            self.fixatoms_mask[3 * ens.fixatoms + 2] = 0

        # Create reduced bead and force object (excluding the fixed atoms. But including the beads at two ends that also moves)
        self.rbeads = Beads(ens.beads.natoms, ens.beads.nbeads)
        self.rbeads.q[:] = ens.beads.q[:]

        self.spring_k = ens.optarrays[
            "spring_k"
        ]  # bind spring force spring_k from NEBMover.
        self.kappa = ens.optarrays[
            "kappa"
        ]  # bind end beads energy constraint constant kappa from NEBMover.

        # Option to vary the spring constant and have larger spring constant at two end point. (This is for the case we have the flat potential at end point)
        self.variable_spring_constant = ens.optarrays[
            "variable_spring_constant"
        ]  # bool variable to decide whether to increase the spring constant at two ends.
        self.VSC_E_ref = ens.optarrays[
            "VSC_E_ref"
        ]  # the reference energy (> instanton_path_energy), below reference energy, we increase the spring constant
        self.VSC_spring_k_max_ratio = ens.optarrays[
            "VSC_spring_k_max_ratio"
        ]  # the spring constant k at the end beads, also the maximum spring constant k when we vary the k.
        self.VSC_k_max = None
        self.VSC_k_ref = None

        if self.variable_spring_constant:
            # in case variable_spring_constant = True, we need to check whether VSC_E_ref & VSC_spring_k_max is provided
            assert (
                self.VSC_E_ref != 0.0
            ), "Must provide the value of reference energy (VSC_E_ref) when we vary the spring constant (variable_spring_constant = True)"
            assert (
                self.VSC_E_ref > self.instanton_path_energy
            ), "The reference energy (VSC_E_ref) must be larger than the energy of end beads of instanton path (instanton_path_energy) when we vary the spring constant (variable_spring_constant = True)"
            print("\n")
            print(
                "@Variable Spring Constant: the current ratio between k_max & k_min for variable spring constant is {}: ".format(
                    self.VSC_spring_k_max_ratio
                )
            )
            print(
                "We use linear interpolation to choose spring constant for beads whose energy is between VSC_E_ref and the energy of end beads."
            )
            print("\n")

            self.VSC_k_max = self.spring_k
            self.VSC_k_ref = self.VSC_k_max / self.VSC_spring_k_max_ratio

        self.energy_shift = ens.optarrays["energy_shift"]

        # bind the gpr model from NEBMover.
        self.gpr_model = ens.gpr_model
        self.coordinate_transformer = ens.coordinate_transformer
        self.ab_initio_pot = np.zeros([self.dbeads.nbeads])
        self.ab_initio_force = np.zeros([self.dbeads.nbeads, 3 * self.dbeads.natoms])

    def initialize_force(self, q):
        """
        initialize rbf & energy. This will enable us to use check_spring_k_kappa in the initialization() step of neb gm in MAPNEBGPRMover
        """
        self.rbeads.q[:, self.fixatoms_mask] = q

        # use Gaussian Process Regression to get the potential and forces for beads.
        self.beads_energy, beads_forces = self.get_gpr_potential_and_forces()

        # Forces for free moving dofs.
        self.rbf = beads_forces.copy()[:, self.fixatoms_mask]

    def __call__(self, mscaled_q):
        """Returns the projection for neb optimization.
        update reduced bead coordinates (&dbeads coordinate) (sticly speaking the free-moving atom parts) with x.
        :param: msacled_x: new mass scaled coordinates for updated freely moving particles.

        rbf: physical forces for reduced beads
        rbq: position for reduced beads
        btau: tangent vector directions.
        """
        # coordinate q.
        q = mscaled_q / np.sqrt(
            self.dbeads.m3[:, self.fixatoms_mask]
        )
        
        self.initialize_force(q)

        # mass scaled coordinate.
        self.mscaled_q = mscaled_q

        # mass weighted force
        mscaled_f = self.rbf / np.sqrt(
            self.dbeads.m3[:, self.fixatoms_mask]
        )  # 1/sqrt(m) * f: mass scaled force.
        self.mscaled_f = mscaled_f

        # Number of images
        nimage = self.dbeads.nbeads
        # Number of atoms that is free to move.
        natom = self.dbeads.natoms - len(self.fixatoms)

        self.spring_forces = np.zeros([nimage, 3 * natom])
        self.end_bead_energy_constraint_forces = np.zeros([2, 3 * natom])
        self.beads_mscaled_distance = npnorm(mscaled_q[1:] - mscaled_q[:-1], axis=1)

        # abbreviated action for the ring polymer instanton path.
        self.action = self.compute_neb_action(nimage, mscaled_q)

        # negative gradient of abbreviated action for each bead. We only compute it for the internal beads (excluding two ends)
        self.action_forces = self.compute_neb_action_force(
            nimage, natom, mscaled_q, mscaled_f
        )

        # compute direction of tangent vector, using either improved methods.
        btau = self.compute_tangent_vector(nimage, natom, mscaled_q)

        # evaluate the nudged elastic band optimization forces for perpendicular action forces and the spring force. (on mass scaled coordinate for free moving atoms.)
        neb_optimization_force = self.compute_neb_optimization_force(
            nimage, natom, btau, mscaled_q, mscaled_f
        )

        self.neb_optimization_force = np.copy(neb_optimization_force)

        neb_optimization_gradient = -neb_optimization_force
        self.neb_optimization_gradient = neb_optimization_gradient

        return self.action, neb_optimization_gradient

    def compute_tangent_vector(self, nimage, natom, mscaled_q):
        """
        we used the improved tangent direction:
        J. Chem. Phys. 113, 9978 (2000); https://doi.org/10.1063/1.1323224
        :param: nimage: number of replica images
        :param: natom: number of atoms (free moving)
        :param: bq: beads coordinate (mass_scaled coordinate)

        :return: btau: unit director for tangent vector of all internal beads in mass_scaled coordinates. (We do not need tangent vector for beads at two ends.)
        """
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
                maxpot = max(
                    abs(beads_energy[ii + 1] - beads_energy[ii]),
                    abs(beads_energy[ii - 1] - beads_energy[ii]),
                )
                minpot = min(
                    abs(beads_energy[ii + 1] - beads_energy[ii]),
                    abs(beads_energy[ii - 1] - beads_energy[ii]),
                )

                if beads_energy[ii + 1] >= beads_energy[ii - 1]:
                    btau[ii] = d2 * maxpot + d1 * minpot
                elif beads_energy[ii + 1] < beads_energy[ii - 1]:
                    btau[ii] = d2 * minpot + d1 * maxpot
            btau[ii] /= npnorm(btau[ii])

        return btau

    def compute_neb_action(self, nimage, mscaled_q):
        """
        compute abbreviated action W. See eq.(10) in J. Chem. Phys. 148, 102334 (2018)
        Note in atomic unit, hbar = kb = 1.

        :param: nimage: number of images (replicas)
        :param: mscaled_q: mass weighted coordinates for free moving atoms [nimag, 3 * natom]

        :return: action: abbreviated action of the ring polymer path
        """
        beads_energy = self.beads_energy

        action = 0

        # sqrt(2 (V - E))
        action_each_bead = np.zeros([nimage])
        for i in range(nimage):
            if beads_energy[i] < self.instanton_path_energy:
                action_each_bead[i] = 0
            else:
                action_each_bead[i] = np.sqrt(
                    2 * (beads_energy[i] - self.instanton_path_energy)
                )

        for j in range(1, nimage):
            rj = mscaled_q[j]
            rj_1 = mscaled_q[j - 1]
            r_dist = npnorm(rj - rj_1)
            action = (
                action
                + 1 / 2 * (action_each_bead[j] + action_each_bead[j - 1]) * r_dist
            )

        return action

    def compute_neb_action_force(self, nimage, natom, mscaled_q, mscaled_f):
        """
        compute the negative gradient of abbreviated action W. (for scaled coordinates.) See eq. (11) in J. Chem. Phys. 148, 102334 (2018).
        Note I will use the same symbol as given in the eq.(11) in the paper.

        :param: nimag: number of images (replica). scalar
        :param: natom: number of freely moving atoms. scalar
        :param: mscaled_q: mass weighted coordinates for free moving atoms. size: [nimag, 3 * natom]
        :param: mscaled_f: mass scaled forces for all beads. size: [nimag, 3 * natom]

        :return: action_force:  the negative gradient of abbreviated action W. (for scaled coordinates) size: [nimag, 3 * natom].
        """
        beads_energy = self.beads_energy
        bead_displs_vector = (
            mscaled_q[1:] - mscaled_q[:-1]
        )  # displacement vector of beads. [nbeads-1, 3 * natom]
        bead_distance = npnorm(
            bead_displs_vector, axis=1
        )  # |r_j - r_{j-1}|  [nbeads -1]
        bead_displs_unit_vector = np.transpose(
            np.transpose(bead_displs_vector) / bead_distance
        )  # unit vector for beads displacement vector [nbeads -1, 3* natom]

        # sqrt(2 (V - E))
        action_each_bead = np.zeros([nimage])
        for i in range(nimage):
            if beads_energy[i] < self.instanton_path_energy:
                action_each_bead[i] = 0
            else:
                action_each_bead[i] = np.sqrt(
                    2 * (beads_energy[i] - self.instanton_path_energy)
                )
        
        action_max = np.max(action_each_bead)
        action_force = np.zeros([nimage, 3 * natom])
        for j in range(1, nimage - 1):
            dj1 = bead_distance[j - 1]  # |r_{j} - r_{j-1}|.  d_{j}
            dj2 = bead_distance[j]  # |r_{j+1} - r_{j}|. d_{j+1}
            dj1_unit_vector = bead_displs_unit_vector[j - 1]  # \hat{d}_{j}
            dj2_unit_vector = bead_displs_unit_vector[j]  # \hat{d}_{j+1}
            fj = mscaled_f[j]

            if action_each_bead[j] == 0:
                # when energy of beads is smaller than the path energy.
                # we make the negative gradient for optimization along the gradient direction, to make bead moves to higher energy.
                gj_force_component = - 0.5 * (1/action_max) * (dj1 + dj2) * fj * 10
                
                warning("the energy of bead " + str(j) + " is  " + str(beads_energy[j]) + \
                         " which is smaller than the end bead energy we choose: " + \
                              str(self.instanton_path_energy) )
            else:
                gj_force_component = 0.5 * (1/action_each_bead[j] * (dj1 + dj2) * fj)

            gj_curvature_component = 0.5 * (
                -(action_each_bead[j] + action_each_bead[j - 1]) * dj1_unit_vector
                + (action_each_bead[j] + action_each_bead[j + 1]) * dj2_unit_vector
            )
            gj = gj_force_component + gj_curvature_component

            action_force[j] = gj

        return action_force

    def compute_spring_force(self, nimage, natom, mscaled_q, mscaled_f, btau):
        """ """
        beads_energy = self.beads_energy

        spring_k_list = np.zeros([nimage - 1])

        if not self.variable_spring_constant:
            # do not vary the spring constant for different beads.
            spring_k_list = np.ones([nimage - 1]) * self.spring_k
        else:
            end_beads_energy = self.instanton_path_energy
            k_change = self.VSC_k_max - self.VSC_k_ref
            E_ref = self.VSC_E_ref
            k_max = self.VSC_k_max

            for i in range(nimage - 1):
                if i != nimage - 2 and i != 0:
                    bead_energy_min = np.min(
                        [beads_energy[i], beads_energy[i + 1]]
                    )  # the minimum of the energy of two beads connected by spring.
                else:
                    # the end beads spring constant is chosen as k_max.
                    spring_k_list[i] = k_max
                    continue

                if bead_energy_min > E_ref:
                    spring_k_list[i] = self.VSC_k_ref
                else:
                    # make the spring constant k change linearly with energy.  The spring is more tight at lower energy.
                    spring_k_list[i] = k_max - k_change * (
                        bead_energy_min - end_beads_energy
                    ) / (E_ref - end_beads_energy)

        # spring forces for beads. Note the spring force at two ends are different from spring forces for internal beads.
        spring_force = np.zeros([nimage, 3 * natom])
        # spring force for internal beads
        for ii in range(1, nimage - 1):
            spring_force[ii] = (
                npnorm(mscaled_q[ii + 1] - mscaled_q[ii]) * spring_k_list[ii]
                - npnorm(mscaled_q[ii] - mscaled_q[ii - 1]) * spring_k_list[ii - 1]
            ) * btau[ii]

        # spring force for end bead 0
        unit_vec_1 = (mscaled_q[1] - mscaled_q[0]) / npnorm(
            mscaled_q[1] - mscaled_q[0]
        )  # unit vector for q[1] - q[0]
        spring_force_bead0 = (
            spring_k_list[0] * npnorm(mscaled_q[1] - mscaled_q[0]) * unit_vec_1
        )
        f0 = mscaled_f[0] / npnorm(mscaled_f[0])  # unit vector along force at beads: 0
        spring_force[0] = (
            spring_force_bead0 - np.dot(spring_force_bead0, f0) * f0
        )  # spring force component transverse to the gradient of potential.

        # spring force for end bead nimag - 1
        unit_vec_2 = (mscaled_q[nimage - 2] - mscaled_q[nimage - 1]) / npnorm(
            mscaled_q[nimage - 2] - mscaled_q[nimage - 1]
        )
        spring_force_bead1 = (
            spring_k_list[nimage - 2]
            * npnorm(mscaled_q[nimage - 2] - mscaled_q[nimage - 1])
            * unit_vec_2
        )
        f1 = mscaled_f[nimage - 1] / npnorm(
            mscaled_f[nimage - 1]
        )  # unit vector along force at beads: nimage - 1
        spring_force[nimage - 1] = (
            spring_force_bead1 - np.dot(spring_force_bead1, f1) * f1
        )  # spring force component transverse to the gradient of potential.

        return spring_force

    def compute_neb_optimization_force(self, nimage, natom, btau, mscaled_q, mscaled_f):
        """
        compute the optimization forces for nudged elastic band beads. See eq.(15 - 22) in J. Chem. Phys. 148, 102334 (2018).

        :param: nimag: number of images (replica). scalar
        :param: natom: number of freely moving atoms. scalar
        :param: btau: tangent vector for internal beads.  size: [nimag, 3 * natoms]
        :param: mscaled_q: mass weighted coordinates for free moving atoms. size: [nimag, 3 * natom]
        :param: mscaled_f: mass scaled forces for all beads. size: [nimag, 3 * natom]

        :return: optimization_force: the optimization force for nudged elastic band. size: [nimag, 3 * natom]
        """
        beads_energy = self.beads_energy

        # kappa: restraint force back to iso-energy contour.
        left_kappa = self.kappa["left"]  # kappa for the left end beads
        right_kappa = self.kappa["right"]  # kappa for the right end beads.

        neb_optimization_force = np.zeros([nimage, 3 * natom])
        self.neb_transverse_force = np.zeros([nimage, 3 * natom])

        spring_force = self.compute_spring_force(
            nimage, natom, mscaled_q, mscaled_f, btau
        )

        # end_beads_energy_constraint_force: force to draw end beads back to isoenergy contours.
        f0 = mscaled_f[0] / npnorm(mscaled_f[0])  # unit vector along force at beads: 0
        f1 = mscaled_f[nimage - 1] / npnorm(
            mscaled_f[nimage - 1]
        )  # unit vector along force at beads: nimage - 1
        end_beads_energy_constraint_force = np.zeros([2, 3 * natom])
        end_beads_energy_constraint_force[0] = (
            f0 * left_kappa * (beads_energy[0] - self.instanton_path_energy)
        )  # kappa * (V(r) - E) * \hat{f}(r) for beads 0
        end_beads_energy_constraint_force[1] = (
            f1 * right_kappa * (beads_energy[nimage - 1] - self.instanton_path_energy)
        )  # kappa * (V(r) - E) * \hat{f}(r) for beads n-1.

        self.spring_forces = spring_force  # store the spring force between beads
        self.end_bead_energy_constraint_forces = end_beads_energy_constraint_force  # store energy constraint force for end beads.

        # for internal beads, transverse force from negative gradient of action.
        for ii in range(1, nimage - 1):
            neb_optimization_force[ii] = (
                self.action_forces[ii]
                - np.dot(self.action_forces[ii], btau[ii]) * btau[ii]
            )

        self.neb_transverse_force = (
            neb_optimization_force  # transverse gradient for interior neb beads.
        )

        # add energy constraint force for two end beads.
        neb_optimization_force[0] = end_beads_energy_constraint_force[0]
        neb_optimization_force[nimage - 1] = end_beads_energy_constraint_force[1]

        # add spring force for all beads.
        neb_optimization_force = neb_optimization_force + spring_force

        return neb_optimization_force

    def get_gpr_potential_and_forces(self):
        """
        Get potential and forces for all beads using the Gaussian Process Regression model.
        When there is ab-initio potential and force available, we use the ab initio value.
        return: potential for all beads.
        """
        test_x = np.copy(self.rbeads.q)
        beads_potential_shift, beads_potential_grad_x, _, _ = (
            self.gpr_model.predict_latent_function(test_x)
        )

        beads_forces = -beads_potential_grad_x
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


class RP_MAP(object):
    """
    Generate Ring polymer for Minimum Action Path (MAP) obtained by NEB method.
    Evolve dynamics of particle on inverted potential along minimum action path.
    The period T of periodic motion (here 2 * total_time for travel from one end to another end) gives beta * hbar (temperature).
    The Ring-polymer for instanton is chosen as evenly spaced beads in time.
    Attributes:

    """

    def __init__(self):
        """
        Initializatioin of RP_MAP
        """
        self.imag_time_period = 0
        self.instanton_temp = 0

        self.skip_neb_mode_bool = False

    def bind(self, nebmover: MAPNEBGPRMover):
        """
        bind function for RP_MAP
        nebmover: MAPNEBMover instance.
        """
        self.prefix = nebmover.options["prefix"]
        self.final_hessian_bool = nebmover.options["final_hessian_bool"]
        self.ab_initio_hessian_bool = nebmover.options["ab_initio_hessian_bool"]
        self.test_gpr_model_along_instanton_path = nebmover.options["test_gpr_model_along_instanton_path"]

        self.energy_shift = nebmover.optarrays["energy_shift"]
        self.output_maker = nebmover.output_maker

        self.neb_beads = nebmover.beads.copy()
        self.dcell = nebmover.cell.copy()
        self.fixatoms = nebmover.fixatoms.copy()

        self.time_step = nebmover.optarrays[
            "instanton_time_step"
        ]  # time step for dynamics along instanton path.
        self.instanton_path_energy = nebmover.optarrays["instanton_path_energy"]

        # Mask to exclude fixed atoms from 3N-arrays
        self.fixatoms_mask = np.ones(3 * nebmover.beads.natoms, dtype=bool)
        if len(nebmover.fixatoms) > 0:
            self.fixatoms_mask[3 * nebmover.fixatoms] = 0
            self.fixatoms_mask[3 * nebmover.fixatoms + 1] = 0
            self.fixatoms_mask[3 * nebmover.fixatoms + 2] = 0

        # ring polymer beads.
        self.rp_bead_number = nebmover.optarrays[
            "instanton_bead_number"
        ]  # bead number for instanton ring polymer
        self.rp_beads = Beads(
            self.neb_beads.natoms, self.rp_bead_number
        )  # bead object for instanton ring polymer
        self.rp_forces = nebmover.forces.copy(self.rp_beads, self.dcell)
        self.rp_hessian = np.eye(0, 0, 0, float)

        self.m3 = np.copy(dstrip(nebmover.beads.m3[0]))  # mass of atoms.

        self.gpr_model = nebmover.gpr_model
        self.coordinate_transformer = nebmover.coordinate_transformer

        # bind the gpr kernel condition
        self.gpr_SE_kernel_number = nebmover.options["gpr_SE_kernel_number"]
        self.gpr_kernel_outputscale = nebmover.optarrays["gpr_kernel_outputscale"]
        self.gpr_kernel_lengthscale_ratio = nebmover.optarrays[
            "gpr_kernel_lengthscale_ratio"
        ]
        self.gpr_noise_std = nebmover.optarrays["gpr_noise_std"]

        # bind the error criterion of the gpr model
        self.gpr_relative_force_error_criterion = nebmover.optarrays[
            "gpr_relative_force_error_criterion"
        ]
        self.gpr_absolute_force_error_criterion = nebmover.optarrays[
            "gpr_absolute_force_error_criterion"
        ]

        self.gpr_fix_internal_dofs_bool = nebmover.options["gpr_fix_internal_dofs_bool"]
        self.gpr_fix_internal_dofs_cutoff = nebmover.options["gpr_fix_internal_dofs_cutoff"]

        # bind the distance cutoff for training data for the gpr model
        self.distance_cutoff_for_training_data = (
            nebmover.options["distance_cutoff_for_training_data"]
        )

        # bind the file that we use to read hessian data
        self.read_gpr_hessian_folder = nebmover.options["read_gpr_hessian_folder"]

        # options about which ab-initio data point we will add to existing training data
        self.add_new_hessian_data_bool = nebmover.options["add_new_hessian_data_bool"]
        self.candidate_hessian_data_number = nebmover.options[
            "candidate_hessian_data_number"
        ]
        self.new_hessian_data_index = nebmover.optarrays["new_hessian_data_index"]
        self.train_hessian_model_bool = nebmover.options["train_hessian_model_bool"]

    def initialize(self, neb_beads, neb_final_step):
        """
        initialize the RP_MAP dynamics. This should be called after beads have converged to minimum action path using line integral nudged elastic band method.
        :param: neb_beads: beads in MAPNEBMover, with optimized geometry for Minimum Action Path.
        :param: neb_forces: LINEBGradientMapper.rforces object.
        :param: step: final step in MAPNEBMover simulation. (Used for output of instanton geometry.)
        """
        self.neb_beads.q[:] = neb_beads.q[:]  # initialize neb beads position.

        # Cubic interpolation of neb beads to enable accurate dynamics evolution.
        self.cubic_spline = ipi.utils.nebinstool.path_cubic_spline_function(
            np.copy(self.neb_beads.q)
        )

        print("use cubic interpolation to generate MAP path")

        self.final_step = neb_final_step

        if self.skip_neb_mode_bool:
            start_time = timer()
            
            self.construct_gpr_model_use_training_data_end_of_neb_stage()
            
            end_time = timer() 
            time_elapsed = (end_time - start_time) / 60
            print(f"the time used for construct \
                  gpr model to predict force along instanton path is: {time_elapsed}")
            pass

    def classical_dynamics_along_MAP(self):
        """
        classical dynamics on the inverted potential -V(x)
        the final time will be 1/2 of the imaginary period.
        :return:  t_list: a list of time of trajectories.
                  v_list: a list of velocity of trajectories.
                  x_list: a list of coordinate of trajectories.
        """
        start_time = timer()
        
        t, r_distance = 0, 0  # time & normalized distance along path.
        x = np.copy(self.neb_beads.q[0])  # coordinate
        v = np.zeros([3 * self.neb_beads.natoms])  # velocity
        v_r = 0  # dr/dt. rate of change for r.

        shifted_V, _, _, _ = self.gpr_model.predict_latent_function(np.array([x]))
        pot = shifted_V[0] + self.energy_shift

        x_list = [x]
        v_list = [v]
        t_list = [t]
        r_list = [r_distance]
        v_r_list = [v_r]
        pot_list = [pot]

        dr = 1000

        while dr > 0:
            old_r_distance = r_distance
            # r is normalized distance along path, in the range of [0, 1]
            t, r_distance, v_r, x, v = self.classical_dynamics_step(t, r_distance, v_r)

            dr = r_distance - old_r_distance
            # check energy conservation
            shifted_V, _, _, _ = self.gpr_model.predict_latent_function(np.array([x]))
            pot = shifted_V[0] + self.energy_shift

            x_list.append(x)
            v_list.append(v)
            t_list.append(t)
            r_list.append(r_distance)
            v_r_list.append(v_r)
            pot_list.append(pot)

        x_list = np.array(x_list)
        v_list = np.array(v_list)
        t_list = np.array(t_list)
        r_list = np.array(r_list)
        v_r_list = np.array(v_r_list)
        pot_list = np.array(pot_list)

        self.analyze_classical_dynamics_along_MAP(v_list, t_list, pot_list)

        end_time = timer()
        time_elapsed = (
            end_time - start_time
        ) / 60  # time elapsed in minutes
        print("the running time for the constrained dynamics along the path is: " + str(time_elapsed) + " min.")

        return t_list, v_list, x_list

    def classical_dynamics_step(self, t, r_distance, v_r):
        """
        evolve dynamics for one time step with dt = self.time_step
        :param:  t: time
                 r: normalized cumulative distance along the path.
                 v_r: rate of change for r.

        :return: t: new time.
                 r: new normalized cumulative distance along the path.
                 v_r: new rate of change for r.
                 x: new coordinate.
                 v: new velocity.
        """
        # parameter for Runge Kutta 4th order algorithm
        m3_matrix = np.diag(self.m3)
        param = [self.gpr_model, m3_matrix, self.cubic_spline]
        dt = self.time_step

        y = np.array([r_distance, v_r])

        new_y = RK4(y, t, ipi.utils.nebinstgprtool.dydt_inverted_pot_gpr, param, dt)
        r_distance = new_y[0]
        v_r = new_y[1]

        t = t + dt
        x = self.cubic_spline(r_distance)
        v = self.cubic_spline(r_distance, nu=1) * v_r

        return t, r_distance, v_r, x, v

    def analyze_classical_dynamics_along_MAP(self, v_list, t_list, pot_list):
        """
        compute the temperature of the instanton path from period of motion.
        Optional: monitor the potential, total energy & kinetic energy.
        """
        # compute the kinetic energy & total energy. check energy conservation.
        pot_list = pot_list - self.energy_shift

        kinetic_energy_list = 0.5 * np.sum(
            np.array(self.m3) * np.power(v_list, 2), axis=1
        )
        total_energy_list = kinetic_energy_list - pot_list  # total_E = K -V.
        pot_list = units.unit_to_user(
            "energy", "electronvolt", pot_list
        )  # convert to eV unit.
        total_energy_list = units.unit_to_user(
            "energy", "electronvolt", total_energy_list
        )
        kinetic_energy_list = units.unit_to_user(
            "energy", "electronvolt", kinetic_energy_list
        )

        # compute the temperature from imaginary time.
        self.imag_time_period = (
            2 * t_list[-1]
        )  # the period of periodic motion is twice the time move from one end to another end.
        self.instanton_temp = 1 / self.imag_time_period

        info(
            "finish evolution of dynamics along Minimum Action path, the period of motion is: {}".format(
                self.imag_time_period
            )
        )

        # print temperature
        temp_kelvin = units.unit_to_user(
            "temperature", "kelvin", self.instanton_temp
        )  # temperature in "kelvin" unit

        print("temperature for instanton path : {} K".format(temp_kelvin))

        # output temperature to a separate file
        file_name = "instanton_temperature.txt"
        with open(file_name, "w") as f:
            f.write("temperature for instanton path : (K) \n")
            f.write(str(temp_kelvin) + "\n")

    def construct_gpr_model_use_training_data_end_of_neb_stage(self):
        """
        construct gpr model by reading training data from file that stored at the end of 'neb' stage.
        """
        neb_final_gpr_folder = "neb_final_gpr_training"
        cartesian_coordinate_x, training_V, training_forces = (
            ipi.utils.nebinstgprtool.read_training_data(prefix=neb_final_gpr_folder)
        )
        training_V_shifted = training_V - self.energy_shift
        training_grad = -training_forces

        # initialize GPR model with training data read from the end of 'neb' stage run.
        self.gpr_model = ipi.utils.gprtools.GPModelWithDerivativesWrapper(
            cartesian_coordinate_x,
            training_V_shifted,
            training_grad,
            self.rp_beads.natoms,
            self.coordinate_transformer,
            gpr_SE_kernel_number=self.gpr_SE_kernel_number,
            kernel_outputscale=self.gpr_kernel_outputscale,
            kernel_lengthscale_ratio=self.gpr_kernel_lengthscale_ratio,
            noise_std=self.gpr_noise_std,
            train_bool= False,
            gpr_fix_internal_dofs_bool= self.gpr_fix_internal_dofs_bool,
            gpr_fix_internal_dofs_cutoff= self.gpr_fix_internal_dofs_cutoff 
        )

        model_hyperparameter_exists = ipi.utils.nebinstgprtool.load_training_hyperparameter_in_gpr_model(
            self.gpr_model, neb_final_gpr_folder
        )

        if not model_hyperparameter_exists:
            # train the model and store the hyper-parameter
            self.gpr_model.train_gpr()
            ipi.utils.nebinstgprtool.store_training_hyperparameter_in_gpr_model(
                self.gpr_model, neb_final_gpr_folder
            )
        

    def interpolate_ring_polymer_beads(self, t_list, v_list, x_list):
        """
        interpolate ring polymer beads from the imaginary time trajectory along Minimum Action Path (MAP).
        t_list , v_list, x_list: list of time / velocity / trajectory from MD simulation along path.
        """
        # interpolate to get ring polymer position.
        rp_t_list, rp_x_list = ipi.utils.nebinstool.interpolate_ring_polymer_beads(
            self.imag_time_period, t_list, x_list, v_list, self.rp_bead_number
        )

        ipi.utils.nebinstool.print_instanton_rp_time(
            "rp_time_FINAL", self.imag_time_period, rp_t_list, self.output_maker
        )

        self.rp_beads.q = rp_x_list

        # print ring polymer instanton geometry.
        shifted_pots, _, _, _ = self.gpr_model.predict_latent_function(rp_x_list)

        self.rp_beads_pots = shifted_pots + self.energy_shift

        ipi.utils.nebinstool.print_neb_instanton_geo(
            "instanton_along_MAP_FINAL",
            self.final_step,
            self.rp_beads.nbeads,
            self.rp_beads.natoms,
            self.neb_beads.names,
            self.rp_beads.q,
            self.rp_beads_pots,
            self.dcell,
            self.energy_shift,
            self.output_maker,
        )

    def compute_ring_polymer_hessian(self):
        """
        compute hessian of ring polymer
        """
        if self.final_hessian_bool:
            rp_beads_q = np.copy(self.rp_beads.q)
            # compute final hessians
            self.rp_hessian = ipi.utils.nebinstool.get_hessian(
                self.rp_beads,
                self.rp_forces,
                rp_beads_q,
                self.rp_beads.natoms,
                self.rp_beads.nbeads,
                self.fixatoms,
            )

            self.rp_beads.q = rp_beads_q

    def optimize_GPR_model_for_dynamics_evolution(self):
        """
        Add training data to GPR model and optimize hyper-parameters to make sure the GPR model generate accurate force along LI-NEB path to have correct temperature
        This is achieved by cross-validation technique:
        (1) Generate 50 data points along the LI-NEB path using cubic interpolation
        (2) Randomly choose 10 data points from them as test data
        (3) Use GPR model to predict the force, compute the force error
        (4) Add more training data into the training set until the force error is small on testing data:
            either relative force error is small or absolute force error is small.
            The selection of the training data is based on the force uncertainty.
        """
        total_data_set_number = 50
        testing_data_number = 10
        # we interpolate the converged LI-NEB path with N + 2 data point, where 2 end data point is already in the training set of GPR model (as they are end points for LI-NEB path)
        LINEB_path_x, _ = ipi.utils.nebinstool.path_cubic_interpolation(
            self.neb_beads.q, total_data_set_number + 2
        )
        LINEB_path_x = LINEB_path_x[1:-1]

        index_list = np.arange(total_data_set_number)
        np.random.shuffle(
            index_list
        )  # shuffle the index to get test data and training data.

        test_x = LINEB_path_x[index_list[:testing_data_number]]
        unused_train_x = LINEB_path_x[index_list[testing_data_number:]]

        # create beads and forces object for testing and training data set
        test_beads = Beads(
            self.neb_beads.natoms, testing_data_number
        )  # bead object for instanton ring polymer
        test_forces = self.rp_forces.copy(test_beads, self.dcell)

        train_beads = Beads(self.neb_beads.natoms, 1)
        train_forces = self.rp_forces.copy(train_beads, self.dcell)

        # compute the ab-initio forces at the test location:
        test_beads.q[:] = test_x
        ab_initio_test_data_f = dstrip(test_forces.f).copy()
        ab_initio_test_data_f_magnitude = np.linalg.norm(
            ab_initio_test_data_f, axis=1
        )  # magnitude of the ab initio force

        # make predictions of forces using GPR model.
        _, gpr_test_grad, _, _ = self.gpr_model.predict_latent_function(test_x)
        gpr_test_f = -gpr_test_grad

        test_f_diff = gpr_test_f - ab_initio_test_data_f
        test_f_diff_magnitude = np.linalg.norm(
            test_f_diff, axis=1
        )  # absolute error on the test force
        test_f_diff_magnitude_ratio = (
            test_f_diff_magnitude / ab_initio_test_data_f_magnitude
        )  # relative error on the test force.

        pass_test_bool = False

        while not pass_test_bool:
            # see if the prediction on the test data is satisfactory:
            pass_test_bool = True
            for i in range(testing_data_number):
                if (
                    test_f_diff_magnitude_ratio[i]
                    > self.gpr_relative_force_error_criterion
                    and test_f_diff_magnitude[i]
                    > self.gpr_absolute_force_error_criterion
                ):
                    pass_test_bool = False
                    break

            if pass_test_bool:
                break

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

            if len(unused_train_x) == 0:
                # all training data set has been used but no satisfactory result achieved on test data.
                print("\n")
                print(
                    "@WARNING: After adding all training data into the gpr model, the prediction of force on unseen test data is still unsatisfactory"
                )
                print("\n")
                break

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
                self.distance_cutoff_for_training_data,
            )

            # update the test error on the unseen test data set
            _, gpr_test_grad, _, _ = self.gpr_model.predict_latent_function(test_x)
            gpr_test_f = -gpr_test_grad

            test_f_diff = gpr_test_f - ab_initio_test_data_f
            test_f_diff_magnitude = np.linalg.norm(
                test_f_diff, axis=1
            )  # absolute error on the test force
            test_f_diff_magnitude_ratio = (
                test_f_diff_magnitude / ab_initio_test_data_f_magnitude
            )  # relative error on the test force.

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

        # update the data in the gpr training data folder.
        train_x = self.gpr_model.train_cartesian_inputs
        train_V = self.gpr_model.train_cartesian_targets[:, 0]
        train_V_to_store = train_V + self.energy_shift
        train_grad = self.gpr_model.train_cartesian_targets[:, 1:]
        train_f_to_store = -train_grad

        ipi.utils.nebinstgprtool.store_training_data(
            train_x, train_V_to_store, train_f_to_store, prefix="neb_final_gpr_training"
        )

    def construct_gpr_hessian_model(self):
        """
        construct the gpr_hessian model, which will predict hessian information using Gaussian Process Regression.
        """
        start_time = timer()
        if self.read_gpr_hessian_folder == "None":
            # create gpr_hessian model using data from gpr model
            print(
                "read_gpr_hessian_folder not provided. Will create gpr_hessian model from training data in gpr model."
            )
            cartesian_coordinate_x = np.copy(self.gpr_model.train_cartesian_inputs)
            training_V_shifted = np.copy(self.gpr_model.train_cartesian_targets[:, 0])
            training_grads = np.copy(self.gpr_model.train_cartesian_targets[:, 1:])

            if not self.add_new_hessian_data_bool:
                raise (
                "Error. You must provide hessian data for gpr_hessian training. \
                  Either add new hessian data (add_new_hessian_data_bool= True) or read hessian data \
                  from read_gpr_hessian_folder"
            )

            if len(self.new_hessian_data_index) == 0:
                raise("Must provide the index of new hessian data point if add_new_hessian_data_bool = True")

            candidate_hessian_point_x, _ = (
                ipi.utils.nebinstool.path_equal_distance_interpolation(
                    np.copy(self.neb_beads.q), self.candidate_hessian_data_number
                )
            )
            
            # use the first data point as the reference point for mean function 
            # when constructing gpr model with hessian
            first_hessian_data_x = candidate_hessian_point_x[self.new_hessian_data_index[0]]
            
            new_beads = Beads(self.neb_beads.natoms, 1)
            new_forces = self.rp_forces.copy(new_beads, self.dcell)
            new_beads.q[0] = first_hessian_data_x
            
            ref_V_shifted = dstrip(new_forces.pots).copy() - self.energy_shift
            ref_grads = -dstrip(new_forces.f).copy()[0] 
            # only 1 bead, so no need to transform the hessian.
            ref_hessians = ipi.utils.nebinstool.get_hessian(
                new_beads,
                new_forces,
                np.copy(new_beads.q),
                self.neb_beads.natoms, 
                1,
                self.fixatoms 
            )

            # initially no hessian training data.
            hessian_data_list = np.array([])
            hessian_index_list = np.array([])

            # construct gpr hessian model. 
            # We have to train it here. First train with only potential and gradient data.
            self.gpr_hessian_model = (
                ipi.utils.gpr_hessian_tools.GPModelWithHessiansWrapper(
                    cartesian_coordinate_x,
                    training_V_shifted,
                    training_grads,
                    hessian_data_list,
                    hessian_index_list,
                    self.rp_beads.natoms,
                    self.coordinate_transformer,
                    self.gpr_SE_kernel_number,
                    self.gpr_kernel_outputscale,
                    self.gpr_kernel_lengthscale_ratio,
                    self.gpr_noise_std,
                    constant_mean_func_bool= False,
                    ref_mean_x= first_hessian_data_x,
                    ref_mean_V= ref_V_shifted,
                    ref_mean_grad_x= ref_grads,
                    ref_mean_hessian_x= ref_hessians,
                    train_bool= True,
                    gpr_fix_internal_dofs_bool= self.gpr_fix_internal_dofs_bool,
                    gpr_fix_internal_dofs_cutoff= self.gpr_fix_internal_dofs_cutoff
                )
            )

            # After train the model with only potential and gradient,
            # the hyper-parameter should be close to the minimum point after adding hessian data.
            # Now add hessian data & re-train the model.
            new_pots = ref_V_shifted + self.energy_shift
            new_grads = np.array([ref_grads])
            new_hessians = np.array([ref_hessians])
            new_hessian_point_x = np.array([first_hessian_data_x])
            ipi.utils.nebinstgprtool.add_hessian_data_to_model(
                self.gpr_hessian_model,
                new_hessian_point_x,
                new_pots,
                new_grads,
                new_hessians,
                self.energy_shift,
                retrain_bool= self.train_hessian_model_bool
            )

        else:
            print(
                "read_gpr_hessian_folder provided. Will read potential & gradients & hessians from folder and create gpr_hessian model."
            )

            # create gpr_hessian model using data read from read_gpr_hessian_folder
            (
                cartesian_coordinate_x,
                training_V,
                training_forces,
                hessian_index_list,
                hessian_data_list,
            ) = ipi.utils.nebinstgprtool.read_training_data_with_hessian(
                self.read_gpr_hessian_folder
            )

            training_V_shifted = training_V - self.energy_shift
            training_grads = -training_forces
            # choose the first data point with hessian information as the reference point for mean function.
            ref_x = cartesian_coordinate_x[hessian_index_list[0]]
            ref_V_shifted = np.array([training_V_shifted[hessian_index_list[0]]])
            ref_grads = training_grads[hessian_index_list[0]]
            ref_hessians = hessian_data_list[0]

            self.gpr_hessian_model = (
                ipi.utils.gpr_hessian_tools.GPModelWithHessiansWrapper(
                    cartesian_coordinate_x,
                    training_V_shifted,
                    training_grads,
                    hessian_data_list,
                    hessian_index_list,
                    self.rp_beads.natoms,
                    self.coordinate_transformer,
                    self.gpr_SE_kernel_number,
                    self.gpr_kernel_outputscale,
                    self.gpr_kernel_lengthscale_ratio,
                    self.gpr_noise_std,
                    constant_mean_func_bool=False,
                    ref_mean_x=ref_x,
                    ref_mean_V=ref_V_shifted,
                    ref_mean_grad_x=ref_grads,
                    ref_mean_hessian_x=ref_hessians,
                    train_bool= False,
                    gpr_fix_internal_dofs_bool= self.gpr_fix_internal_dofs_bool,
                    gpr_fix_internal_dofs_cutoff= self.gpr_fix_internal_dofs_cutoff
                )
            )

            model_hyperparameter_exists = \
                ipi.utils.nebinstgprtool.load_training_hyperparameter_for_gpr_hessian_model(
                    self.gpr_hessian_model,
                    self.read_gpr_hessian_folder
            )

            if model_hyperparameter_exists:
                # the hyper-parameter of the gpr hessian model exists.
                # we do not have to train it.
                # however, if desired (setting train_hessian_model_bool == True), we can train it.
                if (not self.add_new_hessian_data_bool) and self.train_hessian_model_bool:
                    self.gpr_hessian_model.train_model() 
                    ipi.utils.nebinstgprtool.store_training_hyperparameter_in_gpr_hessian_model(
                        self.gpr_hessian_model, self.read_gpr_hessian_folder
                    )
            else:
                # the hyper-parameter of the gpr hessian model does not exist. Must train the model.
                raise(RuntimeError, "The model hyper-parameter gpr_hessian.pth does not exist. Optimizing hessian data directly\
                      without pre-trained hyper-parameter can be inefficient. Please make sure to provide pre-trained hyper-parameter")
        
        end_time = timer()
        time_elapsed = (end_time - start_time) / 60
        print(f"time elapsed for training hessian model is: {time_elapsed} min." )

    def add_new_hessian_data(self):
        """
        (1) compute the new ab initio hessian at new_hessian_data_index.
        (2) Add new hessian data into gpr_hessian_model
        (3) store the updated data set into new folder.
        """
        self.data_destination_folder = self.read_gpr_hessian_folder

        if (
            not self.add_new_hessian_data_bool
        ) and self.read_gpr_hessian_folder == "None":
            raise (
                "Error. You must provide hessian data for gpr_hessian training. \
                  Either add new hessian data (add_new_hessian_data_bool= True) or read hessian data \
                  from read_gpr_hessian_folder"
            )

        if self.add_new_hessian_data_bool:
            # find the location of data point we can compute hessian & the index of data point that we have already computed hessians.
            if self.read_gpr_hessian_folder == "None":
                if len(self.new_hessian_data_index) == 0:
                    raise("Must provide the index of new hessian data point if add_new_hessian_data_bool= True")

                candidate_hessian_point_x, _ = (
                    ipi.utils.nebinstool.path_equal_distance_interpolation(
                        np.copy(self.neb_beads.q), self.candidate_hessian_data_number
                    )
                )
                # index of hessian data that is already computed among candidate data point list.
                self.hessian_index_in_candidate_list = np.array([])

                # the first index of new hessian data index is already used when constructing the model.
                self.hessian_index_in_candidate_list = np.array([self.new_hessian_data_index[0]])
                self.new_hessian_data_index = self.new_hessian_data_index[1:]

            else:
                # read candidate_hessian_point_x, hessian_index_in_candidate_list from self.read_gpr_hessian_folder.
                (candidate_hessian_point_x, self.hessian_index_in_candidate_list) = (
                    ipi.utils.nebinstgprtool.read_candidate_hessian_data_coordinate(
                        self.read_gpr_hessian_folder
                    )
                )

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
                new_forces = self.rp_forces.copy(new_beads, self.dcell)
                new_beads.q = new_hessian_point_x

                new_pots = new_forces.pots
                new_grads = -dstrip(new_forces.f).copy()

                # compute ab initio hessians of new data points.
                new_hessians = ipi.utils.nebinstool.get_hessian(
                    new_beads,
                    new_forces,
                    np.copy(new_beads.q),
                    natoms,
                    new_hessian_data_num,
                    self.fixatoms,
                )

                new_hessians = np.transpose(
                    np.reshape(
                        new_hessians, [3 * natoms, new_hessian_data_num, 3 * natoms]
                    ),
                    (1, 0, 2),
                )

                # add new hessian data (+ pot & gradients) into gpr_hessian model.
                if self.train_hessian_model_bool:
                    retrain_bool = True 
                else:
                    retrain_bool = False

                if retrain_bool:
                    start_t = timer()
                ipi.utils.nebinstgprtool.add_hessian_data_to_model(
                    self.gpr_hessian_model,
                    new_hessian_point_x,
                    new_pots,
                    new_grads,
                    new_hessians,
                    self.energy_shift,
                    retrain_bool= retrain_bool,
                )
                if retrain_bool:
                    end_t = timer()
                    time_elapsed = (end_t - start_t) / 60
                    print(f"the elapsed time for re-training the model is {time_elapsed} min.")

            # create a new data folder with up to date potential, gradient & hessian data.
            # the newly computed hessian will also be stored in this file.
            self.data_destination_folder = (
                ipi.utils.nebinstgprtool.store_training_data_in_gpr_hessian_model(
                    self.gpr_hessian_model, self.energy_shift
                )
            )

            # store the hyper-parameter of the gpr model in the data folder.
            ipi.utils.nebinstgprtool.store_training_hyperparameter_in_gpr_hessian_model(
                self.gpr_hessian_model, self.data_destination_folder
            )

            # store the coordinate of candidate data point for hessian calculation
            # & current index among candidate points that we have already computed hessian.
            self.hessian_index_in_candidate_list = np.concatenate(
                [self.hessian_index_in_candidate_list, self.new_hessian_data_index]
            )

            # store candidate_hessian_point_x, hessian_index_in_candidate_list in data destination folder.
            ipi.utils.nebinstgprtool.store_candidate_hessian_data_coordinate(
                candidate_hessian_point_x,
                self.hessian_index_in_candidate_list,
                self.data_destination_folder,
            )




    def predict_ring_polymer_hessians_using_gpr(self):
        """
        predict hessians of all ring polymer beads using Gaussian Process regression model. (self.gpr_hessian)
        """
        coord = np.copy(self.rp_beads.q)
        nbeads = self.rp_beads.nbeads
        natoms = self.rp_beads.natoms

        hessian_data_point_index = np.arange(nbeads)

        # use Gaussian Process Regression model to predict potentials, gradients and hessians
        pots, grads, hessians, _, _, _ = self.gpr_hessian_model.predict_latent_function(
            coord, hessian_data_point_index, internal_coordinate_bool=False
        )

        # reshape hessians ([nbeads, 3 * natoms, 3 * natoms])
        # to fit the shape of self.rp_hessian: [3 * natoms, nbeads * 3 * natoms]
        self.rp_hessian = np.reshape(
            np.transpose(hessians, (1, 0, 2)), [3 * natoms, nbeads * 3 * natoms]
        )

        # store computed hessians.
        prefix = os.path.join(
            self.data_destination_folder, "nbeads=" + str(int(nbeads))
        )
        ipi.utils.nebinstool.print_instanton_hess(
            prefix, self.rp_hessian, self.output_maker
        )

    def generate_ring_polymer_beads(self, neb_beads, neb_final_step):
        """
        Main function that compute ring-polymer beads from nudged elastic band Minimum action path.
        """
        self.initialize(neb_beads, neb_final_step)

        # optimize GPR model to make sure it will give accurate force for dynamics.
        # Separate point as test set and training set, add training set until the generalization error is small.
        if self.test_gpr_model_along_instanton_path:
            self.optimize_GPR_model_for_dynamics_evolution()

        # start classical dynamics along minimum action path (MEP) on the inverted potential.
        t_list, v_list, x_list = self.classical_dynamics_along_MAP()

        # interpolate the ring polymer beads from the generated trajectory.
        self.interpolate_ring_polymer_beads(t_list, v_list, x_list)

        # code to test the GPModelWithHessianWrapper
        if self.final_hessian_bool:
            if self.ab_initio_hessian_bool:
                # compute ab initio hessian of all ring polymers.
                # Only use this option for benchmark.
                self.compute_ring_polymer_hessian()

                x = self.rp_beads.q
                pots = self.rp_forces.pots
                grads = -np.copy(dstrip(self.rp_forces.f))
                hessians = self.rp_hessian
                ipi.utils.nebinstgprtool.test_gpr_hessian_prediction(
                    self.gpr_model, self.energy_shift, x, pots, grads, hessians,
                    self.gpr_fix_internal_dofs_bool, self.gpr_fix_internal_dofs_cutoff
                )

            else:
                # create gpr hessian model either reading data from input file or using training data from gpr model.
                self.construct_gpr_hessian_model()

                # add new hessian data into GPR model.
                # the location of new hessian data is given by self.new_hessian_data_point_index.
                # candidate_hessian_point_x spaced with equal distance along the path.
                self.add_new_hessian_data()

                # predict hessians of ring polymer beads using Gaussian Process Regression.
                # The result is stored in self.rp_hessians, which will be stored in RESTART file for post-processing.
                self.predict_ring_polymer_hessians_using_gpr()
