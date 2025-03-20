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
import ipi.utils.internalcoordtools  # 1/|ri-rj| : Coloumb matrix.
import ipi.utils.internal.internaltools  # primitive internal coordinate.
import ipi.utils.gprtools
import ipi.utils.nebinstgprtool
import ipi.utils.nebinstool
import ipi.utils.gpr_hessian_tools
import ipi.utils.mintools
import os
from timeit import default_timer as timer
import threading 
import ipi.utils.hessfasttools

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
        fix_dofs= np.array([]),
        asr = "none",
        mode="verlet",
        prefix="neb_instanton",
        tolerances={"gradient": 5e-3, 
                    "gradient_end_bead": 1e-2,
                    "action_forces_sum": 5e-3,
                    "action": 1e-4},
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
        opt = "neb",
        instanton_bead_number=20,
        instanton_path_energy=0.00,
        instanton_temperature=1.0,
        instanton_bead_q=np.zeros(0, float),
        instanton_bead_pot=np.zeros(0, float),
        instanton_hessian=np.eye(0, 0, 0, float),
        neb_inner_loop_step_max = 100,
        spring_k=0.1,
        kappa={"left": 50, "right": 50},
        ENO_order = 3, 
        dynamical_adjust_ratio= {"spring_k": 0.1,
                                 "kappa": 0.2},
        end_bead_energy_converge_value = 1e-3,
        alt_out=5,
        gpr_relative_force_error_criterion=0.05,
        gpr_absolute_force_error_criterion=0.002,
        gpr_force_uncertainty_criterion = 0.001,
        gpr_trust_region=0.1,
        minimum_trust_region= 0.1,
        distance_cutoff_for_training_data= 0.1,
        gpr_kernel_outputscale=np.zeros(0, float),
        gpr_kernel_lengthscale_ratio=np.zeros(0, float),
        gpr_noise_std={
            "pot_noise_prior": 1e-6,
            "force_noise_prior": 1e-4,
            "hessian_noise_prior": 1e-3,
        },
        gpr_SE_kernel_number=1,
        gpr_fix_internal_dofs_bool= True,
        gpr_fix_internal_dofs_cutoff= 1e-3,
        gpr_rigid_internal_dofs_cutoff= 1e-2,
        read_initial_gpr_training_data=False,
        test_gpr_model_along_instanton_path= False,
        final_hessian_bool=False,
        ab_initio_hessian_bool=False,
        read_gpr_hessian_folder="None",
        train_grad_model_bool= True, 
        train_hessian_model_bool= True, 
        add_new_hessian_data_bool= False,
        candidate_hessian_data_number=20,
        new_hessian_data_index=np.zeros(0, int),
        add_new_grad_data_bool= False,
        candidate_grad_data_number= 100,
        new_grad_data_index= np.zeros(0, int),
        selective_hessian_bool= False,
        new_hessian_data_index_rigid_mode= np.zeros(0, int),
        internal_coord = "bond",
        cross_validation_bool= False,
        ridge_regularization_alpha = {
            "force": 0.1,
            "hessian": 0.5
        },
        gpr_covar_inverse_nugget= 1e-8
    ):
        """Initialises NEBMover.

        Args:
           fixcom: An optional boolean which decides whether the centre of mass
              motion will be constrained or not. Defaults to False.
        """
        super(MAPNEBGPRMover, self).__init__(fixcom=fixcom, fixatoms=fixatoms)

        if distance_cutoff_for_training_data > gpr_trust_region:
            distance_cutoff_for_training_data = gpr_trust_region
            print(f"readjust distance cutoff for rejecting training data to trust region value {gpr_trust_region}")

        # parameters to pass in from input.xml
        self.options = {}

        # mode for optimization (geometry optimization algorithm)
        self.options["mode"] = mode
        # optimizer for optimization (neb / string / improved string)
        self.options["opt"] = opt

        self.options["asr"] = asr
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
        self.options["train_hessian_model_bool"] = train_hessian_model_bool
        self.options["train_grad_model_bool"] = train_grad_model_bool

        self.options["add_new_hessian_data_bool"] = add_new_hessian_data_bool
        self.options["candidate_hessian_data_number"] = candidate_hessian_data_number
        
        self.options["add_new_grad_data_bool"] = add_new_grad_data_bool 
        self.options["candidate_grad_data_number"] = candidate_grad_data_number
        
        self.options["gpr_fix_internal_dofs_bool"] = gpr_fix_internal_dofs_bool 
        self.options["gpr_fix_internal_dofs_cutoff"] = gpr_fix_internal_dofs_cutoff
        self.options["gpr_rigid_internal_dofs_cutoff"] = gpr_rigid_internal_dofs_cutoff
        # minimum value for allowed trust region ratio.
        # This is to prevent the algorithm making the trust region ratio too small.
        self.options["minimum_trust_region"] = minimum_trust_region

        # The cutoff for the scaled internal coordinate distnace for training data.
        # The training data is not allowed to be too close to each other, which will make the kernel matrix ill-conditioned.
        self.options["distance_cutoff_for_training_data"] = distance_cutoff_for_training_data

        # Whether compute hessians in the internal coordinate and compute only 1 hessian for rigid modes. 
        self.options["selective_hessian_bool"] = selective_hessian_bool

        self.options["internal_coord"] = internal_coord 

        self.options["cross_validation_bool"] = cross_validation_bool

        # numerical values / arrays. option from input.xml
        self.optarrays = {}
        self.optarrays["fix_dofs"] = fix_dofs  # the cartesian dofs of molecules to be fixed. 
        self.optarrays["energy_shift"] = energy_shift

        self.optarrays["neb_inner_loop_step_max"] = neb_inner_loop_step_max
        self.optarrays["spring_k"] = spring_k
        self.optarrays["kappa"] = kappa
        self.optarrays["ENO_order"] = ENO_order
        self.optarrays["dynamical_adjust_ratio"] = dynamical_adjust_ratio
        self.optarrays["end_bead_energy_converge_value"] = end_bead_energy_converge_value

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

        # for store ab initio grads along the path used for gpr hessian model.
        self.optarrays["new_grad_data_index"] = new_grad_data_index

        # for store ab initio hessians along rigid modes for selective
        # number of beads for gpr hessian model.
        self.optarrays["new_hessian_data_index_rigid_mode"] = new_hessian_data_index_rigid_mode

        # regularization value for linear regression of constrained parts of hessian.
        self.optarrays["ridge_regularization_alpha"] = ridge_regularization_alpha 
        
        # nugget regularization of pseudo-inverse of covariance matrix
        self.optarrays["gpr_covar_inverse_nugget"] = gpr_covar_inverse_nugget
        self.rp_map = RP_MAP()

        # choose optimization method based on optimizer we provide in input.xml
        if self.options["opt"] == "neb":
            self.optimizer = LINEBMethod()
        elif self.options["opt"] == "string":
            self.optimizer = StringMethod()
        elif self.options["opt"] == "improved_string":
            # The bead redistribution part is the same as string method. 
            # but the gradient mapper (gm) is different.
            self.optimizer = StringMethod()
        else:
            raise ValueError("The opt Value does not match any existing options. Please choose either neb/string/improved_string.")
        
        # choose gradient mapper (compute optimization gradient) based on the optimizer we provide in input.xml:
        if self.options["opt"] == "neb":
            self.gm = LINEBGradientMapper()
        elif self.options["opt"] == "string":
            self.gm = StringGradientMapper()
        elif self.options["opt"] == "improved_string":
            self.gm = ImprovedStringGradientMapper()
        else:
            raise ValueError("The opt Value does not match any existing options. Please choose either neb/string/improved_string.")
        
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
        self.optarrays["gpr_force_uncertainty_criterion"] = (
            gpr_force_uncertainty_criterion
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
        self.gpr_force_amplitude_list = []
        self.force_diff_amplitude_list = []

        self.force_diff_amplitude_after_update_list = []
        self.force_diff_ratio_after_update_list = []

        self.coordinate_transformer = None  # coordinate transformer between the Cartesian coordinate and the internal coordinate
        self.gpr_model = None  # Gaussian Process Regression model instance.

         # used to record the time for the calculation.
        self.start_time = timer() 
        
        # record the number of ab initio calculation on beads.
        SharedData.ab_initio_bead_calculation_number = (
            0  
        )

        # number of optimization steps.
        SharedData.inner_loop_optimization_step = 0

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

        # check if fix_dofs is out of range.
        fix_dofs = self.optarrays["fix_dofs"]
        if np.any(fix_dofs >= (3 * self.beads.natoms)):
            raise ValueError("dofs to be fixed is larger than 3 * natoms, this is wrong.")

        # create bead object that is used to add training data to GPR model.
        # We use One Image evaluation method, each time only update bead for one image.
        self.gpr_beads = Beads(self.beads.natoms, 1)
        self.gpr_forces = self.forces.copy(self.gpr_beads, self.cell)

        self.gm.bind(self)
        self.rp_map.bind(self)
        self.optimizer.bind(self)

    def step(self, step=None):
        """
        Does one simulation time step.
        if stage = 'neb', we will do path searching (LI-NEB/string) with Gaussian Process Regression.
        if stage = 'instanton', we will evolve instanton beads along the path.
        if stage = 'converged', we will stop the simulation.
        """
        print(" @NEB Outerloop STEP %d, stage: %s" % (step, self.options["stage"]))

        if step == 0:
            self._initial_step(step)

        if self.coordinate_transformer is None:
            # initialize Gaussian Process Regression(GPR) model and coordiante transformer
            self.initialialize_gpr_model()

            # check the training result on the test data which is unseen by GPR.
            self.check_initial_training_result()

        # Check if we enter the program directly into "instanton" stage:
        if self.options["stage"] == "instanton" and step == 0:
            self.rp_map.skip_neb_mode_bool = True
            print("Skip neb stage. Go directly into instanton stage. \n")
        else:
            self.rp_map.skip_neb_mode_bool = False

        if self.options["stage"] == "neb" and step == 0:
            self._open_neb_output_file()

        # Check if we restarted a converged calculation or the calculation converged.
        if self.options["stage"] == "converged":
            self.converge_stage_motion(step)
        elif self.options["stage"] == "neb":
            self.path_searching_stage_motion(step)
        elif self.options["stage"] == "instanton":
            self.instanton_stage_motion(step)
        else:
            raise ValueError(
                "unrecognized stage parameter. The stage has to be neb or instanton or converged"
            )

    def _initial_step(self, step):
        """
        initial step set up.
        """
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

        # The instanton path energy is defined relative to the energy shift.
        # We perform the transformation only when we start the initial calculation. Not for restarting the calculation.
        self.optarrays["instanton_path_energy"] = (
            self.optarrays["instanton_path_energy"] + self.optarrays["energy_shift"]
        )  # shift the instanton path energy according to energy shift.
        self.gm.instanton_path_energy = self.optarrays["instanton_path_energy"]
        self.rp_map.instanton_path_energy = self.optarrays["instanton_path_energy"]
        self.optimizer.instanton_path_energy = self.optarrays["instanton_path_energy"]

    def _open_neb_output_file(self):
        """
        open the output file for neb optimization.
        """
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
        
        action_info_file_name = "action_info.txt"
        self.action_info_file = open(action_info_file_name, "w")
        self.action_info_file.write("step  action \n")

        action_outloop_info_file_name = "action_info_outloop.txt"
        self.action_outloop_info_file = open(action_outloop_info_file_name, "w")
        self.action_outloop_info_file.write("step action \n")

        file_object_list = {
            "opt_grad": self.optimization_gradient_file,
            "opt_grad_outloop": self.optimization_gradient_outloop_file,
            "geometry_info": self.geometry_info_file,
            "action_info": self.action_info_file,
            "action_info_outloop": self.action_outloop_info_file
        }
        self.optimizer.bind_output_file(file_object_list)

    def path_searching_stage_motion(self, step):
        """
        Use nudged elastic band method to find minimum action path.
        then we switch to the "instanton" stage.
        perform LI-NEB algorithm on the surrogated PES generated by GPR. 
        stop the algorithm either when LI-NEB converge or one bead moves out of the trust region.
        """
        self.optimizer.inner_loop_initialize(step)
        
        # print geometry when outer_loop_step % alt = 0. for record.
        self.print_geometry(step)

        early_stop_bool, outrange_bead_index_list, grad_max_inner_bead, grad_max_end_bead = self.optimizer.inner_loop(step)
        
        # update the bead location. 
        self.beads.q[:] = dstrip(self.optimizer.beads.q[:]).copy()

        # write optimization gradient for each time we update the GPR model
        self.optimization_gradient_outloop_file.write(
            str(step) + " "
            + str(grad_max_inner_bead) + " "
            + str(grad_max_end_bead) + " "
            + str(SharedData.ab_initio_bead_calculation_number) + "\n"
        )

        self.action_outloop_info_file.write(
            str(step) + " "
            + str(self.action) + " "
            + str(SharedData.ab_initio_bead_calculation_number) + "\n "
        )

        # update Gaussian Process Regression model with new training data
        self.update_GPR_model(early_stop_bool, outrange_bead_index_list, step)

        print("optimization step so far for neb stage: " + str(SharedData.inner_loop_optimization_step))

    def instanton_stage_motion(self, step):
        """
        generate instanton ring polymer beads from minimum action path found by NEB.
        """
        info(
            "Now generate instanton path from Minimum Action Path (MAP) found by NEB."
        )
        print("total optimization step for neb stage: " + str(SharedData.inner_loop_optimization_step))

        self.rp_map.generate_ring_polymer_beads(self.beads, step)

        # save the potential, q, temperature, hessian of instanton beads for RESTART.
        self.save_instanton_ring_polymer()

        # ! If we exit here, the RESTART file will not record the hessian and instanton geometry we just computed.
        # therefore, we set ["stage"] == "converged" and exit at next step.
        self.options["stage"] = "converged"

    def converge_stage_motion(self, step):
        """
        The algorithm converges. Wrap up the algorithm.
        """
        # output number of ab-initio calculation.
        ipi.utils.nebinstgprtool.print_ab_initio_calculation_number(
            SharedData.ab_initio_bead_calculation_number, self.output_maker, step
        )
        print(
            "ab initio calculation number : "
            + str(SharedData.ab_initio_bead_calculation_number)
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

    def _select_reference_points(self):
        """
        selects reference points for coordinate transformation.
        Initialize non redundant coordinate transformer.
        choose the point with the highest potential in the initial instanton path as reference point.
        """
        beads_pots = np.copy(self.forces.pots)
        bead_index_at_transition_state = np.argmax(beads_pots)
        ref_x = dstrip(self.beads.q[bead_index_at_transition_state]).copy()

        # Now, we just use the poiont with lowest energy at reactant and product side.
        ref_x_reactant = dstrip(self.beads.q[0]).copy() # coordinate at reactant side.
        ref_x_product = dstrip(self.beads.q[-1]).copy() # coordinate at product side

        ref_x_list = np.array([ref_x, ref_x_reactant, ref_x_product])
        
        return ref_x_list 

    def _initialize_coordinate_transformer(self, ref_x_list):
        """
        Initialize the coordinate transformer that transform the system from the Cartesian coordinate into the internal coordinate.
        """
        names = dstrip(self.beads.names).copy().tolist()
        ref_x = ref_x_list[0]
        # create coordinate_transformer, which handles the transformation from the Cartesian coordinate to internal coordinate.
        # This is for Coulomb matrix type internal coordinate.
        if self.options["internal_coord"] == "Coulomb":
            self.coordinate_transformer = ipi.utils.internalcoordtools.non_redundant_coordinate_transformer(
                self.beads.natoms, ref_x
            )
        elif self.options["internal_coord"] == "bond":
            # This is for internal coordinate that include bond angles and bond distance
            self.coordinate_transformer = ipi.utils.internal.internaltools.non_redundant_coordinate_transformer(
                self.beads.natoms,
                ref_x_list,
                names
            )
        else:
            raise ValueError("The input for internal_coord should be either 'bond' or 'Coulomb' ")

    def _generate_initial_training_data(self):
        """
        Compute potential and force for initial training data.
        We only compute 3 data point (including end beads) as initial training data to avoid computational cost scales with # of beads we add.
        """
        # # choose all NEB beads as initial training data.
        # # We will train the GPR model to optimize hyperparameter using the initial data.
        # train_x = np.copy(self.beads.q)
        # # potential energy has to shift relative to the energy_shift for training.
        # train_V = np.copy(self.forces.pots) - self.optarrays["energy_shift"]
        # train_grad = -np.copy(dstrip(self.forces.f))
        # train_grad = ipi.utils.nebinstool.fixing_dofs(train_grad, self.optarrays["fix_dofs"])
        # # count the # of ab-initio calculation we have done.
        # SharedData.ab_initio_bead_calculation_number = (
        #     SharedData.ab_initio_bead_calculation_number + self.beads.nbeads
        # )

        # choose two end beads and the bead in the middle as initial training data.
        # We will train the GPR model to optimize hyperparameter using the initial data.
        initial_bead_number = 3 
        nbeads = self.beads.nbeads

        self.initial_data_bead = Beads(self.beads.natoms, initial_bead_number)
        self.initial_data_forces = self.forces.copy(self.initial_data_bead, self.cell)
        
        train_x = np.zeros([initial_bead_number, np.shape(self.beads.q)[1]])

        bead_index = np.linspace(0, nbeads - 1, initial_bead_number).astype(int)
        for i in range(initial_bead_number):
            train_x[i] = self.beads.q[bead_index[i]]
            self.initial_data_bead.q[i] = train_x[i]
        
        # potential energy has to shift relative to the energy_shift for training.
        train_V = np.copy(self.initial_data_forces.pots) - self.optarrays["energy_shift"]
        train_grad = - np.copy(dstrip(self.initial_data_forces.f))
        train_grad = ipi.utils.nebinstool.fixing_dofs(train_grad, self.optarrays["fix_dofs"])
        # count the # of ab-initio calculations we have done.
        SharedData.ab_initio_bead_calculation_number = (
            SharedData.ab_initio_bead_calculation_number + initial_bead_number
        )

        return train_x, train_V, train_grad

    def _get_training_data(self):
        """
        Loads or generate initial training data."""
        read_gpr_training_data_bool = self.options["read_initial_gpr_training_data"]
        if not read_gpr_training_data_bool:
            train_x, train_V, train_grad = self._generate_initial_training_data()
        else:
            # read stored training data from folder.
            train_x, stored_train_V, stored_train_f =  (
                ipi.utils.nebinstgprtool.read_training_data(prefix="neb_final_gpr_training")
            )
            train_V = stored_train_V - self.optarrays["energy_shift"] 
            train_grad = - stored_train_f
            train_grad = ipi.utils.nebinstool.fixing_dofs(train_grad, self.optarrays["fix_dofs"])
            # count the number of ab-initio calculation we have done.
            SharedData.ab_initio_bead_calculation_number = (
                SharedData.ab_initio_bead_calculation_number + np.shape(train_x)[0]
            )

            # used to test the overfitting/ underfitting of the GPR model.
            bead_number_to_test = 3
            self.initial_data_bead = Beads(self.beads.natoms, bead_number_to_test)
            self.initial_data_forces = self.forces.copy(self.initial_data_bead, self.cell)

            for i in range(bead_number_to_test):
                self.initial_data_bead.q[i] = train_x[i]
        
        return train_x, train_V, train_grad 

    def _initialize_gpr_model(self, train_x, train_V, train_grad):
        """
        Initialize the GPR model.
        """
        gpr_fixed_internal_dofs = ipi.utils.nebinstgprtool.read_fixed_internal_dofs(prefix= "neb_final_gpr_training")
        fix_dofs = self.optarrays["fix_dofs"]

        training_data_num = np.shape(train_x)[0]

        self.gpr_model = ipi.utils.gprtools.GPModelWithDerivativesWrapper(
            train_x,
            train_V,
            train_grad,
            self.beads.natoms,
            self.coordinate_transformer,
            fix_dofs,
            gpr_SE_kernel_number=self.options["gpr_SE_kernel_number"],
            kernel_outputscale=self.optarrays["gpr_kernel_outputscale"],
            kernel_lengthscale_ratio=self.optarrays["gpr_kernel_lengthscale_ratio"],
            noise_std=self.optarrays["gpr_noise_std"],
            train_bool= False,
            gpr_fix_internal_dofs_bool= self.options["gpr_fix_internal_dofs_bool"],
            gpr_fix_internal_dofs_cutoff= self.options["gpr_fix_internal_dofs_cutoff"],
            gpr_fixed_internal_dofs= gpr_fixed_internal_dofs,
            singular_value_cutoff = self.optarrays["gpr_covar_inverse_nugget"]
        )
        
        read_gpr_training_data_bool = self.options["read_initial_gpr_training_data"]
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
        

    def initialialize_gpr_model(self):
        """
        initialize the gaussian process regression model.
        1. Initialize coordinate transformer to transform between internal coordinate and cartesian coordinate.
        2. initialize GPR_Wrapper, which combines coordinate transformer and GPR model.
        """
        # Initialize non redundant coordinate transformer.
        # choose the point with the highest potential in the initial instanton path as reference point.
        ref_x_list = self._select_reference_points()

        self._initialize_coordinate_transformer(ref_x_list)

        # attach ab_initio potential to self.nebgm.ab_initio_pot and self.nebgm.ab_initio_force.
        # In the LI-NEB algorithm, when there is ab-initio potential & force data available, we will use that potential and force.
        # If the ab-initio data point is not available, we use the potential and force generated by Gaussian Process Regression (GPR)
        # self.gm.ab_initio_pot = np.copy(self.forces.pots)
        # self.gm.ab_initio_force = np.copy(dstrip(self.forces.f))

        # for the training data, we have the option to read it from .txt file or generate it using the current geometry.
        # this provides the flexibility for choosing the training data for the initial model.
        train_x, train_V, train_grad = self._get_training_data()

        self._initialize_gpr_model(train_x, train_V, train_grad)

        # bind the gpr model and coordinate_transformer to the LINEGradientMapper class
        # the LINEBGradientMapper will perform LI-NEB using gpr generated potential and force.
        self.gm.gpr_model = self.gpr_model
        self.gm.coordinate_transformer = self.coordinate_transformer

        self.rp_map.gpr_model = self.gpr_model
        self.rp_map.coordinate_transformer = self.coordinate_transformer

        self.optimizer.gpr_model = self.gpr_model
        self.optimizer.coordinate_transformer = self.coordinate_transformer

    def check_initial_training_result(self):
        """
        check whether the training of GPR model is successful. If not, stop the simulation and report error
        """
        # first check the prediction of the training data. See if there is under-fitting.
        predicted_V_shift, predicted_grad, _, var_grad_x_trace = (
            self.gpr_model.predict_latent_function(self.initial_data_bead.q)
        )

        predicted_forces = -predicted_grad

        ab_initio_V_shift = self.initial_data_forces.pots - self.optarrays["energy_shift"]
        ab_initio_forces = self.initial_data_forces.f

        print("\n")
        print(
            "@initial gpr training info: check the overfitting and underfitting of kernel length scale"
        )

        # check the force noise and potential noise. We can see for force noise of certain internal coordinate, it is quite large.
        # force_range = self.gpr_model.output_normalized_force_range()
        # V_noises, force_noises = self.gpr_model.output_fitted_gpr_model_noises()
        # force_noises_ratio = force_noises / force_range
        # print("potential noise amplitude: " + str(V_noises))
        # print("force noise ratio  (amplitude / range): " + str(force_noises_ratio))
        # print("internal coordinate force range: " + str(force_range))

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
        nbeads = self.initial_data_bead.nbeads

        if nbeads >= 2:
            test_q = self.initial_data_bead.q[0] * 1 / 4 + self.initial_data_bead.q[1] * 3 / 4
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

            SharedData.ab_initio_bead_calculation_number = (
                SharedData.ab_initio_bead_calculation_number + 1
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

            SharedData.ab_initio_bead_calculation_number = (
            SharedData.ab_initio_bead_calculation_number + 1
            )

        pass

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
        ab_initio_beads_forces = ipi.utils.nebinstool.fixing_dofs(ab_initio_beads_forces, self.optarrays["fix_dofs"])
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
            self.options["train_grad_model_bool"]
        )

        # set ab_initio pot and force in nebgm.
        self.gm.ab_initio_pot[bead_index_for_update] = ab_initio_beads_energy
        self.gm.ab_initio_force[bead_index_for_update] = np.copy(ab_initio_beads_forces)

        # count the # of ab-initio calculation we have done.
        SharedData.ab_initio_bead_calculation_number = (
            SharedData.ab_initio_bead_calculation_number + 1
        )

        # compute the difference between ab initio force and gpr force.
        ab_initio_force_amplitude = np.linalg.norm(ab_initio_beads_forces[0])
        force_diff = training_bead_forces - ab_initio_beads_forces[0]
        force_diff_ratio = np.linalg.norm(force_diff) / ab_initio_force_amplitude

        # add |df|/|f| , f and f^{GPR} into the list.
        self.force_diff_ratio_list.append(force_diff_ratio)
        self.ab_initio_force_amplitude_list.append(ab_initio_force_amplitude)
        self.gpr_force_amplitude_list.append(
            np.linalg.norm(training_bead_forces)
        )

        # after update the model
        _, new_outrange_bead_grad, _, _ = self.gpr_model.predict_latent_function(
            self.gpr_beads.q
        )
        new_training_bead_forces = - new_outrange_bead_grad[0]
        new_force_diff = new_training_bead_forces - ab_initio_beads_forces 
        new_force_diff_amplitude = np.linalg.norm(new_force_diff)
        new_force_diff_ratio = new_force_diff_amplitude / ab_initio_force_amplitude

        self.force_diff_amplitude_after_update_list.append(
            new_force_diff_amplitude
        )
        self.force_diff_ratio_after_update_list.append(
            new_force_diff_ratio
        )

        return force_diff_ratio, force_diff

    def update_GPR_model_with_beads_cause_early_stop(self, outrange_bead_index_list):
        """
        compute the ab initio potential and forces for beads far away from the trust region that causes the early stop.
        Among the beads that move out of the trust region, choose the point that has the largest uncertainty
        and update the model.
        The trust region could change at early stage if we add more data.
        """
        force_diff_list = []
        std_grad_x_trace_list = []
        
        # choose the point with largest uncertainty and add it to the training data.
        outrange_bead_index_list = np.array(outrange_bead_index_list)
        outrange_bead_x = np.copy(self.beads.q[outrange_bead_index_list])

        # evaluate the gpr predicted V & f. Also the uncertainty of prediction.
        _, outrange_bead_grad, _, var_grad_x_trace_list = self.gpr_model.predict_latent_function(
            outrange_bead_x
        )
        std_grad_x_trace_list = np.sqrt(var_grad_x_trace_list)
        outrange_bead_with_largest_uncertainty = np.argmax(std_grad_x_trace_list)

        bead_index_for_update = outrange_bead_index_list[outrange_bead_with_largest_uncertainty]
        training_x = np.array([outrange_bead_x[outrange_bead_with_largest_uncertainty]])
        training_bead_forces = - outrange_bead_grad[outrange_bead_with_largest_uncertainty]
        force_diff_ratio, force_diff = self.update_GPR_model_one_bead_subroutine(
            training_x, bead_index_for_update, training_bead_forces
        )

        force_diff_list.append(force_diff)
        self.force_diff_amplitude_list = np.linalg.norm(force_diff_list, axis= 1)

        print(f"@update gpr model due to early stop. bead index: {bead_index_for_update}")



    def update_GPR_model_image_with_large_uncertainty(self, step):
        """
        Use the uncertainty estimation of force from GPR model to choose the data point to add to the training data.
        If the force uncertainty (std in posterior distribution) is larger than cutoff value, then this point is selected. 
        """
        info(
            "Image with large uncertainty is selected for updating GPR model when LI-NEB converges. \n",
            verbosity.low,
        )

        # compute gpr predicted forces and uncertainty prediction.
        beads_potential_shifted, beads_grad_x, _, var_grad_x_uncertainty = self.gpr_model.predict_latent_function(self.beads.q)
        beads_forces = - beads_grad_x
        beads_potential = beads_potential_shifted + self.optarrays["energy_shift"]
        std_grad_x_uncertainty = np.sqrt(var_grad_x_uncertainty)
        
        print(f"uncertainty from GPR prediction: {std_grad_x_uncertainty}")

        large_uncertainty_bool = (std_grad_x_uncertainty > (self.optarrays["gpr_force_uncertainty_criterion"] + 1e-5))
        large_uncertainty_bead_index = np.arange(len(std_grad_x_uncertainty))[large_uncertainty_bool]

        if len(large_uncertainty_bead_index) == 0:
            # the surrogated potential is accurate enough
            self.neb_stage_exit_step(step, beads_potential)
        else:
            beads_number_to_update = len(large_uncertainty_bead_index)
            # create beads and forces object.
            beads_for_update = Beads(self.beads.natoms, beads_number_to_update)
            forces_for_update = self.forces.copy(beads_for_update, self.cell)
            
            training_x = np.copy(self.beads.q[large_uncertainty_bead_index])
            
            beads_for_update.q[:] = training_x

            # compute ab-initio potential and forces.
            ab_initio_beads_energy_for_update = dstrip(forces_for_update.pots).copy()
            ab_initio_shifted_energy_for_update = ab_initio_beads_energy_for_update - self.optarrays["energy_shift"]
            ab_initio_forces_for_update = dstrip(forces_for_update.f).copy() 
            ab_initio_forces_for_update = ipi.utils.nebinstool.fixing_dofs(ab_initio_forces_for_update, self.optarrays["fix_dofs"])
            ab_initio_grad_x_for_update = - ab_initio_forces_for_update

            # update gpr model with new data 
            self.gpr_model.update_model_with_new_data(
                training_x,
                ab_initio_shifted_energy_for_update,
                ab_initio_grad_x_for_update,
                self.options["distance_cutoff_for_training_data"],
                self.options["train_grad_model_bool"]
            )

            # set ab initio pot and force in nebgm
            self.gm.ab_initio_pot[large_uncertainty_bead_index] = np.copy(ab_initio_beads_energy_for_update)
            self.gm.ab_initio_force[large_uncertainty_bead_index] = np.copy(ab_initio_forces_for_update)

            # count the number of ab initio calculations we have done.
            SharedData.ab_initio_bead_calculation_number = (
                SharedData.ab_initio_bead_calculation_number + beads_number_to_update
            )

            # check whether the updated ab-initio forces are close to gpr predicted forces. 
            beads_forces_for_update = beads_forces[large_uncertainty_bead_index]
            force_diff_list = beads_forces_for_update - ab_initio_forces_for_update
            self.force_diff_amplitude_list = np.linalg.norm(force_diff_list, axis= 1)
            self.ab_initio_force_amplitude_list = np.linalg.norm(ab_initio_forces_for_update, axis= 1)
            self.gpr_force_amplitude_list = np.linalg.norm(beads_forces_for_update, axis= 1)
            self.force_diff_ratio_list = (
                self.force_diff_amplitude_list / self.ab_initio_force_amplitude_list
            )

            # check the uncertainty of force for updated potential.
            # increase the gpr_force_uncertainty criterion if the criterion is not met after we have updated the model.
            _, _, _, new_var_grad_x_uncertainty = self.gpr_model.predict_latent_function(self.beads.q)
            new_std_grad_x_uncertainty = np.sqrt(new_var_grad_x_uncertainty)
            max_std_grad_x_uncertainty = np.max(new_std_grad_x_uncertainty)
            if max_std_grad_x_uncertainty > self.optarrays["gpr_force_uncertainty_criterion"]:
                print("@Warning: The uncertainty of gpr prediction is still higher than cutoff criterion after update the model.")
                print(f"max std force uncertainty: {max_std_grad_x_uncertainty}")
                print(f"The force uncertainty criterion will be increased to {max_std_grad_x_uncertainty}")

                self.optarrays["gpr_force_uncertainty_criterion"] = max_std_grad_x_uncertainty

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
            # update GPR model with data points that have high posterior variance (uncertainty)
            # See PHYSICAL REVIEW LETTERS 122, 156001 (2019)
            self.update_GPR_model_image_with_large_uncertainty(step)

        # output info about force diff ratio |f_GPR -f|/|f|
        if len(self.ab_initio_force_amplitude_list) != 0:
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
        self.action_info_file.close()
        self.action_outloop_info_file.close() 

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
        beads_potential_shift, beads_potential_grad_x, _, _ = (
            self.gpr_model.predict_latent_function(np.copy(self.beads.q))
        )
        self.LINEB_pots = (
            beads_potential_shift + self.optarrays["energy_shift"]
        )
        self.LINEB_forces = - beads_potential_grad_x

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
        
        # store fixed dofs.
        ipi.utils.nebinstgprtool.store_fixed_internal_dofs_gpr_model(
            self.gpr_model,
            prefix = neb_gpr_folder_path
        )

    # ------ code below is for auxiliary functions --------------
    def print_geometry(self, step):
        """
        print beads geometry and beads energy.
        """
        pots = self.gm.beads_energy
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
            self.geometry_info_file.write(f"{step} {SharedData.inner_loop_optimization_step} {SharedData.ab_initio_bead_calculation_number} \n")

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

class SharedData:
    """
    class store class variable which will be shared between Optimization Method (LINEBMethod / StringMethod) and MAPNEBGPRMover
    """
    inner_loop_optimization_step = 0
    ab_initio_bead_calculation_number = 0

class DummyMethod(object):
    """
    base Method class to optimize the instanton path.
    """

    def __init__(self):
        """
        """
        # measure the distance of beads from the training data in internal coordinate.  
        self.internal_coordinate_closest_r_list = (
            []
        )  

    
    def bind(self, ens: MAPNEBGPRMover):
        """
        bind beads and cells from MAPNEBGPRMover.
        Also bind Gaussian Process Regression (GPR) model and internal coordinate transformer. 
        """
        self.options= {}
        self.optarrays = {}

        self.beads = ens.beads.copy()
        self.beads.q = ens.beads.q  
        self.cell = ens.cell.copy()
        self.fixatoms = ens.fixatoms.copy()
        self.fixatoms_mask = np.copy(ens.fixatoms_mask)
        self.options["asr"] = ens.options["asr"]
        self.time_step = ens.optarrays["time_step"]

        # optimization method (FIRE / CG/ Quick-Min)
        self.options["mode"]  = ens.options["mode"]
        # for FIRE algorithm.
        self.optarrays["FIRE"] = ens.optarrays["FIRE"]

        self.optarrays["neb_inner_loop_step_max"] = ens.optarrays["neb_inner_loop_step_max"]

        self.optarrays["gpr_trust_region"] = ens.optarrays["gpr_trust_region"]
        # bind the instanton path energy from NEB mover.
        self.instanton_path_energy = ens.optarrays[
            "instanton_path_energy"
        ]  
        self.energy_shift = ens.optarrays["energy_shift"]

        # bind the end beads energy constraint constant kappa.
        self.kappa = ens.optarrays[
            "kappa"
        ]
        # when end bead's energy is within converge value criterion
        # we will assume the end bead is on equal energy surface
        self.end_bead_energy_converge_value = ens.optarrays["end_bead_energy_converge_value"]
        self.options["tolerances"] = ens.options["tolerances"]
        self.dynamical_adjust_ratio = ens.optarrays["dynamical_adjust_ratio"]

        # bind the gradient Mapper.
        self.gpr_model = None
        self.coordinate_transformer = None 
        self.gm = ens.gm 

    def bind_output_file(self, file_object_list):
        """
        bind the output file object.
        """
        self.optimization_gradient_file = file_object_list["opt_grad"]
        self.optimization_gradient_outloop_file = file_object_list["opt_grad_outloop"]
        self.geometry_info_file = file_object_list["geometry_info"]
        self.action_info_file = file_object_list["action_info"]
        self.action_outloop_info_file = file_object_list["action_info_outloop"]

    def inner_loop_initialize(self, step):
        """
        initialize optimization algorithms' parameters (FIRE/ Quick-Min/ CG) for inner loop.
        update parameter for string/ LI-NEB method if necessary.
        """
        # coordinate of free moving atoms
        self.x = np.copy(self.beads.q[:, self.fixatoms_mask])

        # discretized action of the path.
        self.action = None

        # initialize the force to prepare for update of constraint parameters. 
        self.gm.initialize_force(self.x)
        
        # update_constraint_parameters (kappa and spring_k (for LINEB))
        self.update_constraint_parameters()

        mscaled_x = self.x * np.sqrt(
            self.beads.m3[:, self.fixatoms_mask]
        )

        self.action, self.grad_mscaled = self.gm(
            mscaled_x
        )
        # negative gradient of action for each bead in mass scaled coordinate.
        self.f_mscaled = -self.grad_mscaled 

        if step == 0:
             # mass scaled velocity. Used in dynamics optimizatioin algorithm,
            # for example: projected velocity verlet or FIRE 
            self.velocity_mscaled = np.zeros(
                [self.beads.nbeads, 3 * (self.beads.natoms - len(self.fixatoms))]
            )

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

                self.dtmax = self.time_step * self.optarrays["FIRE"]["tmax"]
                self.dtmin = self.time_step * self.optarrays["FIRE"]["tmin"]

                self.Ndelay = self.optarrays["FIRE"]["Ndelay"]
                self.finc = self.optarrays["FIRE"]["finc"]
                self.fdec = self.optarrays["FIRE"]["fdec"]

                self.Nmax = self.optarrays["FIRE"]["Nmax"]
                self.maxstep = self.optarrays["FIRE"]["maxstep"]

                self.Ndn = 0  # number of steps going down hill
                self.Nup = 0  # number of steps going up hill.

    def update_constraint_parameters(self):
        """
        update parameters related to the constraint of the end beads & inner beads (for LI-NEB method).
        """
        dt = self.time_step
        # energy constraint constant for left end bead. 
        left_kappa = self.kappa["left"] 
        # energy constraint constant for right end bead. 
        right_kappa = self.kappa["right"]
        
        # parameter for end bead convergence to energy contour
        end_bead_energy_converge_value = self.end_bead_energy_converge_value
        end_bead_gradient_tolerances = self.options["tolerances"]["gradient_end_bead"]

        kappa_ratio = self.dynamical_adjust_ratio["kappa"]

        max_force_cutoff = np.power(10.0, -3)
        
        # check |dV/dx| * kappa / sqrt(m_H) * (dt)^2. We use stability criterion to set it as 0.5 (empirical value).
        # TODO: Need to change this in case the tunneling atom is not hydrogen atom. 
        # mass of hydrogen in atomic unit. We typically study proton transfer reaction. 
        m_H = 1837  
        
        # check the left end bead.
        max_force2 = np.max(
            np.abs(self.gm.rbf[0])
        )
        max_force2 = np.max([max_force_cutoff, max_force2])
        val2 = max_force2 * np.power(dt, 2) * left_kappa / np.sqrt(m_H)
        left_kappa_scale = kappa_ratio / val2
        self.kappa["left"] = self.kappa["left"] * left_kappa_scale 

        # make sure the kappa value we set is not too large for the convergence.
        if abs(self.gm.beads_energy[0] - self.instanton_path_energy) < end_bead_energy_converge_value:
            left_kappa_for_converge = 0.2 * end_bead_gradient_tolerances / end_bead_energy_converge_value
            self.kappa["left"] = np.min([self.kappa["left"], left_kappa_for_converge])

        # check the right end bead.
        max_force3 = np.max(
            np.abs(self.gm.rbf[-1])
        ) 
        max_force3 = np.max([max_force_cutoff, max_force3])
        val3 = max_force3 * np.power(dt, 2) * right_kappa / np.sqrt(m_H)
        right_kappa_scale = kappa_ratio / val3
        self.kappa["right"] = self.kappa["right"] * right_kappa_scale 
        # make sure the kappa value we set is not too large for the convergence.
        if abs(self.gm.beads_energy[-1] - self.instanton_path_energy) < end_bead_energy_converge_value:
            right_kappa_for_converge = 0.2 * end_bead_gradient_tolerances / end_bead_energy_converge_value
            self.kappa["right"] = np.min([self.kappa["right"], right_kappa_for_converge])

    def optimize_beads_one_step(self):
        """
        Use different optimizer to optimize the LI-NEB beads.
        See: THE JOURNAL OF CHEMICAL PHYSICS 128, 134106 2008 for benchmark of different optimizer's performance
        """
        old_x = np.copy(self.x)
        x_mscaled = self.x * np.sqrt(
            self.beads.m3[:, self.fixatoms_mask]
        )

        # neb move using gradient of LINEBGradient
        # See: J. Chem. Phys. 128, 134106 (2008) for the performance of different optimization algorithm
        if self.options["mode"] == "verlet":
            x_mscaled, self.velocity_mscaled, self.action, self.grad_mscaled = \
            ipi.utils.nebinstool.projected_verlet(
                x_mscaled, 
                self.velocity_mscaled,
                (self.action, self.grad_mscaled),
                self.gm,
                self.time_step
            )
        elif self.options["mode"] == "cg":
            x_mscaled, self.action, self.grad_mscaled, self.conjugate_search_direction= \
            ipi.utils.nebinstool.conjugate_gradient(
                x_mscaled,
                (self.action, self.grad_mscaled),
                self.gm,
                self.conjugate_search_direction,
                self.optarrays["cg_big_step"]
            )
        elif self.options["mode"] == "FIRE":
            fdf0 = (self.action, self.grad_mscaled)
            # one step using FIRE. 
            # the x_mscaled will be updated in the mintools.FIRE() code.
            self.velocity_mscaled, self.alpha, self.Ndn, self.Nup, self.time_step  = \
              ipi.utils.mintools.FIRE(x_mscaled,
                                self.gm,
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
            # update action & mass scaled optimization gradient.
            self.action = self.gm.action 
            self.grad_mscaled = self.gm.optimization_gradient
        else:
            softexit.trigger(
                status="bad",
                message="Only projected velocity verlet (verlet), conjugate gradient (cg) and FIRE are currently implemented. set mode == 'verlet/cg/FIRE' ",
            )

        self.f_mscaled = -self.grad_mscaled
        # update new position
        self.x = x_mscaled / np.sqrt(
            self.beads.m3[:, self.fixatoms_mask]
        )

        self.project_out_asr_mode(old_x)

    def project_out_asr_mode(self, old_x):
        """
        """
        # project out translation & rotational dofs depending on asr mode.
        natoms = self.beads.natoms - len(self.fixatoms)
        m = self.beads.m3[0, self.fixatoms_mask][np.arange(0, 3 * natoms, 3)]

        if (self.options["mode"] == "verlet" or self.options["mode"] == "FIRE"):
            # project out translation & rotational mode in velocity.
            self.velocity_mscaled = ipi.utils.nebinstool.apply_symmetry_projection(m, np.copy(self.x), natoms, self.velocity_mscaled, asr= self.options["asr"], mscaled_bool= True)

        # project out translation & rotational mode from step.
        dx = self.x - old_x 
        projected_dx = ipi.utils.nebinstool.apply_symmetry_projection(m, np.copy(old_x), natoms, dx, asr= self.options["asr"])
        self.x = old_x + projected_dx 

        # update grad_mscaled and f_mscaled.
        x_mscaled = self.x * np.sqrt(
            self.beads.m3[:, self.fixatoms_mask]
        )
        self.action, self.grad_mscaled = self.gm(x_mscaled)
        self.f_mscaled = - self.grad_mscaled 

        self.beads.q[:, self.fixatoms_mask] = self.x

    def step_info(self,
                  outer_loop_step,
                  inner_loop_step,
                  grad_max_inner_bead,
                  grad_max_end_bead,
                  action_forces_sum_amplitude
                  ):
        """
        output the information about convergence check for each inner loop step.
        """
        tolerances = self.options["tolerances"]

        print("\n")
        info(
            "@Inner step summary: Outer loop # {} , inner loop # {}, \n \
              max force gradient for inner bead {:4.2e}, (condition {:4.2e}), \n \
              max force gradient for end bead {:4.2e} (condition {:4.2e}) \n\
              sum of gradient of action for internal beads {:4.2e} (condition {:4.2e}) \n \
              action {} ".format(
                outer_loop_step,
                inner_loop_step,
                grad_max_inner_bead,
                tolerances["gradient"],
                grad_max_end_bead,
                tolerances["gradient_end_bead"],
                action_forces_sum_amplitude,
                tolerances["action_forces_sum"],
                self.action
            ),
            verbosity.low
        )

        # record total number of optimization step
        SharedData.inner_loop_optimization_step = SharedData.inner_loop_optimization_step + 1

        # store the optimization gradient info
        self.optimization_gradient_file.write(
            str(SharedData.inner_loop_optimization_step) + "  "
            + str(grad_max_inner_bead) + "  "
            + str(grad_max_end_bead) + "\n"
        )

        # store the action info 
        self.action_info_file.write(
            str(SharedData.inner_loop_optimization_step) + " "
             + str(self.action) + "\n"
        )

        # check the optimization gradient for LI-NEB / string
        print(
            "beads optimization gradient: "
            + str(npnorm(self.gm.optimization_force, axis=1))
        )

        # check the potential of beads.
        beads_energy_relative_to_instanton_energy = (
            self.gm.beads_energy - self.instanton_path_energy
        ) * units.unit_to_user("energy", "electronvolt", 1)
        print(
            "beads potential relative to instanton path energy (eV): "
            + str(beads_energy_relative_to_instanton_energy)
        )

        # check the distance between beads (equal distance constraint)
        print(
            "distance between beads in mass scaled coordinate: "
            + str(self.gm.beads_mscaled_distance)
        )
        print("\n")
        print(
            "@Finish Inner loop: outer loop step {}, LI-NEB inner loop step {}".format(
                outer_loop_step, inner_loop_step
            )
        )
        print("\n")
        print("\n")


    def inner_loop_step(self, outer_loop_step, inner_loop_step, grad_max_inner_bead, grad_max_end_bead):
        """
        LI-NEB or string method move for one step
        """
        nbeads = self.beads.nbeads

        neb_inner_loop_step_max = self.optarrays["neb_inner_loop_step_max"]

        if self.options["mode"] == "FIRE":
            if inner_loop_step % self.optarrays["FIRE"]["neb_step_update_kappa"] == 0:
                self.update_constraint_parameters()
        else:
            self.update_constraint_parameters()
        
        if inner_loop_step > neb_inner_loop_step_max:
            self.converge = True 
        
        # We have changed the kappa / spring constant, thus recompute grad mapper optimization force.
        x_mscaled = self.x * np.sqrt(
            self.beads.m3[:, self.fixatoms_mask]
        )
        self.action, self.grad_mscaled = self.gm(x_mscaled)
        self.f_mscaled = - self.grad_mscaled 

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
            inner_loop_step,
            self.beads.m3
        )

        # stop the step early if the beads is out of trust region.
        if early_stop_bool:
            return (
                grad_max_inner_bead,
                grad_max_end_bead,
                early_stop_bool,
                outrange_bead_index_list,
            )
    
        # move beads 1 step using gradient of gm.
        # call either FIRE / Quick-MIN / CG method. 
        # for string method, we also need to re-parametrize the path & velocity 
        # if the bead distance constraint is no longer satisfied.
        self.optimize_beads_one_step()

        # compute maximum LI-NEB gradient among all beads. used for monitoring the convergence of LI-NEB.
        grad_norm = npnorm(self.gm.optimization_force, axis=1)

        grad_max_inner_bead = np.amax(grad_norm[1 : nbeads - 1])
        grad_max_end_bead = np.amax(np.array([grad_norm[0], grad_norm[-1]]))
        # amplitude of sum of transverse action force of internal beads.
        action_forces_sum_amplitude = self.gm.action_forces_sum_amplitude

        # output info about calculation.
        self.step_info(
            outer_loop_step,
            inner_loop_step,
            grad_max_inner_bead,
            grad_max_end_bead,
            action_forces_sum_amplitude
        )

        return (
            grad_max_inner_bead,
            grad_max_end_bead,
            early_stop_bool,
            outrange_bead_index_list,
        )

    def inner_loop(self, outer_loop_step):
        """
        the inner loop of Line Integral Nudged Elastic Band / String method.
        The loop will stop once one of the two criteria is met:
        (1) The optimization algorithm converge on the surrogated PES generated by Gaussian Process Regression model.
            This is the case when all the gradient of beads are smaller than the tolerance value.
        (2) One bead move out of the trust region. In this case, PES generated by GPR is not reliable any more,
            we need to early stop the algorithm and compute the ab-initio V & F at that given bead & add to the training data.
            The trust region is defined in the internal coordinate, scaled by the length scale of the squared exponential kernel.
        """
        info(
            " @NEB: start inner loop neb for step {}".format(outer_loop_step),
            verbosity.debug,
        )

        tolerances = self.options[
            "tolerances"
        ]

        inner_loop_step = 0

        early_stop_bool = False 
        # index for beads that move out of trusted region that causes the early stop.
        outrange_bead_index_list = ([])

        # first check the gradient of current geometry before making the move.
        # If it pass the criterion, we do not need to start the inner loop.
        grad_norm = npnorm(self.gm.optimization_force, axis=1)
        
        grad_max_inner_bead = np.amax(grad_norm[1 : self.beads.nbeads - 1])
        grad_max_end_bead = np.amax(np.array([grad_norm[0], grad_norm[-1]]))
        # amplitude of sum of transverse action force of internal beads.
        action_forces_sum_amplitude = self.gm.action_forces_sum_amplitude

        # output the step info
        self.step_info(
            outer_loop_step,
            inner_loop_step,
            grad_max_inner_bead,
            grad_max_end_bead,
            action_forces_sum_amplitude
        )

        print("\n")
        print("@Start outer loop: " + str(outer_loop_step) + "\n")

        self.converge = False

        while (
            grad_max_inner_bead > tolerances["gradient"]
            or grad_max_end_bead > tolerances["gradient_end_bead"]
            or ( self.gm.action_forces_sum_amplitude > tolerances["action_forces_sum"] )
        ):
            inner_loop_step = inner_loop_step + 1  # inner_loop_step == 0: we have not moved the bead.

            (
                grad_max_inner_bead,
                grad_max_end_bead,
                early_stop_bool,
                outrange_bead_index_list,
            ) = self.inner_loop_step(outer_loop_step, inner_loop_step, grad_max_inner_bead, grad_max_end_bead)

            if self.converge:
                # too many optimization steps. The system has converged on the energy surface.
                self.converge = False 
                break 
            
            # beads move out of trust region.
            if early_stop_bool:
                break 
        
        if not early_stop_bool:
            print("@LI-NEB/String converge on GPR PES")
        
        return early_stop_bool, outrange_bead_index_list, grad_max_inner_bead, grad_max_end_bead

class LINEBMethod(DummyMethod):
    """
    Line Integral Nudged Elastic Band method to optimize the instanton path.
    """
    
    def __init__(self):
        """
        """
    
    def bind(self, ens: MAPNEBGPRMover):
        """
        call base class (DummyMethod)'s bind function.
        bind spring constnat k for LINEB algorithm.
        """
        super(LINEBMethod, self).bind(ens)

        self.spring_k = ens.optarrays["spring_k"]

    def update_constraint_parameters(self):
        # update end bead constraint parameter kappa.
        super(LINEBMethod, self).update_constraint_parameters()

        # update spring constant.
        dt = self.time_step 
        bead_number = self.beads.nbeads

        # alternative choice of spring constant: we want spring_k * path_distance / (10 * bead_number) = <g>_action.
        mscaled_x = self.x *  np.sqrt(self.beads.m3[:, self.fixatoms_mask])
        path_distance = np.sum(np.linalg.norm(mscaled_x[1:] - mscaled_x[:-1], axis= 1))
        nimages = self.beads.nbeads
        natoms = self.beads.natoms
        action_force = self.gm.compute_action_force(nimages, natoms)
        average_action_force =  np.mean(np.linalg.norm(action_force[1:-1], axis= 1))
        # near the convergence, the action force can be small. We need a minimum cutoff for spring constant.
        average_action_force_cutoff = 5e-3 
        average_action_force = np.max([average_action_force, average_action_force_cutoff])
        self.spring_k = average_action_force / (path_distance / (10 * bead_number))
        self.gm.spring_k = self.spring_k

        # check spring_k * (dt)^2. We use stability criterion by setting spring_k * dt^2 = 0.25.
        # val1 = spring_k * np.power(dt, 2)
        # spring_k_ratio = self.optarrays["dynamical_adjust_ratio"]["spring_k"]
        # spring_k_scale = spring_k_ratio / val1
        # self.spring_k = self.spring_k * spring_k_scale
        # self.gm.spring_k = self.gm.spring_k * spring_k_scale

class StringMethod(DummyMethod):
    """
    String method/ Improved String Method to optimize the instanton path.
    """
    def __init__(self):
        """
        """
    
    def bind(self, ens: MAPNEBGPRMover):
        """
        call base class (DummyMethod)'s bind function.
        """
        super(StringMethod, self).bind(ens)

    def optimize_beads_one_step(self):
        # call function to use optimizer to move bead for one step.
        super(StringMethod, self).optimize_beads_one_step()

        # redistribute beads along the path use cubic interpolation.
        self.redistribute_beads()
    

    def redistribute_beads(self):
        """
        move beads to the equal distance position using cubic interpolation.
        update the velocity of beads along the path using cubic interpolation.
        update the gradient and force at the new bead location along the path.
        """
        x_mscaled = self.x * np.sqrt(
            self.beads.m3[:, self.fixatoms_mask]
        )

        nbeads = self.beads.nbeads
        bead_distance = np.linalg.norm(x_mscaled[1:] - x_mscaled[:-1], axis= 1)
        path_length = np.sum(bead_distance)
        bead_distance_diff_cutoff = path_length / (10 * (nbeads -1))
        bead_distance_diff = np.abs(bead_distance[1:] - bead_distance[:-1])
        redistribute_bool = (np.max(bead_distance_diff) > bead_distance_diff_cutoff)

        if redistribute_bool:
            # equal spacing interpolation in mass scaled coordinate using cubic spline
            path_coord_cs = ipi.utils.nebinstool.path_cubic_spline_function(np.copy(x_mscaled),
                                                                            np.copy(x_mscaled))
            interpolated_r = np.linspace(0, 1, self.beads.nbeads)

            interpolated_x_mscaled = path_coord_cs(interpolated_r)
            x_mscaled = interpolated_x_mscaled
            self.x = x_mscaled / np.sqrt(
                self.beads.m3[:, self.fixatoms_mask]
            )

            # for MD type optimization algorithm, we also need to interpolate the velocity.
            if (self.options["mode"] == "verlet" or self.options["mode"] == "FIRE"):
                velocity_cs = ipi.utils.nebinstool.path_cubic_spline_function(np.copy(x_mscaled),
                                                                            np.copy(self.velocity_mscaled))
                interpolated_velocity_mscaled = velocity_cs(interpolated_r)
                self.velocity_mscaled = interpolated_velocity_mscaled    


            # update grad_mscaled and f_mscaled.
            self.action, self.grad_mscaled = self.gm(x_mscaled)
            self.f_mscaled = - self.grad_mscaled 

            self.beads.q[:, self.fixatoms_mask] = self.x

# -------  Start the code for GradientMapper -------------- 
class GradientMapper(object):
    """
    Creation of Gradient of target function that will be minimized
    """
    def __init__(self):
        """
        """
        self.kappa = None  # spring constants for beads at two ends.

        self.action = None  # abbreviated action.
        self.action_forces = None  # minus gradient of abbreviated action
        self.optimization_force = (
            None  # neb force for optimization of action with constraints at two ends.
        )

        self.instanton_path_energy = None  # energy E of instanton path in JWKB approximation. See: Section II. A in J. Chem. Phys. 148, 102334 (2018)

        self.gpr_model = None 
        self.coordinate_transformer = None


    def bind(self, ens: MAPNEBGPRMover):
        """
        """
        self.dbeads = ens.beads.copy()
        self.dcell = ens.cell.copy()
        self.fixatoms = ens.fixatoms.copy()
        self.fix_dofs = ens.optarrays["fix_dofs"]
        self.asr = ens.options["asr"]
        
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

        self.kappa = ens.optarrays[
            "kappa"
        ]  # bind end beads energy constraint constant kappa from NEBMover.

        self.energy_shift = ens.optarrays["energy_shift"]

        self.ab_initio_pot = np.zeros([self.dbeads.nbeads])
        self.ab_initio_force = np.zeros([self.dbeads.nbeads, 3 * self.dbeads.natoms])

        self.ENO_order = ens.optarrays["ENO_order"]

    def initialize_force(self, q):
        """
        initialize rbf & energy. This will enable us to use check_spring_k_kappa in the initialization() step of neb gm in MAPNEBGPRMover
        """
        self.rbeads.q[:, self.fixatoms_mask] = q

        # use Gaussian Process Regression to get the potential and forces for beads.
        self.beads_energy, beads_forces = self.get_gpr_potential_and_forces()

        # Forces for free moving dofs.
        self.rbf = beads_forces.copy()[:, self.fixatoms_mask]

        # mass weighted force
        self.mscaled_f = self.rbf / np.sqrt(
            self.dbeads.m3[:, self.fixatoms_mask]
        )  # 1/sqrt(m) * f: mass scaled force.

        self.mscaled_q = q * np.sqrt(
            self.dbeads.m3[:, self.fixatoms_mask]
        )

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
        
        beads_forces = ipi.utils.nebinstool.fixing_dofs(beads_forces, self.fix_dofs)
        
        # For using Simpson's rule to compute action W.  
        midpoint_test_x = (self.rbeads.q[:-1] + self.rbeads.q[1:]) / 2
        midpoint_potential_shift, midpoint_grad_x, _, _ = (
            self.gpr_model.predict_latent_function(
                midpoint_test_x
            )
        )
        
        midpoint_forces = - midpoint_grad_x 
        midpoint_beads_energy = midpoint_potential_shift + self.energy_shift
        
        nimage = self.dbeads.nbeads
        self.midpoint_beads_energy = midpoint_beads_energy
        self.midpoint_rbf = midpoint_forces.copy()[:, self.fixatoms_mask]
        self.midpoint_rbf = ipi.utils.nebinstool.fixing_dofs(self.midpoint_rbf, self.fix_dofs)

        self.mscaled_midpoint_f = self.midpoint_rbf / np.sqrt(
            self.dbeads.m3[:nimage - 1, self.fixatoms_mask]
        )

        return beads_potential, beads_forces


    def compute_tangent_vector(self, nimage, natom):
        """
        we used the improved tangent direction:
        J. Chem. Phys. 113, 9978 (2000); https://doi.org/10.1063/1.1323224
        :param: nimage: number of replica images
        :param: natom: number of atoms (free moving)
        :param: bq: beads coordinate (mass_scaled coordinate)

        :return: btau: unit director for tangent vector of all internal beads in mass_scaled coordinates. (We do not need tangent vector for beads at two ends.)
        """
        mscaled_q = np.copy(self.mscaled_q)
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
    

    def compute_abbreviated_action(self, nimage):
        """
        compute abbreviated action W. See eq.(10) in J. Chem. Phys. 148, 102334 (2018)
        Note: in atomic unit, hbar = kb = 1.

        :param: nimage: number of images (replicas)

        :return: action: abbreviated action of the ring polymer path
        """
        mscaled_q = np.copy(self.mscaled_q)
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
    
    def compute_action_force(self, nimage, natom):
        """
        compute the negative gradient of abbreviated action W. (for scaled coordinates.) See eq. (11) in J. Chem. Phys. 148, 102334 (2018).
        Note I will use the same symbol as given in the eq.(11) in the paper.

        :param: nimag: number of images (replica). scalar
        :param: natom: number of freely moving atoms. scalar

        :return: action_force:  the negative gradient of abbreviated action W. (for scaled coordinates) size: [nimag, 3 * natom].
        """
        mscaled_q = np.copy(self.mscaled_q)
        mscaled_f = np.copy(self.mscaled_f)
        beads_energy = self.beads_energy
        
        bead_displs_vector = (
            mscaled_q[1:] - mscaled_q[:-1]
        )  # displacement vector of beads. [nbeads-1, 3 * natom]
        
        bead_distance = npnorm(
            bead_displs_vector, axis=1
        )  # |r_j - r_{j-1}|  [nbeads -1]

        # unit vector for beads displacement vector [nbeads -1, 3* natom]
        bead_displs_unit_vector = bead_displs_vector / bead_distance[:, np.newaxis] 

        # sqrt(2 (V - E))
        action_each_bead = np.zeros([nimage])
        for i in range(nimage):
            if beads_energy[i] < self.instanton_path_energy:
                action_each_bead[i] = 0
            else:
                action_each_bead[i] = np.sqrt(
                    2 * (beads_energy[i] - self.instanton_path_energy)
                )
        
        action_force = np.zeros([nimage, 3 * natom])
        for j in range(1, nimage - 1):
            dj1 = bead_distance[j - 1]  # |r_{j} - r_{j-1}|.  d_{j}
            dj2 = bead_distance[j]  # |r_{j+1} - r_{j}|. d_{j+1}
            dj1_unit_vector = bead_displs_unit_vector[j - 1]  # \hat{d}_{j}
            dj2_unit_vector = bead_displs_unit_vector[j]  # \hat{d}_{j+1}
            fj = mscaled_f[j]

            if action_each_bead[j] == 0:
                # when energy of beads is smaller than the path energy.
                # we set the force component equal to 0.
                gj_force_component = 0
                
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

# -------- The code below implement method to compute tangent vector / action / action gradient with higher accuracy ----- 
    def compute_ENO_tangent_vector(self):
        """
        compute tangent vector using essentially non-oscillatory scheme.
        See https://dx.doi.org/10.4310/CMS.2003.v1.n2.a10
        """
        mscaled_q = np.copy(self.mscaled_q)
        beads_energy = self.beads_energy
        eno_object = ipi.utils.nebinstool.Essentially_Nonoscillatory_Polynomial(mscaled_q,
                                                                                beads_energy,
                                                                                order = self.ENO_order)
        
        btau = eno_object.compute_tangent_vector()
        return btau 


    def compute_abbreviated_action_Simpson_rule(self, nimage):
        """
        compute abbreviated action W using Simpson's rule. (need potential of mid point beads.)
        Note: in atomic unit, hbar = kb = 1.

        :param: nimage: number of images (replicas)
        """
        mscaled_q = np.copy(self.mscaled_q)
        beads_energy = np.copy(self.beads_energy)

        midpoint_beads_energy = np.copy(self.midpoint_beads_energy)
        
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
        
        # sqrt(2 (V-E))
        midpoint_action_each_bead = np.zeros([nimage - 1])
        for i in range(nimage - 1):
            if midpoint_beads_energy[i] < self.instanton_path_energy:
                midpoint_action_each_bead[i] = 0
            else:
                midpoint_action_each_bead[i] = np.sqrt(
                    2 * (midpoint_beads_energy[i] - self.instanton_path_energy)
                )
        
        # compute action using Simpson's rule
        for j in range(1, nimage):
            rj = mscaled_q[j]
            rj_1 = mscaled_q[j - 1]
            r_dist = npnorm(rj - rj_1)
            
            action = (
                action  
                + 1 / 6 * r_dist * (action_each_bead[j] + action_each_bead[j - 1] + 4 * midpoint_action_each_bead[j - 1])
            )
        
        return action 

    def compute_action_force_Simpson_rule(self, nimage, natom):
        """
        compute the negative optimization gradient (optimization force) for abbreviated action W using Simpson's rule 
        Note: in atomic unit, hbar = kb = 1.

        :param: nimage: number of images (replicas)
        """
        mscaled_q = np.copy(self.mscaled_q)
        mscaled_f = np.copy(self.mscaled_f)
        beads_energy = np.copy(self.beads_energy)
        
        mscaled_midpoint_f = np.copy(self.mscaled_midpoint_f)
        midpoint_beads_energy = np.copy(self.midpoint_beads_energy)

        # displacement vector of beads. [nbeads-1, 3 * natom]
        bead_displs_vector = (
            mscaled_q[1:] - mscaled_q[:-1]
        )  
        
        # |r_j - r_{j-1}|  [nbeads -1]
        bead_distance = npnorm(
            bead_displs_vector, axis=1
        )  

        # unit vector for beads displacement vector [nbeads -1, 3* natom]
        bead_displs_unit_vector = bead_displs_vector / bead_distance[:, np.newaxis] 

        # sqrt(2 (V - E))
        action_each_bead = np.zeros([nimage])
        for i in range(nimage):
            if beads_energy[i] < self.instanton_path_energy:
                action_each_bead[i] = 0
            else:
                action_each_bead[i] = np.sqrt(
                    2 * (beads_energy[i] - self.instanton_path_energy)
                )
        
        # sqrt(2 (V-E))
        midpoint_action_each_bead = np.zeros([nimage - 1])
        for i in range(nimage - 1):
            if midpoint_beads_energy[i] < self.instanton_path_energy:
                midpoint_action_each_bead[i] = 0
            else:
                midpoint_action_each_bead[i] = np.sqrt(
                    2 * (midpoint_beads_energy[i] - self.instanton_path_energy)
                )
        
        action_force = np.zeros([nimage, 3 * natom])
        for j in range(1, nimage -1):
            dj1 = bead_distance[j - 1]  # |r_{j} - r_{j-1}|.  d_{j}
            dj2 = bead_distance[j]  # |r_{j+1} - r_{j}|. d_{j+1}
            dj1_unit_vector = bead_displs_unit_vector[j - 1] # \hat{d}_{j}
            dj2_unit_vector = bead_displs_unit_vector[j] # \hat{d}_{j+1}
            fj = mscaled_f[j]  # f_{j}
            fj_midpoint1 = mscaled_midpoint_f[j - 1]  # f_{j-1/2}
            fj_midpoint2 = mscaled_midpoint_f[j]  # f_{j+1/2}
            
            # compute the contribution to the optimization force by force.
            gj_force_component = 0
            if action_each_bead[j] != 0:
                gj_force_component = gj_force_component + 1/6 * (dj1 + dj2) / action_each_bead[j] * fj 
            if midpoint_action_each_bead[j - 1] != 0:
                gj_force_component  = gj_force_component + 1/3 * dj1 / midpoint_action_each_bead[j - 1] * fj_midpoint1
            if midpoint_action_each_bead[j] != 0:
                gj_force_component = gj_force_component + 1/3 * dj2 / midpoint_action_each_bead[j] * fj_midpoint2
            
            gj_curvature_component = 1/6 * (
                dj2_unit_vector * (action_each_bead[j] + action_each_bead[j + 1] + 4 * midpoint_action_each_bead[j])
                - dj1_unit_vector * (action_each_bead[j] + action_each_bead[j - 1] + 4 * midpoint_action_each_bead[j - 1])
            )

            gj = gj_force_component + gj_curvature_component
            action_force[j] = gj 
        
        return action_force 


class LINEBGradientMapper(GradientMapper):
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
        super(LINEBGradientMapper, self).__init__()
        self.spring_k = None  # spring constants for internal beads

    def bind(self, ens: MAPNEBGPRMover):
        """
        :param: ens: A NEBMover instance.
        Copy beads, cell, forces of NEB mover to itself.
        """
        super(LINEBGradientMapper, self).bind(ens)
        # bind spring force spring_k from NEBMover.
        self.spring_k = ens.optarrays[
            "spring_k"
        ]  

    def __call__(self, mscaled_q):
        """Returns the projected optimization gradient for LI-NEB algorithm..
        update reduced bead coordinates (&dbeads coordinate) (strictly speaking the free-moving atom parts) with x.
        :param: msacled_q: mass scaled coordinates for updated freely moving particles.

        rbf: physical forces for free moving beads.
        rbq: position for freely moving beads.
        btau: tangent vector directions.
        """
        # coordinate q.
        q = mscaled_q / np.sqrt(
            self.dbeads.m3[:, self.fixatoms_mask]
        )
        
        if not np.allclose(self.rbeads.q, q):
            self.initialize_force(q)

        # mass scaled coordinate.
        self.mscaled_q = np.copy(mscaled_q)

        # Number of images
        nimage = self.dbeads.nbeads
        # Number of atoms that is free to move.
        natoms = self.dbeads.natoms - len(self.fixatoms)

        self.spring_forces = np.zeros([nimage, 3 * natoms])
        self.end_beads_energy_constraint_forces = np.zeros([2, 3 * natoms])
        self.beads_mscaled_distance = npnorm(mscaled_q[1:] - mscaled_q[:-1], axis=1)

        # abbreviated action for the ring polymer instanton path.
        self.action = self.compute_abbreviated_action(nimage)

        # negative gradient of abbreviated action for each bead. We only compute it for the internal beads (excluding two ends)
        self.action_forces = self.compute_action_force(
            nimage, natoms
        )

        # Change code above to implement Simpson's rule
        # self.action = self.compute_neb_action_Simpson_rule(nimage)

        # self.action_forces = self.compute_neb_action_force_Simpson_rule(
        #     nimage, natom
        # )

        # #compute direction of tangent vector, using improved methods.
        # self.btau = self.compute_tangent_vector(nimage, natoms)
        # compute direction of tangent vector, using Essentially non-oscillatory method.
        self.btau = self.compute_ENO_tangent_vector()

        # evaluate the nudged elastic band optimization forces for perpendicular action forces and the spring force. 
        self.optimization_force = self.compute_neb_optimization_force(
            nimage, natoms, self.btau
        )

        # project out translation (&rotational) dofs depending on the asr mode. (symmetry of the system.)
        m = self.dbeads.m3[0, self.fixatoms_mask][np.arange(0, 3 * natoms, 3)]
        self.optimization_force = ipi.utils.nebinstool.apply_symmetry_projection(m, q, natoms, self.optimization_force, asr= self.asr, mscaled_bool= True)
        self.transverse_force = ipi.utils.nebinstool.apply_symmetry_projection(m, q, natoms, self.transverse_force, asr= self.asr, mscaled_bool= True)

        # sum of transverse forces for all internal beads 
        # (the action force for the end bead is set to be 0)
        # self.action_forces_sum_amplitude = np.sum(np.linalg.norm(self.action_forces[1: -1], axis=1))
        self.action_forces_sum_amplitude = np.sum(np.linalg.norm(self.transverse_force[1:-1], axis= 1))

        self.optimization_gradient = - self.optimization_force

        return self.action, np.copy(self.optimization_gradient)




    def compute_spring_force(self, nimage, natom, mscaled_q, mscaled_f, btau):
        """ """
        beads_energy = self.beads_energy

        spring_k_list = np.ones([nimage - 1]) * self.spring_k
            
        # spring forces for beads. Note the spring force at two ends are different from spring forces for internal beads.
        spring_force = np.zeros([nimage, 3 * natom])
        # spring force for internal beads
        for ii in range(1, nimage - 1):
            spring_force[ii] = (
                npnorm(mscaled_q[ii + 1] - mscaled_q[ii]) * spring_k_list[ii]
                - npnorm(mscaled_q[ii] - mscaled_q[ii - 1]) * spring_k_list[ii - 1]
            ) * btau[ii]

        # spring force for end bead 0
        # unit vector for q[1] - q[0]
        unit_vec_1 = (mscaled_q[1] - mscaled_q[0]) / npnorm(
            mscaled_q[1] - mscaled_q[0]
        )  
        # spring_force_bead0 = (
        #     unit_vec_1 * np.sqrt(2 * (self.beads_energy[1] - self.instanton_path_energy))
        # )

        spring_force_bead0 = (
            unit_vec_1 * np.linalg.norm(self.action_forces[1])
        )
        # unit vector along force at beads: 0
        f0 = mscaled_f[0] / npnorm(mscaled_f[0]) 
        # spring force component transverse to the gradient of potential.
        spring_force[0] = (
            spring_force_bead0 - np.dot(spring_force_bead0, f0) * f0
        )  

        # spring force for end bead nimag - 1
        unit_vec_2 = (mscaled_q[nimage - 2] - mscaled_q[nimage - 1]) / npnorm(
            mscaled_q[nimage - 2] - mscaled_q[nimage - 1]
        )

        # spring_force_bead1 = (
        #     unit_vec_2 * np.sqrt(2 * (self.beads_energy[nimage -2] - self.instanton_path_energy))
        # )

        spring_force_bead1 = (
            unit_vec_2 * np.linalg.norm(self.action_forces[-2])
        )
        # unit vector along force at beads: nimage - 1
        f1 = mscaled_f[nimage - 1] / npnorm(
            mscaled_f[nimage - 1]
        )  
        # spring force component transverse to the gradient of potential.
        spring_force[nimage - 1] = (
            spring_force_bead1 - np.dot(spring_force_bead1, f1) * f1
        )  

        return spring_force

    def compute_neb_optimization_force(self, nimage, natom, btau):
        """
        compute the optimization forces for nudged elastic band beads. See eq.(15 - 22) in J. Chem. Phys. 148, 102334 (2018).

        :param: nimag: number of images (replica). scalar
        :param: natom: number of freely moving atoms. scalar
        :param: btau: tangent vector for internal beads.  size: [nimag, 3 * natoms]

        :return: optimization_force: the optimization force for nudged elastic band. size: [nimage, 3 * natom]
        """
        mscaled_q = np.copy(self.mscaled_q)
        mscaled_f = np.copy(self.mscaled_f)
        beads_energy = self.beads_energy

        # kappa: constraint force back to iso-energy contour.
        left_kappa = self.kappa["left"]  # kappa for the left end beads
        right_kappa = self.kappa["right"]  # kappa for the right end beads.

        neb_optimization_force = np.zeros([nimage, 3 * natom])

        spring_force = self.compute_spring_force(
            nimage, natom, mscaled_q, mscaled_f, btau
        )

        # end_beads_energy_constraint_force: force to draw end beads back to isoenergy contours.
        # unit vector along force at beads: 0
        f0 = mscaled_f[0] / npnorm(mscaled_f[0]) 
        # unit vector along force at beads: nimage - 1
        f1 = mscaled_f[nimage - 1] / npnorm(
            mscaled_f[nimage - 1]
        )  
        end_beads_energy_constraint_force = np.zeros([2, 3 * natom])
        end_beads_energy_constraint_force[0] = (
            f0 * left_kappa * (beads_energy[0] - self.instanton_path_energy)
        )  # kappa * (V(r) - E) * \hat{f}(r) for beads 0
        end_beads_energy_constraint_force[1] = (
            f1 * right_kappa * (beads_energy[nimage - 1] - self.instanton_path_energy)
        )  # kappa * (V(r) - E) * \hat{f}(r) for beads n-1.

        self.spring_forces = spring_force  # store the spring force between beads
        self.end_beads_energy_constraint_forces = end_beads_energy_constraint_force  # store energy constraint force for end beads.

        # for internal beads, transverse force from negative gradient of action.
        for ii in range(1, nimage - 1):
            neb_optimization_force[ii] = (
                self.action_forces[ii]
                - np.dot(self.action_forces[ii], btau[ii]) * btau[ii]
            )

        # transverse gradient for interior neb beads.
        self.transverse_force = (
            np.copy(neb_optimization_force)  
        )

        # add energy constraint force for two end beads.
        neb_optimization_force[0] = end_beads_energy_constraint_force[0]
        neb_optimization_force[nimage - 1] = end_beads_energy_constraint_force[1]

        # add spring force for all beads.
        neb_optimization_force = neb_optimization_force + spring_force

        return neb_optimization_force

class StringGradientMapper(GradientMapper):
    """
    Creation of Gradient of target function for String method.
    Here the spring force constraint is removed. 
    """
    def __init__(self):
        """
        """
        super(StringGradientMapper, self).__init__()

    def bind(self, ens: MAPNEBGPRMover):
        """
        """
        super(StringGradientMapper, self).bind(ens)
    
    def __call__(self, mscaled_q):
        """
        Returns the projected gradient for string method. (Phys. Rev. B 66, 052301)
        Here, unlike the LI-NEB algorithm. The spring force term is removed. 
        Instead, we will perform re-parametrization if beads move out of position. 
        :param: mscaled_q: mass scaled coordinates for updated freely moving particles.

        rbf: physical forces for free moving beads.
        rbq: position for freely moving beads.
        btau: tangent vector directions.
        """
        # coordinate q.
        q = mscaled_q / np.sqrt(
            self.dbeads.m3[:, self.fixatoms_mask]
        )

        if not np.allclose(self.rbeads.q, q):
            self.initialize_force(q)

        # mass scaled coordinate.
        self.mscaled_q = np.copy(mscaled_q)

        # Number of images
        nimage = self.dbeads.nbeads
        # Number of atoms that is free to move.
        natoms = self.dbeads.natoms - len(self.fixatoms)

        # energy contour constraint force for two end beads.
        self.end_beads_energy_constraint_forces = np.zeros([2, 3 * natoms])
        self.beads_mscaled_distance = npnorm(mscaled_q[1:] - mscaled_q[:-1], axis=1)

        # compute abbreviated action for the ring polymer instanton path.
        self.action = self.compute_abbreviated_action(nimage)
        
        # compute abbreviated action force (negative gradient) for each bead.
        # we only compute action force for inner beads, not for end beads.
        self.action_forces = self.compute_action_force(nimage, natoms)

        # The code below use Simpson's rule to compute action & action force. 
        # This will be more accurate.
        # self.action = self.compute_abbreviated_action_Simpson_rule(nimage)
        # self.action_forces = self.compute_action_force_Simpson_rule(nimage, natoms)

        # compute direction of the tangent vector.
        # self.btau = self.compute_tangent_vector(nimage, natoms)
        # compute direction of the tangent vector, using Essentially Non-oscillatory method.
        self.btau = self.compute_ENO_tangent_vector()

        # evaluate the optimization forces for the string method. 
        self.optimization_force = self.compute_string_optimization_force(nimage, natoms, self.btau)
        # project out translation (&rotation) dofs depending on the asr mode.
        m = self.dbeads.m3[0, self.fixatoms_mask][np.arange(0, 3 * natoms, 3)]
        self.optimization_force = ipi.utils.nebinstool.apply_symmetry_projection(m, q, natoms, self.optimization_force, 
                                                                                 asr= self.asr, mscaled_bool= True)
        self.transverse_force = ipi.utils.nebinstool.apply_symmetry_projection(m, q, natoms, self.transverse_force,
                                                                               asr= self.asr, mscaled_bool= True)
        
        # sum of transverse forces for all internal beads.
        # This will be one convergence criterion.
        self.action_forces_sum_amplitude = np.sum(np.linalg.norm(self.transverse_force[1:-1], axis= 1))

        self.optimization_gradient = - self.optimization_force

        return self.action, np.copy(self.optimization_gradient)

    def compute_string_optimization_force(self, nimage, natoms, btau):
        """
        compute the optimization forces for beads in string method. 

        :param: nimag: number of images (replica). scalar
        :param: natom: number of freely moving atoms. scalar
        :param: btau: tangent vector for internal beads.  size: [nimag, 3 * natoms]
        
        :return: optimization_force: the optimization force for string method. size:[nimage, 3 * natom]
        """
        mscaled_q = np.copy(self.mscaled_q)
        mscaled_f = np.copy(self.mscaled_f)
        beads_energy = self.beads_energy

        # kappa: constraint force that pull beads back to energy contour.
        left_kappa = self.kappa["left"]  # kappa for the left end beads
        right_kappa = self.kappa["right"]  # kappa for the right end beads.

        neb_optimization_force = np.zeros([nimage, 3 * natoms])

        # end_beads_energy_constraint_force: force to draw end beads back to isoenergy contours.
        # unit vector along force at beads: 0
        f0 = mscaled_f[0] / npnorm(mscaled_f[0])
        # unit vector along force at beads: nimage - 1
        f1 = mscaled_f[nimage - 1] / npnorm(
            mscaled_f[nimage - 1]
        )

        end_beads_energy_constraint_force = np.zeros([2, 3 * natoms])
        # kappa * (V(r) - E) * \hat{f}(r) for beads 0
        end_beads_energy_constraint_force[0] = (
            f0 * left_kappa * (beads_energy[0] - self.instanton_path_energy)
        )  
        # kappa * (V(r) - E) * \hat{f}(r) for beads n-1.
        end_beads_energy_constraint_force[1] = (
            f1 * right_kappa * (beads_energy[nimage - 1] - self.instanton_path_energy)
        ) 
        self.end_beads_energy_constraint_forces = end_beads_energy_constraint_force

        # for internal beads, transverse force from negative gradient of action. 
        for ii in range(1, nimage - 1):
            neb_optimization_force[ii] = (
                self.action_forces[ii]
                - np.dot(self.action_forces[ii], btau[ii]) * btau[ii]
            )
        self.transverse_force = np.copy(neb_optimization_force)

        # for two end beads.
        # add energy constraint force for two end beads.
        neb_optimization_force[0] = end_beads_energy_constraint_force[0]
        neb_optimization_force[nimage - 1] = end_beads_energy_constraint_force[1]

        # add force term to end beads to make sure the path along end beads aligns with gradient of potential.
        # spring force for end bead 1
        unit_vec_1 = (mscaled_q[1] - mscaled_q[0]) / npnorm(
            mscaled_q[1] - mscaled_q[0]
        )
        spring_force_bead0 = (
            unit_vec_1 * np.linalg.norm(self.action_forces[1])
        )
        neb_optimization_force[0] = neb_optimization_force[0] + (
            spring_force_bead0 - np.dot(spring_force_bead0, f0) * f0
        )

        # spring force for end bead 2
        unit_vec_2 = (mscaled_q[nimage - 2] - mscaled_q[nimage - 1]) / npnorm(
            mscaled_q[nimage - 2] - mscaled_q[nimage - 1]
        )
        spring_force_bead1 = (
            unit_vec_2 * np.linalg.norm(self.action_forces[-2])
        )
        neb_optimization_force[nimage -1] = neb_optimization_force[nimage -1] + (
            spring_force_bead1 - np.dot(spring_force_bead1, f1) * f1
        )

        return neb_optimization_force

class ImprovedStringGradientMapper(GradientMapper):
    """
    Create the gradient for target function in improved string method. (J. Chem. Phys. 126, 164103 (2007))
    We no longer perform the nudging & projection for the optimization force.
    """
    def __init__(self):
        """
        """
        super(ImprovedStringGradientMapper, self).__init__()

    def bind(self, ens: MAPNEBGPRMover):
        """
        """
        super(ImprovedStringGradientMapper, self).bind(ens)

    def __call__(self, mscaled_q):
        """
        Returns the projected gradient for string method. (Phys. Rev. B 66, 052301)
        Here, unlike the LI-NEB algorithm. The spring force term is removed. 
        Instead, we will perform re-parametrization if beads move out of position. 
        :param: mscaled_q: mass scaled coordinates for updated freely moving particles.

        rbf: physical forces for free moving beads.
        rbq: position for freely moving beads.
        btau: tangent vector directions.
        """
        # coordinate q.
        q = mscaled_q / np.sqrt(
            self.dbeads.m3[:, self.fixatoms_mask]
        )

        if not np.allclose(self.rbeads.q, q):
            self.initialize_force(q)

        # mass scaled coordinate.
        self.mscaled_q = np.copy(mscaled_q)

        # Number of images
        nimage = self.dbeads.nbeads
        # Number of atoms that is free to move.
        natoms = self.dbeads.natoms - len(self.fixatoms)

        # energy contour constraint force for two end beads.
        self.end_beads_energy_constraint_forces = np.zeros([2, 3 * natoms])
        self.beads_mscaled_distance = npnorm(mscaled_q[1:] - mscaled_q[:-1], axis=1)

        # compute abbreviated action for the ring polymer instanton path.
        self.action = self.compute_abbreviated_action(nimage)
        
        # compute abbreviated action force (negative gradient) for each bead.
        # we only compute action force for inner beads, not for end beads.
        self.action_forces = self.compute_action_force(nimage, natoms)

        # The code below use Simpson's rule to compute action & action force. 
        # This will be more accurate.
        # self.action = self.compute_abbreviated_action_Simpson_rule(nimage)
        # self.action_forces = self.compute_action_force_Simpson_rule(nimage, natoms)

        # compute direction of the tangent vector, using Essentially Non-oscillatory method.
        self.btau = self.compute_ENO_tangent_vector()

        # evaluate the optimization forces for the string method. 
        self.optimization_force = self.compute_string_optimization_force(nimage, natoms)
        # project out translation (&rotation) dofs depending on the asr mode.
        m = self.dbeads.m3[0, self.fixatoms_mask][np.arange(0, 3 * natoms, 3)]
        self.optimization_force = ipi.utils.nebinstool.apply_symmetry_projection(m, q, natoms, self.optimization_force, 
                                                                                 asr= self.asr, mscaled_bool= True)
        
        # sum of transverse forces for all internal beads.
        # This will be one convergence criterion.
        self.action_forces_sum_amplitude = np.sum(np.linalg.norm(self.transverse_force[1:-1], axis= 1))

        self.optimization_gradient = - self.optimization_force

        return self.action, np.copy(self.optimization_gradient)

    def compute_string_optimization_force(self, nimage, natoms):
        """
        compute the optimization forces for beads in string method. 

        :param: nimag: number of images (replica). scalar
        :param: natom: number of freely moving atoms. scalar
        :param: btau: tangent vector for internal beads.  size: [nimag, 3 * natoms]
        
        :return: optimization_force: the optimization force for string method. size:[nimage, 3 * natom]
        """
        mscaled_q = np.copy(self.mscaled_q)
        mscaled_f = np.copy(self.mscaled_f)
        beads_energy = self.beads_energy

        # kappa: constraint force that pull beads back to energy contour.
        left_kappa = self.kappa["left"]  # kappa for the left end beads
        right_kappa = self.kappa["right"]  # kappa for the right end beads.

        neb_optimization_force = np.zeros([nimage, 3 * natoms])

        # end_beads_energy_constraint_force: force to draw end beads back to isoenergy contours.
        # unit vector along force at beads: 0
        f0 = mscaled_f[0] / npnorm(mscaled_f[0])
        # unit vector along force at beads: nimage - 1
        f1 = mscaled_f[nimage - 1] / npnorm(
            mscaled_f[nimage - 1]
        )

        end_beads_energy_constraint_force = np.zeros([2, 3 * natoms])
        # kappa * (V(r) - E) * \hat{f}(r) for beads 0
        end_beads_energy_constraint_force[0] = (
            f0 * left_kappa * (beads_energy[0] - self.instanton_path_energy)
        )  
        # kappa * (V(r) - E) * \hat{f}(r) for beads n-1.
        end_beads_energy_constraint_force[1] = (
            f1 * right_kappa * (beads_energy[nimage - 1] - self.instanton_path_energy)
        ) 
        self.end_beads_energy_constraint_forces = end_beads_energy_constraint_force

        # for internal beads, using simplified string method,
        # we do not project out parallel component of action forces.
        for ii in range(1, nimage - 1):
            neb_optimization_force[ii] = self.action_forces[ii]

        self.transverse_force = np.zeros([nimage, 3 * natoms])
        for ii in range(1, nimage -1):
            self.transverse_force[ii] = (self.action_forces[ii]
                - np.dot(self.action_forces[ii], self.btau[ii]) * self.btau[ii])

        # for two end beads.
        # add energy constraint force for two end beads.
        neb_optimization_force[0] = end_beads_energy_constraint_force[0]
        neb_optimization_force[nimage - 1] = end_beads_energy_constraint_force[1]

        # add force term to end beads to make sure the path along end beads aligns with gradient of potential.
        # spring force for end bead 1
        unit_vec_1 = (mscaled_q[1] - mscaled_q[0]) / npnorm(
            mscaled_q[1] - mscaled_q[0]
        )
        spring_force_bead0 = (
            unit_vec_1 * np.linalg.norm(self.action_forces[1])
        )
        neb_optimization_force[0] = neb_optimization_force[0] + (
            spring_force_bead0 - np.dot(spring_force_bead0, f0) * f0
        )

        # spring force for end bead 2
        unit_vec_2 = (mscaled_q[nimage - 2] - mscaled_q[nimage - 1]) / npnorm(
            mscaled_q[nimage - 2] - mscaled_q[nimage - 1]
        )
        spring_force_bead1 = (
            unit_vec_2 * np.linalg.norm(self.action_forces[-2])
        )
        neb_optimization_force[nimage -1] = neb_optimization_force[nimage -1] + (
            spring_force_bead1 - np.dot(spring_force_bead1, f1) * f1
        )

        return neb_optimization_force


# ------ End code for Gradient Mapper -------

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

        self.fix_dofs = np.array(nebmover.optarrays["fix_dofs"])

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
        self.gpr_rigid_internal_dofs_cutoff = nebmover.options["gpr_rigid_internal_dofs_cutoff"]
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
        self.ridge_regularization_alpha = nebmover.optarrays["ridge_regularization_alpha"]
        self.gpr_covar_inverse_nugget = nebmover.optarrays["gpr_covar_inverse_nugget"]

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
            np.copy(self.neb_beads.q),
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
                  gpr model to predict force along instanton path is: {time_elapsed} min")
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

        x_list, v_list, t_list, r_list, v_r_list, pot_list = ([] for _ in range(6))
        data_lists = {
            "x": x_list,
            "v": v_list,
            "t": t_list,
            "r": r_list,
            "v_r": v_r_list,
            "pot": pot_list
        }
        for key, value in zip(data_lists.keys(), [x, v, t, r_distance, v_r, pot]):
            data_lists[key].append(value)

        dr = 1000

        while dr > 0:
            old_r_distance = r_distance
            # r is normalized distance along path, in the range of [0, 1]
            t, r_distance, v_r, x, v = self.classical_dynamics_step(t, r_distance, v_r)

            dr = r_distance - old_r_distance
            # check energy conservation
            shifted_V, _, _, _ = self.gpr_model.predict_latent_function(np.array([x]))
            pot = shifted_V[0] + self.energy_shift

            for key, value in zip(data_lists.keys(), [x, v, t, r_distance, v_r, pot]):
                data_lists[key].append(value)

        for key in data_lists.keys():
            data_lists[key] = np.array(data_lists[key])

        x_list, v_list, t_list, r_list, v_r_list, pot_list = (data_lists[key] for key in ["x", "v", "t", "r", "v_r", "pot"])

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
        v = self.cubic_spline(r_distance, nu= 1) * v_r

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
            ipi.utils.nebinstgprtool.read_training_data(prefix= neb_final_gpr_folder)
        )
        training_V_shifted = training_V - self.energy_shift
        training_grad = -training_forces

        gpr_fixed_internal_dofs = ipi.utils.nebinstgprtool.read_fixed_internal_dofs(
            neb_final_gpr_folder
        )

        # initialize GPR model with training data read from the end of 'neb' stage run.
        self.gpr_model = ipi.utils.gprtools.GPModelWithDerivativesWrapper(
            cartesian_coordinate_x,
            training_V_shifted,
            training_grad,
            self.rp_beads.natoms,
            self.coordinate_transformer,
            self.fix_dofs,
            gpr_SE_kernel_number=self.gpr_SE_kernel_number,
            kernel_outputscale=self.gpr_kernel_outputscale,
            kernel_lengthscale_ratio=self.gpr_kernel_lengthscale_ratio,
            noise_std=self.gpr_noise_std,
            train_bool= False,
            gpr_fix_internal_dofs_bool= self.gpr_fix_internal_dofs_bool,
            gpr_fix_internal_dofs_cutoff= self.gpr_fix_internal_dofs_cutoff,
            gpr_fixed_internal_dofs= gpr_fixed_internal_dofs,
            singular_value_cutoff=  self.gpr_covar_inverse_nugget
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
        
        # store fixed internal dofs.
        ipi.utils.nebinstgprtool.store_fixed_internal_dofs_gpr_model(
            self.gpr_model,
            prefix= neb_final_gpr_folder
        )

        # test training results.
        _, predicted_grad, _, _ = (
            self.gpr_model.predict_latent_function(
                cartesian_coordinate_x
            )
        )
        predicted_forces = - predicted_grad 
        df = np.linalg.norm(
            training_forces - predicted_forces, 
            axis= 1
        )
        ab_initio_force_amplitude = np.linalg.norm(training_forces, axis= 1)
        df_error = df / ab_initio_force_amplitude
        print(f"@gpr_model: relative training error for force: {df_error}")

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
                self.rp_beads.nbeads
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
                self.train_grad_model_bool
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
            new_rigid_mode_rp_force = self.rp_forces.copy(new_rigid_mode_rp_bead, self.dcell)

            self.selective_hessian_calculator.rigid_modes_hessian_preprocess(
                prefix= self.read_gpr_hessian_folder,
                new_train_x= new_train_x_rigid_mode,
                new_rp_bead = new_rigid_mode_rp_bead,
                new_rp_force = new_rigid_mode_rp_force,
                new_rigid_mode_bead_index= self.new_hessian_data_index_rigid_mode
            )
        else:
            self.selective_hessian_calculator.rigid_modes_hessian_preprocess(
                prefix= self.read_gpr_hessian_folder,
                new_rigid_mode_bead_index= self.new_hessian_data_index_rigid_mode
            )

    def construct_new_gpr_hessian_model(self,
                                        candidate_hessian_point_x):
        """
        """
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

        
        # use the first data point as the reference point for mean function 
        # when constructing gpr model with hessian
        first_hessian_data_x = candidate_hessian_point_x[self.new_hessian_data_index[0]]
        
        new_beads = Beads(self.neb_beads.natoms, 1)
        new_forces = self.rp_forces.copy(new_beads, self.dcell)
        new_beads.q[0] = first_hessian_data_x
        
        ref_V_shifted = dstrip(new_forces.pots).copy() - self.energy_shift
        ref_grads = -dstrip(new_forces.f).copy()[0] 

        if self.selective_hessian_bool:
            # initialize the selective_hessian_calculator.
            self.construct_selective_hessian_calculator(candidate_hessian_point_x)

            ref_hessians = self.selective_hessian_calculator.get_hessian(
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

        # For testing the error induced by forward and backward transformation of gradient and hessian.
        ipi.utils.nebinstgprtool.analyze_transformation_between_cartesian_coord_and_internal_coord(
            np.array([first_hessian_data_x]), np.array([ref_grads]), np.array([ref_hessians]), self.coordinate_transformer
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
                self.fix_dofs,
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
                gpr_fix_internal_dofs_cutoff= self.gpr_fix_internal_dofs_cutoff,
                gpr_rigid_internal_dofs_cutoff= self.gpr_rigid_internal_dofs_cutoff,
                ridge_regularization_alpha= self.ridge_regularization_alpha,
                singular_value_cutoff= self.gpr_covar_inverse_nugget
            )
        )

        # test the training error of GPR model.
        ipi.utils.nebinstgprtool.analyze_train_error(self.gpr_hessian_model)

        # After train the model with only potential and gradient,
        # the hyper-parameter should be close to the minimum point after adding hessian data.
        # Now add hessian data & re-train the model.
        new_pots = dstrip(new_forces.pots).copy()
        new_grads = np.array([ref_grads])
        new_hessians = np.array([ref_hessians])
        new_hessian_point_x = np.array([first_hessian_data_x])
        print("We are going to train the gpr model with hessian data.\
                This can be expensive. To add data without training the model, set train_hessian_model_bool= False ")
        ipi.utils.nebinstgprtool.add_hessian_data_to_model(
            self.gpr_hessian_model,
            new_hessian_point_x,
            new_pots,
            new_grads,
            new_hessians,
            self.energy_shift,
            retrain_bool= self.train_hessian_model_bool
        )

        # test the training error of GPR model.
        ipi.utils.nebinstgprtool.analyze_train_error(self.gpr_hessian_model)
        pass
    
    

    def load_gpr_hessian_model(self,
                               candidate_hessian_point_x):
        """
        load gpr hessian model. The hessian are already computed.
        """
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

        gpr_fixed_internal_dofs = ipi.utils.nebinstgprtool.read_fixed_internal_dofs(self.read_gpr_hessian_folder)

        training_V_shifted = training_V - self.energy_shift
        training_grads = -training_forces


        if self.selective_hessian_bool:
            # initialize the selective hessian calculator to compute hessian along rigid modes.
            self.construct_selective_hessian_calculator(candidate_hessian_point_x)
            # update the hessian along the rigid mode.
            hessian_data_list = self.selective_hessian_calculator.update_hessian_rigid_modes(
                cartesian_coordinate_x[hessian_index_list],
                training_forces[hessian_index_list],
                hessian_data_list
            )

        # choose the first data point with hessian information as the reference point for mean function.
        ref_x = cartesian_coordinate_x[hessian_index_list[0]]
        ref_V_shifted = np.array([training_V_shifted[hessian_index_list[0]]])
        ref_grads = training_grads[hessian_index_list[0]]
        ref_hessians = hessian_data_list[0]
        
        #For testing the error induced by forward and backward transformation of gradient and hessian.
        ipi.utils.nebinstgprtool.analyze_transformation_between_cartesian_coord_and_internal_coord(
            np.array([ref_x]), np.array([ref_grads]), np.array([ref_hessians]), self.coordinate_transformer
        )

        self.gpr_hessian_model = (
            ipi.utils.gpr_hessian_tools.GPModelWithHessiansWrapper(
                cartesian_coordinate_x,
                training_V_shifted,
                training_grads,
                hessian_data_list,
                hessian_index_list,
                self.rp_beads.natoms,
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
                ridge_regularization_alpha= self.ridge_regularization_alpha,
                singular_value_cutoff= self.gpr_covar_inverse_nugget
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
            if (not (self.add_new_hessian_data_bool or self.add_new_grad_data_bool)) and self.train_hessian_model_bool:
                print("We are going to train the gpr model with hessian data.\
                This can be expensive. To add data without training the model, set train_hessian_model_bool= False ")
                self.gpr_hessian_model.train_model() 
                ipi.utils.nebinstgprtool.store_training_hyperparameter_in_gpr_hessian_model(
                    self.gpr_hessian_model, self.read_gpr_hessian_folder
                )
        else:
            print("We are going to train the gpr model with hessian data.\
                    This can be expensive. To add data without training the model, set train_hessian_model_bool= False ")
            self.gpr_hessian_model.train_model() 

            ipi.utils.nebinstgprtool.store_training_hyperparameter_in_gpr_hessian_model(
                    self.gpr_hessian_model, self.read_gpr_hessian_folder
            )

        # store fixed internal dofs.
        ipi.utils.nebinstgprtool.store_fixed_internal_dofs_gpr_hessian_model(
            self.gpr_hessian_model,
            self.read_gpr_hessian_folder
        )

        ipi.utils.nebinstgprtool.analyze_train_error(self.gpr_hessian_model)

    def cross_validate_gpr_hessian_model(self,
                                         candidate_hessian_point_x):
        """
        read training data (potential V, gradient, hessians) from folder. 
        split data into training set and cross validation set.
        Perform the cross validation. 
        """
        print("Cross validate the gpr hessian model.")
        print("\n")
        print(
                "read_gpr_hessian_folder provided. \
                Will read potential & gradients & hessians from folder and create gpr_hessian model."
        )
        # read data from read_gpr_hessian_folder.
        (cartesian_coordinate_x,
         potential_data,
         force_data,
         hessian_index_list,
         hessian_data_list,
        ) = ipi.utils.nebinstgprtool.read_training_data_with_hessian(
            self.read_gpr_hessian_folder
        )

        if self.selective_hessian_bool:
            # initialize the selective hessian calculator to compute hessian along rigid modes.
            self.construct_selective_hessian_calculator(candidate_hessian_point_x)
            # update the hessian along the rigid mode.
            hessian_data_list = self.selective_hessian_calculator.update_hessian_rigid_modes(
                cartesian_coordinate_x[hessian_index_list],
                force_data[hessian_index_list],
                hessian_data_list
            )
        

        train_set, cv_set = ipi.utils.nebinstgprtool.split_train_cv_data(
            cartesian_coordinate_x,
            potential_data,
            force_data,
            hessian_index_list,
            hessian_data_list,
            training_ratio = 0.8
        )
        # training data
        train_x, training_V, training_forces, train_hessian_index_list, train_hessian_data_list = train_set 
        training_V_shifted = training_V - self.energy_shift
        # cross validation data.
        cv_x, cv_V, cv_force, cv_hessian_index_list, cv_hessian_data = cv_set 
        

        gpr_fixed_internal_dofs = ipi.utils.nebinstgprtool.read_fixed_internal_dofs(self.read_gpr_hessian_folder)
        training_grads = - training_forces 
        
        # choose the first data point with hessian information as the reference point for mean function.
        ref_x = cartesian_coordinate_x[hessian_index_list[0]]
        ref_V_shifted = np.array([training_V_shifted[hessian_index_list[0]]])
        ref_grads = training_grads[hessian_index_list[0]]
        ref_hessians = hessian_data_list[0]

        # use training data to create gpr_hessian_model
        self.gpr_hessian_model = (
            ipi.utils.gpr_hessian_tools.GPModelWithHessiansWrapper(
                train_x,
                training_V_shifted,
                training_grads,
                train_hessian_data_list,
                train_hessian_index_list,
                self.rp_beads.natoms,
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
                ridge_regularization_alpha= self.ridge_regularization_alpha,
                singular_value_cutoff= self.gpr_covar_inverse_nugget
            )
        )

        # load trained model parameter.
        model_hyperparameter_exists = \
            ipi.utils.nebinstgprtool.load_training_hyperparameter_for_gpr_hessian_model(
                self.gpr_hessian_model,
                self.read_gpr_hessian_folder
        )
        self.gpr_hessian_model.train_model() 
        
        ipi.utils.nebinstgprtool.store_training_hyperparameter_in_gpr_hessian_model(
            self.gpr_hessian_model, self.read_gpr_hessian_folder
        )

        # analyze training data error.
        ipi.utils.nebinstgprtool.analyze_train_error(self.gpr_hessian_model)

        # analyze cross validation data error.
        if len(cv_x) > 0:
            cv_V_shifted = cv_V - self.energy_shift
            cv_grads = - cv_force 
            
            ipi.utils.nebinstgprtool.analyze_cross_validation_error(
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
        start_time = timer()

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

        end_time = timer()
        time_elapsed = (end_time - start_time) / 60
        print(f"time elapsed for training hessian model is: {time_elapsed} min." )

    def add_new_hessian_data(self):
        """
        (1) compute ab initio hessian at new hessian data index.
        (2) add new hessian data into gpr_hessian_model.
        """
        if os.path.exists( os.path.join(self.read_gpr_hessian_folder, "candidate_hessian_data_info.h5") ):
            ab_initio_hessian_file_exists = True 
        else:
            ab_initio_hessian_file_exists = False

        if ab_initio_hessian_file_exists:
            # read candidate_hessian_point_x, hessian_index_in_candidate_list from self.read_gpr_hessian_folder.
            (candidate_hessian_point_x, self.hessian_index_in_candidate_list) = (
                ipi.utils.nebinstgprtool.read_candidate_hessian_data_coordinate(
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
            
            if not ab_initio_hessian_file_exists:
                # the first index of new hessian data index is already used when constructing the model.
                self.hessian_index_in_candidate_list = np.array([self.new_hessian_data_index[0]])
                self.new_hessian_data_index = self.new_hessian_data_index[1:]

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
                if self.selective_hessian_bool:
                    new_hessians = self.selective_hessian_calculator.get_hessian(
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


                ipi.utils.nebinstgprtool.add_hessian_data_to_model(
                    self.gpr_hessian_model,
                    new_hessian_point_x,
                    new_pots,
                    new_grads,
                    new_hessians,
                    self.energy_shift,
                    retrain_bool= False,
                )
        
        return candidate_hessian_point_x, ab_initio_hessian_file_exists 

    def add_new_grad_data(self):
        """
        """
        if os.path.exists( os.path.join(self.read_gpr_hessian_folder, "candidate_grad_data_info.h5") ):
            ab_initio_grad_file_exists = True 
        else:
            ab_initio_grad_file_exists = False

        if ab_initio_grad_file_exists:
            (candidate_grad_point_x, self.grad_index_in_candidate_list) = (
                ipi.utils.nebinstgprtool.read_candidate_grad_data_coordinate(
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
            new_forces = self.rp_forces.copy(new_beads, self.dcell)
            new_beads.q = new_grad_point_x 

            # compute ab initio potentials and forces.
            new_pots = new_forces.pots 
            new_grads = -dstrip(new_forces.f).copy() 

            # add potential and gradient data into the gpr model.
            ipi.utils.nebinstgprtool.add_potential_grad_data_to_model(
                self.gpr_hessian_model,
                new_grad_point_x,
                new_pots,
                new_grads,
                self.energy_shift,
                retrain_bool= False
            )
        
        return candidate_grad_point_x, ab_initio_grad_file_exists

    def store_ab_initio_hessian_and_grad_data(self,
                                              ab_initio_grad_file_exists,
                                              candidate_grad_point_x,
                                              ab_initio_hessian_file_exists,
                                              candidate_hessian_point_x):
        """
        store the computed ab initio gradient and hessian data into data folder.
        """
        # if we have updated the data, we store the data set and training hyper-parameters to a given folder.
        if self.add_new_hessian_data_bool or self.add_new_grad_data_bool:
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
            # store fixed internal dofs in the gpr model
            ipi.utils.nebinstgprtool.store_fixed_internal_dofs_gpr_hessian_model(
                self.gpr_hessian_model,
                self.data_destination_folder
            )

            if self.selective_hessian_bool:
                # store the information about hessian along rigid mode.
                self.selective_hessian_calculator.store_rigid_dofs_hessian(self.data_destination_folder)

            if self.add_new_hessian_data_bool:
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
            else:
                if ab_initio_hessian_file_exists:
                    # store candidate_hessian_point_x, hessian_index_in_candidate_list in data destination folder.
                    # because we do not have new data, this is equivalent to copy the hessian file.
                    ipi.utils.nebinstgprtool.store_candidate_hessian_data_coordinate(
                        candidate_hessian_point_x,
                        self.hessian_index_in_candidate_list,
                        self.data_destination_folder,
                    )

            if self.add_new_grad_data_bool:
                # update the grad index with newly computed data point.
                self.grad_index_in_candidate_list = np.concatenate(
                    [self.grad_index_in_candidate_list, self.new_grad_data_index]
                )

                # store candidate_grad_point_x, grad_index_in_candidate_list in data destination folder.
                ipi.utils.nebinstgprtool.store_candidate_grad_data_coordinate(
                    candidate_grad_point_x,
                    self.grad_index_in_candidate_list,
                    self.data_destination_folder
                )                    
            else:
                if ab_initio_grad_file_exists:
                    # store candidate_grad_point_x, grad_index_in_candidate_list in data destination folder.
                    # because we do not have new data, this is equivalent to copy the grad file.
                    ipi.utils.nebinstgprtool.store_candidate_grad_data_coordinate(
                        candidate_grad_point_x,
                        self.grad_index_in_candidate_list,
                        self.data_destination_folder
                    )

        else:
             if len(self.new_hessian_data_index_rigid_mode) > 0 and self.selective_hessian_bool:
                 self.selective_hessian_calculator.store_rigid_dofs_hessian(self.read_gpr_hessian_folder)
                    

    def add_new_hessian_and_grad_data(self):
        """
        (1) compute the new ab initio hessian at new_hessian_data_index.
        (2) Add new hessian data into gpr_hessian_model
        (3) store the updated data set into new folder.
        """
        self.data_destination_folder = self.read_gpr_hessian_folder

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
        candidate_hessian_point_x, ab_initio_hessian_file_exists = self.add_new_hessian_data()

        # Now we add ab initio grad data along the path into the gpr model.
        candidate_grad_point_x, ab_initio_grad_file_exists = self.add_new_grad_data() 

        # train the model.
        if self.add_new_hessian_data_bool or self.add_new_grad_data_bool:
            if self.train_hessian_model_bool:
                print("We are going to train the gpr model with hessian data.\
                    This can be expensive. To add data without training the model, set train_hessian_model_bool= False ")
                start_t = timer()

                self.gpr_hessian_model.train_model()

                end_t = timer()
                time_elapsed = (end_t - start_t) / 60
                print(f"the elapsed time for re-training the model is {time_elapsed} min.")

            ipi.utils.nebinstgprtool.analyze_train_error(self.gpr_hessian_model)
            pass
                
        # store the computed ab inito gradient and hessian data if we compute new data point. 
        self.store_ab_initio_hessian_and_grad_data(ab_initio_grad_file_exists,
                                                candidate_grad_point_x,
                                                ab_initio_hessian_file_exists,
                                                candidate_hessian_point_x)

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

        _, _, hessians_q, _, _, _ = self.gpr_hessian_model.predict_latent_function(
            coord, hessian_data_point_index, internal_coordinate_bool=True
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

        # start classical dynamics along minimum action path (MAP) on the inverted potential.
        t_list, v_list, x_list = self.classical_dynamics_along_MAP()

        # interpolate the ring polymer beads from the generated trajectory.
        self.interpolate_ring_polymer_beads(t_list, v_list, x_list)

        # code to test the GPModelWithHessianWrapper
        if self.final_hessian_bool:
            if self.ab_initio_hessian_bool:
                # compute ab initio hessian of all ring polymers.
                # Only use this option for benchmark.
                self.compute_ring_polymer_hessian()

            else:
                # create gpr hessian model either reading data from input file or using training data from gpr model.
                self.construct_gpr_hessian_model()

                # add new hessian data into GPR model.
                # the location of new hessian data is given by self.new_hessian_data_point_index.
                # candidate_hessian_point_x spaced with equal distance along the path.
                self.add_new_hessian_and_grad_data()

                # predict hessians of ring polymer beads using Gaussian Process Regression.
                # The result is stored in self.rp_hessians, which will be stored in RESTART file for post-processing.
                self.predict_ring_polymer_hessians_using_gpr()

