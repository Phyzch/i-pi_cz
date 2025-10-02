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
import ipi.utils.nebinstgprtool
from ipi.utils.nebinstool import RK4
import gpr.internal.CoulombInternal # 1/|ri-rj| : Coloumb matrix.
import gpr.internal.ZmatrixInternal # primitive internal coordinate.
import gpr.gprtools 
import gpr.gpr_hessian_tools
import ipi.utils.mintools
import os
from timeit import default_timer as timer
import threading 
import ipi.utils.hessfasttools
from scipy.interpolate import CubicSpline

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
        cal_type= "rate",
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
        gpr_absolute_force_error_criterion=0.001,
        gpr_absolute_potential_error_criterion= 1e-4,
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
        Hessian_interpolation= "GPR",
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
        # type of calculation. rate or tunneling splitting.
        self.options["cal_type"] = cal_type 

        self.options["asr"] = asr
        self.options["stage"] = stage
        self.options["tolerances"] = tolerances
        self.options["alt_out_step"] = alt_out  # step to output geometry.
        self.options["prefix"] = prefix
        self.options["final_hessian_bool"] = final_hessian_bool
        self.options["ab_initio_hessian_bool"] = ab_initio_hessian_bool
        self.options["Hessian_interpolation"] = Hessian_interpolation
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
        else:
            raise ValueError("The opt Value does not match any existing options. Please choose either neb/string/improved_string.")
        
        # choose gradient mapper (compute optimization gradient) based on the optimizer we provide in input.xml:
        if self.options["opt"] == "neb":
            self.gm = LINEBGradientMapper()
        elif self.options["opt"] == "string":
            self.gm = StringGradientMapper()
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
        self.optarrays["gpr_absolute_potential_error_criterion"] = (
            gpr_absolute_potential_error_criterion
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

        # instanton temperature available
        self.instanton_temperature_avail = False

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

        # Check if we enter the program directly into "instanton" stage:
        if self.options["stage"] == "instanton" and step == 0:
            print("Skip neb stage. Go directly into instanton stage. \n")

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
        beads_pots, _, _, _ = self.gpr_model.predict_latent_function(self.beads.q)
        # print initial geometry and energy of neb path.
        ipi.utils.nebinstool.print_neb_instanton_geo(
            self.options["prefix"] + "_initial_",
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
        self.early_stop_bool = early_stop_bool
        self.outrange_bead_index_list = outrange_bead_index_list

        print("optimization step so far for neb stage: " + str(SharedData.inner_loop_optimization_step))

    def instanton_stage_motion(self, step):
        """
        generate instanton ring polymer beads from minimum action path found by NEB.
        """
        info(
            "Now generate instanton path from Minimum Action Path (MAP) found by NEB."
        )
        print("total optimization step for neb stage: " + str(SharedData.inner_loop_optimization_step))

        if not self.instanton_temperature_avail:
            # first compute the temperature along the instanton path and generate ring polymer beads along the path.
            # once done, we predict gpr hessian in the next step.
            self.rp_map.generate_ring_polymer_beads(self.beads, step)
            # set temperature available = True to enable follow up Hessian prediction.
            self.instanton_temperature_avail = True 
        else:
            # use GPR/cubic spline to predict hessians for ring polymer beads or compute it ab-initio way.
            self.rp_map.generate_hessian_along_instanton_path()
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

    def _generate_initial_training_data(self):
        """
        Compute potential and force for initial training data.
        We only compute 3 data point (including end beads) as initial training data to avoid computational cost scales with # of beads we add.
        """
        # choose two end beads and the bead in the middle as initial training data.
        # We will train the GPR model to optimize hyperparameter using the initial data.
        initial_bead_number = 10 
        nbeads = self.beads.nbeads
        # compute forces one by one in case we use the 'centroid' trick to avoid 
        # computing forces for all beads to converge the path.
        
        train_x = np.zeros([initial_bead_number, np.shape(self.beads.q)[1]])
        train_V = np.zeros([initial_bead_number])
        train_grad = np.zeros([initial_bead_number, self.beads.natoms * 3])
        bead_index = np.linspace(0, nbeads - 1, initial_bead_number).astype(int)
        for i in range(initial_bead_number):
            train_x[i] = self.beads.q[bead_index[i]]
            self.gpr_beads.q[0] = train_x[i]
            train_V[i] = np.copy(self.gpr_forces.pots)[0] - self.optarrays["energy_shift"]
            train_grad[i] = - np.copy(dstrip(self.gpr_forces.f))[0]

        # potential energy has to shift relative to the energy_shift for training.
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
        if ((not read_gpr_training_data_bool) and (self.options["stage"] == "neb")):
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
        
        return train_x, train_V, train_grad 

        

    def bind_gpr_model(self, gpr_model: gpr.gprtools.GPModelWithDerivativesWrapper, coordinate_transformer: gpr.internal.ZmatrixInternal.non_redundant_coordinate_transformer):
        """
        bind the gpr model and coordinate_transformer to the LINEGradientMapper class
        the LINEBGradientMapper will perform LI-NEB using gpr generated potential and force.
        """
        self.gpr_model = gpr_model 
        self.coordinate_transformer = coordinate_transformer

        self.gm.gpr_model = gpr_model
        self.gm.coordinate_transformer = coordinate_transformer 

        self.rp_map.gpr_model = gpr_model 
        self.rp_map.coordinate_transformer = self.coordinate_transformer 

        self.optimizer.gpr_model = gpr_model 
        self.optimizer.coordinate_transformer = coordinate_transformer

    def bind_gpr_hessian_model(self, gpr_hessian_model: gpr.gpr_hessian_tools.GPModelWithHessiansWrapper):
        """
        bind the gpr hessian model to rp_map (ring polymer minimum action path)
        """
        self.rp_map.gpr_hessian_model = gpr_hessian_model

    def neb_stage_exit_step(self, step):
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

        beads_pots = self.gm.beads_energy
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

        self.options["cal_type"] = ens.options["cal_type"]

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

        # if cal_type == 'splitting', realign two end beads with respect to the bead it connects to.
        # See J. Chem. Theory Comput. 2016, 12, 787−803
        if self.options["cal_type"] == "splitting":
            m = dstrip(self.beads.m)
            # re-orient bead 0 to align with bead 1.
            self.x[0] = ipi.utils.nebinstgprtool.align_molecules_quaternion(self.x[1], self.x[0], weight= m)
            # re-orient bead -1 to align with bead -2.
            self.x[-1] = ipi.utils.nebinstgprtool.align_molecules_quaternion(self.x[-2], self.x[-1], weight= m)

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

        # coordinate of the reactant. For the tunneling splitting calculation.
        self.q_r1 = self.q_r2 = None 

    def bind(self, ens: MAPNEBGPRMover):
        """
        """
        self.dbeads = ens.beads.copy()
        self.dcell = ens.cell.copy()
        self.fixatoms = ens.fixatoms.copy()
        self.fix_dofs = ens.optarrays["fix_dofs"]
        self.asr = ens.options["asr"]
        # type of calculation: rate or tunneling_splitting.
        self.cal_type = ens.options["cal_type"]
        
        # use it for force evaluation near the potential minima for tunneling splitting calculation.
        self.gpr_absolute_force_error_criterion = ens.optarrays["gpr_absolute_force_error_criterion"]
        self.gpr_absolute_potential_error_criterion = ens.optarrays["gpr_absolute_potential_error_criterion"]

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
        self.forces = ens.forces.copy(self.rbeads, self.dcell)

        self.kappa = ens.optarrays[
            "kappa"
        ]  # bind end beads energy constraint constant kappa from NEBMover.

        self.energy_shift = ens.optarrays["energy_shift"]

        self.ab_initio_pot = np.zeros([self.dbeads.nbeads])
        self.ab_initio_force = np.zeros([self.dbeads.nbeads, 3 * self.dbeads.natoms])

        self.ENO_order = ens.optarrays["ENO_order"]
    
    def read_reactant_info(self):
        if self.cal_type == "splitting":
            # read end point reactant potential, force and hessians.
            reactant1_file = "reactant1_data.h5"
            reactant2_file = "reactant2_data.h5"
            self.q_r1, self.V1, self.f_r1, self.h_r1 = ipi.utils.nebinstgprtool.read_reactant_data(reactant1_file)
            self.q_r2, self.V2, self.f_r2, self.h_r2 = ipi.utils.nebinstgprtool.read_reactant_data(reactant2_file) 

            self.q_r1 = self.q_r1[0]
            self.q_r2 = self.q_r2[0]
            self.pot_cutoff = self.gpr_absolute_potential_error_criterion
            self.force_cutoff = self.gpr_absolute_force_error_criterion

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
        if self.cal_type == "splitting":
            if self.q_r1 is None:
                # read reactant coord, pot, force & hessians from the file.
                self.read_reactant_info()

        test_x = np.copy(self.rbeads.q)
        beads_potential_shift, beads_potential_grad_x, _, _ = (
            self.gpr_model.predict_latent_function(test_x)
        )

        beads_forces = - beads_potential_grad_x
        # the predicted potential is the one relative to the energy shift.
        beads_potential = beads_potential_shift + self.energy_shift

        # for the tunneling splitting calculation. The bead close to reaction minimum. 
        if self.cal_type == "splitting":
            beads_potential[0] = self.V1 
            beads_forces[0] = self.f_r1 

            beads_potential[-1] = self.V2 
            beads_forces[-1] = self.f_r2 

            # internal beads. in case if they are close to the reactant minimum.
            for i in range(1, self.dbeads.nbeads - 1):
                if beads_potential_shift[i] < self.pot_cutoff and np.linalg.norm(beads_forces[i]) < self.force_cutoff:
                    r1_distance = np.linalg.norm(test_x[i] - self.q_r1)
                    r2_distance = np.linalg.norm(test_x[i] - self.q_r2)
                    if (r1_distance < r2_distance):
                        beads_potential[i] = self.V1 + (-self.f_r1) @ (test_x[i] - self.q_r1).T + (test_x[i] - self.q_r1) @ self.h_r1 @ (test_x[i] - self.q_r1).T
                        beads_forces[i] = self.f_r1 + (-self.h_r1) @ (test_x[i] - self.q_r1)
                    else:
                        beads_potential[i] = self.V2 + (-self.f_r2) @ (test_x[i] - self.q_r2).T + (test_x[i] - self.q_r2) @ self.h_r2 @ (test_x[i] - self.q_r2).T 
                        beads_forces[i] = self.f_r2 + (-self.h_r2) @ (test_x[i] - self.q_r2)

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
            elif beads_energy[j] - self.instanton_path_energy < self.gpr_absolute_potential_error_criterion:
                # the gj_force_component will be unstable, set it to zero to avoid instability.
                gj_force_component = 0
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

        if self.cal_type == "splitting":
            # fix the end beads location. 
            neb_optimization_force[0] = np.zeros([3 * natom])
            neb_optimization_force[nimage - 1] = np.zeros([3 * natom])

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
        if self.cal_type == "rate":
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
        elif self.cal_type == "splitting":
            neb_optimization_force[0] = np.zeros([3 * natoms])
            neb_optimization_force[nimage - 1] = np.zeros([3 * natoms])

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

    def bind(self, nebmover: MAPNEBGPRMover):
        """
        bind function for RP_MAP
        nebmover: MAPNEBMover instance.
        """
        self.cal_type = nebmover.options["cal_type"]

        self.prefix = nebmover.options["prefix"]
        self.final_hessian_bool = nebmover.options["final_hessian_bool"]
        self.ab_initio_hessian_bool = nebmover.options["ab_initio_hessian_bool"]
        self.Hessian_interpolation = nebmover.options["Hessian_interpolation"]

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
        
        # bead object for instanton ring polymer
        if self.cal_type == "splitting":
            self.rp_beads = Beads(self.neb_beads.natoms, int(self.rp_bead_number / 2))
        else:
            self.rp_beads = Beads(
                self.neb_beads.natoms, self.rp_bead_number
            )  
        
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

        self.gpr_fix_internal_dofs_bool = nebmover.options["gpr_fix_internal_dofs_bool"]
        self.gpr_fix_internal_dofs_cutoff = nebmover.options["gpr_fix_internal_dofs_cutoff"]
        self.gpr_rigid_internal_dofs_cutoff = nebmover.options["gpr_rigid_internal_dofs_cutoff"]
        # bind the distance cutoff for training data for the gpr model
        self.distance_cutoff_for_training_data = (
            nebmover.options["distance_cutoff_for_training_data"]
        )

        # bind the file that we use to read hessian data
        self.read_gpr_hessian_folder = nebmover.options["read_gpr_hessian_folder"]
        # we use the same folder & data structure to read & store hessians for cubic spline interpolation.
        self.read_cubic_spline_interpolation_hessian_folder = self.read_gpr_hessian_folder

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

        # absolute force error tolerance for gpr model.
        self.gpr_absolute_force_error_criterion = nebmover.optarrays["gpr_absolute_force_error_criterion"]

        # absolute potential error tolerance for gpr model.
        self.gpr_absolute_potential_error_criterion = nebmover.optarrays["gpr_absolute_potential_error_criterion"]

# -------   for generating the ring polymer beads along the instanton path -------
    def classical_dynamics_along_MAP(self, neb_beads):
        """
        classical dynamics on the inverted potential -V(x)
        the final time will be 1/2 of the imaginary period.
        :param: neb_beads: beads in MAPNEBMover, with optimized geometry for Minimum Action Path.
        :return:  t_list: a list of time of trajectories.
                  v_list: a list of velocity of trajectories.
                  x_list: a list of coordinate of trajectories.
        """
        self.neb_beads.q[:] = neb_beads.q[:]  # initialize neb beads position.

        # Cubic interpolation of neb beads to form the path.
        self.cubic_spline = ipi.utils.nebinstool.path_cubic_spline_function(
            np.copy(self.neb_beads.q),
            np.copy(self.neb_beads.q)
        )

        print("use cubic interpolation to generate MAP path")

        # prediction the force using gpr or tylor expansion around reactant minimum.
        self.force_predictor = ipi.utils.nebinstgprtool.force_predictor(self.gpr_model,
                                                                        self.cal_type,
                                                                        self.gpr_absolute_force_error_criterion,
                                                                        self.gpr_absolute_potential_error_criterion,
                                                                        self.energy_shift)

        start_time = timer()
        
        t, r_distance = 0, 0  # time & normalized distance along path.
        x = np.copy(self.neb_beads.q[0])  # coordinate
        v = np.zeros([3 * self.neb_beads.natoms])  # velocity
        v_r = 0  # dr/dt. rate of change for r.

        # shifted_V, _, _, _ = self.gpr_model.predict_latent_function(np.array([x]))
        # pot = shifted_V[0] + self.energy_shift
        shifted_V = self.force_predictor.predict_V(np.array([x]), np.array([0]))
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

        step_num = 0
        
        # shift to move to potential minimum in case the potential is flat in the beginning.
        r_shift = 1e-4

        total_energy_list = []
        while dr > 0:
            old_r_distance = r_distance
            old_pot = pot 
            # r is normalized distance along path, in the range of [0, 1]
            t, r_distance, v_r, x, v = self.classical_dynamics_step(t, r_distance, v_r)

            dr = r_distance - old_r_distance

            if dr < 0 and r_distance < 0.1:
                # small error in force cause simulation fails.
                # simply move particle along the path and set t = 0.
                t = 0 
                r_distance = old_r_distance + r_shift
                dr = - dr
                v_r = 0 
                continue 

            # check energy conservation
            # shifted_V, _, _, _ = self.gpr_model.predict_latent_function(np.array([x]))
            # pot = shifted_V[0] + self.energy_shift
            shifted_V = self.force_predictor.predict_V(np.array([x]), np.array([r_distance]))
            pot = shifted_V[0] + self.energy_shift

            # only evolve half of the trajectory for tunneling splitting calculation.
            if self.cal_type == "splitting":
                 if pot <= old_pot and r_distance > 0.4:
                     break 

            for key, value in zip(data_lists.keys(), [x, v, t, r_distance, v_r, pot]):
                data_lists[key].append(value)
            
            step_num = step_num + 1
            if step_num % 100 == 0:
                print(f"step number: {step_num}")
                print(f"r_distance: {r_distance},  v_r: {v_r}")
                kinetic_energy = 0.5 * np.sum(self.m3 * np.power(v, 2))
                total_energy = kinetic_energy - shifted_V[0]
                print(f"total energy: {total_energy}") 
                total_energy_list.append(total_energy)
            


        for key in data_lists.keys():
            data_lists[key] = np.array(data_lists[key])

        r_list = data_lists["r"]
        t_list = data_lists["t"]
        print("Type of calculation: " + self.cal_type)
        print(f"final r at the end of dynamical evolution: {r_list[-1]}")

        print(f"total energy during evolution: {total_energy_list}")

        x_list, v_list, t_list, r_list, v_r_list, pot_list = (data_lists[key] for key in ["x", "v", "t", "r", "v_r", "pot"])
        
        self.analyze_classical_dynamics_along_MAP(v_list, t_list, pot_list)

        end_time = timer()
        time_elapsed = (
            end_time - start_time
        ) / 60  # time elapsed in minutes
        print("the running time for the constrained dynamics along the path is: " + str(time_elapsed) + " min.")

        return t_list, v_list, x_list, r_list, v_r_list

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
        param = [self.force_predictor, m3_matrix, self.cubic_spline]
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
        if self.cal_type == "rate":
            self.imag_time_period = (
                2 * t_list[-1]
            )  # the period of periodic motion is twice the time move from one end to another end.
        else:
            self.imag_time_period = 4 * t_list[-1]

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

    def interpolate_ring_polymer_beads(self, t_list, v_list, x_list, r_list, v_r_list, step):
        """
        interpolate ring polymer beads from the imaginary time trajectory along Minimum Action Path (MAP).
        t_list , v_list, x_list: list of time / velocity / trajectory from MD simulation along path.
        """
        # interpolate to get ring polymer position.
        rp_t_list, rp_x_list, rp_r_list = ipi.utils.nebinstool.interpolate_ring_polymer_beads(
            self.imag_time_period, 
            t_list, 
            x_list, 
            v_list,
            r_list,
            v_r_list, 
            self.rp_bead_number,
            self.cal_type
        )

        ipi.utils.nebinstool.print_instanton_rp_time(
            "rp_time_FINAL", self.imag_time_period, rp_t_list, self.output_maker
        )

        self.rp_beads.q = rp_x_list
        self.rp_r_list = rp_r_list 

        # print ring polymer instanton geometry.
        shifted_pots = self.force_predictor.predict_V(rp_x_list, rp_r_list)

        self.rp_beads_pots = shifted_pots + self.energy_shift

        ipi.utils.nebinstool.print_neb_instanton_geo(
            "instanton_along_MAP_FINAL",
            step,
            self.rp_beads.nbeads,
            self.rp_beads.natoms,
            self.neb_beads.names,
            self.rp_beads.q,
            self.rp_beads_pots,
            self.dcell,
            self.energy_shift,
            self.output_maker,
        )


    def generate_ring_polymer_beads(self, neb_beads, step):
        """
        Main function that compute ring-polymer beads from nudged elastic band Minimum action path.
        """
        # start classical dynamics along minimum action path (MAP) on the inverted potential.
        t_list, v_list, x_list, r_list, v_r_list = self.classical_dynamics_along_MAP(neb_beads)

        # interpolate the ring polymer beads from the generated trajectory.
        self.interpolate_ring_polymer_beads(t_list, v_list, x_list, r_list, v_r_list, step)

# -------- for generating ring polymer beads along the instanton path END-------

# ------ for hessian calculation ---------------
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

    def add_new_hessian_data(self, hessian_index_in_candidate_list = []):
        """
        add new hessian data for cubic spline interpolation.
        """
        if len(hessian_index_in_candidate_list) != 0:
            common_index = np.intersect1d(
                hessian_index_in_candidate_list,
                self.new_hessian_data_index
            )
            assert (len(common_index) == 0), "At least one data point in newly added hessian index  \
                coincide with the one point that we have already computed hessian."

        candidate_hessian_point_x, _ = (
            ipi.utils.nebinstool.path_equal_distance_interpolation(
                np.copy(self.neb_beads.q), self.candidate_hessian_data_number
            )
        )
        if len(self.new_hessian_data_index) == 0:
            raise ValueError("new hessian index should not be empty.")
        
        new_hessian_point_x = candidate_hessian_point_x[self.new_hessian_data_index]
        new_hessian_data_num = len(new_hessian_point_x)
        natoms = self.neb_beads.natoms
        new_beads = Beads(natoms, new_hessian_data_num)
        new_forces = self.rp_forces.copy(new_beads, self.dcell)
        new_beads.q = new_hessian_point_x

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

        return candidate_hessian_point_x, self.new_hessian_data_index, new_hessians 

    def generate_hessian_data_for_cubic_spline_interpolation(self):
        """
        generate fitting data for the ring polymer interpolation.
        """
        if self.read_cubic_spline_interpolation_hessian_folder != "None":
            (candidate_hessian_point_x, 
            hessian_index_in_candidate_list, 
            hessian_data_list) = ipi.utils.nebinstgprtool.read_cubic_spline_hessian_data(
                self.read_cubic_spline_interpolation_hessian_folder
                )

            # add_new_hessian_data.
            if self.add_new_hessian_data_bool:
                _, new_hessian_index_in_candidate_list, new_hessian_data_list = self.add_new_hessian_data(
                    hessian_index_in_candidate_list
                )
                hessian_index_in_candidate_list = np.concatenate([hessian_index_in_candidate_list,
                                                                new_hessian_index_in_candidate_list])
                hessian_data_list = np.concatenate([hessian_data_list,
                                                    new_hessian_data_list], axis= 0)
        else:
            # compute hessian from ab initio 
            if not self.add_new_hessian_data_bool:
                raise ValueError("Must provide hessian data index for " \
                "cubic spline interpolation if no hessian data is loaded.")
            else:
                (candidate_hessian_point_x, 
                 hessian_index_in_candidate_list,
                 hessian_data_list) = self.add_new_hessian_data()

        # store hessian data.
        if self.add_new_hessian_data_bool:
            hessian_number = len(hessian_index_in_candidate_list)
            prefix = "hessian# " + str(hessian_number)
            ipi.utils.nebinstgprtool.store_cubic_spline_hessian_data(
                prefix,
                candidate_hessian_point_x,
                hessian_index_in_candidate_list,
                hessian_data_list
            )

        return (candidate_hessian_point_x, hessian_index_in_candidate_list, hessian_data_list)

    def predict_ring_polymer_hessians_using_cubic_spline(self):
        """
        Predict ring polymer hessians using the cubic spline method.
        """
        (candidate_hessian_point_x, 
         hessian_index_in_candidate_list, 
         hessian_data_list) = self.generate_hessian_data_for_cubic_spline_interpolation()

        # sort hessian data 
        hessian_data_list = hessian_data_list[np.argsort(hessian_index_in_candidate_list), :]
        hessian_index_in_candidate_list = np.sort(hessian_index_in_candidate_list)       

        # compute r (path distance, range[0,1]) for beads.
        candidate_hessian_data_number = candidate_hessian_point_x.shape[0]
        _, candidate_hessian_point_r = ipi.utils.nebinstool.path_equal_distance_interpolation(
                                                                np.copy(self.neb_beads.q),
                                                               candidate_hessian_data_number
                                                               )
        
        hessian_point_r = candidate_hessian_point_r[hessian_index_in_candidate_list]

        # cubic spline interpolation with hessian data point along the instanton path.
        hessian_cs = CubicSpline(hessian_point_r, hessian_data_list, axis= 0)
        hessians = hessian_cs(self.rp_r_list)

        # reshape hessians ([nbeads, 3 * natoms, 3 * natoms])
        # to fit the shape of self.rp_hessian: [3 * natoms, nbeads * 3 * natoms]
        nbeads = self.rp_beads.nbeads
        natoms = self.rp_beads.natoms
        self.rp_hessian = np.reshape(
            np.transpose(hessians, (1, 0, 2)), [3 * natoms, nbeads * 3 * natoms]
        )

    def generate_hessian_along_instanton_path(self):
        """
        """
        # code to test the GPModelWithHessianWrapper
        if self.final_hessian_bool:
            if self.ab_initio_hessian_bool:
                # compute ab initio hessian of all ring polymers.
                # Only use this option for benchmark.
                self.compute_ring_polymer_hessian()

            else:
                if self.Hessian_interpolation == "GPR":
                    # predict hessians of ring polymer beads using Gaussian Process Regression.
                    # The result is stored in self.rp_hessians, which will be stored in RESTART file for post-processing.
                    self.predict_ring_polymer_hessians_using_gpr()
                elif self.Hessian_interpolation == "CubicSpline":
                    # use cubic spline method to interpolate ring polymer Hessians.
                    # The result is stored in self.rp_hessians, which will be stored in RESTART file for post-processing.
                    self.predict_ring_polymer_hessians_using_cubic_spline()
                else:
                    raise ValueError(f"unrecognized Hessian interpolation method: {self.Hessian_interpolation}")





