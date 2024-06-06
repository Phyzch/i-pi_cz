"""Holds the algorithms to perform nudged elastic band (NEB) calculations to find instanton path.
J. Chem. Phys. 148, 102334 (2018); https://doi.org/10.1063/1.5007180

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


np.set_printoptions(threshold=10000, linewidth=1000)  # Remove in cleanup

__all__ = ["LINEBGradientMapper", "MAPNEBMover", "RP_MAP"]


class MAPNEBMover(Motion):
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
    ):
        """Initialises NEBMover.

        Args:
           fixcom: An optional boolean which decides whether the centre of mass
              motion will be constrained or not. Defaults to False.
        """
        super(MAPNEBMover, self).__init__(fixcom=fixcom, fixatoms=fixatoms)

        # parameters to pass in from input.xml
        self.options = {}
        
        # mode for optimization
        self.options["mode"] = mode  

        self.options["stage"] = stage
        self.options["tolerances"] = tolerances
        self.options["alt_out_step"] = alt_out   # step to output geometry.
        self.options["prefix"] = prefix 
        self.options["final_hessian_bool"] = final_hessian_bool
        
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
        self.rp_map = RP_MAP()  # ring-polymer minimum action path.

        # variables for neb move
        self.velocity_mscaled = None 
        self.x = None 
        self.action = None 
        self.f_mscaled = None 

    def bind(self, ens, beads, nm, cell, bforce, prng, omaker):
        super(MAPNEBMover, self).bind(ens, beads, nm, cell, bforce, prng, omaker)
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

        self.nebgm.bind(self)
        self.rp_map.bind(self)
        
    def step(self, step=None):
        """Does one simulation time step.
        """

        info(" @NEB STEP %d, stage: %s" % (step, self.options["stage"]), verbosity.debug)

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
            self.nebgm.spring_k = self.optarrays["spring_k"]
            self.nebgm.kappa = self.optarrays["kappa"]

            # Only do it for initial calculation. Not for restart.
            self.optarrays["instanton_path_energy"] = self.optarrays["instanton_path_energy"] + self.optarrays["energy_shift"]  # shift the instanton path energy according to energy shift.
            self.nebgm.instanton_path_energy = self.optarrays["instanton_path_energy"]
            self.rp_map.instanton_path_energy = self.optarrays["instanton_path_energy"]


        # Check if we restarted a converged calculation (by mistake)
        if self.options["stage"] == "converged":
            softexit.trigger(
                status="success",
                message="neb calculation converged. Instanton geometry calculation finishes. Exiting simulation",
            )

        if self.options["stage"] == "neb":
            # use nudged elastic band method to find minmum action path.
            # then we will switch to the stage "instanton"
            self.step_neb(step)

        if self.options["stage"] == "instanton":

            # print neb beads geometry and energy.
            ipi.utils.nebinstool.print_neb_instanton_geo(
                self.options["prefix"] + "_neb_FINAL",
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

            # generate instanton ring polymer beads from minimum action path found by NEB.
            info("Now generate instanton path from Minimum Action Path (MAP) found by NEB.")
            self.rp_map.generate_ring_polymer_beads(self.beads, self.forces, step)

            # save the potential , q, temperature, hessian of instanton beads for RESTART.
            self.save_instanton_ring_polymer()

            # ! If we exit here, the RESTART file will not record the hessian and instanton geometry we just computed.
            # therefore, we set ["stage"] == "converged" and exit at next step.
            self.options["stage"] = "converged"
            
            
# --------- NEB method -----------------------
    def step_neb(self, step):
        n_activedim = self.beads.q[0].size - len(self.fixatoms) * 3
        nbeads = self.beads.nbeads
        dt = self.optarrays["time_step"]

        # For first step when we RESTART simulation or when step = 0 (just start simulation.)
        if np.all(self.velocity_mscaled) == None:
            self.neb_initialize()

        # check if spring_k and kappa value is appropriate.
        self.check_spring_k_kappa()

        self.print_geometry(step)
            
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

        
        # check convergence of calculation. 
        # # transverse gradient for interior beads.
        # grad_interior_beads_max = np.amax(np.abs(self.nebgm.neb_transverse_force))
        # # optimization gradient at end beads.
        # grad_end_beads_max = np.amax(np.abs([self.nebgm.neb_optimization_force[0], self.nebgm.neb_optimization_force[nbeads - 1]]))
        # grad_max = np.max([grad_end_beads_max, grad_interior_beads_max])

        grad_max = np.amax(npnorm(self.nebgm.neb_optimization_force, axis = 1))

        self.neb_instanton_exit(step, grad_max)

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



    def neb_initialize(self):
        info(
            " @NEB: calling NEBGradientMapper at step 0",
            verbosity.debug,
        )

        self.velocity_mscaled = np.zeros([self.beads.nbeads, 3 * (self.beads.natoms - len(self.fixatoms))])  # velocity of free moving particles on mass scaled coordinate.
        self.old_f_mscaled = np.zeros([self.beads.nbeads, 3 * (self.beads.natoms - len(self.fixatoms))])  # forces (in the nudged elastic band algorithm) from previous step on mass scaled coordinate
        self.x = np.copy(self.beads.q[:, self.fixatoms_mask])  # coordinate of free moving atoms
        self.old_x = None 
        self.action = None # current action
        self.old_action = None   # action at previous step 
        self.f_mscaled, self.action = self.nebgm(self.x)  # forces at current step on mass scaled coordinate 


    def neb_instanton_exit(self, step, grad_max):
        '''
        check the neb convergence and output info about convergence check
        '''
        tolerances = self.options["tolerances"]

        info("@Exit step : max force gradient {:4.2e} , (condition {:4.2e})".format(
                grad_max, tolerances["gradient"]
            ),
            verbosity.low
            )


       
        print("old action: " + str(self.old_action) + "  new action: " + str(self.action))
        print("inner product between tangent and force direction: " + str(self.nebgm.f_tau_inner_product[1:self.beads.nbeads - 1]))
        print("beads optimization gradient: " + str(npnorm(self.nebgm.neb_optimization_force, axis = 1)))
        print("beads potential relative to instanton path energy (eV): " + str( (self.nebgm.rforces.pots - self.optarrays["instanton_path_energy"]) * units.unit_to_user("energy", "electronvolt", 1)  ))
        print("distance between beads in mass scaled coordinate: " + str( self.nebgm.beads_mscaled_distance ))
        print("\n")

        # for debug
        # print("beads action gradient: " + str(npnorm(self.nebgm.action_forces, axis = 1)))
        # print("beads spring force: " + str(npnorm(self.nebgm.spring_forces, axis = 1) ))
        # print("beads energy constraint force at two ends: " + str(npnorm(self.nebgm.end_bead_energy_constraint_forces, axis = 1)) )
        # print("maximum force: " + str( np.amax(np.abs(self.nebgm.rforces.f)) ))

        print("\n")
        print("finish step {}".format(step))

        print("\n")

 
        if(
          grad_max <= tolerances["gradient"]
        ):
            info( "@Exit step: NEB_instanton: path optimization converged. Step %i \n" % step, verbosity.low)

            # print neb beads geometry and energy.
            ipi.utils.nebinstool.print_neb_instanton_geo(
                self.options["prefix"] + "_neb_FINAL",
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

            # this will make the program switch to "instanton" at next step() function.
            self.options["stage"] = "instanton"
            info("Now generate instanton path from Minimum Action Path (MAP) found by NEB.")

    def print_geometry(self, step):
        '''
        print beads geometry and beads energy.
        '''
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
                self.nebgm.rforces.pots,
                self.cell,
                self.optarrays["energy_shift"],
                self.output_maker
            )

    def save_instanton_ring_polymer(self):
        '''
        save the ring polymer instanton computed in RP_MAP class.
        Therefore, the result can be stored in RESTART file
        '''
        self.optarrays["instanton_temperature"] = self.rp_map.instanton_temp
        self.optarrays["instanton_bead_q"] = self.rp_map.rp_beads.q 
        self.optarrays["instanton_bead_pot"] = self.rp_map.rp_forces.pots
        self.optarrays["instanton_hessian"] = self.rp_map.rp_hessian 

        # print hessian
        if self.options["final_hessian_bool"]:
            ipi.utils.nebinstool.print_instanton_hess(
                self.options["prefix"] + "_FINAL",
                self.optarrays["instanton_hessian"],
                self.output_maker)

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


    def bind(self, ens):
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

    def initialize_rforces(self):
        '''
        initialize force engine for reduced beads rforces and potential.
        self.rforces depends on self.rbeads : self.rforces = ens.forces.copy(self.rbeads, self.dcell)
        updating self.rbeads will cause the program to reevaluate the potential & forces
        '''
        info(
            "Calculating all beads once to get potentials on the endpoints",
            verbosity.medium,
        )
        self.init_allpots = dstrip(self.dforces.pots).copy() # potential of all beads for initial configuration.

        # We want to be greedy about force calls,
        # so we transfer from full beads to the reduced ones.
        # initialize reduced beads force and pot:  self.rforces.pots & self.rforces.f
        tmp_f = self.dforces.f.copy()  # all beads forces.
        tmp_v = self.init_allpots.copy()  # all beads potential.
        self.rforces.transfer_forces_manual(
            new_q=[self.dbeads.q],
            new_v=[tmp_v],
            new_forces=[tmp_f],
        )

    def compute_tangent_vector(self, nimage, natom, mscaled_q, beads_energy):
        '''
        :param: nimage: number of replica images
        :param: natom: number of atoms (free moving)
        :param: bq: beads coordinate (mass_scaled coordinate)
        :param: beads_energy: energy of all beads.

        :return: btau: unit director for tangent vector of all internal beads in mass_scaled coordinates. (We do not need tangent vector for beads at two ends.)
        '''
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

    
    def compute_neb_action(self, nimage, mscaled_q, beads_energy):
        '''
        compute abbreviated action W. See eq.(10) in J. Chem. Phys. 148, 102334 (2018)
        Note in atomic unit, hbar = kb = 1. 
        
        :param: nimage: number of images (replicas)
        :param: mscaled_q: mass weighted coordinates for free moving atoms [nimag, 3 * natom]
        :param: beads_energy: energy of each beads (images).  size [nimage]
        
        :return: action: abbreviated action of the ring polymer path
        '''
        action = 0

        # sqrt(2 (V - E))
        action_each_bead = np.zeros([nimage])
        action_each_bead[1 : nimage - 1] = np.sqrt(2 * (beads_energy[1 : nimage - 1] - self.instanton_path_energy)) 
        if beads_energy[0] > self.instanton_path_energy:
            action_each_bead[0] = np.sqrt(2 * (beads_energy[0] - self.instanton_path_energy))
        if beads_energy[nimage - 1] > self.instanton_path_energy:
            action_each_bead[nimage - 1] = np.sqrt(2 * (beads_energy[nimage - 1] - self.instanton_path_energy))

        for j in range(1 , nimage):
            rj = mscaled_q[j]
            rj_1 = mscaled_q[j - 1]
            r_dist = npnorm(rj - rj_1)  
            action = action + 1/2 * (action_each_bead[j] + action_each_bead[j-1]) * r_dist 
        
        return action 
    
    def compute_neb_action_force(self, nimage, natom, mscaled_q, beads_energy, mscaled_f):
        '''
        compute the negative gradient of abbreviated action W. (for scaled coordinates.) See eq. (11) in J. Chem. Phys. 148, 102334 (2018).
        Note I will use the same symbol as given in the eq.(11) in the paper.

        :param: nimag: number of images (replica). scalar 
        :param: natom: number of freely moving atoms. scalar 
        :param: mscaled_q: mass weighted coordinates for free moving atoms. size: [nimag, 3 * natom]
        :param: beads_energy: potential energy of each beads (images)  size : [nimage]
        :param: mscaled_f: mass scaled forces for all beads. size: [nimag, 3 * natom]

        :return: action_force:  the negative gradient of abbreviated action W. (for scaled coordinates) size: [nimag, 3 * natom].
        '''
        bead_displs_vector = mscaled_q[1:] - mscaled_q[:-1]  # displacement vector of beads. [nbeads-1, 3 * natom]

        bead_distance = npnorm( bead_displs_vector , axis = 1)  # |r_j - r_{j-1}|  [nbeads -1]

        bead_displs_unit_vector = np.transpose(np.transpose(bead_displs_vector) / bead_distance) # unit vector for beads displacement vector [nbeads -1, 3* natom] 

        # sqrt(2 (V - E))
        action_each_bead = np.zeros([nimage])
        action_each_bead[1 : nimage - 1] = np.sqrt(2 * (beads_energy[1 : nimage - 1] - self.instanton_path_energy)) # sqrt(2 (V - E))
        if beads_energy[0] > self.instanton_path_energy:
            action_each_bead[0] = np.sqrt(2 * (beads_energy[0] - self.instanton_path_energy))
        if beads_energy[nimage - 1] > self.instanton_path_energy:
            action_each_bead[nimage - 1] = np.sqrt(2 * (beads_energy[nimage - 1] - self.instanton_path_energy))
        
        action_force = np.zeros([nimage, 3 * natom])
        for j in range(1 , nimage-1):
            dj1 = bead_distance[j-1]  #|r_{j} - r_{j-1}|.  d_{j}
            dj2 = bead_distance[j] # |r_{j+1} - r_{j}|. d_{j+1}
            dj1_unit_vector = bead_displs_unit_vector[j-1] # \hat{d}_{j}
            dj2_unit_vector = bead_displs_unit_vector[j]  # \hat{d}_{j+1}
            fj = mscaled_f[j]

            gj = 0.5 * ( 1/action_each_bead[j] * (dj1 + dj2) * fj - (action_each_bead[j] + action_each_bead[j-1]) * dj1_unit_vector + (action_each_bead[j] + action_each_bead[j+1]) * dj2_unit_vector )
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

    def compute_neb_optimization_force(self, nimage, natom, btau, mscaled_q, beads_energy, mscaled_f):
        '''
        compute the optimization forces for nudged elastic band beads. See eq.(15 - 22) in J. Chem. Phys. 148, 102334 (2018).

        :param: nimag: number of images (replica). scalar 
        :param: natom: number of freely moving atoms. scalar 
        :param: btau: tangent vector for internal beads.  size: [nimag, 3 * natoms]
        :param: mscaled_q: mass weighted coordinates for free moving atoms. size: [nimag, 3 * natom]
        :param: beads_energy: potential energy of each beads (images)  size : [nimage]
        :param: mscaled_f: mass scaled forces for all beads. size: [nimag, 3 * natom]

        :return: optimization_force: the optimization force for nudged elastic band. size: [nimag, 3 * natom]
        '''
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
        spring_force_bead0 = spring_k * (mscaled_q[1] - mscaled_q[0]) 
        f0 = mscaled_f[0] / npnorm(mscaled_f[0])   # unit vector along force at beads: 0
        spring_force[0] = spring_force_bead0 - np.dot(spring_force_bead0 , f0) * f0  # spring force transverse to gradient.

        # spring force for end bead nimag - 1
        spring_force_bead1 = spring_k * (mscaled_q[nimage - 2] - mscaled_q[nimage - 1])
        f1 = mscaled_f[nimage - 1] / npnorm(mscaled_f[nimage - 1])  # unit vector along force at beads: nimage - 1 
        spring_force[nimage - 1] = spring_force_bead1 - np.dot(spring_force_bead1 , f1) * f1  # spring force transverse to gradient.

        # end_beads_spring_force: force to draw end beads back to isoenergy contours.
        end_beads_spring_force = np.zeros([2, 3 * natom])
        end_beads_spring_force[0] = mscaled_f[0] / npnorm(mscaled_f[0]) * left_kappa * (beads_energy[0] - self.instanton_path_energy)  # kappa * (V(r) - E) * \hat{f}(r) for beads 0
        end_beads_spring_force[1] = mscaled_f[nimage - 1] / npnorm(mscaled_f[nimage - 1]) * right_kappa * (beads_energy[nimage -1] - self.instanton_path_energy)  # kappa * (V(r) - E) * \hat{f}(r) for beads n-1.

        self.spring_forces = spring_force   # store the spring force between beads
        self.end_bead_energy_constraint_forces = end_beads_spring_force  # store energy constraint force for end beads.

        # for internal beads, transverse force from negative gradient of action.
        for ii in range(1, nimage - 1):
            neb_optimization_force[ii] = self.action_forces[ii] - np.dot(self.action_forces[ii] , btau[ii]) * btau[ii]
        
        self.neb_transverse_force = neb_optimization_force  # transverse gradient for interior neb beads.

        neb_optimization_force[0] = end_beads_spring_force[0]
        neb_optimization_force[nimage - 1] = end_beads_spring_force[1]

        neb_optimization_force = neb_optimization_force + spring_force 

        return neb_optimization_force 
    



    def __call__(self, x):
        """Returns the potential for all beads and the gradient.
        update reduced bead coordinates (&dbeads coordinate) (sticly speaking the free-moving atom parts) with x.
        x = q[:, self.fixatoms_mask] : new coordinates for updated freely moving particles.

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

        # initialize self.rforces with forces and pots from self.dbeads.
        if self.init_allpots is None:
            self.initialize_rforces()

        # energy for reudced beads. All potential energy of beads are needed.
        beads_energy= dstrip(self.rforces.pots).copy()  # beads energy.  rforces.pots here is the potential for the single bead (all atoms)

        # Forces for reduced beads
        rbf = dstrip(self.rforces.f).copy()[:, self.fixatoms_mask]
        # mass weighted force
        mscaled_f = rbf / np.sqrt( self.dbeads.m3[: , self.fixatoms_mask] )  # 1/sqrt(m) * f: mass scaled force.
        self.mscaled_f = mscaled_f

        # Number of images
        nimage = self.dbeads.nbeads
        # Number of atoms that is free to move.
        natom = self.dbeads.natoms - len(self.fixatoms)

        self.spring_forces = np.zeros([nimage, 3 * natom])
        self.end_bead_energy_constraint_forces = np.zeros([2, 3 * natom ])
        self.beads_mscaled_distance = npnorm(mscaled_q[1:] - mscaled_q[:-1] , axis = 1)

        # abbreviated action for the ring polymer instanton path.
        self.action = self.compute_neb_action(nimage, mscaled_q, beads_energy)
        # negative gradient of abbreviated action for each bead. We only compute it for the internal beads (excluding two ends)
        self.action_forces = self.compute_neb_action_force(nimage, natom, mscaled_q, beads_energy, mscaled_f)

        # compute direction of tangent vector, using either improved methods.
        btau = self.compute_tangent_vector(nimage, natom, mscaled_q, beads_energy)

        # compute inner product for mass scaled force and tangent vector btau
        self.f_tau_inner_product = self.compute_force_tangent_vector_inner_product( mscaled_f, btau, nimage)

        # evaluate the nudged elastic band forces for perpendicular action forces and the spring force. (on mass scaled coordinate for free moving atoms.)
        neb_optimization_force = self.compute_neb_optimization_force(nimage, natom, btau, mscaled_q, beads_energy, mscaled_f)

        self.neb_optimization_force = np.copy(neb_optimization_force)

        return neb_optimization_force, self.action 
    
class RP_MAP(object):
    '''
    Generate Ring polymer for Minimum Action Path (MAP) obtained by NEB method.
    Evolve dynamics of particle on inverted potential along minimum action path. 
    The period T of periodic motion (here 2 * total_time for travel from one end to another end) gives beta * hbar (temperature).
    The Ring-polymer for instanton is chosen as evenly spaced beads in time.
    Attributes:

    '''
    def __init__(self):
        """
        Initializatioin of RP_MAP
        """
        self.bead_path_x = None  # coordinate of interpolated beads along MAP
        self.bead_path_r = None  # cumulative sum of distance along MAP

        self.imag_time_period = 0
        self.instanton_temp = 0


    def bind(self, nebmover):
        """
        bind function for RP_MAP
        nebmover: MAPNEBMover instance.
        """
        self.prefix = nebmover.options["prefix"]
        self.final_hessian_bool = nebmover.options["final_hessian_bool"]

        self.energy_shift = nebmover.optarrays["energy_shift"]
        self.output_maker = nebmover.output_maker

        self.path_interpolation_bead_number = nebmover.optarrays["path_interpolation_bead_number"]  # bead number for cubic interpolation of neb beads along minimum action path.
                # for cubic interpolation of neb_beads

        self.neb_beads = nebmover.beads.copy()
        self.dcell = nebmover.cell.copy()
        self.fixatoms = nebmover.fixatoms.copy()


        self.time_step = nebmover.optarrays["instanton_time_step"]  # time step for dynamics along instanton path.
        self.instanton_path_energy = nebmover.optarrays["instanton_path_energy"]

        # Mask to exclude fixed atoms from 3N-arrays
        self.fixatoms_mask = np.ones(3 * nebmover.beads.natoms, dtype=bool)
        if len(nebmover.fixatoms) > 0:
            self.fixatoms_mask[3 * nebmover.fixatoms] = 0
            self.fixatoms_mask[3 * nebmover.fixatoms + 1] = 0
            self.fixatoms_mask[3 * nebmover.fixatoms + 2] = 0

        # ring polymer beads.
        self.rp_bead_number = nebmover.optarrays["instanton_bead_number"] # bead number for instanton ring polymer
        self.rp_beads = Beads(self.neb_beads.natoms, self.rp_bead_number)  # bead object for instanton ring polymer
        self.rp_forces = nebmover.forces.copy(self.rp_beads, self.dcell)
        self.rp_hessian = np.eye(0,0,0, float)

        # particle that perform classical dynamics on inverted potential.
        self.cl_bead = Beads(self.neb_beads.natoms, 1)
        self.m3 = np.copy(dstrip(nebmover.beads.m3[0]))  # mass of atoms.
        self.cl_forces = nebmover.forces.copy(self.cl_bead, self.dcell)




    
    def initialize(self, neb_beads, neb_forces, neb_final_step):
        '''
        initialize the RP_MAP dynamics. This should be called after beads have converged to minimum action path using nudged elastic band method.
        :param: neb_beads: beads in MAPNEBMover, with optimized geometry for Minimum Action Path.
        :param: neb_forces: LINEBGradientMapper.rforces object. 
        :param: step: final step in MAPNEBMover simulation. (Used for output of instanton geometry.)
        '''
        self.neb_beads.q[:] = neb_beads.q[:]  # initialize neb beads position.
        self.cl_bead.q[:] = [neb_beads.q[0]]  # classical particle is initialized at one end of optimized neb beads

        # initialize potential and forces for cl_forces object (forces object for classical particle moving on inverted potential.)
        tmp_v = np.array([dstrip(neb_forces.pots).copy()[0]])   # all beads potential.
        tmp_f = np.array([dstrip(neb_forces.f).copy()[0]])    # all beads forces
        
        self.cl_forces.transfer_forces_manual(
            new_q = [self.cl_bead.q],
            new_v = [tmp_v],
            new_forces = [tmp_f],
        )
    
        # Cubic interpolation of neb beads to enable accurate dynamics evolution.
        self.bead_path_x, self.bead_path_r = \
                ipi.utils.nebinstool.path_cubic_interpolation(self.neb_beads.q, self.path_interpolation_bead_number) 


        print("use cubic interpolation to generate MAP path. The cumulative distance r along the path:  " + str(self.bead_path_r))

        self.final_step = neb_final_step 
   

    def cl_dynamics_along_MAP(self):
        '''
        classical dynamics on the inverted potential -V(x) 
        the final time will be 1/2 of the imaginary period.
        :return:  t_list: a list of time of trajectories.
                  v_list: a list of velocity of trajectories.
                  x_list: a list of coordinate of trajectories.
        '''
        t = 0
        r = 0    # distance traveled along MEP
        x = np.copy(self.bead_path_x[0])  # coordinate
        self.cl_bead.q[0] = np.copy(x)  # update bead location.  
        v = np.zeros([3 * self.cl_bead.natoms])   # velocity
        
        end_bead_index = 1 
        end_bead_r = self.bead_path_r[end_bead_index]  # cumulative sum of distance along Minimum action path (MAP).

        tau = self.bead_path_x[end_bead_index] - self.bead_path_x[end_bead_index - 1]  
        tau = tau / np.linalg.norm(tau)       # tangent direction of the path

        f = - dstrip(self.cl_forces.f).copy()[0]  # compute negative force (force in inverted potential)
        a = f / self.m3 # acceleration
        a = np.dot(a , tau) * tau  # project acceleration along the tangent direction of the path.

        pot = self.cl_forces.pots[0]

        x_list = [x]
        v_list = [v]
        a_list = [a]
        t_list = [t]
        r_list = [r]
        pot_list = [pot]


        while end_bead_index < self.path_interpolation_bead_number:
            tau, t, x, v, a, r, end_bead_index, end_bead_r = \
                        self.cl_dynamics_step(tau, t, x, v, a, r, end_bead_index, end_bead_r)  # evolve particle along Minimum action path on inverted potential with one time step.
            
            pot = self.cl_forces.pots[0]

            x_list.append(x)
            v_list.append(v)
            t_list.append(t)
            r_list.append(r)
            a_list.append(a)
            pot_list.append(pot)

        x_list = np.array(x_list)
        v_list = np.array(v_list)

        kinetic_energy_list = 0.5 * np.sum(np.array(self.m3) * np.power(v_list,2), axis = 1)

        t_list = np.array(t_list)
        r_list = np.array(r_list)
        a_list = np.array(a_list)
        pot_list = np.array(pot_list)

        pot_list = pot_list - self.energy_shift  # ground state energy shift
        total_energy_list = kinetic_energy_list - pot_list  # total_E = K - V.

        pot_list = units.unit_to_user("energy", "electronvolt", pot_list)   # convert to eV unit.
        total_energy_list = units.unit_to_user("energy", "electronvolt", total_energy_list)
        kinetic_energy_list = units.unit_to_user("energy", "electronvolt", kinetic_energy_list)

        a_norm = npnorm(a_list, axis = 1)

        self.imag_time_period = 2 * t_list[-1]  # period of periodic motion is twice of time move from one end to another end. (= beta hbar corresponds to imaginary time.)
        self.instanton_temp = 1 / self.imag_time_period  # temperature of instanton path.
        info("finish evolution of dynamics along Minimum action path, the period of motion is : {} " + str(self.imag_time_period) )

        # for debug.
        print("list of time for dynamic evolution of each time step: " + str(t_list) )
        print("\n")
        print("list of distance along path for dynamics evolution of each time step: " + str(r_list) )
        print("\n")
        print("acceleration amplitude :  " + str(a_norm) )
        print("\n")
        print("potential of points (eV): " + str(pot_list) )
        print("\n")
        print("kinetic energy (eV): " + str( kinetic_energy_list ))
        print("\n")
        print("total energy (eV): " + str(total_energy_list))
        print("\n")

        return t_list, v_list, x_list 


    def cl_dynamics_step(self, tau, t, x, v, a, r, end_bead_index, end_bead_r):
        '''
        evolve dynamics for one step with dt = self.time_step
        :param: tau: tangent direction along path
                t: time
                x: coordinate
                v: velocity
                a: acceleration
                r: cumulative distance along the path.
                end_bead_index: the bead index for the end point of the current segment. The dynamics is along current straight segement line.
                end_bead_r: cumulative distance along the path until the end point.
        '''
        # param for RK4.
        param = [self.cl_bead, self.cl_forces, self.m3, tau]
        dt = self.time_step
        
        old_x = np.copy(x)
        old_v = np.copy(v)
        y = np.array([ np.copy(x), np.copy(v) ])
        
        new_y = RK4(y, t, dydt_inverted_pot, param, dt)
        x = np.copy(new_y[0])
        v = np.copy(new_y[1])

        dx = x - old_x 
        dr = np.linalg.norm(dx)

        if (r + dr) < end_bead_r :
            # move bead normally along tau.
            
            # update r and bead location in cl_bead
            r = r + dr 
            self.cl_bead.q[0] = np.copy(x)

            # update accleration:
            a = -dstrip(self.cl_forces.f).copy()[0] / self.m3  # negative force (-f), force in inverted potential.
            a = np.dot(a, tau) * tau 

            # update time:
            t = t + dt 
        else:
            # adjust the time step 
            dt_right = self.time_step 
            dt_left = 0
            target_dr = end_bead_r - r 
            # bisect search for the time step 
            old_y = np.copy([np.copy(old_x), np.copy(old_v)])
            
            dt , new_y = ipi.utils.nebinstool.bisect_dt(dt_right, dt_left, old_y, t, param, target_dr)

            # update x & r
            r = end_bead_r 
            x = np.copy(self.bead_path_x[end_bead_index])
            self.cl_bead.q[0] = np.copy(x)

            # update velocity
            v = np.copy(new_y[1])

            # update time 
            t = t + dt 

            # update end_bead_index
            end_bead_index = end_bead_index + 1 
            
            if end_bead_index > self.path_interpolation_bead_number - 1:
                # end of path.
                return tau, t, x, v, a, r, end_bead_index, end_bead_r 
            
            end_bead_r = self.bead_path_r[end_bead_index]

            # update tangent vector
            tau = self.bead_path_x[end_bead_index] - self.bead_path_x[end_bead_index - 1]
            tau = tau / npnorm(tau)

            # update acceleration (acceleration should align along the new tau)
            a = -dstrip(self.cl_forces.f).copy()[0] / self.m3  # negative force (-f), force in inverted potential.
            a = np.dot(a, tau) * tau

            # reorient velocity but keep the kinetic energy
            scaled_v = np.sqrt(self.m3) * v 
            scaled_tau = tau * np.sqrt(self.m3)
            scaled_tau = scaled_tau / npnorm(scaled_tau)
            scaled_v = npnorm(scaled_v) * scaled_tau   # re-orient sqrt(m) * v 
            v = scaled_v / np.sqrt(self.m3)  # now v conserve the energy and align with direction of tau.

        return tau, t, x, v, a, r, end_bead_index, end_bead_r
    
    def interpolate_ring_polymer_beads(self, t_list, v_list, x_list):
        '''
        interpolate ring polymer beads from the imaginary time trajectory along Minimum Action Path (MAP).
        t_list , v_list, x_list: list of time / velocity / trajectory from MD simulation along path.
        '''
        # interpolate to get ring polymer position.
        rp_t_list, rp_x_list = ipi.utils.nebinstool.interpolate_ring_polymer_beads(self.imag_time_period, t_list, x_list, v_list, self.rp_bead_number)

        ipi.utils.nebinstool.print_instanton_rp_time("rp_time_FINAL", self.imag_time_period, rp_t_list, self.output_maker)

        self.rp_beads.q = rp_x_list 

        # print ring polymer instanton geometry. 
        pots = self.rp_forces.pots 
        ipi.utils.nebinstool.print_neb_instanton_geo(
            "instanton_along_MAP_FINAL",
            self.final_step,
            self.rp_beads.nbeads,
            self.rp_beads.natoms,
            self.neb_beads.names,
            self.rp_beads.q,
            pots,
            self.dcell,
            self.energy_shift,
            self.output_maker
        )

    def compute_ring_polymer_hessian(self):
        '''
        compute hessian of ring polymer
        '''
        if self.final_hessian_bool:
            # compute final hessian.

            # create bead and forces object for computing hessian.
            hess_rp_beads = self.rp_beads.copy() 
            hess_forces = self.rp_forces.copy(hess_rp_beads, self.dcell)

            self.rp_hessian = ipi.utils.nebinstool.get_hessian(
                hess_rp_beads,
                hess_forces,
                self.rp_beads.q,
                self.rp_beads.natoms,
                self.rp_beads.nbeads,
                self.fixatoms
            )
        



    def print_temperature(self):
        '''
        output temperature to the log file and a separate file
        '''
        temp_kelvin = units.unit_to_user("temperature", "kelvin", self.instanton_temp)  # temperature in "kelvin" unit

        print("temperature for instanton path : {} K".format(temp_kelvin))

        # output temperature to a separate file
        outfile = self.output_maker.get_output( "instanton_temperature.txt", "w")
        print("temperature for instanton path : (K)", file = outfile)
        print(str(temp_kelvin), file = outfile)
        outfile.close_stream()

    def generate_ring_polymer_beads(self, neb_beads, neb_forces, neb_final_step):
        '''
        Main function that compute ring-polymer beads from nudged elastic band Minimum action path.
        '''
        self.initialize(neb_beads, neb_forces, neb_final_step)
        
        # start classical dynamics along minimum action path (MEP) on inverted potential.
        t_list, v_list, x_list  = self.cl_dynamics_along_MAP()

        # print the temperature for the found minimum action path in Kelvin unit.
        self.print_temperature()

        # interpolate the ring polymer beads from the generated trajectory. 
        self.interpolate_ring_polymer_beads(t_list, v_list, x_list)

        # compute hessian of ring polymers
        self.compute_ring_polymer_hessian()
        



    


