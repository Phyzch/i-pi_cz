"""Holds the algorithms to perform nudged elastic band (NEB) calculations.
J. Chem. Phys. 113, 9901 (2000); https://doi.org/10.1063/1.1329672

The algorithms are first implemented by Michele Ceriotti and Benjamin Helfrecht, 2015.
Considerably reworked by Karen Fidanyan in 2021.
Algorithm for using li-neb to search instanton path.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2021 i-PI developers
# See the "licenses" directory for full license information.

import numpy as np
from numpy.linalg import norm as npnorm
import time
from ipi.utils import units
from ipi.engine.motion import Motion
from ipi.utils.depend import dstrip
from ipi.utils.softexit import softexit
from ipi.utils.mintools import Damped_BFGS, FIRE
from ipi.utils.messages import verbosity, info
from ipi.engine.beads import Beads
from ipi.utils.nebinstool import print_neb_instanton_geo

np.set_printoptions(threshold=10000, linewidth=1000)  # Remove in cleanup

__all__ = ["LINEBGradientMapper", "MAPNEBMover"]


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
        self.spring_k = None    # spring constants for internal beads  TODO: convert unit from anstrom^-2 to internal unit
        self.kappa = None   # spring constants for beads at two ends.  TODO: convert unit from eV-1 angstrom^-1 to internal unit.

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

        self.spring_k = ens.optarrays["spring_k"] # bind spring force spring_k from NEBMover
        self.kappa = ens.optarrays["kappa"] # bind energy constraint force kappa from NEBMover 
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
        kappa = self.kappa   # kappa: restraint force back to iso-energy contour
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
        end_beads_spring_force[0] = mscaled_f[0] / npnorm(mscaled_f[0]) * kappa * (beads_energy[0] - self.instanton_path_energy)  # kappa * (V(r) - E) * \hat{f}(r) for beads 0
        end_beads_spring_force[1] = mscaled_f[nimage - 1] / npnorm(mscaled_f[nimage - 1]) * kappa * (beads_energy[nimage -1] - self.instanton_path_energy)  # kappa * (V(r) - E) * \hat{f}(r) for beads n-1.

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
    



class MAPNEBMover(Motion):
    """Nudged elastic band routine. for minimum action path (MAP)

    Attributes:
    """

    def __init__(
        self,
        fixcom=False,
        fixatoms=None,
        mode = "verlet",
        prefix = "neb_instanton",
        tolerances = { "gradient": 5e-3},
        energy_shift = 0.00,
        time_step = np.power(10.0 , -3),
        stage = "neb",
        instanton_bead_number = 20,
        instanton_path_energy = 0.00,
        spring_k = 10,
        kappa = 1000,
        alt_out = 5
    ):
        """Initialises NEBMover.

        Args:
           fixcom: An optional boolean which decides whether the centre of mass
              motion will be constrained or not. Defaults to False.
        """
        super(MAPNEBMover, self).__init__(fixcom=fixcom, fixatoms=fixatoms)

        # TODO parameters to pass in from input.xml
        self.options = {}
        
        # mode for optimization
        self.options["mode"] = mode  

        self.options["stage"] = stage
        self.options["tolerances"] = tolerances
        self.options["alt_out_step"] = alt_out   # step to output geometry.
        self.options["prefix"] = prefix 
        # numerical values / arrays. option from input.xml
        self.optarrays = {}
        self.optarrays["energy_shift"] = energy_shift 
        self.optarrays["spring_k"] = spring_k
        self.optarrays["kappa"] = kappa
        self.optarrays["instanton_path_energy"] = instanton_path_energy 
        self.optarrays["time_step"] = time_step
        self.optarrays["instanton_bead_number"] = instanton_bead_number


        # convert unit for spring_k , kappa 
        self.optarrays["spring_k"] = self.optarrays["spring_k"] / np.power( units.unit_to_internal("length", "angstrom", 1) , 2)  # input unit: angstrom^{-2}
        self.optarrays["kappa"] = self.optarrays["kappa"] / ( units.unit_to_internal("length" , "angstrom", 1) * units.unit_to_internal("energy", "electronvolt", 1) )

        # need to figure out how the program handle unit.
        self.optarrays["instanton_path_energy"] = self.optarrays["instanton_path_energy"] + self.optarrays["energy_shift"]  # shift the instanton path energy according to energy shift.

        self.nebgm = LINEBGradientMapper()

    def bind(self, ens, beads, nm, cell, bforce, prng, omaker):
        super(MAPNEBMover, self).bind(ens, beads, nm, cell, bforce, prng, omaker)
        if len(self.fixatoms) == len(self.beads[0]):
            softexit.trigger(
                status="bad",
                message="WARNING: all atoms are fixed, geometry won't change. Exiting simulation.",
            )

        # fixatoms mask.
        self.fixatoms_mask = np.ones(3 * self.beads.natoms, dtype=bool)
        if len(self.fixatoms) > 0:
            self.fixatoms_mask[3 * self.fixatoms] = 0
            self.fixatoms_mask[3 * self.fixatoms + 1] = 0
            self.fixatoms_mask[3 * self.fixatoms + 2] = 0

        self.nebgm.bind(self)
            

    def step(self, step=None):
        """Does one simulation time step.
        """

        info(" @NEB STEP %d, stage: %s" % (step, self.options["stage"]), verbosity.debug)

        # Check if we restarted a converged calculation (by mistake)
        if self.options["stage"] == "converged":
            softexit.trigger(
                status="success",
                message="NEB has already converged. Exiting simulation.",
            )

        if self.options["stage"] == "neb":
            self.step_neb(step)



# --------- NEB method -----------------------
    def step_neb(self, step):
        n_activedim = self.beads.q[0].size - len(self.fixatoms) * 3
        nbeads = self.beads.nbeads
        dt = self.optarrays["time_step"]

        if step == 0:
            self.neb_initialize()
        
        self.print_geometry(step)
            
        if self.options["mode"] == "verlet":
            # Only initialize velocity for fresh start, not for RESTART
            dx_mscaled = dt * self.velocity_mscaled + 0.5 * self.f_mscaled * np.power(dt, 2)
            dx = dx_mscaled / np.sqrt(self.beads.m3[:, self.fixatoms_mask])

            # update position
            self.old_x = self.x 
            self.x = self.x + dx 
            self.beads.q[:, self.fixatoms_mask] = self.x 

            self.old_f_mscaled = self.f_mscaled # record old force
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

        
        # check convergence
        # transverse gradient for interior beads.
        grad_interior_beads_max = np.amax(np.abs(self.nebgm.neb_transverse_force))
        # optimization gradient at end beads.
        grad_end_beads_max = np.amax(np.abs([self.nebgm.neb_optimization_force[0], self.nebgm.neb_optimization_force[nbeads - 1]]))
        grad_max = np.max([grad_end_beads_max, grad_interior_beads_max])

        self.neb_instanton_exit(step, grad_max)


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


    def neb_instanton_exit(self, step,grad_max):
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
        print("\n")
        
        # for debug
        # print("beads potential relative to instanton path energy (kcal/mol): " + str( (self.nebgm.rforces.pots - self.optarrays["instanton_path_energy"]) * 627.503  ))
        # print("distance between beads in mass scaled coordinate: " + str( self.nebgm.beads_mscaled_distance ))
        # print("\n")

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
            self.stage = "converged"

            # print neb beads geometry and energy.
            print_neb_instanton_geo(
                self.options["prefix"] + "_neb_FINAL",
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

            softexit.trigger(
                status = "success",
                message = "NEB finished successfully at step %i" % step
            )
        
    def print_geometry(self, step):
        '''
        print beads geometry and beads energy.
        '''
        if (
            self.options["alt_out_step"] > 0 and np.mod(step, self.options["alt_out_step"]) == 0
        ):
            print_neb_instanton_geo(
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



