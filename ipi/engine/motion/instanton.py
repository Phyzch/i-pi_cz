"""
Contains classes for instanton  calculations.

Algorithms implemented by Yair Litman and Mariana Rossi, 2017
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


import numpy as np
import warnings

np.set_printoptions(suppress=True, linewidth=1000)
import time
import sys
from importlib import util

from ipi.engine.beads import Beads
from ipi.engine.normalmodes import NormalModes
from ipi.engine.motion import Motion
from ipi.utils.depend import dstrip
from ipi.utils.softexit import softexit
from ipi.utils.messages import verbosity, info
from ipi.utils import units
from ipi.utils.mintools import nichols, Powell, Davidon_Fletcher_Powell
from ipi.engine.motion.geop import L_BFGS
from ipi.utils.instools import (
    banded_hessian,
    invmul_banded,
    red2comp,
    get_imvector,
    print_instanton_geo,
    Fix,
)
from ipi.utils.instools import print_instanton_hess, diag_banded, ms_pathway
from ipi.utils.hesstools import get_hessian, clean_hessian, get_dynmat


__all__ = ["InstantonMotion"]


class InstantonMotion(Motion):
    """Instanton motion class.

    Attributes:
        biggest_step: max allowed step size
        old_force: force on previous step
        hessian:

        mode= type of instanton calculation
        tolerances:
            energy: change in energy tolerance for ending minimization
            force: force/change in force tolerance foe ending minimization
            position: change in position tolerance for ending minimization}
        
        biggest_step: The maximum step size during the optimization.
        old_pos: The previous step positions during the optimization.
        old_pot: The previous step potential energy during the optimization
        old_force:  The previous step force during the optimization
        
        opt: The geometry optimization algorithm to be used
        discretization: Allows for non uniform time discretization
        
        alt_out: (Alternative output) Prints different formatting of outputs for geometry, hessian and bead potential energies.
        All quantities are also accessible from typical i-pi output infrastructure. Default to 1, which prints
        every step. -1 will suppress the output (except the last one). Any other positive number will set the frequency (in steps) with
        which the quantities are written to file.

        prefix: Prefix of the output files.
        delta: Initial stretch amplitude.
        hessian_init: Boolean which decides whether the initial hessian is going to be computed.
        hessian: Stored  Hessian matrix
        hessian_update: The way to update the hessian after each movement
        hessian_asr: Removes the zero frequency vibrational modes depending on the symmerty of the system.
        glist_lbfgs: List of previous gradients (g_n+1 - g_n) for L-BFGS. Number of entries = corrections_lbfgs
        qlist_lbfgs: List of previous positions (x_n+1 - x_n) for L-BFGS. Number of entries = corrections_lbfgs
        scale_lbfgs: Scale choice for the initial hessian.
        corrections_lbfgs: Number of corrections to be stored for L-BFGS
        ls_options: Options for line search methods.
        hessian_final:  Boolean which decides whether the hessian after the optimization will be computed.
        energy_shift: zero of energy (usually it corresponds to reactant state)
    """

    def __init__(
        self,
        fixcom=False,
        fixatoms=None,
        mode="None",
        tolerances={"energy": 1e-5, "force": 1e-4, "position": 1e-3},
        biggest_step=0.3,
        old_pos=np.zeros(0, float),
        old_pot=np.zeros(0, float),
        old_force=np.zeros(0, float),
        opt="None",
        max_e=0.0,
        max_ms=0.0,
        discretization=np.zeros(0, float),
        alt_out=1,
        prefix="instanton",
        delta=np.zeros(0, float),
        hessian_init=None,
        hessian=np.eye(0, 0, 0, float),
        fric_hessian=np.eye(0, 0, 0, float),
        hessian_update=None,
        hessian_asr=None,
        qlist_lbfgs=np.zeros(0, float),
        glist_lbfgs=np.zeros(0, float),
        scale_lbfgs=1,
        corrections_lbfgs=5,
        ls_options={"tolerance": 1e-1, "iter": 100},
        old_direction=np.zeros(0, float),
        hessian_final="False",
        energy_shift=np.zeros(0, float),
        friction=False,
        frictionSD=True,
        eta=np.eye(0, 0, 0, float),
        fric_spec_dens=np.zeros(0, float),
        fric_spec_dens_ener=0.0,
    ):
        """Initialises InstantonMotion."""

        super(InstantonMotion, self).__init__(fixcom=fixcom, fixatoms=fixatoms)

        self.options = {}  # Optimization options

        # Optimization mode
        self.options["mode"] = mode

        # Generic optimization
        # self.big_step = biggest_step
        # self.tolerances = tolerances

        self.options["tolerances"] = tolerances   # tolerance for termination of optimization
        self.options["save"] = alt_out    # frequency for outputting geometry, hessian and bead potential.
        self.options["prefix"] = prefix  # prefix for output file
        self.options["hessian_final"] = hessian_final  # bool: whether compute final hessian.

        self.options["max_e"] = max_e  # used for interpolation of beads potential energy, maximum energy
        self.options["max_ms"] = max_ms  # used for interpolation of beads potential energy, maximum path length distance.
        self.options["discretization"] = discretization  # bool variable for different form of discretization, not implemented. 
        self.options["friction"] = friction  # add friction to dynamics, only apply in Nicholas optimizer.
        if not friction:
            self.options["frictionSD"] = False
        else:
            self.options["frictionSD"] = frictionSD

        self.options["fric_spec_dens"] = fric_spec_dens
        self.options["fric_spec_dens_ener"] = fric_spec_dens_ener

        self.optarrays = {}  # Optimization arrays
        self.optarrays["big_step"] = biggest_step   # biggest step for instanton optimization
        self.optarrays["energy_shift"] = energy_shift  # zero point of energy (energy of reactant state)
        self.optarrays["delta"] = delta   # initial stretch amplitude for optimization.
        
        # previous step potential, force and bead coordinates.
        self.optarrays["old_x"] = old_pos 
        self.optarrays["old_u"] = old_pot
        self.optarrays["old_f"] = old_force 

        # We set the default optimization algorithm depending on the mode.
        # default optimization method for "rate": nicholas,  default optimization method for tunneling splitting: limited meory BFGS
        if mode == "rate":
            if opt == "None":
                opt = "nichols"
            self.options["opt"] = opt

        elif mode == "splitting":
            if opt == "None":
                opt = "lbfgs"
            self.options["opt"] = opt

        if (
            self.options["opt"] == "nichols"
            or self.options["opt"] == "NR"
            or self.options["opt"] == "lanczos"
        ):
            if self.options["friction"]:  # and not self.options["frictionSD"]:
                self.options["eta0"] = eta

            self.options["hessian_update"] = hessian_update   # the way to update hessian. ['powell' or 'recompute']. powell method is the quasi-Newton method
            self.options["hessian_asr"] = hessian_asr  # 'poly' / 'crystal' / 'none', used to work with rotational & translational dof.
            self.options["hessian_init"] = hessian_init  # bool, whether to compute initial hessian
            self.optarrays["hessian"] = hessian 

            if self.options["friction"] and self.options["frictionSD"]:
                self.optarrays["fric_hessian"] = fric_hessian

            if self.options["opt"] == "nichols":
                self.optimizer = NicholsOptimizer()   # call __init__() function of Nicholas Optimizer (it's __init__ func of DummyOptimizer)
            else:
                if self.options["friction"]:
                    raise ValueError(
                        "\nPlease select nichols opt algorithm for an instanton calculation with friction\n"
                    )
                if self.options["opt"] == "NR":
                    self.optimizer = NROptimizer()  # Newton Raphson optimizer is recommended. 
                else:
                    self.optimizer = LanczosOptimizer()  # this is mentioned in JCTC (Locating instantons in many degrees of freedom) paper

        elif self.options["opt"] == "lbfgs":  # limited-memory BFGS algorithm (https://en.wikipedia.org/wiki/Limited-memory_BFGS)
            self.optimizer = LBFGSOptimizer()
            self.optarrays["hessian"] = hessian  # Only for initial (to spread) or final
            self.options["hessian_asr"] = hessian_asr

            self.options["corrections"] = corrections_lbfgs  # the number of past vectors to store 
            self.options["scale"] = scale_lbfgs  # scale choice of lbfgs
            self.options["ls_options"] = ls_options  # line searching algorithm options

            self.optarrays["qlist"] = qlist_lbfgs # previous position difference for bfgs, s_{k}= x_{k+1} - x_{k}
            self.optarrays["glist"] = glist_lbfgs # previous gradient difference for bfgs, y_{k}= \nbla f(x_{k+1}) - \nbla f(x_{k})
            self.optarrays["d"] = old_direction # previous direction of CG or SD optimization 

        if self.options["opt"] == "NR" or self.options["opt"] == "lanczos":
            info(
                "Note that we need scipy to use NR or lanczos. If storage and diagonalization of the full hessian is not a "
                "problem use nichols even though it may not be as efficient.",
                verbosity.low,
            )
            found = util.find_spec("scipy")
            if found is None:
                softexit.trigger(
                    "Scipy is required to use NR or lanczos optimization but could not be found"
                )

        if self.options["friction"]:
            found = util.find_spec("scipy")
            if found is None:
                softexit.trigger(
                    "Scipy is required to use friction in a instanton calculation but could not be found"
                )

    def bind(self, ens, beads, nm, cell, bforce, prng, omaker):
        """Binds beads, cell, bforce and prng to InstantonMotion

        Args:
        beads: The beads object from which the bead positions are taken.
        nm: A normal modes object used to do the normal modes transformation.
        cell: The cell object from which the system box is taken.
        bforce: The forcefield object from which the force and virial are taken.
        prng: The random number generator object which controls random number generation.
        """

        super(InstantonMotion, self).bind(ens, beads, nm, cell, bforce, prng, omaker)

        # Redefine normal modes
        self.nm = NormalModes(
            transform_method="matrix", open_paths=np.arange(self.beads.natoms)
        )

        self.nm.bind(self.ensemble, self, Beads(self.beads.natoms, self.beads.nbeads))
        if self.options["mode"] == "rate":
            self.rp_factor = 2
        elif self.options["mode"] == "splitting":
            self.rp_factor = 1

        self.optimizer.bind(self)

    def step(self, step=None):
        self.optimizer.step(step)


class PesMapper(object):
    """Creation of the multi-dimensional function to compute the physical potential and forces

    Attributes:
        dbeads:  copy of the bead object
        dcell:   copy of the cell object
        dforces: copy of the forces object
    """

    def __init__(self):
        self.fcount = 0
        pass

    def bind(self, mapper):
        self.dbeads = mapper.beads.copy()
        self.dcell = mapper.cell.copy()
        self.dforces = mapper.forces.copy(self.dbeads, self.dcell)

        # self.nm = mapper.nm
        # self.rp_factor = mapper.rp_factor
        if self.dbeads.nbeads > 1:
            self.C = mapper.nm.transform._b2o_nm
        else:
            self.C = 1

        self.omegak = mapper.rp_factor * mapper.nm.get_o_omegak()

        self.fix = mapper.fix
        self.coef = mapper.coef

        max_ms = mapper.options["max_ms"]
        max_e = mapper.options["max_e"]

        if max_ms > 0 or max_e > 0:
            self.spline = True

            if max_ms > 0:
                self.max_ms = max_ms
            else:
                self.max_ms = 1000000
            if max_e > 0:
                self.max_e = max_e
            else:
                self.max_e = 10000000
        else:
            self.spline = False

    def initialize(self, q, forces):
        """Initialize potential and forces"""
        self.save(forces.pots, -forces.f)

    def set_pos(self, x):
        """Set the positions"""
        self.dbeads.q = x

    def save(self, e, g):
        """Stores potential and forces in this class for convenience"""
        self.pot = e
        self.f = -g

    def interpolation(self, full_q, full_mspath, get_all_info=False):
        """Creates the reduced bead object from which energy and forces will be
        computed and interpolates the results to the full size
        """
        if self.spline:
            # implement reduced beads. evaluate forces and energy with reduced beads number.
            try:
                from scipy.interpolate import interp1d
            except ImportError:
                softexit.trigger(
                    status="bad", message="Scipy required to use  max_ms >0"
                )

            indexes = list()
            indexes.append(0)
            old_index = 0
            for i in range(1, self.dbeads.nbeads):
                if (full_mspath[i] - full_mspath[old_index] > self.max_ms) or (
                    np.absolute(self.pot[i] - self.pot[old_index]) > self.max_e
                ):
                    indexes.append(i)
                    old_index = i
            if self.dbeads.nbeads - 1 not in indexes:
                indexes.append(self.dbeads.nbeads - 1)
            info(
                "The reduced RP for this step has {} beads.".format(len(indexes)),
                verbosity.low,
            )
            if len(indexes) <= 2:
                softexit.trigger(
                    status="bad",
                    message="Too few beads fulfill criteria. Please reduce max_ms or max_e",
                )
        else:
            indexes = np.arange(self.dbeads.nbeads)

        # Create reduced bead and force object and evaluate forces
        reduced_b = Beads(self.dbeads.natoms, len(indexes))
        reduced_b.q[:] = full_q[indexes]  # reduced beads' position
        reduced_b.m[:] = self.dbeads.m    # reduced beads' mass for different atoms.
        reduced_b.names[:] = self.dbeads.names

        reduced_cell = self.dcell.copy()
        reduced_forces = self.dforces.copy(reduced_b, reduced_cell)

        # Evaluate energy and forces (and maybe friction)
        rpots = reduced_forces.pots  # reduced energy
        rforces = reduced_forces.f  # reduced gradient

        # Interpolate if necessary to get full pot and forces
        if self.spline:
            red_mspath = full_mspath[indexes]
            spline = interp1d(red_mspath, rpots.T, kind="cubic")  # create interpolation function: V = V(s), here s is mean-square path length.
            full_pot = spline(full_mspath).T  # interpolate to get the potential of full beads polymer
            spline = interp1d(red_mspath, rforces.T, kind="cubic")  # create interpolation function: F = F(s), here s is mean-square path length
            full_forces = spline(full_mspath).T  # interpolate to get the force of full beads polymer.
        else:
            full_pot = rpots
            full_forces = rforces
        if get_all_info:
            return full_pot, full_forces, indexes, reduced_forces
        else:
            return full_pot, full_forces

    def __call__(self, x, new_disc=True):
        """Computes energy and gradient for optimization step"""
        self.fcount += 1
        full_q = x.copy()
        full_mspath = ms_pathway(full_q, self.dbeads.m3)  # mass scaled pathway, full_mspath : list, path length until bead i.
        full_pot, full_forces = self.interpolation(full_q, full_mspath)
        self.dbeads.q[:] = x[:]
         # update coordinate (full_q), potential (full_pot) and forces (full_forces) in self.dforces (Forces class) object
        self.dforces.transfer_forces_manual([full_q], [full_pot], [full_forces]) 
        info("UPDATE of forces and extras", verbosity.debug)

        self.save(full_pot, -full_forces)  # update self.pot & self.f (potential & force)
        return self.evaluate()

    def evaluate(self):
        """Evaluate the energy and forces including:
        - non uniform discretization
        - friction term (if required)
        """

        e = self.pot.copy()
        g = -self.f.copy()

        e = e * (self.coef[1:,0] + self.coef[:-1,0]) / 2  # TODO bug here, should be self.coef[1:, 0]
        g = g * (self.coef[1:] + self.coef[:-1]) / 2

        return e, g


class FrictionMapper(PesMapper):
    """Creation of the multi-dimensional function to compute the physical potential and forces,
    as well as the friction terms"""

    def __init__(self, frictionSD, eta0):
        super(FrictionMapper, self).__init__()
        self.frictionSD = frictionSD
        self.eta0 = eta0

    def bind(self, mapper):
        super(FrictionMapper, self).bind(mapper)
        from scipy.interpolate import interp1d
        from scipy.linalg import sqrtm
        from scipy.integrate import quad

        self.sqrtm = sqrtm
        self.quad = quad
        self.interp1d = interp1d

    def save(self, e, g, eta=None):
        """Stores potential and forces in this class for convenience"""
        super(FrictionMapper, self).save(e, g)
        self.eta = eta

    def initialize(self, q, forces):
        """Initialize potential, forces and friction"""

        if self.frictionSD:
            eta = np.array(forces.extras["friction"]).reshape(
                (q.shape[0], q.shape[1], q.shape[1])
            )
        else:
            eta = np.zeros((q.shape[0], q.shape[1], q.shape[1]))
            for i in range(self.dbeads.nbeads):
                eta[i] = self.eta0

        self.check_eta(eta)

        self.save(forces.pots, -forces.f, eta)

    def check_eta(self, eta):
        for i in range(self.dbeads.nbeads):
            assert (
                eta[i] - eta[i].T
                == np.zeros((self.dbeads.natoms * 3, self.dbeads.natoms * 3))
            ).all()
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                self.sqrtm(
                    eta[i] + np.eye(self.dbeads.natoms * 3) * 0.000000000001
                )  # dgdq = s ** 0.5 -> won't work for multiD
            except Warning:
                print(eta[i])
                softexit.trigger("The provided friction is not positive definite")

    def set_fric_spec_dens(self, fric_spec_dens_data, fric_spec_dens_ener):
        """Computes and sets the laplace transform of the friction tensor"""
        # from ipi.utils.mathtools import LT_friction

        # from scipy.interpolate import interp1d
        if len(fric_spec_dens_data) == 0:
            LT_fric_spec_dens = np.ones((1000, 2))
            LT_fric_spec_dens[:, 0] = np.arange(self.omegak.shape)
        else:
            invcm2au = units.unit_to_internal("frequency", "inversecm", 1)

            # We perform the spline in inversecm for numerical reasons
            freq = fric_spec_dens_data[:, 0] * invcm2au
            spline = self.interp1d(
                freq,
                fric_spec_dens_data[:, 1],
                kind="cubic",
                fill_value=0.0,
                bounds_error=False,
            )

            if fric_spec_dens_ener == 0 or fric_spec_dens_ener / invcm2au < freq[0]:
                norm = 1.0  # spline(freq[0])*freq[0]*invcm2au
            elif fric_spec_dens_ener / invcm2au > freq[-1]:
                norm = 1.0  # spline(freq[-1])*freq[-10]*invcm2au
            else:
                # norm = spline(fric_spec_dens_ener / invcm2au) * fric_spec_dens_ener
                norm = spline(fric_spec_dens_ener / invcm2au)

            fric_spec_dens = spline(self.omegak)
            LT_fric_spec_dens = fric_spec_dens / norm
            # LT_fric_spec_dens = LT_friction(self.omegak / invcm2au, spline) / norm

        self.fric_LTwk = np.multiply(self.omegak, LT_fric_spec_dens)[:, np.newaxis]
        info(units.unit_to_user("frequency", "inversecm", self.omegak), verbosity.debug)

    def get_fric_rp_hessian(self, fric_hessian, eta, SD):
        """Creates the friction hessian from the eta derivatives
        THIS IS ONLY DONE FOR THE ACTIVE MODES"""

        nphys = self.fix.nactive * 3
        ndof = self.dbeads.nbeads * self.fix.nactive * 3
        nbeads = self.dbeads.nbeads

        s = eta

        dgdq = np.zeros(s.shape)
        for i in range(nbeads):
            dgdq[i] = self.sqrtm(s[i] + np.eye(nphys) * 0.000000000001)

        h_fric = np.zeros((ndof, ndof))

        # Block diag:
        if SD:
            gq = self.obtain_g(s)
            gq_k = np.dot(self.C, gq)
            prefactor = np.dot(self.C.T, self.fric_LTwk * gq_k)
            for n in range(self.dbeads.nbeads):
                for j in range(nphys):
                    for k in range(nphys):
                        aux_jk = 0
                        for i in range(nphys):
                            if dgdq[n, i, j] != 0:
                                aux_jk += (
                                    0.5
                                    * prefactor[n, i]
                                    * fric_hessian[n, i, j, k]
                                    / dgdq[n, i, j]
                                )
                        h_fric[nphys * n + j, nphys * n + k] = aux_jk
        # Cross-terms:
        for nl in range(nbeads):
            for ne in range(nbeads):
                prefactor = 0
                for alpha in range(nbeads):
                    prefactor += (
                        self.C[alpha, nl] * self.C[alpha, ne] * self.fric_LTwk[alpha]
                    )
                for j in range(nphys):
                    for k in range(nphys):
                        suma = np.sum(dgdq[nl, :, j] * dgdq[ne, :, k])
                        h_fric[nphys * nl + j, nphys * ne + k] = prefactor * suma
        return h_fric

    def obtain_g(self, s):
        """Computes g from s"""

        nphys = self.dbeads.natoms * 3

        ss = np.zeros(s.shape)

        for i in range(self.dbeads.nbeads):
            ss[i] = self.sqrtm(
                s[i] + np.eye(nphys) * 0.000000001
            )  # ss = s ** 0.5 -> won't work for multiD

        q = self.dbeads.q.copy()
        gq = np.zeros(self.dbeads.q.copy().shape)
        for nd in range(3 * self.dbeads.natoms):
            try:
                spline = self.interp1d(
                    q[:, nd], ss[:, nd, nd], kind="cubic"
                )  # spline for each dof
                for nb in range(1, self.dbeads.nbeads):
                    gq[nb, nd] = self.quad(spline, q[0, nd], q[nb, nd])[
                        0
                    ]  # Cumulative integral along the path for each dof
                # for i in range(self.dbeads.nbeads):
                #    print(q[i, nd],q[i,nd],ss[i, nd, nd],gq[i, nd])
            except ValueError:
                gq[:, nd] = 0

        return gq

    def compute_friction_terms(self):
        """Computes friction component of the energy and gradient"""

        s = self.eta

        nphys = self.dbeads.natoms * 3

        dgdq = np.zeros(s.shape)
        for i in range(self.dbeads.nbeads):
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                try:
                    dgdq[i] = self.sqrtm(
                        s[i] + np.eye(nphys) * 0.00000001
                    )  # dgdq = s ** 0.5 -> won't work for multiD
                except Warning:
                    print(s[i])
                    softexit.trigger("The provided friction is not positive definite")

        gq = self.obtain_g(s)
        gq_k = np.dot(self.C, gq)
        e = 0.5 * np.sum(self.fric_LTwk * gq_k**2)

        f = np.dot(self.C.T, self.fric_LTwk * gq_k)
        g = np.zeros(f.shape)
        for i in range(self.dbeads.nbeads):
            g[i, :] = np.dot(dgdq[i], f[i])

        return e, g

    def get_full_extras(self, reduced_forces, full_mspath, indexes):
        """Get the full extra strings"""
        diction = {}
        for key in reduced_forces.extras.keys():
            if str(key) != "raw":
                red_data = np.array(reduced_forces.extras[key])
                if self.spline:
                    red_mspath = full_mspath[indexes]
                    spline = self.interp1d(red_mspath, red_data.T, kind="cubic")
                    full_data = spline(full_mspath).T
                else:
                    full_data = red_data
            else:
                full_data = reduced_forces.extras[key]
            diction[key] = full_data

        return diction

    def __call__(self, x, new_disc=True):
        """Computes energy and gradient for optimization step"""
        self.fcount += 1
        full_q = x.copy()
        full_mspath = ms_pathway(full_q, self.dbeads.m3)
        full_pot, full_forces, indexes, reduced_forces = self.interpolation(
            full_q, full_mspath, get_all_info=True
        )

        full_extras = self.get_full_extras(reduced_forces, full_mspath, indexes)
        if self.frictionSD:
            full_eta = np.zeros(
                (self.dbeads.nbeads, self.dbeads.natoms * 3, self.dbeads.natoms * 3)
            )
            for n in range(self.dbeads.nbeads):
                full_eta[n] = full_extras["friction"][n].reshape(
                    self.dbeads.natoms * 3, self.dbeads.natoms * 3
                )
        else:
            full_eta = self.eta

        info(
            "We expect friction tensor evaluated at the first RP frequency",
            verbosity.debug,
        )

        # This forces the update of the forces and the extras
        self.dbeads.q[:] = x[:]
        self.dforces.transfer_forces_manual(
            [full_q], [full_pot], [full_forces], [full_extras]
        )
        self.save(full_pot, -full_forces, full_eta)
        return self.evaluate()

    def evaluate(self):
        """Evaluate the energy and forces including:
        - non uniform discretization
        - friction term
        """

        e = self.pot.copy()
        g = -self.f.copy()

        e_friction, g_friction = self.compute_friction_terms()
        e += e_friction
        g += g_friction

        e = e * (self.coef[1:] + self.coef[:-1]) / 2
        g = g * (self.coef[1:] + self.coef[:-1]) / 2

        return e, g


class SpringMapper(object):
    """Creation of the multi-dimensional function to compute full or half ring polymer potential
    and forces.
    """

    def __init__(self):
        self.pot = None
        self.f = None
        pass

    def bind(self, mapper):
        self.temp = mapper.temp
        self.fix = mapper.fix
        self.coef = mapper.coef
        self.dbeads = mapper.beads.copy()
        # self.nm = mapper.nm
        # self.rp_factor = mapper.rp_factor
        if self.dbeads.nbeads > 1:
            self.C = mapper.nm.transform._b2o_nm
        else:
            self.C = 1
        self.omegak = mapper.rp_factor * mapper.nm.get_o_omegak()
        self.omegan = mapper.rp_factor * mapper.nm.omegan

        # Computes the spring hessian if the optimization modes requires it
        if (
            mapper.options["opt"] == "nichols"
            or mapper.options["opt"] == "NR"
            or mapper.options["opt"] == "lanczos"
        ):
            self.h = self.spring_hessian(
                natoms=self.fix.fixbeads.natoms,
                nbeads=self.fix.fixbeads.nbeads,
                m3=self.fix.fixbeads.m3[0],
                omega2=(self.omegan) ** 2,
                coef=self.coef,
            )

    def set_coef(self, coef):
        """Sets coefficients for non-uniform instanton calculation"""
        self.coef = coef.reshape(-1, 1)

    def save(self, e, g):
        """Stores potential and forces in this class for convenience"""
        self.pot = e
        self.f = -g

    def __call__(self, x, ret=True, new_disc=True):
        """Computes spring energy and gradient for instanton optimization step"""

        if new_disc:
            coef = self.coef
        elif new_disc == "one":
            coef = np.ones(self.coef.shape)
        else:
            coef = new_disc.reshape(self.coef.shape)

        if x.shape[0] == 1:  # only one bead
            self.dbeads.q = x
            e = 0.0
            g = np.zeros(x.shape[1])
            self.save(e, g)

        else:
            self.dbeads.q = x
            e = 0.00
            g = np.zeros(self.dbeads.q.shape, float)

            # OLD reference
            # for i in range(self.dbeads.nbeads - 1):
            #    dq = self.dbeads.q[i + 1, :] - self.dbeads.q[i, :]
            #    e += self.omega2 * 0.5 * np.dot(self.dbeads.m3[0] * dq, dq)
            # for i in range(0, self.dbeads.nbeads - 1):
            #    #g[i, :] += self.omega2 * (self.dbeads.q[i, :] - self.dbeads.q[i + 1, :])
            #    g[i, :] += self.dbeads.m3[i, :] * self.omega2 * (self.dbeads.q[i, :] - self.dbeads.q[i + 1, :])
            # for i in range(1, self.dbeads.nbeads):
            #    #g[i, :] +=  self.omega2 * (self.dbeads.q[i, :] - self.dbeads.q[i - 1, :])
            #    g[i, :] += self.dbeads.m3[i, :] * self.omega2 * (self.dbeads.q[i, :] - self.dbeads.q[i - 1, :])
            gq_k = np.dot(self.C, self.dbeads.q)  # normal mode coordinates. 
            g = self.dbeads.m3[0] * np.dot(
                self.C.T, gq_k * (self.omegak**2)[:, np.newaxis]
            )

            # TODO this is a bug here. they forget to compute e.
            e = 0.5 * np.sum( np.power(self.omegak,2)[:, np.newaxis] * (self.dbeads.m3 * np.power(gq_k, 2)) )

            # With new discretization #This can be expressed as matrix multp
            if False:  # ALBERTO
                for i in range(self.dbeads.nbeads - 1):
                    dq = (self.dbeads.q[i + 1, :] - self.dbeads.q[i, :]) / np.sqrt(
                        coef[i + 1]
                    )  # coef[0] and coef[-1] do not enter
                    e += self.omega2 * 0.5 * np.dot(self.dbeads.m3[0] * dq, dq)
                for i in range(0, self.dbeads.nbeads - 1):
                    g[i, :] += (
                        self.dbeads.m3[i, :]
                        * self.omega2
                        * (
                            self.dbeads.q[i, :] / coef[i + 1]
                            - self.dbeads.q[i + 1, :] / coef[i + 1]
                        )
                    )
                for i in range(1, self.dbeads.nbeads):
                    g[i, :] += (
                        self.dbeads.m3[i, :]
                        * self.omega2
                        * (
                            self.dbeads.q[i, :] / coef[i]
                            - self.dbeads.q[i - 1, :] / coef[i]
                        )
                    )

            self.save(e, g)

        if ret:
            return e, g

    @staticmethod
    def spring_hessian(natoms, nbeads, m3, omega2, mode="half", coef=None):
        """Compute the 'spring hessian'

        OUT    h       = hessian with only the spring terms ('spring hessian')
        """
        if coef is None:
            coef = np.ones(nbeads + 1).reshape(-1, 1)

        # Check size of discretization:
        if coef.size != nbeads + 1:
            print("@spring_hessian: discretization size error")
            sys.exit()

        info(" @spring_hessian", verbosity.high)
        ii = natoms * 3
        h = np.zeros([ii * nbeads, ii * nbeads])

        if nbeads == 1:
            return h

        # Diagonal
        h_sp = m3 * omega2
        diag1 = np.diag(h_sp)
        # diag2 = np.diag(2.0 * h_sp)

        if mode == "half":
            i = 0
            h[i * ii : (i + 1) * ii, i * ii : (i + 1) * ii] += diag1 / coef[1]
            i = nbeads - 1
            h[i * ii : (i + 1) * ii, i * ii : (i + 1) * ii] += diag1 / coef[-2]
            for i in range(1, nbeads - 1):
                h[i * ii : (i + 1) * ii, i * ii : (i + 1) * ii] += diag1 * (
                    1.0 / coef[i] + 1.0 / coef[i + 1]
                )
        elif mode == "splitting" or mode == "full":
            for i in range(0, nbeads):
                h[i * ii : (i + 1) * ii, i * ii : (i + 1) * ii] += diag1 * (
                    1.0 / coef[i] + 1.0 / coef[i + 1]
                )
        else:
            raise ValueError("We can't compute the spring hessian.")

        # Non-Diagonal
        ndiag = np.diag(-h_sp)
        # Quasi-band
        for i in range(0, nbeads - 1):
            h[i * ii : (i + 1) * ii, (i + 1) * ii : (i + 2) * ii] += ndiag * (
                1.0 / coef[i + 1]
            )
            h[(i + 1) * ii : (i + 2) * ii, i * ii : (i + 1) * ii] += ndiag * (
                1.0 / coef[i + 1]
            )

        # Corner
        if mode == "full":
            h[0:ii, (nbeads - 1) * ii : (nbeads) * ii] += ndiag / coef[0]
            h[(nbeads - 1) * ii : (nbeads) * ii, 0:ii] += ndiag / coef[0]

        return h

    # def __call__(self, x, ret=True, new_disc=True):
    #    """Computes spring energy and gradient for instanton optimization step"""


class Mapper(object):
    """Creation of the multi-dimensional function that is the proxy between all the energy and force components and the optimization algorithm.
    It also handles fixatoms"""

    def __init__(self, esum=False):
        self.sm = SpringMapper()  # spring term
        self.gm = PesMapper()  # physical potential energy term
        self.esum = esum

    def initialize(self, q, forces):
        self.gm.initialize(q, forces)

        e1, g1 = self.gm.evaluate()  # compute physical potential e1 & gradient g1.  e1 is a matrix of shape [nbeads]
        e2, g2 = self.sm(q)   # compute spring potential e2 and gradient g2
        g = self.fix.get_active_vector(g1 + g2, 1)
        e = np.sum(e1) + np.sum(e2) 

        self.save(e, g)

    def save(self, e, g):
        self.pot = e
        self.f = -g

    def bind(self, dumop):
        self.temp = dumop.temp
        self.beads = dumop.beads
        self.forces = dumop.forces
        self.cell = dumop.cell
        self.nm = dumop.nm
        self.rp_factor = dumop.rp_factor

        self.fixatoms = dumop.fixatoms
        self.fix = dumop.fix
        self.fixbeads = self.fix.fixbeads

        self.options = dumop.options

        self.coef = np.ones(self.beads.nbeads + 1).reshape(-1, 1)
        self.set_coef(self.options["discretization"])

        self.friction = self.options["friction"]
        if self.friction:
            self.frictionSD = self.options["frictionSD"]
            self.gm = FrictionMapper(self.frictionSD, self.options["eta0"])
            self.gm.bind(self)
            self.gm.set_fric_spec_dens(
                dumop.options["fric_spec_dens"], dumop.options["fric_spec_dens_ener"]
            )
        else:
            self.gm.bind(self)

        self.sm.bind(self)

    def set_coef(self, coef):
        """Sets coeficients for non-uniform instanton calculation"""
        self.coef[:] = coef.reshape(-1, 1)

    def __call__(self, x, mode="all", apply_fix=True, new_disc=True, ret=True):
        if mode == "all":
            e1, g1 = self.sm(x, new_disc)  # e1 is a number: energy term of spring potential.
            e2, g2 = self.gm(x, new_disc)  # e2 is an array of size [nbeads]. physical potential energy of each bead.
            e = np.sum(e1) + np.sum(e2)
            g = np.add(g1, g2)

        elif mode == "physical":
            e, g = self.gm(x, new_disc)
        elif mode == "springs":
            e, g = self.sm(x, new_disc)
        else:
            softexit.trigger("Mode not recognized when calling  FullMapper")

        if apply_fix:
            g = self.fix.get_active_vector(g, 1)

        if mode == "all":
            self.save(np.sum(e), g)

        if self.esum:
            e = np.sum(e)

        if ret:
            return e, g


class DummyOptimizer:
    """Dummy class for all optimization classes"""

    def __init__(self):
        """Initializes object for PesMapper (physical potential, forces and hessian)
        and SpringMapper ( spring potential,forces and hessian)"""

        self.options = {}  # Optimization options
        self.optarrays = {}  # Optimization arrays

        self.mapper = Mapper()  # handle all potentials.

        self.exit = False
        self.init = False

        # output maximum gradient. 
        self.gradient_file = None 

    def bind(self, geop):
        """
        Bind optimization options and call bind function of Mappers (get beads, cell,forces)
        check whether force size,  Hessian size from  match system size
        """

        self.beads = geop.beads
        self.cell = geop.cell
        self.forces = geop.forces
        self.fixcom = geop.fixcom
        self.fixatoms = geop.fixatoms

        self.fix = Fix(self.fixatoms, self.beads, self.beads.nbeads)
        self.nm = geop.nm
        self.rp_factor = geop.rp_factor

        self.output_maker = geop.output_maker

        # The resize action must be done before the bind

        if geop.optarrays["old_x"].size != self.beads.q.size:
            if geop.optarrays["old_x"].size == 0:
                geop.optarrays["old_x"] = np.zeros(
                    (self.beads.nbeads, 3 * self.beads.natoms), float
                )
            else:
                raise ValueError("Old positions size does not match system size")
        if geop.optarrays["old_u"].size != self.beads.nbeads:
            if geop.optarrays["old_u"].size == 0:
                geop.optarrays["old_u"] = np.zeros(self.beads.nbeads, float)
            else:
                raise ValueError("Old potential energy size does not match system size")
        if geop.optarrays["old_f"].size != self.beads.q.size:
            if geop.optarrays["old_f"].size == 0:
                geop.optarrays["old_f"] = np.zeros(
                    (self.beads.nbeads, 3 * self.beads.natoms), float
                )
            else:
                raise ValueError("Old forces size does not match system size")

        # Temperature
        self.temp = geop.ensemble.temp
        if geop.ensemble.temp == -1.0 or geop.ensemble.temp == 1.0:
            # This is due to a little inconsistency on the default value
            if self.beads.nbeads != 1:
                raise ValueError(
                    "Temperature must be specified for an Instanton calculation "
                )

        # Optimization mode
        self.options["mode"] = geop.options["mode"]

        # Generic optimization
        if geop.options["discretization"].size != self.beads.nbeads + 1:
            if geop.options["discretization"].size == 0:
                geop.options["discretization"] = np.ones(self.beads.nbeads + 1, float)
            else:
                raise ValueError("Discretization coefficients do not match system size")

        self.options["max_ms"] = geop.options["max_ms"]
        self.options["max_e"] = geop.options["max_e"]
        self.options["discretization"] = geop.options["discretization"]
        self.options["friction"] = geop.options["friction"]
        self.options["frictionSD"] = geop.options["frictionSD"]
        if self.options["friction"]:
            self.options["eta0"] = geop.options["eta0"]
            self.options["fric_spec_dens"] = geop.options["fric_spec_dens"]
            self.options["fric_spec_dens_ener"] = geop.options["fric_spec_dens_ener"]
        self.options["tolerances"] = geop.options["tolerances"]
        self.optarrays["big_step"] = geop.optarrays["big_step"]
        self.optarrays["old_x"] = geop.optarrays["old_x"]
        self.optarrays["old_u"] = geop.optarrays["old_u"]
        self.optarrays["old_f"] = geop.optarrays["old_f"]
        self.options["opt"] = geop.options["opt"]  # optimization algorithm

        # Generic instanton
        self.options["save"] = geop.options["save"]
        self.options["prefix"] = geop.options["prefix"]
        self.optarrays["delta"] = geop.optarrays["delta"]
        self.options["hessian_final"] = geop.options["hessian_final"]
        self.optarrays["energy_shift"] = geop.optarrays["energy_shift"]

        # self.fix = Fix(geop.beads.natoms, geop.fixatoms, geop.beads.nbeads)

        self.mapper.bind(self)

    def initial_geo(self):
        """Generates the initial instanton geometry by stretching the transitions-state geometry along the mode with imaginary frequency"""

        info(
            " @GEOP: We stretch the initial geometry with an 'amplitude' of {:4.2f}".format(
                self.optarrays["delta"]
            ),
            verbosity.low,
        )

        fix_onebead = Fix(self.fixatoms, self.beads, 1)
        # get initial hessian of 1 bead, excluding the fixed atoms.
        active_hessian = fix_onebead.get_active_vector(
            self.optarrays["initial_hessian"], 2
        )
        # get eigenvector along the imaginary mode for one bead.
        active_imvector = get_imvector(
            active_hessian, fix_onebead.fixbeads.m3[0].flatten()
        )
        # convert eigenvector for active dof into full dof by setting fixed atom index to 0.
        imvector = fix_onebead.get_full_vector(active_imvector, 1).flatten()
        # extend beads along the imaginary vector direction. note this is physical coordinate, not mass-weighted coordinate.
        for i in range(self.beads.nbeads):
            self.beads.q[i, :] += (
                self.optarrays["delta"]
                * np.cos(i * np.pi / float(self.beads.nbeads - 1))
                * imvector[:]
            )

    def exitstep(self, d_x_max, step):
        """Exits the simulation step. Computes time, checks for convergence."""
        self.qtime += time.time()

        tolerances = self.options["tolerances"]
        d_u = self.forces.pot - self.optarrays["old_u"].sum()
        # active_force = self.fix.get_active_vector(self.forces.f, 1) + self.im.f

        active_force = self.mapper.f

        info(
            " @Exit step: Energy difference: {:4.2e}, (condition: {:4.2e})".format(
                np.absolute(d_u / self.fix.fixbeads.natoms), tolerances["energy"]
            ),
            verbosity.low,
        )
        info(
            " @Exit step: Maximum force component: {:4.2e}, (condition: {:4.2e})".format(
                np.amax(np.absolute(active_force)), tolerances["force"]
            ),
            verbosity.low,
        )
        info(
            " @Exit step: Maximum component step component: {:4.2e}, (condition: {:4.2e})".format(
                d_x_max, tolerances["position"]
            ),
            verbosity.low,
        )

        if (
            (np.absolute(d_u / self.mapper.sm.dbeads.natoms) <= tolerances["energy"])
            and (
                (np.amax(np.absolute(active_force)) <= tolerances["force"])
                or (
                    np.linalg.norm(
                        self.forces.f.flatten() - self.optarrays["old_f"].flatten()
                    )
                    <= 1e-08
                )
            )
            and (d_x_max <= tolerances["position"])
        ):
            print_instanton_geo(
                self.options["prefix"] + "_FINAL",
                step,
                self.beads.nbeads,
                self.beads.natoms,
                self.beads.names,
                self.beads.q,
                self.forces.f,
                self.forces.pots,
                self.cell,
                self.optarrays["energy_shift"],
                self.output_maker,
            )
            if not self.options["hessian_final"]:
                info("We are not going to compute the final hessian.", verbosity.low)
                info(
                    "Warning, The current hessian is not the real hessian is only an approximation .",
                    verbosity.low,
                )

            else:
                info("We are going to compute the final hessian", verbosity.low)
                current_hessian = get_hessian(
                    gm=self.mapper.gm,
                    x0=self.beads.q.copy(),
                    natoms=self.beads.natoms,
                    nbeads=self.beads.nbeads,
                    fixatoms=self.fixatoms,
                    friction=self.options["frictionSD"],
                )

                if self.options["friction"] and self.options["frictionSD"]:
                    friction_hessian = current_hessian[1]
                    self.optarrays["fric_hessian"][:] = self.fix.get_full_vector(
                        friction_hessian, 4
                    )
                    # self.optarrays["fric_hessian"][:] = friction_hessian #ALBERTO
                    print_instanton_hess(
                        self.options["prefix"] + "fric_FINAL",
                        step,
                        self.optarrays["fric_hessian"],
                        self.output_maker,
                    )

                    phys_hessian = current_hessian[0]

                else:
                    phys_hessian = current_hessian

                # self.optarrays["hessian"][:] = self.fix.get_full_vector(phys_hessian, 2) #ALBERTO
                self.optarrays["hessian"][:] = phys_hessian  # ALBERTO

            print_instanton_hess(
                self.options["prefix"] + "_FINAL",
                step,
                self.optarrays["hessian"],
                self.output_maker,
            )

            return True
            # If we just exit here, the last step (including the last hessian) will not be in the RESTART file

        return False

    def output_max_gradient(self, step):
        '''
        output gradient to self.gradient_file
        '''
        if self.gradient_file.closed:
            warnings.warn("gradient file is closed when we try to output maximum gradient. Error")
            self.exit = True 
        else:
            active_force = self.mapper.f 
            maximum_force = np.amax(np.absolute(active_force))

            precision = 10
            formatted_max_force = "{:.{}f}".format(maximum_force,precision)
            self.gradient_file.write(str(step) + "  ")
            self.gradient_file.write(formatted_max_force + "\n")

    def update_pos_force(self):
        """Update positions and forces"""

        self.beads.q[:] = self.mapper.gm.dbeads.q[:]

        # This forces the update of the forces
        self.forces.transfer_forces(self.mapper.gm.dforces)

    def update_old_pos_force(self):
        """Update 'old' positions and forces arrays"""

        self.optarrays["old_x"][:] = self.beads.q
        self.optarrays["old_u"][:] = self.forces.pots
        self.optarrays["old_f"][:] = self.forces.f

    def print_geo(self, step):
        """Small interface to call the function that prints thet instanton geometry"""

        if (
            self.options["save"] > 0 and np.mod(step, self.options["save"]) == 0
        ) or self.exit:
            print_instanton_geo(
                self.options["prefix"],
                step,
                self.beads.nbeads,
                self.beads.natoms,
                self.beads.names,
                self.beads.q,
                self.forces.f,
                self.forces.pots,
                self.cell,
                self.optarrays["energy_shift"],
                self.output_maker,
            )



    def pre_step(self, step=None, adaptative=False):
        """General tasks that have to be performed before actual step"""

        if self.exit:
            # exit the program here.
            # close the gradient output file
            self.gradient_file.close()

            softexit.trigger(
                status="success",
                message="Geometry optimization converged. Exiting simulation",
            )

        # initialize instanton geometry at step 0.
        if not self.init:
            self.initialize(step)  

        if adaptative:
            softexit.trigger(
                status="bad",
                message="Adaptative discretization is not fully implemented",
            )
            # new_coef = <implement_here>
            # self.mapper.set_coef(coef)
            raise NotImplementedError

        self.qtime = -time.time()
        info("\n Instanton optimization STEP {}".format(step), verbosity.low)

        # get the active array (excluding fixed atoms) according to the key of arrays.
        # see utils.instools.py FIX class, get_active_array() function.  "old_u", "big_step", "delta", "energy_shift", "initial_hessian", "old_x", "old_f", "d", "hessian", "qlist", "glist" (for lbfgs), "fric_hessian"
        activearrays = self.fix.get_active_array(self.optarrays)  

        return activearrays

    def step(self, step=None):
        """Dummy simulation time step which does nothing."""
        pass

    def opt_coef(self, coef):
        # func = lambda x: 2 * np.sum(x) - x[0] - x[-1]
        def func(x):
            return 2 * np.sum(x) - x[0] - x[-1]

        coef = np.absolute(coef)
        s = func(coef)
        coef *= 2 * self.sm.dbeads.nbeads / s
        # c0   = 2*self.sm.dbeads.nbeads - 2*np.sum(coef)
        # coef = np.insert(coef,0,c0)

        # self.im.set_coef(coef)

        fphys = self.gm.dforces.f * ((coef[1:] + coef[:-1]) / 2).reshape(-1, 1)
        e, gspring = self.sm(self.sm.dbeads.q)
        return np.amax(np.absolute(-gspring + fphys))


class HessianOptimizer(DummyOptimizer):
    """Instanton Rate calculation"""

    def bind(self, geop):
        # call bind function from DummyOptimizer
        super(HessianOptimizer, self).bind(geop)

        self.options["hessian_update"] = geop.options["hessian_update"]
        self.options["hessian_asr"] = geop.options["hessian_asr"]

        if len(self.fixatoms) > 0:
            info(" 'fixatoms' is enabled. Setting asr to None", verbosity.low)
            self.options["hessian_asr"] = "none"
        #        self.output_maker = geop.output_maker
        self.options["hessian_init"] = geop.options["hessian_init"]
        self.optarrays["initial_hessian"] = None
        print(geop.optarrays["hessian"].size)

        if geop.optarrays["hessian"].size != (
            self.beads.natoms * 3 * self.beads.q.size
        ):
            if geop.optarrays["hessian"].size == (self.beads.natoms * 3) ** 2:
                self.optarrays["initial_hessian"] = geop.optarrays["hessian"].copy()
                geop.optarrays["hessian"] = np.zeros(
                    (self.beads.natoms * 3, self.beads.q.size), float
                )

            elif geop.optarrays["hessian"].size == 0 and geop.options["hessian_init"]:
                info(
                    " Initial hessian is not provided. We are going to compute it.",
                    verbosity.low,
                )
                geop.optarrays["hessian"] = np.zeros(
                    (self.beads.natoms * 3, self.beads.q.size)
                )

                if (
                    (self.beads.q - self.beads.q[0]) == 0
                ).all() and self.beads.nbeads > 1:
                    raise ValueError(
                        """We need an initial hessian in order to create our initial
                    instanton geometry. Please provide a (1-bead) hessian or an initial instanton geometry."""
                    )

            else:
                raise ValueError(
                    " 'Hessian_init' is false, an initial hessian (of the proper size) must be provided."
                )

        self.optarrays["hessian"] = geop.optarrays["hessian"]
        if self.options["friction"]:
            if geop.options["eta0"].shape == (0, 0):
                geop.options["eta0"] = np.zeros(
                    (self.beads.natoms * 3, self.beads.natoms * 3)
                )
            assert geop.options["eta0"].shape == (
                self.beads.natoms * 3,
                self.beads.natoms * 3,
            ), "Please provide a friction tensor with the appropiate shape"
            self.options["eta0"] = geop.options["eta0"]

            if self.options["frictionSD"]:
                if geop.optarrays["fric_hessian"].shape != (
                    self.beads.nbeads,
                    self.beads.natoms * 3,
                    self.beads.natoms * 3,
                    self.beads.natoms * 3,
                ):
                    if geop.options["hessian_init"]:
                        geop.optarrays["fric_hessian"] = np.zeros(
                            (
                                self.beads.nbeads,
                                self.beads.natoms * 3,
                                self.beads.natoms * 3,
                                self.beads.natoms * 3,
                            )
                        )
                    else:
                        raise ValueError(
                            """
              'Hessian_init' is false, 'friction' is true so an initial fric_hessian (of the proper size) must be provided.
                    """
                        )
                self.optarrays["fric_hessian"] = geop.optarrays["fric_hessian"]

    def initialize(self, step):
        if step == 0:
            info(" @GEOP: Initializing INSTANTON", verbosity.low)

            # open file that output maximum gradient at each time step.
            gradient_file_name = self.options["prefix"]+"_gradient.txt"
            gradient_file_name = self.output_maker.prefix + "." + gradient_file_name
            self.gradient_file = open(gradient_file_name, "w")

            # for nbeads = 1, using Nichols' algorithm (walking uphill along one direction), we will reach the transition state. Thus (bead = 1) == Classical TS search. 
            if self.beads.nbeads == 1:
                info(" @GEOP: Classical TS search", verbosity.low)

            else:
                # If the coordinates in all the imaginary time slices are the same
                if ((self.beads.q - self.beads.q[0]) == 0).all():
                    self.initial_geo()  # initialize the geometry of instanton by stretching along imaginary frequency mode.
                    
                    self.options["hessian_init"] = True

                else:
                    info(
                        " @GEOP: Starting from the provided geometry in the extended phase space",
                        verbosity.low,
                    )
                    # in case hessian is of size [3*natom, 3*natom]: hessian of a single bead. 
                    # self.optarrays["initial_hessian"] = hessian. See bind() function for optimizer.
                    # otherwise, self.optarrays["initial_hessian"] = None , hessian is already provided from the file
                    if not (self.optarrays["initial_hessian"] is None):
                        raise ValueError(
                            " You have to provided a hessian with size (3 x natoms)^2 but also geometry in"
                            " the extended phase space (nbeads>1). Please check the inputs\n"
                        )

        # Initialize all the mappers for potential and forces. compute forces & potential. 
        self.mapper.initialize(self.beads.q, self.forces)

        # compute hessian for the initial instanton geometry. here full_hessian has shape [3 * natoms, 3 * natoms * nbeads]
        # full_hessian is computed using finite difference method ( (f(x+h)-f(x-h))/(2h) )
        if self.options["hessian_init"]:
            full_hessian = get_hessian(
                gm=self.mapper.gm,
                x0=self.beads.q.copy(),
                natoms=self.beads.natoms,
                nbeads=self.beads.nbeads,
                fixatoms=self.fixatoms,
                friction=self.options["frictionSD"],
            )
            if self.options["friction"] and self.options["frictionSD"]:
                phys_hessian = full_hessian[0]
                friction_hessian = full_hessian[1]
                # self.optarrays["fric_hessian"][:] = self.fix.get_full_vector( friction_hessian, 4 ) #ALBERTO
                self.optarrays["fric_hessian"][:] = friction_hessian[:]
            else:
                phys_hessian = full_hessian

            # self.optarrays["hessian"][:] = self.fix.get_full_vector(phys_hessian, 2) #ALBERTO
            self.optarrays["hessian"][:] = phys_hessian

            # save hessian.
            print_instanton_hess(self.options["prefix"] + "_initial",
                                0, 
                                self.optarrays["hessian"],
                                self.output_maker)

        #   self.gm.save(self.forces.pots, self.forces.f)
        self.update_old_pos_force()

        self.init = True

    def update_hessian(self, update, active_hessian, new_x, d_x, d_g):
        """Update hessian
        :param: update: self.options["hessian_update"]: "powell" or "recompute"
        :param: active_hessian: reduced physical hessian of active atoms. [3 * self.fix.fixbeads.natoms, 3 * self.fix.fixbeads.natoms * nbeads]. here self.fix.fixbeads.natoms is number of active atoms after subtracting fixed atoms.
        :param: new_x: coordinate array of all atoms in new position. (including fixed atoms)  size: [nbeads, 3 * natoms]
        :param: d_x: displacement array for active atoms, shape: [nbeads, 3 * self.fix.fixbeads.natoms]
        :param: d_g: finite difference of gradient, shape [nbeads, 3 * self.fix.fixbeads.natoms]
        Both hessian and active_hessian is updated.
        """

        if update == "powell":
            i = self.fix.fixbeads.natoms * 3
            for j in range(self.fix.fixbeads.nbeads):
                aux = active_hessian[:, j * i : (j + 1) * i]
                dg = d_g[j, :]
                dx = d_x[j, :]
                Powell(dx, dg, aux)
            phys_hessian = active_hessian
            if self.options["friction"]:
                info(
                    "Powell update for friction hessian is not implemented. We move on without updating it. In all tested cases this is not a problem",
                    verbosity.medium,
                )
        elif update == "DFP":   # need to update dictionary "hessian_update" in /inputs/motion/instanton.py. add "DFP" to "options"
            i = self.fix.fixbeads.natoms * 3 
            for j in range(self.fix.fixbeads.nbeads):
                aux = active_hessian[:, j * i : (j+1) * i]
                dg = d_g[j, :]
                dx = d_x[j, :]
                Davidon_Fletcher_Powell(dx, dg, aux)  # here aux is updated in the program, thus active_hessian is also updated.
            phys_hessian = active_hessian
            info("Customary DFP method Using Davidon_Fletcher_Powell method to update  Hessian", verbosity.medium,)


        elif update == "recompute":
            active_hessian = get_hessian(
                gm=self.mapper.gm,
                x0=new_x,
                natoms=self.beads.natoms,
                nbeads=self.beads.nbeads,
                fixatoms=self.fixatoms,
                friction=self.options["frictionSD"],
            )

            if self.options["friction"] and self.options["frictionSD"]:
                phys_hessian = active_hessian[0]
                friction_hessian = active_hessian[1]
                self.optarrays["fric_hessian"][:] = self.fix.get_full_vector(
                    friction_hessian, 4
                )
            else:
                phys_hessian = active_hessian
        
        
        self.optarrays["hessian"][:] = self.fix.get_full_vector(phys_hessian, 2)  # transform phys_hessian into full_hessian and assign to "hessian" in optarrays.

    def print_hess(self, step):
        if (
            self.options["save"] > 0 and np.mod(step, self.options["save"]) == 0
        ) or self.exit:
            print_instanton_hess(
                self.options["prefix"],
                step,
                self.optarrays["hessian"],
                self.output_maker,
            )
            if self.options["friction"]:
                print_instanton_hess(
                    self.options["prefix"] + "_fric",
                    step,
                    self.optarrays["fric_hessian"],
                    self.output_maker,
                )

    def post_step(self, step, new_x, d_x, activearrays):
        """General tasks that have to be performed after finding the new step"""

        d_x_max = np.amax(np.absolute(d_x))
        info("Current step norm = {}".format(d_x_max), verbosity.medium)

        # Get energy and forces(f) for the new position
        self.mapper(new_x, ret=False)

        # Update force and gradient
        # get new active force. compute gradient difference d_g = - d_f = old_f - new_f
        f = self.fix.get_active_vector(self.mapper.gm.f, t=1)
        d_g = np.subtract(activearrays["old_f"], f)

        # Update hessian.
        self.update_hessian(
            self.options["hessian_update"], activearrays["hessian"], new_x, d_x, d_g
        )

        # Update position and forces. in the optimization class
        self.update_pos_force()

        #  Print geometry & hessian.
        self.print_geo(step)
        self.print_hess(step)

        self.output_max_gradient(step)

        # Check Exit and only then update old arrays
        self.exit = self.exitstep(d_x_max, step)

        # assign the new_u, new_f, new_x to old_u, old_f, old_x. update old arrays
        self.update_old_pos_force()


class NicholsOptimizer(HessianOptimizer):
    """Class that implements a nichols optimizations. It can find first order saddle points or minimum"""

    def bind(self, geop):
        # call bind function from HessianOptimizer
        super(NicholsOptimizer, self).bind(geop)

    def initialize(self, step):
        # call initialize function from HessianOptimizer
        super(NicholsOptimizer, self).initialize(step)

    def step(self, step=None):
        """Does one simulation step."""

        # check if simulation is complete & exit.  check whether system has been initialized, if not, initialize the instanton geometry. 
        # return activearrays, excluding the atom to be fixed. 
        activearrays = self.pre_step(step)  

        # First construct complete hessian from reduced
        h0 = red2comp(
            activearrays["hessian"],
            self.fix.fixbeads.nbeads,
            self.fix.fixbeads.natoms,
            self.mapper.coef,
        )

        # Add spring terms to the physical hessian
        h = np.add(self.mapper.sm.h, h0)

        # Add friction terms to the hessian
        if self.options["friction"]:
            eta_active = self.fix.get_active_vector(self.mapper.gm.eta, 5)
            if self.options["frictionSD"]:
                h_fric = self.mapper.gm.get_fric_rp_hessian(
                    activearrays["fric_hessian"], eta_active, self.options["frictionSD"]
                )
            else:
                h_fric = self.mapper.gm.get_fric_rp_hessian(
                    None, eta_active, self.options["frictionSD"]
                )
            h = np.add(h, h_fric)

        # Get eigenvalues and eigenvector of hessian.  excluding rotational & translational dof.
        d, w = clean_hessian(
            h,
            self.fix.fixbeads.q,
            self.fix.fixbeads.natoms,
            self.fix.fixbeads.nbeads,
            self.fix.fixbeads.m,
            self.fix.fixbeads.m3,
            self.options["hessian_asr"],
        )

        # d,w =np.linalg.eigh(h1) #Cartesian
        # print 3 lowest frequencies
        info(
            "\n@Nichols: 1st freq {} cm^-1".format(
                units.unit_to_user(
                    "frequency", "inversecm", np.sign(d[0]) * np.sqrt(np.absolute(d[0]))
                )
            ),
            verbosity.medium,
        )
        info(
            "@Nichols: 2nd freq {} cm^-1".format(
                units.unit_to_user(
                    "frequency", "inversecm", np.sign(d[1]) * np.sqrt(np.absolute(d[1]))
                )
            ),
            verbosity.medium,
        )
        info(
            "@Nichols: 3rd freq {} cm^-1".format(
                units.unit_to_user(
                    "frequency", "inversecm", np.sign(d[2]) * np.sqrt(np.absolute(d[2]))
                )
            ),
            verbosity.medium,
        )
        # info('@Nichols: 4th freq {} cm^-1'.format(units.unit_to_user('frequency','inversecm',np.sign(d[3])*np.sqrt(np.absolute(d[3])))),verbosity.medium)
        # info('@Nichols: 8th freq {} cm^-1\n'.format(units.unit_to_user('frequency','inversecm',np.sign(d[7])*np.sqrt(np.absolute(d[7])))),verbosity.medium)


        # Find new movement direction
        if self.options["mode"] == "rate":
            d_x = nichols(
                self.mapper.f,
                d,
                w,
                self.fix.fixbeads.m3,
                activearrays["big_step"],
            )
        elif self.options["mode"] == "splitting":
            # splitting corresponds to the kink path, which is minimum value of a linear polymer connecting two minimums.
            d_x = nichols(
                self.mapper.f,
                d,
                w,
                self.fix.fixbeads.m3,
                activearrays["big_step"],
                mode=0,
            )

        # Rescale step if necessary
        if np.amax(np.absolute(d_x)) > activearrays["big_step"]:
            info(
                "Step norm, scaled down to {}".format(activearrays["big_step"]),
                verbosity.low,
            )
            d_x *= activearrays["big_step"] / np.amax(np.absolute(d_x))

        # Get the new full-position
        d_x_full = self.fix.get_full_vector(d_x, t=1)  # convert from active atom d_x array to full atom d_x array(setting fixed atom dx = 0)
        new_x = self.optarrays["old_x"].copy() + d_x_full

        self.post_step(step, new_x, d_x, activearrays)


class NROptimizer(HessianOptimizer):
    """Class that implements a Newton-Raphson optimizations. It can find first order saddle points or minima"""

    def bind(self, geop):
        # call bind function from HessianOptimizer
        super(NROptimizer, self).bind(geop)

    def initialize(self, step):
        # call initialize function from HessianOptimizer
        super(NROptimizer, self).initialize(step)

    def step(self, step=None):
        """Does one simulation time step."""
        activearrays = self.pre_step(step)

        dyn_mat = get_dynmat(
            #activearrays["hessian"], self.mapper.sm.dbeads.m3, self.mapper.sm.dbeads.nbeads  
            activearrays["hessian"], self.fix.fixbeads.m3, self.fix.fixbeads.nbeads
        )
        h_up_band = banded_hessian(
            dyn_mat, self.mapper.sm, masses=False, shift=0.0000001
        )  # create upper band dynmat matrix

        fff = activearrays["old_f"] * (self.mapper.coef[1:] + self.mapper.coef[:-1]) / 2
        f = (fff + self.mapper.sm.f).reshape(                        # here f is spring force + physical force
            self.mapper.sm.dbeads.natoms * 3 * self.mapper.sm.dbeads.nbeads, 1
        )
        f = np.multiply(f, self.mapper.sm.dbeads.m3.reshape(f.shape) ** -0.5)  # mass weighted coordinate

        d_x = invmul_banded(h_up_band, f).reshape(self.mapper.sm.dbeads.q.shape)  # inverse of hessian * force. NR step.
        d_x = np.multiply(d_x, self.mapper.sm.dbeads.m3**-0.5)  # transform back to physical coordinate.

        # Rescale step if necessary
        if np.amax(np.absolute(d_x)) > activearrays["big_step"]:
            info(
                "Step norm, scaled down to {}".format(activearrays["big_step"]),
                verbosity.low,
            )
            d_x *= activearrays["big_step"] / np.amax(np.absolute(d_x))

        # Get the new full-position
        d_x_full = self.fix.get_full_vector(d_x, t=1)
        new_x = self.optarrays["old_x"].copy() + d_x_full

        self.post_step(step, new_x, d_x, activearrays)


class LanczosOptimizer(HessianOptimizer):
    """Class that implements a modified Nichols algorithm based on Lanczos diagonalization to avoid constructing and diagonalizing
    the full (3*natoms*nbeads)^2 matrix"""

    def bind(self, geop):
        # call bind function from HessianOptimizer
        super(LanczosOptimizer, self).bind(geop)

    def initialize(self, step):
        # call initialize function from HessianOptimizer
        super(LanczosOptimizer, self).initialize(step)

    def step(self, step=None):
        """Does one simulation step."""

        activearrays = self.pre_step(step)

        f = self.mapper.f.reshape(
            self.fix.fixbeads.natoms * 3 * self.fix.fixbeads.nbeads, 1
        )

        # banded = False
        banded = True   # choose the banded form.
        if banded:
            # BANDED Version
            # MASS-scaled
            dyn_mat = get_dynmat(
                activearrays["hessian"], self.fix.fixbeads.m3, self.fix.fixbeads.nbeads
            )

            h_up_band = banded_hessian(
                dyn_mat, self.mapper.sm, masses=False, shift=0.0000001
            )  # create upper band matrix
            f = np.multiply(f, self.fix.fixbeads.m3.reshape(f.shape) ** -0.5)
            # CARTESIAN
            # h_up_band = banded_hessian(activearrays["hessian"], self.sm.masses=True)  # create upper band matrix

            d = diag_banded(h_up_band)  # three lowest eigenvalues of hessian 
        else:
            # FULL dimensions version
            h_0 = red2comp(
                activearrays["hessian"],
                self.sm.dbeads.nbeads,
                self.sm.dbeads.natoms,
                self.mapper.coef,
            )
            h_test = np.add(self.sm.h, h_0)  # add spring terms to the physical hessian
            d, w = clean_hessian(
                h_test,
                self.sm.dbeads.q,
                self.sm.dbeads.natoms,
                self.sm.dbeads.nbeads,
                self.sm.dbeads.m,
                self.sm.dbeads.m3,
                None,
            )
            # CARTESIAN
            # d,w =np.linalg.eigh(h_test) #Cartesian
        info(
            "\n@Lanczos: 1st freq {} cm^-1".format(
                units.unit_to_user(
                    "frequency", "inversecm", np.sign(d[0]) * np.sqrt(np.absolute(d[0]))
                )
            ),
            verbosity.medium,
        )
        info(
            "@Lanczos: 2nd freq {} cm^-1".format(
                units.unit_to_user(
                    "frequency", "inversecm", np.sign(d[1]) * np.sqrt(np.absolute(d[1]))
                )
            ),
            verbosity.medium,
        )
        info(
            "@Lanczos: 3rd freq {} cm^-1\n".format(
                units.unit_to_user(
                    "frequency", "inversecm", np.sign(d[2]) * np.sqrt(np.absolute(d[2]))
                )
            ),
            verbosity.medium,
        )

        if d[0] > 0:
            if d[1] / 2 > d[0]:
                alpha = 1
                lamb = (2 * d[0] + d[1]) / 4
            else:
                alpha = (d[1] - d[0]) / d[1]
                lamb = (3 * d[0] + d[1]) / 4  # midpoint between b[0] and b[1]*(1-alpha/2)
        elif d[1] < 0:  # Jeremy Richardson
            if d[1] >= d[0] / 2:
                alpha = 1
                lamb = (d[0] + 2 * d[1]) / 4
            else:
                alpha = (d[0] - d[1]) / d[0]
                lamb = (d[0] + 3 * d[1]) / 4
        # elif d[1] < 0:  # Litman for Second Order Saddle point
        #    alpha = 1
        #    lamb = (d[1] + d[2]) / 4
        #    print 'WARNING: We are not using the standard Nichols'
        #    print 'd_x', d_x[0],d_x[1]

        else:  # Only d[0] <0
            alpha = 1
            lamb = (d[0] + d[1]) / 4

        if banded:
            h_up_band[-1, :] = h_up_band[-1, :] - np.ones(h_up_band.shape[1]) * lamb  # B - lamb * I. change diagonal part
            d_x = invmul_banded(h_up_band, f) * alpha             # FIXME: bug here. dx = invmul_banded(h_up_band, f) * alpha.
        else:
            h_test =  h_test - np.eye(h_test.shape[0]) * lamb   # FIXME: bug here, the right one should be : h_test = h_test - np.eye(h_test.shape[0]) * lamb.  d_x = np.linalg.solve(h_test,f) * alpha
            d_x = np.linalg.solve(h_test, f) * alpha 

        d_x.shape = self.fix.fixbeads.q.shape

        # MASS-scaled
        d_x = np.multiply(d_x, self.fix.fixbeads.m3**-0.5)

        # Rescale step if necessary
        if np.amax(np.absolute(d_x)) > activearrays["big_step"]:
            info(
                "Step norm, scaled down to {}".format(activearrays["big_step"]),
                verbosity.low,
            )
            d_x *= activearrays["big_step"] / np.amax(np.absolute(d_x))

        # Get the new full-position
        d_x_full = self.fix.get_full_vector(d_x, t=1)
        new_x = self.optarrays["old_x"].copy() + d_x_full

        self.post_step(step, new_x, d_x, activearrays)


class LBFGSOptimizer(DummyOptimizer):
    def bind(self, geop):
        # call bind function from DummyOptimizer
        super(LBFGSOptimizer, self).bind(geop)

        if geop.optarrays["hessian"].size == (self.beads.natoms * 3) ** 2:
            self.optarrays["initial_hessian"] = geop.optarrays["hessian"].copy()
            geop.optarrays["hessian"] = np.zeros(
                (self.beads.natoms * 3, self.beads.q.size)
            )

        if geop.options["hessian_final"]:
            self.options["hessian_asr"] = geop.options["hessian_asr"]
            if geop.optarrays["hessian"].size == 0:
                geop.optarrays["hessian"] = np.zeros(
                    (self.beads.natoms * 3, self.beads.q.size)
                )
            self.optarrays["hessian"] = geop.optarrays["hessian"]

        # self.sm.bind(self, self.options["discretization"])

        # Specific for LBFGS
        self.options["corrections"] = geop.options["corrections"]
        self.options["ls_options"] = geop.options["ls_options"]
        if geop.optarrays["qlist"].size != (
            self.options["corrections"] * self.beads.q.size
        ):
            if geop.optarrays["qlist"].size == 0:
                geop.optarrays["qlist"] = np.zeros(
                    (self.options["corrections"], self.beads.q.size), float
                )
            else:
                raise ValueError("qlist size does not match system size")
        if geop.optarrays["glist"].size != (
            self.options["corrections"] * self.beads.q.size
        ):
            if geop.optarrays["glist"].size == 0:
                geop.optarrays["glist"] = np.zeros(
                    (self.options["corrections"], self.beads.q.size), float
                )
            else:
                raise ValueError("qlist size does not match system size")

        self.optarrays["qlist"] = geop.optarrays["qlist"]
        self.optarrays["glist"] = geop.optarrays["glist"]

        if geop.options["scale"] not in [0, 1, 2]:
            raise ValueError("Scale option is not valid")

        self.options["scale"] = geop.options["scale"]

        if geop.optarrays["d"].size != self.beads.q.size:
            if geop.optarrays["d"].size == 0:
                geop.optarrays["d"] = np.zeros(
                    (self.beads.nbeads, 3 * self.beads.natoms), float
                )
            else:
                raise ValueError("Initial direction size does not match system size")

        self.optarrays["d"] = geop.optarrays["d"]

        self.mapper.esum = True

    def initialize(self, step):
        if step == 0:
            info(" @GEOP: Initializing instanton", verbosity.low)

            if self.beads.nbeads == 1:
                raise ValueError(
                    "We can not perform an splitting calculation with nbeads =1"
                )

            else:
                if ((self.beads.q - self.beads.q[0]) == 0).all():
                    # If the coordinates in all the imaginary time slices are the same
                    self.initial_geo()
                else:
                    info(
                        " @GEOP: Starting from the provided geometry in the extended phase space",
                        verbosity.low,
                    )

        # This must be done after the stretching and before the self.d.
        # Initialize all the mapper
        self.mapper.initialize(self.beads.q, self.forces)
        # if self.mapper.sm.f is None:
        #    self.mapper.sm(self.beads.q, ret=False)  # Init instanton mapper

        if (
            self.optarrays["old_x"]
            == np.zeros((self.beads.nbeads, 3 * self.beads.natoms), float)
        ).all():
            self.optarrays["old_x"][:] = self.beads.q

        # Specific for LBFGS
        if np.linalg.norm(self.optarrays["d"]) == 0.0:
            # f = self.forces.f + self.mapper.sm.f
            f = self.mapper.f
            self.optarrays["d"] += dstrip(f) / np.sqrt(np.dot(f.flatten(), f.flatten()))

        self.update_old_pos_force()
        self.init = True

    def post_step(self, step, activearrays):
        """General tasks that have to be performed after the  actual step"""

        # Update
        self.optarrays["qlist"][:] = self.fix.get_full_vector(
            activearrays["qlist"], t=3
        )
        self.optarrays["glist"][:] = self.fix.get_full_vector(
            activearrays["glist"], t=3
        )
        self.optarrays["d"][:] = self.fix.get_full_vector(activearrays["d"], t=1)

        self.update_pos_force()

        self.print_geo(step)

        # Check Exit and only then update old arrays
        d_x_max = np.amax(
            np.absolute(np.subtract(self.beads.q, self.optarrays["old_x"]))
        )
        self.exit = self.exitstep(d_x_max, step)
        self.update_old_pos_force()

    def step(self, step=None):
        """Does one simulation step."""

        activearrays = self.pre_step(step)

        e, g = self.mapper(self.beads.q)
        fdf0 = (e, g)

        # Do one step. Update the position and force inside the mapper.
        print(
            activearrays["big_step"],
            self.options["ls_options"]["tolerance"],
            self.options["tolerances"]["energy"],
            self.options["ls_options"]["iter"],
            self.options["corrections"],
            self.options["scale"],
            step,
        )

        L_BFGS(
            activearrays["old_x"],
            activearrays["d"],
            self.mapper,
            activearrays["qlist"],
            activearrays["glist"],
            fdf0,
            activearrays["big_step"],
            self.options["ls_options"]["tolerance"]
            * self.options["tolerances"]["energy"],
            self.options["ls_options"]["iter"],
            self.options["corrections"],
            self.options["scale"],
            step,
        )

        self.post_step(step, activearrays)
