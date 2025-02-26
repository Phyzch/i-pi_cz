"""
utility code for neb_instanton.py. 
Written by Chenghao Zhang & Niri Govind, Pacific Northwest National Laboratory (chenghao.zhang@pnnl.gov), 2024.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from ipi.engine.beads import Beads
from ipi.utils.messages import verbosity, info
from ipi.utils import units
import ipi.utils.mathtools as mt
import os
from ipi.utils.depend import dstrip
import ipi
from ipi.utils.softexit import softexit

def print_neb_instanton_geo(
    prefix, step, nbeads, natoms, names, q, pots, cell, shift, output_maker
):
    """
    adapted from instool.py: print_instanton_geo
    Alternative (but very useful) output of the instanton geometry and potential energy
    """

    outfile = output_maker.get_output(prefix + "_" + str(step) + ".ener", "w")
    print("#Bead    Energy (eV)", file=outfile)
    for i in range(nbeads):
        print(
            (
                str(i)
                + "     "
                + str(units.unit_to_user("energy", "electronvolt", pots[i] - shift))
            ),
            file=outfile,
        )
    outfile.close_stream()

    # print out the coordinate in .xyz form
    unit = "angstrom"

    a, b, c, alpha, beta, gamma = mt.h2abc_deg(cell.h)

    outfile = output_maker.get_output(prefix + "_" + str(step) + ".xyz", "w")

    for i in range(nbeads):
        print(natoms, file=outfile)

        print(
            (
                "CELL(abcABC):  %f %f %f %f %f %f cell{atomic_unit}  Traj: positions{%s}   Bead:       %i"
                % (a, b, c, alpha, beta, gamma, unit, i)
            ),
            file=outfile,
        )

        for j in range(natoms):
            print(
                names[j],
                str(units.unit_to_user("length", unit, q[i, 3 * j])),
                str(units.unit_to_user("length", unit, q[i, 3 * j + 1])),
                str(units.unit_to_user("length", unit, q[i, 3 * j + 2])),
                file=outfile,
            )

    outfile.close_stream()


def print_instanton_rp_time(prefix, image_time_period, rp_t_list, output_maker):
    """
    print info about imaginary time of ring polymer beads we print out.
    """
    nbead = len(rp_t_list)
    outfile = output_maker.get_output(prefix + ".txt", "w")
    print(
        "imaginary time for periodic motion: {}".format(image_time_period), file=outfile
    )
    print(" #Bead  time", file=outfile)
    for i in range(nbead):
        print(str(i) + "    " + str(rp_t_list[i]), file=outfile)

    outfile.close_stream()


def path_cubic_interpolation(neb_bead_q, interpolation_bead_number):
    """
    do cubic spline interpolation of minimum action path.
    return coordinate (x) and distance from initial point (r) along interpolated beads.
    :param: neb_bead_q:  coordinate of nudged elastic band bead
    :param: interpolation_bead_number: number of point to interpolate beads.

    :return: bead_path_x : the coordinate of interpolated points along minimum action path.
             bead_path_r: the cumulative distance from initial point along minimum action path.
    """
    neb_bead_q_array = np.array(neb_bead_q)
    neb_bead_number = len(neb_bead_q)

    neb_bead_distance = np.linalg.norm(neb_bead_q[1:] - neb_bead_q[:-1], axis=1)
    neb_bead_path_r = np.concatenate([[0], np.cumsum(neb_bead_distance)])
    # make the variable in the range of [0, 1]
    neb_bead_path_r_scaled = neb_bead_path_r / neb_bead_path_r[-1]

    cs = CubicSpline(
        neb_bead_path_r_scaled, neb_bead_q_array, axis=0, bc_type="natural"
    )  # object for cubic spline interpolation. interpolate along axis 0.

    cs1 = CubicSpline(np.arange(neb_bead_number), neb_bead_path_r_scaled)

    new_a = np.linspace(0, neb_bead_number - 1, num=interpolation_bead_number)

    new_bead_r_scaled = cs1(new_a)

    bead_path_x = cs(new_bead_r_scaled)

    bead_distance = np.linalg.norm(bead_path_x[1:] - bead_path_x[:-1], axis=1)

    bead_path_r = np.concatenate(
        [[0], np.cumsum(bead_distance)]
    )  # distance from initial beads.

    return bead_path_x, bead_path_r


def path_cubic_spline_function(neb_bead_q):
    """
    return cubic spline function of minimum action path using the location of neb beads.
    The spline function x = Cs(r), will r is the normalized distance along the path. (at the end of path, r=1).

    :param: neb_bead_q: coordinate of nudged elastic band bead

    :return cs: CubicSpline function : scipy.interpolate.CubicSpline
    """
    neb_bead_q_array = np.array(neb_bead_q)
    neb_bead_distance = np.linalg.norm(neb_bead_q[1:] - neb_bead_q[:-1], axis=1)
    neb_bead_path_r = np.concatenate([[0], np.cumsum(neb_bead_distance)])
    # make the variable in the range of [0, 1]
    neb_bead_path_r_scaled = neb_bead_path_r / neb_bead_path_r[-1]

    cs = CubicSpline(
        neb_bead_path_r_scaled, neb_bead_q_array, axis=0, bc_type="natural"
    )

    return cs


def path_equal_distance_interpolation(neb_bead_q, interpolation_bead_number):
    """
    interpolate the path to find points spaced with equal distance along the path.
    :param: neb_bead_q:  coordinate of nudged elastic band bead.
    :param: interpolation_bead_number: number of point to interpolate beads.
    """
    neb_bead_q_array = np.array(neb_bead_q)
    neb_bead_number = len(neb_bead_q)
    neb_bead_distance = np.linalg.norm(neb_bead_q[1:] - neb_bead_q[:-1], axis=1)
    neb_bead_path_r = np.concatenate([[0], np.cumsum(neb_bead_distance)])
    # make the variable in the range of [0, neb_bead_number]
    neb_bead_path_r_scaled = neb_bead_path_r / neb_bead_path_r[-1] * neb_bead_number

    cs = CubicSpline(neb_bead_path_r_scaled, neb_bead_q_array, axis=0)

    interpolate_r_scaled = np.linspace(
        0, neb_bead_number, num=interpolation_bead_number
    )

    bead_path_q = cs(interpolate_r_scaled)

    bead_distance = np.linalg.norm(bead_path_q[1:] - bead_path_q[:-1], axis=1)

    bead_path_r = np.concatenate(
        [[0], np.cumsum(bead_distance)]
    )  # distance from initial beads.

    return bead_path_q, bead_path_r


def interpolate_ring_polymer_beads(
    period, t_list, x_list, v_list, instanton_bead_number
):
    """
    interpolate the position of ring polymer beads along minmium action path.
    x[bead_i] = x(t = period / (2N) * bead_i) with total N beads. We only record half ring-polymer as it folds back to itself.

    :param:  period: imaginary time evolution period (= beta hbar)
             t_list: list of time for trajectory recorded.
             v_list: list of velocity for the trajectory recorded.
             x_list: list of coordinate for trajectory recorded
             instanton_bead_number: bead number for ring-polymers along instanton path to interpolate.

    :return: rp_t_list: time list for instanton ring-polymer
             rp_x_list: coordinate list for instanton ring-polymer
    """
    rp_t_list = np.linspace(
        0, period / 2, instanton_bead_number, endpoint=False
    )  # i * beta * hhbar / (2N) : here i = 0, ..., N-1.
    rp_t_list = rp_t_list + period / (
        4 * instanton_bead_number
    )  # (1/2 + i) * (beta * hbar / (2N)) : here i = 0, .., N-1

    print("imaginary time list for ring polymer: " + str(rp_t_list))

    t_list_len = len(t_list)

    rp_x_list = []

    # interpolate the internal beads
    t_index_start = 0
    for i in range(instanton_bead_number):
        rp_t = rp_t_list[i]

        for t_index in range(t_index_start, t_list_len - 1):
            if rp_t >= t_list[t_index] and rp_t < t_list[t_index + 1]:
                # interpolate using velocity and acceleration.
                dt = t_list[t_index + 1] - t_list[t_index]
                rp_dt = rp_t - t_list[t_index]

                dx = x_list[t_index + 1] - x_list[t_index]
                rp_dx = v_list[t_index] * rp_dt + np.power(rp_dt / dt, 2) * (
                    dx - v_list[t_index] * dt
                )  # velocity and acceleration contribution.

                rp_x = x_list[t_index] + rp_dx
                rp_x_list.append(rp_x)

                t_index_start = t_index
                break

    rp_x_list = np.array(rp_x_list)

    return rp_t_list, rp_x_list


def RK4(y, t, dydt, param, h):
    """
    Evolve system one step further using 4th order Runge_Kutta method.
    :param t: time
    :param y: variable
    :param dydt : first order derivative function. dydt (y,t , param)
    :param param: parameter for dydt function
    :param h: time step
    :return:
    """
    k1 = h * dydt(y, t, param)
    k2 = h * dydt(y + 0.5 * k1, t + 0.5 * h, param)
    k3 = h * dydt(y + 0.5 * k2, t + 0.5 * h, param)
    k4 = h * dydt(y + k3, t + h, param)

    new_y = y + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return new_y


def compute_r_acceleration_along_path(
    negative_f: np.ndarray,
    jacobian: np.ndarray,
    jacobian_rate: np.ndarray,
    m3_matrix: np.ndarray,
    v_r: np.ndarray,
):
    """
    :param: negative_f: negative force.
    :param: jacobian: dx/dr
    :param: jacobian_rate: d(dx/dr)/dt
    :param: m3_matrix: diagonal matrix. diagonal element is m3.
    :param: v_r: dr/dt. velocity for r.
    """
    jacobian_transpose = np.transpose(jacobian)
    term1 = np.dot(jacobian, negative_f)
    term2 = np.matmul(np.matmul(jacobian_transpose, m3_matrix), jacobian_rate) * v_r

    denominator = np.matmul(np.matmul(jacobian_transpose, m3_matrix), jacobian)

    a_r = (term1 - term2) / denominator

    return a_r


def dydt_inverted_pot(y, t, param):
    """
    y = [r, v_r].
    r is normalized distance along the path.
    y[0] = r,  y[1] = v_r
    dydt[0] = v_r,  dydt[1] = a_r  (acceleration of r on inverted potential)
    param = [cl_bead, cl_forces, m3_matrix, cubic_spline]
    here cl_bead is the bead for classical dynamics.
         cl_forces is force engine for classical dynamics.
         set cl_bead.q[0] = x. Then we can call force engine to get potential and force.

         m3_matrix: mass. 2d diagonal matrix. size: [3 * natoms, 3* natoms].
                    The diagonal element is m3.
        cubic_spline: cubic spline function that return coordinate x(r).

    acceleration d^2 r/dt^2 is from constrained dynamics.
    See eq.(13) in Witkin, A. (1997). Computer graphics, 9, 27
    """
    r_distance = y[0]
    v_r = y[1]

    cl_bead = param[0]
    cl_forces = param[1]
    m3_matrix = param[2]
    cubic_spline = param[3]

    x = cubic_spline(
        r_distance, nu=0
    )  # coordinate of the system from cubic spline (vector)
    dx_dr = cubic_spline(r_distance, nu=1)  # jacobian dx/dr (vector)
    dx_dr_second_deriv = cubic_spline(
        r_distance, nu=2
    )  # second derivative d^2 x/ dr^2 (vector)

    dx_dr_rate = (
        dx_dr_second_deriv * v_r
    )  # d(dx/dr)/dt: rate of change for the jacobian (vector)

    # compute the negative force on the up-side down potential.
    cl_bead.q[0] = x
    forces = cl_forces.f[0]
    negative_f = -forces

    # compute acceleration of r.
    a_r = compute_r_acceleration_along_path(
        negative_f, dx_dr, dx_dr_rate, m3_matrix, v_r
    )

    dydt = np.array([v_r, a_r])

    return dydt


def print_instanton_hess(prefix, hessian, output_maker):
    """Print physical part of the instanton hessian"""
    hessian_file_name = prefix + ".hess"
    if os.path.exists(hessian_file_name):
        os.rename(hessian_file_name, hessian_file_name + "#")
    with open(hessian_file_name, "w") as f:
        hessian_1d = hessian.flatten()
        hessian_size = hessian.size
        for i in range(hessian_size):
            f.write(str(hessian_1d[i]) + " ")

        f.write("\n")


def get_hessian(rp_beads, rp_forces, x0, natoms, nbeads=1,  fixdofs = [], d=0.001):
    """
    Adapted from hesstool.py
    Compute hessian as finite difference of force.
    The intermediate steps are written as a temporary files so the full hessian calculations is only ONE step.
    The hessian for fixed dofs will be set as 0 and we will skip its calculation.
    IN     rp_beads: bead object for ring polymer
           rp_forces: forces object for ring polymer
           x0       = position vector
           natoms   = number of atoms
           nbeads   = number of beads
           fix_dofs = indexes of fixed dofs
           d        = displacement

    OUT    h       = physical hessian ( (natoms-len(fixatoms) )*3 , nbeads*( natoms-len(fixatoms) )*3)
    """
    info(" @get_hessian: Computing hessian", verbosity.low)
    ii = natoms * 3
    activedof = np.delete(np.arange(ii), fixdofs)
    ncalc = ii - len(fixdofs)  # for each bead, # of free dofs need calculation.
    if x0.size != natoms * 3 * nbeads:
        raise ValueError(
            "The position vector is not consistent with the number of atoms/beads."
        )

    h = np.zeros((ii, ii * nbeads), float)

    # Check if there is a temporary file:
    i0 = -1

    for i in range(ii, -1, -1):
        try:
            b = np.loadtxt("hessian_" + str(i) + ".tmp")
        except IOError:
            pass
        else:
            h[:, :] = b[:, :]
            i0 = i
            print(("We have found a temporary file ( hessian_" + str(i) + ".tmp). "))
            if (
                b.shape == h.shape
            ):  # Check that the last temporary file was properly written
                break
            else:
                continue

    # Start calculation:
    # deep copy data in case rp_beads.q == x0.
    if isinstance(x0, ipi.utils.depend.depend_array):
        x0_copy = np.copy(dstrip(x0))
    elif isinstance(x0, np.ndarray):
        x0_copy = np.copy(x0)

    for j in range(i0 + 1, ii):
        if j in fixdofs:
            continue
        else:
            ndone = len(activedof[activedof < j])
            info(
                " @get_hessian: Computing hessian: %d of %d" % (ndone + 1, ncalc),
                verbosity.low,
            )
            x = x0_copy.copy()

            # PLUS
            x[:, j] = x0_copy[:, j] + d
            rp_beads.q[:] = x  # update bead location.
            g1 = -rp_forces.f  # gradient = - force.

            # Minus
            x[:, j] = x0_copy[:, j] - d
            rp_beads.q[:] = x
            g2 = -rp_forces.f  # gradient = - force.

            # COMBINE
            g = (g1 - g2) / (2 * d)
            # Set hessian component along fixed dofs as 0.
            g[:, fixdofs] = 0
            
            h[j, :] = g.flatten()

            # save hessian temporary file (record hessian up to row j.)
            file = open("hessian_" + str(j) + ".tmp", "w")
            np.savetxt(file, h)
            file.close()

    # remove hessian temporary file
    for i in range(ii):
        try:
            os.remove("hessian_" + str(i) + ".tmp")
        except OSError:
            pass

    return h

def projected_verlet(x0, v0, fdf0, fdf,  dt):
    """
    projected velocity verlet algorithm.
    velocity verlet & project the velocity along force direction. 
    This is one type of steepest descent algorithm.

    :param: x0: initial coordinate
    :param: v0: initial velocity 
    :param: fdf0: (func, gradient): initial function and gradient values
    :param: fdf: gradient mapper. func, grad = fdf(x_mscaled)
    :param: dt: time step
    """
    _, g0 = fdf0  # g0: initial gradient.
    negative_g0 = -g0
    # update new position 
    dx = dt * v0 + 0.5 * negative_g0 * np.power(
        dt, 2
    )
    x = x0 + dx 

    func , g = fdf(
        x
    )
    negative_g = -g

    v = v0 + dt * (negative_g0 + negative_g) / 2 

    # project the velocity along the direction of the current force.
    negative_g_unit_vector = negative_g / np.linalg.norm(g)

    v_g_inner_product = np.inner(
        negative_g_unit_vector.flatten(), v.flatten()
    )

    if v_g_inner_product < 0:
        v = np.zeros(v.shape)
    else:
        v = v_g_inner_product * negative_g_unit_vector 

    return x, v, func, g 

def conjugate_gradient(x0, fdf0, fdf, initial_search_direction, big_step, 
                       line_search_cutoff = 0.1,
                       ):
    """
    Use conjugate gradient method to find local minimum for neb algorithm.
    The function to optimize of neb method is ill-defined, because we perform the projection of gradient.
    Therefore, we use the simpliest criterion for line search: 
    the gradient at new location is smaller than the gradient at old location.

    We use the Polak Ribere version of conjugate gradient method to update search direction.
    We restart the search direction as gradient direction when |df * df0|/|df0|^2 > 0.1. 
    This algorithm perform one step of CG method. Do line search and then update search direction. 

    :param: x0: initial coordinate.
    :param: fdf0: (func, gradient)
    :param: fdf: mapper function.  func, gradient = fdf(x)
    :param: big_step: biggest step for cg method. Used to perform backtracking.
    :param: backtrack_ratio: ratio to scale the step for line search backtracking.
    :param: restart_check: value to restart the cg search direction as negative gradient direction.
    """
    action0, g0 = fdf0
    g0_norm = np.linalg.norm(g0)
    
    # notation in Nocedal & Wright
    p0 = initial_search_direction 
    if np.inner(p0.flatten(), g0.flatten()) > 0:
        # the conjugate search direction is no longer the descent direction, refresh the p0 as -g0.
        p0 = -g0
    
    p0_norm = np.linalg.norm(p0)
    g0_component = np.inner(g0.flatten(), p0.flatten()) / p0_norm

    search_step = big_step 

    # implement a simple Newton's step 
    # a finite difference is used to estimate g'(x) (hessian along search direction)
    dx = 0.01 
    x_fd = x0 + dx * p0 
    _, g_fd = fdf(x_fd)
    g_fd_component = np.inner(g_fd.flatten(), p0.flatten()) / p0_norm 
    dg0 = (g_fd_component - g0_component) / dx 
    
    # newton's step x = x0 + g(x0) / g'(x0)
    x = x0 - p0 * g0_component / dg0 

    step_size = np.linalg.norm(x - x0) / np.linalg.norm(p0)
    action, g = fdf(x)
    g_component = np.inner(g.flatten(), p0.flatten())/ p0_norm
    # update search direction
    # check if we need to refresh the search direction as negative gradient.
    g0_flatten = g0.flatten()
    g_flatten = g.flatten() 

    # Polak Ribere version of conjugate gradient
    beta = np.inner(
        g_flatten, ( g_flatten - g0_flatten ) 
        )/ np.inner(
            g0_flatten, g0_flatten
        )
    
    # update search direction.
    p = beta * p0 - g

    search_direction = p 

    return x, action, g, search_direction


def apply_symmetry_projection_single_bead(m, q, natoms, vec, asr = "none", mscaled_bool= False):
    """
    Removes translation (and rotations) for one bead depending on the asr mode.
    The projection is performed in the mass scaled coordinate: sqrt(m) * q.
    :param: m: mass of atoms. shape: [natoms]
    :param: q: coordinate q for a single bead. shape: [3 * natoms] 
    :param: natoms: number of atoms in molecules.
    :param: asr: mode of symmetry. options["None", "crystal", "poly"]
    :param: vec: vectors to project out translation & rotation motions. shape: [natoms] 
    :param: mscaled_bool: indicate whether the vec is already scaled by mass or not. 
    Adapted from ipi/engine/motion/phonons.py apply_asr.
    
    :return: projected_vec: vectors after we project out translation & rotational motions.
    """
    assert np.shape(q)[0] == 3 * natoms, "The natoms doesn't match beads_q shape."
    assert m.shape[0] == natoms
    ism = 1 / np.sqrt(np.repeat(m, 3))

    if asr == "none":
        return vec 
    elif asr == "crystal":
        # project out translation dofs.
        D = np.zeros((3, 3 * natoms), float)
        D[0] = np.tile([1, 0, 0], natoms) / ism
        D[1] = np.tile([0, 1, 0], natoms) / ism 
        D[2] = np.tile([0, 0, 1], natoms) / ism 
        for k in range(3):
            D[k] = D[k] / np.linalg.norm(D[k])

        if not mscaled_bool:
            # convert vector to mass scaled coordinate
            mscaled_vec = vec / ism 
            transfmatrix = np.eye(3 * natoms) - np.dot(D.T, D)
            projected_mscaled_vec = transfmatrix @ mscaled_vec 
            projected_vec = projected_mscaled_vec * ism 
        else:
            # the vector is already scaled by mass.
            transfmatrix = np.eye(3 * natoms) - np.dot(D.T, D)
            projected_vec = transfmatrix @ vec 

        return projected_vec 

    elif asr == "poly":
        # project out translation and rotation dofs.
        # compute center of mass
        com = (np.dot(
            np.transpose(q.reshape((natoms, 3)), (1,0)), m
            ) / np.sum(m)
            )  
        qminuscom = q.reshape((natoms, 3)) - com[np.newaxis, :]
        # Computes the momentum of inertia 
        moi = np.zeros((3, 3), float)
        for k in range(natoms):
            moi = moi - (
                np.dot(
                    np.cross(qminuscom[k], np.identity(3)),
                    np.cross(qminuscom[k], np.identity(3))
                ) 
                * m[k]
            )

        U = (np.linalg.eig(moi))[1] # rotation axis.
        R = np.dot(qminuscom, U) 
        # translation & rotation motion to project out.
        D = np.zeros((6, 3 * natoms), float)
        D[0] = np.tile([1, 0, 0], natoms) / ism
        D[1] = np.tile([0, 1, 0], natoms) / ism 
        D[2] = np.tile([0, 0, 1], natoms) / ism 
        for i in range(3 * natoms):
            iatom = i // 3
            idof = np.mod(i, 3)
            D[3, i] = (
                R[iatom, 1] * U[idof, 2] - R[iatom, 2] * U[idof, 1]
            ) / ism[i]
            D[4, i] = (
                R[iatom, 2] * U[idof, 0] - R[iatom, 0] * U[idof, 2]
            ) / ism[i]
            D[5, i] = (
                R[iatom, 0] * U[idof, 1] - R[iatom, 1] * U[idof, 0]
            ) / ism[i]
        
        # Compute unit vectors
        for k in range(6):
            D[k] = D[k] / np.linalg.norm(D[k])
        

        if not mscaled_bool:
            # convert vector to mass scaled coordinate
            mscaled_vec = vec / ism 
            transfmatrix = np.eye(3 * natoms) - np.dot(D.T, D)
            projected_mscaled_vec = transfmatrix @ mscaled_vec 
            projected_vec = projected_mscaled_vec * ism 
        else:
            transfmatrix = np.eye(3 * natoms) - np.dot(D.T, D)
            projected_vec = transfmatrix @ vec 

        return projected_vec 

    else:
        raise ValueError(f"unsupported asr mode. we only take 'none', 'crystal' and 'asr'. Current value: {asr}")

def apply_symmetry_projection(m, beads_q, natoms, vec, asr= "none", mscaled_bool= False):
    """
    Removes translation (and rotations) for beads depending on the asr mode.
    The projection is performed in the mass scaled coordinate: sqrt(m) * q.
    :param: m: mass of atoms. shape: [natoms]
    :param: beads_q: coordinate q for all bead. shape: [nbeads, 3 * natoms] 
    :param: natoms: number of atoms in molecules.
    :param: asr: mode of symmetry. options["None", "crystal", "poly"]
    :param: vec: vectors to project out translation & rotation motions. shape: [nbeads, 3 * natom]
    Adapted from ipi/engine/motion/phonons.py apply_asr.

    Calls apply_symmetry_projection_single_bead function to project out trans & rotation dofs for a single bead.
    """
    nbeads = np.shape(beads_q)[0]
    projected_vecs = []
    for bead_index in range(nbeads):
        projected_vec = apply_symmetry_projection_single_bead(m, 
                                                              beads_q[bead_index], 
                                                              natoms, 
                                                              vec[bead_index], 
                                                              asr= asr, 
                                                              mscaled_bool= mscaled_bool)
        projected_vecs.append(projected_vec)
    
    projected_vecs = np.array(projected_vecs)
    
    return projected_vecs 

class Essentially_Nonoscillatory_Polynomial(object):
    """
    construct essentially non-oscillatory polynomial (ENO) to represent the path.
    The tangent direction of the path can be given by the first derivative of the polynomial using the upwind scheme.
    See: https://dx.doi.org/10.4310/CMS.2003.v1.n2.a10
    """
    def __init__(self, beads_q, beads_energy, order):
        """
        :param: beads_q: [nbeads, 3 * natom]. beads coordinate that represent the path.
        :param: beads_energy: the energy of beads. Used for upwind scheme to construct tangent vector.
        :param: order: the order of the ENO polynomial we will construct. 
        """
        self.beads_q = np.copy(beads_q)
        self.beads_energy = np.copy(beads_energy)
        self.order = order 
        self.beads_number = np.shape(self.beads_q)[0]

    def compute_parametrization(self):
        """
        compute the parameter alpha for the path. 
        The path is parametrized by the normalized arc length alpha. so f(0) = q[0], f(1) = q[-1].  
        parameter alpha in range [0, 1].

        """
        q_distance = np.linalg.norm(self.beads_q[1:] - self.beads_q[:-1], axis= 1)
        distance_sum = np.sum(q_distance)
        normalized_q_distance = q_distance / distance_sum 
        alpha = np.cumsum(normalized_q_distance)
        alpha = np.concatenate([[0], alpha])
        # parameter alpha: distance along the path. This parametrize the path.
        self.alpha = alpha 
    
    def compute_Newton_divided_difference(self):
        """
        compute Newton's divided difference to the order given by self.order.
        """
        beads_q_shape = np.shape(self.beads_q)
        beads_number = self.beads_number
        alpha = self.alpha 
        difference_matrix = np.zeros(np.concatenate([self.order + 1, 
                                                      beads_q_shape])
                                                      )
        difference_matrix[0] = np.copy(self.beads_q)
        for l in range(1, self.order + 1):
            for i in range(0, beads_number - l):
                difference_matrix[l, i] = (difference_matrix[l -1, i + 1] - difference_matrix[l -1, i]) / (alpha[i + l] - alpha[i])
        
        self.difference_matrix = difference_matrix
    
    def compute_ENO_polynomial(self):
        """
        compute essentially non-oscillatory polynomial to the order : self.order. 
        We use numpy.Polynomial to construct the approximate polynomial. 
        See Algorithm 3.1 in https://dx.doi.org/10.4310/CMS.2003.v1.n2.a10
        """
        beads_number = self.beads_number 
        ndofs = np.shape(self.beads_q)[1]
        # The order of polynomial we will construct.
        order = self.order 
        
        # shape: [beads_number -1, ndofs]
        polynomial_ENO_list = []

        for i in range(beads_number - 1):
            # polynomial_ENO[i] is associated with interval [alpha_{i}, alpha_{i+1}].
            # A list of polynomials along each dofs.
            polynomial_ENO = []
            for n in range(ndofs):
                polynomial_ENO_term = np.polynomial.Polynomial([self.beads_q[i, n]])
                polynomial_ENO.append(polynomial_ENO_term)

            # (alpha - alpha[i]) * f[alpha[i], alpha[i+1]]
            for n in range(ndofs):
                first_order_term = np.polynomial.Polynomial([ -self.alpha[i], 1]) * self.difference_matrix[1, i, n] 
                polynomial_ENO[n] = polynomial_ENO[n] + first_order_term 

            k_min_list = np.zeros((order + 1))
            k_min_list[1] = i 
            for l in range(2, order + 1):
                a = self.difference_matrix[l, k_min_list[l-1]]
                if k_min_list[l-1] == 0:
                    c = a 
                    k_min_list[l] = k_min_list[l-1]
                else:
                    b = self.difference_matrix[l, k_min_list[l-1] - 1]
                    if np.linalg.norm(a) > np.linalg.norm(b):
                        c = b 
                        k_min_list[l] = k_min_list[l-1] - 1 
                    else:
                        c = a 
                        k_min_list[l] = k_min_list[l-1]
                
                # c * \prod_{m - k_min[l-1]}^{k_min[l-1] + l -1} (alpha - alpha[m])
                for n in range(ndofs):
                    order_l_term = np.polynomial.Polynomial([c[n]])
                    for m in range(k_min_list[l-1], k_min_list[l-1] + l):
                        order_l_term = order_l_term * np.polynomial.Polynomial([- self.alpha[m], 1])
                
                    polynomial_ENO[n] = polynomial_ENO[n] + order_l_term 
                
            polynomial_ENO_list.append(polynomial_ENO)
        
        self.polynomial_ENO_list = polynomial_ENO_list
    
    def compute_tangent_vector_subroutine(self):
        """
        compute tangent vector as the first derivative of polynomial
        """
        beads_number = self.beads_number 
        ndofs = np.shape(self.beads_q)[1]
        beads_energy = self.beads_energy 

        btau = np.zeros((beads_number, ndofs), float)
        for ii in range(1, beads_number -1):
            # tau minus 
            d1 = np.zeros([ndofs])
            for n in range(ndofs):
                # tangent direction is defined as the derivative of the polynomial.
                deriv = self.polynomial_ENO_list[ii -1][n].deriv()
                d1[n] = deriv(self.alpha[ii])
            
            # tau plus 
            d2 = np.zeros([ndofs])
            for n in range(ndofs):
                # tangent direction is defined as the derivative of the polynomial.
                deriv = self.polynomial_ENO_list[ii][n].deriv()
                d2[n] = deriv(self.alpha[ii])

            # Improved tangent estimate:
            # Energy of images: E(ii + 1) < E(ii) < E(ii - 1)
            if beads_energy[ii + 1] < beads_energy[ii] < beads_energy[ii - 1]:
                btau[ii] = d1 
            elif beads_energy[ii - 1] <= beads_energy[ii] <= beads_energy[ii + 1]:
                btau[ii] = d2 
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
            btau[ii] = btau[ii] / np.linalg.norm(btau[ii])
        
        return btau 

    def compute_tangent_vector(self):
        """
        compute tangent vector using first derivative of the ENO polynomial. 
        call compute_parameterization(), compute_Newton_divided_difference(),
             compute_ENO_polynomial(),  compute_tangent_vector_subroutine() 
        """
        # compute parameter alpha that parameterize the path.
        self.compute_parametrization()
        
        self.compute_Newton_divided_difference()
        
        # compute Essentially non-oscillatory polynomial using Newton divided difference 
        self.compute_ENO_polynomial()
        
        # compute the tangent vector as first order derivative of ENO polynomial. 
        btau = self.compute_tangent_vector_subroutine()

        return btau


        