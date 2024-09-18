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
            if rp_t > t_list[t_index] and rp_t < t_list[t_index + 1]:
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


def get_hessian(rp_beads, rp_forces, x0, natoms, nbeads=1, fixatoms=[], d=0.001):
    """
    Adapted from hesstool.py
    Compute hessian as finite difference of force.
    The intermediate steps are written as a temporary files so the full hessian calculations is only ONE step.

    IN     rp_beads: bead object for ring polymer
           rp_forces: forces object for ring polymer
           x0       = position vector
           natoms   = number of atoms
           nbeads   = number of beads
           fixatoms = indexes of fixed atoms
           d        = displacement

    OUT    h       = physical hessian ( (natoms-len(fixatoms) )*3 , nbeads*( natoms-len(fixatoms) )*3)
    """

    info(" @get_hessian: Computing hessian", verbosity.low)
    fixdofs = list()
    for i in fixatoms:
        fixdofs.extend(
            [3 * i, 3 * i + 1, 3 * i + 2]
        )  # add all fixdofs attached to fix atoms.
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
    TODO: This CG method doesn't converge for neb code.

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

    # bisect search using gradient along search direction.
    x_end = x0 + p0 * search_step 
    action_end , g_end = fdf(x_end)
    g_end_component = np.inner(g_end.flatten(), p0.flatten())/ p0_norm
    while g_end_component < 0:
        # increase the searach step until we can make sure that there is one minimum between x_end and x0.
        search_step = search_step * 2
        x_end = x0 + p0 * search_step 
        action_end , g_end = fdf(x_end)
        g_end_component = np.inner(g_end.flatten(), p0.flatten())/ p0_norm

    x_low = x0    # the point with negative gradient along p0.
    x_high = x_end  # the point with positive gradient along p0.
    g_low = g0 
    g_high = g_end

    while(1):
        g_low_component = np.inner(g_low.flatten(), p0.flatten()) / p0_norm
        g_high_component = np.inner(g_high.flatten(), p0.flatten()) / p0_norm
        
        if abs(g_low_component) < line_search_cutoff * abs(g0_component):
            x = x_low 
            break 
        
        if abs(g_high_component) < line_search_cutoff * abs(g0_component):
            x = x_high 
            break

        # bisect 
        x_middle = (x_low + x_high) / 2
        action_middle , g_middle = fdf(x_middle)
        g_middle_component = np.inner(g_middle.flatten(), p0.flatten())/ p0_norm 

        if g_middle_component < 0:
            x_low = x_middle 
            g_low = g_middle 
        else:
            x_high = x_middle
            g_high = g_middle


    step_size = np.linalg.norm(x - x0) / np.linalg.norm(p0)
    action, g = fdf(x)
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

    # restart the search direction if it degrades.
    check = np.abs(np.inner(g_flatten, g0_flatten))/ np.power(np.linalg.norm(g0) , 2)
    if check > 0.1:
        p = -g

    search_direction = p 

    return x, action, g, search_direction
    
    
    