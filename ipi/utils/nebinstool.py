import numpy as np
from scipy.interpolate import CubicSpline
from ipi.engine.beads import Beads
from ipi.utils.messages import verbosity, info
from ipi.utils import units
import ipi.utils.mathtools as mt
import os 
from ipi.utils.depend import dstrip


def print_neb_instanton_geo(
    prefix, step, nbeads, natoms, names, q, pots, cell, shift, output_maker
):
    """
    adapted from instool.py: print_instanton_geo
    Alternative (but very useful) output of the instanton geometry and potential energy"""

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
    '''
    print info about imaginary time of ring polymer beads we print out.
    '''
    nbead = len(rp_t_list)
    outfile = output_maker.get_output(prefix + ".txt", "w")
    print("imaginary time for periodic motion: {}".format(image_time_period) , file = outfile)
    print(" #Bead  time", file = outfile)
    for i in range(nbead):
        print(
            str(i) + "    "
            + str(rp_t_list[i]), 
            file = outfile
        )
    
    outfile.close_stream()



def path_cubic_interpolation(neb_bead_q, interpolation_bead_number):
    '''
    do cubic interpolation of minimum action path. 
    return coordinate (x) and distance from initial point (r) along interpolated beads.
    :param: neb_bead_q:  coordinate of nudged elastic band bead
    :param: interpolation_bead_number: number of point to interpolate beads.
    
    :return: bead_path_x : the coordinate of interpolated points along minimum action path.
             bead_path_r: the cumulative distance from initial point along minimum action path.
    '''
    neb_bead_q_array = np.array(neb_bead_q)
    neb_bead_number = len(neb_bead_q)

    a = np.arange(neb_bead_number)
    b = neb_bead_q_array

    cs = CubicSpline(a, b, axis = 0)  # object for cubic spline interpolation. interpolate along axis 0.

    new_a = np.linspace(0, a[-1], num = interpolation_bead_number)  

    bead_path_x = cs(new_a)

    bead_distance = np.linalg.norm(bead_path_x[1:] - bead_path_x[:-1], axis = 1)

    bead_path_r = np.concatenate([[0], np.cumsum(bead_distance)]) # distance from initial beads.

    return bead_path_x, bead_path_r 

def interpolate_ring_polymer_beads(period, t_list, x_list, v_list, instanton_bead_number):
    '''
    interpolate the position of ring polymer beads along minmium action path. 
    x[bead_i] = x(t = period / (2N) * bead_i) with total N beads. We only record half ring-polymer as it folds back to itself.
    
    :param:  period: imaginary time evolution period (= beta hbar)
             t_list: list of time for trajectory recorded.
             v_list: list of velocity for the trajectory recorded.
             x_list: list of coordinate for trajectory recorded
             instanton_bead_number: bead number for ring-polymers along instanton path to interpolate.
    
    :return: rp_t_list: time list for instanton ring-polymer
             rp_x_list: coordinate list for instanton ring-polymer
    '''
    rp_t_list = np.linspace(0, period / 2, instanton_bead_number, endpoint = False) # i * beta * hhbar / (2N) : here i = 0, ..., N-1.
    rp_t_list = rp_t_list + period / (4 * instanton_bead_number)  # (1/2 + i) * (beta * hbar / (2N)) : here i = 0, .., N-1

    print("time list for ring polymer: " + str(rp_t_list))
    print("time list for evolution: " + str(t_list))
        
    t_list_len = len(t_list)

    rp_x_list = []

    # interpolate the internal beads
    t_index_start = 0
    for i in range(instanton_bead_number):
        rp_t = rp_t_list[i]

        for t_index in range(t_index_start, t_list_len):
            if rp_t > t_list[t_index] and rp_t < t_list[t_index + 1]:
                # interpolate using velocity and acceleration.
                dt = t_list[t_index + 1] - t_list[t_index]
                rp_dt = rp_t - t_list[t_index]
                
                dx = x_list[t_index + 1] - x_list[t_index]
                rp_dx = v_list[t_index] * rp_dt + np.power(rp_dt / dt , 2) * (dx - v_list[t_index] * dt)  # velocity and acceleration contribution.
                
                rp_x = x_list[t_index] + rp_dx 
                rp_x_list.append(rp_x)

                t_index_start = t_index
                break 

        

    rp_x_list = np.array(rp_x_list)
    
    return rp_t_list, rp_x_list 

def RK4(y , t , dydt , param , h):
    '''
    Evolve system one step further using 4th order Runge_Kutta method.
    :param t: time
    :param y: variable
    :param dydt : first order derivative function. dydt (y,t , param)
    :param param: parameter for dydt function
    :param h: time step
    :return:
    '''
    k1 = h * dydt( y , t, param )
    k2 = h * dydt( y + 0.5 * k1 , t + 0.5 * h , param)
    k3 = h * dydt( y + 0.5 * k2 , t + 0.5 * h , param)
    k4 = h * dydt( y + k3 , t + h , param)

    y = y + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4 )

    return y

def dydt_inverted_pot(y, t, param):
    '''
    y=[x,v]. That is y[0] = x. y[1] = v.
    dydt[0] = v. dydt[1] = a (inverted pot)
    param = [cl_beads, cl_forces, m3, tau]
    cl_beads: bead object that record the coordinate of current particle
    cl_forces: force object that connect to force engine to compute force (Depending on bead object's location)
    m3 : mass. size : [3 * natom]
    tau: tangent direction of motion. unit vector.
    '''
    x = y[0]
    v = y[1]

    cl_beads = param[0]
    cl_forces = param[1]
    m3 = param[2]
    tau = param[3]

    # update coordinate of bead object to enable the forces object to compute force
    if (cl_beads.q[0] != x).any() :
        cl_beads.q[0] = np.copy(x)

    a = -dstrip(cl_forces.f).copy()[0] / m3  # negative force (-f), force in inverted potential.
    a = np.dot(a, tau) * tau 

    dydt = np.array([ v, a ])

    return dydt 


def get_hessian(
    rp_beads, rp_forces, x0, natoms, nbeads=1, fixatoms=[], d=0.001
):
    """
    Adopted from hesstool.py
    Compute the physical hessian given a function to evaluate energy and forces (gm).
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
        fixdofs.extend([3 * i, 3 * i + 1, 3 * i + 2])  # add all fixdofs attached to fix atoms.
    ii = natoms * 3
    activedof = np.delete(np.arange(ii), fixdofs)
    ncalc = ii - len(fixdofs)  #for each bead, # of free dofs need calculation.
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
    for j in range(i0 + 1, ii):
        if j in fixdofs:
            continue
        else:
            ndone = len(activedof[activedof < j])
            info(
                " @get_hessian: Computing hessian: %d of %d" % (ndone + 1, ncalc),
                verbosity.low,
            )
            x = x0.copy()

            # PLUS
            x[:, j] = x0[:, j] + d
            rp_beads.q = x  # update bead location.
            g1 = -rp_forces.f  # gradient = - force.

            # Minus
            x[:, j] = x0[:, j] - d
            rp_beads.q = x 
            g2 = -rp_forces.f # gradient = - force.

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
