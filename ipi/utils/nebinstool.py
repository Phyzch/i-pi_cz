import numpy as np
from scipy.interpolate import CubicSpline
from ipi.engine.beads import Beads
from ipi.utils.messages import verbosity, info
from ipi.utils import units
import ipi.utils.mathtools as mt


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
