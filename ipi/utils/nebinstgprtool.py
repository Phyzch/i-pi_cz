'''
utility module for neb_instanton_gpr.py module.
Written by Chenghao Zhang, Pacific Northwest National Laboratory (chenghao.zhang@pnnl.gov)
'''
import numpy as np 
from ipi.utils.internalcoordtools import non_redundant_coordinate_transformer
from ipi.utils.gprtools import GPModelWithDerivativesWrapper
from ipi.engine.beads import Beads
from ipi.utils.depend import dstrip
import re 
import os 
from ipi.utils.nebinstool import RK4
import ipi.utils.nebinstool 

def check_neb_early_stop(beads_x, trust_region_distance, gpr_model: GPModelWithDerivativesWrapper,
                         outerloop_step, inner_loop_neb_step):
    '''
    check early stoage criterion for LI-NEB algorithm with machine learning.
    If the bead move out of trusted region, then we stop the current move.
    Notice because of automatic relevance determination (ARD) (https://mogp-emulator.readthedocs.io/en/latest/methods/proc/ProcAutomaticRelevanceDetermination.html#:~:text=We%20describe%20here%20the%20method,scales%20in%20the%20covariance%20models.)
    each internal dimension has one length scale, therefore, the internal distance q should be scaled by kernel length scale.

    :param: beads_x: cartesian coordinate X of neb beads 
    :param: trust_region_ratio: cutoff for rmax / neb_path_length. If beads move out of trust region, we stop the inner neb loop. 
    :param: gpr_model: model to perform the Gaussian Process Regression.
    :param: outerloop_step: step index for the outer loop.
    :param: inner_loop_neb_step: step index for the inner loop.

    :return: early_stop_bool: bool variable to indicate whether there is bead out of trust region.
             out_range_bead_index: the bead index that move out of the trusted region.
    '''
    early_stop_bool = False 
    out_range_bead_index_list = []

    # kernel output scale and kernel length scale of kernels
    kernel_output_scale = gpr_model.output_kernel_outputscale()
    kernel_length_scale = gpr_model.output_kernel_lengthscale()

    # normalize the output scale:
    output_scale_sum = np.sum(kernel_output_scale)
    kernel_output_scale_normalized = kernel_output_scale / output_scale_sum
    # effective kernel lengthscale for scaling internal coordinate. l_eff^{-2} = sum_{n} output_scale_n / (l_n)^2.   
    effective_kernel_length_scale = np.power(np.sum(kernel_output_scale_normalized[:, np.newaxis] / np.power(kernel_length_scale, 2) , axis = 0), -0.5)

    # the location of current beads in internal coordinate
    beads_free_moving_internal_coordinate = gpr_model.get_free_moving_internal_coordinate(np.copy(beads_x))
    
    # distance cutoff for trust region.
    distance_cutoff = trust_region_distance
    # the location of training data in internal coordinate.
    gpr_training_free_moving_internal_coordinate = gpr_model.output_free_moving_training_internal_inputs()
    
    # compute the distance and find beads that move out of the trusted region.
    nbeads = np.shape(beads_x)[0]
    internal_coordinate_closest_r_list = []
    for bead_index in range(nbeads):
        bead_internal_q = beads_free_moving_internal_coordinate[bead_index]

        # distance between gpr training data and beads.
        internal_coordinate_r = np.linalg.norm( (bead_internal_q[np.newaxis, :] - gpr_training_free_moving_internal_coordinate) / effective_kernel_length_scale , axis = 1) 
        
        nearest_gpr_data_index = np.argmin(internal_coordinate_r)

        # distance r between beads and the closest gpr training data
        internal_coordinate_closest_r = internal_coordinate_r[nearest_gpr_data_index]
        
        internal_coordinate_closest_r_list.append(internal_coordinate_closest_r)
        if internal_coordinate_closest_r > distance_cutoff:
            early_stop_bool = True 
            out_range_bead_index_list.append(bead_index)
    
    internal_coordinate_closest_r_list = np.array(internal_coordinate_closest_r_list)
    
    # output early stop information
    print("\n")
    print("@early stop info: outer loop: {},  inner loop: {} ".format(outerloop_step, inner_loop_neb_step))
    # print("effective kernel length scale: " + str(effective_kernel_length_scale))
    print("internal coordinate distance cutoff: " + str(distance_cutoff))
    print("distance for beads to nearest GPR point: " + str(internal_coordinate_closest_r_list))
    print("\n")

    if early_stop_bool:
        print("\n")
        print("@Early Stop for Inner Loop. outer loop: {},  inner loop: {} ".format(outerloop_step, inner_loop_neb_step))
        print("bead index that cause early stop (starting from 0) : " + str(out_range_bead_index_list))
        print("distance for beads out of trust region: " + str(internal_coordinate_closest_r_list[out_range_bead_index_list]))
        print("distance cutoff: {}".format(distance_cutoff))
        print("\n")

    return early_stop_bool, out_range_bead_index_list, internal_coordinate_closest_r_list


def print_ab_initio_calculation_number(ab_initio_calculation_number, output_maker, step):
    '''
    print number of ab initio calculation during GPR optimization. Used to see how much computational effort GPR saves
    '''
    file_name = "ab_initio_force_number step " + str(step) + ".txt"
    with open(file_name, "w") as f:
        f.write("ab initio calculation number:  " + str(ab_initio_calculation_number) + "\n")


def check_gpr_fitting_error(gpr_beads, gpr_forces, gpr_model : GPModelWithDerivativesWrapper, energy_shift, q):
    '''
    use gpr_beads and gpr_forces as force engine to compute ab-initio force and potential.
    use gpr model to predict the potential and force. 
    compare the difference between gpr prediction and ab-initio result.
    
    :param: gpr_beads: beads to store location of q.
    :param: gpr_forces: force engine to output ab-initio potential V and forces f.
    :param: gpr_model: Gaussian Process Regression Model.
    :param: energy shift: energy shift of ab-initio potential.
    :param: q: coordinate q to evaluate the Gaussian Process Regression error.
    '''
    gpr_beads.q[0] = q
    ab_initio_force = gpr_forces.f[0]
    ab_initio_pot = gpr_forces.pots[0] - energy_shift

    predicted_V_shift, predicted_V_grad, _, _ = gpr_model.predict_latent_function(gpr_beads.q)
    predicted_gpr_bead_force = - predicted_V_grad[0]
    predicted_V_shift = predicted_V_shift[0]

    test_V_error = np.abs((predicted_V_shift - ab_initio_pot) / ab_initio_pot)
    test_df = ab_initio_force - predicted_gpr_bead_force
    test_df_error = np.linalg.norm(test_df) / np.linalg.norm(ab_initio_force)   

    print("V error for test data " + str(test_V_error) + "   V value: " + str(ab_initio_pot) + "  predicted V value: " + str(predicted_V_shift) )
    print("f error for test data " + str(test_df_error)+ "   force amplitude: " + str(np.linalg.norm(ab_initio_force)) +  "  predicted force amplitude:  " + str(np.linalg.norm(predicted_gpr_bead_force)) )
    print("\n")
    return predicted_V_shift, predicted_gpr_bead_force, ab_initio_pot, ab_initio_force
    

def store_training_data(cartesian_coordinate_x, V, forces, prefix):
    '''
    store the initial training data for training of GPR model.
    In this way, when we do fine-tuning of hyper-parameter for GPR model, we do not to compute ab-initio potential and force again.

    :param: cartesian_coordinate_x: cartesian coordinate of training data. in atomic unit
    :param:  V: potential V (without shifted by energy shift). in Hatree unit
    :param: forces: forces at data point. in Hatree / atomic unit.
    :param: output_maker: output_maker provided by i-pi program. For output streaming.
    '''
    training_bead_number = np.shape(cartesian_coordinate_x)[0]
    dofs = np.shape(cartesian_coordinate_x)[1]
    # for cartesian coordinate_x
    coordinate_file_name = prefix + "_coord.txt"
    if os.path.exists(coordinate_file_name):
        os.rename(coordinate_file_name, "#" + coordinate_file_name)
    with open(coordinate_file_name, "w") as f:
        f.write("Total Bead number: \n")
        f.write(str(training_bead_number) + "\n")

        f.write("#Bead     cartesian coordinate \n")
        for i in range(training_bead_number):
            f.write(str(i) + "    ")
            for j in range(dofs):
                f.write(str(cartesian_coordinate_x[i,j]) + " ")
            f.write("\n")
    
    # for potential V.
    V_file_name = prefix + "_pot.txt"
    if os.path.exists(V_file_name):
        os.rename(V_file_name, "#" + V_file_name)
    with open(V_file_name, "w") as f:
        f.write("Total Bead number: \n")
        f.write(str(training_bead_number) + "\n")

        f.write("#Bead   Energy(Hatree) \n")
        for i in range(training_bead_number):
            f.write(str(i) + "    " + str(V[i]) + "\n")
    
    # for force f:
    force_file_name = prefix + "_force.txt"
    if os.path.exists(force_file_name):
        os.rename(force_file_name, "#" + force_file_name)
    with open(force_file_name, "w") as f:
        f.write("Total Bead number: \n")
        f.write(str(training_bead_number) + "\n")

        f.write("#Bead  Force (Hatree / a.u.) \n")
        for i in range(training_bead_number):
            f.write(str(i) + "    ")
            for j in range(dofs):
                f.write(str(forces[i,j]) + " ")
            f.write("\n")
    
def extract_number_from_line(line):
    line = re.split(' ', line.strip())
    line = [ele for ele in line if ele != '']

    return line 

def read_training_data(prefix):
    '''
    read coordinate, potential V and force f for training data.
    '''
    coordinate_file_name = prefix + "_coord.txt"
    V_file_name = prefix + "_pot.txt"
    force_file_name = prefix + "_force.txt"

    assert os.path.exists(coordinate_file_name), "data: coordinate file: " + str(coordinate_file_name) + "  does not exist."
    assert os.path.exists(V_file_name), "data: potential V file: " + str(V_file_name) + "  does not exist."
    assert os.path.exists(force_file_name), "data: force f file: " + str(force_file_name) + "  does not exist"

    # read coordinate.
    cartesian_coordinate_x = []
    with open(coordinate_file_name, "r") as f:
        lines = f.readlines()
        bead_number = int(extract_number_from_line(lines[1])[0])

        start_line_index = 3
        for bead_index in range(bead_number):
            line_index = start_line_index + bead_index 
            line = extract_number_from_line(lines[line_index])[1:]              # the first number is bead index.
            bead_x = np.array(list(map(float, line)))
            cartesian_coordinate_x.append(bead_x)
        
    cartesian_coordinate_x = np.array(cartesian_coordinate_x)

    # read potential V
    training_V = []
    with open(V_file_name, "r") as f:
        lines = f.readlines()
        bead_number = int(extract_number_from_line(lines[1])[0])

        start_line = 3
        for bead_index in range(bead_number):
            line_index = bead_index + start_line 
            V = float(extract_number_from_line(lines[line_index])[1])
            training_V.append(V)
    
    training_V = np.array(training_V)

    # read force
    training_forces = []
    with open(force_file_name, "r") as f:
        lines = f.readlines()
        bead_number = int(extract_number_from_line(lines[1])[0])

        start_line = 3
        for bead_index in range(bead_number):
            line_index = bead_index + start_line 
            line = extract_number_from_line(lines[line_index])[1:]
            force = np.array(list(map(float, line)))
            training_forces.append(force)
    
    training_forces = np.array(training_forces)

    return cartesian_coordinate_x, training_V, training_forces


def dydt_inverted_pot_gpr(y, t, param):
    '''
    y=[x,v]. That is y[0] = x. y[1] = v.
    dydt[0] = v. dydt[1] = a (inverted pot)
    param = [cl_beads, cl_forces, m3, tau]
    cl_beads: bead object that record the coordinate of current particle
    gpr_model: gaussian process regression model 
    m3 : mass. size : [3 * natom]
    tau: tangent direction of motion. unit vector.
    '''
    x = y[0]
    v = y[1]

    gpr_model = param[0]
    m3 = param[1]
    tau = param[2]

    # update coordinate of bead object to enable the forces object to compute force
    _ , grad_V, _, _ = gpr_model.predict_latent_function(np.array([x]))
    negative_f = grad_V[0]

    # compute acceleration along the path.
    a = ipi.utils.nebinstool.compute_acceleration_along_path(negative_f, tau, m3)

    dydt = np.array([ v, a ])

    return dydt 

def bisect_dt_gpr(dt_right, dt_left, old_y, t, param, target_dr):
    '''
    using bisection search method to find the appropriate dt value to go to end beads.
    '''
    old_x = np.copy(old_y[0])
    dr = 1000

    while abs(target_dr - dr) > 0.0005:
        dt = (dt_left + dt_right)/2
        new_y = RK4(np.copy(old_y), t, dydt_inverted_pot_gpr, param, dt)
        new_x = np.copy(new_y[0])
        dr = np.linalg.norm(new_x - old_x)

        if dr > target_dr:
            # dt is too large. make dt smaller.
            dt_right = dt 
        else:
            # dt is too small. make dt larger.
            dt_left = dt 
    
    return dt, new_y 