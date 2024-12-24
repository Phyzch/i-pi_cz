"""
utility module for neb_instanton_gpr.py module.
Written by Chenghao Zhang & Niri Govind, Pacific Northwest National Laboratory (chenghao.zhang@pnnl.gov), 2024.
"""

import numpy as np
from ipi.utils.gprtools import GPModelWithDerivativesWrapper
import re
import os
import ipi.utils.nebinstgprtool
from ipi.utils.nebinstool import RK4
import ipi.utils.nebinstool
from ipi.utils.depend import dstrip
from ipi.utils.gpr_hessian_tools import GPModelWithHessiansWrapper
# import ipi.utils.internalcoordtools
import ipi.utils.internal.internaltools
import shutil


def check_neb_early_stop(
    beads_x,
    trust_region_distance,
    gpr_model: GPModelWithDerivativesWrapper,
    outerloop_step,
    inner_loop_neb_step,
):
    """
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
    """
    early_stop_bool = False
    out_range_bead_index_list = []

    # kernel output scale and kernel length scale of kernels
    kernel_output_scale = gpr_model.output_kernel_outputscale()
    kernel_length_scale = gpr_model.output_kernel_lengthscale()

    # normalize the output scale:
    output_scale_sum = np.sum(kernel_output_scale)
    kernel_output_scale_normalized = kernel_output_scale / output_scale_sum
    # effective kernel lengthscale for scaling internal coordinate. l_eff^{-2} = sum_{n} output_scale_n / (l_n)^2.
    effective_kernel_length_scale = np.power(
        np.sum(
            kernel_output_scale_normalized[:, np.newaxis]
            / np.power(kernel_length_scale, 2),
            axis=0,
        ),
        -0.5,
    )

    # the location of current beads in internal coordinate
    beads_free_moving_internal_coordinate = (
        gpr_model.get_free_moving_internal_coordinate(np.copy(beads_x))
    )

    # distance cutoff for trust region.
    distance_cutoff = trust_region_distance
    # the location of training data in internal coordinate.
    gpr_training_free_moving_internal_coordinate = (
        gpr_model.output_free_moving_training_internal_inputs()
    )

    # compute the distance and find beads that move out of the trusted region.
    nbeads = np.shape(beads_x)[0]
    internal_coordinate_closest_r_list = []
    for bead_index in range(nbeads):
        bead_internal_q = beads_free_moving_internal_coordinate[bead_index]

        # distance between gpr training data and beads.
        internal_coordinate_r = np.linalg.norm(
            (
                bead_internal_q[np.newaxis, :]
                - gpr_training_free_moving_internal_coordinate
            )
            / effective_kernel_length_scale,
            axis=1,
        )

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
    print(
        "@early stop info: outer loop: {},  inner loop: {} ".format(
            outerloop_step, inner_loop_neb_step
        )
    )
    # print("effective kernel length scale: " + str(effective_kernel_length_scale))
    print("internal coordinate distance cutoff: " + str(distance_cutoff))
    print(
        "distance for beads to nearest GPR point: "
        + str(internal_coordinate_closest_r_list)
    )
    print("\n")

    if early_stop_bool:
        print("\n")
        print(
            "@Early Stop for Inner Loop. outer loop: {},  inner loop: {} ".format(
                outerloop_step, inner_loop_neb_step
            )
        )
        print(
            "bead index that cause early stop (starting from 0) : "
            + str(out_range_bead_index_list)
        )
        print(
            "distance for beads out of trust region: "
            + str(internal_coordinate_closest_r_list[out_range_bead_index_list])
        )
        print("distance cutoff: {}".format(distance_cutoff))
        print("\n")

    return (
        early_stop_bool,
        out_range_bead_index_list,
        internal_coordinate_closest_r_list,
    )


def print_ab_initio_calculation_number(
    ab_initio_calculation_number, output_maker, step
):
    """
    print number of ab initio calculation during GPR optimization. Used to see how much computational effort GPR saves
    """
    file_name = "ab_initio_force_number step " + str(step) + ".txt"
    with open(file_name, "w") as f:
        f.write(
            "ab initio calculation number (including initial training data):  " + str(ab_initio_calculation_number) + "\n"
        )


def check_gpr_fitting_error(
    gpr_beads, gpr_forces, gpr_model: GPModelWithDerivativesWrapper, energy_shift, q
):
    """
    use gpr_beads and gpr_forces as force engine to compute ab-initio force and potential.
    use gpr model to predict the potential and force.
    compare the difference between gpr prediction and ab-initio result.

    :param: gpr_beads: beads to store location of q.
    :param: gpr_forces: force engine to output ab-initio potential V and forces f.
    :param: gpr_model: Gaussian Process Regression Model.
    :param: energy shift: energy shift of ab-initio potential.
    :param: q: coordinate q to evaluate the Gaussian Process Regression error.
    """
    gpr_beads.q[0] = q
    ab_initio_force = gpr_forces.f[0]
    ab_initio_pot = gpr_forces.pots[0] - energy_shift

    beads_q = dstrip(gpr_beads.q).copy()
    predicted_V_shift, predicted_V_grad, _, _ = gpr_model.predict_latent_function(
        beads_q
    )
    predicted_gpr_bead_force = -predicted_V_grad[0]
    predicted_V_shift = predicted_V_shift[0]

    test_V_error = np.abs((predicted_V_shift - ab_initio_pot) / ab_initio_pot)
    test_df = ab_initio_force - predicted_gpr_bead_force
    test_df_error = np.linalg.norm(test_df) / np.linalg.norm(ab_initio_force)

    print(
        "V error for test data "
        + str(test_V_error)
        + "   V value: "
        + str(ab_initio_pot)
        + "  predicted V value: "
        + str(predicted_V_shift)
    )
    print(
        "f error for test data "
        + str(test_df_error)
        + "   force amplitude: "
        + str(np.linalg.norm(ab_initio_force))
        + "  predicted force amplitude:  "
        + str(np.linalg.norm(predicted_gpr_bead_force))
    )
    print("\n")
    return predicted_V_shift, predicted_gpr_bead_force, ab_initio_pot, ab_initio_force


def store_training_data(cartesian_coordinate_x, V, forces, prefix):
    """
    store the initial training data for training of GPR model.
    In this way, when we do fine-tuning of hyper-parameter for GPR model, we do not to compute ab-initio potential and force again.

    :param: cartesian_coordinate_x: cartesian coordinate of training data. in atomic unit
    :param:  V: potential V (without shifted by energy shift). in Hatree unit
    :param: forces: forces at data point. in Hatree / atomic unit.
    :param: output_maker: output_maker provided by i-pi program. For output streaming.
    """
    training_bead_number = np.shape(cartesian_coordinate_x)[0]
    ndofs = np.shape(cartesian_coordinate_x)[1]
    # create folder with prefix
    if os.path.exists("#" + prefix):
        shutil.rmtree("#" + prefix)

    if os.path.exists(prefix):
        shutil.move(prefix, "#" + prefix)

    os.mkdir(prefix)

    # for cartesian coordinate_x
    coordinate_file_name = os.path.join(prefix, "coord.txt")
    if os.path.exists(coordinate_file_name):
        os.rename(coordinate_file_name, coordinate_file_name + "#")
    with open(coordinate_file_name, "w") as f:
        f.write("Total Bead number: \n")
        f.write(str(training_bead_number) + "\n")

        f.write("#Bead     cartesian coordinate \n")
        for i in range(training_bead_number):
            f.write(str(i) + "    ")
            for j in range(ndofs):
                f.write(str(cartesian_coordinate_x[i, j]) + " ")
            f.write("\n")

    # for potential V.
    V_file_name = os.path.join(prefix, "pot.txt")
    if os.path.exists(V_file_name):
        os.rename(V_file_name, V_file_name + "#")
    with open(V_file_name, "w") as f:
        f.write("Total Bead number: \n")
        f.write(str(training_bead_number) + "\n")

        f.write("#Bead   Energy(Hatree) \n")
        for i in range(training_bead_number):
            f.write(str(i) + "    " + str(V[i]) + "\n")

    # for force f:
    force_file_name = os.path.join(prefix, "force.txt")
    if os.path.exists(force_file_name):
        os.rename(force_file_name, force_file_name + "#")
    with open(force_file_name, "w") as f:
        f.write("Total Bead number: \n")
        f.write(str(training_bead_number) + "\n")

        f.write("#Bead  Force (Hatree / a.u.) \n")
        for i in range(training_bead_number):
            f.write(str(i) + "    ")
            for j in range(ndofs):
                f.write(str(forces[i, j]) + " ")
            f.write("\n")


def store_training_data_with_hessian(
    cartesian_coordinate_x, V, forces, hessian_index_list, hessians, prefix
):
    """
    store the training data (coord, pot, grad) + hessian
    """
    ndofs = np.shape(cartesian_coordinate_x)[1]

    store_training_data(cartesian_coordinate_x, V, forces, prefix)

    # for hessian h:
    hessian_file_name = os.path.join(prefix, "hessian.txt")
    if os.path.exists(hessian_file_name):
        os.rename(hessian_file_name, hessian_file_name + "#")

    hessian_data_num = len(hessian_index_list)
    with open(hessian_file_name, "w") as f:
        f.write("Total hessian number: \n")
        f.write(str(int(hessian_data_num)) + "\n")

        f.write("Bead_index  Hessian (Hatree/ a.u.^2) \n")
        for i in range(hessian_data_num):
            f.write(str(hessian_index_list[i]) + "   ")
            for j in range(ndofs):
                for k in range(ndofs):
                    f.write(str(hessians[i, j, k]) + " ")

            f.write("\n")


def store_candidate_hessian_data_coordinate(
    candidate_hessian_point_x, used_hessian_index_in_candidate_list, prefix
):
    """
    store the information about which data point we have used hessian in gpr model and
    what are the potential (candidate) data points we can compute hessians and add to gpr model.
    :param: candidate_hessian_point_x: coordinate of candidate data points that we can compute hessians.
    :param: hessian_index_in_candidate_list: the index of data point that we have already computed hessians.
    :param: prefix: name of folders that we will store info
    """
    assert os.path.exists(prefix), "the prefix folder should have already been created."
    candidate_point_number = len(candidate_hessian_point_x)
    ndofs = np.shape(candidate_hessian_point_x)[1]

    # write the coordinate of candidate hessian data points.
    hessian_coordinate_file_name = os.path.join(prefix, "candidate_hessian_coord.txt")
    if os.path.exists(hessian_coordinate_file_name):
        os.rename(hessian_coordinate_file_name, hessian_coordinate_file_name + "#")
    with open(hessian_coordinate_file_name, "w") as f:
        f.write("Total candidate point number: \n")
        f.write(str(candidate_point_number) + "\n")

        f.write("#Index   cartesian coordinate \n")
        for i in range(candidate_point_number):
            f.write(str(i) + "    ")
            for j in range(ndofs):
                f.write(str(candidate_hessian_point_x[i, j]) + " ")
            f.write("\n")

    # write the index of data point that we have already computed hessian information.
    hessian_index_file_name = os.path.join(
        prefix, "hessian_index_in_candidate_point_list.txt"
    )

    if os.path.exists(hessian_index_file_name):
        os.rename(hessian_index_file_name, hessian_index_file_name + "#")

    used_hessian_point_num = len(used_hessian_index_in_candidate_list)
    with open(hessian_index_file_name, "w") as f:
        f.write("Index for data point that we have computed hessians. \n")
        for i in range(used_hessian_point_num):
            used_hessian_index = int(used_hessian_index_in_candidate_list[i])
            f.write(str(used_hessian_index) + " ")
        f.write("\n")

def store_candidate_grad_data_coordinate(
        candidate_grad_point_x, used_grad_index_in_candidate_list, prefix
):
    """
    store the information about which data point we have used gradient in gpr model and
    what are the potential (candidate) data points we can compute gradients and add to gpr model.
    :param: candidate_grad_point_x: coordinate for candidate points that we can compute gradients.
    :param: used_grad_index_in_candidate_list: the index of data points that we have already computed gradients.
    :param: prefix: name of folders that we will store info
    """
    assert os.path.exists(prefix), "the prefix folder should have already been created."
    candidate_point_number = len(candidate_grad_point_x)
    ndofs = np.shape(candidate_grad_point_x)[1]

    # write the coordinate of candidate gradient data points
    gradient_coordinate_file_name = os.path.join(prefix, "candidate_gradient_coord.txt")
    if os.path.exists(gradient_coordinate_file_name):
        os.rename(gradient_coordinate_file_name, gradient_coordinate_file_name + "#")

    with open(gradient_coordinate_file_name, "w") as f:
        f.write("Total candidate point number: \n")
        f.write(str(candidate_point_number) + "\n")

        f.write("#Index   cartesian coordinate \n")
        for i in range(candidate_point_number):
            f.write(str(i) + "    ")
            for j in range(ndofs):
                f.write(str(candidate_grad_point_x[i, j]) + " ")
            f.write("\n")
    
    # write the index of gradient data point that we have already computed gradient information.
    grad_index_file_name = os.path.join(
        prefix, "grad_index_in_candidate_point_list.txt"
    )

    if os.path.exists(grad_index_file_name):
        os.rename(grad_index_file_name, grad_index_file_name + "#")

    used_grad_point_num = len(used_grad_index_in_candidate_list)
    with open(grad_index_file_name, "w") as f:
        f.write("Index for data point that we have computed gradients. \n")
        for i in range(used_grad_point_num):
            used_grad_index = int(used_grad_index_in_candidate_list[i])
            f.write(str(used_grad_index) + " ")
        f.write("\n")

def extract_number_from_line(line):
    line = re.split(" ", line.strip())
    line = [ele for ele in line if ele != ""]

    return line


def read_training_data(prefix):
    """
    read coordinate, potential V and force f for training data.
    """
    coordinate_file_name = os.path.join(prefix, "coord.txt")
    V_file_name = os.path.join(prefix, "pot.txt")
    force_file_name = os.path.join(prefix, "force.txt")

    assert os.path.exists(coordinate_file_name), (
        "data: coordinate file: " + str(coordinate_file_name) + "  does not exist."
    )
    assert os.path.exists(V_file_name), (
        "data: potential V file: " + str(V_file_name) + "  does not exist."
    )
    assert os.path.exists(force_file_name), (
        "data: force f file: " + str(force_file_name) + "  does not exist"
    )

    # read coordinate.
    cartesian_coordinate_x = []
    with open(coordinate_file_name, "r") as f:
        lines = f.readlines()
        bead_number = int(extract_number_from_line(lines[1])[0])

        start_line_index = 3
        for bead_index in range(bead_number):
            line_index = start_line_index + bead_index
            line = extract_number_from_line(lines[line_index])[
                1:
            ]  # the first number is bead index.
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


def read_hessian_data(prefix, ndofs):
    """
    """
    hessian_file_name = os.path.join(prefix, "hessian.txt")
    assert os.path.exists(hessian_file_name), (
        "data: hessian file: " + str(hessian_file_name) + " does not exist"
    )

    hessian_index_list = []
    hessian_data_list = []

    with open(hessian_file_name, "r") as f:
        lines = f.readlines()
        bead_with_hessian_number = int(extract_number_from_line(lines[1])[0])
        start_line_index = 3
        for index in range(bead_with_hessian_number):
            line_index = start_line_index + index
            line = extract_number_from_line(lines[line_index])

            bead_with_hessian_index = int(float(line[0]))
            hessian_index_list.append(bead_with_hessian_index)

            hessian_data = np.array(list(map(float, line[1:])))
            assert len(hessian_data) == np.power(
                ndofs, 2
            ), "the length of hessian data read from file is wrong."
            hessian_data = np.reshape(hessian_data, (ndofs, ndofs))
            hessian_data_list.append(hessian_data)

    hessian_index_list = np.array(hessian_index_list)
    hessian_data_list = np.array(hessian_data_list)

    return hessian_index_list, hessian_data_list 

def read_training_data_with_hessian(prefix):
    """
    read coordinate, potential V, force f and hessian h from training data
    """
    cartesian_coordinate_x, training_V, training_forces = read_training_data(prefix)

    ndofs = np.shape(training_forces)[1]
    # read hessian
    hessian_index_list, hessian_data_list = read_hessian_data(prefix, 
                                                              ndofs)

    return (
        cartesian_coordinate_x,
        training_V,
        training_forces,
        hessian_index_list,
        hessian_data_list,
    )


def read_candidate_hessian_data_coordinate(prefix):
    """
    read the information about which data point we have used hessian in gpr model and
    what are the potential (candidate) data points we can compute hessians and add to gpr model.
    :param: prefix: name of folders that we will load info
    """
    assert os.path.exists(prefix), "the prefix folder should have already been created."

    # read candidate hessian coordinate
    hessian_coordinate_file_name = os.path.join(prefix, "candidate_hessian_coord.txt")
    candidate_hessian_point_x = []
    with open(hessian_coordinate_file_name, "r") as f:
        lines = f.readlines()
        bead_number = int(extract_number_from_line(lines[1])[0])

        start_line_index = 3
        for bead_index in range(bead_number):
            line_index = start_line_index + bead_index
            line = extract_number_from_line(lines[line_index])[
                1:
            ]  # the first number is bead index.
            bead_x = np.array(list(map(float, line)))
            candidate_hessian_point_x.append(bead_x)

    candidate_hessian_point_x = np.array(candidate_hessian_point_x)

    # read the index of used point in candidate list.
    hessian_index_file_name = os.path.join(
        prefix, "hessian_index_in_candidate_point_list.txt"
    )

    with open(hessian_index_file_name, "r") as f:
        lines = f.readlines()
        line = extract_number_from_line(lines[1])
        used_hessian_index_in_candidate_list = np.array(list(map(int, line)))

    return candidate_hessian_point_x, used_hessian_index_in_candidate_list

def read_candidate_grad_data_coordinate(prefix):
    """
    read the information about which data point we have used hessian in gpr model and
    what are the potential (candidate) data points we can compute hessians and add to gpr model.
    :param: prefix: name of folders that we will load info
    """
    assert os.path.exists(prefix), "the prefix folder should have already been created."

    # read candidate hessian coordinate
    grad_coordinate_file_name = os.path.join(prefix, "candidate_gradient_coord.txt")
    candidate_grad_point_x = []
    with open(grad_coordinate_file_name, "r") as f:
        lines = f.readlines()
        bead_number = int(extract_number_from_line(lines[1])[0])

        start_line_index = 3
        for bead_index in range(bead_number):
            line_index = start_line_index + bead_index
            line = extract_number_from_line(lines[line_index])[
                1:
            ]  # the first number is bead index.
            bead_x = np.array(list(map(float, line)))
            candidate_grad_point_x.append(bead_x)

    candidate_grad_point_x = np.array(candidate_grad_point_x)

    # read the index of used point in candidate list.
    grad_index_file_name = os.path.join(
        prefix, "grad_index_in_candidate_point_list.txt"
    )

    with open(grad_index_file_name, "r") as f:
        lines = f.readlines()
        line = extract_number_from_line(lines[1])
        used_grad_index_in_candidate_list = np.array(list(map(int, line)))

    return candidate_grad_point_x, used_grad_index_in_candidate_list


def dydt_inverted_pot_gpr(y, t, param):
    """
    y = [r, v_r].
    That is y[0] = r, y[1] = v_r = dr/dt.
    dydt[0] = v_r, dydt[1] = a (acceleration of r on inverted potential.)
    param = [gpr_model, m3, cubic_spline]
    here gpr_model: gaussian process regression model
         m3_matrix: mass. 2d diagonal matrix. size: [3 * natoms, 3* natoms]. The diagonal element is m3.
         cubic_spline: cubic spline function that return x(r).

    acceleration: d^2 r/ dt^2 is from constrained dynamics.
    See eq.(13) in Witkin, A. (1997). Computer graphics, 9, 27
    """
    r_distance = y[0]
    v_r = y[1]

    gpr_model = param[0]
    m3_matrix = param[1]
    cubic_spline = param[2]

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

    # compute the negative force in upside down potential.
    _, grad_V, _, _ = gpr_model.predict_latent_function(np.array([x]))
    negative_f = grad_V[0]

    # compute the acceleration of r.
    a_r = ipi.utils.nebinstool.compute_r_acceleration_along_path(
        negative_f, dx_dr, dx_dr_rate, m3_matrix, v_r
    )

    dydt = np.array([v_r, a_r])

    return dydt


def compute_frobenius_norm(input_matrix: np.ndarray):
    """
    compute the frobenius norm of the matrix.
    """
    # make the matrix has 3 dimensions
    if len(np.shape(input_matrix)) == 2:
        matrix = input_matrix[np.newaxis, :, :]
    elif len(np.shape(input_matrix)) == 3:
        matrix = input_matrix
    else:
        raise "the shape of matrix for computing frobenius norm has to have 2 or 3 dimensions"

    matrix_transpose = np.transpose(matrix, (0, 2, 1))
    frobenius_norm = np.sqrt(
        np.trace(np.matmul(matrix_transpose, matrix), axis1=1, axis2=2)
    )

    if len(np.shape(input_matrix)) == 2:
        # for 2d matrix, we return its frobenius norm (scalar)
        frobenius_norm = frobenius_norm[0]

    return frobenius_norm


def compute_relative_matrix_error_with_frobenius_norm(
    ab_initio_matrix: np.ndarray, predicted_matrix: np.ndarray
):
    """
    compute the relative error of predicted matrix (B) regarding the ab-initio matrix (A).
    ||A - B|| / ||A||. Here ||*|| is the Frobenius norm.
    """
    diff_matrix = predicted_matrix - ab_initio_matrix
    diff_norm = compute_frobenius_norm(diff_matrix)
    ab_initio_norm = compute_frobenius_norm(ab_initio_matrix)

    relative_error = diff_norm / ab_initio_norm
    return relative_error


def check_gpr_hessian_model_lengthscale(gpr_hessian_model: GPModelWithHessiansWrapper):
    """
    check the length scale of squared exponential kernel function of Gaussian Process Regression model which predicts hessian.
    By inspecting length scale, we can see if there is over-fitting of the model.
    """
    gpr_kernel_number = gpr_hessian_model.gpr_SE_kernel_number
    (
        gpr_hessian_kernel_outputscale,
        gpr_hessian_lengthscale_list,
        gpr_hessian_lengthscale_ratio_list,
    ) = gpr_hessian_model.check_gpr_lengthscale()

    print("gpr hessian output scale: " + str(gpr_hessian_kernel_outputscale))

    for i in range(gpr_kernel_number):
        # we have options to add multiple Gaussian Process Regression kernels.
        print(
            "kernel: "
            + str(i)
            + " gpr hessian model length scale: "
            + str(gpr_hessian_lengthscale_list[i])
        )
        print(
            "kernel: "
            + str(i)
            + " gpr hessian length scale ratio: "
            + str(gpr_hessian_lengthscale_ratio_list[i])
        )

    return (
        gpr_hessian_kernel_outputscale,
        gpr_hessian_lengthscale_list,
        gpr_hessian_lengthscale_ratio_list,
    )


def compare_ab_initio_hessian_and_predicted_hessian(
    cartesian_x_with_hessian,
    gradients_with_hessian,
    hessians_full,
    hessian_data_point_index_list,
    gpr_hessian_model: GPModelWithHessiansWrapper,
    internal_coordinate_bool,
    training_data_bool,
    print_hessian=False,
):
    """
    compare the result of ab initio hessian and hessian predicted by Gaussian Process Regression model.
    :param: cartesian_x_with_hessian: the Cartesian coordinate of data with hessian information.
    :param: gradients_with_hessian: the gradient of data with hessian information.
    :param: hessians_full: the hessian data.
    :param: hessian_data_point_index_list: the index of hessian data point in the data set we need to compare and show.
    :param: gpr_hessian_model: Gaussian Process Regression model that predicts hessian information.
    :param: internal_coordinate_bool: Bool variable. If true, return hessian in internal coordinate. If false, return hessian in Cartesian coordinate.
    :param: training_data_bool: whether the data we are comparing is training data or testing data. This will make output information different.
    """
    data_num = len(hessian_data_point_index_list)

    ab_initio_grads = gradients_with_hessian[hessian_data_point_index_list]
    # transform hessian into internal coordinate
    ab_initio_hessians = hessians_full[hessian_data_point_index_list]

    ab_initio_hessians_q = gpr_hessian_model.coordinate_transformer.transform_cartesian_hessian_to_internal_hessian(
        cartesian_x_with_hessian[hessian_data_point_index_list],
        gradients_with_hessian[hessian_data_point_index_list],
        ab_initio_hessians,
    )
    
    index_to_show_list = [0, int(data_num / 2), data_num - 1]

    if internal_coordinate_bool:
        # the predicted results in internal coordinate.
        predicted_pots, predicted_grads_q, predicted_hessians_q, _, _, _ = (
            gpr_hessian_model.predict_latent_function(
                cartesian_x_with_hessian,
                hessian_data_point_index_list,
                internal_coordinate_bool=True,
            )
        )

        free_moving_dofs = gpr_hessian_model.FixingDofs.free_moving_dofs
        free_moving_dofs_2d = np.meshgrid(
            free_moving_dofs, free_moving_dofs, indexing="ij"
        )

        selected_ab_initio_hessian_q = ab_initio_hessians_q[
            :, free_moving_dofs_2d[0], free_moving_dofs_2d[1]
        ]
        selected_predicted_hessian_q = predicted_hessians_q[
            :, free_moving_dofs_2d[0], free_moving_dofs_2d[1]
        ]

        # compute the relative error of matrix using Frobenius norm as measure
        relative_hessian_error = compute_relative_matrix_error_with_frobenius_norm(
            selected_ab_initio_hessian_q, selected_predicted_hessian_q
        )
        if training_data_bool:
            print(
                "training case: relative hessian error:  " + str(relative_hessian_error)
            )
        else:
            print("test case: relative hessian error:  " + str(relative_hessian_error))

        print("\n")

        if print_hessian:
            for index_to_show in index_to_show_list:
                if training_data_bool:
                    print(
                        "training case "
                        + str(index_to_show)
                        + " : ab initio: "
                        + str(selected_ab_initio_hessian_q[index_to_show])
                    )
                    print(
                        "training case "
                        + str(index_to_show)
                        + " : predicted: "
                        + str(selected_predicted_hessian_q[index_to_show])
                    )
                else:
                    print(
                        "test case "
                        + str(index_to_show)
                        + " : ab initio: "
                        + str(selected_ab_initio_hessian_q[index_to_show])
                    )
                    print(
                        "test case "
                        + str(index_to_show)
                        + " : predicted: "
                        + str(selected_predicted_hessian_q[index_to_show])
                    )

                print("\n")

            for i in range(3):
                print("\n")

        return selected_ab_initio_hessian_q, selected_predicted_hessian_q

    else:
        # the predicted results in Cartesian coordinate.
        predicted_pots, predicted_grads, predicted_hessians, _, _, _ = (
            gpr_hessian_model.predict_latent_function(
                cartesian_x_with_hessian,
                hessian_data_point_index_list,
                internal_coordinate_bool=False,
            )
        )

        # compute the relative error of matrix using Frobenius norm as measure
        relative_hessian_error = compute_relative_matrix_error_with_frobenius_norm(
            ab_initio_hessians, predicted_hessians
        )
        if training_data_bool:
            print(
                "training case: relative hessian error:  " + str(relative_hessian_error)
            )
        else:
            print("test case: relative hessian error:  " + str(relative_hessian_error))

        print("\n")

        if print_hessian:
            for index_to_show in index_to_show_list:
                if training_data_bool:
                    print(
                        "training case "
                        + str(index_to_show)
                        + " : ab initio: "
                        + str(ab_initio_hessians[index_to_show])
                    )
                    print(
                        "training case "
                        + str(index_to_show)
                        + " : predicted: "
                        + str(predicted_hessians[index_to_show])
                    )
                else:
                    print(
                        "test case "
                        + str(index_to_show)
                        + " : ab initio: "
                        + str(ab_initio_hessians[index_to_show])
                    )
                    print(
                        "test case "
                        + str(index_to_show)
                        + " : predicted: "
                        + str(predicted_hessians[index_to_show])
                    )

                print("\n")

            for i in range(3):
                print("\n")

        return ab_initio_hessians, predicted_hessians


def test_gpr_hessian_prediction(
    gpr_model: GPModelWithDerivativesWrapper,
    energy_shift,
    cartesian_x_with_hessian,
    pots_with_hessian_before_shift,
    grads_with_hessian,
    hessians_data,
    cartesian_fix_dofs,
    gpr_fix_internal_dofs_bool,
    gpr_fix_internal_dofs_cutoff
):
    """
    test gaussian process regression model that predict hessian.
    The parameter below are x, potentials, gradients and hessians of data.
    :param: cartesian_x_with_hessian:  coordinate of data with hessian information we need to test with GPR model.
    :param: pots_with_hessian_before_shift: potentials of data before we do the energy shift with self.energy_shift.
    :param: grads_with_hessian: gradients of data.
    :param: hessians_data: hessians of data point.
    """
    # collect gradient, force & hessian data from gpr model. These data will also be used to initialize gpr_hessian_model.
    cartesian_x_gpr_model = np.copy(gpr_model.train_cartesian_inputs)
    pots_gpr_model = np.copy(gpr_model.train_cartesian_targets[:, 0])
    grads_gpr_model = np.copy(gpr_model.train_cartesian_targets[:, 1:])

    gpr_model_training_data_num = len(cartesian_x_gpr_model)

    # training data point with hessians.
    nbeads_with_hessian = len(cartesian_x_with_hessian)
    natoms = gpr_model.natom

    # shift the potential data with energy shift.
    pots_with_hessian = pots_with_hessian_before_shift - energy_shift

    # we choose the training points that we use to train the gpr model.
    train_hessian_data_point_index_array = np.arange(0, nbeads_with_hessian, 3)
    test_hessian_data_point_index_array = np.delete(
        np.arange(0, nbeads_with_hessian), train_hessian_data_point_index_array
    )

    # Previous shape: [3 * natoms, nbeads * 3 * natoms].
    # change the shape of hessian to [nbeads, 3 * natoms, 3 * natoms]
    hessians_full = np.transpose(
        np.reshape(hessians_data, [3 * natoms, nbeads_with_hessian, 3 * natoms]),
        (1, 0, 2),
    )

    # add hessian data into training data
    train_cartesian_x = np.concatenate(
        [
            cartesian_x_gpr_model,
            cartesian_x_with_hessian[train_hessian_data_point_index_array],
        ]
    )
    train_pots = np.concatenate(
        [pots_gpr_model, pots_with_hessian[train_hessian_data_point_index_array]]
    )
    train_grads = np.concatenate(
        [grads_gpr_model, grads_with_hessian[train_hessian_data_point_index_array]]
    )
    train_hessians = hessians_full[train_hessian_data_point_index_array]
    hessian_data_point_index_in_training_data = (
        np.arange(len(train_hessian_data_point_index_array))
        + gpr_model_training_data_num
    )

    # initialize the gpr_hessian_model's training parameter with  output scale and length scale of previously trained model.
    kernel_lengthscale_initio_value = gpr_model.output_kernel_lengthscale()
    kernel_outputscale_initio_value = gpr_model.output_kernel_outputscale()
    # prepare the reference point with V, grad, hessian. The mean function of Gaussian Process Regression model is the Taylor expansion around such reference point.
    hessian_ref_data_point_index = 0
    ref_x = cartesian_x_with_hessian[hessian_ref_data_point_index]
    ref_V = np.array([pots_with_hessian[hessian_ref_data_point_index]])
    ref_grads = grads_with_hessian[hessian_ref_data_point_index]
    ref_hessians = hessians_full[hessian_ref_data_point_index]

    coordinate_transformer = gpr_model.coordinate_transformer
    # create Gaussian Process Regression model which can predict hessian information.
    # set the mean function of gpr hessian model as Taylor expansion aroudn the reference point.
    gpr_hessian_model = GPModelWithHessiansWrapper(
        train_cartesian_x,
        train_pots,
        train_grads,
        train_hessians,
        hessian_data_point_index_in_training_data,
        natoms,
        coordinate_transformer,
        cartesian_fix_dofs,
        gpr_model.gpr_SE_kernel_number,
        gpr_model.kernel_outputscale,
        gpr_model.kernel_lengthscale_ratio,
        gpr_model.noise_std,
        kernel_lengthscale_initio_value=kernel_lengthscale_initio_value,
        kernel_outputscale_initio_value=kernel_outputscale_initio_value,
        constant_mean_func_bool=False,
        ref_mean_x=ref_x,
        ref_mean_V=ref_V,
        ref_mean_grad_x=ref_grads,
        ref_mean_hessian_x=ref_hessians,
        gpr_fix_internal_dofs_bool= gpr_fix_internal_dofs_bool,
        gpr_fix_internal_dofs_cutoff= gpr_fix_internal_dofs_cutoff
    )

    internal_coordinate_with_hessian = coordinate_transformer.get_internal_coordinate_q(
        cartesian_x_with_hessian
    )
    internal_coordinate_with_hessian = internal_coordinate_with_hessian[
        :, gpr_hessian_model.FixingDofs.free_moving_dofs
    ]

    # check the length scale of gpr model after we finish the training. This way we can see if over-fitting happens.
    (
        gpr_hessian_kernel_outputscale,
        gpr_hessian_lengthscale_list,
        gpr_hessian_lengthscale_ratio_list,
    ) = check_gpr_hessian_model_lengthscale(gpr_hessian_model)

    # test the prediction of hessian of training data.
    ab_initio_train_hessian_q, predicted_train_hessian_q = (
        compare_ab_initio_hessian_and_predicted_hessian(
            cartesian_x_with_hessian,
            grads_with_hessian,
            hessians_full,
            train_hessian_data_point_index_array,
            gpr_hessian_model,
            internal_coordinate_bool=True,
            training_data_bool=True,
        )
    )

    # test data case for hessians in internal coordinate.
    ab_initio_test_hessian_q, predicted_test_hessian_q = (
        compare_ab_initio_hessian_and_predicted_hessian(
            cartesian_x_with_hessian,
            grads_with_hessian,
            hessians_full,
            test_hessian_data_point_index_array,
            gpr_hessian_model,
            internal_coordinate_bool=True,
            training_data_bool=False,
        )
    )

    # test data case for hessians in Cartesian coordinate
    ab_initio_test_hessian, predicted_test_hessian = (
        compare_ab_initio_hessian_and_predicted_hessian(
            cartesian_x_with_hessian,
            grads_with_hessian,
            hessians_full,
            test_hessian_data_point_index_array,
            gpr_hessian_model,
            internal_coordinate_bool=False,
            training_data_bool=False,
        )
    )

    pass


def add_hessian_data_to_model(
    gpr_hessian_model: GPModelWithHessiansWrapper,
    train_data_coordinate,
    train_pots,
    train_gradients,
    train_ab_initio_hessians,
    energy_shift,
    retrain_bool=True,
):
    """
    simple function to add data with hessian information (coordinate + pot + gradients + hessian) into the gpr_hessian_model.
    We also shift the potential energy before we add the data into gpr_hessian_model.
    We assume all data points have hessian information.
    """
    train_hessian_data_num = len(train_data_coordinate)
    new_train_x = np.copy(train_data_coordinate)
    new_train_V = np.copy(train_pots) - energy_shift
    new_train_grad_x = np.copy(train_gradients)
    new_train_hessian = np.copy(train_ab_initio_hessians)
    new_hessian_data_point_index = np.arange(train_hessian_data_num)
    gpr_hessian_model.update_model_with_new_data(
        new_train_x,
        new_train_V,
        new_train_grad_x,
        new_train_hessian,
        new_hessian_data_point_index,
        retrain_bool=retrain_bool,
    )

def add_potential_grad_data_to_model(
    gpr_hessian_model: GPModelWithHessiansWrapper,
    train_data_coordinate,
    train_pots,
    train_gradients,
    energy_shift,
    retrain_bool= True
):
    """
    simple function to add data with only potential and gradient information into the gpr_hessian_model.
    We also shift the potential energy before we add the data into gpr_hessian_model.
    we assume all data points do not have hessian information.
    """
    new_train_x = np.copy(train_data_coordinate)
    new_train_V = np.copy(train_pots) - energy_shift 
    new_train_grad_x = np.copy(train_gradients)
    new_train_hessian = np.array([])
    new_hessian_data_point_index = np.array([])

    gpr_hessian_model.update_model_with_new_data(
        new_train_x,
        new_train_V,
        new_train_grad_x,
        new_train_hessian,
        new_hessian_data_point_index,
        retrain_bool= retrain_bool
    )


def store_training_data_in_gpr_hessian_model(
    gpr_hessian_model: GPModelWithHessiansWrapper, energy_shift
):
    """
    store coordinate, potential, gradient & hessian data into folder: named by prefix.
    :param: gpr_hessian_model:  the gpr model that is capable of predicting hessian information.
    :param: prefix: the prefix of the folder.
    """
    cartesian_x = np.copy(gpr_hessian_model.train_cartesian_input)
    pots = np.copy(gpr_hessian_model.train_V) + energy_shift
    gradients = np.copy(gpr_hessian_model.train_cartesian_gradient)
    forces = -gradients

    hessian_index_list = np.copy(
        gpr_hessian_model.training_data_hessian_data_point_index
    )
    hessians = np.copy(gpr_hessian_model.train_cartesian_hessian)

    # prefix for the folder
    gradients_num = len(gradients)
    hessians_num = len(hessian_index_list)

    prefix = "grad# " + str(gradients_num) + " hessian# " + str(hessians_num)

    store_training_data_with_hessian(
        cartesian_x, pots, forces, hessian_index_list, hessians, prefix=prefix
    )

    return prefix

def store_training_hyperparameter_in_gpr_model(
        gpr_model: GPModelWithDerivativesWrapper,
        folder_path
):
    """
    store the hyper-parameter of the trained gpr model
    """
    file_name = "gpr.pth"
    file_path = os.path.join(folder_path, file_name)

    gpr_model.save_model(file_path)

def load_training_hyperparameter_in_gpr_model(
        gpr_model: GPModelWithDerivativesWrapper,
        folder_path
):
    """
    load the hyper-parameter of the trained gpr model
    """
    file_name = "gpr.pth"
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        gpr_model.load_model(file_path)
        model_hyperparameter_exists = True 
    else:
        print(f"The file {file_path} does not exists. Can not load hyper-parameter for gpr model. Need to train it.")
        model_hyperparameter_exists = False 

    return model_hyperparameter_exists

def store_training_hyperparameter_in_gpr_hessian_model(
        gpr_hessian_model: GPModelWithHessiansWrapper,
        folder_path
):
    """
    store the hyper-parameter of the trained gpr model 
    """
    file_name = "gpr_hessian.pth"
    file_path = os.path.join(folder_path, file_name)

    gpr_hessian_model.save_model(file_path)

def load_training_hyperparameter_for_gpr_hessian_model(
        gpr_hessian_model: GPModelWithHessiansWrapper,
        folder_path
):
    """
    load the hyper-parameter of the trained gpr model.
    """
    file_name = "gpr_hessian.pth"
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        gpr_hessian_model.load_model(file_path)
        model_hyperparameter_exists = True
    else:
        print(f"The file {file_path} does not exist. Can not load hyper-parameter for gpr hessian model. Need to train it.")
        model_hyperparameter_exists = False 
    
    return model_hyperparameter_exists

def analyze_train_error(gpr_hessian_model: GPModelWithHessiansWrapper):
    """
    analyze the training error in gpr hessian model.
    """
    coord = gpr_hessian_model.train_cartesian_input 
    hessian_data_point_index = np.array(
        list(
            map(
                int, 
                gpr_hessian_model.training_data_hessian_data_point_index
                )
            )
        )

    # predict hessians.
    predicted_pots, predicted_grads, predicted_hessians, _, _, _ = gpr_hessian_model.predict_latent_function(
        coord, hessian_data_point_index, internal_coordinate_bool= False 
    )

    # ab initio training data.
    ab_initio_training_hessians = gpr_hessian_model.train_cartesian_hessian
    ab_initio_train_V = gpr_hessian_model.train_V 
    ab_initio_train_cartesian_gradient = gpr_hessian_model.train_cartesian_gradient

    if len(ab_initio_training_hessians) > 0:
        # compute the relative error in training hessian data.
        relative_hessian_error = compute_relative_matrix_error_with_frobenius_norm(
            predicted_hessians, ab_initio_training_hessians
        )

        print(f"train data: relative hessian error for ring polymer beads: {relative_hessian_error}")

    # compute the relative error in training potential
    V_error = np.abs(ab_initio_train_V - predicted_pots) / np.abs(ab_initio_train_V)
    print(f"train data: error of potential prediction: {V_error}")
    
    # compute the relative error in training grads
    df = np.linalg.norm(ab_initio_train_cartesian_gradient - predicted_grads, axis= 1)
    ab_initio_force_amplitude = np.linalg.norm(ab_initio_train_cartesian_gradient, axis= 1)
    df_error = df / ab_initio_force_amplitude

    print(f"train data: error of force prediction: {df_error}")

    pass 


def analyze_transformation_between_cartesian_coord_and_internal_coord(coord_x,
                                                                      grad_x,
                                                                      hessian_x,
                                                                      coordinate_transformer: ipi.utils.internal.internaltools.non_redundant_coordinate_transformer):
    """
    analyze the transformation of gradient and hessian between the Cartesian coordinate
    and the internal coordinate. 
    We perform the transformation of gradients and hessian from Cartesian coordinate 
    to internal coordinate, then transform back.
    We then compare the difference between the original gradient and hessian and 
    the transformed gradient and hessian.
    :param: coord_x: cartesian coordinate of points x.
    :param: grad_x: gradient of data points in cartesian coordinate.
    :param: hessian_x: the hessian of data points in cartesian coordinate.
    :param: coordinate_transformer: class object that transform gradient and hessian between cartesian coordinate and internal coordinate.
    """
    # transform the gradient and hessian from cartesian coordinate into internal coordinate.
    grad_q = coordinate_transformer.transform_cartesian_gradient_to_internal_gradient(coord_x, 
                                                                                      grad_x)
    
    hessian_q = coordinate_transformer.transform_cartesian_hessian_to_internal_hessian(coord_x, 
                                                                                       grad_x, 
                                                                                       hessian_x)

    # transform back
    back_transformed_grad_x = coordinate_transformer.transform_internal_gradient_to_cartesian_gradient(coord_x,
                                                                                                       grad_q)
    
    back_transformed_hessian_x = coordinate_transformer.transform_internal_hessian_to_cartesian_hessian(coord_x,
                                                                                                        grad_q,
                                                                                                        hessian_q)
    
    # Now test the relative difference of gradient 
    dg = np.linalg.norm(grad_x - back_transformed_grad_x, axis= 1)
    ab_initio_grad_amplitude = np.linalg.norm(grad_x, axis= 1)
    dg_error = dg / ab_initio_grad_amplitude

    print(f"error after forward & backward transform the gradient: {dg_error}")
    
    # Now test the relative 
    relative_hessian_error = compute_relative_matrix_error_with_frobenius_norm(
        hessian_x, back_transformed_hessian_x
    )

    print(f"relative hessian error after forward & backward transform of hessian: {relative_hessian_error}")


def perturb_training_point(bead_path_x: np.ndarray,
                           new_data_index: np.ndarray,
                           gpr_hessian_model: GPModelWithHessiansWrapper,
                           perturb_amplitude: np.ndarray = 0.1):
    """
    Add perturbation to the training data point.
    To avoid causing trouble to the GPR model, we do not perturb the training point along the fixed internal dofs.
    For free moving dofs, we set the perturbation amplitude dq = q_range / N, here N is number of candidate data point (N=20 by default).
    
    :param: bead_path_x: the coordinate of beads along the path.
    :param: new_data_index: the index of data point along the path that we plan to add new gradient or hessian data.
    :param: gpr_hessian_model: The gaussian process regression model that can predict hessian.
    """
    coordinate_transformer = gpr_hessian_model.coordinate_transformer
    # the total number of candidate data points
    x_ref = np.copy(bead_path_x[new_data_index])
    # transform cartesian coordinate into internal coordinate.
    bead_path_q = coordinate_transformer.get_internal_coordinate_q(bead_path_x)
    # the coordinate of data points that we are going to perturb
    q_ref = np.copy(bead_path_q[new_data_index])
    new_data_point_num = len(new_data_index)

    free_moving_dofs = gpr_hessian_model.FixingDofs.free_moving_dofs
    free_moving_ndofs = len(free_moving_dofs)

    bead_path_q_range = np.max(bead_path_q, axis= 0) - np.min(bead_path_q, axis= 0)
    dq = bead_path_q_range * perturb_amplitude 

    # perturbation for data point along free moving dofs
    q_perturb = dq[free_moving_dofs] * np.random.uniform(-1, 1, (new_data_point_num, free_moving_ndofs))
    # add perturbation
    perturbed_q = np.copy(q_ref)
    perturbed_q[:, free_moving_dofs] = q_ref[:, free_moving_dofs] + q_perturb 

    # transform back to the Cartesian coordinate.
    perturbed_x = coordinate_transformer.get_cartesian_coordinate_x(x_ref, perturbed_q, q_cutoff= pow(10.0, -4))

    return perturbed_x 