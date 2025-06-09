"""
utility module for neb_instanton_gpr.py module.
Written by Chenghao Zhang & Niri Govind, Pacific Northwest National Laboratory (chenghao.zhang@pnnl.gov), 2024.
"""

import numpy as np
from gpr.gprtools import GPModelWithDerivativesWrapper
import re
import os
from ipi.utils.nebinstool import RK4
import ipi.utils.nebinstool
from ipi.utils.depend import dstrip
from gpr.gpr_hessian_tools import GPModelWithHessiansWrapper
# import ipi.utils.internalcoordtools
import gpr.internal.ZmatrixInternal
import shutil
from collections import namedtuple
import h5py

def check_neb_early_stop(
    beads_x,
    trust_region_distance,
    gpr_model: GPModelWithDerivativesWrapper,
    outerloop_step,
    inner_loop_neb_step,
    m3
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

    # #For dbebug
    # gpr_training_data_cartesian_coordinate = np.copy(gpr_model.train_cartesian_inputs)
    # ref_Bq = gpr_model.coordinate_transformer.compute_delocalized_wilson_matrix_Bq(
    #     np.array([gpr_model.coordinate_transformer.ref_x])
    # )[0]
    # # Vt is eigenvector for internal coordinate.
    # U, S, Vt = np.linalg.svd(ref_Bq, full_matrices= False)
    # free_moving_internal_dofs = gpr_model.FixingDofs.free_moving_dofs 

    # compute the distance and find beads that move out of the trusted region.
    nbeads = np.shape(beads_x)[0]
    internal_coordinate_closest_r_list = []
    for bead_index in range(nbeads):
        bead_internal_q = beads_free_moving_internal_coordinate[bead_index]

        # distance between gpr training data and beads.
        internal_coordinate_diff =  (
            (
                bead_internal_q[np.newaxis, :]
                - gpr_training_free_moving_internal_coordinate
            )
            / effective_kernel_length_scale)

        internal_coordinate_r = np.linalg.norm(internal_coordinate_diff, axis= 1)

        nearest_gpr_data_index = np.argmin(internal_coordinate_r)

        # distance r between beads and the closest gpr training data
        internal_coordinate_closest_r = internal_coordinate_r[nearest_gpr_data_index]

    #    # Cartesian coordinate
    #     cartesian_coordinate_diff = (
    #         beads_x[bead_index] - gpr_training_data_cartesian_coordinate
    #     )[nearest_gpr_data_index] 
    #     dalton_unit = ipi.utils.units.UnitMap["mass"]["dalton"]
    #     # atom mass in dalton unit
    #     atom_mass = m3[0] / dalton_unit
    #     # mass in dalton unit, length in angstrom unit. 
    #     mass_scaled_cartesian_coordinate_diff = (np.sqrt(atom_mass) * cartesian_coordinate_diff 
    #                                              / ipi.utils.units.UnitMap["length"]["angstrom"])
    #     mass_scaled_cartesian_distance = np.linalg.norm(mass_scaled_cartesian_coordinate_diff)

        internal_coordinate_closest_r_list.append(internal_coordinate_closest_r)
        if internal_coordinate_closest_r > distance_cutoff:
            early_stop_bool = True
            out_range_bead_index_list.append(bead_index)

            # # analyze the internal coordinate that causes early stop.
            # internal_coordinate_diff_closest_point = internal_coordinate_diff[nearest_gpr_data_index]
            # free_moving_internal_coordinate_index_sorted = np.argsort(- np.abs(internal_coordinate_diff_closest_point))
            # internal_coordinate_index_sorted = free_moving_internal_dofs[free_moving_internal_coordinate_index_sorted]
            # print(f"outrange bead index: {bead_index}. internal coordinate that causes early stop: {internal_coordinate_index_sorted[:5]}." 
            #       f"internal coordinate distance: {internal_coordinate_diff_closest_point[free_moving_internal_coordinate_index_sorted[:5]]}"
            #       f"mass scaled distance in cartesian coordinate: {mass_scaled_cartesian_distance}")
            
            # eigenvector = Vt[internal_coordinate_index_sorted[0]]
            # cartesian_coordinate_index_sorted = np.argsort(- np.abs(eigenvector))
            # pass 

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
    training_bead_number, ndofs = cartesian_coordinate_x.shape 
    # create folder with prefix
    backup_prefix = "#" + prefix 
    if os.path.exists(backup_prefix):
        shutil.rmtree(backup_prefix)
    if os.path.exists(prefix):
        shutil.move(prefix, backup_prefix)
    os.mkdir(prefix)

    # use HDF5 file store data
    h5_file_path = os.path.join(prefix, "training_data.h5")

    # write data to hdf5 file 
    with h5py.File(h5_file_path, "w") as h5f:
        h5f.create_dataset("cartesian_coordinate_x", data= cartesian_coordinate_x, compression= 'gzip')
        h5f.create_dataset("V", data= V, compression= "gzip")
        h5f.create_dataset("forces", data= forces, compression= "gzip")
        h5f.attrs["training_bead_number"] = training_bead_number
        h5f.attrs["ndofs"] = ndofs
    
    print(f"Training data successfully stored in {h5_file_path}")


def store_training_data_with_hessian(
    cartesian_coordinate_x, V, forces, hessian_index_list, hessians, prefix
):
    """
    store the training data (coord, pot, grad) + hessian
    """
    store_training_data(cartesian_coordinate_x, V, forces, prefix)
    
    # use HDF5 file store data
    h5_file_path = os.path.join(prefix, "training_data.h5")
    with h5py.File(h5_file_path, "a") as h5f:
        # the index of data point that stores hessian information.
        h5f.create_dataset("hessian_index_list", data= hessian_index_list, compression= "gzip")
        h5f.create_dataset("hessians", data= hessians, compression= "gzip")


def read_training_data(prefix):
    """
    read coordinate, potential V and force f for training data.
    """
    # read data from use HDF5 file 
    h5_file_path = os.path.join(prefix, "training_data.h5")
    assert os.path.exists(h5_file_path), "training data (training_data.h5) for gpr model does not exist."
    
    with h5py.File(h5_file_path, "r") as h5f:
        cartesian_coordinate_x = np.array(h5f["cartesian_coordinate_x"])
        training_V = np.array(h5f["V"])
        training_forces = np.array(h5f["forces"])

    return cartesian_coordinate_x, training_V, training_forces

def read_hessian_data(prefix):
    """
    """
    h5_file_path = os.path.join(prefix, "training_data.h5")

    with h5py.File(h5_file_path, "r") as h5f:
        # the index of data point that contains hessian information.
        hessian_index_list = np.array(h5f["hessian_index_list"])
        hessian_data_list = np.array(h5f["hessians"])

    return hessian_index_list, hessian_data_list 

def read_training_data_with_hessian(prefix):
    """
    read coordinate, potential V, force f and hessian h from training data
    """
    cartesian_coordinate_x, training_V, training_forces = read_training_data(prefix)
    
    # read hessian
    hessian_index_list, hessian_data_list = read_hessian_data(prefix)

    return (
        cartesian_coordinate_x,
        training_V,
        training_forces,
        hessian_index_list,
        hessian_data_list,
    )

def store_fixed_internal_dofs_gpr_model(gpr_model: GPModelWithDerivativesWrapper,
                              prefix):
    """
    store fixed internal dofs in gpr_hessian_model
    """
    fixed_internal_dofs = gpr_model.output_fixed_internal_dofs()
    
    file_path = os.path.join(prefix, "fixed_internal_dofs.txt")
    with open(file_path, "w") as f:
        for dof in fixed_internal_dofs:
            f.write(str(dof) + " ")
        f.write('\n')

def store_fixed_internal_dofs_gpr_hessian_model(gpr_hessian_model: GPModelWithHessiansWrapper,
                              prefix):
    """
    store fixed internal dofs in gpr_hessian_model
    """
    fixed_internal_dofs = gpr_hessian_model.output_fixed_internal_dofs()
    
    file_path = os.path.join(prefix, "fixed_internal_dofs.txt")
    with open(file_path, "w") as f:
        for dof in fixed_internal_dofs:
            f.write(str(dof) + " ")
        f.write('\n')
    
def store_rigid_internal_dofs_gpr_hessian_model(gpr_hessian_model: GPModelWithHessiansWrapper,
                                                prefix):
    """
    store rigid internal dofs in gpr hessian model.
    """
    rigid_internal_dofs = gpr_hessian_model.output_rigid_internal_dofs()

    file_path = os.path.join(prefix, "rigid_internal_dofs.txt")
    with open(file_path, "w") as f:
        for dof in rigid_internal_dofs:
            f.write(str(dof) + " ")
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
    candidate_point_number, ndofs = candidate_hessian_point_x.shape 

    h5_file_path = os.path.join(prefix, "candidate_hessian_data_info.h5")
    with h5py.File(h5_file_path, "w") as h5f:
        h5f.create_dataset("candidate_hessian_point_x", 
                           data= candidate_hessian_point_x, 
                           compression= "gzip")
        
        h5f.attrs["candidate_point_number"] = candidate_point_number
    
    # write the index of data point that we have already computed hessian information.
    # we want this data in readable format
    hessian_index_file_name = os.path.join(
        prefix, "hessian_index_in_candidate_point_list.txt"
    )
    used_hessian_point_num = len(used_hessian_index_in_candidate_list)
    with open(hessian_index_file_name, "w") as f:
        f.write("Index for data point that we have computed hessians. \n")
        indices = [str(int(used_hessian_index_in_candidate_list[i])) for i in range(used_hessian_point_num)]
        f.write(" ".join(indices) + "\n")
        f.write("\n")

def store_candidate_grad_data_coordinate(
        candidate_grad_point_x, used_grad_data_index, prefix
):
    """
    store the information about which data point we have used gradient in gpr model and
    what are the potential (candidate) data points we can compute gradients and add to gpr model.
    :param: candidate_grad_point_x: coordinate for candidate points that we can compute gradients.
    :param: used_grad_index: the index of data points that we have already computed gradients.
    :param: prefix: name of folders that we will store info
    """
    assert os.path.exists(prefix), "the prefix folder should have already been created."
    candidate_point_number, ndofs = candidate_grad_point_x.shape 

    h5_file_path = os.path.join(prefix, "candidate_grad_data_info.h5")
    with h5py.File(h5_file_path, "w") as h5f:
        h5f.create_dataset("candidate_grad_point_x", 
                           data= candidate_grad_point_x, 
                           compression= "gzip")
        
        h5f.attrs["candidate_point_number"] = candidate_point_number
    
    # write the index of gradient data point that we have already computed gradient information.
    grad_index_file_name = os.path.join(
        prefix, "grad_index_in_candidate_point_list.txt"
    )
    used_grad_point_num = len(used_grad_data_index)
    with open(grad_index_file_name, "w") as f:
        f.write("Index for data point that we have computed gradients. \n")
        f.write(" ".join(str(int(used_grad_data_index[i])) for i in range(used_grad_point_num)) + " ")
        f.write("\n")

def extract_number_from_line(line):
    line = re.split(" ", line.strip())
    line = [ele for ele in line if ele != ""]

    return line

def read_fixed_internal_dofs(prefix):
    """
    read fixed internal dofs from file.
    If file exists, read the data.
    else: return None.
    """
    file_path = os.path.join(prefix, "fixed_internal_dofs.txt")
    
    fixed_internal_dofs = None
    if os.path.exists(file_path):
        with open (file_path, "r") as f:
            lines = f.readlines()
            fixed_internal_dofs = extract_number_from_line(lines[0])
            fixed_internal_dofs = np.array(list(map(int, fixed_internal_dofs)))
    
    return fixed_internal_dofs

def read_rigid_internal_dofs(prefix):
    """
    read rigid internal dofs from file.
    If file exists, read the data.
    else: return None.
    """
    file_path = os.path.join(prefix, "rigid_internal_dofs.txt")

    rigid_internal_dofs = None 
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
            rigid_internal_dofs = np.array(extract_number_from_line(lines[0])).astype(int)

    return rigid_internal_dofs
    

def read_candidate_hessian_data_coordinate(prefix):
    """
    read the information about which data point we have used hessian in gpr model and
    what are the potential (candidate) data points we can compute hessians and add to gpr model.
    :param: prefix: name of folders that we will load info
    """
    assert os.path.exists(prefix), "the prefix folder should have already been created."

    # read candidate hessian coordinate
    h5_file_path = os.path.join(prefix, "candidate_hessian_data_info.h5")
    assert os.path.exists(h5_file_path), "candidate hessian data point file does not exist"

    with h5py.File(h5_file_path, "r") as h5f:
        candidate_hessian_point_x = np.array(h5f["candidate_hessian_point_x"])
    
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

    h5_file_path = os.path.join(prefix, "candidate_grad_data_info.h5")
    assert os.path.exists(h5_file_path), "candidate grad data point file does not exist"
    
    with h5py.File(h5_file_path, "r") as h5f:
        candidate_grad_point_x = np.array(h5f["candidate_grad_point_x"])
    
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
        raise "the shape of matrix for computing frobenius norm has to have dimensions 2 or 3."

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
        cartesian_x, pots, forces, hessian_index_list, hessians, prefix= prefix
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

def split_train_cv_data(cartesian_coordinate_x,
                        potential_data,
                        force_data,
                        hessian_index_list,
                        hessian_data_list,
                        training_ratio= 0.6):
    """
    split the data into training set and cross validation (cv) set. 
    we only put data point with hessian information into cross validation set.
    training_ratio is ratio of training data in all data set.
    """
    if len(hessian_index_list) < 5:
        train_x = cartesian_coordinate_x
        train_pot = potential_data 
        train_force = force_data 
        train_hessian_index_list = hessian_index_list
        train_hessian_data = hessian_data_list 

        cv_x = np.array([])
        cv_pot = np.array([])
        cv_force = np.array([])
        cv_hessian_index_list = np.array([])
        cv_hessian_data = np.array([])
    else:
        hessian_data_num = len(hessian_index_list)
        cv_hessian_data_num = round((1- training_ratio) * hessian_data_num)
        train_hessian_data_num = hessian_data_num - cv_hessian_data_num

        all_hessian_index = np.copy(hessian_index_list)
        # use a seed to replicate the result.
        np.random.seed(42)
        np.random.shuffle(all_hessian_index)
        cv_hessian_index = np.sort(all_hessian_index[:cv_hessian_data_num])
        train_hessian_index = np.sort(all_hessian_index[cv_hessian_data_num:])

        hessian_index_in_cv_data_bool = [1 if hessian_index_list[i] in cv_hessian_index else 0 for i in range(hessian_data_num)]
        cv_index = np.nonzero(hessian_index_in_cv_data_bool)[0]  # index in hessian_data for cross validation hessians.
        train_index =  np.delete(np.arange(hessian_data_num), cv_index) # index in hessian data for training hessians.
        
        hessian_index_shift = np.cumsum(hessian_index_in_cv_data_bool)[train_index]

        # cross validation data
        cv_x = cartesian_coordinate_x[cv_hessian_index]
        cv_pot = potential_data[cv_hessian_index]
        cv_force = force_data[cv_hessian_index]
        cv_hessian_index_list = np.arange(cv_hessian_data_num)
        cv_hessian_data = hessian_data_list[cv_index]

        # training data.
        train_x = np.delete(cartesian_coordinate_x, cv_hessian_index, axis= 0)
        train_pot = np.delete(potential_data, cv_hessian_index)
        train_force = np.delete(force_data, cv_hessian_index, axis= 0)
        train_hessian_index_list = train_hessian_index - hessian_index_shift
        train_hessian_data = hessian_data_list[train_index]

    Dataset = namedtuple('Dataset', ['x', 'pot', 'force', 'hessian_index', 'hessian_data'])
    train_set = Dataset(train_x, train_pot, train_force, train_hessian_index_list, train_hessian_data)
    cv_set = Dataset(cv_x, cv_pot, cv_force, cv_hessian_index_list, cv_hessian_data)
    
    return train_set, cv_set 


def analyze_force_error(coord,
                        ab_initio_cartesian_gradient, 
                        predicted_cartesian_grads, 
                        gpr_hessian_model: GPModelWithHessiansWrapper,
                        data_type= "train data"):
    """
    analyze the error in force prediction.
    In internal coordinate, force along constrained dofs are predicted by Linear regression.
    force along free moving dofs are predicted by gaussian process regression.
    We analyze the error in force prediction for both of these two components.
    """
    df = np.linalg.norm(ab_initio_cartesian_gradient - predicted_cartesian_grads, axis= 1)
    ab_initio_force_amplitude = np.linalg.norm(ab_initio_cartesian_gradient, axis= 1)
    df_error = df / ab_initio_force_amplitude

    print(f"{data_type}: error of force prediction: {df_error}")

    ab_initio_grad_q = gpr_hessian_model.coordinate_transformer.transform_cartesian_gradient_to_internal_gradient(
        coord,
        ab_initio_cartesian_gradient
    )

    predicted_grad_q = gpr_hessian_model.coordinate_transformer.transform_cartesian_gradient_to_internal_gradient(
        coord,
        predicted_cartesian_grads
    )

    # error for force component that is predicted by linear regression.
    constrained_dofs = gpr_hessian_model.FixingDofs.constrained_internal_dofs

    if len(constrained_dofs) > 0:
        constrained_force_error = (np.linalg.norm(ab_initio_grad_q[:, constrained_dofs] 
                                                - predicted_grad_q[:, constrained_dofs],
                                                axis= 1) / 
                                                np.linalg.norm(ab_initio_grad_q[:, constrained_dofs], 
                                                                axis= 1)
                               )
        absolute_constrained_force_error = np.linalg.norm(ab_initio_grad_q[:, constrained_dofs] - predicted_grad_q[:, constrained_dofs],
                                                axis= 1)
        print(f"{data_type}: error in constrained internal dofs for force prediction (linear regression): {constrained_force_error}")


    # error for force component that is predicted by gaussian process regression. 
    free_moving_dofs = gpr_hessian_model.FixingDofs.free_moving_dofs
    free_moving_force_error = (np.linalg.norm(ab_initio_grad_q[:, free_moving_dofs] 
                                              - predicted_grad_q[:, free_moving_dofs], 
                                              axis= 1) / 
                                              np.linalg.norm(ab_initio_grad_q[:, free_moving_dofs],
                                                             axis= 1)
                                              )
    absolute_free_moving_force_error = np.linalg.norm(ab_initio_grad_q[:, free_moving_dofs]  - predicted_grad_q[:, free_moving_dofs], 
                                              axis= 1)
    print(f"{data_type}: error in free moving internal dofs for force prediction: (GPR model): {free_moving_force_error}")

    print("\n")
    # print(f"{data_type}: absolute error for force prediction {df}\n")
    # print(f"{data_type}: absolute error in constrained internal dofs for force prediction (linear regression): {absolute_constrained_force_error}")
    # print(f"{data_type}: absolute error in free moving internal dofs for force prediction (GPR model): {absolute_free_moving_force_error}")

    pass 

def analyze_hessian_error(coord,
                          predicted_cartesian_gradients,
                          ab_initio_cartesian_gradients,
                          hessian_data_point_index, 
                          predicted_hessians, 
                          ab_initio_hessians,
                          gpr_hessian_model: GPModelWithHessiansWrapper,
                          data_type= "train data"):
    """
    analyze error in hessian prediction.
    In internal coordinate, the hessian for free moving dofs are predicted by Gaussian Process Regression.
    the hessian for constrained internal dofs are predicted by linear regression. 
    We analyze error for both two block components.
    """
    relative_hessian_error = compute_relative_matrix_error_with_frobenius_norm(
        predicted_hessians, ab_initio_hessians
    )

    print(f"{data_type}: relative hessian error for ring polymer beads: {relative_hessian_error}")

    ab_initio_hessian_q = gpr_hessian_model.coordinate_transformer.transform_cartesian_hessian_to_internal_hessian(
            coord[hessian_data_point_index],
            ab_initio_cartesian_gradients[hessian_data_point_index], 
            ab_initio_hessians
            )
        
    predicted_hessian_q = gpr_hessian_model.coordinate_transformer.transform_cartesian_hessian_to_internal_hessian(
            coord[hessian_data_point_index], 
            predicted_cartesian_gradients[hessian_data_point_index], 
            predicted_hessians
            )
    
    # error for hessian component that is predicted by linear regression. (block diagonal term)
    constrained_dofs_2d_index = np.copy(gpr_hessian_model.FixingDofs.constrained_internal_dofs_2d_index)
    constrained_ab_initio_hessian_q = ab_initio_hessian_q[:,
                                                          constrained_dofs_2d_index[0], 
                                                            constrained_dofs_2d_index[1]]
    constrained_predicted_hessian_q = predicted_hessian_q[:,
                                                          constrained_dofs_2d_index[0],
                                                            constrained_dofs_2d_index[1]]
    constrained_hessian_error = compute_relative_matrix_error_with_frobenius_norm(
        constrained_predicted_hessian_q, 
        constrained_ab_initio_hessian_q
    )
    print(f"{data_type}: relative hessian error for constrained dofs (modeled by linear regression): {constrained_hessian_error}")

    # error for hessian component that is predicted by linear regression (cross term between rigid mode and flexible mode)
    cross_term_2d_index = np.copy(gpr_hessian_model.FixingDofs.cross_term_2d_index)
    cross_term_ab_initio_hessian_q = ab_initio_hessian_q[:,
                                                         cross_term_2d_index[0],
                                                         cross_term_2d_index[1]]
    cross_term_predicted_hessian_q = predicted_hessian_q[:,
                                                         cross_term_2d_index[0],
                                                         cross_term_2d_index[1]]
    cross_term_hessian_error = compute_relative_matrix_error_with_frobenius_norm(
        cross_term_predicted_hessian_q,
        cross_term_ab_initio_hessian_q
    )
    print(f"{data_type}: relative hessian error for cross term (modeled by linear regression): {cross_term_hessian_error}")

    # error for hessian component that is predicted by linear regression.
    free_moving_dofs_2d_index = gpr_hessian_model.FixingDofs.free_moving_dofs_2d_index
    free_moving_ab_initio_hessian_q = ab_initio_hessian_q[:,
                                                          free_moving_dofs_2d_index[0],
                                                          free_moving_dofs_2d_index[1]]
    free_moving_predicted_hessian_q = predicted_hessian_q[:,
                                                          free_moving_dofs_2d_index[0],
                                                          free_moving_dofs_2d_index[1]]
    free_moving_hessian_error = compute_relative_matrix_error_with_frobenius_norm(
        free_moving_predicted_hessian_q,
        free_moving_ab_initio_hessian_q
    )
    print(f"{data_type}: relative hessian error for free moving dofs of ring polymers beads \
           (modeled by Gaussian Process Regression): {free_moving_hessian_error}")
    
    pass


def analyze_train_error(gpr_hessian_model: GPModelWithHessiansWrapper):
    """
    analyze the training error in gpr hessian model.
    """
    coord = gpr_hessian_model.train_cartesian_input 
    hessian_data_point_index = np.array(
        gpr_hessian_model.training_data_hessian_data_point_index
        ).astype(int)

    # predict hessians.
    predicted_pots, predicted_cartesian_grads, predicted_hessians, _, _, _ = gpr_hessian_model.predict_latent_function(
        coord, hessian_data_point_index, internal_coordinate_bool= False 
    )

    # ab initio training data.
    ab_initio_training_hessians = gpr_hessian_model.train_cartesian_hessian
    ab_initio_train_V = gpr_hessian_model.train_V 
    ab_initio_train_cartesian_grads = gpr_hessian_model.train_cartesian_gradient

    # compute the relative error in training potential
    V_error = np.abs(ab_initio_train_V - predicted_pots) / np.abs(ab_initio_train_V)
    print(f"train data: error of potential prediction: {V_error}")
    print("\n")

    # compute the relative error in training grads
    analyze_force_error(coord,
                        ab_initio_train_cartesian_grads,
                        predicted_cartesian_grads,
                        gpr_hessian_model,
                        data_type= "train data")
    for _ in range(2):
        print("\n")
    # compute the relative error in training hessian data.
    if len(ab_initio_training_hessians) > 0:
        analyze_hessian_error(coord,
                              predicted_cartesian_grads,
                              ab_initio_train_cartesian_grads,
                              hessian_data_point_index,
                              predicted_hessians,
                              ab_initio_training_hessians,
                              gpr_hessian_model)

        for _ in range(2):
            print("\n")
    pass 

def analyze_cross_validation_error(gpr_hessian_model: GPModelWithHessiansWrapper,
                                   cv_coord,
                                   cv_pots,
                                   cv_ab_initio_grads,
                                   cv_hessian_data_point_index,
                                   cv_ab_initio_hessians):
    """
    Test the performance of gpr_hessian_model on the cross validation data.
    :param: cv_coord: coordinate for cross-validation data.
    :param: cv_hessian_data_point_index: hessian data point in cross validation data.
    :param: cv_ab_initio_gradient: ab initio gradient for cross validation data.
    :param: cv_ab_initio_hessian: hessian for cross validation data.
    """
    cv_hessian_data_point_index = np.array(cv_hessian_data_point_index).astype(int)

    # predict potential, gradients, hessians in cartesian coordinate.
    predicted_pots, predicted_cartesian_grads, predicted_cartesian_hessians, _, _, _ = (
        gpr_hessian_model.predict_latent_function(
            cv_coord,
            cv_hessian_data_point_index,
            internal_coordinate_bool= False
        )
    )

    # compute relative error in potential for cross validation data.
    V_error = np.abs(cv_pots - predicted_pots) / np.abs(cv_pots)
    print(f"cross validation data: error of potential prediction {V_error}")
    print("\n")
    
    # compute the relative error in cross validation gradient:
    analyze_force_error(
        cv_coord,
        cv_ab_initio_grads,
        predicted_cartesian_grads,
        gpr_hessian_model,
        data_type= "cross validation data"
    )
    for _ in range(2):
        print("\n")
    # compute the relative hessian error in cross validation gradient.
    if len(cv_ab_initio_hessians) > 0:
        analyze_hessian_error(
            cv_coord,
            predicted_cartesian_grads,
            cv_ab_initio_grads,
            cv_hessian_data_point_index,
            predicted_cartesian_hessians,
            cv_ab_initio_hessians,
            gpr_hessian_model,
            data_type= "cross validation data"
        )

        for _ in range(2):
            print("\n")

    pass

def analyze_transformation_between_cartesian_coord_and_internal_coord(coord_x,
                                                                      grad_x,
                                                                      hessian_x,
                                                                      coordinate_transformer: gpr.internal.ZmatrixInternal.non_redundant_coordinate_transformer):
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

def compute_path_tangent_vector(bead_path: np.ndarray, fixed_dofs):
    """
    compute the tangent vector along the path.
    """
    bead_number = np.shape(bead_path)[0]
    tangent_vector = np.zeros(bead_path.shape)
    
    tangent_vector[0] = bead_path[1] - bead_path[0]
    tangent_vector[0, fixed_dofs] = 0
    tangent_vector[0] = tangent_vector[0] / np.linalg.norm(tangent_vector[0])

    tangent_vector[-1] = bead_path[-1] - bead_path[-2]
    tangent_vector[-1, fixed_dofs] = 0
    tangent_vector[-1] = tangent_vector[-1] / np.linalg.norm(tangent_vector[-1])

    for index in range(1, bead_number - 1):
        tangent_vector[index] = bead_path[index + 1] - bead_path[index - 1]
        tangent_vector[index, fixed_dofs] = 0
        tangent_vector[index] = tangent_vector[index] / np.linalg.norm(tangent_vector[index])
    
    return tangent_vector