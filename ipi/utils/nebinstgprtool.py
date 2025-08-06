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

def extract_number_from_line(line):
    line = re.split(" ", line.strip())
    line = [ele for ele in line if ele != ""]

    return line

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

def read_cubic_spline_hessian_data(prefix):
    """
    read x, hessian for cubic spline interpolation.
    The data storage format is compatible with hessian data used in gpr modeling.
    """
    assert os.path.exists(prefix), f"the prefix folder: {prefix} should have already been created."
    
    # read hessians.
    hessian_file_path = os.path.join(prefix, "training_data.h5")
    assert os.path.exists(hessian_file_path), "hessian data file does not exist."
    with h5py.File(hessian_file_path, "r") as h5f:
        hessian_data_list = np.array(h5f["hessians"])

    # read candidate hessian data points' coordinate.
    candidate_x_file_path = os.path.join(prefix, "candidate_hessian_data_info.h5")
    assert os.path.exists(candidate_x_file_path), "candidate hessian data point file does not exist."
    with h5py.File(candidate_x_file_path, "r") as h5f:
        candidate_hessian_point_x = np.array(h5f["candidate_hessian_point_x"])
    
    # read the index of hessian data point in candidate list.
    hessian_index_file_name = os.path.join(
        prefix, "hessian_index_in_candidate_point_list.txt"
    )
    assert os.path.exists(hessian_index_file_name), "hessian index file does not exist."
    with open(hessian_index_file_name, "r") as f:
        lines = f.readlines()
        line = extract_number_from_line(lines[1])
        hessian_index_in_candidate_list = np.array(list(map(int, line)))

    return candidate_hessian_point_x, hessian_index_in_candidate_list, hessian_data_list

def store_cubic_spline_hessian_data(prefix,
                                    candidate_hessian_point_x,
                                    hessian_index_in_candidate_list,
                                    hessian_data_list):
    """
    store x, hessian for cubic spline interpolation.
    The data storage format is compatible with hessian data used in gpr modeling.
    """
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    # store hessians.
    hessian_file_path = os.path.join(prefix, "training_data.h5")
    with h5py.File(hessian_file_path, "w") as h5f:
        h5f.create_dataset("hessians", data= hessian_data_list)
    
    # store candidate hessian data points' coordinate.
    candidate_x_file_path = os.path.join(prefix, "candidate_hessian_data_info.h5")
    with h5py.File(candidate_x_file_path, "w") as h5f:
        h5f.create_dataset("candidate_hessian_point_x", data= candidate_hessian_point_x)
    
    # write the index of data point that we have already computed hessian information.
    # we want this data in human readable format
    hessian_index_file_name = os.path.join(
        prefix, "hessian_index_in_candidate_point_list.txt"
    )
    hessian_data_number = len(hessian_index_in_candidate_list)
    with open(hessian_index_file_name, "w") as f:
        f.write("Index for data point that we have computed hessians. \n")
        indices = [str(int(hessian_index_in_candidate_list[i])) for i in range(hessian_data_number)]
        f.write(" ".join(indices) + "\n")
    

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