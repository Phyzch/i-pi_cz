'''
Analyze the training and prediction of Gaussian Process Regression model. 
python gpr_analyze.py RESTART -hf <hessian_folder> -index <neb_bead_index_with_highest_energy> -f <ab_initio_hessian_file> 
'''
import numpy as np
import sys
import re 

import argparse
from ipi.utils.messages import verbosity, info

# Chenghao Zhang 2024.
# Read gpr model and parameter & then predict hessian.
import ipi.utils.nebinstgprtool
import ipi.utils.gpr_hessian_tools
from ipi.engine.simulation import Simulation
from ipi.utils.internalcoordtools import non_redundant_coordinate_transformer
from ipi.utils.depend import dstrip

def parse_input():
    # INPUT
    parser = argparse.ArgumentParser(
        description= """
        File that read the gaussian process regression model.
        """
    )

    parser.add_argument(
        "input",
        help="RESTART file"
    )

    parser.add_argument(
        "-hf",
        "--hessian_folder",
        help= "folder that contains gradient and hessian information."
    )

    parser.add_argument(
        "-index",
        "--bead_index_at_ts",
        help= "bead index that has the highest energy. Used for construct coordinate transformer."
    )

    parser.add_argument(
        "-f",
        "--file",
        help= "path to the ab initio hessian file"
    )

    args = parser.parse_args()
    inputt = args.input
    hessian_folder = args.hessian_folder 
    bead_index = int(args.bead_index_at_ts)
    ab_initio_hessian_file = args.file

    return inputt, hessian_folder, ab_initio_hessian_file, bead_index 

def read_instanton_data(inputt):
    """
    read the neb path and energy_shift & other data from RESTART file
    """
    simulation= Simulation.load_from_xml(
        inputt, custom_verbosity= "quiet", request_banner= False, read_only= True
    )

    energy_shift = simulation.syslist[0].motion.optarrays["energy_shift"]
    rp_beads_q = simulation.syslist[0].motion.optarrays["instanton_bead_q"]

    motion = simulation.syslist[0].motion 

    return energy_shift, rp_beads_q, motion 

def read_gpr_model(hessian_folder, energy_shift, motion, bead_index):
    """
    read and construct gaussian process regression model
    """
    (
        cartesian_coordinate_x,
        training_V,
        training_forces,
        hessian_index_list,
        hessian_data_list,
    ) = ipi.utils.nebinstgprtool.read_training_data_with_hessian(
        hessian_folder
    )

    training_V_shifted = training_V - energy_shift
    training_grads = -training_forces
    # choose the first data point with hessian information as the reference point for mean function.
    ref_x = cartesian_coordinate_x[hessian_index_list[0]]
    ref_V_shifted = np.array([training_V_shifted[hessian_index_list[0]]])
    ref_grads = training_grads[hessian_index_list[0]]
    ref_hessians = hessian_data_list[0]

    # read parameter from RESTART file.
    natoms = motion.beads.natoms
    gpr_SE_kernel_number = motion.options["gpr_SE_kernel_number"]
    gpr_kernel_outputscale = motion.optarrays["gpr_kernel_outputscale"]
    gpr_kernel_lengthscale_ratio = motion.optarrays["gpr_kernel_lengthscale_ratio"]
    gpr_noise_std = motion.optarrays["gpr_noise_std"]
    gpr_fix_internal_dofs_bool = motion.options["gpr_fix_internal_dofs_bool"]
    gpr_fix_internal_dofs_cutoff = motion.options["gpr_fix_internal_dofs_cutoff"] 

    # generate coordinate transformer
    neb_bead_q = motion.beads.q 
    ref_x = dstrip(neb_bead_q[bead_index]).copy() 
    coordinate_transformer = non_redundant_coordinate_transformer(
            natoms, ref_x
    )

    # construct the gaussian process regression model.
    gpr_hessian_model = (
        ipi.utils.gpr_hessian_tools.GPModelWithHessiansWrapper(
            cartesian_coordinate_x,
            training_V_shifted,
            training_grads,
            hessian_data_list,
            hessian_index_list,
            natoms,
            coordinate_transformer,
            gpr_SE_kernel_number,
            gpr_kernel_outputscale,
            gpr_kernel_lengthscale_ratio,
            gpr_noise_std,
            constant_mean_func_bool=False,
            ref_mean_x=ref_x,
            ref_mean_V=ref_V_shifted,
            ref_mean_grad_x=ref_grads,
            ref_mean_hessian_x=ref_hessians,
            train_bool= False,
            gpr_fix_internal_dofs_bool= gpr_fix_internal_dofs_bool,
            gpr_fix_internal_dofs_cutoff= gpr_fix_internal_dofs_cutoff
            )
    )

    # load hyper-parameter
    model_hyperparameter_exists = \
            ipi.utils.nebinstgprtool.load_training_hyperparameter_for_gpr_hessian_model(
                    gpr_hessian_model,
                    hessian_folder
            )

    return gpr_hessian_model, hessian_data_list, ref_x, hessian_index_list 

def predict_hessians(gpr_hessian_model:ipi.utils.gpr_hessian_tools.GPModelWithHessiansWrapper,
                     rp_beads_q):
    """
    predict the hessian of ring polymer beads.
    """
    coord = np.copy(rp_beads_q)
    nbeads = np.shape(rp_beads_q)[0]

    hessian_data_point_index = np.arange(nbeads)
    pots, grads, hessians, _, _, _ = gpr_hessian_model.predict_latent_function(
            coord, hessian_data_point_index, internal_coordinate_bool=False
        )
    
    return hessians

def extract_number_from_line(line):
    line = re.split(" ", line.strip())
    line = [ele for ele in line if ele != ""]

    return line


def read_ab_initio_hessian_from_file(hessian_file_path):
    """
    Read the hessian from the file.
    """
    with open(hessian_file_path, "r") as f:
        lines = f.readlines() 
        line = extract_number_from_line(lines[0])

        hessian_data = np.array(list(map(float, line)))

        return hessian_data 

def transform_ab_initio_hessian(hessian_data, natoms, nbeads):
    """
    transform the hessian from 1d to shape [3 * natoms, nbeads * 3 * natoms],
    then to [nbeads, 3 * natom, 3* natom]
    """
    assert len(hessian_data) == np.power(3 * natoms, 2) * nbeads, "the length of hessian does not match nbeads and natoms" 
    hessian_data = np.reshape(hessian_data, (3 * natoms, nbeads, 3 * natoms))
    hessian_data = np.transpose(hessian_data, (1, 0, 2))

    return hessian_data 

def analyze_gpr_predicted_hessian():
    """
    compute hessians of ring polymer beads
    """
    inputt, hessian_folder, ab_initio_hessian_file, bead_index  = parse_input()

    energy_shift, rp_beads_q, motion = read_instanton_data(inputt)

    gpr_hessian_model, training_hessians, ref_x, hessian_index_list = read_gpr_model(hessian_folder, energy_shift, motion, bead_index)
    
    hessians = predict_hessians(gpr_hessian_model,
                               rp_beads_q)
    
    # read ab initio hessian file
    ab_initio_hessian_data_1d = read_ab_initio_hessian_from_file(ab_initio_hessian_file)
    natoms = motion.beads.natoms
    nbeads = rp_beads_q.shape[0]
    ab_initio_hessian_data = transform_ab_initio_hessian(ab_initio_hessian_data_1d, natoms, nbeads)

    # measure the distance between hessian
    relative_hessian_error = ipi.utils.nebinstgprtool.compute_relative_matrix_error_with_frobenius_norm(
        hessians, ab_initio_hessian_data
    ) 
    
    print(f"relative hessian error for ring polymer beads: {relative_hessian_error}")

    pass 

analyze_gpr_predicted_hessian()

