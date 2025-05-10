'''
Analyze the training and prediction of Gaussian Process Regression model. 
python gpr_analyze.py RESTART -hf <hessian_folder> -index <neb_bead_index_with_highest_energy> -f <ab_initio_hessian_file> -xf <ab_initio_hessian_coordinate_file>
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
import ipi.utils.internal.CoulombInternal
import ipi.utils.internal.ZmatrixInternal

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

    parser.add_argument(
        "-xf",
        "--test_coord_file",
        help= "path to the coordinate file. Coordinate of ab initio hessian."
    )

    parser.add_argument(
        "-i",
        "--internal_coordinate",
        default= "bond",
        help= "choice of internal coordinate (Coulomb matrix (Coulomb) or Zmatrix (bond))."
    )

    args = parser.parse_args()
    inputt = args.input
    hessian_folder = args.hessian_folder 
    bead_index = int(args.bead_index_at_ts)
    ab_initio_hessian_file = args.file
    ab_initio_hessian_coordinate_file = args.test_coord_file
    internal_coordinate = args.internal_coordinate
    return inputt, hessian_folder, ab_initio_hessian_file, ab_initio_hessian_coordinate_file, bead_index, internal_coordinate 

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

def read_gpr_model(hessian_folder, energy_shift, motion, bead_index, internal_coordinate):
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
    names = dstrip(motion.beads.names).copy().tolist()
    gpr_SE_kernel_number = motion.options["gpr_SE_kernel_number"]
    gpr_kernel_outputscale = motion.optarrays["gpr_kernel_outputscale"]
    gpr_kernel_lengthscale_ratio = motion.optarrays["gpr_kernel_lengthscale_ratio"]
    gpr_noise_std = motion.optarrays["gpr_noise_std"]
    gpr_fix_internal_dofs_bool = motion.options["gpr_fix_internal_dofs_bool"]
    gpr_fix_internal_dofs_cutoff = motion.options["gpr_fix_internal_dofs_cutoff"] 
    gpr_rigid_internal_dofs_cutoff = motion.options["gpr_rigid_internal_dofs_cutoff"]
    gpr_covar_inverse_nugget = motion.optarrays["gpr_covar_inverse_nugget"]
    ridge_regularization_alpha = motion.optarrays["ridge_regularization_alpha"]
    fix_dofs = motion.optarrays["fix_dofs"]

    # generate coordinate transformer
    neb_bead_q = motion.beads.q 
    ref_x = dstrip(neb_bead_q[bead_index]).copy() 
    if internal_coordinate == "bond":
        coordinate_transformer = ipi.utils.internal.ZmatrixInternal.non_redundant_coordinate_transformer(
            natoms, [ref_x], names
        )
    else:
        coordinate_transformer = ipi.utils.internal.CoulombInternal.non_redundant_coordinate_transformer(
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
            fix_dofs,
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
            gpr_fix_internal_dofs_cutoff= gpr_fix_internal_dofs_cutoff,
            gpr_rigid_internal_dofs_cutoff= gpr_rigid_internal_dofs_cutoff,
            singular_value_cutoff= gpr_covar_inverse_nugget,
            ridge_regularization_alpha= ridge_regularization_alpha
            )
    )

    # load hyper-parameter
    model_hyperparameter_exists = \
            ipi.utils.nebinstgprtool.load_training_hyperparameter_for_gpr_hessian_model(
                    gpr_hessian_model,
                    hessian_folder
            )

    return gpr_hessian_model, hessian_data_list, ref_x, hessian_index_list 


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

def read_xyz_file(filepath):
    '''
    parse .xyz file and read the data. For neb path file.
    10 (natom)
    cell information: CELL(abcABC):  8.182510 14.645380 13.360360 90.000000 90.000000 90.000000 cell{atomic_unit}  Traj: positions{angstrom}   Bead:       0
    H 0.010812203277898587 -0.022055074478126723 3.756470544713579e-08  (in angstrom)
    C -0.008150199299932109 1.0598052296273865 -1.7071277595125512e-09
    C 1.1724225086600273 1.800009493654883 9.6358607690528e-10
    N -1.2095666449923337 3.0876153242033966 -1.3632577538519434e-10
    O 1.2026677672180992 3.088597110655394 -4.2879931388220246e-10
    H -0.01228888563076768 3.4051293535112706 1.7510952526588003e-09
    H -2.099638894426458 3.572150862782702 -1.5734165495657747e-09
    C -1.2284687912195427 1.771845731273054 7.5123657142115e-10
    H 2.145266551789177 1.2929512365820284 -6.448242274084185e-10
    H -2.1780199232400053 1.2311657381224785 6.854082778715344e-11
    '''
    with open(filepath, "r") as f:
        lines = f.readlines()
        lines = [line.strip('\n') for line in lines]
        num_lines = len(lines)
        natom = int(re.split(' ', lines[0].strip())[0])  # number of atoms
        nbeads = int(num_lines / (natom + 2))  # number of beads

        beads_q = []
        line_index = 0
        for bead_index in range(nbeads):
            line_index = line_index + 1  #  line records # of atom
            # Cell information line.
            line_index = line_index + 1
            
            # read atoms coordinate 
            q_list = []
            atom_names = []
            for atom_index in range(natom):
                line = re.split(' ', lines[line_index].strip())
                line = [ele for ele in line if ele != '']
                atom_names = atom_names + [line[0]]

                q = list(map(float, line[1:])) # coordinate
                q_list = q_list + q 

                line_index = line_index + 1 
            
            beads_q.append(q_list)
        
        beads_q = np.array(beads_q)

        return atom_names, beads_q, nbeads 
     

def transform_ab_initio_hessian(hessian_data, natoms, nbeads):
    """
    transform the hessian from 1d to shape [3 * natoms, nbeads * 3 * natoms],
    then to [nbeads, 3 * natom, 3* natom]
    """
    assert len(hessian_data) == np.power(3 * natoms, 2) * nbeads, "the length of hessian does not match nbeads and natoms" 
    hessian_data = np.reshape(hessian_data, (3 * natoms, nbeads, 3 * natoms))
    hessian_data = np.transpose(hessian_data, (1, 0, 2))

    return hessian_data 

def predict_ring_polymer_hessians(gpr_hessian_model:ipi.utils.gpr_hessian_tools.GPModelWithHessiansWrapper,
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


def analyze_train_error(gpr_hessian_model: ipi.utils.gpr_hessian_tools.GPModelWithHessiansWrapper):
    """
    analyze the training error in gpr hessian model
    """
    coord = gpr_hessian_model.train_cartesian_input 
    hessian_data_point_index = gpr_hessian_model.training_data_hessian_data_point_index

    # predict hessians.
    predicted_pots, predicted_grads, predicted_hessians, _, _, _ = gpr_hessian_model.predict_latent_function(
        coord, hessian_data_point_index, internal_coordinate_bool= False 
    )

    # ab initio training data.
    ab_initio_training_hessians = gpr_hessian_model.train_cartesian_hessian
    ab_initio_train_V = gpr_hessian_model.train_V 
    ab_initio_train_cartesian_gradient = gpr_hessian_model.train_cartesian_gradient

    # compute the relative error in training hessian data.
    relative_hessian_error = ipi.utils.nebinstgprtool.compute_relative_matrix_error_with_frobenius_norm(
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

def analyze_test_error(gpr_hessian_model,
                       motion,
                       ab_initio_hessian_coordinate_file,
                       ab_initio_hessian_file):
    """
    analyze error in the test data.
    """
    atom_names, ab_initio_data_beads_q, nbeads = read_xyz_file(ab_initio_hessian_coordinate_file)
    # the read data is in angstrom unit, so we have to convert it to atomic unit.
    angstrom_to_au = 1.8897261
    ab_initio_data_beads_q = ab_initio_data_beads_q * angstrom_to_au

    # compute the GPR_predicted hessian.
    gpr_predicted_hessians = predict_ring_polymer_hessians(gpr_hessian_model,
                            ab_initio_data_beads_q)
    
    # read ab initio test hessian data from the file.
    ab_initio_hessian_data_1d = read_ab_initio_hessian_from_file(ab_initio_hessian_file)
    natoms = motion.beads.natoms
    ab_initio_hessian_data = transform_ab_initio_hessian(ab_initio_hessian_data_1d, natoms, nbeads)

    # measure the distance between hessian
    relative_hessian_error = ipi.utils.nebinstgprtool.compute_relative_matrix_error_with_frobenius_norm(
        gpr_predicted_hessians, ab_initio_hessian_data
    ) 
    
    print(f"test data: relative hessian error for ring polymer beads: {relative_hessian_error}")

    pass

def analyze_gpr_predicted_hessian():
    """
    compute hessians of ring polymer beads
    """
    inputt, hessian_folder, ab_initio_hessian_file, ab_initio_hessian_coordinate_file, bead_index, internal_coordinate  = parse_input()

    energy_shift, rp_beads_q, motion = read_instanton_data(inputt)

    gpr_hessian_model, training_hessians, ref_x, hessian_index_list = read_gpr_model(hessian_folder, energy_shift, motion, bead_index,
                                                                                     internal_coordinate)
    
    # analyze the training error.
    analyze_train_error(
        gpr_hessian_model
    )


    analyze_test_error(gpr_hessian_model,
                       motion,
                       ab_initio_hessian_coordinate_file,
                       ab_initio_hessian_file)

    pass 

analyze_gpr_predicted_hessian()

