from ipi.engine.motion import Motion 
from ipi.utils.depend import dstrip
import numpy as np 
import ipi.utils.nebinstgprtool
import ipi.utils.nebinstool
import re
import os 
from gpr.gprtools import GPModelWithDerivativesWrapper

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
