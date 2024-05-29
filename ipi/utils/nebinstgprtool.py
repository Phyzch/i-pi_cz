'''
utility module for neb_instanton_gpr.py module
'''
import numpy as np 
from ipi.utils.internalcoordtools import non_redundant_coordinate_transformer
from ipi.utils.gprtools import GPModelWithDerivativesWrapper
from ipi.engine.beads import Beads
from ipi.utils.depend import dstrip

def check_neb_early_stop(beads_x, gpr_model: GPModelWithDerivativesWrapper):
    '''
    check early stoage criterion for neb algorithm with machine learning.
    If the bead move out of trusted region, then we stop the current move.
    This means if the distance between beads and nearest gpr point in internal coordinate "q" exceeds 2 * sigma, 
    (where sigma is the length scale of gpr kernel), then the bead is out of trusted region.

    :param: beads_x: cartesian coordinate X of neb beads 
    :param: gpr_model: model to perform the Gaussian Process Regression.

    :return: early_stop_bool: bool variable to indicate whether there is bead out of trust region.
             out_range_bead_index: the bead index that move out of the trusted region.
    '''
    early_stop_bool = False 
    out_range_bead_index = -1
    
    coordinate_transformer = gpr_model.coordinate_transformer

    # the location of current beads in internal coordinate
    beads_internal_coordinate = coordinate_transformer.get_internal_coordinate_q(np.copy(beads_x))
    neb_path_internal_coordinate_length = np.sum(np.linalg.norm(beads_internal_coordinate[1:] - beads_internal_coordinate[:-1], axis = 1))

    # distance cutoff for trust region.
    distance_cutoff = neb_path_internal_coordinate_length * 0.05
    # the location of training data in internal coordinate.
    gpr_training_internal_coordinate = gpr_model.output_training_internal_inputs()
    

    # compute the distance and find beads that move out of the trusted region.
    nbeads = np.shape(beads_x)[0]
    internal_coordinate_r_closest_list = []
    for bead_index in range(nbeads):
        bead_internal_q = beads_internal_coordinate[bead_index]

        # scaled distance between gpr training data and beads.
        internal_coordinate_r = np.sqrt(np.sum(np.power(bead_internal_q - gpr_training_internal_coordinate , 2) , axis = 1))
        
        nearest_gpr_data_index = np.argmin(internal_coordinate_r)

        # scaled distance r between beads and the closest gpr training data
        internal_coordinate_r_closest = np.linalg.norm(bead_internal_q - gpr_training_internal_coordinate[nearest_gpr_data_index])
        
        internal_coordinate_r_closest_list.append(internal_coordinate_r_closest)
        if internal_coordinate_r_closest > distance_cutoff:
            early_stop_bool = True 
            out_range_bead_index = bead_index 
            break
    
    # for debug
    print("\n")
    print("internal coordinate distance cutoff: " + str(distance_cutoff))
    print("distance for beads to nearest GPR point: " + str(internal_coordinate_r_closest_list))
    print("\n")

    return early_stop_bool, out_range_bead_index


def print_ab_initio_calculation_number(ab_initio_calculation_number, output_maker):
    '''
    print number of ab initio calculation during GPR optimization. Used to see how much computational effort GPR saves
    '''
    outfile = output_maker.get_output(" ab_initio_force_number.txt", "w")
    print("ab initio calculation number" + str(ab_initio_calculation_number), file = outfile)
    outfile.close_stream()
    