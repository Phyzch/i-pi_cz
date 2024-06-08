'''
utility module for neb_instanton_gpr.py module
'''
import numpy as np 
from ipi.utils.internalcoordtools import non_redundant_coordinate_transformer
from ipi.utils.gprtools import GPModelWithDerivativesWrapper
from ipi.engine.beads import Beads
from ipi.utils.depend import dstrip

def check_neb_early_stop(beads_x, trust_region_ratio, gpr_model: GPModelWithDerivativesWrapper, initial_scaled_internal_coordinate_neb_path_length, initial_effective_kernel_length_scale):
    '''
    check early stoage criterion for neb algorithm with machine learning.
    If the bead move out of trusted region, then we stop the current move.
    This means if the distance between beads and nearest gpr point in internal coordinate "q" exceeds 2 * sigma, 
    (where sigma is the length scale of gpr kernel), then the bead is out of trusted region.
    Notice because of automatic relevance determination (ARD) (https://mogp-emulator.readthedocs.io/en/latest/methods/proc/ProcAutomaticRelevanceDetermination.html#:~:text=We%20describe%20here%20the%20method,scales%20in%20the%20covariance%20models.)
    We should use scaled coordinate with respect to the kernel length scale.

    :param: beads_x: cartesian coordinate X of neb beads 
    :param: trust_region_ratio: cutoff for rmax / neb_path_length. If beads move out of trust region, we stop the inner neb loop. 
    :param: gpr_model: model to perform the Gaussian Process Regression.
    :param: initial_scaled_internal_coordinate_neb_path_length: the initial neb path length in internal coordinate scaled by kernel length scale. 
    :return: early_stop_bool: bool variable to indicate whether there is bead out of trust region.
             out_range_bead_index: the bead index that move out of the trusted region.
    '''
    early_stop_bool = False 
    out_range_bead_index = -1
    
    coordinate_transformer = gpr_model.coordinate_transformer

    # kernel output scale and kernel length scale of kernels
    kernel_output_scale = gpr_model.output_kernel_outputscale()
    kernel_length_scale = gpr_model.output_kernel_lengthscale()
    kernel_number = gpr_model.gpr_SE_kernel_number

    # normalize the output scale:
    output_scale_sum = np.sum(kernel_output_scale)
    kernel_output_scale_normalized = kernel_output_scale / output_scale_sum
    # effective kernel lengthscale for scaling internal coordinate. l_eff^{-2} = sum_{n} output_scale_n / (l_n)^2.   
    effective_kernel_length_scale = np.power(np.sum(kernel_output_scale_normalized[:, np.newaxis] / np.power(kernel_length_scale, 2) , axis = 0), -0.5)

    # the location of current beads in internal coordinate
    beads_internal_coordinate = coordinate_transformer.get_internal_coordinate_q(np.copy(beads_x))

    # distance cutoff for trust region.
    distance_cutoff = initial_scaled_internal_coordinate_neb_path_length * trust_region_ratio
    # the location of training data in internal coordinate.
    gpr_training_internal_coordinate = gpr_model.output_training_internal_inputs()
    

    # compute the distance and find beads that move out of the trusted region.
    nbeads = np.shape(beads_x)[0]
    internal_coordinate_r_closest_list = []
    for bead_index in range(nbeads):
        bead_internal_q = beads_internal_coordinate[bead_index]

        # distance between gpr training data and beads.
        internal_coordinate_r = np.linalg.norm( (bead_internal_q[np.newaxis, :] - gpr_training_internal_coordinate) / initial_effective_kernel_length_scale , axis = 1) 
        
        nearest_gpr_data_index = np.argmin(internal_coordinate_r)

        # distance r between beads and the closest gpr training data
        internal_coordinate_r_closest = internal_coordinate_r[nearest_gpr_data_index]
        
        internal_coordinate_r_closest_list.append(internal_coordinate_r_closest)
        if internal_coordinate_r_closest > distance_cutoff:
            if early_stop_bool == False:
                early_stop_bool = True 
                out_range_bead_index = bead_index 
            
    
    # for debug
    print("\n")
    print("effective kernel length scale: " + str(effective_kernel_length_scale))
    print("initial effective kernel length scale: " + str(initial_effective_kernel_length_scale))
    print("internal coordinate distance cutoff: " + str(distance_cutoff))
    print("distance for beads to nearest GPR point: " + str(internal_coordinate_r_closest_list))
    print("\n")

    if early_stop_bool:
        print("\n")
        print("bead index that cause early stop (starting from 0) : " + str(bead_index))
        print("@Early Stop for Inner Loop")
        print("\n")

    return early_stop_bool, out_range_bead_index


def print_ab_initio_calculation_number(ab_initio_calculation_number, output_maker):
    '''
    print number of ab initio calculation during GPR optimization. Used to see how much computational effort GPR saves
    '''
    outfile = output_maker.get_output(" ab_initio_force_number.txt", "w")
    print("ab initio calculation number:  " + str(ab_initio_calculation_number), file = outfile)
    outfile.close_stream()

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

    predicted_V_shift, predicted_V_grad, _, _ = gpr_model.predict_observable(gpr_beads.q)
    predicted_gpr_bead_force = - predicted_V_grad[0]
    predicted_V_shift = predicted_V_shift[0]

    test_V_error = np.abs((predicted_V_shift - ab_initio_pot) / ab_initio_pot)
    test_df = ab_initio_force - predicted_gpr_bead_force
    test_df_error = np.linalg.norm(test_df) / np.linalg.norm(ab_initio_force)   

    print("V error for test data " + str(test_V_error))
    print("f error for test data " + str(test_df_error))

    return predicted_V_shift, predicted_gpr_bead_force, ab_initio_pot, ab_initio_force
    
