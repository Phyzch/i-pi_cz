import gpr.internal.CoulombInternal
import gpr.internal.ZmatrixInternal
from ipi.engine.motion import Motion 
from ipi.utils.depend import dstrip
import numpy as np 

def select_reference_points(motion: Motion):
    """
    selects reference points for coordinate transformation.
    Initialize non redundant coordinate transformer.
    choose the point with the highest potential in the initial instanton path as reference point.
    """
    beads_pots = np.copy(motion.forces.pots)
    bead_index_at_transition_state = np.argmax(beads_pots)
    ref_x = dstrip(motion.beads.q[bead_index_at_transition_state]).copy()

    # Now, we just use the poiont with lowest energy at reactant and product side.
    ref_x_reactant = dstrip(motion.beads.q[0]).copy() # coordinate at reactant side.
    ref_x_product = dstrip(motion.beads.q[-1]).copy() # coordinate at product side

    ref_x_list = np.array([ref_x, ref_x_reactant, ref_x_product])
    
    return ref_x_list 

def create_coordinate_transformer(motion:Motion, ref_x_list):
    """
    Initialize the coordinate transformer that transform the system 
    from the Cartesian coordinate into the internal coordinate.
    """
    names = dstrip(motion.beads.names).copy().tolist()
    ref_x = ref_x_list[0]
    # create coordinate_transformer, 
    # which handles the transformation from the Cartesian coordinate to internal coordinate.
    # This is for Coulomb matrix type internal coordinate.
    internal_coord = motion.options["internal_coord"]
    if internal_coord == "Coulomb":
        coordinate_transformer = gpr.internal.CoulombInternal.non_redundant_coordinate_transformer(
            motion.beads.natoms, ref_x 
        )
    elif internal_coord == "bond":
        # This is for internal coordinate that include bond angles and bond distance
        coordinate_transformer = gpr.internal.ZmatrixInternal.non_redundant_coordinate_transformer(
                motion.beads.natoms,
                ref_x_list,
                names
        )
    else:
        raise ValueError("The input for internal_coord should be either 'bond' or 'Coulomb' ")

    return coordinate_transformer