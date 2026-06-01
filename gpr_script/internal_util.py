import gpr.internal.CoulombInternal
import gpr.internal.ZmatrixInternal
import gpr.internal.MixedInternalCartesian
from ipi.engine.motion import Motion 
from ipi.utils.depend import dstrip
import numpy as np 
import h5py 

def select_reference_points(motion: Motion):
    """
    selects reference points for coordinate transformation.
    Initialize non redundant coordinate transformer.
    choose the point with the highest potential in the initial instanton path as reference point.
    """
    # previous ways.
    # beads_pots = np.copy(motion.forces.pots)
    # bead_index_at_transition_state = np.argmax(beads_pots)
    # ref_x = dstrip(motion.beads.q[bead_index_at_transition_state]).copy()

    # # Now, we just use the poiont with lowest energy at reactant and product side.
    # ref_x_reactant = dstrip(motion.beads.q[0]).copy() # coordinate at reactant side.
    # ref_x_product = dstrip(motion.beads.q[-1]).copy() # coordinate at product side

    # ref_x_list = np.array([ref_x, ref_x_reactant, ref_x_product])
    
    ref_x_list = dstrip(motion.beads.q).copy()
    return ref_x_list 

def create_coordinate_transformer(motion:Motion, ref_x_list, load):
    """
    Initialize the coordinate transformer that transform the system 
    from the Cartesian coordinate into the internal coordinate.
    :param: load: whether to load data for coordinate transformer.
    """
    names = dstrip(motion.beads.names).copy().tolist()
    ref_x = ref_x_list[0]

    neb_final_gpr_folder = "neb_final_gpr_training"

    # create coordinate_transformer, 
    # which handles the transformation from the Cartesian coordinate to internal coordinate.
    # This is for Coulomb matrix type internal coordinate.
    internal_coord = motion.options["internal_coord"]
    nonbonded_atom_index = motion.optarrays["nonbonded_atom_index"]
    if internal_coord == "Coulomb":
        coordinate_transformer = gpr.internal.CoulombInternal.non_redundant_coordinate_transformer(
            motion.beads.natoms, 
            ref_x,
            load= load, 
            load_file_path= neb_final_gpr_folder 
        )
    elif internal_coord == "bond":
        # This is for internal coordinate that include bond angles and bond distance
        coordinate_transformer = gpr.internal.ZmatrixInternal.non_redundant_coordinate_transformer(
                motion.beads.natoms,
                ref_x_list,
                names,
                load,
                load_file_path= neb_final_gpr_folder
        )
    elif internal_coord == "IRZ":
        # This is for internal coordinate that include bond angles and bond distance
        coordinate_transformer = gpr.internal.ZmatrixInternal.non_redundant_coordinate_transformer(
                motion.beads.natoms,
                ref_x_list,
                names,
                load,
                load_file_path= neb_final_gpr_folder,
                inverse_distance= True
        )
    elif internal_coord == "MIC":
        coordinate_transformer = gpr.internal.MixedInternalCartesian.non_redundant_coordinate_transformer(
            motion.beads.natoms,
            ref_x_list,
            names,
            load,
            load_file_path= neb_final_gpr_folder,
            nonbonded_atom_index= nonbonded_atom_index
        )

    else:
        raise ValueError("The input for internal_coord should be either 'bond' or 'Coulomb' ")

    return coordinate_transformer

def output_internal_coord(coordinate_transformer: gpr.internal.ZmatrixInternal.non_redundant_coordinate_transformer):
    """
    output the singular vector corresponds to the internal coordinate.
    """
    Vh = np.copy(coordinate_transformer.ref_Vh)
    internal_coord_num = coordinate_transformer.nonzero_S_index_len
    ref_x = coordinate_transformer.ref_x
    # store data in hdf5 file.

    with h5py.File('internal_coord.h5', "w") as h5f:
        h5f.attrs['ndofs'] = internal_coord_num
        h5f.create_dataset('modes', data= Vh, compression= 'gzip')
        h5f.create_dataset('ref_x', data= ref_x, compression= 'gzip')
    