"""
Implement mixed internals and Cartesian descriptors (MIC). See J. Chem. Theory Comput. 2024, 20, 3766−3778
This is achieved by using HDLC(hybrid delocalized internal coordinate) in geoMETRIC code, 
and set atoms that we don't want to form covalent bond (substrate in JCTC 2024) as non-bonded atoms. 
"""
import numpy as np
import gpr.internal.molecule 
from .DummyInternal import dummy_non_redundant_coordinate_transformer
from .internalcoord import DelocalizedInternalCoordinates

class non_redundant_coordinate_transformer(dummy_non_redundant_coordinate_transformer):
    """
    perform transformation between Cartesian coordinate and internal coordinate,
    """
    def __init__(self, 
                natom: int,
                ref_x_list: np.ndarray,
                elem: list = None,
                load: bool= False,
                load_file_path= None,
                nonbonded_atom_index= [], 
                ):
        """
        :param: natoms number of atoms in molecule
        :param: ref_x_list: a list of reference coordinate x, which we will use to create the topology of molecule.
                By default, we choose the first ref_x in ref_x_list as the coordinate of the created molecule object.
        :param: elem: elements of atoms in molecules: in ipi, this is self.beads.names.
        :param: nonbonded_atom_index: index of atoms that are not bonded. 
        """
        ref_x = ref_x_list[0]

        super().__init__(natom, ref_x)

        self.ref_x_list = ref_x_list
        self.internal_coord_type = "MIC"
        
        if load:
            self.load_coordinate_transformer(load_file_path)
        
        if np.size(self.ref_x) != 3 * natom:
            raise (
                "The size of reference point for initializing non redundant coordinate is not 3 * natom: size of ref_x: {} , natom: {}".format(
                    np.size(self.ref_x), natom
                )
            )

        # create molecule object. 
        molecule = gpr.internal.molecule.create_molecule(
            natom,
            elem,
            ref_x_list,
            molecule_index= 0,
            nonbonded_atom_index= nonbonded_atom_index
        )

        # create hybrid delocalized internal coordinate.
        self.dlc_coord = DelocalizedInternalCoordinates(molecule, connect= False, addcart= True, inverse_distance= False)

        # record unitary matrix U and singular value matrix S for the reference point.
        self.ref_U = self.dlc_coord.ref_U 
        self.ref_UT = self.ref_U.T 
        self.ref_S = self.dlc_coord.S
        self.ref_Vh = self.dlc_coord.ref_Vh  
        self.nonzero_S_index_len = len(self.ref_S)

        print(f"Number of nonzero dofs: {self.nonzero_S_index_len}")