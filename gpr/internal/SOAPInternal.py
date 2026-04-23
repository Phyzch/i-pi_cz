from .DummyInternal import dummy_non_redundant_coordinate_transformer
import numpy as np
import h5py 
import os 
from ase.io import read 
from ase import Atoms 
from .soap import DLC_SOAP

class non_redundant_coordinate_transformer(dummy_non_redundant_coordinate_transformer):
    """
    perform transformation between Cartesian coordinate and internal coordinate.
    """
    def __init__(self,
                 natom: int,
                 ref_x_list: np.ndarray,
                 geometry_file_path: str,
                 r_cut: float= 5.0,
                 n_max: int= 8,
                 l_max: int= 8,
                 load: bool= False,
                 load_file_path= None,
                ):
        ref_x = ref_x_list[0]
        super().__init__(natom, ref_x)

        self.ref_x_list = ref_x_list
        self.internal_coord_type = "SOAP"

        if load:
            self.load_coordinate_transformer(load_file_path)

        if np.size(self.ref_x) != 3 * natom:
            raise (
                "The size of reference point for initializing non redundant coordinate is not 3 * natom: size of ref_x: {} , natom: {}".format(
                    np.size(self.ref_x), natom
                )
            )

        # create molecule object from geometry file.
        # the positions need to be updated as ref_x. 
        try:
            molecule = read(geometry_file_path)
        except Exception as e:
            raise Exception("Failed to read geometry file: {}. Error message: {}".format(geometry_file_path, str(e)))
        
        molecule.set_positions(ref_x.reshape((natom, 3)))

        self.dlc_coord = DLC_SOAP(ref_x_list, molecule, natom, r_cut, n_max, l_max)
        # record unitary matrix U and singular value matrix S for the reference point.
        self.ref_U = self.dlc_coord.ref_U 
        self.ref_UT = self.ref_U.T 
        self.ref_S = self.dlc_coord.S
        self.ref_Vh = self.dlc_coord.ref_Vh  
        self.nonzero_S_index_len = len(self.ref_S)

        print(f"Number of nonzero dofs: {self.nonzero_S_index_len}")
    