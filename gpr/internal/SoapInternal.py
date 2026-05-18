from .soap import DLC_SOAP
import numpy as np 
import gpr.internal.molecule 
from .DummyInternal import dummy_non_redundant_coordinate_transformer
from ase.io import read 

class non_redundant_coordinate_transformer(dummy_non_redundant_coordinate_transformer):
    """
    perform transformation between Cartesian coordinate and soap descriptor.
    """
    def __init__(self,
                 natom: int,
                 ref_x_list: np.ndarray,
                 elem: list= None,
                 load: bool= False,
                 load_file_path= None,
                 geometry_file_path = None,
                 soap_parameter = None, 
                 ):
        """
        :param: natoms: number of atoms in molecule
        :param: ref_x_list: a list of reference coordinate x, which we will use to create the topology of molecule.
                By default, we choose the first ref_x in ref_x_list as the coordinate of the created molecule object.
        :param: elem: elements of atoms in molecules: in ipi, this is self.beads.names.
        :param: molecule_coord_file: file path that records coordinate of molecules.
        """
        ref_x = ref_x_list[0]

        super().__init__(natom, ref_x)

        self.ref_x_list = ref_x_list
        self.internal_coord_type = "soap"

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
        molecule.set_positions(ref_x.reshape(self.natom, 3))

        if soap_parameter is None:
            r_cut, n_max, l_max = [5, 8, 8]
        else:
            r_cut, n_max, l_max = [soap_parameter["r_cut"], soap_parameter["n_max"], soap_parameter["l_max"]]
        
        self.dlc_coord = DLC_SOAP(ref_x_list, molecule, natom, r_cut, n_max, l_max)

        # record unitary matrix U and singular value matrix S for the reference point.
        self.ref_U = self.dlc_coord.ref_U 
        self.ref_UT = self.ref_U.T 
        self.ref_S = self.dlc_coord.S
        self.ref_Vh = self.dlc_coord.ref_Vh  
        self.nonzero_S_index_len = len(self.ref_S)

        print(f"Number of nonzero dofs: {self.nonzero_S_index_len}")