from .internalcoord import DelocalizedInternalCoordinates
import numpy as np
import gpr.internal.molecule 
import h5py 
import os 
from .DummyInternal import dummy_non_redundant_coordinate_transformer

class non_redundant_coordinate_transformer(dummy_non_redundant_coordinate_transformer):
    """
    perform transformation between Cartesian coordinate and internal coordinate.
    """
    def __init__(self, 
                 natom: int,
                 ref_x_list: np.ndarray,
                 elem: list = None, 
                 load: bool= False,
                 load_file_path= None,
                 inverse_distance: bool = False,
                 ):
        """
        :param: natoms: number of atoms in molecule
        :param: ref_x_list: a list of reference coordinate x, which we will use to create the topology of molecule.
                By default, we choose the first ref_x in ref_x_list as the coordinate of the created molecule object.
        :param: elem: elements of atoms in molecules: in ipi, this is self.beads.names.
        """
        ref_x = ref_x_list[0]

        super().__init__(natom, ref_x)

        self.ref_x_list = ref_x_list
        self.internal_coord_type = "bond"
        
        if load:
            self.load_coordinate_transformer(load_file_path)

        if np.size(self.ref_x) != 3 * natom:
            raise (
                "The size of reference point for initializing non redundant coordinate is not 3 * natom: size of ref_x: {} , natom: {}".format(
                    np.size(self.ref_x), natom
                )
            )

        # create molecule object
        molecule = gpr.internal.molecule.create_molecule(
            natom,
            elem,
            self.ref_x_list,
            molecule_index= 0  # we choose the first ref_x as the coordinate of the newly created molecule object.
        )

        # create delocalized internal coordinate object
        self.dlc_coord = DelocalizedInternalCoordinates(molecule, connect= True, addcart= False, inverse_distance= inverse_distance)

        # record unitary matrix U and singular value matrix S for the reference point.
        self.ref_U = self.dlc_coord.ref_U 
        self.ref_UT = self.ref_U.T 
        self.ref_S = self.dlc_coord.S
        self.ref_Vh = self.dlc_coord.ref_Vh  
        self.nonzero_S_index_len = len(self.ref_S)

        print(f"Number of nonzero dofs: {self.nonzero_S_index_len}")
    
    def store_coordinate_transformer(self, file_path):
        """
        store internal_coord_type & ref_x_list for the coordinate transformer 
        """
        file_name = os.path.join(file_path, "coordinate_transformer.hdf5")
        with h5py.File(file_name, "w") as f:
            f.create_dataset("internal_coord_type", data= self.internal_coord_type)
            f.create_dataset("ref_x_list", data= self.ref_x_list)
            f.create_dataset("ref_x", data= self.ref_x)
        
        print("data for coordinate transformer successfully stored")

    def load_coordinate_transformer(self, file_path):
        """
        load internal_coord_type & ref_x_list for coordinate transformer.
        """
        file_name = os.path.join(file_path, "coordinate_transformer.hdf5")
        try:
            with h5py.File(file_name, "r") as f:
                self.internal_coord_type = f["internal_coord_type"][()].decode() 
                self.ref_x_list = np.array(f["ref_x_list"])
                self.ref_x = np.array(f["ref_x"])
        except:
            print("@Warning: Fails to load ref_x for coordinate transformer.")
