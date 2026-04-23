from ase.io import read 
from ase.atoms import Atoms as ASEAtoms
from dscribe.descriptors import SOAP  
import numpy as np 
from collections import OrderedDict
import warnings 
from geometric.internal import InternalCoordinates

class SOAPDescriptor(InternalCoordinates):
    """
    Create SOAP descriptor.
    """
    def __init__(self, 
                 molecule: ASEAtoms,
                 natom: int,
                 r_cut: float = 5,
                 n_max: int = 8,
                 l_max: int = 8
                 ):
        """
        r_cut: A cutoff for SOAP descriptor in angstroms. 
        n_max: The maximum degree of radial basis functions.
        l_max: the maximum degree of spherical harmonics. 
        """
        super().__init__()
        self.molecule = molecule 
        self.natom = natom
        
        # this is the hashmap to store the wilson B matrix for different coordinate. 
        #The key is the hash of the coordinate, and the value is the wilson B matrix. 
        self.stored_wilsonB = OrderedDict() 
        

        # create SOAP descriptor for the molecule
        species = set() 
        species.update(molecule.get_chemical_symbols())

        soap = SOAP(
            species = species, 
            r_cut= r_cut,
            n_max= n_max,
            l_max= l_max,
            average= "outer",
            sparse= False
        )

        self.soap = soap # descriptor object 
        
        feature_vectors = soap.create(molecule, n_jobs= 1)
        self.feature_dim = feature_vectors.shape[0]

        # How to use soap descriptor:
        # feature_vectors = soap.create(molecule, n_jobs= 1)
        # derivatives, feature_vectors = soap.derivatives(molecule, return_descriptor= True, n_jobs= 1)
        self.CacheWarning = False 
        self.stored_second_derivative = OrderedDict() # this is the hashmap to store the second derivative for different coordinate.

    # Return SOAP descriptor
    def calculate(self, xyz):
        """
        Return SOAP descriptor for a given molecule
        """
        molecule = self.molecule.copy()
        xyz3 = xyz.reshape((self.natom, 3))
        assert molecule.get_positions().shape == xyz3.shape, "The shape of input coordinate is not correct."
        molecule.set_positions(xyz3)

        feature_vectors = self.soap.create(molecule, n_jobs= 1)
        return feature_vectors
    
    # return derivatives of SOAP descriptor 
    def derivatives(self, xyz):
        """
        Return derivatives of SOAP descriptor for a given molecule
        """
        molecule = self.molecule.copy()
        xyz3 = xyz.reshape((self.natom, 3))
        assert molecule.get_positions().shape == xyz3.shape, "The shape of input coordinate is not correct."
        molecule.set_positions(xyz3)

        derivatives = self.soap.derivatives(molecule, return_descriptor= False, n_jobs= 1)
        
        derivatives = derivatives[0].transpose((-1, 0, 1)) # shape [feature_dim, natom, 3]
        return derivatives

    def second_derivatives(self, xyz):
        """
        Return second derivatives of SOAP descriptor for a given molecule
        # This array has dimensions:
        # 1) Number of features
        # 2) Number of atoms
        # 3) 3
        # 4) Number of atoms
        # 5) 3
        """
        # Can only do numerical second derivative. 
        # use hashmap to avoid repetitive computation.
        xhash = hash(xyz.tobytes())
        if xhash in self.stored_second_derivative:
            ans = self.stored_second_derivative[xhash]
            return ans 

        second_derivatives = np.zeros((self.feature_dim, self.natom, 3, self.natom * 3))
        # loop over 3 * natom dimension to calculate the numrical second derivative
        for i in range(self.natom * 3):
            delta = 1e-4
            xyz_plus = np.copy(xyz)
            xyz_plus[i] = xyz_plus[i] + delta 

            xyz_minus = np.copy(xyz)
            xyz_minus[i] = xyz_minus[i] - delta 

            deriv_plus = self.derivatives(xyz_plus)
            deriv_minus = self.derivatives(xyz_minus)

            second_deriv = (deriv_plus - deriv_minus) / (2 * delta)
            second_derivatives[:, :, :, i] = second_deriv

        second_derivatives = np.reshape(second_derivatives, (self.feature_dim, self.natom, 3, self.natom, 3))

        self.stored_second_derivative[xhash] = second_derivatives
        if len(self.stored_second_derivative) > 100 and not self.CacheWarning:
            warnings.warn("The number of stored second derivative is larger than 100. This may cause memory issue.")
            self.CacheWarning = True
        
        return second_derivatives
    

    def GInverse(self, xyz):
        return self.GInverse_SVD(xyz)
