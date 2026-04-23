from ase.io import read 
from ase.atoms import Atoms as ASEAtoms
from dscribe.descriptors import SOAP  
import numpy as np 
from collections import OrderedDict
import warnings 

class SOAPDescriptor():
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
        self.feature_dim = feature_vectors.shape[1]

        # How to use soap descriptor:
        # feature_vectors = soap.create(molecule, n_jobs= 1)
        # derivatives, feature_vectors = soap.derivatives(molecule, return_descriptor= True, n_jobs= 1)
    

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
    
    def wilsonB(self, xyz):
        """
        Return Wilson B matrix. This is the flattened version of the first derivative.
        We also use hashmap to avoid repetitive computation."""
        xhash = hash(xyz.tobytes())
        if xhash in self.stored_wilsonB:
            return self.stored_wilsonB[xhash]

        WilsonB = [] 
        Der = self.derivatives(xyz) 
        for i in range(Der.shape[0]):
            WilsonB.append(Der[i].flatten())
        self.stored_wilsonB[xhash] = np.array(WilsonB)

        ans = np.array(WilsonB)
        return ans 

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
        return second_derivatives
    

class DLC_SOAP():
    """
    Create delocalized internal coordinate created from SOAP descriptor.
    """
    def __init__(self, 
                 ref_x_list,
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
        self.Prims = SOAPDescriptor(molecule, natom, r_cut, n_max, l_max)
        self.na = natom 
        self.molecule = molecule
        self.ref_x_list = ref_x_list # ref coordinate list for building the delocalized internal coordinates.

        self.build_dlc() 

    def build_dlc(self):
        """
        The coordinate of ref_x_list will be used to build the reference delocalized internal coordinate.
        """
        ref_num = self.ref_x_list.shape[0]
        Bmat_list = [] 
        for index in range(ref_num):
            xyz = self.ref_x_list[index]
            Bmat = self.Prims.wilsonB(xyz)
            Bmat_list.append(Bmat)
        
        Bmat_list = np.array(Bmat_list)
        Bmat_average = np.mean(Bmat_list, axis= 0)
        # SVD decomposition of Bmat
        U, S, Vh = np.linalg.svd(Bmat_average, full_matrices= False)

        natom = self.na
        # If we do not include information about the position of 
        # the center of mass and the orientation of molecular system.
        # The number of nonzero eigenvalue should be 3N-6.
        dlc_na = 3 * natom - 6

        assert (
            np.size(S) >= dlc_na
        ), "number of nonzero singular value of B is smaller than 3n-6. Wrong"

        # sort singular value according to their absolute values. descending order
        s_index = np.array(range(len(S)))
        nonzero_S_index = s_index[: dlc_na]
        nonzero_S = S[nonzero_S_index]

        # sanity check in case we have non-zero sinuglar value number larger than 3n-6.
        zero_S_index = s_index[dlc_na :]
        zero_S = S[zero_S_index]
        if np.size(zero_S) != 0:
            zero_s_max = np.max(np.abs(zero_S))
            if zero_s_max > np.power(10.0, -4) * np.min(np.abs(nonzero_S)):
                # nonzero value is too large
                warnings.warn(
                    "zero singular value of matrix B is too large. zero_s_max: {}  min(nonzero_s): {}".format(
                        zero_s_max, np.min(np.abs(nonzero_S))
                    )
                )

        S_nonredundant = S[nonzero_S_index]
        print(f"All non-redundant singular values: {S_nonredundant}")

        # truncate nonzero singular value.
        U = U[:, :dlc_na]
        Vh = Vh[:dlc_na, :]
        S = S[:dlc_na]

        # record U matrix and singular value matrix S.
        self.ref_U = U  
        self.ref_UT = U.T
        self.S = S
        self.ref_Vh = Vh 
    
    def calculate(self, coords):
        """
        Calculate Delocalized Internal Coordinate given the Cartesian coordinate.
        """
        # Primitive Internal Coordinate.
        PrimVals = self.Prims.calculate(coords)
        
        vals = PrimVals @ self.ref_U 
        
        return vals 
    
    def derivatives(self, coords):
        """
        Calculate the change of the DLCs with respect to the Cartesian coordinates. 
        This computes derivatives for individual Cartesian coordinates.
        The returned array has dimensions:
        1) Number of delocalized internal coordinates
        2) Number of atoms
        3) 3
        """
        # shape: [Ninternal, natom, 3]
        PrimDers = self.Prims.derivatives(coords)

        ders = np.tensordot(self.ref_U, PrimDers, axes= (0,0))
        return np.array(ders)
    
    def second_derivatives(self, coords):
        """
        Calculate the second derivatives of DLCs with respect to the Cartesian coordinate.
        This array has dimensions:
        1) Number of delocalized internal coordinates
        2) Number of atoms
        3) 3
        4) Number of atoms
        5) 3
        """
        PrimDers = self.Prims.second_derivatives(coords)
        ders = np.tensordot(self.ref_U, PrimDers, axes= (0,0))

        return ders

    def inverse_second_derivatives(self, coords):
        """
        Calculate the d^x/dq^2. here q is DLC, x is cartesian coordinate.
        d^x/dq^2 = (-1) (dx/dq)^3 (dq/dx)^2 = (-1) * ((B)^{-1})^3 * second_derivative
        """
        # d^2 q/ dx^2, shape: [3n-6, n, 3, n, 3]
        ders = self.second_derivatives(coords)
        
        natom = self.na 
        # shape: [3n-6, 3n, 3n]
        hessian_q_xx = ders.reshape((ders.shape[0], natom * 3, natom * 3))

        # dq/dx : shape[3n-6, 3n]
        Bq = self.wilsonB(coords)
        # shape: [3n, 3n-6]. (Bq)^-1.  (dq/dx)^(-1)
        inverse_Bq = np.linalg.pinv(Bq)
        # shape: [3n-6, 3n-6, 3n]
        h1 = np.einsum('ijk, jl -> ilk', hessian_q_xx, inverse_Bq)
        # shape: [3n-6, 3n-6, 3n-6]
        h2 = np.einsum('ijk, kl -> ijl', h1, inverse_Bq)
        # shape: [3n, 3n-6, 3n -6]
        h3 = np.einsum('ijk, li -> ljk', h2, inverse_Bq)

        hessian_x_qq = (-1) * h3 

        return hessian_x_qq 

    def calcGradCart(self, xyz, gradq):
        """
        calculate the gradient in Cartesian coordinate. df/dx.
        Gx = B^T * gradq.
        """
        Bmat = self.wilsonB(xyz)
        # Cartesian coordinate gradient.
        Gx = np.transpose(Bmat) @ gradq 
        return Gx 