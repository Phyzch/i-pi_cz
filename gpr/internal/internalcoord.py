"""
use internal.py function in geometric package.
"""
from geometric.internal import * 
import geometric.internal
from .molecule import NewMolecule
# List of Primitive Internal Coordinate.
import warnings 

def convert_bohrs_degrees(prims, values):
    """ Convert values of primitive ICs (or differences) from
    weighted atomic units to Bohrs and degrees.
    The unit of translation will be Bohrs.
    The unit of rotation will be degrees. """
    converted = np.array(values).copy()
    for ic, c in enumerate(prims):
        if type(c) in [TranslationX, TranslationY, TranslationZ]:
            w = 1.0
        elif hasattr(c, 'w'):
            w = c.w
        else:
            w = 1.0
        if type(c) in [TranslationX, TranslationY, TranslationZ, CartesianX, CartesianY, CartesianZ, Distance, LinearAngle, CentroidDistance]:
            factor = 1  # here we assume the unit of translation will be Bohrs.
        elif c.isAngular:
            factor = 180.0 / np.pi
        converted[ic] /= w
        converted[ic] *= factor
    return converted

class PrimitiveInternalCoordinates(InternalCoordinates):
    """
    Primitive Redundant Internal Coordinate.
    This code is a copy of PrimitiveInternalCoordinates class in geometric: internal.py 
    Need to rewrite it to make it simplier. 
    We do not implement TRIC and constraint in the current class. 
    This is to simplify the implementation.
    """
    def __init__(self, molecule: Molecule, connect=True, addcart=False, **kwargs):
        super(PrimitiveInternalCoordinates, self).__init__()
        # connect = True corresponds to "traditional" internal coordinates with minimum spanning bonds
        # connect = False, addcart = True corresponds to HDLC
        # connect = False, addcart = False corresponds to TRIC
        self.connect = connect
        self.addcart = addcart

        self.Internals = []  # List of Primitive Internal coordinates.

        self.elem = molecule.elem
        # Atomic mass array
        self.mass = np.repeat([PeriodicTable[i] for i in self.elem], 3)
        # Here each molecule object corresponds to one frame in geomeTRIC.
        self.makePrimitives(molecule)

    def makePrimitives(self, molecule: Molecule):
        """
        Make primitive internal coordinates based on atom connectivity topology of the molecule.
        """
        # build topology for molecules based on atom radius and distance between atoms. 
        molecule.build_topology() 

        # reference coordinates in Bohr.
        coords = molecule.xyz.flatten()
        # Make a distance matrix mapping atom pairs to inter-atomic distances.
        AtomIterator, dxij = molecule.distance_matrix(pbc= False)

        # record the distance between atoms (this belongs to Internal Coordinate.)
        D = {} 
        for index, r in zip(AtomIterator, dxij[0]):
            assert index[0] < index[1]
            D[tuple(index)] = r 
        
        # create non-covalent bond to characterize inter-molecular interaction.
        # Or use HDIC. The TRIC currently is not implemented.
        # Add non-covalent bonds into self.Internal.
        noncov = self.add_noncov(molecule, D)

        # Add an internal coordinate for bonded atom pairs. (Distance between bonded atom pairs)
        for (a,b) in molecule.topology.edges():
            self.add(Distance(a,b))
        
        # Linear Angle threshold -- corresponds about 162 degrees.
        self.LinThre = 0.95

        # Add angle between bonded molecules.
        # Or bending internal coords corresponding to linear 
        # The angular internal coordinates is added to self.Internals
        self.AngDict = self.add_angle(molecule, noncov, coords)

        # Out of Plane internal coordinate.
        # The out of plane internal coordinate is added to self.Internals.
        self.add_out_of_plane(molecule, noncov, coords)

        # Creating dihedrals
        # The dihedral angle internal coordinate is added to self.Internals.
        self.add_dihedral(molecule, noncov, coords)

    def add_noncov(self, molecule: Molecule, D: dict):
        """
        add non-covalent bond between molecular fragments.
        This is to treat inter-molecular interaction.
        Transition rotation Internal Coordinate (TRIC), Hybrid Delocalized Internal Coordinate (HDIC)
        and traditional approach (all connected method) differs from each other 
        in how we generate internal coordinates that characterize inter-molecular interaction.
        Here we only implement the traditional way.

        :param: molecule: Molecule class object. topology of atom connectivity.
        :param: D: dictionary of distance between atom pairs.
        """
        # create a weighted graph. The weight of edge is intr-atom distance r.
        dgraph = nx.Graph()
        
        for i in range(molecule.na):
            dgraph.add_node(i)

        for k, v in D.items():
            # key is AtomIterator tuple (i,j)
            # v is distance r.
            dgraph.add_edge(k[0], k[1], weight= v)
        
        # minimum spanning trees.
        mst = sorted(
            list(
                    nx.minimum_spanning_edges(dgraph, data= False)
                )
            )
        
        # build a list of noncovalent distances.
        noncov = []
        # Connect all non-bonded fragments together.
        if self.connect:
            for edge in mst:
                if edge not in list(molecule.topology.edges()):
                    # This edge is between non-bonded fragments.
                    molecule.topology.add_edge(edge[0], edge[1])
                    noncov.append(edge)
        else:
            if self.addcart:
                for i in range(molecule.na):
                    self.add(CartesianX(i, w= 1.0))
                    self.add(CartesianY(i, w= 1.0))
                    self.add(CartesianZ(i, w= 1.0))
            else:
                raise NotImplementedError("We do not implement Translation Rotation Internal Coordinate Here.\
                                           This may change in the future.")

        return noncov 
    
    def add_angle(self, molecule: Molecule, noncov, coords):
        """
        Add angles and linear angles for bonded atoms.
        :param: molecule: Molecular object. Contains information about connectivity between atoms.
        :param: noncov: list of noncov bonds (record edge (a,b))
        :param: coords: coordinates of molecule (3 * Natom.)
        """
        AngDict = defaultdict(list)
        for b in molecule.topology.nodes():
            for a in molecule.topology.neighbors(b):
                for c in molecule.topology.neighbors(b):
                    if a < c:
                        Ang = Angle(a, b, c)

                        # check if bond is non-covalent bond.
                        is_ab_noncov = (min(a,b), max(a,b)) in noncov
                        is_bc_noncov = (min(b,c), max(b,c)) in noncov 
                        # results for not noncov.
                        nnc = (is_ab_noncov & is_bc_noncov)

                        if np.abs(np.cos(Ang.value(coords))) < self.LinThre:
                            # not linear.
                            self.add(Angle(a, b, c))
                            AngDict[b].append(Ang)
                        elif self.connect or (not self.addcart):
                            # Add linear angle IC's
                            # LPW 2019-02-16: Linear angle ICs work well for "very" linear angles in molecules (e.g. HCCCN)
                            # but do not work well for "almost" linear angles in noncovalent systems (e.g. H2O6).
                            # Bringing back old code to use "translations" for the latter case, but should be investigated
                            # more deeply in the future.
                            # LPW 2022-02-15: Linear angle ICs have been improved, and should no longer require resetting if the
                            # atoms in the angle go through a large rotation. They are currently being used.
                            if nnc == 0:
                                # one of (a,b) and (b,c) is not in noncov, which means that are covalent bond.
                                # Add two internal coordinates corresponding to the bending of the linear angle.
                                self.add(LinearAngle(a, b, c, 0))
                                self.add(LinearAngle(a, b, c, 1))
                            else:
                                # Use translation in case there is non-covalent linear bond.
                                # Unit vector connecting atoms a and c
                                nac= molecule.xyz[c] - molecule.xyz[a]
                                nac = nac / np.linalg.norm(nac)
                                # Dot products of this vector with the Cartesian axes
                                dots = [np.abs(np.dot(ei, nac)) for ei in np.eye(3)]
                                # Functions for adding Cartesian coordinate
                                # carts = [CartesianX, CartesianY, CartesianZ]
                                trans = [TranslationX, TranslationY, TranslationZ]
                                w = np.array([-1.0, 2.0, -1.0])
                                # Add two of the most perpendicular Cartesian coordinates
                                for i in np.argsort(dots)[:2]:
                                    self.add(trans[i]([a, b, c], w=w))
        
        return AngDict 

    def add_out_of_plane(self, 
                         molecule:Molecule, 
                         noncov,
                         coords):
        """
        Add out of plane internal coordinate.
        :param: molecule: Molecular object. Contains information about connectivity between atoms.
        :param: noncov: list of noncov bonds (record edge (a,b))
        :param: coords: coordinates of molecule (3 * Natom.)
        """
        # out of plane internal coordinate.
        for b in molecule.topology.nodes():
            for a in molecule.topology.neighbors(b):
                for c in molecule.topology.neighbors(b):
                    for d in molecule.topology.neighbors(b):
                        if a < c < d:
                            nnc = (min(a, b), max(a, b)) in noncov
                            nnc += (min(b, c), max(b, c)) in noncov
                            nnc += (min(b, d), max(b, d)) in noncov
                            # if nnc >= 1: continue
                            for i, j, k in sorted(list(itertools.permutations([a, c, d], 3))):
                                Ang1 = Angle(b,i,j)
                                Ang2 = Angle(i,j,k)
                                if np.abs(np.cos(Ang1.value(coords))) > self.LinThre: continue
                                if np.abs(np.cos(Ang2.value(coords))) > self.LinThre: continue
                                # This is to take care the case that 4 molecules is in a plane (BH3), the possible out of plane bending.
                                if np.abs(np.dot(Ang1.normal_vector(coords), Ang2.normal_vector(coords))) > self.LinThre:  # linear.
                                    self.delete(Angle(i, b, j))
                                    self.add(OutOfPlane(b, i, j, k))
                                    break
        

    def add_dihedral(self,
                     molecule: Molecule,
                     noncov,
                     coords):
        """
        Add dihedral angle between 4 atoms as internal coordinate.
        :param: molecule: Molecular object. Contains information about connectivity between atoms.
        :param: noncov: list of noncov bonds (record edge (a,b))
        :param: coords: coordinates of molecule (3 * Natom.)
        """
        # Find groups of atoms that are in straight lines. This is for creating dihedrals.
        atom_lines = [list(i) for i in molecule.topology.edges()]
        # This will make atom_lines include line object (a,b,c,d,e,..) where atom aligns in a line.
        while True:
            # For a line of two atoms (one bond):
            # AB-AC
            # AX-AY
            # i.e. AB is the first one, AC is the second one
            # AX is the second-to-last one, AY is the last one
            # AB-AC-...-AX-AY
            # AB-(AC, AX)-AY
            atom_lines0 = deepcopy(atom_lines)
            for aline in atom_lines:
                # Imagine a line of atoms going like ab-ac-ax-ay.
                # Our job is to extend the line until there are no more
                ab = aline[0]  # node 0 of edge aline
                ay = aline[-1] # node 1 of edge aline.
                for aa in molecule.topology.neighbors(ab):
                    if aa not in aline:
                        # If the angle that AA makes with AB and ALL other atoms AC in the line are linear:
                        # Add AA to the front of the list
                        if all([np.abs(np.cos(Angle(aa, ab, ac).value(coords))) > self.LinThre for ac in aline[1:] if ac != ab]):
                            aline.insert(0, aa)
                for az in molecule.topology.neighbors(ay):
                    if az not in aline:
                        if all([np.abs(np.cos(Angle(ax, ay, az).value(coords))) > self.LinThre for ax in aline[:-1] if ax != ay]):
                            aline.append(az)

            # If no further atoms add to liner line, break.
            if atom_lines == atom_lines0: break
        
        # unique set of atom lines.
        atom_lines_uniq = []
        for atom_line in atom_lines:     
            if tuple(atom_line) not in set(atom_lines_uniq):
                atom_lines_uniq.append(tuple(atom_line))
        
        # Normal dihedral code
        for aline in atom_lines_uniq:
            # Go over ALL pairs of atoms in a line
            for (b, c) in itertools.combinations(aline, 2):  # generate a pair of atom indices.
                if b > c: (b, c) = (c, b)
                # Go over all neighbors of b
                for a in molecule.topology.neighbors(b):
                    # Go over all neighbors of c
                    for d in molecule.topology.neighbors(c):
                        # Make sure the end-atoms are not in the line and not the same as each other
                        if a not in aline and d not in aline and a != d:
                            nnc = (min(a, b), max(a, b)) in noncov
                            nnc += (min(b, c), max(b, c)) in noncov
                            nnc += (min(c, d), max(c, d)) in noncov
                            # print aline, a, b, c, d
                            Ang1 = Angle(a,b,c)
                            Ang2 = Angle(b,c,d)
                            # Eliminate dihedrals containing angles that are almost linear
                            # (should be eliminated already)
                            if np.abs(np.cos(Ang1.value(coords))) > self.LinThre: continue
                            if np.abs(np.cos(Ang2.value(coords))) > self.LinThre: continue
                            self.add(Dihedral(a, b, c, d))

    # Return internal coordinates:
    def calculate(self, xyz):
        """
        Return Internal coordinates corresponds to xyz.
        """
        answer = []
        for Internal in self.Internals:
            answer.append(Internal.value(xyz))
        return np.array(answer)

    # Return derivatives of internal coordinates.
    def derivatives(self, xyz):
        # self.calculate(xyz)
        answer = []
        for Internal in self.Internals:
            answer.append(Internal.derivative(xyz))
        # This array has dimensions:
        # 1) Number of internal coordinates
        # 2) Number of atoms
        # 3) 3
        return np.array(answer)

    def second_derivatives(self, xyz):
        # self.calculate(xyz)
        answer = []
        for Internal in self.Internals:
            answer.append(Internal.second_derivative(xyz))
        # This array has dimensions:
        # 1) Number of internal coordinates
        # 2) Number of atoms
        # 3) 3
        # 4) Number of atoms
        # 5) 3
        return np.array(answer)
    
    # Auxiliary function.
    def add(self, dof):
        if dof not in self.Internals:
            self.Internals.append(dof)
    
    def delete(self, dof):
        for ii in range(len(self.Internals))[::-1]:
            if dof == self.Internals[ii]:
                del self.Internals[ii]
    
    def __repr__(self):
        lines = ["Internal coordinate system (atoms numbered from 1):"]
        typedict = OrderedDict()
        for Internal in self.Internals:
            lines.append(Internal.__repr__())
            if str(type(Internal)) not in typedict:
                typedict[str(type(Internal))] = 1
            else:
                typedict[str(type(Internal))] += 1
        if len(lines) > 1000:
            # Print only summary if too many
            lines = []
        for k, v in typedict.items():
            lines.append("%s : %i" % (k, v))
        return '\n'.join(lines)


    def __eq__(self, other):
        answer = True
        for i in self.Internals:
            if i not in other.Internals:
                answer = False
        for i in other.Internals:
            if i not in self.Internals:
                answer = False
        return answer

    def __ne__(self, other):
        return not self.__eq__(other)

class DelocalizedInternalCoordinates(InternalCoordinates):
    """
    Delocalized Internal Coordinate.
    This code is based on geometric.internal.DelocalizedInternalCoordinates
    I need a simplier version, so I write this one.
    """
    def __init__(self, molecule: NewMolecule,  connect= True, addcart=False):
        super(DelocalizedInternalCoordinates, self).__init__()
        # HDLC is given by (connect = False, addcart = True)
        # Standard DLC is given by (connect = True, addcart = False)
        # TRIC is given by (connect = False, addcart = False)
        self.connect = connect
        # Add Cartesian coordinates to all.
        self.addcart = addcart

        self.molecule = molecule
        # The DLC contains an instance of primitive internal coordinates.
        self.Prims = PrimitiveInternalCoordinates(molecule, connect=connect, addcart=addcart)
        self.na = molecule.na
        # Atomic mass array
        self.mass = np.repeat([PeriodicTable[i] for i in molecule.elem], 3)

        # Build DLCs
        # xyzs in molecule.xyzs is already in bohr unit.
        self.build_dlc(molecule.xyzs)
    
    def build_dlc(self, xyzs):
        """
        Build delocalized internal coordinate.
        param: xyz: Cartesian coordinate.
        """
        # flatten the xyzs
        xyzs = np.reshape(xyzs, (xyzs.shape[0], -1))
        # list of Bmat.
        Bmat_list = []
        for index in range(xyzs.shape[0]):
            xyz = xyzs[index]
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

    def GInverse(self, xyz):
        return self.GInverse_SVD(xyz)

    def calcGradCart(self, xyz, gradq):
        """
        calculate the gradient in Cartesian coordinate. df/dx.
        Gx = B^T * gradq.
        """
        Bmat = self.wilsonB(xyz)
        # Cartesian coordinate gradient.
        Gx = np.transpose(Bmat) @ gradq 
        return Gx 
    
    def __eq__(self, other):
        return self.Prims == other.Prims

    def __ne__(self, other):
        return not self.__eq__(other)

