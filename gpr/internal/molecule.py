from geometric.molecule import *

import numpy as np
from ipi.utils.messages import warning, info 
from geometric.nifty import ang2bohr
try:
    import networkx as nx
except ImportError:
    warning("utils/internal/molecule.py: Cannot import optional NetworkX module, topology tools won't work.")

def union_topology(molecule_list, 
                   molecule_index= 0):
    """
    Create a Molecule object.
    Take the union of topology of molecules (Molecule object) in the molecule_list as the topology of the new molecule.
    Create fragments using the topology of the new molecule.
    The coordinate of the new molecule is given by the xyz coordinate of the old molecule in molecule list:
    molecule_list[molecule_index]
    """
    molecule_num = len(molecule_list)
    if molecule_num <= molecule_index:
        raise ValueError(f"The molecule index {molecule_index} is larger than the number of molecules: {molecule_num} in molecule_list.")

    new_molecule = copy.deepcopy(molecule_list[molecule_index])

    # check natom, elem is the same:
    for molecule in molecule_list:
        assert molecule.elem == new_molecule.elem
        assert molecule.na == new_molecule.na 

    # take the union of bonds.
    bond_list = []
    for molecule in molecule_list:
        bonds = molecule.bonds
        bond_list = bond_list + bonds 

    bond_list = sorted(list(set(bond_list)))
    new_molecule.bonds = copy.deepcopy(bond_list)

    # take the union of topology. Use nx.compose() function.
    for index, molecule in enumerate(molecule_list):
        if index == molecule_index: 
            continue
        # take union of topology.
        new_molecule.topology = nx.compose(new_molecule.topology, molecule.topology)
    
    # create subgraph (fragments) from the new topology
    G = new_molecule.topology
    new_molecule.molecules = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    for g in new_molecule.molecules: g.__class__ = MyG 

    return new_molecule 

def create_molecule(natoms, elem, xyz_list, molecule_index= 0):
    """
    Create a Molecule object.
    The topology of the new molecule object is the union of topology for all molecules, each with the coordinate in the xyz_list.
    The xyz for new molecule is given by xyz_list[molecule_index].

    This is used for the case that reactant , TS & product have different atom connectivity graph.
    Therefore, we have to take union of them.
    """
    molecule_list = []
    assert molecule_index < len(xyz_list), f"molecule index {molecule_index} > number of xyz coordinate: {len(xyz_list)}"
    for xyz in xyz_list:
        molecule = NewMolecule(natoms, xyz, elem)
        # build topology
        molecule.build_topology()

        molecule_list.append(molecule)
    
    new_molecule = union_topology(molecule_list,
                                  molecule_index)
    
    return new_molecule


class NewMolecule(Molecule):
    """ Inherit from geometric.Molecule
    Unit system:  Angstroms.
    
    self.na: number of atoms.
    self.xyz: coordinate of atoms in molecules. [Natoms, 3] 2d array.
    self.elem: elements of atoms in molecules
    """

    def __init__(self,
                 natoms: int, 
                 xyz: np.ndarray, 
                 elem: list,
                 fnm = None, ftype = None, top = None, ttype = None,
                 **kwargs):
        """
        Create a Molecule object.

        Parameters
        ----------
        natoms: number of atoms.
        xyz: xyz coordinate of atoms in the molecule. shape: [3 * natoms]. Need to reshape it to [Natoms, 3] when set it to self.xyz.
        elem: elements of atoms. In ipi, this is self.beads.names
        Fac : float, optional
            Multiplicative factor to covalent radii criterion for deciding whether two atoms are bonded
            Default value of 1.2 is reasonable, 1.4 will produce lots of bonds
        """
        super(NewMolecule, self).__init__(fnm, ftype, top, ttype, **kwargs)
        # Data container.  All of the data is stored in here.
        self.Data = {}

        # number of atoms.
        self.na = natoms
        self.xyz = np.reshape(xyz, [natoms, 3])
        self.xyzs = self.xyz[np.newaxis, :]
        self.elem = elem 

    def __deepcopy__(self, memo):
        """
        Custom deepcopy method. Modified from geomeTRIC code.
        """
        New = NewMolecule(self.na,
                       self.xyz,
                       self.elem)
        
        # Copy over variables not contained in self.Data
        New.built_bonds = self.built_bonds
        New.top_settings = copy.deepcopy(self.top_settings)

        New.na = self.na 
        New.xyz = copy.deepcopy(self.xyz)
        New.elem = copy.deepcopy(self.elem)

        for key in self.Data:
            if key in ['xyzs']:
                # These variables are lists of NumPy arrays, NetworkX graph objects, or others with
                # explicitly defined copy() methods.
                New.Data[key] = []
                for i in range(len(self.Data[key])):
                    New.Data[key].append(copy.deepcopy(self.Data[key][i]))
            elif key in ['topology']:
                # These are NetworkX graph objects or other variables with explicitly defined copy() methods.
                New.Data[key] = self.Data[key].copy()
            elif key in ['molecules']:
                # fragments
                New.Data[key] = []
                for i in range(len(self.Data[key])):
                    New.Data[key].append(self.Data[key][i].copy())
            elif key in ['bonds']:
                # List of lists of 2 integers.
                New.Data[key] = []
                for i in range(len(self.Data[key])):
                    New.Data[key].append(self.Data[key][i][:])
            elif key in ['elem']:
                if not isinstance(self.Data[key], list):
                    raise RuntimeError('Expected data attribute %s to be a list, but it is %s' % (key, str(type(self.Data[key]))))
                # Lists of strings or floats.
                New.Data[key] = self.Data[key][:]
            else:
                raise RuntimeError("Failed to copy key %s" % key)
        
        return New 

    def build_bonds(self):
        """ 
        Build the bond connectivity graph.
        Simplified version without using grid algorithm. Also change the unit (unit is bohr in i-PI.)
        adapted from Molecule.build_bonds() function from geometric
        """
        Fac = self.top_settings['Fac']  # 1.2 by default.
        mindist = 1.0 # Any two atoms that are closer than this distance are bonded.

        # Create an atom-wise list of covalent radii.
        # Molecule object can have its own set of radii that overrides the global ones
        # Here .get(a, b) will return top_settings['radii'] if a exist, otherwise return b, which is default radii.
        R = np.array([self.top_settings['radii'].get(i, (Radii[Elements.index(i)-1] if i in Elements else 0.0)) for i in self.elem])
        # convert R from angstrom to Bohr unit.
        # Bohr unit is used in i-PI socket. 
        R = R * ang2bohr

        # Create a list of 2-tuples corresponding to combinations of atomic indices.
        # This is much faster than using itertools.combinations.
        # All unique atom pairs (i,j) where i < j in a Numpy array.
        # For example: self.na = 4, output: [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
        AtomIterator = np.ascontiguousarray(np.vstack(
                                                        (np.fromiter(
                                                            itertools.chain(
                                                                *[
                                                                    [i]*(self.na - i - 1) for i in range(self.na)
                                                                    ]
                                                                ),dtype=np.int32
                                                            ), 
                                                        np.fromiter(
                                                            itertools.chain(
                                                                *[range(i+1,self.na) for i in range(self.na)]
                                                                ),dtype=np.int32
                                                            )
                                                            )
                                                        ).T
                                                        )

        # Create a list of thresholds for determining whether a certain interatomic distance is considered to be a bond.
        BT0 = R[AtomIterator[:,0]]
        BT1 = R[AtomIterator[:,1]]
        BondThresh = (BT0+BT1) * Fac
        # mindist: any atom closer than this is bonded.
        # BondThresh = (BondThresh > mindist) * BondThresh + (BondThresh < mindist) * mindist  
        BondThresh = np.clip(BondThresh, a_min= mindist, a_max= None)

        # compute distance between atoms.
        dxij = AtomContact(self.xyz[np.newaxis, :], AtomIterator)[0]  

        # Create a list of atoms that each atom is bonded to.
        atom_bonds = [[] for i in range(self.na)]
        bond_bool = dxij < BondThresh

        for i, a in enumerate(bond_bool):
            if not a: continue
            # i : index for bond.
            # a : bool variable. If form the bond.
            # ii. jj: index for atoms in molecule.
            (ii, jj) = AtomIterator[i]
            if ii == jj: continue

            atom_bonds[ii].append(jj)
            atom_bonds[jj].append(ii)

        bondlist = []
        for i, bi in enumerate(atom_bonds):
            for j in bi:
                if i == j: continue

                if i < j:
                    bondlist.append((i, j))
                else:
                    bondlist.append((j, i))

        bondlist = sorted(list(set(bondlist)))

        # delete duplicate element in list and sort list.
        # This data can be accessed as self.bonds.
        self.Data['bonds'] = sorted(list(set(bondlist))) 

        self.built_bonds = True
