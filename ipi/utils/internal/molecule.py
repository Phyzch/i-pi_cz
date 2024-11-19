from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import itertools
import os
import re
import sys
import sysconfig
import json
from collections import OrderedDict, namedtuple, Counter
from ctypes import *
from datetime import date

import numpy as np
from numpy import sin, cos, arccos
from numpy.linalg import multi_dot
from ipi.utils.messages import warning, info 

try:
    import networkx as nx
except ImportError:
    warning("utils/internal/molecule.py: Cannot import optional NetworkX module, topology tools won't work.")



# ======================================================================#
# |                                                                    |#
# |              Chemical file format conversion module                |#
# |                                                                    |#
# |                Lee-Ping Wang (leeping@ucdavis.edu)                 |#
# |                    Last updated March 31, 2019                     |#
# |                                                                    |#
# |   This code is part of geomeTRIC and is covered under the          |#
# |   geomeTRIC copyright notice and BSD 3-clause license.             |#
# |   Please see https://github.com/leeping/geomeTRIC for details.     |#
# |                                                                    |#
# |   Special note:                                                    |#
# |   This file was copied over from ForceBalance to geomeTRIC         |#
# |   in order to lighten the dependencies of the latter.              |#
# |   Please make sure this file is up-to-date in                      |#
# |   both the 'geomeTRIC' and 'forcebalance' modules.                 |#
# |                                                                    |#
# |   Feedback and suggestions are encouraged.                         |#
# |                                                                    |#
# |   What this is for:                                                |#
# |   Converting a molecule between file formats                       |#
# |   Loading and processing of trajectories                           |#
# |   (list of geometries for the same set of atoms)                   |#
# |   Concatenating or slicing trajectories                            |#
# |   Combining molecule metadata (charge, Q-Chem rem variables)       |#
# |                                                                    |#
# |   Supported file formats:                                          |#
# |   See the __init__ method in the Molecule class.                   |#
# |                                                                    |#
# |   Note to self / developers:                                       |#
# |   Please make this file as standalone as possible                  |#
# |   (i.e. don't introduce dependencies).  If we load an external     |#
# |   library to parse a file, do so with 'try / except' so that       |#
# |   the module is still usable even if certain parts are missing.    |#
# |   It's better to be like a Millennium Falcon. :P                   |#
# |                                                                    |#
# |   At present, when I perform operations like adding two objects,   |#
# |   the sum is created from deep copies of data members in the       |#
# |   originals. This is because copying by reference is confusing;    |#
# |   suppose if I do B += A and modify something in B; it should not  |#
# |   change in A.                                                     |#
# |                                                                    |#
# |   A consequence of this is that data members should not be too     |#
# |   complicated; they should be things like lists or dicts, and NOT  |#
# |   contain references to themselves.                                |#
# |                                                                    |#
# |   To-do list: Handling of comments is still not very good.         |#
# |   Comments from previous files should be 'passed on' better.       |#
# |                                                                    |#
# |              Contents of this file:                                |#
# |              0) Names of data variables                            |#
# |              1) Imports                                            |#
# |              2) Subroutines                                        |#
# |              3) Molecule class                                     |#
# |                a) Class customizations (add, getitem)              |#
# |                b) Instantiation                                    |#
# |                c) Core functionality (read, write)                 |#
# |                d) Reading functions                                |#
# |                e) Writing functions                                |#
# |                f) Extra stuff                                      |#
# |              4) "main" function (if executed)                      |#
# |                                                                    |#
# |                   Required: Python 2.7 or 3.6                      |#
# |                             (2.6, 3.5 and earlier untested)        |#
# |                             NumPy 1.6                              |#
# |                   Optional: Mol2, PDB, DCD readers                 |#
# |                    (can be found in ForceBalance)                  |#
# |                    NetworkX package (for topologies)               |#
# |                                                                    |#
# |             Thanks: Todd Dolinsky, Yong Huang,                     |#
# |                     Kyle Beauchamp (PDB)                           |#
# |                     John Stone (DCD Plugin)                        |#
# |                     Pierre Tuffery (Mol2 Plugin)                   |#
# |                     #python IRC chat on FreeNode                   |#
# |                                                                    |#
# |             Contributors: Leah Isseroff Bendavid                   |#
# |                           Yudong Qiu                               |#
# |                                                                    |#
# |             Instructions:                                          |#
# |                                                                    |#
# |               To import:                                           |#
# |                 from molecule import Molecule                      |#
# |               To create a Molecule object:                         |#
# |                 MyMol = Molecule(fnm)                              |#
# |               To convert to a new file format:                     |#
# |                 MyMol.write('newfnm.format')                       |#
# |               To concatenate geometries:                           |#
# |                 MyMol += MyMolB                                    |#
# |                                                                    |#
# ======================================================================#

# =========================================#
# |     DECLARE VARIABLE NAMES HERE       |#
# |                                       |#
# |  Any member variable in the Molecule  |#
# | class must be declared here otherwise |#
# | the Molecule class won't recognize it |#
# =========================================#
# | Data attributes in FrameVariableNames |#
# | must be a list along the frame axis,  |#
# | and they must have the same length.   |#
# =========================================#
# xyzs       = List of arrays of atomic xyz coordinates
# comms      = List of comment strings
# boxes      = List of 3-element or 9-element arrays for periodic boxes
# qm_grads   = List of arrays of gradients (i.e. negative of the atomistic forces) from QM calculations
# qm_espxyzs = List of arrays of xyz coordinates for ESP evaluation
# qm_espvals = List of arrays of ESP values
# qm_zpe     = Zero point energy, kcal/mol (from a qchem freq calculation)
# qm_entropy = Entropy contribution at STP, cal/mol.K (from a qchem freq calculation)
# qm_enthalpy= Enthalpic contribution at STP, excluding electronic energy and ZPE, kcal/mol (from a qchem freq calculation)

FrameVariableNames = {'xyzs', 'comms', 'boxes', 'qm_hessians', 'qm_grads', 'qm_energies', 'qm_interaction',
                      'qm_espxyzs', 'qm_espvals', 'qm_extchgs', 'qm_mulliken_charges', 'qm_mulliken_spins', 'qm_zpe',
                      'qm_entropy', 'qm_enthalpy', 'qm_bondorder'}
#=========================================#
#| Data attributes in AtomVariableNames  |#
#| must be a list along the atom axis,   |#
#| and they must have the same length.   |#
#=========================================#
# elem       = List of elements
# partial_charge = List of atomic partial charges
# atomname   = List of atom names (can come from MM coordinate file)
# atomtype   = List of atom types (can come from MM force field)
# tinkersuf  = String that comes after the XYZ coordinates in TINKER .xyz or .arc files
# resid      = Residue IDs (can come from MM coordinate file)
# resname    = Residue names
# terminal   = List of true/false denoting whether this atom is followed by a terminal group.
AtomVariableNames = {'elem', 'partial_charge', 'atomname', 'atomtype', 'tinkersuf', 'resid', 'resname', 'qcsuf',
                     'qm_ghost', 'chain', 'altloc', 'icode', 'terminal'}
#=========================================#
#| This can be any data attribute we     |#
#| want but it's usually some property   |#
#| of the molecule not along the frame   |#
#| atom axis.                            |#
#=========================================#
# bonds      = A list of 2-tuples representing bonds.  Carefully pruned when atom subselection is done.
# fnm        = The file name that the class was built from
# qcrems     = The Q-Chem 'rem' variables stored as a list of OrderedDicts
# qctemplate = The Q-Chem template file, not including the coordinates or rem variables
# charge     = The net charge of the molecule
# mult       = The spin multiplicity of the molecule
MetaVariableNames = {'fnm', 'ftype', 'qcrems', 'qctemplate', 'qcerr', 'charge', 'mult', 'bonds', 'topology',
                     'molecules'}
# Variable names relevant to quantum calculations explicitly
QuantumVariableNames = {'qcrems', 'qctemplate', 'charge', 'mult', 'qcsuf', 'qm_ghost', 'qm_energies', 'qm_grads', 'qm_hessians',
                        'qm_interaction', 'qm_espxyzs', 'qm_espvals', 'qm_extchgs', 'qm_mulliken_charges', 'qm_mulliken_spins',
                        'qm_zpe', 'qm_entropy', 'qm_enthalpy','qm_bondorder'}
# Superset of all variable names.
AllVariableNames = QuantumVariableNames | AtomVariableNames | MetaVariableNames | FrameVariableNames


# Covalent radii from Cordero et al. 'Covalent radii revisited' Dalton Transactions 2008, 2832-2838.
Radii = [0.31, 0.28, # H and He
         1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, # First row elements
         0.00, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, # Second row elements
         # 1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, # Second row elements
         2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.61, 1.52, 1.50,
         1.24, 1.32, 1.22, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16, # Third row elements, K through Kr
         2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42,
         1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40, # Fourth row elements, Rb through Xe
         2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98,
         1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, # Fifth row elements, s and f blocks
         1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36,
         1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50, # Fifth row elements, d and p blocks
         2.60, 2.21, 2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69] # Sixth row elements

# A list that gives you the element if you give it the atomic number, hence the 'none' at the front.
Elements = ["None",'H','He',
            'Li','Be','B','C','N','O','F','Ne',
            'Na','Mg','Al','Si','P','S','Cl','Ar',
            'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr',
            'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe',
            'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb',
            'Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn',
            'Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt']

# Dictionary of atomic masses ; also serves as the list of elements (periodic table)
#
# Atomic mass data was updated on 2020-05-07 from NIST:
# "Atomic Weights and Isotopic Compositions with Relative Atomic Masses"
# https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses
# using All Elements -> preformatted ASCII table.
#
# The standard atomic weight was provided in several different formats:
# Two numbers in brackets as in [1.00784,1.00811] : The average value of the two limits is used.
# With parentheses(uncert) as in 4.002602(2) : The parentheses was split off and all significant digits are used.
# A single number in brackets as in [98] : The single number was used
# Not provided (for Am, Z=95 and up): The mass number of the lightest isotope was used
PeriodicTable = OrderedDict([("H", 1.007975), ("He", 4.002602), # First row
                             ("Li", 6.9675), ("Be", 9.0121831), ("B", 10.8135), ("C", 12.0106), ("N", 14.006855), ("O", 15.99940), ("F", 18.99840316), ("Ne", 20.1797), # Second row Li-Ne
                             ("Na", 22.98976928), ("Mg", 24.3055), ("Al", 26.9815385), ("Si", 28.085), ("P", 30.973762), ("S", 32.0675), ("Cl", 35.4515), ("Ar", 39.948), # Third row Na-Ar
                             ("K", 39.0983), ("Ca", 40.078), ("Sc", 44.955908), ("Ti", 47.867), ("V", 50.9415), ("Cr", 51.9961), ("Mn", 54.938044), ("Fe", 55.845), ("Co", 58.933194), # Fourth row K-Kr
                             ("Ni", 58.6934), ("Cu", 63.546), ("Zn", 65.38), ("Ga", 69.723), ("Ge", 72.63), ("As", 74.921595), ("Se", 78.971), ("Br", 79.904), ("Kr", 83.798),
                             ("Rb", 85.4678), ("Sr", 87.62), ("Y", 88.90584), ("Zr", 91.224), ("Nb", 92.90637), ("Mo", 95.95), ("Tc", 98.), ("Ru", 101.07), ("Rh", 102.9055), # Fifth row Rb-Xe
                             ("Pd", 106.42), ("Ag", 107.8682), ("Cd", 112.414), ("In", 114.818), ("Sn", 118.71), ("Sb", 121.76), ("Te", 127.6), ("I", 126.90447), ("Xe", 131.293),
                             ("Cs", 132.905452), ("Ba", 137.327), ("La", 138.90547), ("Ce", 140.116), ("Pr", 140.90766), ("Nd", 144.242), ("Pm", 145.), ("Sm", 150.36), # Sixth row Cs-Rn
                             ("Eu", 151.964), ("Gd", 157.25), ("Tb", 158.92535), ("Dy", 162.5), ("Ho", 164.93033), ("Er", 167.259), ("Tm", 168.93422), ("Yb", 173.054),
                             ("Lu", 174.9668), ("Hf", 178.49), ("Ta", 180.94788), ("W", 183.84), ("Re", 186.207), ("Os", 190.23), ("Ir", 192.217), ("Pt", 195.084),
                             ("Au", 196.966569), ("Hg", 200.592), ("Tl", 204.3835), ("Pb", 207.2), ("Bi", 208.9804), ("Po", 209.), ("At", 210.), ("Rn", 222.),
                             ("Fr", 223.), ("Ra", 226.), ("Ac", 227.), ("Th", 232.0377), ("Pa", 231.03588), ("U", 238.02891), ("Np", 237.), ("Pu", 244.), # Seventh row Fr-Og
                             ("Am", 241.), ("Cm", 243.), ("Bk", 247.), ("Cf", 249.), ("Es", 252.), ("Fm", 257.), ("Md", 258.), ("No", 259.),
                             ("Lr", 262.), ("Rf", 267.), ("Db", 268.), ("Sg", 271.), ("Bh", 272.), ("Hs", 270.), ("Mt", 276.), ("Ds", 281.),
                             ("Rg", 280.), ("Cn", 285.), ("Nh", 284.), ("Fl", 289.), ("Mc", 288.), ("Lv", 293.), ("Ts", 292.), ("Og", 294.)])

def getElement(mass):
    return PeriodicTable.keys()[np.argmin([np.abs(m-mass) for m in PeriodicTable.values()])]

def elem_from_atomname(atomname):
    """ Given an atom name, attempt to get the element in most cases. """
    return re.search('[A-Z][a-z]*',atomname).group(0)


def nodematch(node1,node2):
    # Matching two nodes of a graph.  Nodes are equivalent if the elements are the same
    return node1['e'] == node2['e']

def cartesian_product2(arrays):
    """ Form a Cartesian product of two NumPy arrays. """
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=np.int32)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def AtomContact(xyz, pairs, box=None, displace=False):
    """
    Compute distances between pairs of atoms.

    Parameters
    ----------
    xyz : np.ndarray
        N_frames*N_atoms*3 (3D) array of atomic positions
        If you only have a single set of positions, pass in xyz[np.newaxis, :]
    pairs : list
        List of 2-tuples of atom indices
    box : np.ndarray, optional
        N_frames*3 (2D) array of periodic box vectors
        If you only have a single set of positions, pass in box[np.newaxis, :]
    displace : bool
        If True, also return N_frames*N_pairs*3 array of displacement vectors

    Returns
    -------
    np.ndarray
        N_frames*N_pairs (2D) array of minimum image convention distances
    np.ndarray (optional)
        if displace=True, N_frames*N_pairs*3 array of displacement vectors
    """
    # Obtain atom selections for atom pairs
    parray = np.array(pairs)
    sel1 = parray[:,0]
    sel2 = parray[:,1]
    xyzpbc = xyz.copy()
    # Minimum image convention: Place all atoms in the box
    # [0, xbox); [0, ybox); [0, zbox)
    if box is not None:
        xyzpbc /= box[:,np.newaxis,:]
        xyzpbc = xyzpbc % 1.0
    # Obtain atom selections for the pairs to be computed
    # These are typically longer than N but shorter than N^2.
    xyzsel1 = xyzpbc[:,sel1,:]
    xyzsel2 = xyzpbc[:,sel2,:]
    # Calculate xyz displacement
    dxyz = xyzsel2-xyzsel1
    # Apply minimum image convention to displacements
    if box is not None:
        dxyz = np.mod(dxyz+0.5, 1.0) - 0.5
        dxyz *= box[:,np.newaxis,:]
    dr2 = np.sum(dxyz**2,axis=2)
    dr = np.sqrt(dr2)
    if displace:
        return dr, dxyz
    else:
        return dr


class MyG(nx.Graph):
    """
    graph representing connectivity of atoms in molecules.
    For each node: 
    self.nodes() : atom. (represented by number.)
    self.nodes[i]['e']: elements of atom: 'H', 'C', 'O', 'N'
    self.nodes[i]['x']: coordinates of the atom. 
    self.nodes[i]['n']: atom names
    """
    def __init__(self):
            super(MyG,self).__init__()
    def __eq__(self, other):
        # This defines whether two MyG objects are "equal" to one another.
        return nx.is_isomorphic(self, other, node_match=nodematch)
    def __hash__(self):
        """ The hash function is something we can use to discard two things that are obviously not equal.  Here we neglect the hash. """
        return 1
    def L(self):
        """ Return a list of the sorted atom numbers in this graph. """
        return sorted(list(self.nodes()))
    def AStr(self):
        """ Return a string of atoms, which serves as a rudimentary 'fingerprint' : '99,100,103,151' . """
        return ','.join(['%i' % i for i in self.L()])
    def e(self):
        """ Return an array of the elements.  For instance ['H' 'C' 'C' 'H']. """
        elems = nx.get_node_attributes(self,'e')
        return [elems[i] for i in self.L()]
    def ef(self):
        """ Create an Empirical Formula For example: C2H4"""
        Formula = list(self.e())
        return ''.join([('%s%i' % (k, Formula.count(k)) if Formula.count(k) > 1 else '%s' % k) for k in sorted(set(Formula))])
    def x(self):
        """ Get a list of the coordinates. """
        coors = nx.get_node_attributes(self,'x')
        return np.array([coors[i] for i in self.L()])


class Molecule(object):
    """ From Lee-Ping Wang's general file format conversion class.
    Unit system:  Angstroms.
    
    self.na: number of atoms.
    self.xyz: coordinate of atoms in molecules. [Natoms, 3] 2d array.
    self.elem: elements of atoms in molecules
    """

    def __init__(self,
                 natoms: int, 
                 xyz: np.ndarray, 
                 elem: list,
                 **kwargs):
        """
        Create a Molecule object.

        Parameters
        ----------
        natoms: number of atoms.
        xyz: xyz coordinate of atoms in the molecule. shape: [3 * natoms]. Need to reshape it to [Natoms, 3] when set it to self.xyz.
        elem: elements of atoms. In ipi, this is self.beads.names
        build_topology : bool, optional
            Build the molecular topology consisting of: topology (overall connectivity graph),
            molecules (list of connected subgraphs), bonds (if not explicitly read in), default True
        Fac : float, optional
            Multiplicative factor to covalent radii criterion for deciding whether two atoms are bonded
            Default value of 1.2 is reasonable, 1.4 will produce lots of bonds
        """
        # bool that whether we have built bonds.
        self.built_bonds = False

        # number of atoms.
        self.na = natoms
        self.xyz = np.reshape(xyz, [natoms, 3])
        self.elem = elem 

        ## Topology settings
        self.top_settings = {'Fac' : kwargs.get('Fac', 1.2),
                             'read_bonds' : False,
                             'fragment' : kwargs.get('fragment', False),
                             'radii' : kwargs.get('radii', {})}

        # Data container.  All of the data is stored in here.
        self.Data = {}

    def __getattr__(self, key):
        """ Whenever we try to get a class attribute, it first tries to get the attribute from the Data dictionary. """
        if key in self.Data:
            return self.Data[key]
        
        return getattr(super(Molecule, self), key)

    def __setattr__(self, key, value):
        """ Whenever we try to get a class attribute, it first tries to get the attribute from the Data dictionary. """
        ## These attributes return a list of attribute names defined in this class, that belong in the chosen category.
        ## For example: self.FrameKeys should return set(['xyzs','boxes']) if xyzs and boxes exist in self.Data
        if key in AllVariableNames:
            self.Data[key] = value
        return super(Molecule,self).__setattr__(key, value)


    def build_bonds(self):
        """ 
        Build the bond connectivity graph.
        Simplified version without using grid algorithm.
        """
        Fac = self.top_settings['Fac']  # 1.2 by default.
        mindist = 1.0 # Any two atoms that are closer than this distance are bonded.

        # Create an atom-wise list of covalent radii.
        # Molecule object can have its own set of radii that overrides the global ones
        # Here .get(a, b) will return top_settings['radii'] if a exist, otherwise return b, which is default radii.
        R = np.array([self.top_settings['radii'].get(i, (Radii[Elements.index(i)-1] if i in Elements else 0.0)) for i in self.elem])

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
        BondThresh = np.clip(BondThresh, a_min= mindist)

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


    def build_topology(self, **kwargs):
        """
        Create self.topology and self.molecules; these are graph
        representations of the individual molecules (fragments)
        contained in the Molecule object.

        Adopted from geomeTRIC molecule.py:build_topology(). 
        Simplify the code by only providing options to build graph using interatomic distances.

        Parameters
        ----------
        """
        if self.na > 100000:
            warning("Warning: Large number of atoms (%i), topology building may take a long time" % self.na)

        # Build bonds from connectivity graph.
        self.build_bonds()

        # Create a NetworkX graph object to hold the bonds. Use self.bonds created in self.build_bonds()
        G = MyG()
        for i, a in enumerate(self.elem):
            G.add_node(i)
            if 'atomname' in self.Data:
                nx.set_node_attributes(G,{i: self.atomname[i]}, name='n')  # atom name
            nx.set_node_attributes(G,{i: a}, name='e')  # atom element. "H", "C", "O"
            nx.set_node_attributes(G,{i:self.xyz[i]}, name='x') # atom coordinate.

        for (i, j) in self.bonds:  # here self.bonds: is self.Data['bonds']. see __getattr__ func
            G.add_edge(i, j)
        # The Topology is simply the NetworkX graph object.
        self.topology = G
        # LPW: Molecule.molecules is a funny misnomer... it should be fragments or substructures or something
        self.molecules = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        for g in self.molecules: g.__class__ = MyG

