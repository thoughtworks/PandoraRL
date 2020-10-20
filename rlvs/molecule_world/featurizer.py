from openbabel import pybel
from openbabel import openbabel as ob
import numpy as np
from .molecule import Molecule
import scipy.sparse as sp

class Featurizer():    
    def __init__(self):

        # dict of atom codes for one hot encoding
        self.atom_codes = {}
        self.class_codes = {}
        metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
                          + list(range(37, 51)) + list(range(55, 84))
                          + list(range(87, 104)))

        atom_classes = [
            (1, 'H'),
            (5, 'B'),
            (6, 'C'),
            (7, 'N'),
            (8, 'O'),
            (15, 'P'),
            (16, 'S'),
            (34, 'Se'),
            ([9, 17, 35, 53], 'halogen'),
            (metals, 'metal')
        ]

        for code, (atom, name) in enumerate(atom_classes):
            if type(atom) is list:
                for a in atom:
                    self.atom_codes[a] = code
            else:
                self.atom_codes[atom] = code
            self.class_codes[code] = name
        self.num_classes = len(atom_classes)

        self.SMARTS = ['[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
                '[a]',
                '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
                '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
                '[r]']
        self.smarts_labels = ['hydrophobic', 'aromatic', 'acceptor', 'donor',
                                    'ring']

        self.__PATTERNS = []
        for smarts in self.SMARTS:
            self.__PATTERNS.append(pybel.Smarts(smarts))

    def get_atom_features(self, atom, molecule_type):
        #TODO: add SMARTS patterns 
        '''
        INPUT
        atom: OB Atom object
        molecule_type: 1 for ligand, -1 for protein
        
        OUTPUT
        
        features:
            [
                x,
                y,
                z,
                encoding (10 bit one hot encoding),
                hyb (1,2 or 3),
                heavy_valence (integer),
                hetero_valence (integer),
                partial_charge (float),
                molecule_type (1 for ligand, -1 for protein)
            ]
        '''
        features = []
        
        # x y z coordinates
        features.extend([atom.GetX(), atom.GetY(), atom.GetZ()])

        # one hot encode atomic number
        encoding = np.zeros(self.num_classes)
        encoding[self.atom_codes[atom.GetAtomicNum()]] = 1
        features.extend(encoding)
        
        # hybridization, heavy valence, hetero valence, partial charge
        named_features = [atom.GetHyb(), atom.GetHvyDegree(), atom.GetHeteroDegree(), atom.GetPartialCharge()]
        features.extend(named_features)
        
        #molecule type
        molecule_type = molecule_type
        features.append(molecule_type)
        
        return features
    
    def get_mol_features(self, obmol, molecule_type, bond_verbose=False):
        
        num_atoms = obmol.NumAtoms()

        idx_node_tuples = []

        # Calculate SMARTS features
        mol_py = pybel.Molecule(obmol)
        smarts_patterns = self.find_smarts(mol_py) # shape = (num_atoms, num_smarts_patterns)

        for atom in ob.OBMolAtomIter(obmol):
            # add only heavy atoms
            # if atom.GetAtomicNum() > 1:
            #     atom_id = atom.GetIndex()
            #     atom_feats = self.get_atom_features(atom, molecule_type)
            #     idx_node_tuples.append((atom_id, atom_feats))

            atom_id = atom.GetIndex()
            atom_feats = self.get_atom_features(atom, molecule_type) # list of features
            
            atom_feats.extend(smarts_patterns[atom_id])

            idx_node_tuples.append((atom_id, atom_feats))

        idx_node_tuples.sort()
        idxs, nodes = list(zip(*idx_node_tuples))

        nodes = np.vstack(nodes)

        # Get bond lists with reverse edges included
        edge_list = [
            (bond.GetBeginAtom().GetIndex(), bond.GetEndAtom().GetIndex()) for bond in ob.OBMolBondIter(obmol)
        ]

        # Get canonical adjacency list
        canon_adj_list = [[] for mol_id in range(len(nodes))]
        for edge in edge_list:
            canon_adj_list[edge[0]].append(edge[1])
            canon_adj_list[edge[1]].append(edge[0])
        
        return nodes, canon_adj_list

    def find_smarts(self, molecule):
            """Find atoms that match SMARTS patterns.

            Parameters
            ----------
            molecule: pybel.Molecule

            Returns
            -------
            features: np.ndarray
                NxM binary array, where N is the number of atoms in the `molecule`
                and M is the number of patterns. `features[i, j]` == 1.0 if i'th
                atom has j'th property
            """


            features = np.zeros((len(molecule.atoms), len(self.__PATTERNS)))

            for (pattern_id, pattern) in enumerate(self.__PATTERNS):
                atoms_with_prop = np.array(list(*zip(*pattern.findall(molecule))),
                                        dtype=int) - 1
                features[atoms_with_prop, pattern_id] = 1.0
            return features
