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
                encoding (9 bit one hot encoding),
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

        # edges = np.zeros((num_atoms, num_atoms))
        # for bond in ob.OBMolBondIter(obmol):
        #     edges[bond.GetBeginAtom().GetIndex(), bond.GetEndAtom().GetIndex()] = 1
        #     edges[bond.GetEndAtom().GetIndex(), bond.GetBeginAtom().GetIndex()] = 1 # bi-directional

        # adj_mat = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        #                 shape=(num_atoms, num_atoms), dtype=np.float32)

        # adj_mat = adj_mat + adj_mat.T.multiply(adj_mat.T > adj_mat) - adj_mat.multiply(adj_mat.T > adj_mat)
            
        for atom in ob.OBMolAtomIter(obmol):
            # add only heavy atoms
            # if atom.GetAtomicNum() > 1:
            #     atom_id = atom.GetIndex()
            #     atom_feats = self.get_atom_features(atom, molecule_type)
            #     idx_node_tuples.append((atom_id, atom_feats))

            atom_id = atom.GetIndex()
            atom_feats = self.get_atom_features(atom, molecule_type)
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
