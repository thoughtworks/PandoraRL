import numpy as np
import h5py
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from openbabel import pybel
import os


def center_molecule(coords):
    '''
    shift origin to the center of molecule
    '''
    centroid = coords.mean(axis = 0).reshape(-1, 3)
    shifted_coords = coords - centroid
    
    return shifted_coords

def molecule_to_grid(max_dist, resolution, coords, features, display=False):
    '''
    create a 3d grid (a box)
    
    max_dist: maximum distance between any atom and box center
    '''
    num_features = features.shape[1]
    box_size = int(np.ceil(2 * max_dist / resolution + 1))
    grid = np.zeros((box_size, box_size, box_size, num_features))
    
    #grid_coords = (coords + max_dist) / resolutions
    
    grid_coords = coords.round().astype(int)
    X, Y, Z,atom_type = [], [], [], []

    for (x,y,z), f in zip(grid_coords, features):
        X.append(x)
        Y.append(y)
        Z.append(z)
        atom_type.append(f[0]) #first feature channel
        grid[x,y,z] += f
        
    if display:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')

        img = ax.scatter(X,Y,Z, c = atom_type, s = 100)
        fig.colorbar(img)
        plt.show()
        
    return grid

class Featurizer():    
    def __init__(self):

        # dict of atom codes for one hot encoding
        self.atom_codes = {}
        self.class_codes = {}
        metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
                          + list(range(37, 51)) + list(range(55, 84))
                          + list(range(87, 104)))

        atom_classes = [
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
        atom: pybel Atom object
        molecule_type: 1 for ligand, -1 for protein
        
        OUTPUT
        [(x,y,z), [features]] 
        where features:
            - atom code (TEMPORARY FEATURE)
            - encoding (9 bit one hot encoding)
            - hyb (1,2 or 3)
            - heavy_valence (integer)
            - hetero_valence (integer)
            - partial_charge (float)
            - molecule_type (1 for ligand, -1 for protein)
        '''
        features = []
        
        features.append(self.atom_codes[atom.atomicnum])
        
        # one hot encode atomic number
        encoding = np.zeros(self.num_classes)
        encoding[self.atom_codes[atom.atomicnum]] = 1
        features.extend(encoding)
        
        # hybridization, heavy valence, hetero valence, partial charge
        named_features = [atom.hyb, atom.heavydegree, atom.heterodegree, atom.partialcharge]
        features.extend(named_features)
        
        #molecule type
        molecule_type = molecule_type
        features.append(molecule_type)
        
        return atom.coords, features
    
    def get_mol_features(self, mol, molecule_type):
        num_atoms = len(mol.atoms)
        coords, features = [], []
        for atom in mol.atoms:
            # add only heavy atoms
            if atom.atomicnum > 1:
                crds, feats = self.get_atom_features(atom, molecule_type)
                coords.append(crds)
                features.append(feats)
        coords = np.array(coords)
        features = np.array(features)

        return coords, features
