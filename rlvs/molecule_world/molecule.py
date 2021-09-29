import numpy as np
import torch

from openbabel import pybel
from openbabel import openbabel as ob

from scipy.spatial.transform import Rotation
from .rmsd import RMSD

from .helper_functions import read_to_OB
from .types import MoleculeType


class Molecule:
    '''
    Molecule represents collection of Atoms in 3D space
    - num_atoms: number of atoms in molecule
    - num_features: number of features per atom
    - atom_features: stack of features of each atom (shape = (num_atoms, num_features))
    - canon_adj_list: list of neighbors of each node (len = num_atoms)
    - origin: origin (x,y,z) w.r.t which all other coords are defined
    '''
    T = lambda x,y,z : np.array([[1,0,0,x], [0,1,0,y], [0,0,1,z], [0,0,0,1]]).astype(float)
    
    def __init__(self, path=None, filetype=None):
        self.path = path
        self.filetype=filetype
        self.rmsd = RMSD(self)
        self.pdb_structure = None
        self.atoms = None

    @property
    def data(self):
        return self.atoms

    def get_atom_features(self):
        return np.array(self.atoms.features)

    @property
    def n_atoms(self):
        return len(self.atoms)

    # [deprecated]
    def get_ordered_features(self):
        return self.atom_features[np.argsort(self.correct_order)]

    def get_coords(self):
        return self.atoms.coords

    def distance(self, coordinates):
        func = lambda features: np.linalg.norm(coordinates - features[:, :3], axis=1)
        return func(self.data.x)
    
    def set_coords(self, new_coords):
        assert new_coords.shape == (self.n_atoms, 3)
        self.atoms.coords = new_coords
        
    def atom_range(self, axis):
        values = self.get_coords()[:,axis]
        minimum, maximum = min(values), max(values)
        return minimum, maximum, abs(maximum-minimum)
        
    def get_centroid(self):
        return self.get_coords().mean(axis=0)
    
    def homogeneous(self):
        ''' returns homogeneous form of coordinates (column major form) for translation and rotation matrix multiplication '''
        return np.concatenate((self.get_coords(), np.ones((self.get_coords().shape[0], 1))), axis = 1).T

    def update_pose(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
        old_coords = np.copy(self.get_coords())
        self.translate(x, y, z)
        self.rotate('xyz', [roll, pitch, yaw], True)
        new_coords = self.get_coords()
        delta_change = np.abs(new_coords - old_coords)
        return delta_change.mean(axis = 0)
        
    def translate(self,x,y,z):
        ''' translate along axes '''
        self.set_coords((Molecule.T(x,y,z)@self.homogeneous()).T[:, 0:3])
        
    def rotate(self, axis_seq, angles, degrees):
        '''
        - parameters are the same as scipy's "from_euler" parameters.
        - rotations occur in order of axis sequence. 
        '''
        [x,y,z] = self.get_centroid()
        
        R = (Rotation.from_euler(seq=axis_seq, angles=angles, degrees=degrees)).as_matrix()
        R = np.concatenate((np.concatenate((R, np.array([0,0,0]).reshape(3, -1)), axis = 1), np.array([0,0,0,1]).reshape(-1, 4)), axis = 0)

        self.set_coords((Molecule.T(x,y,z)@(R@(Molecule.T(-x,-y,-z)@self.homogeneous()))).T[:,0:3])

        assert(self.get_coords().shape == (self.n_atoms, 3))
        
    def crop(self, center_coord, x, y, z):
        check_range = lambda x, min_x, max_x: (x >= min_x and x < max_x)
        center_coord = center_coord.reshape(1,3)
        min_x, max_x = center_coord[:,0][0] - x, center_coord[:,0][0] + x
        min_y, max_y = center_coord[:,1][0] - y, center_coord[:,1][0] + y
        min_z, max_z = center_coord[:,2][0] - z, center_coord[:,2][0] + z

        crop_condition = lambda atom: check_range(atom.coord[0], min_x, max_x) \
            and check_range(atom.coord[1], min_y, max_y) \
            and check_range(atom.coord[2], min_z, max_z)

        self.atoms.crop(crop_condition)

    def apply_crop_mask(self, mask):
        self.atom_features = self.atom_features[mask, :]
        self.n_atoms = self.atom_features.shape[0]
    
    def summary(self):
        print(f"Number of heavy atoms in molecule = {self.n_atoms}")
        print(f"Number of features = {self.n_feat}")
        print(f"Max distance between atoms along\nx:{self.atom_range(0)}\ny:{self.atom_range(1)}\nz:{self.atom_range(2)}")
        

    def save(self, output_path, output_type):
        obmol = read_to_OB(filename=self.path, filetype=self.filetype, prepare=True)
        assert(obmol.NumAtoms() == len(self.atoms))
        for atom in ob.OBMolAtomIter(obmol):
            atom.SetVector(*(self.atoms[atom.GetIndex()].coord))

        mol_py = pybel.Molecule(obmol)
        mol_py.write(format=output_type, filename=output_path, overwrite=True)
