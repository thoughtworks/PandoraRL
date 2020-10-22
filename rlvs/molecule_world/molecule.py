import numpy as np
from scipy.spatial.transform import Rotation
from deepchem.feat.mol_graphs import ConvMol
from .rmsd import RMSD


class Molecule(ConvMol):
    '''
    Molecule represents collection of Atoms in 3D space
    - num_atoms: number of atoms in molecule
    - num_features: number of features per atom
    - atom_features: stack of features of each atom (shape = (num_atoms, num_features))
    - canon_adj_list: list of neighbors of each node (len = num_atoms)
    - origin: origin (x,y,z) w.r.t which all other coords are defined
    '''
    
    def __init__(self, atom_features, canon_adj_list, max_deg=10, min_deg=0, origin=[0,0,0], path=None):

        super(Molecule, self).__init__(atom_features=atom_features, adj_list=canon_adj_list, max_deg=max_deg, min_deg=min_deg)
        self.path = path
        self.origin = np.array(origin, copy=True).astype(float).reshape(1,3) #initially coords are wrt (0,0,0)
        self.T = lambda x,y,z : np.array([[1,0,0,x], [0,1,0,y], [0,0,1,z], [0,0,0,1]]).astype(float)
        self.rmsd = RMSD(self)

    def get_coords(self):
        return self.atom_features[:, 0:3]

    def set_coords(self, new_coords):
        assert new_coords.shape==(self.n_atoms, 3)
        self.atom_features[:, 0:3] = new_coords

    def randomize(self, box_size):
        x, y, z, r, p, y_ = np.random.uniform(-box_size, box_size, (6,)) * 10
        self.update_pose(x, y, z, r, p, y)
        
    def atom_range(self, axis):
        values = self.get_coords()[:,axis]
        minimum, maximum = min(values), max(values)
        return minimum, maximum, abs(maximum-minimum)
        
    def get_centroid(self):
        return self.get_coords().mean(axis=0)
    
    def homogeneous(self):
        ''' returns homogeneous form of coordinates (column major form) for translation and rotation matrix multiplication '''
        return np.concatenate((self.get_coords(), np.ones((self.get_coords().shape[0], 1))), axis = 1).T

    def update_pose(self, x, y, z, roll, pitch, yaw):
        old_coords = np.copy(self.get_coords())
        self.translate(x, y, z)
        self.rotate('xyz', [roll, pitch, yaw], True)
        new_coords = self.get_coords()
        delta_change = new_coords - old_coords
        return delta_change.mean(axis = 0)
        
    def translate(self,x,y,z):
        ''' translate along axes '''
        self.set_coords((self.T(x,y,z)@self.homogeneous()).T[:, 0:3])
        
    def rotate(self, axis_seq, angles, degrees):
        '''
        - parameters are the same as scipy's "from_euler" parameters.
        - rotations occur in order of axis sequence. 
        '''
        [x,y,z] = self.get_centroid()
        
        R = (Rotation.from_euler(seq=axis_seq, angles=angles, degrees=degrees)).as_matrix()
        R = np.concatenate((np.concatenate((R, np.array([0,0,0]).reshape(3, -1)), axis = 1), np.array([0,0,0,1]).reshape(-1, 4)), axis = 0)

        self.set_coords((self.T(x,y,z)@(R@(self.T(-x,-y,-z)@self.homogeneous()))).T[:,0:3])

        assert(self.get_coords().shape == (self.n_atoms, 3))
        
    def crop(self, center_coord, x, y, z):
        check_range = lambda x, min_x, max_x: True if (x >= min_x and x < max_x) else False
        center_coord = center_coord.reshape(1,3)
        min_x, max_x = center_coord[:,0][0] - x, center_coord[:,0][0] + x
        min_y, max_y = center_coord[:,1][0] - y, center_coord[:,1][0] + y
        min_z, max_z = center_coord[:,2][0] - z, center_coord[:,2][0] + z
        
        mask = np.zeros(self.n_atoms, dtype=bool)
        for i, (X,Y,Z) in enumerate(self.get_coords()):
            mask[i] = check_range(X, min_x, max_x) and check_range(Y, min_y, max_y) and check_range(Z, min_z, max_z)
            
        self.apply_crop_mask(mask)

    def apply_crop_mask(self, mask):
        self.atom_features = self.atom_features[mask, :]
        self.n_atoms = self.atom_features.shape[0]
    
    def summary(self):
        print(f"Number of heavy atoms in molecule = {self.n_atoms}")
        print(f"Number of features = {self.n_feat}")
        print(f"Max distance between atoms along\nx:{self.atom_range(0)}\ny:{self.atom_range(1)}\nz:{self.atom_range(2)}")
        
