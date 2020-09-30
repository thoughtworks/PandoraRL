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
    
    def __init__(self, atom_features, canon_adj_list, max_deg=10, min_deg=0, origin=[0,0,0]):

        super(Molecule, self).__init__(atom_features=atom_features, adj_list=canon_adj_list, max_deg=max_deg, min_deg=min_deg)
      
        self.origin = np.array(origin, copy=True).astype(float).reshape(1,3) #initially coords are wrt (0,0,0)
        self.T = lambda x,y,z : np.array([[1,0,0,x], [0,1,0,y], [0,0,1,z], [0,0,0,1]]).astype(float)
        self.rmsd = RMSD(self)

    def randomize(self, box_size):
        x, y_ = np.random.uniform(0, box_size/4, (2,))
        self.update_pose(x, y_, 0, 0,0,0)
        
    def atom_range(self, axis):
        values = self.coords[:,axis]
        minimum, maximum = min(values), max(values)
        return minimum, maximum, abs(maximum-minimum)
                          
    # def shift_origin(self, new_origin):
    #     self.origin = np.array(new_origin, copy=True).astype(float).reshape(1,3)
    #     self.coords -= self.origin
    #     assert(self.coords.shape == (self.num_atoms, 3))
        
    def get_centroid(self):
        return self.coords.mean(axis=0)
    
    def homogeneous(self):
        ''' returns homogeneous form of coordinates (column major form) for translation and rotation matrix multiplication '''
        return np.concatenate((self.coords, np.ones((self.coords.shape[0], 1))), axis = 1).T

    def update_pose(self, x, y, z, roll, pitch, yaw):
        self.translate(x, y, z)
        self.rotate('xyz', [roll, pitch, yaw], True)
        
    def translate(self,x,y,z):
        ''' translate along axes '''
        self.coords = (self.T(x,y,z)@self.homogeneous()).T[:, 0:3]
        
    def rotate(self, axis_seq, angles, degrees):
        '''
        - parameters are the same as scipy's "from_euler" parameters.
        - rotations occur in order of axis sequence. 
        '''
        [x,y,z] = self.get_centroid()
        
        R = (Rotation.from_euler(seq=axis_seq, angles=angles, degrees=degrees)).as_matrix()
        R = np.concatenate((np.concatenate((R, np.array([0,0,0]).reshape(3, -1)), axis = 1), np.array([0,0,0,1]).reshape(-1, 4)), axis = 0)

        self.coords = (self.T(x,y,z)@(R@(self.T(-x,-y,-z)@self.homogeneous()))).T[:,0:3]

        assert(self.coords.shape == (self.num_atoms, 3))
        
    def crop(self, center_coord, x, y, z):
        check_range = lambda x, min_x, max_x: True if (x >= min_x and x < max_x) else False
        center_coord = center_coord.reshape(1,3)
        min_x, max_x = center_coord[:,0][0] - x, center_coord[:,0][0] + x
        min_y, max_y = center_coord[:,1][0] - y, center_coord[:,1][0] + y
        min_z, max_z = center_coord[:,2][0] - z, center_coord[:,2][0] + z
        
        mask = np.zeros(self.num_atoms, dtype=bool)
        for i, (X,Y,Z) in enumerate(self.coords):
            mask[i] = check_range(X, min_x, max_x) and check_range(Y, min_y, max_y) and check_range(Z, min_z, max_z)
            
        self.apply_crop_mask(mask)

    def apply_crop_mask(self, mask):
        self.coords = self.coords[mask, :]
        self.features = self.features[mask, :]
        self.num_atoms = self.features.shape[0]
        
        
    def set_in_grid(self, grid):
        grid_coords = grid.convert_to_grid_coords(self.coords)
        for (x,y,z), f in zip(grid_coords, self.features):
            grid.tensor4D[x,y,z] += f
            
    def plot(self, grid, s=20, alpha=1, color='r'):
        grid_coords = grid.convert_to_grid_coords(self.coords)
        origin = grid.convert_to_grid_coords(self.origin)
        X, Y, Z = [], [], []
        for (x,y,z) in grid_coords:
            X.append(x)
            Y.append(y)
            Z.append(z)
        grid.ax.scatter(X,Y,Z, s=s, alpha=alpha, color=color)
        grid.ax.scatter([origin[0,0]], [origin[0,1]], [origin[0,2]], s=s, alpha=1, color=color, marker = "+")
        #print(f"Plotting X={X}, Y={Y}, Z={Z}")
    
    def summary(self):
        print(f"Number of heavy atoms in molecule = {self.num_atoms}")
        print(f"Number of features = {self.num_features}")
        print(f"Max distance between atoms along\nx:{self.atom_range(0)}\ny:{self.atom_range(1)}\nz:{self.atom_range(2)}")
        
