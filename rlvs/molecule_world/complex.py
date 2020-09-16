import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import copy

class Complex:
    __GOOD_FIT = 0.006
    def __init__(self, protein, ligand, max_dist=100, resolution=1):
        '''
        max_dist : maximum distance between any atom and box center
        '''

        self.max_dist = max_dist
        self.resolution = resolution
        self.num_features = ligand.num_features

        self.protein = protein

        self.ligand = ligand
        self.__ligand = copy.deepcopy(ligand)
        
        self.box_size = int(np.ceil(2 * self.max_dist / self.resolution + 1))
        self.ligand.randomize(self.box_size)
        self.update_tensor()
        
        
    def convert_to_grid_coords(self, coords):
        
        assert(coords.shape[1]==3)
        return ((coords + self.max_dist) / self.resolution).round().astype(int)
        
    def new_3D_display(self, fullbox):

        fig = plt.figure(figsize=(4,4))
        self.ax = fig.add_subplot(111, projection='3d')
        if fullbox:
            self.ax.set_xlim(0, self.box_size)
            self.ax.set_ylim(0, self.box_size)
            self.ax.set_zlim(0, self.box_size)
            
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
    def update_tensor(self, update_mols=False):
        '''
        create a new 4D tensor

        - mols : list of molecule objects to join in same tensor

            '''

        invalid = False
        self.tensor4D = np.zeros((self.box_size, self.box_size, self.box_size, self.num_features))
        
        for mol in [self.protein, self.ligand]:
            grid_coords = self.convert_to_grid_coords(mol.coords)
            features = mol.features
            in_box = ((grid_coords >= 0) & (grid_coords < self.box_size)).all(axis=1)

            if update_mols:
                mol.apply_crop_mask(mask = in_box)

            if all(in_box)==False:
                invalid = True

            for (x,y,z), f in zip(grid_coords[in_box, :], features[in_box, :]):
                self.tensor4D[x,y,z] += f

        if invalid:
            raise Exception("Some atoms are outside the box and will be discarded.")

    def score(self):
        rmsd = self.ligand.rmsd(self.__ligand)
        return np.sinh(rmsd**0.7 + np.arcsinh(1))**-1

    @property
    def rmsd(self):
        return self.ligand.rmsd(self.__ligand)
    
    @property
    def perfect_fit(self):
        rmsd = self.ligand.rmsd(self.__ligand)
        return rmsd < self.__GOOD_FIT

    def plot_tensor(self, feature_axis, protein_alpha=0.1, protein_color='orange', ligand_alpha=1, ligand_color='blue',fullbox=1):

        self.new_3D_display(fullbox=fullbox)
        x,y,z = np.where(self.tensor4D[:,:,:,feature_axis])

        for coords in zip(x,y,z):
            if self.tensor4D[coords][feature_axis]==-1:
                self.ax.scatter(*coords, alpha=protein_alpha, color=protein_color)
            elif self.tensor4D[coords][feature_axis]==1:
                self.ax.scatter(*coords, alpha=ligand_alpha, color=ligand_color)
                
    def crop_tensor(self, center_coord, x, y, z):
        '''
        - assumes center is wrt 0,0,0
        '''

        grid_center_coord = self.convert_to_grid_coords(center_coord)
        transform = lambda x: int(round(x/self.resolution))
        x,y,z = transform(x), transform(y), transform(z)

        min_x, max_x = grid_center_coord[:,0][0] - x, grid_center_coord[:,0][0] + x
        min_y, max_y = grid_center_coord[:,1][0] - y, grid_center_coord[:,1][0] + y
        min_z, max_z = grid_center_coord[:,2][0] - z, grid_center_coord[:,2][0] + z

        print(min_x, max_x, min_y, max_y, min_z, max_z)
        print(self.tensor4D.shape)
        mask = np.zeros(self.tensor4D.shape[0:3], dtype=bool)
        mask[min_x:max_x, min_y:max_y, min_z:max_z] = True
        self.tensor4D[~mask] = np.zeros(self.tensor4D.shape[3])
        print(self.tensor4D.shape)
