import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import copy

class Complex:
    __GOOD_FIT = 0.006
    def __init__(self, protein, ligand):
        '''
        max_dist : maximum distance between any atom and box center
        '''
        self.num_features = ligand.n_feat

        self.protein = protein

        self.ligand = ligand
        self.__ligand = copy.deepcopy(ligand) 
        self.ligand.randomize(10) # TODO: move it out of the protein
        
    # def convert_to_grid_coords(self, coords):
        
    #     assert(coords.shape[1]==3)
    #     return ((coords + self.max_dist) / self.resolution).round().astype(int)
        
    # def new_3D_display(self, fullbox):

    #     fig = plt.figure(figsize=(4,4))
    #     self.ax = fig.add_subplot(111, projection='3d')
    #     if fullbox:
    #         self.ax.set_xlim(0, self.box_size)
    #         self.ax.set_ylim(0, self.box_size)
    #         self.ax.set_zlim(0, self.box_size)
            
    #     self.ax.set_xlabel('X')
    #     self.ax.set_ylabel('Y')
    #     self.ax.set_zlabel('Z')
        
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

    # def plot_tensor(self, feature_axis, protein_alpha=0.1, protein_color='orange', ligand_alpha=1, ligand_color='blue',fullbox=1):

    #     self.new_3D_display(fullbox=fullbox)
    #     x,y,z = np.where(self.tensor4D[:,:,:,feature_axis])

    #     for coords in zip(x,y,z):
    #         if self.tensor4D[coords][feature_axis]==-1:
    #             self.ax.scatter(*coords, alpha=protein_alpha, color=protein_color)
    #         elif self.tensor4D[coords][feature_axis]==1:
    #             self.ax.scatter(*coords, alpha=ligand_alpha, color=ligand_color)
                
    # def crop_tensor(self, center_coord, x, y, z):
    #     '''
    #     - assumes center is wrt 0,0,0
    #     '''

    #     grid_center_coord = self.convert_to_grid_coords(center_coord)
    #     transform = lambda x: int(round(x/self.resolution))
    #     x,y,z = transform(x), transform(y), transform(z)

    #     min_x, max_x = grid_center_coord[:,0][0] - x, grid_center_coord[:,0][0] + x
    #     min_y, max_y = grid_center_coord[:,1][0] - y, grid_center_coord[:,1][0] + y
    #     min_z, max_z = grid_center_coord[:,2][0] - z, grid_center_coord[:,2][0] + z

    #     print(min_x, max_x, min_y, max_y, min_z, max_z)
    #     print(self.tensor4D.shape)
    #     mask = np.zeros(self.tensor4D.shape[0:3], dtype=bool)
    #     mask[min_x:max_x, min_y:max_y, min_z:max_z] = True
    #     self.tensor4D[~mask] = np.zeros(self.tensor4D.shape[3])
    #     print(self.tensor4D.shape)
