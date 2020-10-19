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
        
    def score(self):
        rmsd = self.ligand.rmsd(self.__ligand)
        if rmsd > 8:
            raise Exception("BAD RMSD")

        return np.sinh(rmsd**0.7 + np.arcsinh(1))**-1

    @property
    def rmsd(self):
        return self.ligand.rmsd(self.__ligand)
    
    @property
    def perfect_fit(self):
        rmsd = self.ligand.rmsd(self.__ligand)
        return rmsd < self.__GOOD_FIT
