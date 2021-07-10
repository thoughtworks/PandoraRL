import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import copy
from rlvs.agents.utils import batchify
from torch_geometric.data import Data
import torch

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
        
    def score(self):
        rmsd = self.ligand.rmsd(self.__ligand)
        if rmsd > 200:
            raise Exception("BAD RMSD")
        if rmsd > 100:
            return 0
        return np.sinh(rmsd**0.25 + np.arcsinh(1))**-1

    @property
    def rmsd(self):
        return self.ligand.rmsd(self.__ligand)
    
    @property
    def perfect_fit(self):
        rmsd = self.ligand.rmsd(self.__ligand)
        return rmsd < self.__GOOD_FIT

    @property
    def data(self):
        batched = batchify([self.protein, self.ligand])
        # add edges based on distnace
        batch = torch.tensor([0] * batched.x.shape[0])
        return Data(x=batched.x, edge_index=batched.edge_index, batch=batch)
