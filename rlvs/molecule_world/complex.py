import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import copy
from rlvs.agents.utils import batchify, interacting_edges, \
    molecule_median_distance, timeit
from torch_geometric.data import Data
import torch
from rlvs.constants import ComplexConstants

class Complex:
    def __init__(self, protein, ligand, original_ligand, interacting_edges=None):
        '''
        max_dist : maximum distance between any atom and box center
        '''
        # self.num_features = ligand.n_feat

        self.protein = protein

        self.ligand = ligand
        self.original_ligand = original_ligand 
        self.interacting_edges = interacting_edges
        self.update_interacting_edges()
        self.previous_rmsd = None

    def crop(self, x, y, z):
        self.protein.crop(self.ligand.get_centroid(), x, y, z)
        
    def score(self):
        print(
             "complex Stats: interacting Edges: ", self.interacting_edges.shape,
             "Ligand Shape", self.ligand.data.x.shape,
             "Protein Shape", self.protein.data.x.shape
        )
        rmsd = self.ligand.rmsd(self.original_ligand)
        score = np.sinh(rmsd**0.25 + np.arcsinh(1))**-1
        multiplier = 1
        if self.previous_rmsd is not None:
            multiplier = -1 if self.previous_rmsd < rmsd else 1

        self.previous_rmsd = rmsd
        
        if rmsd > ComplexConstants.RMSD_THRESHOLD:
            raise Exception("BAD RMSD")

        if multiplier > 0:
            if rmsd < 7:
                multiplier = 5
            if rmsd < 5:
                multiplier = 7.5
            if rmsd < 3:
                multiplier = 10
            if rmsd < 2:
                multiplier = 100
                
        return multiplier * score

    def update_interacting_edges(self):
        if self.interacting_edges is not None:
            print(
            "complex Stats: interacting Edges: ", self.interacting_edges.shape,
            "Ligand Shape", self.ligand.data.x.shape,
            "Protein Shape", self.protein.data.x.shape
            )

            return
        
        self.interacting_edges = interacting_edges(
            self.protein, self.ligand
        )

    @property
    def rmsd(self):
        return self.ligand.rmsd(self.original_ligand)

    @property
    def perfect_fit(self):
        rmsd = self.ligand.rmsd(self.original_ligand)
        return rmsd < ComplexConstants.GOOD_FIT

    @property
    def data(self):
        batched = batchify([self.protein, self.ligand], data=False)
        edge_index = torch.hstack([
            batched.edge_index,
            self.interacting_edges
            ])
        batch = torch.tensor([0] * batched.x.shape[0])
        return Data(x=batched.x.detach().clone(), edge_index=edge_index.detach().clone(), batch=batch)
