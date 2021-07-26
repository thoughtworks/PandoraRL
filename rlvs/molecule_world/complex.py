import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import copy
from rlvs.agents.utils import batchify, interacting_edges, \
    molecule_median_distance, timeit
from torch_geometric.data import Data
import torch

class Complex:
    __GOOD_FIT = 0.006
    DIST_THRESHOLD = 75

    def __init__(self, protein, ligand, original_ligand=None):
        '''
        max_dist : maximum distance between any atom and box center
        '''
        self.num_features = ligand.n_feat

        self.protein = protein

        self.ligand = ligand
        self.original_ligand = copy.deepcopy(ligand) if original_ligand is \
            None else original_ligand
        self._interacting_edges = None
        self.update_interacting_edges()

    def score(self):
        rmsd = self.ligand.rmsd(self.original_ligand)
        if rmsd > 8:
            raise Exception("BAD RMSD")
        if rmsd > 5:
            return 0
        return np.sinh(rmsd**0.25 + np.arcsinh(1))**-1

    def update_interacting_edges(self):
        self._interacting_edges = interacting_edges(
            self.protein, self.ligand, self.DIST_THRESHOLD
        )

        print(
            "complex Stats: interacting Edges: ", self._interacting_edges.shape,
            "Ligand Shape", self.ligand.data.x.shape,
            "Protein Shape", self.protein.data.x.shape
        )

    @property
    def rmsd(self):
        return self.ligand.rmsd(self.original_ligand)

    @property
    def perfect_fit(self):
        rmsd = self.ligand.rmsd(self.original_ligand)
        return rmsd < self.__GOOD_FIT

    @property
    def data(self):
        batched = batchify([self.protein, self.ligand])
        edge_index = torch.hstack([
            batched.edge_index,
            self._interacting_edges
            ])
        batch = torch.tensor([0] * batched.x.shape[0])
        return Data(x=batched.x, edge_index=edge_index, batch=batch)
