import numpy as np
import matplotlib.pyplot as plt
import copy
import torch

from rlvs.agents.utils import batchify, interacting_edges, \
    molecule_median_distance, timeit
from torch_geometric.data import Data
from rlvs.constants import ComplexConstants

from .scoring.vina_score import VinaScore

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
        self.vina = VinaScore(protein, ligand)

    def crop(self, x, y, z):
        self.protein.crop(self.ligand.get_centroid(), x, y, z)

    def vina_score(self):
        return self.vina.total_energy()

    def score(self):
        print(
             "complex Stats: interacting Edges: ", self.interacting_edges.shape,
             "Ligand Shape", self.ligand.data.x.shape,
             "Protein Shape", self.protein.data.x.shape
        )
        rmsd = self.ligand.rmsd(self.original_ligand)
        rmsd_score = np.sinh(rmsd**0.25 + np.arcsinh(1))**-1
        
        if rmsd > ComplexConstants.RMSD_THRESHOLD:
            raise Exception("BAD RMSD")

        vina_score = 10 if (vina_score:=self.vina.total_energy() > 10) else vina_score

        return rmsd_score - vina_score

    def randomize_ligand(self, action_shape):
        self.ligand.randomize(ComplexConstants.BOUNDS, action_shape)

    def reset_ligand(self):
        self.ligand.set_coords(self.original_ligand.get_coords().data.numpy())
        
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
