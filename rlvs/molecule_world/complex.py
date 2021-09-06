import numpy as np
import matplotlib.pyplot as plt
import copy
import torch

from rlvs.agents.utils import batchify, interacting_edges, \
    molecule_median_distance, timeit
from torch_geometric.data import Data
from rlvs.constants import ComplexConstants

from .named_atom import H
from .scoring.vina_score import VinaScore
from .bond import InterMolecularBond
from .types import BondType

class Complex:
    def __init__(self, protein, ligand, original_ligand, interacting_edges=None):
        self.protein = protein
        self.ligand = ligand
        self.original_ligand = original_ligand 

        self.vina = VinaScore(protein, ligand)
        
        self.inter_molecular_interactions = self.inter_molecular_bonds()
        self.interacting_edges = interacting_edges
        self.update_interacting_edges()
        self.inter_molecular_edges = torch.vstack([
            bond.edge for bond in self.inter_molecular_interactions
        ]).t().contiguous()

    def crop(self, x, y, z):
        self.protein.crop(self.ligand.get_centroid(), x, y, z)

    def vina_score(self):
        return self.vina.total_energy()

    def inter_molecular_bonds(self):
        n_p_atoms = len(self.protein.atoms)
        p_atoms = [atom.idx for atom in self.protein.atoms.where(lambda x: x.is_heavy_atom)]
        l_atoms = [atom.idx for atom in self.ligand.atoms.where(lambda x: x.is_heavy_atom)]

        adg_mat = np.array(np.meshgrid(l_atoms, p_atoms)).T.reshape(-1,2)

        inter_molecular_edges = [
            InterMolecularBond(
                self.protein.atoms[p_idx],
                self.ligand.atoms[l_idx],
                None,
                update_edge=False,
                ligand_offset=n_p_atoms
            ) for l_idx, p_idx in adg_mat ]

        for edge in inter_molecular_edges:
            if BondType.is_hydrogen_bond(edge.p_atom, edge.l_atom) or\
               BondType.is_hydrogen_bond(edge.l_atom, edge.p_atom):
                edge.update_bond_type(BondType.HYDROGEN)

            if BondType.is_hydrophobic_1(edge.p_atom, edge.l_atom):
                edge.update_bond_type(BondType.HYDROPHOBIC)

            if BondType.is_multi_polar_halogen(edge.p_atom, edge.l_atom):
                edge.update_bond_type(BondType.MULTI_POLAR_HALOGEN)

            if BondType.is_halogen(edge.p_atom, edge.l_atom):
                edge.update_bond_type(BondType.HALOGEN_BOND)

            if BondType.is_amide_stacking(edge.p_atom, edge.l_atom):
                edge.update_bond_type(BondType.AMIDE_STACKING)

            if BondType.is_pi_stacking(edge.p_atom, edge.l_atom):
                edge.update_bond_type(BondType.PI_STACKING)

            if BondType.is_salt_bridge(edge.p_atom, edge.l_atom):
                edge.update_bond_type(BondType.SALT_BRIDGE)

            if BondType.is_cation_pi(edge.p_atom, edge.l_atom):
                edge.update_bond_type(BondType.CATION_PI)
                
        return inter_molecular_edges
        

    
    def score(self):
        complex_saperation = np.linalg.norm(self.protein.atoms.centroid - self.ligand.atoms.centroid)

        print(
            "complex Stats: InterMolecularBond: ", self.inter_molecular_edges.shape,
            "Ligand Shape", self.ligand.data.x.shape,
            "Protein Shape", self.protein.data.x.shape,
            "Centroid Saperation: ", complex_saperation
        )

        # rmsd = self.ligand.rmsd(self.original_ligand)
        # rmsd_score = np.sinh(rmsd**0.25 + np.arcsinh(1))**-1

        # Introduce saperation as an exit criterio

        
        vina_score = self.vina.total_energy()
        
        if complex_saperation > ComplexConstants.DISTANCE_THRESHOLD or\
           vina_score > ComplexConstants.VINA_SCORE_THRESHOLD or vina_score == 0:
            raise Exception(f"BAD State: VinaScore: {vina_score}, distance: {complex_saperation}")

        return -vina_score

    def randomize_ligand(self, action_shape):
        self.ligand.randomize(ComplexConstants.BOUNDS, action_shape)

    def reset_ligand(self):
        self.ligand.set_coords(self.original_ligand.get_coords().data.numpy())
        
    def update_interacting_edges(self):
        if self.interacting_edges is not None:
            print(
            "complex Stats: interacting Edges: ", self.inter_molecular_bonds.shape,
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
