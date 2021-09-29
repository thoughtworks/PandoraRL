import numpy as np
import matplotlib.pyplot as plt
import copy
import torch

from rlvs.agents.utils import batchify, interacting_edges, \
    molecule_median_distance, timeit, USE_CUDA
from torch_geometric.data import Data
from rlvs.constants import ComplexConstants

from .named_atom import H
from .scoring.vina_score import VinaScore
from .bond import InterMolecularBond
from .types import BondType

import logging

class Complex:
    def __init__(self, protein, ligand, original_ligand):
        self.protein = protein
        self.ligand = ligand
        self.original_ligand = original_ligand 

        self.vina = VinaScore(protein, ligand)
        
        self.inter_molecular_interactions = self.inter_molecular_bonds()
        self.update_edges()
        
        logging.debug(
            f"""complex Stats: InterMolecularBond: {self.inter_molecular_edges.shape},
            Ligand Shape: {self.ligand.data.x.shape},
            Protein Shape: {self.protein.data.x.shape},
            Initial Vina Score: {self.vina.total_energy()}"""
        )

    def crop(self, x, y, z):
        self.protein.crop(self.ligand.get_centroid(), x, y, z)

        self.inter_molecular_interactions = self.inter_molecular_bonds()
        self.update_edges()

    def update_edges(self):
        self.inter_molecular_edges = torch.vstack([
            bond.edge for bond in self.inter_molecular_interactions
        ]).t().contiguous()

        self.inter_molecular_edges = self.inter_molecular_edges.cuda() if USE_CUDA \
            else self.inter_molecular_edges
        
        self.inter_molecular_edge_attr = torch.vstack([
            torch.tensor([
                bond.feature,
                bond.feature
            ], dtype=torch.float32)
            for bond in self.inter_molecular_interactions
        ])

        self.inter_molecular_edge_attr = self.inter_molecular_edge_attr.cuda() if USE_CUDA \
            else self.inter_molecular_edge_attr

    
    def vina_score(self):
        return self.vina.total_energy()

    def inter_molecular_bonds(self):
        n_p_atoms = len(self.protein.atoms)
        p_atoms = [atom.idx for atom in self.protein.atoms.where(lambda x: x.is_heavy_atom)]
        l_atoms = [atom.idx for atom in self.ligand.atoms.where(lambda x: x.is_heavy_atom)]

        adg_mat = np.array(np.meshgrid(l_atoms, p_atoms)).T.reshape(-1,2)
        edge_count = {}

        inter_molecular_edges = [
            InterMolecularBond(
                self.protein.atoms[p_idx],
                self.ligand.atoms[l_idx],
                None,
                update_edge=False,
                ligand_offset=n_p_atoms,
                bond_type=0
            ) for l_idx, p_idx in adg_mat ]

        for edge in inter_molecular_edges:
            if BondType.is_hydrogen_bond(edge.p_atom, edge.l_atom) or\
               BondType.is_hydrogen_bond(edge.l_atom, edge.p_atom):
                edge.update_bond_type(BondType.HYDROGEN)
                edge_count[BondType.HYDROGEN] = edge_count.get(BondType.HYDROGEN, 0) + 1

            if BondType.is_weak_hydrogen_bond(edge.p_atom, edge.l_atom):
                edge.update_bond_type(BondType.WEAK_HYDROGEN)
                edge_count[BondType.WEAK_HYDROGEN] = edge_count.get(BondType.WEAK_HYDROGEN, 0) + 1

            if BondType.is_hydrophobic_1(edge.p_atom, edge.l_atom):
                edge.update_bond_type(BondType.HYDROPHOBIC)
                edge_count[BondType.HYDROPHOBIC] = edge_count.get(BondType.HYDROPHOBIC, 0) + 1

            if BondType.is_multi_polar_halogen(edge.p_atom, edge.l_atom):
                edge.update_bond_type(BondType.MULTI_POLAR_HALOGEN)
                edge_count[BondType.MULTI_POLAR_HALOGEN] = edge_count.get(BondType.MULTI_POLAR_HALOGEN, 0) + 1

            if BondType.is_halogen(edge.p_atom, edge.l_atom):
                edge.update_bond_type(BondType.HALOGEN_BOND)
                edge_count[BondType.HALOGEN_BOND] = edge_count.get(BondType.HALOGEN_BOND, 0) + 1

            if BondType.is_amide_stacking(edge.p_atom, edge.l_atom):
                edge.update_bond_type(BondType.AMIDE_STACKING)
                edge_count[BondType.AMIDE_STACKING] = edge_count.get(BondType.AMIDE_STACKING, 0) + 1

            if BondType.is_pi_stacking(edge.p_atom, edge.l_atom):
                edge.update_bond_type(BondType.PI_STACKING)
                edge_count[BondType.PI_STACKING] = edge_count.get(BondType.PI_STACKING, 0) + 1

            if BondType.is_salt_bridge(edge.p_atom, edge.l_atom):
                edge.update_bond_type(BondType.SALT_BRIDGE)
                edge_count[BondType.SALT_BRIDGE] = edge_count.get(BondType.SALT_BRIDGE, 0) + 1

            if BondType.is_cation_pi(edge.p_atom, edge.l_atom):
                edge.update_bond_type(BondType.CATION_PI)
                edge_count[BondType.CATION_PI] = edge_count.get(BondType.CATION_PI, 0) + 1

        print("Total updated intermolecular edges: ", edge_count)
        logging.debug(f"Total updated intermolecular edges: {edge_count}")
        return inter_molecular_edges
        

    
    def score(self):
        complex_saperation = np.linalg.norm(self.original_ligand.atoms.centroid - self.ligand.atoms.centroid)

        print(
            "complex Stats: InterMolecularBond: ", self.inter_molecular_edges.shape,
            "Ligand Shape", self.ligand.data.x.shape,
            "Protein Shape", self.protein.data.x.shape,
            "Centroid Saperation: ", complex_saperation
        )

        rmsd = self.ligand.rmsd(self.original_ligand)
        rmsd_score = np.sinh(rmsd**0.25 + np.arcsinh(1))**-1

        # Introduce saperation as an exit criterio

        
        vina_score = self.vina.total_energy()

        logging.debug(f"Centroid Saperation: {complex_saperation}, vina score: {vina_score}")
        
        if complex_saperation > ComplexConstants.DISTANCE_THRESHOLD or\
           vina_score > ComplexConstants.VINA_SCORE_THRESHOLD or vina_score == 0:
            raise Exception(f"BAD State: VinaScore: {vina_score}, distance: {complex_saperation}")

        # Found that adding weights to rmsd_score was not having much effect, rmsd_score was mostly being used when the first term went to zero.
        return 0.7**(vina_score) + rmsd_score
        #return -vina_score

    def randomize_ligand(self, action_shape):
        self.ligand.randomize(ComplexConstants.BOUNDS, action_shape)

    def reset_ligand(self):
        self.ligand.set_coords(self.original_ligand.get_coords().data.numpy())
        
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
            self.inter_molecular_edges
            ])

        edge_attr = torch.vstack([
            batched.edge_attr,
            self.inter_molecular_edge_attr
        ])
        batch = torch.tensor([0] * batched.x.shape[0])
        batch = batch.cuda() if USE_CUDA else batch
        
        return Data(
            x=batched.x.detach().clone(),
            edge_index=edge_index.detach().clone(),
            edge_attr=edge_attr.detach().clone(),
            batch=batch)

    def save(self, path, filetype=None):
        ligand_filetype = filetype if filetype is not None else self.ligand.filetype
        protein_filetype =  self.protein.filetype
        self.ligand.save(f'{path}.{ligand_filetype}', ligand_filetype)
        self.protein.save(f'{path}_protein.{protein_filetype}', protein_filetype)

