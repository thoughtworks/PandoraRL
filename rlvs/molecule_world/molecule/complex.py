import numpy as np
import torch

from rlvs.agents.utils import batchify
from torch_geometric.data import Data
from rlvs.constants import ComplexConstants
from rlvs.molecule_world.scoring import VinaScore, Reward, RMSD

from .inter_molecular_bond import InterMolecularBond, BondEncoder

import logging


class Complex:
    def __init__(self, protein, ligand, original_ligand):
        self.protein = protein
        self.ligand = ligand
        self.original_ligand = original_ligand

        self.vina = VinaScore(protein, ligand)

        self.all_inter_molecular_interactions = self.all_inter_molecular_bonds()
        self.update_edges()

        logging.debug(
            f"""complex Stats: InterMolecularBond: {self.inter_molecular_edges.shape},
            Ligand Shape: {self.ligand.data.x.shape},
            Protein Shape: {self.protein.data.x.shape},
            Initial Vina Score: {self.vina.total_energy()}"""
        )

    def crop(self, x, y, z):
        self.protein.crop(self.ligand.get_centroid(), x, y, z)
        self.all_inter_molecular_interactions = self.all_inter_molecular_bonds()
        self.update_edges()

    def update_edges(self):
        inter_molecular_interactions = [
            BondEncoder.generate_encoded_bond_types(edge)
            for edge in self.all_inter_molecular_interactions
            if edge.distance <= ComplexConstants.DISTANCE_THRESHOLD
         ]

        inter_molecular_interactions = [
            edge for edge in inter_molecular_interactions
            if edge.bond_type is not None and edge.bond_type != 0

        ]

        for edge in inter_molecular_interactions:
            edge.reset_interaction_strengths()

        for edge in inter_molecular_interactions:
            edge.update_interaction_strengths()

        if len(inter_molecular_interactions) == 0:
            self.inter_molecular_edges = None
            return

        self.inter_molecular_edges = torch.vstack([
            bond.edge for bond in inter_molecular_interactions
        ]).t().contiguous()

        self.inter_molecular_edge_attr = torch.vstack([
            torch.tensor([
                bond.feature,
                bond.feature
            ], dtype=torch.float32)
            for bond in inter_molecular_interactions
        ])

    def vina_score(self):
        return self.vina.total_energy()

    def update_pose(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
        self.ligand.update_pose(x, y, z, roll, pitch, yaw)
        self.update_edges()

    def all_inter_molecular_bonds(self):
        n_p_atoms = len(self.protein.atoms)
        p_atoms = [atom.idx for atom in self.protein.atoms.where(lambda x: x.is_heavy_atom)]
        l_atoms = [atom.idx for atom in self.ligand.atoms.where(lambda x: x.is_heavy_atom)]

        adg_mat = np.array(np.meshgrid(l_atoms, p_atoms)).T.reshape(-1, 2)

        return [
            InterMolecularBond(
                self.protein.atoms[p_idx],
                self.ligand.atoms[l_idx],
                None,
                update_edge=False,
                ligand_offset=n_p_atoms,
                bond_type=0
            ) for l_idx, p_idx in adg_mat
        ]

    def randomize_ligand(self, action_shape, test=False):
        self.ligand.randomize(ComplexConstants.BOUNDS, action_shape, test=test)

    def reset_ligand(self):
        self.ligand.set_coords(self.original_ligand.get_coords())

    @property
    def rmsd(self):
        return self.ligand.rmsd(self.original_ligand)

    @property
    def ligand_centroid_saperation(self):
        original_ligand_centroid = RMSD.centroid(self.original_ligand.get_coords())
        current_ligand_centroid = RMSD.centroid(self.ligand.get_coords())

        return np.linalg.norm(original_ligand_centroid - current_ligand_centroid)

    @property
    def perfect_fit(self):
        rmsd = self.ligand.rmsd(self.original_ligand)
        return rmsd < ComplexConstants.GOOD_FIT

    @property
    def data(self):
        batched = batchify([self.protein, self.ligand], data=False)
        if self.inter_molecular_edges is None:
            edge_index = batched.edge_index
            edge_attr = batched.edge_attr

        else:
            edge_index = torch.hstack([
                batched.edge_index,
                self.inter_molecular_edges
            ])
            edge_attr = torch.vstack([
                batched.edge_attr,
                self.inter_molecular_edge_attr
            ])

        pos = batched.x[:, :3]

        batch = torch.tensor([0] * batched.x.shape[0])

        return Data(
            pos=pos.detach().clone(),
            x=batched.x.detach().clone(),
            edge_index=edge_index.detach().clone(),
            edge_attr=edge_attr.detach().clone(),
            batch=batch)

    def save(self, path, filetype=None):
        ligand_filetype = filetype if filetype is not None else self.ligand.filetype
        ligand_name = self.ligand.path.split('/')[-1].split('.')[0]
        self.ligand.save(f'{path}_{ligand_name}.{ligand_filetype}', ligand_filetype)
