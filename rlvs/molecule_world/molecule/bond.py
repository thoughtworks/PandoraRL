from collections import namedtuple
import numpy as np
import torch

from .types import MoleculeType, BondType
from rlvs.config import Config

class Bond:
    def __init__(self, idx, atom_a, atom_b, bond_length, update_edge=True, bond_type=0, ob_bond = None):
        self.idx = idx
        self.atom_a = atom_a
        self.atom_b = atom_b
        self._distance = np.linalg.norm(atom_a.coord - atom_b.coord)
        self.lenght = bond_length
        self.bond_type = bond_type

        self.is_amide = None
        self.is_aromatic = None
        self.is_carbonyl = None
        self.update_edge = update_edge

        if ob_bond is not None:
            self.is_amide = ob_bond.IsAmide()
            self.is_aromatic = ob_bond.IsAromatic()
            self.is_carbonyl = ob_bond.IsCarbonyl()
            
        if update_edge:
            atom_a.add_bond(self)
            atom_b.add_bond(self)
            atom_a.update_hydrogens(atom_b, self)
            atom_b.update_hydrogens(atom_a, self)

    @property
    def distance(self):
        self._distance = np.linalg.norm(self.atom_a.coord - self.atom_b.coord)
        return self._distance

    @property
    def surface_distance(self):
        return self._distance - (self.atom_a.VDWr + self.atom_b.VDWr)
    
    @property
    def edge(self):
        return torch.tensor([
            [self.atom_a.idx, self.atom_b.idx],
            [self.atom_b.idx, self.atom_a.idx]
        ], dtype=torch.long)

    def encoding(self):
        return BondType.encoding(self.bond_type)

    def bond_distance(self):
        return [self.distance]
    
    @property
    def feature(self):
        config = Config.get_instance()
        features = []
        for prop in config.edge_features:
            features.extend(getattr(self, prop)())

        return features


    def saperation(self, dest, named_atom):
        source = self.atom_a if self.atom_a == named_atom else \
            self.atom_b if self.atom_b == named_atom else None

        if source is None:
            raise Exception(f"atom not found")

        return np.sqrt(np.sum((source.coord - dest.coord) ** 2))

    def angle(self, atom1=None, atom2=None, atom3=None, named_atom2=None, named_atom3=None):
        if atom2 is None:            
            atom2 = self.atom_a if self.atom_a == named_atom2 else \
            self.atom_b if self.atom_b == named_atom2 else None

        if atom3 is None:            
            atom3 = self.atom_a if self.atom_a == named_atom2 else \
            self.atom_b if self.atom_b == named_atom2 else None

        if atom2 is None:
            raise Exception("atom2 not found")

        if atom3 is None:
            raise Exception("atom3 not found")

        a = atom1.coord
        b = atom2.coord
        c = atom3.coord
        
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    def update_bond_type(self, bond_type):
        if self.bond_type is None:
            self.bond_type = bond_type
        else:
            self.bond_type = self.bond_type | bond_type
