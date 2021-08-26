from collections import namedtuple
import numpy as np
import torch
from enum import IntEnum

class BondType(IntEnum):
    COVALENT = 0
    HYDROGEN = 1
    HYDROPHOBIC = 2


class Bond:
    def __init__(self, atom_a, atom_b, bond_length, update_edge=True, bond_type=None):
        self.atom_a = atom_a
        self.atom_b = atom_b
        self.distance = np.sqrt(np.sum((atom_a.coord - atom_b.coord) ** 2))
        self.lenght = bond_length
        self.bond_type = bond_type

        if update_edge:
            atom_a.add_bond(self)
            atom_b.add_bond(self)
            atom_a.update_hydrogens(atom_b, self)
            atom_b.update_hydrogens(atom_a, self)

    @property
    def edge(self):
        return torch.tensor([
            [self.atom_a.idx, self.atom_b.idx],
            [self.atom_b.idx, self.atom_a.idx]
        ], dtype=torch.long)

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


class InterMolecularBond(Bond):
    def __init__(self, p_atom, l_atom, bond_length, update_edge=True, bond_type=None, ligand_offset=0):
        super(
            InterMolecularBond, self
        ).__init__(p_atom, l_atom, bond_length, update_edge, bond_type)

        self.l_atom = l_atom
        self.p_atom = p_atom
        self.ligand_offset = ligand_offset

    @property
    def edge(self):
        return torch.tensor([
            [self.p_atom.idx, self.l_atom.idx + self.ligand_offset],
            [self.l_atom.idx + self.ligand_offset, self.p_atom.idx]
        ], dtype=torch.long)

    @property
    def feature(self):
        encoding = np.zeros(len(BondType), dtype=int)
        encoding[self.bond_type] = 1
        return np.append(encoding, self.distance)


class HydrogenBond(InterMolecularBond):
    def __init__(self, idx, donor, acceptor, ligand_offset=0):
        super(
            HydrogenBond, self
        ).__init__(donor, acceptor, None,
                   update_edge=False, bond_type=BondType.HYDROGEN, ligand_offset=ligand_offset)
        self.idx = idx

    @property
    def donor(self):
        return self.atom_a

    @property
    def acceptor(self):
        return self.atom_b

    
class HydrophobicBond(InterMolecularBond):
    def __init__(self, idx, p_atom, l_atom, ligand_offset=0):
        super(
            HydrophobicBond, self
        ).__init__(p_atom, l_atom, None,
                   update_edge=False, bond_type=BondType.HYDROPHOBIC, ligand_offset=ligand_offset)
        self.idx = idx
