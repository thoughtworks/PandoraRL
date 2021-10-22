from .bond import Bond
from .types import BondType, MoleculeType
from rlvs.constants import BondThresholds, Vina
from .named_atom import H
import torch

class InterMolecularBond(Bond):
    @staticmethod
    def copy(bond):
        return InterMolecularBond(
            bond.atom_a,
            bond.atom_b,
            bond.lenght,
            update_edge=bond.update_edge,
            bond_type=bond.bond_type,
            ligand_offset=bond.ligand_offset
        )

    def __init__(self, atom_a, atom_b, bond_length, update_edge=True, bond_type=None, ligand_offset=0):
        super(
            InterMolecularBond, self
        ).__init__(-1, atom_a, atom_b, bond_length, update_edge, bond_type)

        self.p_atom = atom_a if atom_a.molecule_type == MoleculeType.PROTEIN else atom_b
        self.l_atom = atom_a if atom_a.molecule_type == MoleculeType.LIGAND else atom_b
        self.ligand_offset = ligand_offset

    @property
    def edge(self):
        return torch.tensor([
            [self.p_atom.idx, self.l_atom.idx + self.ligand_offset],
            [self.l_atom.idx + self.ligand_offset, self.p_atom.idx]
        ], dtype=torch.long)


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

    @property
    def is_valid(self):
        valid_h_bonds = [
            h_bond for h_bond in self.donor.hydrogens
            if (
                    h_bond.distance < Vina.DONOR_HYDROGEN_DISTANCE
                    and h_bond.saperation(
                        self.acceptor, H
                    ) < Vina.ACCEPTOR_HYDROGEN_DISTANCE
                    and h_bond.angle(
                        atom1=self.acceptor,
                        named_atom2=H,
                        atom3=self.donor
                    ) >= Vina.HYDROGEN_BOND_ANGLE
            )
        ]

        if not valid_h_bonds:
            return False

        return self.surface_distance <= Vina.HYDROGEN_BOND_SURFACE_THRESHOLD


    
class HydrophobicBond(InterMolecularBond):
    def __init__(self, idx, p_atom, l_atom, ligand_offset=0):
        super(
            HydrophobicBond, self
        ).__init__(p_atom, l_atom, None,
                   update_edge=False, bond_type=BondType.HYDROPHOBIC, ligand_offset=ligand_offset)
        self.idx = idx


class BondEncoder:
    @classmethod
    def generate_encoded_bond_types(cls, edge):
        edge = InterMolecularBond.copy(edge)
        if cls.is_hydrogen_bond(edge) \
           and edge.distance <= BondThresholds.HYDROGEN:
            edge.update_bond_type(BondType.HYDROGEN)

        if BondType.is_weak_hydrogen_bond(edge.p_atom, edge.l_atom) \
           and edge.distance <= BondThresholds.WEAK_HYDROGEN:
            edge.update_bond_type(BondType.WEAK_HYDROGEN)

        if BondType.is_hydrophobic_1(edge.p_atom, edge.l_atom) \
           and edge.distance <= BondThresholds.HYDROPHOBIC:
                edge.update_bond_type(BondType.HYDROPHOBIC)

        if BondType.is_multi_polar_halogen(edge.p_atom, edge.l_atom) \
           and edge.distance <= BondThresholds.MULTI_POLAR_HALOGEN:
                edge.update_bond_type(BondType.MULTI_POLAR_HALOGEN)

        if BondType.is_halogen(edge.p_atom, edge.l_atom) \
           and edge.distance <= BondThresholds.HALOGEN:
                edge.update_bond_type(BondType.HALOGEN_BOND)

        if BondType.is_amide_stacking(edge.p_atom, edge.l_atom) \
           and edge.distance <= BondThresholds.AMIDE_STACKING:
            edge.update_bond_type(BondType.AMIDE_STACKING)

        if BondType.is_pi_stacking(edge.p_atom, edge.l_atom) \
           and edge.distance <= BondThresholds.PI_STACKING:
                edge.update_bond_type(BondType.PI_STACKING)            

        if BondType.is_salt_bridge(edge.p_atom, edge.l_atom) \
           and edge.distance <= BondThresholds.SALT_BRIDGE:
                edge.update_bond_type(BondType.SALT_BRIDGE)

        if BondType.is_cation_pi(edge.p_atom, edge.l_atom) \
           and edge.distance <= BondThresholds.CATION_PI:
                edge.update_bond_type(BondType.CATION_PI)

        if BondType.is_repulsive_bond(edge):
                edge.update_bond_type(BondType.REPULSIVE)
                
        return edge

    @staticmethod
    def is_hydrogen_bond(bond):
        h_bond = None
        if  BondType.is_hydrogen_bond(bond.atom_a, bond.atom_b):
            h_bond = HydrogenBond(bond.idx, bond.atom_a, bond.atom_b)
            
        elif BondType.is_hydrogen_bond(bond.atom_b, bond.atom_a):
            h_bond = HydrogenBond(bond.idx, bond.atom_b, bond.atom_a)

        else:
            return False

        return h_bond.is_valid
            
