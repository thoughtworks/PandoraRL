from enum import IntEnum
from .named_atom import H, C, O, N, S, F, Cl, Br, I

class BondType(IntEnum):
    COVALENT = 1
    HYDROGEN = 2
    HYDROPHOBIC = 4
    PI_STACKING = 8
    SALT_BRIDGE = 16
    AMIDE_STACKING = 32
    HALOGEN_BOND = 64
    MULTI_POLAR_HALOGEN = 128
    CATION_PI = 256

    @classmethod
    def encoding(cls, bond_type):
        return [bond_type >> _type & 1 for _type in range(len(cls) - 1, -1, -1)]

    @staticmethod
    def is_hydrogen_bond(atom1, atom2):
        is_H_bond_functional_group = lambda atom1, atom2: (O == atom1 and (N == atom2 or O == atom2)) or\
            (N == atom1 and (O == atom2 or N == atom2 or S == atom2)) or (S == atom1 and N == atom2)

        return len(atom1.hydrogens) > 0 and atom2.acceptor and \
            is_H_bond_functional_group(atom1, atom2)

    @staticmethod
    def is_hydrophobic(p_atom, l_atom):
        return p_atom == C and l_atom == C and p_atom.hydrophobic and l_atom.hydrophobic

    @staticmethod
    def is_hydrophobic_1(p_atom, l_atom):
        return (
            p_atom == C and l_atom == C and not (p_atom.aromatic and l_atom.aromatic)
        ) or (
            p_atom == C and l_atom in [F, Cl, Br, I]
        ) or (
            p_atom == S and l_atom == C and l_atom.aromatic
        )

    @staticmethod
    def is_multi_polar_halogen(p_atom, l_atom):
        return (
            p_atom in [C, N] and l_atom in [F, Cl] and p_atom.amide
        )


    @staticmethod
    def is_halogen(p_atom, l_atom):
        return (
            p_atom in [O, N, S] and l_atom in [Cl, Br, I]
        )
    
    
    @staticmethod
    def is_amide_stacking(p_atom, l_atom):
        return (
            p_atom == C and l_atom == C and p_atom.amide and l_atom.aromatic
        )    

    @staticmethod
    def is_pi_stacking(p_atom, l_atom):
        return (
            p_atom == C and l_atom == C and p_atom.aromatic and l_atom.aromatic
        )    

    @staticmethod
    def is_salt_bridge(p_atom, l_atom):
        is_nitrogen = lambda atom: atom == N and atom.partial_charge > 0
        is_oxygen = lambda atom: atom == O and atom.partial_charge < 0
        return (
            is_nitrogen(p_atom) and is_oxygen(l_atom)
        ) or (
            is_nitrogen(l_atom) and is_oxygen(p_atom)
        )

    @staticmethod
    def is_cation_pi(p_atom, l_atom):
        is_nitrogen = lambda atom: atom == N and atom.partial_charge > 0
        is_carbon = lambda atom: atom == C and atom.aromatic
        return (
            is_nitrogen(p_atom) and is_carbon(l_atom)
        ) or (
            is_nitrogen(l_atom) and is_carbon(p_atom)
        )   

class MoleculeType(IntEnum):
    PROTEIN = -1
    LIGAND = 1
