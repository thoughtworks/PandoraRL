from enum import IntEnum
from rlvs.constants import H, C, O, N, S

class BondType(IntEnum):
    COVALENT = 0
    HYDROGEN = 1
    HYDROPHOBIC = 2

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



class MoleculeType(IntEnum):
    PROTEIN = -1
    LIGAND = 1
