from rlvs.molecule_world.named_atom import NamedAtom

class ComplexConstants:
    DISTANCE_THRESHOLD = 75
    GOOD_FIT = 0.006
    RMSD_THRESHOLD = 12
    DISTANCE_THRESHOLD = 70
    VINA_SCORE_THRESHOLD = 200
    BOUNDS = 4

class Rewards:
    PENALTY = -1000


class AgentConstants:
    ACTOR_LEARNING_RATE = 0.00005
    CRITIQ_LEARNING_RATE = 0.0001
    TAU = 0.001

    GAMMA = 0.99

    BATCH_SIZE = 32
    BUFFER_SIZE = 200000
    EXPLORATION_EPISODES = 10000


class Vina:
    DONOR_HYDROGEN_DISTANCE = 1.1
    ACCEPTOR_HYDROGEN_DISTANCE = 2.5
    HYDROGEN_BOND_ANGLE = 120

class Features:
    HYDROPHOBIC = 20
    VDWr = 18
    COORD = slice(3)


RESIDUES = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLU": 5,
    "GLN": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
    "XXX": 20
}



H = NamedAtom(1, 'H', 3)
C = NamedAtom(6, 'C', 5)
N = NamedAtom(7, 'N', 6)
O = NamedAtom(8, 'O', 7)
S = NamedAtom(16, 'S', 9)
