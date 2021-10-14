from rlvs.molecule_world.molecule import NamedAtom

class ComplexConstants:
    GOOD_FIT = 0.1
    RMSD_THRESHOLD = 4
    DISTANCE_THRESHOLD = 4
    VINA_SCORE_THRESHOLD = 200
    BOUNDS = 4
    TRANSLATION_DELTA = 0.1
    ROTATION_DELTA = 0.1

class Rewards:
    PENALTY = -100


class AgentConstants:
    ACTOR_LEARNING_RATE = 0.001
    CRITIQ_LEARNING_RATE = 0.0001
    TAU = 0.01

    GAMMA = 0.99

    BATCH_SIZE = 32
    BUFFER_SIZE = 5000
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
