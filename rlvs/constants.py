from rlvs.molecule_world.molecule import NamedAtom


class ComplexConstants:
    GOOD_FIT = 0.1
    RMSD_THRESHOLD = 12 #4
    DISTANCE_THRESHOLD = 8 #4
    VINA_SCORE_THRESHOLD = 200
    BOUNDS = 4
    TRANSLATION_DELTA = 0.5#0.1
    ROTATION_DELTA = 5 #0.1


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
    HYDROGEN_BOND_SURFACE_THRESHOLD = -0.7


class BondThresholds:
    HYDROGEN = 3.0
    WEAK_HYDROGEN = 3.4
    HYDROPHOBIC = 4.0
    HALOGEN = 3.5
    MULTI_POLAR_HALOGEN = 3.6
    AMIDE_STACKING = 4.0
    PI_STACKING = 4.0
    SALT_BRIDGE = 4.0
    CATION_PI = 4.0


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

KD_HYDROPHOBICITYA = {
    'ALA': 2,
    'CYS': 3,
    'ASP': 0,
    'GLU': 0,
    'PHE': 3,
    'GLY': 1,
    'HIS': 0,
    'ILE': 3,
    'LYS': 0,
    'LEU': 3,
    'MET': 2,
    'ASN': 0,
    'PRO': 1,
    'GLN': 0,
    'ARG': 0,
    'SER': 1,
    'THR': 1,
    'VAL': 3,
    'TRP': 1,
    'XXX': 4,
    'TYR': 1
}

CONFORMATION_SIMILARITY = {
    'ALA': 0,
    'CYS': 0,
    'ASP': 2,
    'GLU': 0,
    'PHE': 4,
    'GLY': 3,
    'HIS': 4,
    'ILE': 6,
    'LYS': 0,
    'LEU': 0,
    'MET': 0,
    'ASN': 2,
    'PRO': 1,
    'GLN': 0,
    'ARG': 0,
    'SER': 5,
    'THR': 6,
    'VAL': 6,
    'TRP': 4,
    'XXX': 7,
    'TYR': 4
}

Z_SCORES = {
    "ALA": [0.24, -2.32,  0.60, -0.14,  1.3],
    "ARG": [3.52,  2.50, -3.50,  1.99, -0.17],
    "ASN": [3.05,  1.62,  1.04, -1.15,  1.61],
    "ASP": [3.98,  0.93,  1.93, -2.46,  0.75],
    "CYS": [0.84, -1.67,  3.71,  0.18, -2.65],
    "GLU": [3.11,  0.26, -0.11, -3.04, -0.25],
    "GLN": [1.75,  0.50, -1.44, -1.34,  0.66],
    "GLY": [2.05, -4.06,  0.36, -0.82, -0.38],
    "HIS": [2.47,  1.95,  0.26,  3.90,  0.09],
    "ILE": [3.89, -1.73, -1.71, -0.84,  0.26],
    "LEU": [4.28, -1.30, -1.49, -0.72,  0.84],
    "LYS": [2.29,  0.89, -2.49,  1.49,  0.31],
    "MET": [-2.85, -0.22,  0.47,  1.94, -0.98],
    "PHE": [-4.22,  1.94,  1.06,  0.54, -0.62],
    "PRO": [-1.66,  0.27,  1.84,  0.70,  2.00],
    "SER": [2.39, -1.07,  1.15, -1.39,  0.67],
    "THR": [0.75, -2.18, -1.12, -1.46, -0.4],
    "TRP": [-4.36,  3.94,  0.59,  3.44, -1.59],
    "TYR": [-2.54,  2.44,  0.43,  0.04, -1.47],
    "VAL": [-2.59, -2.64, -1.54, -0.85, -0.02],
    "XXX": [0,     0,     0,     0,     0]
}
