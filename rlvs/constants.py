from collections import namedtuple
from rlvs.molecule_world.named_atom import NamedAtom

HydrogenBondPair = namedtuple("HydrogenBondPair", ["idx", "donor", "acceptor"])

class ComplexConstants:
    DISTANCE_THRESHOLD = 75
    GOOD_FIT = 0.006
    RMSD_THRESHOLD = 12
    BOUNDS = 10

class Rewards:
    PENALTY = -100


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

    


H = NamedAtom(1, 'H', 3)
C = NamedAtom(6, 'C', 5)
N = NamedAtom(7, 'N', 6)
O = NamedAtom(8, 'O', 7)
S = NamedAtom(16, 'S', 9)
