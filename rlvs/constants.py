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
