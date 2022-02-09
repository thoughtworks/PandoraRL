import numpy as np
from .rmsd import RMSD
from .vina_score import VinaScore

from rlvs.config import Config


class Reward:
    __reward_functions = {
        'rmsd': lambda pl_complex: RMSDReward(pl_complex),
        'vina': lambda pl_complex: VinaReward(pl_complex)
    }

    PENALTY = -100

    def __init__(self, pl_complex):
        self.pl_complex = pl_complex

    @classmethod
    def get_reward_function(cls, pl_complex):
        config = Config.get_instance()
        function = cls.__reward_functions[config.reward_function]

        return function(pl_complex)

    def __call__(self):
        pass


class VinaReward(Reward):
    def __init__(self, pl_complex):
        Reward.__init__(self, pl_complex)
        self.vina_score = VinaScore(pl_complex.protein, pl_complex.ligand)

    def __call__(self):
        return self.vina_score.total_energy()


class RMSDReward(Reward):
    def __init__(self, pl_complex):
        Reward.__init__(self, pl_complex)
        self.rmsd = RMSD(pl_complex.ligand)

    def __call__(self):
        rmsd = self.rmsd(self.pl_complex.original_ligand)
        rmsd_score = np.sinh(rmsd + np.arcsinh(0))**-1

        return 200 if self.pl_complex.perfect_fit else rmsd_score
