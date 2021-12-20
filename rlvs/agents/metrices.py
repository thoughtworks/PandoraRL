import numpy as np


class RMSDMetric:
    def __init__(self, episode, molecule, initial_rmsd):
        self.episode = episode
        self.molecule = molecule
        self.initial_rmsd = initial_rmsd
        self.min_rmsd = initial_rmsd
        self.max_rmsd = initial_rmsd
        self.divergence = False

    def update_rmsd(self, rmsd):
        if self.min_rmsd > rmsd:
            self.min_rmsd = rmsd

        if self.max_rmsd < rmsd:
            self.max_rmsd = rmsd

    def diverged(self, divergence=True):
        self.divergence = divergence

    def dict(self):
        return {
            "episode": self.episode,
            "initial_rmsd": self.initial_rmsd,
            "min_rmsd": self.min_rmsd,
            "max_rmsd": self.max_rmsd,
            "divergence": self.divergence
        }


class Metric:
    __instance = None
    __SAVE_PATH = "metric.npy"

    def __init__(self):
        self.losses = []
        self.episode_loss = {}
        self.rmsd_metrices = {}

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = Metric()

        return cls.__instance

    def init_rmsd(self, episode, molecule, initial_rmsd):
        if episode not in self.rmsd_metrices:
            self.rmsd_metrices[episode] = []

        self.rmsd_metrices[episode].append(RMSDMetric(episode, molecule, initial_rmsd))

    def cache_loss(self, episode, loss):
        self.losses.append(loss)
        episode_loss = self.episode_loss.get(episode, [])
        episode_loss.append(loss)
        self.episode_loss[episode] = episode_loss

    def cache_rmsd(self, episode, rmsd, molecule_index):
        if episode not in self.rmsd_metrices:
            raise Exception(f"RMSD metric for {episode} not initialised")

        if molecule_index >= len(self.rmsd_metrices[episode]):
            raise Exception(f"RMSD metric for {molecule_index} not initialised")

        self.rmsd_metrices[episode][molecule_index].update_rmsd(rmsd)

    def cache_divergence(self, episode, divergence, molecule_index):
        if episode not in self.rmsd_metrices:
            raise Exception(f"RMSD metric for {episode} not initialised")

        if molecule_index >= len(self.rmsd_metrices[episode]):
            raise Exception(f"RMSD metric for {molecule_index} not initialised")

        self.rmsd_metrices[episode][molecule_index].diverged(divergence)

    def get_loss(self, episode=None):
        if episode is None:
            return self.losses

        return self.episode_loss.get(episode, [])

    @classmethod
    def save(cls, metrices, root_path):
        with open(f'{root_path}_{cls.__SAVE_PATH}', 'wb') as f:
            np.save(f, metrices)

    @classmethod
    def load(cls, root_path):
        metrices = Metric()
        with open(f'{root_path}_{cls.__SAVE_PATH}', 'rb') as f:
            metrices = np.load(f, allow_pickle=True)

        return metrices.item()
