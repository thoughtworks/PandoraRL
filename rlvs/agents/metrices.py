import numpy as np
import matplotlib.pyplot as plt


class MoleculeMetric:
    def __init__(self, episode, molecule, initial_rmsd, initial_coord):
        self.episode = episode
        self.molecule = molecule
        self.initial_coord = initial_coord
        self.actions = []
        self.rewards = []
        self.divergence = False
        self.rmsds = [initial_rmsd]
        self.rewards = []

    @property
    def initial_rmsd(self):
        return self.rmsds[0]

    @property
    def final_rmsd(self):
        return self.rmsds[-1]

    @property
    def max_rmsd(self):
        return max(self.rmsds)

    @property
    def min_rmsd(self):
        return min(self.rmsds)

    def update_rmsd(self, rmsd):
        self.rmsds.append(rmsd)

    def update_action_reward(self, action, reward):
        self.actions.append(action)
        self.rewards.append(reward)

    def diverged(self, divergence=True):
        self.divergence = divergence


class Metric:
    __instance = None
    __SAVE_PATH = "metric.npy"

    def __init__(self):
        self.losses = []
        self.episode_loss = {}
        self.molecule_metrices = {}

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = Metric()

        return cls.__instance

    def init_rmsd(self, episode, molecule, initial_rmsd, initial_coord):
        if episode not in self.molecule_metrices:
            self.molecule_metrices[episode] = []

        self.molecule_metrices[episode].append(
            MoleculeMetric(episode, molecule, initial_rmsd, initial_coord)
        )

    def cache_loss(self, episode, loss):
        self.losses.append(loss)
        episode_loss = self.episode_loss.get(episode, [])
        episode_loss.append(loss)
        self.episode_loss[episode] = episode_loss

    def cache_rmsd(self, episode, rmsd, molecule_index):
        if episode not in self.molecule_metrices:
            raise Exception(f"RMSD metric for {episode} not initialised")

        if molecule_index >= len(self.molecule_metrices[episode]):
            raise Exception(f"RMSD metric for {molecule_index} not initialised")

        self.molecule_metrices[episode][molecule_index].update_rmsd(rmsd)

    def cache_action_reward(self, episode, action, reward, molecule_index):
        if episode not in self.molecule_metrices:
            raise Exception(f"RMSD metric for {episode} not initialised")

        if molecule_index >= len(self.molecule_metrices[episode]):
            raise Exception(f"RMSD metric for {molecule_index} not initialised")

        self.molecule_metrices[episode][molecule_index].update_action_reward(action, reward)

    def cache_divergence(self, episode, divergence, molecule_index):
        if episode not in self.molecule_metrices:
            raise Exception(f"RMSD metric for {episode} not initialised")

        if molecule_index >= len(self.molecule_metrices[episode]):
            raise Exception(f"RMSD metric for {molecule_index} not initialised")

        self.molecule_metrices[episode][molecule_index].diverged(divergence)

    def get_loss(self, episode=None):
        if episode is None:
            return self.losses

        return self.episode_loss.get(episode, [])

    def plot_rmsd_trend(self, episode, root_path):
        molecule_metrics = self.molecule_metrices[episode]
        for idx, metric in enumerate(molecule_metrics):
            plt.plot(metric.rmsds, marker='.', label=f'{idx}_{metric.molecule}')

        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel('rmsd')
        plt.savefig(f'{root_path}_{episode}_rmsd_trend.png', bbox_inches='tight')
        plt.close()

    def plot_loss_trend(self, root_path):
        average_losses = [np.mean(loss) for loss in self.episode_loss.values()]
        plt.plot(average_losses, marker='.')
        plt.xlabel('episodes')
        plt.ylabel('loss')
        plt.savefig(f'{root_path}_loss_trend.png', bbox_inches='tight')
        plt.close()

    @classmethod
    def save(cls, metrices, root_path):
        with open(f'{root_path}_{cls.__SAVE_PATH}', 'wb') as f:
            np.save(f, metrices)

    @classmethod
    def load(cls, path, root=True):
        metrices = Metric()
        file_path = f'{path}_{cls.__SAVE_PATH}' if root else path
        with open(file_path, 'rb') as f:
            metrices = np.load(f, allow_pickle=True)

        return metrices.item()
