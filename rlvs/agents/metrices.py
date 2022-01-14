import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from rlvs.config import Config


class MoleculeMetric:
    def __init__(self, episode, molecule, initial_rmsd, initial_coord):
        self.episode = episode
        self.molecule = molecule
        self.initial_coord = initial_coord
        self.actions = []
        self.rewards = []
        self.divergence = False
        self.rmsds = [initial_rmsd]

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
        self.test_metrices = {}

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = Metric()

        return cls.__instance

    def init_rmsd(self, episode, molecule, initial_rmsd, initial_coord, test=False):
        metric = self.molecule_metrices if not test else self.test_metrices
        if episode not in metric:
            metric[episode] = []

        metric[episode].append(
            MoleculeMetric(episode, molecule, initial_rmsd, initial_coord)
        )

    def cache_loss(self, episode, loss):
        self.losses.append(loss)
        episode_loss = self.episode_loss.get(episode, [])
        episode_loss.append(loss)
        self.episode_loss[episode] = episode_loss

    def cache_rmsd(self, episode, rmsd, molecule_index, test=False):
        metric = self.molecule_metrices if not test else self.test_metrices
        if episode not in metric:
            raise Exception(f"RMSD metric for {episode} not initialised")

        if molecule_index >= len(metric[episode]):
            raise Exception(f"RMSD metric for {molecule_index} not initialised")

        metric[episode][molecule_index].update_rmsd(rmsd)

    def cache_action_reward(self, episode, action, reward, molecule_index, test=False):
        metric = self.molecule_metrices if not test else self.test_metrices
        if episode not in metric:
            raise Exception(f"RMSD metric for {episode} not initialised")

        if molecule_index >= len(metric[episode]):
            raise Exception(f"RMSD metric for {molecule_index} not initialised")

        metric[episode][molecule_index].update_action_reward(action, reward)

    def cache_divergence(self, episode, divergence, molecule_index, test=False):
        metric = self.molecule_metrices if not test else self.test_metrices
        if episode not in metric:
            raise Exception(f"RMSD metric for {episode} not initialised")

        if molecule_index >= len(metric[episode]):
            raise Exception(f"RMSD metric for {molecule_index} not initialised")

        metric[episode][molecule_index].diverged(divergence)

    def get_loss(self, episode=None):
        if episode is None:
            return self.losses

        return self.episode_loss.get(episode, [])

    def plot_rmsd_trend(self, episode, root_path, test=False):
        marker = ',' if test else '.'
        plt.figure(figsize=(10, 7))
        file_name = f'{root_path}_{episode}_rmsd_trend.png' if not test \
            else f'{root_path}_{episode}_test_rmsd_trend.png'
        metric = self.molecule_metrices if not test else self.test_metrices
        molecule_metrics = metric[episode]
        for idx, metric in enumerate(molecule_metrics):
            plt.plot(
                metric.rmsds, marker=marker, linewidth=1,
                label=f'{idx}_{metric.molecule}', zorder=1
            )

            if test:
                self.__plot_rmsd_actions(
                    plt, metric.rmsds, metric.actions
                )

        plt.legend(
            loc='upper right', bbox_to_anchor=(1.3, 1.05), shadow=True
        )
        plt.xlabel('iterations')
        plt.ylabel('rmsd')
        plt.savefig(file_name, bbox_inches='tight', dpi=600)
        plt.close()

    def __plot_rmsd_actions(self, plt, rmsds, actions):
        action_markers = [
            "p", "*", "v", "^", "<", ">", "d", "X", "s", "P", "+", "x"
        ]

        def agg(acc, val):
            idx, x = val
            sym = acc.get(actions[idx], [])
            sym.append((idx+1, x))
            acc[actions[idx]] = sym
            return acc

        plot_points = reduce(agg, enumerate(rmsds[1:]), {})
        for act in plot_points:
            x, y = zip(*plot_points[act])
            plt.scatter(x, y, marker=action_markers[act], zorder=2, s=2)

    def plot_loss_trend(self, root_path):
        average_losses = [np.mean(loss) for loss in self.episode_loss.values()]
        plt.plot(average_losses, marker='.')
        plt.xlabel('episodes')
        plt.ylabel('loss')
        plt.savefig(f'{root_path}_loss_trend.png', bbox_inches='tight')
        plt.close()

    def has_diverged(self, episode):
        config = Config.get_instance()
        molecule_metric = self.molecule_metrices[episode]

        if len(molecule_metric) > 1:
            return any(metric.divergence for metric in molecule_metric)

        rmsds = molecule_metric[0].rmsds
        x = np.array(range(len(rmsds)))
        y = np.array(rmsds)
        xmean = np.mean(x)
        ymean = np.mean(y)
        slope = np.sum((x - xmean)*(y - ymean))/np.sum((x - xmean)**2)
        print("RMSD Slope:", slope)
        return slope > config.divergence_slope

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
