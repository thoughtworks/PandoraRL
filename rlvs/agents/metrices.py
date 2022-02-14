import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from rlvs.config import Config


class MoleculeMetric:
    def __init__(self, episode, molecule, initial_score, initial_coord):
        self.episode = episode
        self.molecule = molecule
        self.initial_coord = initial_coord
        self.actions = []
        self.rewards = []
        self.divergence = False
        self.scores = [initial_score]

    @property
    def initial_score(self):
        return self.scores[0]

    @property
    def final_score(self):
        return self.scores[-1]

    @property
    def max_score(self):
        return max(self.scores)

    @property
    def min_score(self):
        return min(self.scores)

    def update_score(self, score):
        self.scores.append(score)

    def update_action_reward(self, action, reward):
        self.actions.append(action)
        self.rewards.append(reward)

    def diverged(self, divergence=True):
        self.divergence = divergence

    def get_data_frame(self):
        '''
        episode, molecule, action, score
        '''

        df = pd.DataFrame({
            'episode': self.episode,
            'molecule': self.molecule,
            'action': [-1]+self.actions,
            'score': self.scores
        })

        return df


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

    def init_score(self, episode, molecule, initial_score, initial_coord, test=False):
        metric = self.molecule_metrices if not test else self.test_metrices
        if episode not in metric:
            metric[episode] = []

        metric[episode].append(
            MoleculeMetric(episode, molecule, initial_score, initial_coord)
        )

    def cache_loss(self, episode, loss):
        self.losses.append(loss)
        episode_loss = self.episode_loss.get(episode, [])
        episode_loss.append(loss)
        self.episode_loss[episode] = episode_loss

    def cache_score(self, episode, score, molecule_index, test=False):
        metric = self.molecule_metrices if not test else self.test_metrices
        if episode not in metric:
            raise Exception(f"SCORE metric for {episode} not initialised")

        if molecule_index >= len(metric[episode]):
            raise Exception(f"SCORE metric for {molecule_index} not initialised")

        metric[episode][molecule_index].update_score(score)

    def cache_action_reward(self, episode, action, reward, molecule_index, test=False):
        metric = self.molecule_metrices if not test else self.test_metrices
        if episode not in metric:
            raise Exception(f"SCORE metric for {episode} not initialised")

        if molecule_index >= len(metric[episode]):
            raise Exception(f"SCORE metric for {molecule_index} not initialised")

        metric[episode][molecule_index].update_action_reward(action, reward)

    def cache_divergence(self, episode, divergence, molecule_index, test=False):
        metric = self.molecule_metrices if not test else self.test_metrices
        if episode not in metric:
            raise Exception(f"SCORE metric for {episode} not initialised")

        if molecule_index >= len(metric[episode]):
            raise Exception(f"SCORE metric for {molecule_index} not initialised")

        metric[episode][molecule_index].diverged(divergence)

    def get_loss(self, episode=None):
        if episode is None:
            return self.losses

        return self.episode_loss.get(episode, [])

    def plot_score_trend(self, episode, root_path, test=False, actions=False):
        marker = ',' if actions else '.'
        plt.figure(figsize=(10, 7))
        file_name = f'{root_path}_{episode}_score_trend.png' if not test \
            else f'{root_path}_{episode}_test_score_trend.png'
        metric = self.molecule_metrices if not test else self.test_metrices
        molecule_metrics = metric[episode]
        for idx, metric in enumerate(molecule_metrics):
            plt.plot(
                metric.scores, marker=marker, linewidth=1,
                label=f'{idx}_{metric.molecule}', zorder=1
            )

            if actions:
                self.__plot_score_actions(
                    plt, metric.scores, metric.actions
                )

        plt.legend(
            loc='upper right', bbox_to_anchor=(1.3, 1.05), shadow=True
        )
        plt.xlabel('iterations')
        plt.ylabel('score')
        plt.savefig(file_name, bbox_inches='tight', dpi=600)
        plt.close()

    def __plot_score_actions(self, plt, scores, actions):
        action_markers = [
            "p", "*", "v", "^", "<", ">", "d", "X", "s", "P", "+", "x"
        ]

        def agg(acc, val):
            idx, x = val
            sym = acc.get(actions[idx], [])
            sym.append((idx+1, x))
            acc[actions[idx]] = sym
            return acc

        plot_points = reduce(agg, enumerate(scores[1:]), {})
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
        return episode % 10 == 0
        config = Config.get_instance()
        molecule_metric = self.molecule_metrices[episode]

        if len(molecule_metric) > 1:
            return any(metric.divergence for metric in molecule_metric)

        scores = molecule_metric[0].scores
        x = np.array(range(len(scores)))
        y = np.array(scores)

        xmean = np.mean(x)
        ymean = np.mean(y)
        slope = np.sum((x - xmean)*(y - ymean))/np.sum((x - xmean)**2)
        print("SCORE Slope:", slope)
        return slope > config.divergence_slope

    def generate_data_frame(self, test=False):
        metrices = self.molecule_metrices if not test else self.test_metrices

        return pd.concat(
            (pd.concat(metric.get_data_frame() for metric in runs) for runs in metrices.values())
        )

    @classmethod
    def save(cls, metrices, root_path):
        with open(f'{root_path}_{cls.__SAVE_PATH}', 'wb') as f:
            np.save(f, metrices)

    @classmethod
    def save_csv(cls, metrices, root_path, test=False):
        df = metrices.generate_data_frame(test)
        df.to_csv(f'{root_path}_trend.csv', index=False)

    @classmethod
    def load(cls, path, root=True):
        metrices = Metric()
        file_path = f'{path}_{cls.__SAVE_PATH}' if root else path
        with open(file_path, 'rb') as f:
            metrices = np.load(f, allow_pickle=True)

        return metrices.item()
