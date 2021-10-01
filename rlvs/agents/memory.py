import random
import numpy as np
import torch

class Samples:
    def __init__(self, samples):
        self.samples = samples

    @property
    def len(self):
        return len(self.samples)

    @property
    def states(self):
        return [val['state'] for val in self.samples]

    @property
    def actions(self):
        return np.vstack([val['action'] for val in self.samples])

    @property
    def rewards(self):
        return np.vstack([val['reward'] for val in self.samples])

    @property
    def next_states(self):
        return [val['next_state'] for val in self.samples]

    @property
    def terminals(self):
        return np.vstack([val['done'] for val in self.samples])

    def stats(self):
        print(f'''States: {self.states.shape}
        Actions: {self.actions.shape}
        Rewards: {self.rewards.shape}
        Next States: {self.next_sates.shape}
        Terminals: {self.terminals.shape}''')

class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            samples = random.sample(self._samples, len(self._samples))
        else:
            samples = random.sample(self._samples, no_samples)

        return [Samples([sample]) for sample in samples]

    @property
    def num_samples(self):
        return len(self._samples)

    def has_samples(self, size):
        return len(self._samples) >= size
