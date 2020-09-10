import numpy as np
from .block import Block

class ActionSpace:
    def __init__(self, action_bounds):
        self.action_bounds = np.asarray(action_bounds)
        self.n_outputs = self.action_bounds.shape[1]

    def sample(self):
        return np.diff(
            self.action_bounds, axis = 0
        ).flatten() * np.random.random_sample(self.action_bounds[0].shape
        ) + self.action_bounds[0]

class Env:
    def __init__(self):
        self.block = Block.get_block()
        self.input_shape = (*self.block.sandbox.shape, 1)
        self.action_space = ActionSpace(self.block.action_bounds)

    def reset(self):
        self.block = Block.get_block()
        self.input_shape = (*self.block.sandbox.shape, 1)
        state = np.expand_dims(self.block.sandbox.reshape(self.input_shape), axis=0)
        return state.astype(dtype='float32')

    def step(self, action):
        try:
            self.block.update_sandbox(*action)
        except:
            reward = -0.0001
            self.block.update_sandbox()
        else: 
            reward = self.block.score()
            
        state = np.expand_dims(self.block.sandbox.reshape(self.input_shape), axis=0)
        terminal = (reward == -0.0001 or self.block.perfect_fit)
        return state.astype(dtype='float32'), reward, terminal
