import numpy as np
from .complex import Complex
from .molecule import Molecule
from .helper_functions import get_molecules
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
        self.ligand, self.protein = get_molecules()
        self._complex = Complex(self.protein, self.ligand)
        self.input_shape = (1, *self._complex.tensor4D.shape)
        single_step = np.array([10, 10, 10, 10, 10, 10])
        action_bounds = np.array([-1*single_step, single_step])
        self.action_space = ActionSpace(action_bounds)

    def reset(self):
        self.ligand, self.protein = get_molecules()
        self._complex = Complex(self.protein, self.ligand)
        self.input_shape = (1, *self._complex.tensor4D.shape)
        state = np.expand_dims(self._complex.tensor4D.reshape(self.input_shape), axis=0)
        return state

    def step(self, action):
        terminal = False
        try:
            self.ligand.update_pose(*action)
            self._complex.update_tensor()
        except:
            reward = -1
            terminal = True
        else: 
            reward = self._complex.score()

        state = np.expand_dims(self._complex.tensor4D.reshape(self.input_shape), axis=0)
        terminal = (terminal or self._complex.perfect_fit)
        return state.astype(dtype='float32'), reward, terminal



# initialize complex/reset
# state
# apply delta transformations (step function)
#
