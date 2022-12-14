import numpy as np
from .molecule.complex import Complex
from .datastore.datasets import DataStore
from rlvs.constants import ComplexConstants, Rewards
from rlvs.molecule_world.scoring import Reward


class ActionSpace:
    def __init__(self, action_bounds):
        self.action_bounds = np.asarray(action_bounds)
        self.n_outputs = self.action_bounds.shape[1]
        self.degree_of_freedom = self.n_outputs * 2

        self._action_vectors = [
            [-ComplexConstants.TRANSLATION_DELTA, 0, 0, 0, 0, 0],
            [ComplexConstants.TRANSLATION_DELTA, 0, 0, 0, 0, 0],
            [0, -ComplexConstants.TRANSLATION_DELTA, 0, 0, 0, 0],
            [0, ComplexConstants.TRANSLATION_DELTA, 0, 0, 0, 0],
            [0, 0, -ComplexConstants.TRANSLATION_DELTA, 0, 0, 0],
            [0, 0, ComplexConstants.TRANSLATION_DELTA, 0, 0, 0],
            [0, 0, 0, -ComplexConstants.ROTATION_DELTA, 0, 0],
            [0, 0, 0, ComplexConstants.ROTATION_DELTA, 0, 0],
            [0, 0, 0, 0, -ComplexConstants.ROTATION_DELTA, 0],
            [0, 0, 0, 0, ComplexConstants.ROTATION_DELTA, 0],
            [0, 0, 0, 0, 0, -ComplexConstants.ROTATION_DELTA],
            [0, 0, 0, 0, 0, ComplexConstants.ROTATION_DELTA]
        ]

    def get_action(self, predicted_action_index):
        return self._action_vectors[predicted_action_index]

    def sample(self):
        return np.diff(
            self.action_bounds, axis=0
        ).flatten() * np.random.random_sample(
            self.action_bounds[0].shape
        ) + self.action_bounds[0]


class Env:
    def __init__(self):
        DataStore.init()
        self.protein, self.ligand = DataStore.next()
        self._complex = Complex(self.protein, self.ligand)
        self.input_shape = self._complex.tensor4D.shape
        single_step = np.array([10, 10, 10, 10, 10, 10])
        action_bounds = np.array([-1*single_step, single_step])
        self.action_space = ActionSpace(action_bounds)

    def reset(self):
        self.protein, self.ligand = DataStore.next()
        self._complex = Complex(self.protein, self.ligand)
        self.input_shape = self._complex.tensor4D.shape
        state = np.expand_dims(self._complex.tensor4D.reshape(self.input_shape), axis=0)

        return state

    def step(self, action):
        terminal = False

        try:
            self.ligand.update_pose(*action)
            self._complex.update_tensor()
            reward = self._complex.score()
            terminal = self._complex.perfect_fit
        except Exception:
            reward = -1
            terminal = True

        state = np.expand_dims(self._complex.tensor4D.reshape(self.input_shape), axis=0)
        return state.astype(dtype='float32'), reward, terminal


class GraphEnv:
    def __init__(self, complex=None, single_step=np.array([1, 1, 1]), test=False):
        action_bounds = np.array([-1*single_step, single_step])
        self.action_space = ActionSpace(action_bounds)

        if complex is None:
            DataStore.init(crop=True)
            self._complex = DataStore.next(False)
            self._complex.randomize_ligand(self.action_space.n_outputs)
        else:
            self._complex = complex

        self.reward = Reward.get_reward_function(self._complex)
        self.reward()
        self.input_shape = self._complex.protein.get_atom_features().shape[1]
        self.edge_shape = self._complex.inter_molecular_edge_attr.shape[1]

    @property
    def is_legal_state(self):
        return self.reward.is_legal

    def reset(self, test=False):
        self._complex.reset_ligand()
        self._complex.update_edges()
        print("RESET RMSD", self._complex.rmsd)
        while True:
            self._complex = DataStore.next(False, test=test)
            self.reward = Reward.get_reward_function(self._complex)
            self.reward()
            original_vina_score = self._complex.vina.total_energy()
            self._complex.randomize_ligand(self.action_space.n_outputs, test=test)
            print(
                "Complex: ", self._complex.protein.path,
                "Original VinaScore:", original_vina_score,
                "Randomized RMSD:", (np.round(self._complex.rmsd, 4)),
                "Randomized Vina Score:", self._complex.vina.total_energy()
            )

            if self.is_legal_state:
                break

            self._complex.reset_ligand()

        self.input_shape = self._complex.protein.get_atom_features().shape[1]

        state = None
        return self._complex, state

    def step(self, action):
        terminal = False
        self._complex.update_pose(*action)
        reward = self.reward()

        if not self.is_legal_state:
            print(
                f'Illegal state Centroid saperation:{self._complex.ligand_centroid_saperation}'
            )

            reward = Rewards.PENALTY
            terminal = True

        return reward, terminal

    def save_complex_files(self, path, filetype="pdb"):
        self._complex.save(path, filetype)


class TestGraphEnv(GraphEnv):
    def __init__(self, scaler, protein_path, ligand_path, protein_filetype, ligand_filetype):
        self.protein_filetype = protein_filetype
        self.ligand_filetype = ligand_filetype
        self.protein_path = protein_path
        self.ligand_path = ligand_path
        self.scaler = scaler

        protein = OB_to_mol(
                    read_to_OB(filename=f'{self.protein_path}', filetype=self.protein_filetype),
                    mol_type=-1,
                    path=f'{self.protein_path}'
                )
        if self.ligand_filetype=="smiles_string":
            ligand = OB_to_mol(
                smiles_to_OB(self.ligand_path, prepare=True),
                mol_type=1,
                path=f'{self.ligand_path}'
            )
        elif self.ligand_filetype=="pdb":
            ligand = OB_to_mol(
                read_to_OB(filename=f'{self.ligand_path}', filetype=self.ligand_filetype, prepare=True),
                mol_type=1,
                path=f'{self.ligand_path}'
            )
        else:
            ligand = OB_to_mol(
                read_to_OB(filename=f'{self.ligand_path}', filetype=self.ligand_filetype, prepare=False),
                mol_type=1,
                path=f'{self.ligand_path}'
            )
        super(TestGraphEnv, self).__init__(complex=Complex(protein, ligand))

    def reset(self):
        self.__init__(self.scaler, self.protein_path, self.ligand_path, self.protein_filetype, self.ligand_filetype)
        return self._complex, self.get_state()

    def step(self, action):
        terminal = False

        delta_change = self._complex.ligand.update_pose(*action)
        terminal = (delta_change < 0.01).all()
            
        state = None #self.get_state()
        return self._complex, state, terminal
