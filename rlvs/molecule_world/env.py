import numpy as np
from .complex import Complex
from .molecule import Molecule
from .datasets import DataStore
from deepchem.feat.mol_graphs import MultiConvMol, ConvMol
import tensorflow as tf
from .helper_functions import *
from rlvs.constants import ComplexConstants, Rewards    

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
        except:
            reward = -1
            terminal = True

        state = np.expand_dims(self._complex.tensor4D.reshape(self.input_shape), axis=0)
        return state.astype(dtype='float32'), reward, terminal
    

class GraphEnv:
    def __init__(self, complex=None):
        # single_step = np.array([10, 10, 10, 10, 10, 10])
        single_step = np.array([1, 1, 1])
        action_bounds = np.array([-1*single_step, single_step])
        self.action_space = ActionSpace(action_bounds)
        
        if complex is None:
            DataStore.init(crop=True)
            self._complex = DataStore.next(False)            
            self._complex.randomize_ligand(self.action_space.n_outputs)
        else:
            self._complex = complex
        
        self.input_shape = self._complex.protein.get_atom_features().shape[1]
        self.edge_shape = self._complex.inter_molecular_edge_attr.shape[1]

    def reset(self):
        self._complex.reset_ligand()
        print("RESET RMSD", self._complex.rmsd)
        while True:
            self._complex = DataStore.next(False)
            original_vina_score = self._complex.vina.total_energy()
            self._complex.randomize_ligand(self.action_space.n_outputs)
            print(
                "Complex: ", self._complex.protein.path,
                "Original VinaScore:", original_vina_score,
                "Randomized RMSD:", (rmsd:=np.round(self._complex.rmsd, 4)),
                "Randomized Vina Score:", self._complex.vina.total_energy()
            )
            self._complex.previous_rmsd = rmsd
            if rmsd < ComplexConstants.RMSD_THRESHOLD:
                break

            self._complex.reset_ligand()
        
        self.input_shape = self._complex.protein.get_atom_features().shape[1]

        state = None
        return self._complex, state

    def step(self, action):
        terminal = False
        try:
            delta_change = self._complex.ligand.update_pose(*action)
            reward = self._complex.score()
            terminal = self._complex.perfect_fit
        except Exception as e:
            print(e)
            reward = Rewards.PENALTY
            terminal = True

        return reward, terminal

    def save_complex_files(self, path, filetype="pdb"):
        filetype = filetype if filetype is not None else self._complex.ligand.filetype
        self._complex.ligand.save(f'{path}.{filetype}', filetype)

        
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
