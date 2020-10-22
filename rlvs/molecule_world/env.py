import numpy as np
from .complex import Complex
from .molecule import Molecule
from .datasets import DataStore
from deepchem.feat.mol_graphs import MultiConvMol, ConvMol
import tensorflow as tf

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
    def __init__(self):
        DataStore.init(crop=False)
        self.protein, self.ligand = DataStore.next(False)
        self._complex = Complex(self.protein, self.ligand)
        
        self.input_shape = self._complex.protein.get_atom_features().shape[1]

        single_step = np.array([50, 50, 50, 10, 10, 10])
        action_bounds = np.array([-1*single_step, single_step])
        self.action_space = ActionSpace(action_bounds)

    def reset(self):
        self.protein, self.ligand = DataStore.next(False)
        self._complex = Complex(self.protein, self.ligand)
        print("Complex: ", self.protein.path, "Randomized RMSD:", np.round(self._complex.rmsd, 4))
        self.input_shape = self._complex.protein.get_atom_features().shape[1]

        state = self.get_state()
        return self._complex, state

    def step(self, action, testing=False):
        terminal = False
       
        try:
            delta_change = self._complex.ligand.update_pose(*action)
            reward = self._complex.score()
            if testing:
                # print(delta_change)
                terminal = (delta_change < 0.01).all()
            else:   
                terminal = self._complex.perfect_fit
        except:
            reward = -1
            terminal = True
            
        state = self.get_state()
        return self._complex, state, reward, terminal

    def get_state(self, complexes=None):
        if complexes is None:
            proteins = [self._complex.protein]
            ligands = [self._complex.ligand]
        else:
            proteins = [m_complex.protein for m_complex in complexes]
            ligands = [m_complex.ligand for m_complex in complexes]

            
        protein_batch  = self.get_graphcnn_input(self.mols_to_inputs(proteins))
        ligand_batch  = self.get_graphcnn_input(self.mols_to_inputs(ligands))

        state = [protein_batch, ligand_batch] 
        # protein_batch is list of tensors for agglomerated protein molecules 
        # ligand_batch is list of tensors for agglomerated ligand molecules 

        return state

    @staticmethod
    def mols_to_inputs( mols):
        multiConvMol = ConvMol.agglomerate_mols(mols)
        n_samples = np.array([len(mols)])
        all_atom_features = multiConvMol.get_atom_features()
        # scaling of first 3 features, i.e., coordinates
        all_atom_features[:, 0:3] = DataStore.scaler.transform(all_atom_features[:, 0:3])
        inputs = [all_atom_features, multiConvMol.deg_slice,
                    np.array(multiConvMol.membership), n_samples]
        for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
            inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
        return inputs

    @staticmethod
    def get_graphcnn_input(inputs):
        atom_features = np.expand_dims(inputs[0], axis=0)
        degree_slice = np.expand_dims(tf.cast(inputs[1], dtype=tf.int32), axis=0)
        membership = np.expand_dims(tf.cast(inputs[2], dtype=tf.int32), axis=0)
        n_samples = np.expand_dims(tf.cast(inputs[3], dtype=tf.int32), axis=0)
        deg_adjs = [np.expand_dims(tf.cast(deg_adj, dtype=tf.int32), axis=0) for deg_adj in inputs[4:]]

        in_layer = atom_features

        gc_in = [in_layer, degree_slice, membership, n_samples] + deg_adjs
        
        return gc_in
