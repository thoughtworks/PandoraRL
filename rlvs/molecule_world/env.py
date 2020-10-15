import numpy as np
from .complex import Complex
from .molecule import Molecule
from .datasets import DataStore
from deepchem.feat.mol_graphs import MultiConvMol, ConvMol

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

        single_step = np.array([10, 10, 10, 10, 10, 10])
        action_bounds = np.array([-1*single_step, single_step])
        self.action_space = ActionSpace(action_bounds)

    def reset(self):
        self.protein, self.ligand = DataStore.next(False)
        self._complex = Complex(self.protein, self.ligand)
        
        self.input_shape = self._complex.protein.get_atom_features().shape[1]

        state = self.get_state()
        return state

    def step(self, action):
        terminal = False
       
        try:
            self._complex.ligand.update_pose(*action)
            reward = self._complex.score()
            terminal = self._complex.perfect_fit
        except:
            reward = -1
            terminal = True
            
        state = self.get_state()
        return state, reward, terminal

    def get_state(self):
        # state = [] # state is a list of 2 lists of tensors 
        # for mol in [self._complex.protein, self._complex.ligand]:
        #     atom_features = np.expand_dims(mol.get_atom_features(), axis=0)
        #     degree_slice = np.expand_dims((mol.deg_slice.astype(dtype='int32')), axis=0)
        #     deg_adjs = [np.expand_dims(deg_adj.astype(dtype='int32'), axis=0) for deg_adj in mol.get_deg_adjacency_lists()[1:]]

        #     gc_in = [atom_features, degree_slice] + deg_adjs
        #     state.append(gc_in)

        # since this is single batch
        proteins = [self._complex.protein]
        ligands = [self._complex.ligand]
        protein_batch  = get_graphcnn_input(mols_to_inputs(proteins))
        ligand_batch  = get_graphcnn_input(mols_to_inputs(ligands))

        state = [protein_batch, ligand_batch] 
        # protein_batch is list of tensors for agglomerated protein molecules 
        # ligand_batch is list of tensors for agglomerated ligand molecules 

        return state

    def mols_to_inputs(mols):
        multiConvMol = ConvMol.agglomerate_mols(mols)
        n_samples = np.array([len(mols)])
        inputs = [multiConvMol.get_atom_features(), multiConvMol.deg_slice,
                    np.array(multiConvMol.membership), n_samples]
        for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
            inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
        return inputs
    
    def get_graphcnn_input(inputs):
        atom_features = np.expand_dims(inputs[0], axis=0)
        degree_slice = np.expand_dims(tf.cast(inputs[1], dtype=tf.int32), axis=0)
        membership = np.expand_dims(tf.cast(inputs[2], dtype=tf.int32), axis=0)
        n_samples = np.expand_dims(tf.cast(inputs[3], dtype=tf.int32), axis=0)
        deg_adjs = [np.expand_dims(tf.cast(deg_adj, dtype=tf.int32), axis=0) for deg_adj in inputs[4:]]

        in_layer = atom_features

        gc_in = [in_layer, degree_slice, membership, n_samples] + deg_adjs
        
        return gc_in