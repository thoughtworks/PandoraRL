from .atom import Atoms
from .molecule import Molecule, MoleculeType
import numpy as np

class Ligand(Molecule):
    molecule_type = MoleculeType.LIGAND
    def __init__(self, obmol, path=None, name=None):
        super(Ligand, self).__init__(path)
        self.atoms = Atoms(self.molecule_type, obmol)
        self.atom_features = self.atoms.features

    def randomize(self, box_size, action_shape):
        random_pose = np.random.uniform(-box_size, box_size, (action_shape,)) * 10
        print("Randomized", random_pose)
        self.update_pose(*random_pose)
