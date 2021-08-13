from .atom import Atoms
from .molecule import Molecule, MoleculeType
import numpy as np

class Ligand(Molecule):
    molecule_type = MoleculeType.LIGAND
    def __init__(self, obmol, path=None, name=None):
        super(Ligand, self).__init__(path)
        self.atoms = Atoms(self.molecule_type, obmol)
        self.atom_features = self.atoms.features

    def randomize(self, box_size):
        x, y, z, r, p, y_ = np.random.uniform(-box_size, box_size, (6,)) * 10
        print("Randomized", x, y, z, r, p, y_)
        self.update_pose(x, y, z, r, p, y_)
