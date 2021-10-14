from .atom import Atoms
from .molecule import Molecule, MoleculeType
import numpy as np

from ..helper_functions import read_to_OB, randomizer
import logging

class Ligand(Molecule):
    molecule_type = MoleculeType.LIGAND
    
    def __init__(self, path=None, name=None, filetype=None):
        super(Ligand, self).__init__(path, filetype=filetype)
        obmol = read_to_OB(filename=path, filetype=filetype)
        self.atoms = Atoms(self.molecule_type, obmol)
        self.atom_features = self.atoms.features

    def randomize(self, box_size, action_shape):
        random_pose = randomizer(action_shape)
        print("Randomized", random_pose)
        logging.info(f'Randomized Pose: {random_pose}')
        self.update_pose(*random_pose)
