from .atom import Atoms
from .molecule import Molecule, MoleculeType

class Ligand(Molecule):
    molecule_type = MoleculeType.LIGAND
    def __init__(self, obmol, path=None, name=None):
        super(Ligand, self).__init__(path)
        self.atoms = Atoms(self.molecule_type, obmol)
        self.atom_features = self.atoms.features
