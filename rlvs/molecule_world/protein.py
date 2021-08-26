from .atom import Atoms
from .molecule import Molecule, MoleculeType
from Bio import PDB

from .helper_functions import read_to_OB

class Protein(Molecule):
    molecule_type = MoleculeType.PROTEIN
    def __init__(self, path=None, name="XX", filetype=None):
        super(Protein, self).__init__(path, filetype)
        obmol = read_to_OB(filename=path, filetype=filetype)
        parser = PDB.PDBParser()
        self.pdb_structure = parser.get_structure(name, path)
        self.atoms = Atoms(self.molecule_type, obmol, self.pdb_structure)
        self.atom_features = self.atoms.features
        
        
