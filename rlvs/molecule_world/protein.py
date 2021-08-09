from .atom import Atoms
from .molecule import Molecule, MoleculeType
from Bio import PDB

class Protein(Molecule):
    molecule_type = MoleculeType.PROTEIN
    def __init__(self, obmol, path=None, name="XX"):
        super(Protein, self).__init__(path)
        parser = PDB.PDBParser()
        self.pdb_structure = parser.get_structure(name, path)
        self.atoms = Atoms(self.molecule_type, obmol, self.pdb_structure)
        self.atom_features = self.atoms.features
        
        
