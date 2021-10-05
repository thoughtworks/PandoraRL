from .atom import Atoms
from .molecule import Molecule, MoleculeType
from Bio import PDB
from openbabel import pybel
from openbabel import openbabel as ob

import numpy as np

from .helper_functions import read_to_OB

class Protein(Molecule):
    molecule_type = MoleculeType.PROTEIN
    def __init__(self, path=None, name="XX", filetype=None):
        super(Protein, self).__init__(path, filetype)
        obmol = read_to_OB(filename=path, filetype=filetype)
        parser = PDB.PDBParser()
        self.name=name
        self.pdb_structure = parser.get_structure(name, path)
        self.atoms = Atoms(self.molecule_type, obmol, self.pdb_structure)
        self.atom_features = self.atoms.features

    def save(self, output_path, output_type):
        obmol = read_to_OB(filename=self.path, filetype=self.filetype, prepare=True)
        new_obmol = ob.OBMol()
        for atom in self.atoms:
            ob_atom = obmol.GetAtom(atom.atom_idx + 1)
            coord = np.array(atom.coord, dtype=np.float64)
            ob_atom.SetVector(*coord)
            new_obmol.AddAtom(ob_atom)

        for bond in self.atoms.bonds:
            ob_bond = obmol.GetBond(bond.idx)
            new_obmol.AddBond(ob_bond)
            
        mol_py = pybel.Molecule(new_obmol)
        mol_py.write(format=output_type, filename=output_path, overwrite=True)

        
