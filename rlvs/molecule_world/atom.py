import numpy as np
import torch

from openbabel import openbabel as ob
from .molecule import MoleculeType
from .featurizer import Featurizer


class Atoms:
    def __init__(self, molecule_type, obmol, pdb_structure=None):
        self._atoms = []
        self.featurizer = Featurizer(obmol)
        if molecule_type == MoleculeType.PROTEIN:
            pdb_atoms = [atom for atom in pdb_structure.get_atoms()]
            self._atoms = [
                Atom(
                    ob_atom.GetIndex(),
                    molecule_type,
                    pdb_atoms[ob_atom.GetIndex()].name,
                    ob_atom.GetAtomicNum(),
                    pdb_atoms[ob_atom.GetIndex()].coord,
                    ob_atom.GetHyb(),
                    ob_atom.GetHvyDegree(),
                    ob_atom.GetHeteroDegree(),
                    ob_atom.GetPartialCharge(),
                    pdb_atoms[ob_atom.GetIndex()]
                )
                for ob_atom in ob.OBMolAtomIter(obmol)
            ]

        elif molecule_type == MoleculeType.LIGAND:
            self._atoms = [
                Atom(
                    atom.GetIndex(),
                    molecule_type,
                    self.featurizer.atom_codes[atom.GetAtomicNum()],
                    atom.GetAtomicNum(),
                    [atom.GetX(), atom.GetY(), atom.GetZ()],
                    atom.GetHyb(),
                    atom.GetHvyDegree(),
                    atom.GetHeteroDegree(),
                    atom.GetPartialCharge()
                )
                for atom in ob.OBMolAtomIter(obmol)
            ]

        self.bonds = [
            Bond(bond.GetBeginAtom().GetIndex(), bond.GetEndAtom().GetIndex())
            for bond in ob.OBMolBondIter(obmol)
        ]

        self.features = torch.tensor([
            atom.features(self.featurizer) for atom in self._atoms
        ], dtype=torch.float)

        self.edge_index = torch.vstack([bond.edge for bond in self.bonds]).t().contiguous()

    @property
    def x(self):
        return self.features

    @property
    def coords(self):
        return self.features[:,:3]

    @coords.setter
    def coords(self, coords):
        self.features[:, :3] = torch.from_numpy(coords)
        
        for idx, coord in enumerate(coords):
            self._atoms[idx].coord = coord

    def __len__(self):
        return len(self._atoms)

    def where(self, condition):
        return [atom for atom in self._atoms if condition(atom)]


class Atom:
    def __init__(
            self, idx, molecule_type,
            name=None, atomic_num=None,
            coord=None, hyb=None, hvy_degree=None,
            hetro_degree=None, partial_charge=None,
            pdb_atom=None
    ):
        self.idx = idx
        self._type = molecule_type
        self.atomic_num = atomic_num
        self.name = name
        self.coord = coord
        self.pbd_atom = pdb_atom
        self.hyb = hyb
        self.hvy_degree = hvy_degree
        self.hetro_degree = hetro_degree
        self.partial_charge = partial_charge

    @property
    def molecule_type(self):
        return self._type.value

    @property
    def is_c_alpha(self):
        return self.name == 'CA'

    @property
    def is_heavy_metal(self):
        return self.name == 'metal' and self.atomic_num > 20

    @property
    def is_heavy_atom(self):
        return self.atomic_num > 5

    def features(self, featurizer):
        return featurizer.featurize(self)


class Bond:
    def __init__(self, atom_a, atom_b):
        self.atom_a = atom_a
        self.atom_b = atom_b

    @property
    def edge(self):
        return torch.tensor([[self.atom_a, self.atom_b], [self.atom_b, self.atom_a]], dtype=torch.long)

# Element type
#  name
#  element
#  position
#  get_features: similar to featurizer object
#  c_alpha
#  heavy_atom
