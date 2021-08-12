import numpy as np
import torch

from openbabel import openbabel as ob
from .molecule import MoleculeType
from .featurizer import Featurizer


class Atoms:
    def __init__(self, molecule_type, obmol, pdb_structure=None):
        self._atoms = []
        featurizer = Featurizer(obmol)
        if molecule_type == MoleculeType.PROTEIN:
            pdb_atoms = [atom for atom in pdb_structure.get_atoms()]
            self._atoms = [
                Atom(
                    (idx:=ob_atom.GetIndex()),
                    molecule_type,
                    pdb_atoms[idx].name,
                    (atm_no:=ob_atom.GetAtomicNum()),
                    pdb_atoms[idx].coord,
                    ob_atom.GetHyb(),
                    ob_atom.GetHvyDegree(),
                    ob_atom.GetHeteroDegree(),
                    ob_atom.GetPartialCharge(),
                    ob.GetVdwRad(atm_no),
                    featurizer.smarts_patterns[idx],
                    pdb_atoms[idx],
                    ob_atom.GetResidue().GetName()
                )
                for ob_atom in ob.OBMolAtomIter(obmol)
            ]

        elif molecule_type == MoleculeType.LIGAND:
            self._atoms = [
                Atom(
                    (idx:=atom.GetIndex()),
                    molecule_type,
                    featurizer.atom_codes[(atm_no:=atom.GetAtomicNum())],
                    atm_no,
                    np.array([atom.GetX(), atom.GetY(), atom.GetZ()]),
                    atom.GetHyb(),
                    atom.GetHvyDegree(),
                    atom.GetHeteroDegree(),
                    atom.GetPartialCharge(),
                    ob.GetVdwRad(atm_no),
                    featurizer.smarts_patterns[idx]
                )
                for atom in ob.OBMolAtomIter(obmol)
            ]

        self.bonds = [
            Bond(
                self._atoms[bond.GetBeginAtom().GetIndex()],
                self._atoms[bond.GetEndAtom().GetIndex()],
                bond.GetLength()
            )
            for bond in ob.OBMolBondIter(obmol)
        ]

        self.features = torch.tensor([
            atom.features(featurizer) for atom in self._atoms
        ], dtype=torch.float)

        self.edge_index = torch.vstack(
            [bond.edge for bond in self.bonds]
        ).t().contiguous()

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

    def __iter__(self):
        return self._atoms.__iter__()

    def where(self, condition):
        return [atom for atom in self._atoms if condition(atom)]


class Atom:
    def __init__(
            self, idx=-1, molecule_type=MoleculeType.LIGAND,
            name=None, atomic_num=None,
            coord=None, hyb=None, hvy_degree=None,
            hetro_degree=None, partial_charge=None,
            VDWr=None, smarts=None, pdb_atom=None,
            residue=None
            
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
        self.VDWr = VDWr
        self.hydrophobic,\
            self.aromatic,\
            self.acceptor,\
            self.donor,\
            self.ring = np.array(
            smarts, dtype=bool
        )

        self.residue = residue
        self.bonds = np.array([])

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

    def add_bond(self, bond):
        self.bonds = np.append(self.bonds, bond)


class Bond:
    def __init__(self, atom_a, atom_b, bond_length, update_edge=True):
        self.atom_a = atom_a
        self.atom_b = atom_b
        self.distance = np.sqrt(np.sum((atom_a.coord - atom_b.coord) ** 2))
        self.lenght = bond_length

        if update_edge:
            atom_a.add_bond(self)
            atom_b.add_bond(self)

    @property
    def edge(self):
        return torch.tensor([
            [self.atom_a.idx, self.atom_b.idx],
            [self.atom_b.idx, self.atom_a.idx]
        ], dtype=torch.long)
