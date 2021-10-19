# Code referenced from https://gitlab.com/cheminfIBB/pafnucy

from openbabel import pybel
from openbabel import openbabel as ob
import numpy as np

import torch
from torch_geometric.data import Data
from .types import MoleculeType

from rlvs.constants import RESIDUES, Z_SCORES

ob.obErrorLog.SetOutputLevel(0)


class Featurizer():
    def __init__(self, obmol=None):

        # dict of atom codes for one hot encoding
        self.atom_codes = {}
        self.class_codes = {}
        metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
                  + list(range(37, 51)) + list(range(55, 84))
                  + list(range(87, 104)))

        atom_classes = [
            (1, 'H'),
            (5, 'B'),
            (6, 'C'),
            (7, 'N'),
            (8, 'O'),
            (15, 'P'),
            (16, 'S'),
            (34, 'Se'),
            ([9, 17, 35, 53], 'halogen'),
            (metals, 'metal')
        ]


        self.residues = RESIDUES

        for code, (atom, name) in enumerate(atom_classes):
            if type(atom) is list:
                for a in atom:
                    self.atom_codes[a] = code
            else:
                self.atom_codes[atom] = code
            self.class_codes[code] = name
        self.num_classes = len(atom_classes)

        self.SMARTS = [
            '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
            '[a]',
            '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
            '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
            '[r]'
        ]
        self.smarts_labels = ['hydrophobic', 'aromatic', 'acceptor', 'donor',
                              'ring']

        self.__PATTERNS = []
        for smarts in self.SMARTS:
            self.__PATTERNS.append(pybel.Smarts(smarts))

        self.smarts_patterns = None
        if obmol is not None:
            mol_py = pybel.Molecule(obmol)
            self.smarts_patterns = self.find_smarts(mol_py)

    def featurize(self, atom):
        '''
        INPUT
        atom: OB Atom object
        molecule_type: 1 for ligand, 0 for protein

        OUTPUT

        features:
            [
                x,
                y,
                z,
                encoding (10 bit one hot encoding),
                hyb (1,2 or 3),
                heavy_valence (integer),
                hetero_valence (integer),
                partial_charge (float),
                is_heavy_atom,
                vander_wal_radius,
                molecule_type (1 for ligand, -1 for protein),
                smarts_patterns(5)
                encoding residue type for protein atoms
            ]
        '''
        features = []
        # features.extend(atom.coord)

        # one hot encode atomic number
        encoding = np.zeros(self.num_classes, dtype=int)
        encoding[self.atom_codes[atom.atomic_num]] = 1
        features.extend(encoding)

        # hybridization, heavy valence, hetero valence, partial charge
        named_features = [
            atom.hyb, atom.hvy_degree,
            atom.hetro_degree, atom.partial_charge
        ]

        residue = np.zeros(len(self.residues), dtype=int)
        z_scores = [0] * 5

        if atom.molecule_type == MoleculeType.PROTEIN:
            residue[self.residues.get(atom.residue.upper(), 20)] = 1
            z_scores = Z_SCORES.get(atom.residue.upper(), [0, 0, 0, 0, 0])


        features.extend(named_features)
        features.extend([int(atom.is_heavy_atom)])
        features.extend([atom.VDWr])
        features.extend([atom.molecule_type])
        features.extend(self.smarts_patterns[atom.idx])
        features.extend(residue)
        features.extend(z_scores)
            
        return features

    def find_smarts(self, molecule):
        """Find atoms that match SMARTS patterns.

        Parameters
        ----------
        molecule: pybel.Molecule

        Returns
        -------
        features: np.ndarray
        NxM binary array, where N is the number of atoms in the `molecule`
            and M is the number of patterns. `features[i, j]` == 1.0 if i'th
            atom has j'th property
        """

        features = np.zeros((len(molecule.atoms), len(self.__PATTERNS)))

        for (pattern_id, pattern) in enumerate(self.__PATTERNS):
            atoms_with_prop = np.array(list(*zip(*pattern.findall(molecule))),
                                       dtype=int) - 1
            features[atoms_with_prop, pattern_id] = 1.0
        return features
