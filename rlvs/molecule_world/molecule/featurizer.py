# Code referenced from https://gitlab.com/cheminfIBB/pafnucy

from openbabel import pybel
from openbabel import openbabel as ob
import numpy as np

import torch
from torch_geometric.data import Data
from .types import MoleculeType

from rlvs.constants import RESIDUES, Z_SCORES, KD_HYDROPHOBICITYA, CONFORMATION_SIMILARITY
from rlvs.config import Config

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

        self.config = Config.get_instance()
        self.residues = RESIDUES
        self.hydrophobicity = KD_HYDROPHOBICITYA
        self.conf_sim = CONFORMATION_SIMILARITY
        self.hydophobicity_max_grps = max(np.unique(list(self.hydrophobicity.values())))
        self.conf_sim_max_grps = max(np.unique(list(self.conf_sim.values())))

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

    def atom_type_encoding(self, atom):
        encoding = np.zeros(self.num_classes, dtype=int)
        encoding[self.atom_codes[atom.atomic_num]] = 1

        return encoding

    def interaction_strengths(self, atom):
        return atom.inter_molecular_interactions.features

    def atom_named_features(self, atom):
        return [
            atom.hyb, atom.hvy_degree,
            atom.hetro_degree, atom.partial_charge
        ]

    def is_heavy_atom(self, atom):
        return [int(atom.is_heavy_atom)]

    def VDWr(self, atom):
        return [atom.VDWr]

    def molecule_type(self, atom):
        return [atom.molecule_type]

    def smarts_pattern_encoding(self, atom):
        return self.smarts_patterns[atom.idx]
    
    def residue_labels(self, atom):
        residue = np.zeros(len(self.residues), dtype=int)
        if atom.molecule_type == MoleculeType.PROTEIN:
            residue[self.residues.get(atom.residue.upper(), 20)] = 1

        return residue

    def kd_hydophobocitya(self, atom):
        kd_hydophobocitya = np.zeros(len(np.unique(list(self.hydrophobicity.values()))))
        
        if atom.molecule_type == MoleculeType.PROTEIN:
            kd_hydophobocitya[self.hydrophobicity.get(atom.residue.upper(), self.hydophobicity_max_grps)] = 1

        return kd_hydophobocitya

    def conformational_similarity(self, atom):
        conformational_similarity = np.zeros(len(np.unique(list(self.conf_sim.values()))))

        if atom.molecule_type == MoleculeType.PROTEIN:
            conformational_similarity[self.conf_sim.get(atom.residue.upper(), self.conf_sim_max_grps)] = 1

        return conformational_similarity


    def z_scores(self, atom):
        if atom.molecule_type == MoleculeType.PROTEIN:
            return Z_SCORES.get(atom.residue.upper(), [0, 0, 0, 0, 0])

        return [0] * 5
            
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
        for method in self.config.node_features:
            features.extend(getattr(self, method)(atom))
            
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
