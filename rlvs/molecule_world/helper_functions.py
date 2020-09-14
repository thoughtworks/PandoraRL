from openbabel import pybel
from openbabel import openbabel as ob
import os
import numpy as np
from molecule import Molecule
from .complex import Complex

DATA_PATH = f'{path.dirname(path.abspath(__file__))}/../../pdbqt_data/'
PROTEIN_FILES = ['6Y2F_MOD.pdbqt']
LIGAND_FILES = [
    'letermovir.pdbqt',
    'lopinavir.pdbqt',
    'raltegravir.pdbqt',
    'ritonavir.pdbqt',
    'tipranavir.pdbqt'
]

# TODO: For faster processing
protein = OB_to_mol(read_to_OB(filename=f'{DATA_PATH}{PROTEIN_FILES[0]}', filetype="pdbqt"), mol_type=-1)
bound_ligand = OB_to_mol(read_to_OB(filename=f'{DATA_PATH}{a-ketoamide-13b.pdbqt}', filetype="pdbqt"), mol_type=1)



class Featurizer():    
    def __init__(self):

        # dict of atom codes for one hot encoding
        self.atom_codes = {}
        self.class_codes = {}
        metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
                          + list(range(37, 51)) + list(range(55, 84))
                          + list(range(87, 104)))

        atom_classes = [
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

        for code, (atom, name) in enumerate(atom_classes):
            if type(atom) is list:
                for a in atom:
                    self.atom_codes[a] = code
            else:
                self.atom_codes[atom] = code
            self.class_codes[code] = name
        self.num_classes = len(atom_classes)

    def get_atom_features(self, atom, molecule_type):
        #TODO: add SMARTS patterns 
        '''
        INPUT
        atom: OB Atom object
        molecule_type: 1 for ligand, -1 for protein
        
        OUTPUT
        [(x,y,z), [features]] 
        where features:
            - atom code (TEMPORARY FEATURE)
            - encoding (9 bit one hot encoding)
            - hyb (1,2 or 3)
            - heavy_valence (integer)
            - hetero_valence (integer)
            - partial_charge (float)
            - molecule_type (1 for ligand, -1 for protein)
        '''
        features = []
        
        features.append(self.atom_codes[atom.GetAtomicNum()])
        
        # one hot encode atomic number
        encoding = np.zeros(self.num_classes)
        encoding[self.atom_codes[atom.GetAtomicNum()]] = 1
        features.extend(encoding)
        
        # hybridization, heavy valence, hetero valence, partial charge
        named_features = [atom.GetHyb(), atom.GetHvyDegree(), atom.GetHeteroDegree(), atom.GetPartialCharge()]
        features.extend(named_features)
        
        #molecule type
        molecule_type = molecule_type
        features.append(molecule_type)
        
        return [atom.GetX(), atom.GetY(), atom.GetZ()], features
    
    def get_mol_features(self, obmol, molecule_type, bond_verbose=False):
        num_atoms = obmol.NumAtoms()
        coords, features = [], []
        for atom in ob.OBMolAtomIter(obmol):
            # add only heavy atoms
            if atom.GetAtomicNum() > 1:
                crds, feats = self.get_atom_features(atom, molecule_type)
                coords.append(crds)
                features.append(feats)
        coords = np.array(coords)
        features = np.array(features)
        print(f"Shape of coords: {coords.shape}")
        print(f"Shape of features: {features.shape}")
        if bond_verbose:
            print(f"Bond information:\n")
            for bond in ob.OBMolBondIter(obmol):
                print(bond.GetBeginAtom().GetType(), bond.GetEndAtom().GetType())
        return coords, features
    
def read_to_OB(filename, filetype):
    obconversion = ob.OBConversion()
    obconversion.SetInFormat(filetype)
    obmol = ob.OBMol()

    notatend = obconversion.ReadFile(obmol, filename)
    print(obmol.GetFormula())
    return obmol

def OB_to_mol(obmol, mol_type):
    
    f = Featurizer()
    coords, feats = f.get_mol_features(obmol=obmol, molecule_type=mol_type, bond_verbose=0)
    mol = Molecule(coords=coords, features=feats)
    return mol

def get_molecules(ligand_path=None, protein_path=None):
    global protein
    ligand_path = f'{DATA_PATH}{LIGAND_FILES[np.random.randint(len(LIGAND_FILES))]}'
    protien_path = f'{DATA_PATH}{PROTEIN_FILES[0]}'
    ligand = OB_to_mol(read_to_OB(filename=ligand_path, filetype="pdbqt"), mol_type=1)
    #protein = OB_bto_mol(read_to_OB(filename=protein_path, filetype="pdbqt"), mol_type=-1)
    protein.crop(bound_ligand.get_centroid(), 10, 10, 10)
    return protein, ligand
