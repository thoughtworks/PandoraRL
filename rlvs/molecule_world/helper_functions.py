from openbabel import pybel
from openbabel import openbabel as ob
from os import path
import numpy as np
from .molecule import Molecule
from .complex import Complex
from .featurizer import Featurizer

DATA_PATH = f'{path.dirname(path.abspath(__file__))}/../../pdbqt_data/'
PROTEIN_FILES = ['6Y2F_MOD.pdbqt']
LIGAND_FILES = [
    'letermovir.pdbqt',
    'lopinavir.pdbqt',
    'raltegravir.pdbqt',
    'ritonavir.pdbqt',
    'tipranavir.pdbqt'
]

def read_to_OB(filename, filetype):
    obconversion = ob.OBConversion()
    obconversion.SetInFormat(filetype)
    obmol = ob.OBMol()

    notatend = obconversion.ReadFile(obmol, filename)
    # print(obmol.GetFormula())
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

# TODO: For faster processing
#protein = OB_to_mol(read_to_OB(filename=f'{DATA_PATH}{PROTEIN_FILES[0]}', filetype="pdbqt"), mol_type=-1)
#bound_ligand = OB_to_mol(read_to_OB(filename=f'{DATA_PATH}a-ketoamide-13b.pdbqt', filetype="pdbqt"), mol_type=1)
