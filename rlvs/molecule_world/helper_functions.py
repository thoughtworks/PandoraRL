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

def smiles_to_OB(smile_string, prepare=False):
    mol_py = pybel.readstring("smi", smile_string)
    
    if prepare:
       mol_py.make3D(steps=500) 
       mol_py.calccharges(model="gasteiger")
    obmol = mol_py.OBMol
    return obmol

def read_to_OB(filename, filetype, prepare=False):
    obconversion = ob.OBConversion()
    obconversion.SetInFormat(filetype)
    obmol = ob.OBMol()

    notatend = obconversion.ReadFile(obmol, filename)
    if prepare:
        mol_py = pybel.Molecule(obmol)
        mol_py.addh()
        mol_py.calccharges(model="gasteiger")
        obmol = mol_py.OBMol

    return obmol

def OB_to_mol(obmol, mol_type, path=None):
    f = Featurizer()
    nodes, canon_adj_list = f.get_mol_features(obmol=obmol, molecule_type=mol_type, bond_verbose=0)
    mol = Molecule(atom_features=nodes, canon_adj_list=canon_adj_list, path=path)
    return mol

def mol_to_OB(mol, filetype, scaler, prepare):
    # TODO: do we need to recenter coords now?? 
    # all_features = scaler.inverse_transform(mol.get_ordered_features())
    all_features = mol.get_ordered_features()
    if filetype=="smiles_string":
        obmol = smiles_to_OB(mol.path, prepare=prepare)
    else:   
        obmol = read_to_OB(mol.path, filetype, prepare=prepare)
    assert(obmol.NumAtoms()==all_features.shape[0])
    for atom_id, atom in enumerate(ob.OBMolAtomIter(obmol)):
        atom.SetVector(*(all_features[atom_id, 0:3]))

    return obmol

def OBs_to_file(obmol_ligand, filename, filetype):

    mol_py = pybel.Molecule(obmol_ligand)
    mol_py.write(format=filetype, filename=filename, overwrite=True)
    return
