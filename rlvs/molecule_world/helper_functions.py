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

def OB_to_mol(obmol, mol_type, path=None):
    f = Featurizer()
    nodes, canon_adj_list = f.get_mol_features(obmol=obmol, molecule_type=mol_type, bond_verbose=0)
    mol = Molecule(atom_features=nodes, canon_adj_list=canon_adj_list, path=path)
    return mol

def mol_to_OB(mol, filetype, scaler):
    # TODO: do we need to recenter coords now?? 
    # all_features = scaler.inverse_transform(mol.get_ordered_features())
    all_features = mol.get_ordered_features()
    obmol = read_to_OB(mol.path, filetype)
    assert(obmol.NumAtoms()==mol.get_coords().shape[0])
    for atom_id, atom in enumerate(ob.OBMolAtomIter(obmol)):
        atom.SetVector(*(all_features[atom_id, 0:3]))

    return obmol

def OBs_to_file(obmol_protein, obmol_ligand, filename, filetype):

    mol_py = pybel.Molecule(obmol_ligand)
    mol_py.write(format=filetype, filename=filename, overwrite=True)
    # outputfile = pybel.Outputfile(format=filetype, filename=filename, overwrite=True)
    # outputfile.write(pybel.Molecule(obmol_protein))
    # outputfile.write(pybel.Molecule(obmol_ligand)) 
    # outputfile.close()
    return
