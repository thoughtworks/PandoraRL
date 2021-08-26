from openbabel import pybel
from openbabel import openbabel as ob
from os import path
import numpy as np
from .complex import Complex

ob.obErrorLog.SetOutputLevel(0)

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
