from openbabel import pybel
from openbabel import openbabel as ob
import numpy as np
from rlvs.constants import ComplexConstants

ob.obErrorLog.SetOutputLevel(0)
RANDOM_POS_SIGN = 1
box_size = ComplexConstants.RMSD_THRESHOLD  # 0.1


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

    obconversion.ReadFile(obmol, filename)
    if prepare:
        mol_py = pybel.Molecule(obmol)
        mol_py.addh()
        mol_py.calccharges(model="gasteiger")
        obmol = mol_py.OBMol

    return obmol


def randomizer(action_shape, test=False):
    global box_size, RANDOM_POS_SIGN

    if not test:
        pose = np.random.uniform(-box_size, box_size, (action_shape,))
    else:
        pose = np.random.uniform(
            -ComplexConstants.RMSD_THRESHOLD,  ComplexConstants.RMSD_THRESHOLD, (action_shape,)
        )
    return pose
