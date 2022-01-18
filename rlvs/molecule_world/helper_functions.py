from openbabel import pybel
from openbabel import openbabel as ob
import numpy as np
from rlvs.constants import ComplexConstants

ob.obErrorLog.SetOutputLevel(0)
box_size = 8 #0.1
RANDOM_POS_SIGN = 1


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
        pose = RANDOM_POS_SIGN * np.random.uniform(0, box_size, (action_shape,))
        box_size = min(box_size + 0.005, ComplexConstants.RMSD_THRESHOLD)
    else:
        pose = RANDOM_POS_SIGN * np.random.uniform(
            0,  ComplexConstants.RMSD_THRESHOLD, (action_shape,)
        )

    RANDOM_POS_SIGN *= -1
    return pose
