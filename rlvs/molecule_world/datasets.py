import os
from os import path
from .helper_functions import OB_to_mol, read_to_OB
import numpy as np

ROOT_PATH = f'{path.dirname(path.abspath(__file__))}/../../data'

class Data:
    DATA_PATH = None
    def __init__(self):
        self._complexes = []
        self.complexes_path = os.listdir(self.DATA_PATH)
        
    @property
    def random(self):
        return self._complexes[np.random.randint(len(self._complexes))]
    
class PafnucyData(Data):
    DATA_PATH=f'{ROOT_PATH}/pafnucy_data/complexes'

    def __init__(self):
        super(PafnucyData, self).__init__()
        self.complexes_path.remove('.DS_Store')
        self.complexes_path.remove('affinities.csv')

        self._complexes = [
            (
                OB_to_mol(
                    read_to_OB(filename=f'{self.DATA_PATH}/{complex}/{complex}_pocket.mol2', filetype="mol2"),
                    mol_type=-1
                ), OB_to_mol(
                    read_to_OB(filename=f'{self.DATA_PATH}/{complex}/{complex}_ligand.mol2', filetype="mol2"),
                    mol_type=1
                )
            ) for complex in self.complexes_path
        ]
        

class DudeProteaseData(Data):
    DATA_PATH=f'{ROOT_PATH}/dude_protease'

    def __init__(self):
        super(DudeProteaseData, self).__init__()
        self.complexes_path.remove('.DS_Store')

        self._complexes = [
            (
                OB_to_mol(
                    read_to_OB(filename=f'{self.DATA_PATH}/{complex}/receptor.pdb', filetype="pdb"),
                    mol_type=-1
                ), OB_to_mol(
                    read_to_OB(filename=f'{self.DATA_PATH}/{complex}/crystal_ligand.mol2', filetype="mol2"),
                    mol_type=1
                )
            ) for complex in self.complexes_path
        ]

    
    
class DataStore:
    DATA_STORES = []

    @classmethod
    def init(cls):
        cls.DATA_STORES = [PafnucyData(), DudeProteaseData()]

    @classmethod
    def next(cls):
        return cls.DATA_STORES[np.random.randint(0, len(cls.DATA_STORES))].random
