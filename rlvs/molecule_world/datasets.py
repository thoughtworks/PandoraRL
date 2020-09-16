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
        return self._complexes[np.random.randint(len(self._complexes))]()
    
class PafnucyData(Data):
    DATA_PATH=f'{ROOT_PATH}/pafnucy_data/complexes'

    def __init__(self):
        super(PafnucyData, self).__init__()
        if '.DS_Store' in self.complexes_path:
            self.complexes_path.remove('.DS_Store')
        if 'affinities.csv' in self.complexes_path:            
            self.complexes_path.remove('affinities.csv')

        self._complexes = [
            lambda: (
                OB_to_mol(
                    read_to_OB(filename=f'{self.DATA_PATH}/{complex}/{complex}_pocket.mol2', filetype="mol2"),
                    mol_type=-1
                ), OB_to_mol(
                    read_to_OB(filename=f'{self.DATA_PATH}/{complex}/{complex}_ligand.mol2', filetype="mol2"),
                    mol_type=1
                )
            ) for complex in self.complexes_path
        ]

class PDBQTData(Data):
    DATA_PATH=f'{ROOT_PATH}/pdbqt_data'

    def __init__(self):
        super(DudeProteaseData, self).__init__()
        self.complexes_path = [
            ('6Y2F_MOD.pdbqt', 'a-ketoamide-13b.pdbqt')
        ]

        self._complexes = [
            lambda: (
                OB_to_mol(
                    read_to_OB(filename=f'{self.DATA_PATH}/{complex}', filetype="pdbqt"),
                    mol_type=-1
                ), OB_to_mol(
                    read_to_OB(filename=f'{self.DATA_PATH}/{ligand}', filetype="pdbqt"),
                    mol_type=1
                )
            ) for complex, ligand in self.complexes_path
        ]

class DudeProteaseData(Data):
    DATA_PATH=f'{ROOT_PATH}/dude_protease'

    def __init__(self):
        super(DudeProteaseData, self).__init__()
        if '.DS_Store' in self.complexes_path:
            self.complexes_path.remove('.DS_Store')


        self._complexes = [
            lambda: (
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
        cls.DATA_STORES = [PDBQTData(), PafnucyData(), DudeProteaseData()]

    @classmethod
    def next(cls):
        return cls.DATA_STORES[0].random
        #return cls.DATA_STORES[np.random.randint(0, len(cls.DATA_STORES))].random
