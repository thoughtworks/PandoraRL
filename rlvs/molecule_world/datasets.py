import os
from os import path
from .helper_functions import OB_to_mol, read_to_OB
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .protein import Protein
from .ligand import Ligand
from .complex import Complex

ROOT_PATH = f'{path.dirname(path.abspath(__file__))}/../../data'


class Data:
    DATA_PATH = None
    def __init__(self):
        self._complexes = []
        self.complexes_path = os.listdir(self.DATA_PATH)

    def get_molecules(self, complex, crop=True):
        if crop:
            complex.crop(10, 10, 10)
            
        return complex

    @property
    def random(self):
        return self._complexes[np.random.randint(len(self._complexes))]


class PafnucyData(Data):
    DATA_PATH = f'{ROOT_PATH}/pafnucy_data/complexes'

    def __init__(self):
        super(PafnucyData, self).__init__()
        if '.DS_Store' in self.complexes_path:
            self.complexes_path.remove('.DS_Store')
        if 'affinities.csv' in self.complexes_path:
            self.complexes_path.remove('affinities.csv')

        self._complexes = [
            Complex(
                Protein(
                    read_to_OB(
                        filename=f'{self.DATA_PATH}/{complex}/{complex}_pocket.pdb',
                        filetype="pdb"
                    ),
                    path=f'{self.DATA_PATH}/{complex}/{complex}_pocket.pdb'
                ), Ligand(
                    read_to_OB(
                        filename=f'{self.DATA_PATH}/{complex}/{complex}_ligand.mol2',
                        filetype="mol2"
                    ),
                    path=f'{self.DATA_PATH}/{complex}/{complex}_ligand.mol2'
                ), Ligand(
                    read_to_OB(
                        filename=f'{self.DATA_PATH}/{complex}/{complex}_ligand.mol2',
                        filetype="mol2"
                    ),
                    path=f'{self.DATA_PATH}/{complex}/{complex}_ligand.mol2'
                )
            ) for complex in self.complexes_path
        ]


class SARSVarients(Data):
    DATA_PATH = f'{ROOT_PATH}/SARS_variants/'

    def __init__(self):
        super(SARSVarients, self).__init__()
        if '.DS_Store' in self.complexes_path:
            self.complexes_path.remove('.DS_Store')

        self._complexes = [
            Complex(
                Protein(
                    read_to_OB(
                        filename=f'{self.DATA_PATH}/{complex}/{complex}_protein.pdb',
                        filetype="pdb"
                    ),
                    path=f'{self.DATA_PATH}/{complex}/{complex}_protein.pdb'
                ), Ligand(
                    read_to_OB(
                        filename=f'{self.DATA_PATH}/{complex}/{complex}_ligand.pdbqt',
                        filetype="pdbqt"
                    ),
                    path=f'{self.DATA_PATH}/{complex}/{complex}_ligand.pdbqt'
                ), Ligand(
                    read_to_OB(
                        filename=f'{self.DATA_PATH}/{complex}/{complex}_ligand.pdbqt',
                        filetype="pdbqt"
                    ),
                    path=f'{self.DATA_PATH}/{complex}/{complex}_ligand.pdbqt'
                )
            ) for complex in self.complexes_path
        ]


class PDBQTData(Data):
    DATA_PATH = f'{ROOT_PATH}/pdbqt_data'

    def __init__(self):
        super(PDBQTData, self).__init__()
        self.complexes_path = [
            ('6Y2F_MOD.pdb', 'a-ketoamide-13b.pdbqt')
        ]

        self._complexes = [
            Complex(
                Protein(
                    read_to_OB(
                        filename=f'{self.DATA_PATH}/{complex}',
                        filetype="pdb"
                    ),
                    path=f'{self.DATA_PATH}/{complex}'
                ), Ligand(
                    read_to_OB(
                        filename=f'{self.DATA_PATH}/{ligand}',
                        filetype="pdbqt"
                    ),
                    path=f'{self.DATA_PATH}/{ligand}'
                ), Ligand(
                    read_to_OB(
                        filename=f'{self.DATA_PATH}/{ligand}',
                        filetype="pdbqt"
                    ),
                    path=f'{self.DATA_PATH}/{ligand}'
                )
            ) for complex, ligand in self.complexes_path
        ]


class PDBQTData_2(Data):
    DATA_PATH = f'{ROOT_PATH}/pdbqt_data_2'

    def __init__(self):
        super(PDBQTData_2, self).__init__()
        if '.DS_Store' in self.complexes_path:
            self.complexes_path.remove('.DS_Store')
        

        self._complexes = [
            Complex(
                Protein(
                    read_to_OB(
                        filename=f'{self.DATA_PATH}/{complex}/{complex}_protein.pdb',
                        filetype="pdb"
                    ),
                    path=f'{self.DATA_PATH}/{complex}/{complex}_protein.pdb'
                ), Ligand(
                    read_to_OB(
                        filename=f'{self.DATA_PATH}/{complex}/{complex}_ligand.mol2',
                        filetype="mol2"
                    ),
                    path=f'{self.DATA_PATH}/{complex}/{complex}_ligand.mol2'
                ), Ligand(
                    read_to_OB(
                        filename=f'{self.DATA_PATH}/{complex}/{complex}_ligand.mol2',
                        filetype="mol2"
                    ),
                    path=f'{self.DATA_PATH}/{complex}/{complex}_ligand.mol2'
                )
            ) for complex in self.complexes_path
        ]


class DudeProteaseData(Data):
    DATA_PATH = f'{ROOT_PATH}/dude_protease'

    def __init__(self):
        super(DudeProteaseData, self).__init__()
        if '.DS_Store' in self.complexes_path:
            self.complexes_path.remove('.DS_Store')

        self._complexes = [
            Complex(
                Protein(
                    read_to_OB(
                        filename=f'{self.DATA_PATH}/{complex}/receptor.pdb',
                        filetype="pdb"
                    ),
                    path=f'{self.DATA_PATH}/{complex}/receptor.pdb'
                ), Ligand(
                    read_to_OB(
                        filename=f'{self.DATA_PATH}/{complex}/crystal_ligand.mol2',
                        filetype="mol2"
                    ),
                    path=f'{self.DATA_PATH}/{complex}/crystal_ligand.mol2'
                ), Ligand(
                    read_to_OB(
                        filename=f'{self.DATA_PATH}/{complex}/crystal_ligand.mol2',
                        filetype="mol2"
                    ),
                    path=f'{self.DATA_PATH}/{complex}/crystal_ligand.mol2'
                )
                
            ) for complex in self.complexes_path
        ]


class DataStore:
    DATA_STORES = []
    DATA = []

    @classmethod
    def init(cls, crop=True):
        cls.DATA_STORES = [PDBQTData(), PDBQTData_2()]
        cls.load(crop)
        # cls.scaler = cls.normalize()

    @classmethod
    def load(cls, crop=True):
        cls.DATA = [
            store.get_molecules(complex, crop)
            for store in cls.DATA_STORES
            for complex in store._complexes
        ]

    @classmethod
    def next(cls, crop=True):
        return cls.DATA[np.random.randint(0, len(cls.DATA))]

    @classmethod
    def normalize(cls):
        X = []
        for complex in cls.DATA:
            protein, ligand = complex.protein, complex.ligand
            X.extend(list(protein.get_atom_features()))
            X.extend(list(ligand.get_atom_features()))
        X = np.asarray(X)
        scaler = MinMaxScaler()
        scaler.fit(X)
        return scaler
