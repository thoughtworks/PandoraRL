import os
from os import path
from .helper_functions import OB_to_mol, read_to_OB
import numpy as np
from sklearn.preprocessing import MinMaxScaler

ROOT_PATH = f'{path.dirname(path.abspath(__file__))}/../../data'

class Data:
    DATA_PATH = None
    def __init__(self):
        self._complexes = []
        self.complexes_path = os.listdir(self.DATA_PATH)

    def get_molecules(self, complex, crop=True):
        protein, ligand = complex
        if crop:
            protein.crop(ligand.get_centroid(), 10, 10, 10)
            
        return protein, ligand

    @property
    def random(self):
        return self._complexes[np.random.randint(len(self._complexes))]
    
class PafnucyData(Data):
    DATA_PATH=f'{ROOT_PATH}/pafnucy_data/complexes'

    def __init__(self):
        super(PafnucyData, self).__init__()
        if '.DS_Store' in self.complexes_path:
            self.complexes_path.remove('.DS_Store')
        if 'affinities.csv' in self.complexes_path:            
            self.complexes_path.remove('affinities.csv')

        self._complexes = [
            (
                OB_to_mol(
                    read_to_OB(filename=f'{self.DATA_PATH}/{complex}/{complex}_pocket.mol2', filetype="mol2"),
                    mol_type=-1,
                    path=f'{self.DATA_PATH}/{complex}/{complex}_pocket.mol2'
                ), OB_to_mol(
                    read_to_OB(filename=f'{self.DATA_PATH}/{complex}/{complex}_ligand.mol2', filetype="mol2"),
                    mol_type=1,
                    path=f'{self.DATA_PATH}/{complex}/{complex}_ligand.mol2'                    
                )
            ) for complex in self.complexes_path
        ]

class PDBQTData(Data):
    DATA_PATH=f'{ROOT_PATH}/pdbqt_data'

    def __init__(self):
        super(PDBQTData, self).__init__()
        self.complexes_path = [
            ('6Y2F_MOD.pdbqt', 'a-ketoamide-13b.pdbqt')
        ]

        self._complexes = [
            (
                OB_to_mol(
                    read_to_OB(filename=f'{self.DATA_PATH}/{complex}', filetype="pdbqt"),
                    mol_type=-1,
                    path=f'{self.DATA_PATH}/{complex}'                    
                ), OB_to_mol(
                    read_to_OB(filename=f'{self.DATA_PATH}/{ligand}', filetype="pdbqt"),
                    mol_type=1,
                    path=f'{self.DATA_PATH}/{ligand}'
                )
            ) for complex, ligand in self.complexes_path
        ]

class PDBQTData_2(Data):
    DATA_PATH=f'{ROOT_PATH}/pdbqt_data_2'

    def __init__(self):
        super(PDBQTData_2, self).__init__()
        if '.DS_Store' in self.complexes_path:
            self.complexes_path.remove('.DS_Store')
        

        self._complexes = [
            (
                OB_to_mol(
                    read_to_OB(filename=f'{self.DATA_PATH}/{complex}/{complex}_pocket.pdb', filetype="pdb"),
                    mol_type=-1,
                    path=f'{self.DATA_PATH}/{complex}/{complex}_pocket.pdb'
                ), OB_to_mol(
                    read_to_OB(filename=f'{self.DATA_PATH}/{complex}/{complex}_ligand.mol2', filetype="mol2"),
                    mol_type=1,
                    path=f'{self.DATA_PATH}/{complex}/{complex}_ligand.mol2'                    
                )
            ) for complex in self.complexes_path
        ]
        

class DudeProteaseData(Data):
    DATA_PATH=f'{ROOT_PATH}/dude_protease'

    def __init__(self):
        super(DudeProteaseData, self).__init__()
        if '.DS_Store' in self.complexes_path:
            self.complexes_path.remove('.DS_Store')


        self._complexes = [
            (
                OB_to_mol(
                    read_to_OB(filename=f'{self.DATA_PATH}/{complex}/receptor.pdb', filetype="pdb"),
                    mol_type=-1,
                    path=f'{self.DATA_PATH}/{complex}/receptor.pdb'
                ), OB_to_mol(
                    read_to_OB(filename=f'{self.DATA_PATH}/{complex}/crystal_ligand.mol2', filetype="mol2"),
                    mol_type=1,
                    path=f'{self.DATA_PATH}/{complex}/crystal_ligand.mol2'
                )
            ) for complex in self.complexes_path
        ]

    
    
class DataStore:
    DATA_STORES = []
    DATA = []

    @classmethod
    def init(cls, crop=True):
        cls.DATA_STORES = [PDBQTData_2(), PDBQTData(), PafnucyData()]#, DudeProteaseData()]
        cls.load(crop)
        cls.normalize()

    @classmethod
    def load(cls, crop=True):
        cls.DATA = [store.get_molecules(complex, crop) for store in cls.DATA_STORES for complex in store._complexes]            
                
    @classmethod
    def next(cls, crop=True):
        # datastore = cls.DATA_STORES[np.random.randint(0, len(cls.DATA_STORES))]
        # return datastore.get_molecules(datastore.random, crop)
        return cls.DATA[np.random.randint(0, len(cls.DATA))]

    @classmethod
    def normalize(cls):
        X = []
        for protein, ligand in cls.DATA:
            X.extend(list(protein.get_coords()))
            X.extend(list(ligand.get_coords()))
        X = np.asarray(X)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        for i, (protein, ligand) in enumerate(cls.DATA):
            protein.set_coords(X_scaled[i:i+protein.n_atoms,:])
            ligand.set_coords(X_scaled[i+protein.n_atoms:i+protein.n_atoms+ligand.n_atoms, :])
