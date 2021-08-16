from openbabel import pybel
from openbabel import openbabel as ob
import os
from os import path

ROOT_PATH = f'{path.dirname(path.abspath(__file__))}/data'
def convert_obmol(obmol, dest_path, in_filetype, out_filetype='pdb'):
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats(in_filetype, out_filetype)
    pdb_content = obConversion.WriteString(obmol)
    with open(dest_path, 'w') as pdb_file:
        print("writing pdb")
        pdb_file.write(pdb_content)


def read_to_OB(filename, filetype):
    obconversion = ob.OBConversion()
    obconversion.SetInFormat(filetype)
    obmol = ob.OBMol()

    notatend = obconversion.ReadFile(obmol, filename)
    return obmol

class Data:
    DATA_PATH = None
    def __init__(self):
        self.file_paths = []
        self.complexes_path = os.listdir(self.DATA_PATH)


class PafnucyData(Data):
    DATA_PATH=f'{ROOT_PATH}/pafnucy_data/complexes'

    def __init__(self):
        super(PafnucyData, self).__init__()
        if '.DS_Store' in self.complexes_path:
            self.complexes_path.remove('.DS_Store')
        if 'affinities.csv' in self.complexes_path:            
            self.complexes_path.remove('affinities.csv')
        print(self.complexes_path)
        self.file_paths = [
            {
                "obmol": read_to_OB(f'{self.DATA_PATH}/{complex}/{complex}_pocket.mol2', 'mol2'),
                "dest_path": f'{self.DATA_PATH}/{complex}/{complex}_pocket.pdb',
                "in_filetype": 'mol2'
             } for complex in self.complexes_path
        ]


class SARSVarients(Data):
    DATA_PATH=f'{ROOT_PATH}/SARS_variants'

    def __init__(self):
        super(SARSVarients, self).__init__()
        if '.DS_Store' in self.complexes_path:
            self.complexes_path.remove('.DS_Store')
        print(self.complexes_path)
        self.file_paths = [
            {
                "obmol": read_to_OB(f'{self.DATA_PATH}/{complex}/{complex}_protein.pdbqt', 'pdbqt'),
                "dest_path": f'{self.DATA_PATH}/{complex}/{complex}_protein.pdb',
                "in_filetype": 'pdbqt'
             } for complex in self.complexes_path
        ]


class PDBQTData(Data):
    DATA_PATH=f'{ROOT_PATH}/pdbqt_data'

    def __init__(self):
        super(PDBQTData, self).__init__()
        self.complexes_path = [
            '6Y2F_MOD.pdbqt'
        ]

        self.file_paths = [
            {
                "obmol": read_to_OB(f'{self.DATA_PATH}/6Y2F_MOD.pdbqt', 'pdbqt'),
                "dest_path": f'{self.DATA_PATH}/6Y2F_MOD.pdb',
                "in_filetype": 'pdbqt'
             } 
        ]


def migrate():
    files = SARSVarients().file_paths
    for args in files:
        convert_obmol(**args)
        

if __name__ == '__main__':
    migrate()
