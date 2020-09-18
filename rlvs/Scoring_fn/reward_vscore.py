# script to run Autodock Vina and obtain binding affinity
# run command: python3 reward_vscore.py --receptor <receptor.pdbqt> --ligand <receptor.pdbqt> --vina_executable <path to ./vina>
        
import __future__

import textwrap
import math
import os
import sys
import glob
import pickle


################################## MODIFY THIS VARIABLE TO POINT TO THE AUTODOCK VINA EXECUTABLE ##################################
vina_executable = "/PATH/TO/VINA_1_1_2/vina"
###################################################################################################################################

class PDB:

    def __init__ (self):
        self.AllAtoms={}
        self.NonProteinAtoms = {}
        self.max_x = -9999.99
        self.min_x = 9999.99
        self.max_y = -9999.99
        self.min_y = 9999.99
        self.max_z = -9999.99
        self.min_z = 9999.99
        self.rotateable_bonds_count = 0
        self.protein_resnames = ["ALA", "ARG", "ASN", "ASP", "ASH", "ASX", "CYS", "CYM", "CYX", "GLN", "GLU", "GLH", "GLX", "GLY", "HIS", "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "LYN", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
        self.aromatic_rings = []
        self.charges = [] # a list of points
        self.OrigFileName = ""

def getCommandOutput2(command):
    child = os.popen(command)
    data = child.read()
    err = child.close()
    if err:
        raise RuntimeError('%s failed w/ exit code %d' % (command, err))
    return data

class vscore:

   def __init__(self, ligand_pdbqt_filename, receptor, parameters, line_header, actual_filename_if_ligand_is_list="", actual_filename_if_receptor_is_list=""): # must be a more elegant way of doing this

        receptor_pdbqt_filename = receptor.OrigFileName
        ligand_pdbqt_filename = ligand.OrigFileName
        
        # Get vina score and format
        vina_output = getCommandOutput2(parameters.params['vina_executable'] + ' --score_only --receptor ' + receptor_pdbqt_filename + ' --ligand ' + ligand_pdbqt_filename)
        vina_output = vina_output.split("\n")

        vina_affinity = 0.0
        vina_gauss_1 = 0.0
        vina_gauss_2 = 0.0
        vina_repulsion = 0.0
        vina_hydrophobic = 0.0
        vina_hydrogen = 0.0
        for item in vina_output:
            item = item.strip()
            if "Affinity" in item: vina_affinity = float(item.replace("Affinity: ","").replace(" (kcal/mol)",""))
            if "gauss 1" in item: vina_gauss_1 = float(item.replace("gauss 1     : ",""))
            if "gauss 2" in item: vina_gauss_2 = float(item.replace("gauss 2     : ",""))
            if "repulsion" in item: vina_repulsion = float(item.replace("repulsion   : ",""))
            if "hydrophobic" in item: vina_hydrophobic = float(item.replace("hydrophobic : ",""))
            if "Hydrogen" in item: vina_hydrogen = float(item.replace("Hydrogen    : ",""))

        # Only binding affinity returned
        self.vina_output = vina_affinity
        #self.vina_output = [vina_affinity, vina_gauss_1, vina_gauss_2, vina_repulsion, vina_hydrophobic, vina_hydrogen]
        
class command_line_parameters:

    params = {}

    def __init__(self, parameters):

        global vina_executable

        # first, set defaults
        self.params['receptor'] = ''
        self.params['ligand'] = ''
        self.params['vina_executable'] = vina_executable
        
        # now get user inputed values

        for index in range(len(parameters)):
            item = parameters[index]
            if item[:1] == '-': # so it's a parameter key value
                key = item.replace('-','').lower()

                value = parameters[index+1]
                if key in list(self.params.keys()):
                    self.params[key] = value
                    parameters[index] = ""
                    parameters[index + 1] = ""

        # make a list of all the command-line parameters not used
        error = ""
        for index in range(1,len(parameters)):
            item = parameters[index]
            if item != "": error = error + item + " "

        if error != "":
            print("WARNING: The following command-line parameters were not used:")
            print(("\t" + error + "\n"))

    def okay_to_proceed(self):
        if self.params['receptor'] != '' and self.params['ligand'] != '' and self.params['vina_executable'] != '':
            return True
        else: return False


cmd_params = command_line_parameters(sys.argv[:])

if cmd_params.okay_to_proceed() is False:
    print("ERROR: You need to specify the ligand and receptor PDBQT files, as\nwell as the full path the the AutoDock Vina 1.1.2 executable, using the\n-receptor, -ligand, and -vina_executable tags from the command line.\nThe -ligand tag can also specify an AutoDock Vina output file.\n")
    sys.exit(0)

lig = cmd_params.params['ligand']
rec = cmd_params.params['receptor']

# load the rec into an array so you only have to load it from the disk once

receptor = PDB()
receptor.OrigFileName = rec

# load the ligand into an array so you only have to load it from the disk once

ligand = PDB()
ligand.OrigFileName = lig

line_header = "\t"

score = vscore(ligand, receptor, cmd_params, line_header, lig, rec)
print("binding_affinity:", score.vina_output)
