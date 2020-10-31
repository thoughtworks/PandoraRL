import ast
import sys

import joblib
import os

from src.main.Path import Path

param = ast.literal_eval(sys.argv[1])

protein_input = param["protein_file_path"]
protein_filetype = os.path.splitext(protein_input)[1][1:]

ligand_filetype = param['ligand_input_type']


if ligand_filetype != "smiles_string":
    ligand_input = param['ligand_file_path']
    ligand_filetype = os.path.splitext(ligand_input)[1][1:]
else:
    ligand_input = param['ligand_input']

# specify model path
actor_weights = "./Results/run5_actor.h5"
critic_weights = "./Results/run5_critic.h5"
scaler_filename = "./Results/run5_scaler.save"

print("**** Setting up RL Agent")

from rlvs.molecule_world.env import TestGraphEnv
from rlvs.agents.ddpg_agent import DDPGAgentGNN

###
from rlvs.molecule_world.datasets import DataStore
DataStore.init(crop=False)
joblib.dump(DataStore.scaler, scaler_filename) 
###

scaler = joblib.load(scaler_filename)
print(ligand_filetype, ligand_input, protein_filetype, protein_input)
env = TestGraphEnv(
    scaler=scaler,
    protein_path=protein_input,
    ligand_path=ligand_input,
    protein_filetype=protein_filetype,
    ligand_filetype=ligand_filetype,
)
agent = DDPGAgentGNN(env, weights_path="", log_filename=param["log_path"])
print("*****Running RL Agent")
agent.test(max_steps=Path.MAX_STEPS, path_actor_weights=actor_weights, path_critic_weights=critic_weights)

# convert complex to pdbqt
print("***** Saving output file")
env.save_complex_files(path=param["output_path"], filetype="pdb")
