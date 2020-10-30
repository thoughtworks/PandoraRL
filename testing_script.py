import ast
import sys

from rlvs.molecule_world.env import TestGraphEnv
from rlvs.agents.ddpg_agent import DDPGAgentGNN
import joblib

from src.main.Path import Path
param = ast.literal_eval(sys.argv[1])
protein_input = param["protein_file_path"]
ligand_input = param["ligand_file_path"]

# specify model path
actor_weights = "./Results/run5_actor.h5"
critic_weights = "./Results/run5_critic.h5"
scaler_filename = "./Results/run5_scaler.save"

###
from rlvs.molecule_world.datasets import DataStore
DataStore.init(crop=False)
joblib.dump(DataStore.scaler, scaler_filename) 
###

scaler = joblib.load(scaler_filename) 
max_steps = 10
env = TestGraphEnv(scaler=scaler, protein_path=protein_input, ligand_path=ligand_input, protein_filetype="pdbqt", ligand_filetype="pdbqt")
agent = DDPGAgentGNN(env, weights_path="", log_filename=param["log_path"])
agent.test(max_steps=max_steps, path_actor_weights=actor_weights, path_critic_weights=critic_weights)

#convert complex to pdbqt
env.save_complex_files(path=param["output_path"],filetype="pdbqt")
