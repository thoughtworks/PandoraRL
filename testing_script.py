from rlvs.molecule_world.env import TestGraphEnv
from rlvs.agents.ddpg_agent import DDPGAgentGNN
import joblib
import os

from src.main.Path import Path

protein_input = Path.PROTEIN_FILE_PATH
protein_filetype = os.path.splitext(protein_input)[1][1:]

# ligand file input
ligand_input = Path.LIGAND_FILE_PATH
ligand_filetype = os.path.splitext(ligand_input)[1][1:]
# ligand string input
ligand_input = ""
ligand_filetype = "smiles_string"

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
env = TestGraphEnv(
    scaler=scaler, 
    protein_path=protein_input, 
    ligand_path=ligand_input, 
    protein_filetype=protein_filetype, 
    ligand_filetype=ligand_filetype,
    prepare=False
)
agent = DDPGAgentGNN(env, weights_path = "", log_filename=Path.TESTING_LOG_FILE_PATH)
agent.test(max_steps=max_steps, path_actor_weights=actor_weights, path_critic_weights=critic_weights)

#convert complex to pdbqt
env.save_complex_files(path=f"./Results/a-ketoamide_output_{max_steps}.pdbqt",filetype="pdbqt")