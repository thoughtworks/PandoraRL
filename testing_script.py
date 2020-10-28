from rlvs.molecule_world.env import TestGraphEnv
from rlvs.agents.ddpg_agent import DDPGAgentGNN
import joblib

protein_input = "./data/pafnucy_data/complexes/4mrz/4mrz_pocket.mol2" #from user
ligand_input = "./data/pafnucy_data/complexes/4mrz/4mrz_ligand.mol2" #from user

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

env = TestGraphEnv(scaler=scaler, protein_path=protein_input, ligand_path=ligand_input, protein_filetype="mol2", ligand_filetype="mol2")
agent = DDPGAgentGNN(env, weights_path = "", log_filename="./testing_logfile.log")
agent.test(max_steps=0, path_actor_weights=actor_weights, path_critic_weights=critic_weights)

#convert complex to pdbqt
env.save_complex_files(path="./Results/ligand_output.pdbqt",filetype="pdbqt")
