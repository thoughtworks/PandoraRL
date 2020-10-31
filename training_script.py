from rlvs.molecule_world.env import GraphEnv
from rlvs.agents.ddpg_agent import DDPGAgentGNN
env = GraphEnv()
print(env.input_shape)

run_id = 0
folder = "./model/"
path_prefix = f"{folder}run{run_id}_"
agent = DDPGAgentGNN(env, weights_path=path_prefix+"weights_intermediate", log_filename=path_prefix+"training_log.log")
actions =  agent.play(50000)
agent.save_weights(path_prefix+"weights_final")
