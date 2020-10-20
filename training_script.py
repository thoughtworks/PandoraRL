from rlvs.molecule_world.env import GraphEnv
from rlvs.agents.ddpg_agent import DDPGAgentGNN
env = GraphEnv()
print(env.input_shape)

agent = DDPGAgentGNN(env, log_filename="./training_logfile.log")
actions =  agent.play(1000)
agent.save_weights("./model_weights")