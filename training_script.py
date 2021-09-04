from rlvs.constants import ComplexConstants  # , AgentConstants
from rlvs.molecule_world.env import GraphEnv
from rlvs.agents.ddpg_agent import DDPGAgentGNN

import os

ComplexConstants.DISTANCE_THRESHOLD = 50

env = GraphEnv()
run_id = os.getenv('RUNID', 0)
folder = "./model/"
path_prefix = f"{folder}run{run_id}_"
agent = DDPGAgentGNN(
    env,
    weights_path=path_prefix+"weights_intermediate",
    complex_path=path_prefix+"ligand_intermediate",
    log_filename=path_prefix+"training_log.log"
)
actions = agent.play(50000)
agent.save_weights(path_prefix+"weights_final")
