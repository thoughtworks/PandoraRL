from rlvs.constants import ComplexConstants  # , AgentConstants
from rlvs.molecule_world.env import GraphEnv
from rlvs.agents.ddpg_agent import DDPGAgentGNN

import os
import logging

ComplexConstants.DISTANCE_THRESHOLD = 10

run_id = os.getenv('RUNID', 0)
folder = "./model/"
path_prefix = f"{folder}run{run_id}_"
log_filename=path_prefix+"training_log.log"

logging.basicConfig(
            filename=log_filename,
            filemode='w',
            format='%(message)s',
            datefmt='%I:%M:%S %p',
            level=logging.DEBUG
        )

env = GraphEnv()
agent = DDPGAgentGNN(
    env,
    weights_path=path_prefix+"weights_intermediate",
    complex_path=path_prefix+"ligand_intermediate"
)
actions = agent.play(50000)
agent.save_weights(path_prefix+"weights_final")
