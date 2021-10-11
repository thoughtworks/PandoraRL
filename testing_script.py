from rlvs.constants import ComplexConstants  # , AgentConstants
from rlvs.molecule_world.env import GraphEnv
from rlvs.agents.dqn_agent import DQNAgentGNN

import os
import logging
import numpy as np

run_id = os.getenv('RUNID', 0)
folder = "./model_test/"
path_prefix = f"{folder}run{run_id}_"
log_filename=path_prefix+"training_log.log"

logging.basicConfig(
            filename=log_filename,
            filemode='w',
            format='%(message)s',
            datefmt='%I:%M:%S %p',
            level=logging.DEBUG
        )

env = GraphEnv(single_step=np.array([1]))
agent = DQNAgentGNN(
    env,
    weights_path=path_prefix+"weights_intermediate",
    complex_path=path_prefix+"ligand_intermediate"
)

agent.load_weights(f'{agent.weights_path}_actor', f'{agent.weights_path}_critic')
actions = agent.test(10)
