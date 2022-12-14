from rlvs.molecule_world.env import GraphEnv
from rlvs.agents.dqn_agent import DQNAgentGNN
from rlvs.agents.ddpg_agent import DDPGAgentGNN
from rlvs.agents.metrices import Metric
from rlvs.config import Config

import os
import logging
import numpy as np


class TrainingAgents:
    @staticmethod
    def get_agent(algorithm: str):
        agents = {
            'dqn': DQNAgentGNN,
            'ddpg': DDPGAgentGNN
        }

        return agents.get(algorithm)


run_id = os.getenv('RUNID', 0)
folder = os.getenv('OUTPUT', "./model/")

config_path = os.getenv('CONFIG', './config.json')
Config.init(config_path)
config = Config.get_instance()
metric = Metric.get_instance()

path_prefix = f"{folder}/run{run_id}_"
log_filename = path_prefix+"training_log.log"

TRAINING_AGENT = TrainingAgents.get_agent(config.training_algorithm)

logging.basicConfig(
            filename=log_filename,
            filemode='w',
            format='%(message)s',
            datefmt='%I:%M:%S %p',
            level=logging.DEBUG
        )

env = GraphEnv(single_step=np.array(config.single_step))
agent = TRAINING_AGENT(
    env,
    weights_path=path_prefix+"weights_intermediate",
    complex_path=path_prefix+"ligand_intermediate"
)
actions = agent.play(1500)
agent.save_weights(path_prefix+"weights", "final")

metric.plot_loss_trend(path_prefix)
Metric.save(metric, f'{path_prefix}_final')
