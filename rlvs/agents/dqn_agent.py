# refered from https://github.com/germain-hug/Deep-RL-Keras
# Source DDPG: https://raw.githubusercontent.com/ghliu/pytorch-ddpg/master/ddpg.py
# Source DDPG Training: https://raw.githubusercontent.com/ghliu/pytorch-ddpg/master/main.py

import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import Adam

from rlvs.network import ActorDQN
from rlvs.constants import AgentConstants
from rlvs.config import Config

from .memory import Memory
from .metrices import Metric
from .utils import soft_update, hard_update, to_tensor, to_numpy, batchify, \
    FLOAT, LONG, INT32, USE_CUDA, use_device
import logging

criterion = nn.SmoothL1Loss()


class DQNAgentGNN:
    ACTOR_LEARNING_RATE = AgentConstants.ACTOR_LEARNING_RATE
    TAU = AgentConstants.TAU

    GAMMA = AgentConstants.GAMMA

    BATCH_SIZE = AgentConstants.BATCH_SIZE
    BUFFER_SIZE = AgentConstants.BUFFER_SIZE
    EXPLORATION_EPISODES = AgentConstants.EXPLORATION_EPISODES
    LEARN_INTERVAL = 4

    def __init__(
            self, env, weights_path,
            complex_path=None, warmup=32, prate=0.00005, is_training=1
    ):
        self.metrices = Metric.get_instance()
        self.input_shape = env.input_shape
        self.edge_shape = env.edge_shape
        self.action_shape = env.action_space.degree_of_freedom
        self.eps = 0.99
        self.decay = 0.99
        self.memory = Memory(self.BUFFER_SIZE)
        self.env = env
        self.warm_up_steps = warmup
        self.is_training = is_training

        self._actor = ActorDQN(
            self.input_shape,
            self.edge_shape,
            self.action_shape,
            self.ACTOR_LEARNING_RATE,
            self.TAU
        )

        self._actor_target = ActorDQN(
            self.input_shape,
            self.edge_shape,
            self.action_shape,
            self.ACTOR_LEARNING_RATE,
            self.TAU
        )

        if USE_CUDA:
            self._actor.cuda()
            self._actor_target.cuda()

        self._actor_optim = Adam(self._actor.parameters(), lr=prate)

        hard_update(self._actor_target, self._actor)  # Make sure target is with the same weight

        self.weights_path = weights_path
        self.complex_path = complex_path

    def memorize(self, state, action, reward, next_state, done):
        self.memory.add_sample({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })

    def act(self, data, test=False):
        if random.random() > self.eps or test:
            with torch.no_grad():
                predicted_action = self._actor(data)

            print("PREDICTED", predicted_action)

            return np.argmax(to_numpy(predicted_action[0]))
        else:
            return random.choice(np.arange(self.action_shape))

    def learn(self, sync=False):
        losses = []
        batches = self.memory.sample(self.BATCH_SIZE)
        for batch in batches:  # TODO: implement batching for graphs
            complex_batched = batchify(batch.states)
            next_complex_batched = batchify(batch.next_states)

            actions = to_tensor(batch.actions, dtype=LONG)
            rewards = to_tensor(batch.rewards)
            terminals = to_tensor(batch.terminals, dtype=INT32)
            self._actor_optim.zero_grad()
            complex_batched = use_device(complex_batched)
            next_complex_batched = use_device(next_complex_batched)
            predicted_targets = self._actor(complex_batched).gather(1, actions)
            labels_next = self._actor_target(next_complex_batched).detach().max(1)[0].unsqueeze(1)
            complex_batched.cpu()
            next_complex_batched.cpu()
            labels = (rewards + (self.GAMMA * labels_next*(1-terminals))).type(FLOAT)

            loss = criterion(predicted_targets, labels)
            loss.backward()
            self._actor_optim.step()
            losses.append(to_numpy(loss.data))

        if sync:
            soft_update(self._actor_target, self._actor, self.TAU)

        return np.array(losses).mean()

    def play(self, num_train_episodes):
        num_steps = 0
        max_episode_length = 100
        max_reward = 0
        i_episode = 0
        losses = []
        loss = 0
        sync_counter = 0
        config = Config.get_instance()

        while i_episode < num_train_episodes:
            m_complex_t, state_t = self.env.reset()
            molecule_index = 0
            protein_name = self.env._complex.protein.path.split('/')[-1]
            self.metrices.init_rmsd(
                i_episode, protein_name,
                self.env._complex.rmsd,
                self.env._complex.ligand.get_centroid()
            )

            episode_return, episode_length, terminal = 0, 0, False
            episode_loss = []
            while not (terminal or (episode_length == max_episode_length)):
                data = m_complex_t.data
                data = use_device(data)
                action = self.act(data)
                data.cpu()
                molecule_action = self.env.action_space.get_action(action)
                reward, terminal = self.env.step(molecule_action)

                self.metrices.cache_rmsd(i_episode, self.env._complex.rmsd, molecule_index)
                self.metrices.cache_action_reward(
                    i_episode, action, reward, molecule_index
                )
                self.metrices.cache_divergence(i_episode, terminal, molecule_index)

                d_store = False if episode_length == max_episode_length else terminal
                reward = 0 if episode_length == max_episode_length else reward
                self.memorize(data, [action], reward, m_complex_t.data, d_store)

                if (num_steps := num_steps % self.LEARN_INTERVAL) == 0:
                    if self.memory.has_samples(self.BATCH_SIZE):
                        sync_counter = (sync_counter + 1) % 5
                        loss = self.learn(sync_counter == 0)

                        self.metrices.cache_loss(i_episode, loss)

                        losses.append(loss)
                        episode_loss.append(loss)

                self.log(action, reward, episode_length, i_episode, loss)

                if m_complex_t.perfect_fit:
                    molecule_index += 1
                    m_complex_t, state_t = self.env.reset()
                    protein_name = self.env._complex.protein.path.split('/')[-1]

                    self.metrices.init_rmsd(
                        i_episode,
                        protein_name,
                        self.env._complex.rmsd,
                        self.env._complex.ligand.get_centroid()
                    )

                num_steps += 1
                episode_return += reward
                episode_length += 1

            self.eps = max(self.eps * self.decay, 0.01)
            max_reward = max_reward if max_reward > episode_return else episode_return

            print(
                f"Episode: {i_episode + 1} \
                Return: {episode_return} \
                episode_length: {episode_length} \
                Max Reward; {max_reward} \
                Actor loss: {np.mean(episode_loss)}"
            )

            logging.info(
                f"Episode: {i_episode + 1} \
                Return: {episode_return} \
                episode_length: {episode_length} \
                Max Reward; {max_reward} \
                Actor loss: {np.mean(episode_loss)}"
            )

            if i_episode >= config.test_from_episode and \
               not self.metrices.has_diverged(i_episode):
                self.save_weights(self.weights_path, i_episode)
                self.env.save_complex_files(f'{self.complex_path}_{i_episode}')
                self.metrices.plot_rmsd_trend(i_episode, self.weights_path)

                if config.run_tests:
                    self.test(10, train_episode=i_episode, save=False)
                    self.metrices.plot_rmsd_trend(i_episode, self.weights_path, test=True)

                Metric.save(self.metrices, self.weights_path)

            i_episode += 1

    def test(self, number_of_tests, train_episode=0, save=True):
        i = 0
        max_episode_length = 200
        while i < number_of_tests:
            m_complex_t, state_t = self.env.reset(test=True)
            protein_name = self.env._complex.protein.path.split('/')[-1]
            self.metrices.init_rmsd(
                train_episode, protein_name,
                self.env._complex.rmsd,
                self.env._complex.ligand.get_centroid(),
                test=True
            )

            episode_return, episode_length, terminal = 0, 0, False

            while not (terminal or (episode_length == max_episode_length)):
                data = m_complex_t.data
                data = data.cuda() if USE_CUDA else data
                action = self.act(data, test=True)
                data.cpu()
                molecule_action = self.env.action_space.get_action(action)
                reward, terminal = self.env.step(molecule_action)
                self.metrices.cache_rmsd(
                    train_episode, self.env._complex.rmsd, i, test=True
                )

                self.metrices.cache_action_reward(
                    train_episode, action, reward, i, test=True
                )

                self.metrices.cache_divergence(train_episode, terminal, i, test=True)

                self.log(action, reward, episode_length, i, 0, test=True)
                episode_return += reward
                episode_length += 1
                if save:
                    self.env.save_complex_files(f'{self.complex_path}_{i}_test_{episode_length}')
            i += 1

    def log(self, action, reward, episode_length, i_episode, loss, test=False):
        test_log = "Test" if test else ""
        print(
            test_log,
            "Action:", np.round(np.array(action), 4),
            "Reward:", np.round(reward, 10),
            "E_i:", episode_length,
            "RMSD: ", self.env._complex.rmsd,
            "E:", i_episode,
            "loss", loss
        )
        logging.info(
            f"{test_log} Action: {np.round(np.array(action), 4)}, \
            Reward: {np.round(reward, 4)}, \
            E_i: {episode_length}, E: {i_episode},\
            RMSD: {np.round(self.env._complex.rmsd, 4)}, LOSS: {loss}"
        )

    def save_weights(self, path, episode):
        torch.save(self._actor.state_dict(), f'{path}_{episode}_actor')

    def load_weights(self, path_actor, path_critic):
        self._actor.load_state_dict(torch.load(path_actor))
