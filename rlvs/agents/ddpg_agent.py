# refered from https://github.com/germain-hug/Deep-RL-Keras
# Source DDPG: https://raw.githubusercontent.com/ghliu/pytorch-ddpg/master/ddpg.py
# Source DDPG Training: https://raw.githubusercontent.com/ghliu/pytorch-ddpg/master/main.py

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from rlvs.network import ActorGNN, CriticGNN
from rlvs.constants import AgentConstants

from .metrices import Metric
from .memory import Memory
from .utils import soft_update, hard_update, to_tensor, to_numpy, batchify, USE_CUDA
from .noise import OrnsteinUhlenbeckActionNoise
import logging

criterion = nn.MSELoss()


class DDPGAgentGNN:
    ACTOR_LEARNING_RATE = AgentConstants.ACTOR_LEARNING_RATE
    CRITIQ_LEARNING_RATE = AgentConstants.CRITIQ_LEARNING_RATE
    TAU = AgentConstants.TAU

    GAMMA = AgentConstants.GAMMA

    BATCH_SIZE = AgentConstants.BATCH_SIZE
    BUFFER_SIZE = AgentConstants.BUFFER_SIZE
    EXPLORATION_EPISODES = AgentConstants.EXPLORATION_EPISODES

    def __init__(
            self, env, weights_path, complex_path=None,
            warmup=32, prate=0.00005, is_training=1
    ):
        self.metrices = Metric.get_instance()
        self.input_shape = env.input_shape
        self.edge_shape = env.edge_shape
        self.action_shape = env.action_space.n_outputs
        self.eps = 1.0
        self.decay_epsilon = 1/50000
        self.action_bounds = env.action_space.action_bounds
        self.memory = Memory(self.BUFFER_SIZE)
        self.env = env
        self.exploration_noise = OrnsteinUhlenbeckActionNoise(size=self.env.action_space.n_outputs)
        self.warm_up_steps = warmup
        self.is_training = is_training

        self._actor = ActorGNN(
            self.input_shape,
            self.edge_shape,
            self.action_shape,
            self.ACTOR_LEARNING_RATE,
            self.TAU
        )
        self._actor_target = ActorGNN(
            self.input_shape,
            self.edge_shape,
            self.action_shape,
            self.ACTOR_LEARNING_RATE,
            self.TAU
        )
        self._actor_optim = Adam(self._actor.parameters(), lr=prate)

        self._critiq = CriticGNN(
            self.input_shape,
            self.edge_shape,
            self.action_shape,
            self.CRITIQ_LEARNING_RATE,
            self.TAU
        )
        self._critiq_target = CriticGNN(
            self.input_shape,
            self.edge_shape,
            self.action_shape,
            self.CRITIQ_LEARNING_RATE,
            self.TAU
        )
        self._critiq_optim = Adam(self._critiq.parameters(), lr=prate)

        if USE_CUDA:
            self._actor.cuda()
            self._actor_target.cuda()
            self._critiq.cuda()
            self._critiq_target.cuda()

        hard_update(self._actor_target, self._actor)  # Make sure target is with the same weight
        hard_update(self._critiq_target, self._critiq)

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

    def update_network(self):

        batch = self.memory.sample(self.BATCH_SIZE)

        complex_batched = batchify(batch.states)
        next_complex_batched = batchify(batch.next_states)

        actions = batch.actions
        rewards = batch.rewards
        terminals = batch.terminals

        # Prepare for the target q batch
        next_q_values = self._critiq_target([
            next_complex_batched,
            self._actor_target(next_complex_batched),
        ])

        with torch.no_grad():
            target_q_batch = to_tensor(
                rewards
            ) + self.CRITIQ_LEARNING_RATE * to_tensor(
                terminals.astype(np.float)
            ) * next_q_values

            self._critiq.zero_grad()

        q_batch = self._critiq([complex_batched, to_tensor(actions)])

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self._critiq_optim.step()

        # Actor update
        self._actor.zero_grad()

        # critic policy update
        policy_loss = -self._critiq([
            complex_batched,
            self._actor(complex_batched)
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self._actor_optim.step()

        # Target update
        soft_update(self._actor_target, self._actor, self.TAU)
        soft_update(self._critiq_target, self._critiq, self.TAU)
        print("Value Loss: ", value_loss, " policy loss: ", policy_loss)
        return to_numpy(value_loss), to_numpy(policy_loss)

    def get_predicted_action(self, data, step=None, decay_epsilon=True):
        # Explore AdaptiveParamNoiseSpec, with normalized action space
        # https://github.com/l5shi/Multi-DDPG-with-parameter-noise/blob/master/Multi_DDPG_with_parameter_noise.ipynb
        action = to_numpy(
            self._actor(data)[0]
        )

        if step is not None:
            action += self.is_training * max(
                self.eps, 0
            ) * self.exploration_noise.generate(step)

        if decay_epsilon:
            self.eps -= self.decay_epsilon

        return action

    def random_action(self):
        return np.random.uniform(-1., 1., self.action_shape)

    def get_action(self, action):
        action *= self.action_bounds[1]  # Remove as the multiplier has been removed
        return np.clip(action, *self.action_bounds)

    def play(self, num_train_episodes):
        returns = []
        num_steps = 0
        max_episode_length = 100
        max_reward = 0
        i_episode = 0

        while i_episode < num_train_episodes:
            critic_losses = []
            actor_losses = []
            m_complex_t, state_t = self.env.reset()

            protein_name = self.env._complex.protein.path.split('/')[-1]
            self.metrices.init_rmsd(i_episode, protein_name, self.env._complex.rmsd)

            episode_return, episode_length, terminal = 0, 0, False

            self.exploration_noise = OrnsteinUhlenbeckActionNoise(
                size=self.env.action_space.n_outputs
            )

            while not (terminal or (episode_length == max_episode_length)):
                data = m_complex_t.data

                if num_steps <= self.warm_up_steps:
                    predicted_action = self.random_action()
                else:
                    predicted_action = self.get_predicted_action(
                        data, episode_length
                    )

                action = self.get_action(predicted_action)

                reward, terminal = self.env.step(action)
                d_store = False if episode_length == max_episode_length else terminal

                self.metrices.cache_rmsd(i_episode, protein_name, self.env._complex.rmsd)
                self.metrices.cache_divergence(i_episode, terminal)

                reward = 0 if episode_length == max_episode_length else reward
                self.memorize(data, [predicted_action], reward, m_complex_t.data, d_store)

                episode_return += reward
                episode_length += 1

                self.log(action, np.round(reward, 4), episode_length, i_episode)

                if num_steps > self.warm_up_steps:
                    critic_loss, actor_loss = self.update_network()
                    self.metrices.cache_loss(
                        i_episode, {
                            "critic": critic_loss,
                            "actor": actor_loss
                        })

                    critic_losses.append(critic_loss)
                    actor_losses.append(actor_loss)

                if (num_steps + 1) % 10 == 0 and i_episode > 100:
                    self.env.save_complex_files(f'{self.complex_path}_{i_episode}_{num_steps}')
                    Metric.save(self.metrices, self.complex_path)

                if m_complex_t.perfect_fit:
                    m_complex_t, state_t = self.env.reset()
                    protein_name = self.env._complex.protein.path.split('/')[-1]
                    self.metrices.init_rmsd(i_episode, protein_name, self.env._complex.rmsd)

                num_steps += 1

            returns.append([
                i_episode + 1,
                episode_length,
                episode_return,
                np.mean(critic_losses),
                np.mean(actor_losses)
            ])

            max_reward = max_reward if max_reward > episode_return else episode_return

            print(
                f"Episode: {i_episode + 1} \
                Return: {episode_return} \
                episode_length: {episode_length} \
                Max Reward; {max_reward} \
                Critic Loss: {np.mean(critic_losses)} \
                Actor loss: {np.mean(actor_losses)}"
            )

            logging.info(
                f"Episode: {i_episode + 1} \
                Return: {episode_return} \
                episode_length: {episode_length} \
                Max Reward; {max_reward} \
                Critic Loss: {np.mean(critic_losses)} \
                Actor loss: {np.mean(actor_losses)}"
            )

            if i_episode % 10 == 0:
                self.save_weights(self.weights_path)
                self.env.save_complex_files(f'{self.complex_path}_{i_episode}')

            i_episode += 1

        return returns

    def log(self, action, reward, episode_length, i_episode):
        print(
            "Action:", np.round(np.array(action), 4),
            "Reward:", np.round(reward, 4),
            "E_i:", episode_length,
            "RMSD: ", self.env._complex.rmsd,
            "E:", i_episode
        )
        logging.info(
            f"Action: {np.round(np.array(action), 4)},\
              Reward: {np.round(reward, 4)}, \
              E_i: {episode_length}, E: {i_episode}, \
              RMSD: {np.round(self.env._complex.rmsd, 4)}"
        )

    def save_weights(self, path):
        torch.save(self._actor.state_dict(), f'{path}_actor')
        torch.save(self._critiq.state_dict(), f'{path}_critic')

    def load_weights(self, path_actor, path_critic):
        self._critiq.load_state_dict(torch.load(path_critic))
        self._actor.load_state_dict(torch.load(path_actor))
