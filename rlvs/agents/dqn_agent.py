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

from .memory import Memory
from .utils import soft_update, hard_update, to_tensor, to_numpy, batchify, FLOAT
from .noise import OrnsteinUhlenbeckActionNoise

criterion = nn.MSELoss()

import logging
            

class DQNAgentGNN:
    ACTOR_LEARNING_RATE = AgentConstants.ACTOR_LEARNING_RATE 
    TAU  = AgentConstants.TAU                 
                                               
    GAMMA = AgentConstants.GAMMA               
                                               
    BATCH_SIZE = AgentConstants.BATCH_SIZE          
    BUFFER_SIZE = AgentConstants.BUFFER_SIZE         
    EXPLORATION_EPISODES = AgentConstants.EXPLORATION_EPISODES
    LEARN_INTERVAL = 4


    def __init__(self, env, weights_path, complex_path=None, warmup=32, prate=0.00005, is_training=1):
        self.input_shape = env.input_shape
        self.edge_shape = env.edge_shape
        self.action_shape = env.action_space.degree_of_freedom
        self.eps = 1.0
        self.decay_epsilon = 0.996
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

        self._actor_optim = Adam(self._actor.parameters(), lr=prate)


        hard_update(self._actor_target, self._actor) # Make sure target is with the same weight

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

    def act(self, data):
        if random.random() > self.eps:
            print("Actual Action")
            self._actor.eval()
            with torch.no_grad():
                predicted_action = self._actor(data)

            self._actor.train()

            return np.argmax(to_numpy(predicted_action[0]))
        else:
            print("Random Action")
            return random.choice(np.arange(self.action_shape))

    def learn(self):
        batch = self.memory.sample(self.BATCH_SIZE)

        complex_batched = batchify(batch.states)
        next_complex_batched = batchify(batch.next_states)

        actions = torch.from_numpy(batch.actions)
        rewards = to_tensor(batch.rewards)
        terminals = batch.terminals

        self._actor.train()
        self._actor_target.eval()

        predicted_targets = self._actor(complex_batched).gather(1,actions)

        with torch.no_grad():
            labels_next = self._actor_target(next_complex_batched).detach().max(1)[0].unsqueeze(1)

        labels = torch.tensor(rewards + (self.GAMMA * labels_next*(1-terminals)), dtype=torch.float32)
        loss = criterion(predicted_targets,labels)
        self._actor_optim.zero_grad()
        loss.backward()
        self._actor_optim.step()

        soft_update(self._actor_target, self._actor, self.TAU)
        return loss

    def play(self, num_train_episodes):
        returns = []
        num_steps = 0
        max_episode_length = 100
        max_reward = 0
        i_episode = 0
        losses = []
        loss = 0

        while i_episode < num_train_episodes:
            m_complex_t, state_t = self.env.reset()
            episode_return, episode_length, terminal = 0, 0, False
            losses = []
            while not (terminal or (episode_length == max_episode_length)):
                data = m_complex_t.data
                action = self.act(data)
                molecule_action = self.env.action_space.get_action(action)
                reward, terminal = self.env.step(molecule_action)
                d_store = False if episode_length == max_episode_length else terminal
                reward = 0 if episode_length == max_episode_length else reward
        
                self.memorize(data, [action], reward, m_complex_t.data, d_store)

                if num_steps % self.LEARN_INTERVAL == 0:
                    if num_steps > self.warm_up_steps:
                        self.warm_up_steps = -1
                        num_steps = 0
                        loss = self.learn()
                        losses.append(to_numpy(loss))

                self.log(action, reward, episode_length, i_episode, loss)               
                if m_complex_t.perfect_fit:
                    m_complex_t, state_t = self.env.reset()
                num_steps += 1
                episode_return += reward
                episode_length += 1

            max_reward = max_reward if max_reward > episode_return else episode_return
            self.eps = max(self.eps * self.decay_epsilon, 0.01)

            print(
                f"Episode: {i_episode + 1} \
                Return: {episode_return} \
                episode_length: {episode_length} \
                Max Reward; {max_reward} \
                Actor loss: {np.mean(losses)}"
            )

            logging.info(
                f"Episode: {i_episode + 1} \
                Return: {episode_return} \
                episode_length: {episode_length} \
                Max Reward; {max_reward} \
                Actor loss: {np.mean(losses)}"
            )
            if i_episode%10 == 0:
                self.save_weights(self.weights_path)
                self.env.save_complex_files(f'{self.complex_path}_{i_episode}')

            i_episode += 1

            




    def log(self, action, reward, episode_length, i_episode, loss):
        print(
            "Action:", np.round(np.array(action), 4),
            "Reward:", np.round(reward, 10),
            "E_i:", episode_length,
            "RMSD: ", self.env._complex.rmsd,
            "E:", i_episode,
            "loss", loss
        )
        logging.info(f"Action: {np.round(np.array(action), 4)}, Reward: {np.round(reward, 4)}, E_i: {episode_length}, E: {i_episode}, RMSD: {np.round(self.env._complex.rmsd, 4)}, LOSS: {loss}")

    def save_weights(self, path):
        torch.save(self._actor.state_dict(), f'{path}_actor')
        

    def load_weights(self, path_actor, path_critic):
        self._actor.load_state_dict(torch.load(path_actor))


