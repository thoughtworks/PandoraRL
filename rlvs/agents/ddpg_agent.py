# refered from https://github.com/germain-hug/Deep-RL-Keras
# Source DDPG: https://raw.githubusercontent.com/ghliu/pytorch-ddpg/master/ddpg.py
# Source DDPG Training: https://raw.githubusercontent.com/ghliu/pytorch-ddpg/master/main.py

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from rlvs.network import ActorGNN, CriticGNN

from .memory import Memory
from .utils import soft_update, hard_update, to_tensor, to_numpy
from .noise import OrnsteinUhlenbeckActionNoise

criterion = nn.MSELoss()

import logging
            

class DDPGAgent:
    ACTOR_LEARNING_RATE  = 0.00005
    CRITIQ_LEARNING_RATE = 0.00005
    TAU                  = 0.001

    BATCH_SIZE           = 32
    BUFFER_SIZE          = 20000

    def __init__(self, env, warmup=32, prate=0.00005, is_training=1):
        self.input_shape = env.input_shape
        self.action_shape = env.action_space.n_outputs
        self.eps = 0.9
        self.action_bounds = env.action_space.action_bounds
        self.memory = Memory(self.BUFFER_SIZE)
        self.env = env
        self.exploration_noise = OrnsteinUhlenbeckActionNoise(size=self.env.action_space.n_outputs)
        self.warm_up_steps = warmup
        self.is_training = is_training

        self._actor = ActorGNN(
            self.input_shape,
            self.action_shape,
            self.ACTOR_LEARNING_RATE,
            self.TAU
        )
        self._actor_target = ActorGNN(
            self.input_shape,
            self.action_shape,
            self.ACTOR_LEARNING_RATE,
            self.TAU
        )
        self._actor_optim = dam(self._actor.parameters(), lr=prate)


        self._critiq = CriticGNN(
            self.input_shape,
            self.action_shape,
            self.CRITIQ_LEARNING_RATE,
            self.TAU
        )        
        self._critiq_target = CriticGNN(
            self.input_shape,
            self.action_shape,
            self.CRITIQ_LEARNING_RATE,
            self.TAU
        )        
        self._critiq_optim = dam(self._critiq.parameters(), lr=prate)

        hard_update(self._actor_target, self._actor) # Make sure target is with the same weight
        hard_update(self._critiq_target, self._critiq)        

    def memorize(self, state, action, reward, next_state, done):
        self.memory.add_sample({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    
    def update_network(self):
        # 
        # state_batch, action_batch, reward_batch, \
        # next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)
        
        batch = self.memory.sample(self.BATCH_SIZE)
        batch_len = len(batch)
        
        states = np.array([val['state'] for val in batch])
        actions = np.array([val['action'] for val in batch])
        rewards = np.array([val['reward'] for val in batch])
        next_states = np.array([val['next_state'] for val in batch ])
        terminals = np.array([val['done'] for val in batch ])
        # Prepare for the target q batch
        next_q_values = self._critiq_target([
            to_tensor(next_state, volatile=True),
            self._actor_target(to_tensor(next_state, volatile=True)),
        ])
        
        next_q_values.volatile=False

        target_q_batch = to_tensor(rewards) + \
            self.CRITIQ_LEARNING_RATE*to_tensor(terminals.astype(np.float))*next_q_values

        # Critic update
        self._critiq.zero_grad()

        q_batch = self._critiq([ to_tensor(states), to_tensor(actions) ])
        
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self._critiq_optim.step()

        # Actor update
        self._actor.zero_grad()

        policy_loss = -self._critiq([
            to_tensor(states),
            self._actor(to_tensor(states))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self._actor_optim.step()

        # Target update
        soft_update(self._actor_target, self.actor, self.tau)
        soft_update(self._critiq_target, self._critiq, self.tau)
        return value_loss, policy_loss

    def get_predicted_action(self, state, step=None, decay_epsilon=True):
        # Explore AdaptiveParamNoiseSpec, with normalized action space
        # https://github.com/l5shi/Multi-DDPG-with-parameter-noise/blob/master/Multi_DDPG_with_parameter_noise.ipynb
        action = to_numpy(
            self._actor(to_tensor(state))
        ).squeeze(0)
        print("ACTION", action)
        
        if step is not None:
            action += self.is_training*max(self.eps, 0)*self.exploration_noise.generate(step)

        if decay_epsilon:
            self.eps *= self.eps


        return action

    def random_action(self):
        return np.random.uniform(-1.,1.,self.action_shape)

    def get_action(self, action):
        action *= self.action_bounds[1]
        
        x, y, t = np.clip(action, *self.action_bounds)
        return np.array([np.round(x), np.round(y), t]) #TODO: Revalidate this area
    
        
    def play(self, num_train_episodes):
        returns      = []
        num_steps    = 0
        max_episode_length = 500
        max_reward = 0
        i_episode = 0
        
        while i_episode < num_train_episodes:
            critic_losses = []
            actor_losses = []
            state_t, episode_return, episode_length, terminal = self.env.reset(), 0, 0, False

            self.exploration_noise = OrnsteinUhlenbeckActionNoise(size=self.env.action_space.n_outputs)

            while not (terminal or (episode_length == max_episode_length)):

                if i_episode <= self.warm_up_steps:
                    predicted_action = agent.random_action()
                else:
                    predicted_action = self.get_predicted_action(state_t, episode_length)

                action = self.get_action(predicted_action)
                state_t_1, reward, terminal = self.env.step(action)
                
                d_store = False if episode_length == max_episode_length else terminal
                reward = 0 if episode_length == max_episode_length else reward
                
                self.memorize(state_t[0], predicted_action, reward, state_t_1[0], d_store)

                num_steps += 1                
                
                episode_return += reward
                episode_length += 1
                state_t = state_t_1

                self.log(action, np.round(reward, 4), episode_length)

                if i_episode > self.warm_up_steps:
                    critic_loss, actor_loss = self.update_network()
                    critic_losses.append(critic_loss)
                    actor_losses.append(actor_loss)
                
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

            i_episode += 1

        return returns

    def log(self, action, reward, episode_length):
        print("Action:", action, "Reward:", np.round(reward, 4), "E_i:", episode_length,
              "Block state:", [
                  self.env.block.block_x, self.env.block.block_y, np.round(self.env.block.rotate_angle, 2), self.env.block.shift_x, self.env.block.shift_y
              ], "Dist:",  np.round(self.env.block.distance(), 4))
