# refered from https://github.com/sol0invictus/RL_Codes/blob/master/DDPG/v1/ddpg.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, Convolution2D, Conv2D, MaxPooling2D, Concatenate, Input
from tensorflow.keras.optimizers import Adam

from .memory import Memory
from .network import Actor, Critic

class DDPGAgent:
    MAX_EPSILON = 1
    MIN_EPSILON = 0.01
    ACTOR_LEARNING_RATE = 1e-3,
    CRITIQ_LEARNING_RATE = 1e-3,
    EPSILON_MIN_ITER = 5000
    DELAY_TRAINING = 300
    GAMMA = 0.95
    BATCH_SIZE = 32
    TAU = 0.08
    RANDOM_REWARD_STD = 1.0
    
    def __init__(self, env):
        '''
Create a DDPG agent
Parameters: 
action_bounds (array): array with min and max action values, with shape (2, number of output values)
        example: 
        np.array([
          [x_min, y_min, theta_min], 
          [x_max, y_max, theta_max]
        ])
        
        '''
        self.input_shape = env.input_shape
        self.multi_output_count = env.action_space.n_outputs
        self.eps = 0.9
        self.action_bounds = env.action_space.action_bounds
        self.memory = Memory(500000)
        self.env = env

        self._actor = Actor(self.input_shape, self.multi_output_count, 1e-3)
        self._critiq = Critic(self.input_shape, self.multi_output_count, 1e-3)

    def get_action(self, state, noise_scale):
        # Explore AdaptiveParamNoiseSpec, with normalized action space
        # https://github.com/l5shi/Multi-DDPG-with-parameter-noise/blob/master/Multi_DDPG_with_parameter_noise.ipynb
        action = self.action_bounds[1] * self._actor.predict(state).flatten()
        action += noise_scale * np.random.randn(self.multi_output_count)
        import random

        return [random.uniform(*bounds) for bounds in self.action_bounds.transpose()]
        # return np.clip(action, *self.action_bounds)
    
    def play(self, num_train_episodes):
        returns = []
        q_losses = []
        mu_losses = []
        num_steps = 0
        action_noise=0.9
        max_episode_length=50
        decay = 0.9
        r = False

        for i_episode in range(num_train_episodes):
            state_t, episode_return, episode_length, terminal = self.env.reset(), 0, 0, False
            print("Actiual states: ", self.env.block.shift_x, self.env.block.shift_y, self.env.block.rotate_angle)
            while not (terminal or (episode_length == max_episode_length)):
                # For the first `start_steps` steps, use randomly sampled actions
                # in order to encourage exploration.
                
                if np.random.rand() < decay:
                  action = self.get_action(state_t, action_noise)
                  r = True
                else:
                  action = self.env.action_space.sample()
                  r = False
                
                # Keep track of the number of steps done
                num_steps += 1                
                # Step the env
                state_t_1, reward, terminal = self.env.step(action)
                episode_return += reward
                episode_length += 1
                
                # Ignore the "done" signal if it comes from hitting the time
                # horizon (that is, when it's an artificial terminal signal
                # that isn't based on the agent's state)
                d_store = False if episode_length == max_episode_length else terminal
                
                # Store experience to replay buffer
                self.memory.add_sample([state_t[0], action, reward, state_t_1[0], d_store])
                
                # Assign next state to be the current state on the next round
                state_t = state_t_1
                returns.append([self.env.block.sandbox, self.env.block.original_sandbox, reward])

                print(r, action, reward, episode_length)
                
            self.update_network(episode_length, mu_losses, q_losses)
            decay *= decay
            action_noise *= 0.7
            print("Episode:", i_episode + 1, "Return:", episode_return, 'episode_length:', episode_length)

        return (returns,q_losses,mu_losses)


    def update_network(self, episode_length, mu_losses, q_losses):
        gamma=0.99
        decay=0.99
        episode_length = 1
        for _ in range(episode_length):
            batch = self.memory.sample(self.BATCH_SIZE)
        
            states = np.array([val[0] for val in batch])
            actions = np.array([val[1] for val in batch])
            rewards = np.array([val[2] for val in batch])
            next_states = np.array([val[3] for val in batch ])
            terminals = np.array([val[4] for val in batch ])

            state_tensor=tf.convert_to_tensor(states)

            target_actions = self._actor.predict_target(next_states)
            target_q = self._critiq.predict_target(next_states, target_actions)
            
            train_q = (rewards + (1 - terminals) * gamma * target_q.reshape((32,))).reshape((32,1))
            self._critiq.fit(states, actions, train_q)
            predicted_actions = self._actor.tensors(states)

            action_grads, state_grads = self._critiq.action_gradients(tf.convert_to_tensor(states), predicted_actions)
            self._actor.fit(states, action_grads)

            self._actor.update_target_network()
            self._critiq.update_target_network()
        
