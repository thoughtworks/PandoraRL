# refered from https://github.com/germain-hug/Deep-RL-Keras

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, Convolution2D, Conv2D, MaxPooling2D, Concatenate, Input
from tensorflow.keras.optimizers import Adam

from .memory import Memory
from .network import Actor, Critic
from .noise import OrnsteinUhlenbeckActionNoise

class DDPGAgent:
    ACTOR_LEARNING_RATE  = 0.00005
    CRITIQ_LEARNING_RATE = 0.00005
    TAU                  = 0.001

    DELAY_TRAINING       = 300
    GAMMA                = 0.99

    BATCH_SIZE           = 32
    BUFFER_SIZE          = 20000

    RANDOM_REWARD_STD    = 1.0
    NOISE_SCALE          = 0.001
    
    
    def __init__(self, env):
        self.input_shape = env.input_shape
        self.action_shape = env.action_space.n_outputs
        self.eps = 0.9
        self.action_bounds = env.action_space.action_bounds
        self.memory = Memory(self.BUFFER_SIZE)
        self.env = env
        self.exploration_noise = OrnsteinUhlenbeckActionNoise(size=self.env.action_space.n_outputs)
        self.noise_scaling = self.NOISE_SCALE * (self.action_bounds[1] - self.action_bounds[0])

        self._actor = Actor(
            self.input_shape,
            self.action_shape,
            self.ACTOR_LEARNING_RATE,
            self.TAU
        )

        self._critiq = Critic(
            self.input_shape,
            self.action_shape,
            self.CRITIQ_LEARNING_RATE,
            self.TAU
        )

    def get_predicted_action(self, state, step=None):
        # Try Brownian Motion as a candidate for stochastic noise, currently using Ornstein–Uhlenbeck process
        # Explore AdaptiveParamNoiseSpec, with normalized action space
        # https://github.com/l5shi/Multi-DDPG-with-parameter-noise/blob/master/Multi_DDPG_with_parameter_noise.ipynb
        action = self._actor.predict(state).flatten()

        if step is not None:
            action += self.exploration_noise.generate(step)

        return action
        
    def get_action(self, action):
        action *= self.action_bounds[1]
        
        x, y, t = np.clip(action, *self.action_bounds)
        return np.array([np.round(x), np.round(y), t])
    
    def play(self, num_train_episodes):
        returns      = []
        q_losses     = []
        mu_losses    = []
        num_steps    = 0
        action_noise = 0.99
        max_episode_length = 500
        decay        = 0.99
        noise_ep     = 10
        mean         = 0
        stdev        = 0
        r = False

        for i_episode in range(num_train_episodes):
            state_t, episode_return, episode_length, terminal = self.env.reset(), 0, 0, False
            self.exploration_noise = OrnsteinUhlenbeckActionNoise(size=self.env.action_space.n_outputs)
            while not (terminal or (episode_length == max_episode_length)):
                # For the first `start_steps` steps, use randomly sampled actions
                # in order to encourage exploration.
                predicted_action = self.get_predicted_action(state_t, episode_length)
                action = self.get_action(predicted_action)
                state_t_1, reward, terminal = self.env.step(action)
                d_store = False if episode_length == max_episode_length else terminal
                
                self.memorize(state_t[0], predicted_action, reward, state_t_1[0], d_store)

                num_steps += 1                
                
                episode_return += reward
                episode_length += 1
                state_t = state_t_1
                
                print(action, reward, episode_length, [self.env.block.block_x, self.env.block.block_y, self.env.block.rotate_angle, self.env.block.shift_x, self.env.block.shift_y, self.env.block.distance()])
                
                xx, xy = self.update_network()
                print("critic loss", xx)
                
            #mean, stdev = self.gather_stats()
            returns.append([i_episode + 1, episode_length, mean, stdev])

            print("Episode:", i_episode + 1, "Return:", episode_return, 'episode_length:', episode_length, 'stats (m, s)', [mean, stdev])

        return returns

    def memorize(self, state, action, reward, next_state, done):
        self.memory.add_sample({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })

    def learn_current_action(self, state, action, reward, next_state, terminal):
        target_action = self._actor.predict_target(next_state)
        target_q = self._critiq.predict_target([next_state, target_action])
        
        critic_target = (
            reward + (1 - terminal) * self.GAMMA * target_q.reshape((1,))
        ).reshape((1,1))
        
        self._critiq.train_on_batch(state, np.array([action]), critic_target)
        predicted_action = self._actor.predict(state)

        action_grads = self._critiq.action_gradients(state, predicted_action)
        self._actor.train(state, action, action_grads[0])
        
    def update_network(self):
        batch = self.memory.sample(self.BATCH_SIZE)
        batch_len = len(batch)
        
        states = np.array([val['state'] for val in batch])
        actions = np.array([val['action'] for val in batch])
        rewards = np.array([val['reward'] for val in batch])
        next_states = np.array([val['next_state'] for val in batch ])
        terminals = np.array([val['done'] for val in batch ])

        target_actions = self._actor.predict_target(next_states)
        target_q = self._critiq.predict_target([next_states, target_actions])
        
        critic_target = (
            rewards + (1 - terminals) * self.GAMMA * target_q.reshape((batch_len,))
        ).reshape((batch_len,1))
        
        critic_lose = self._critiq.train_on_batch(states, actions, critic_target)
        predicted_actions = self._actor.predict(states)

        action_grads = self._critiq.action_gradients(states, predicted_actions)
        actor_lose = self._actor.train(states, actions, action_grads[0])

        self._actor.update_target_network()
        self._critiq.update_target_network()

        return critic_lose, actor_lose
        
    def gather_stats(self):
        print("Gatthering Stats")
        score = []
        step_count = 0
        for k in range(10):
            old_state = self.env.reset()
            cumul_r, done = 0, False
            while not (done or step_count == 500):
                step_count += 1
                a = self.get_action(old_state)
                old_state, r, done = self.env.step(a)
                cumul_r += r
                score.append(cumul_r)
        return np.mean(np.array(score)), np.std(np.array(score))
