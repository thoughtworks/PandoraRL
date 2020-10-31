# refered from https://github.com/germain-hug/Deep-RL-Keras
# GraphNN https://github.com/vermaMachineLearning/keras-deep-graph-learning
# Follow the examples to build the graph

# A(G(P), G(L)) -> <pose>
# C([G(P), G(L)], <pose>) -> <Q>


import numpy as np

from .memory import Memory
from .noise import OrnsteinUhlenbeckActionNoise
from rlvs.network import Actor, Critic, Actor3D, Critic3D, ActorGNN, CriticGNN
import tensorflow as tf
from tensorflow.keras.layers import Concatenate

import logging

class DDPGAgent:
    ACTOR_LEARNING_RATE  = 0.00005
    CRITIQ_LEARNING_RATE = 0.00005
    TAU                  = 0.001

    BATCH_SIZE           = 32
    BUFFER_SIZE          = 20000    
    
    def __init__(self, env):
        self.input_shape = env.input_shape
        self.action_shape = env.action_space.n_outputs
        self.eps = 0.9
        self.action_bounds = env.action_space.action_bounds
        self.memory = Memory(self.BUFFER_SIZE)
        self.env = env
        self.exploration_noise = OrnsteinUhlenbeckActionNoise(size=self.env.action_space.n_outputs)

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
        # Explore AdaptiveParamNoiseSpec, with normalized action space
        # https://github.com/l5shi/Multi-DDPG-with-parameter-noise/blob/master/Multi_DDPG_with_parameter_noise.ipynb
        action = self._actor.predict(state).flatten()
        print("ACTION", action)
        if step is not None:
            noise = self.exploration_noise.generate(step)
            print("noise:", noise)
            action += noise
        return action
        
    def get_action(self, action):
        action *= self.action_bounds[1]
        
        x, y, t = np.clip(action, *self.action_bounds)
        return np.array([np.round(x), np.round(y), t])

    def log(self, action, reward, episode_length):
        print("Action:", action, "Reward:", np.round(reward, 4), "E_i:", episode_length,
              "Block state:", [
                  self.env.block.block_x, self.env.block.block_y, np.round(self.env.block.rotate_angle, 2), self.env.block.shift_x, self.env.block.shift_y
              ], "Dist:",  np.round(self.env.block.distance(), 4))
        
    def play(self, num_train_episodes):
        returns      = []
        num_steps    = 0
        max_episode_length = 500
        max_reward = 0

        for i_episode in range(num_train_episodes):
            critic_losses = []
            actor_losses = []
            state_t, episode_return, episode_length, terminal = self.env.reset(), 0, 0, False
            self.exploration_noise = OrnsteinUhlenbeckActionNoise(size=self.env.action_space.n_outputs)
            while not (terminal or (episode_length == max_episode_length)):
                predicted_action = self.get_predicted_action(state_t, episode_length)
                action = self.get_action(predicted_action)
                state_t_1, reward, terminal = self.env.step(action)
                d_store = False if episode_length == max_episode_length else terminal
                
                self.memorize(state_t[0], predicted_action, reward, state_t_1[0], d_store)

                num_steps += 1                
                
                episode_return += reward
                episode_length += 1
                state_t = state_t_1

                self.log(action, np.round(reward, 4), episode_length)
                # return 0
                critic_loss, actor_loss = self.update_network()
                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)
                
            returns.append([i_episode + 1, episode_length, episode_return, np.mean(critic_losses), np.mean(actor_losses)])
            max_reward = max_reward if max_reward > episode_return else episode_return
            print("Episode:", i_episode + 1, "Return:", episode_return, 'episode_length:', episode_length, 'Max Reward', max_reward, "Critic Loss: ", np.mean(critic_losses), " Actor loss: ", np.mean(actor_losses))

        return returns

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
        batch_len = len(batch)
        
        states = np.array([val['state'] for val in batch])
        actions = np.array([val['action'] for val in batch])
        rewards = np.array([val['reward'] for val in batch])
        next_states = np.array([val['next_state'] for val in batch ])
        terminals = np.array([val['done'] for val in batch ])
        state_tensor = tf.convert_to_tensor(states)   

        action_gradient, actor_loss = self._critiq.action_gradients(states, self._actor.actor)
        self._actor.optimize(action_gradient)

        critic_loss = self._critiq.train(states, actions, rewards, terminals, next_states, self._actor.actor_target)

        self._actor.update_target_network()
        self._critiq.update_target_network()
        print(critic_loss, actor_loss)
        return critic_loss, actor_loss
        
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
                old_state, max_reward, done = self.env.step(a)
                cumul_r += max_reward
                score.append(cumul_r)
        return np.mean(np.array(score)), np.std(np.array(score))


class DDPGAgent3D(DDPGAgent):
    ACTOR_LEARNING_RATE  = 0.00005
    CRITIQ_LEARNING_RATE = 0.00005
    TAU                  = 0.001
    
    GAMMA                = 0.99

    BATCH_SIZE           = 32
    BUFFER_SIZE          = 20000
        
    def __init__(self, env):
        # self.input_shape = env.input_shape
        self.action_shape = env.action_space.n_outputs
        self.eps = 0.9
        self.action_bounds = env.action_space.action_bounds
        self.memory = Memory(self.BUFFER_SIZE)
        self.env = env
        self.exploration_noise = OrnsteinUhlenbeckActionNoise(size=self.env.action_space.n_outputs)

        self._actor = Actor(
            # self.input_shape,
            self.action_shape,
            self.ACTOR_LEARNING_RATE,
            self.TAU
        )

        self._critiq = Critic(
            # self.input_shape,
            self.action_shape,
            self.CRITIQ_LEARNING_RATE,
            self.TAU
        )

    def get_action(self, action):
        action *= self.action_bounds[1]
        
        x, y, z, r, p, _y = np.clip(action, *self.action_bounds)
        return np.array([np.round(x), np.round(y), np.round(z), r, p, _y])
    
    def log(self, action, reward, episode_length):
        print(
            "Action:", action,
            "Reward:", np.round(reward, 4),
            "E_i:", episode_length,
            "RMSD: ", self.env._complex.rmsd
        )


class DDPGAgentGNN(DDPGAgent):
    ACTOR_LEARNING_RATE  = 0.00005
    CRITIQ_LEARNING_RATE = 0.0001
    TAU                  = 0.001
    
    GAMMA                = 0.99

    BATCH_SIZE           = 32
    BUFFER_SIZE          = 200000
    EXPLORATION_EPISODES = 10000
        
    def __init__(self, env, log_filename, weights_path):
        self.input_shape = env.input_shape
        self.action_shape = env.action_space.n_outputs
        self.eps = 0.9
        self.action_bounds = env.action_space.action_bounds
        self.memory = Memory(self.BUFFER_SIZE)
        self.env = env
        self.exploration_noise = OrnsteinUhlenbeckActionNoise(size=self.env.action_space.n_outputs, theta=0.15, sigma=0.3, n_steps_annealing=self.EXPLORATION_EPISODES)

        self._actor = ActorGNN(
            self.input_shape,
            self.action_shape,
            self.action_bounds,
            self.ACTOR_LEARNING_RATE,
            self.TAU
        )

        self._critiq = CriticGNN(
            self.input_shape,
            self.action_shape,
            self.CRITIQ_LEARNING_RATE,
            self.TAU
        )

        logging.basicConfig(
            filename=log_filename,
            filemode='w',
            format='%(message)s', 
            datefmt='%I:%M:%S %p', 
            level=logging.DEBUG
        )
        self.weights_path = weights_path

    def get_predicted_action(self, state, step=None, cur_episode=0):
        # Explore AdaptiveParamNoiseSpec, with normalized action space
        # https://github.com/l5shi/Multi-DDPG-with-parameter-noise/blob/master/Multi_DDPG_with_parameter_noise.ipynb
        action = self._actor.predict(state).flatten()
        print("ACTION", action)
        if step is not None and cur_episode < self.EXPLORATION_EPISODES:
            noise = self.exploration_noise.generate(step)
            print("noise:", noise)
            action = np.clip(action + noise, -1, 1)
        return action


    def get_action(self, action):
        action *= self.action_bounds[1]
        
        x, y, z, r, p, _y = np.clip(action, *self.action_bounds)
        return np.array([x, y, z, r, p, _y])
    
    def log(self, action, reward, episode_length, i_episode):
        print(
            "Action:", action,
            "Reward:", np.round(reward, 4),
            "E_i:", episode_length,
            "RMSD: ", self.env._complex.rmsd,
            "E:",i_episode
        )
        logging.info(f"Action: {action}, Reward: {np.round(reward, 4)}, E_i: {episode_length}, E: {i_episode}, RMSD: {np.round(self.env._complex.rmsd, 4)}")

    def play(self, num_train_episodes):
        returns      = []
        num_steps    = 0
        max_episode_length = 50000
        max_reward = 0

        for i_episode in range(num_train_episodes):
            critic_losses = []
            actor_losses = []
            
            m_complex_t, state_t = self.env.reset()
            episode_return, episode_length, terminal = 0, 0, False
            while not (terminal or (episode_length == max_episode_length)):
                predicted_action = self.get_predicted_action(state_t, episode_length, i_episode)
                action = self.get_action(predicted_action)
                m_complex_t_1, state_t_1, reward, terminal = self.env.step(action)
                d_store = False if episode_length == max_episode_length else terminal
                self.memorize(m_complex_t, predicted_action, reward, m_complex_t_1, d_store)
                
                num_steps += 1                
                
                episode_return += reward
                episode_length += 1
                state_t = state_t_1

                self.log(action, np.round(reward, 4), episode_length, i_episode)
                if episode_length % 5 == 0 and self.memory.num_samples > 32:
                    self.update_network(critic_losses, actor_losses)                

            # training_length = 20 if episode_length > 20 else episode_length

            # for i in range(training_length):
            # print(f"E_i:{i_episode + 1} {i}/{training_length}")
            self.update_network(critic_losses, actor_losses)                
                
            returns.append([i_episode + 1, episode_length])
            max_reward = max_reward if max_reward > episode_return else episode_return
            print("Episode:", i_episode + 1, "Return:", episode_return, 'episode_length:', episode_length, 'Max Reward', max_reward, "Critic Loss: ", np.mean(critic_losses), " Actor loss: ", np.mean(actor_losses))
            logging.info(f"Episode: {i_episode + 1}, Return: {episode_return}, episode_length: {episode_length} , Max Reward: {max_reward} , Critic Loss: {np.mean(critic_losses)} , Actor loss: {np.mean(actor_losses)}\n\n")

            # save weights periodically
            if i_episode%10 == 0:
                self.save_weights(self.weights_path)

        return returns
        

        
    def update_network(self, critic_losses, actor_losses):
        batch = self.memory.sample(self.BATCH_SIZE)

        states = self.env.get_state([val['state'] for val in batch])
        actions = np.array([val['action'] for val in batch])
        rewards = np.array([val['reward'] for val in batch])
        next_states = self.env.get_state([val['next_state'] for val in batch ])
        terminals = np.array([val['done'] for val in batch ])
        

        action_gradient, actor_loss = self._critiq.action_gradients(states, self._actor.actor)

        self._actor.optimize(action_gradient)

        critic_loss = self._critiq.train(states, actions, rewards, terminals, next_states, self._actor.actor_target)

        critic_losses.append(critic_loss)
        actor_losses.append(actor_loss)

        self._actor.update_target_network()
        self._critiq.update_target_network()

        print(f"C : {critic_loss}, A: {actor_loss}")
        logging.info(f"C : {critic_loss}, A: {actor_loss}")

    def save_weights(self, path):
        self._actor.save(path)
        self._critiq.save(path)

    def load_weights(self, path_actor, path_critic):
        self._critiq.load_weights(path_critic)
        self._actor.load_weights(path_actor)

    def test(self, max_steps, path_actor_weights, path_critic_weights):
        self.load_weights(path_actor=path_actor_weights, path_critic=path_critic_weights)

        old_complex, old_state = self.env.reset() #new ligand
        for step in range(max_steps):
            action = self.get_action(self.get_predicted_action(old_state))
            print(f"Action = {action}")
            updated_complex, old_state, terminal = self.env.step(action)
            print(f"RMSD = {self.env._complex.rmsd}")
            if terminal:
                break
