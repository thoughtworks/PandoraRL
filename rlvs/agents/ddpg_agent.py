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

        if step is not None:
            action += self.exploration_noise.generate(step)

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
                
            returns.append([i_episode + 1, episode_length])
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
        
        x, y, z, r, p, y = np.clip(action, *self.action_bounds)
        return np.array([np.round(x), np.round(y), np.round(z), r, p, y])
    
    def log(self, action, reward, episode_length):
        print(
            "Action:", action,
            "Reward:", np.round(reward, 4),
            "E_i:", episode_length,
            "RMSD: ", self.env._complex.rmsd
        )


class DDPGAgentGNN(DDPGAgent):
    ACTOR_LEARNING_RATE  = 0.00005
    CRITIQ_LEARNING_RATE = 0.00005
    TAU                  = 0.001
    
    GAMMA                = 0.99

    BATCH_SIZE           = 10
    BUFFER_SIZE          = 20000
        
    def __init__(self, env):
        self.input_shape = env.input_shape
        self.action_shape = env.action_space.n_outputs
        self.eps = 0.9
        self.action_bounds = env.action_space.action_bounds
        self.memory = Memory(self.BUFFER_SIZE)
        self.env = env
        self.exploration_noise = OrnsteinUhlenbeckActionNoise(size=self.env.action_space.n_outputs)

        self._actor = ActorGNN(
            self.input_shape,
            self.action_shape,
            self.ACTOR_LEARNING_RATE,
            self.TAU
        )

        self._critiq = CriticGNN(
            self.input_shape,
            self.action_shape,
            self.CRITIQ_LEARNING_RATE,
            self.TAU
        )

    def get_action(self, action):
        action *= self.action_bounds[1]
        
        x, y, z, r, p, y = np.clip(action, *self.action_bounds)
        return np.array([np.round(x), np.round(y), np.round(z), r, p, y])
    
    def log(self, action, reward, episode_length, i_episode):
        print(
            "Action:", action,
            "Reward:", np.round(reward, 4),
            "E_i:", episode_length,
            "E:", i_episode,
            "RMSD: ", np.round(self.env._complex.rmsd, 4)
        )

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
                
                self.memorize(state_t, np.array([predicted_action]), np.array(reward), state_t_1, np.array(d_store))

                num_steps += 1                
                
                episode_return += reward
                episode_length += 1
                state_t = state_t_1

                self.log(action, np.round(reward, 4), episode_length, i_episode)
                self.update_network(critic_losses, actor_losses)                
                
            returns.append([i_episode + 1, episode_length])
            max_reward = max_reward if max_reward > episode_return else episode_return
            print("Episode:", i_episode + 1, "Return:", episode_return, 'episode_length:', episode_length, 'Max Reward', max_reward, "Critic Loss: ", np.mean(critic_losses), " Actor loss: ", np.mean(actor_losses))

        return returns
        

        
    def update_network(self, critic_losses, actor_losses):
        batch = self.memory.sample(self.BATCH_SIZE)
        batch_len = len(batch)

        c_losses, a_losses = [], []

        for val in batch:
            action_gradient, actor_loss = self._critiq.action_gradients(val['state'], self._actor.actor)
            self._actor.optimize(action_gradient)
            critic_loss = self._critiq.train(
                val['state'],
                val['action'],
                val['reward'],
                val['done'],
                val['next_state'],
                self._actor.actor_target
            )

            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)

            c_losses.append(critic_loss)
            a_losses.append(actor_loss)

        print("Critic Training Loss: ", np.mean(c_losses), "Actor Training Loss: ", np.mean(a_losses) )

                

        self._actor.update_target_network()
        self._critiq.update_target_network()

