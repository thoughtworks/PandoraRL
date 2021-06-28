import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .graph_cnn import GraphConv

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class CriticGNN(nn.Module):
    def __init__(self, input_shape, action_shape, learning_rate, tau=0.001, init_w=3e-3):
        super(Actor, self).__init__()
        self._learning_rate = learning_rate
        self._tau = tau
        
        self.ligand_gcn_in = GraphConv(input_shape, 16)
        self.ligand_gcn_out = GraphConv(16, 50)

        self.protein_gcn_in = GraphConv(input_shape, 16)
        self.protein_gcn_out = GraphConv(16, 50)

        self.policy_layer_in = nn.Linear(50 + 50, 60)
        self.policy_layer_hidden = nn.Linear(60 + action_shape, 10)
        self.policy_layer_out = nn.Linear(10, 1)
        
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.ligand_gcn_in = fanin_init(self.ligand_gcn_in.weight.data.size())
        self.ligand_gcn_out = fanin_init(self.ligand_gcn_out.weight.data.size())

        self.protein_gcn_in = fanin_init(self.protein_gcn_in.weight.data.size())
        self.protein_gcn_out = fanin_init(self.protein_gcn_out.weight.data.size())

        self.policy_layer_in = fanin_init(self.policy_layer_in.weight.data.size())
        self.policy_layer_hidden = fanin_init(self.policy_layer_hidden.weight.data.size())
        self.policy_layer_out.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, state):
        protein, ligand, action = state

        protein_data, protein_edge_index = protein.data, protein.adgacency
        protein_data = self.protein_gcn_in(protein_data, protein_edge_index)
        protein_data = F.relu(protein_data)
        protein_data = F.dropout(protein_data, training=self.training)
        protein_data = self.protein_gcn_out(protein_data, protein_edge_index)

        ligand_data, ligand_edge_index = ligand.data, ligand.adgacency
        ligand_data = self.ligand_gcn_in(ligand_data, ligand_edge_index)
        ligand_data = F.relu(ligand_data)
        ligand_data = F.dropout(ligand_data, training=self.training)
        ligand_data = self.ligand_gcn_out(ligand_data, ligand_edge_index)

        molecule_data = torch.cat((protein_data, ligand_data), dim=1)

        finger_print = F.relu(self.policy_layer_in(molecule_data))
        policy = self.policy_layer_hidden(torch.cat([finger_print, action], 1))
        policy = self.policy_layer_out(F.relu(policy))

        return policy



# class Critic:
#     GAMMA = 0.99
#     def __init__(self, input_shape, action_shape, learning_rate, tau=0.001):
#         self._tau = tau
#         self._learning_rate = learning_rate
#         self._state_shape = input_shape
#         self._action_shape = action_shape
        
#         self.critic, self._input_layer, self._action_layer = self._create_network()
#         self.critic_target, self._target_input_layer, self._target_action_layer = self._create_network()
#         self.critic_target.set_weights(self.critic.get_weights())
#         self.optimizer = Adam(learning_rate=learning_rate)
    
#     def update_target_network(self):
#         W, target_W = self.critic.get_weights(), self.critic_target.get_weights()
#         for i in range(len(W)):
#             target_W[i] = self._tau * W[i] + (1 - self._tau)* target_W[i]
#         self.critic_target.set_weights(target_W)

#     def train(self, states, actions, rewards, terminals, next_states, target_actor):
#         next_state_tensor = tf.convert_to_tensor(next_states)
#         with tf.GradientTape() as tape:
#             next_actions = target_actor(next_states)
#             target_q = self.critic_target([next_states, next_actions])
#             critic_target = (
#                 rewards + (1 - terminals) * self.GAMMA * target_q
#             )

#             qvals = self.critic([states, actions]) 

#             critic_loss = tf.reduce_mean((qvals - critic_target)**2)
#             q_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
            
#         self.optimizer.apply_gradients(zip(q_gradient, self.critic.trainable_variables))
#         return critic_loss.numpy()

#     def action_gradients(self, states, actor):
#         state_tensor = tf.convert_to_tensor(states)
#         with tf.GradientTape() as tape2:
#             action_tensor = actor(states)
#             q = self.critic([state_tensor, action_tensor])
#             action_loss =  -tf.reduce_mean(q)
#             action_gradient =  tape2.gradient(action_loss, actor.trainable_variables)         

#         return action_gradient, action_loss.numpy()
    

#     def save(self, path):
#         self.critic.save_weights(filepath=path+'_critic.h5', save_format="h5")

#     def load_weights(self, path):
#         self.critic.load_weights(path)


#     def _create_network(self):
#         conv_model = Input(shape=self._state_shape)

#         conv_model_1 = Conv2D(64, (5, 5),  activation='relu')(conv_model)
#         conv_model_1 = Conv2D(64, (4, 4),  activation='relu')(conv_model_1)
#         conv_model_1 = Conv2D(64, (3, 3),  activation='relu')(conv_model_1)
#         conv_model_1 = MaxPooling2D(pool_size=(2, 2))(conv_model_1)
        
        
#         conv_model_1 = Flatten()(conv_model_1)
#         conv_model_1 = Dense(256, activation='relu')(conv_model_1)
#         conv_model_1 = Model(inputs=conv_model, outputs=conv_model_1)

#         action_model = Input(shape=[self._action_shape])
#         action_model_1 = Dense(256, activation='linear')(action_model)
#         action_model_1 = Model(inputs=action_model, outputs=action_model_1)

#         intermediate = Concatenate()([conv_model_1.output, action_model_1.output])
#         intermediate = Dense(256, activation='relu')(intermediate)
#         intermediate = Dropout(0.5)(intermediate)

#         value_layer = Dense(1, activation='linear', kernel_initializer=RandomUniform())(intermediate)

#         model = Model([conv_model, action_model], value_layer)
#         return model, conv_model, action_model


# class Critic3D(Critic):
#     def __init__(self, input_shape, action_shape, learning_rate, tau=0.001):
#         super(Critic3D, self).__init__(input_shape, action_shape, learning_rate, tau)

#     def _create_network(self):
#         conv_model = Input(shape=self._state_shape)

#         conv_model_1 = Conv3D(64, (5, 5, 5),  activation='relu')(conv_model)
#         conv_model_1 = Conv3D(64, (4, 4, 4),  activation='relu')(conv_model_1)
#         conv_model_1 = Conv3D(64, (3, 3, 3),  activation='relu')(conv_model_1)
#         conv_model_1 = MaxPooling3D(pool_size=(2, 2, 2))(conv_model_1)
        
        
#         conv_model_1 = Flatten()(conv_model_1)
#         conv_model_1 = Dense(256, activation='relu')(conv_model_1)
#         conv_model_1 = Model(inputs=conv_model, outputs=conv_model_1)

#         action_model = Input(shape=[self._action_shape])
#         action_model_1 = Dense(256, activation='linear')(action_model)
#         action_model_1 = Model(inputs=action_model, outputs=action_model_1)

#         intermediate = Concatenate()([conv_model_1.output, action_model_1.output])
#         intermediate = Dense(256, activation='relu')(intermediate)
#         intermediate = Dropout(0.5)(intermediate)

#         value_layer = Dense(1, activation='linear', kernel_initializer=RandomUniform())(intermediate)

#         model = Model([conv_model, action_model], value_layer)

#         return model, conv_model, action_model 
    
# class CriticGNN(Critic):
#     def __init__(self, input_shape, action_shape, learning_rate, tau=0.001):
#         self.adjecency_rank = 10
#         super(CriticGNN, self).__init__(input_shape, action_shape, learning_rate, tau)

#     def _create_molecule_network(self, jj=0):
#         features_input = Input(shape=(None, self._state_shape,), batch_size=1, name=f"critic_Feature_{jj}") 
#         degree_slice_input = Input(shape=(11,2), dtype=tf.int32, batch_size=1, name=f"critic_Degree_slice_{jj}")
#         membership = Input(shape=(None,), dtype=tf.int32, name=f'critic_membership_{jj}', batch_size=1)
#         n_samples = Input(shape=(1,), dtype=tf.int32, name=f'critic_n_samples_{jj}', batch_size=1)
#         deg_adjs_input = [Input(shape=(None,None,), dtype=tf.int32, batch_size=1, name=f"critic_deg_adjs_{jj}_{i}") for i in  range(self.adjecency_rank)]
        
#         input_states = [features_input, degree_slice_input, membership, n_samples] + deg_adjs_input
#         graph_layer = GraphConv(layer_id=jj, out_channel=64, activation_fn=tf.nn.relu)(input_states)

#         graph_pool_in = [graph_layer, degree_slice_input, membership, n_samples] + deg_adjs_input
#         graph_pool = GraphPool()(graph_pool_in)
#         dense_layer = Dense(128, activation=tf.nn.relu)(graph_pool)
#         graph_gather_layer = GraphGather(activation_fn=tf.nn.tanh)([dense_layer, membership, n_samples])
#         return input_states, graph_gather_layer

#     def _create_network(self):

#         ip_1, graph_gather_layer_1 = self._create_molecule_network(0)
#         ip_2, graph_gather_layer_2 = self._create_molecule_network(1)

#         mol1_model = Model(inputs=ip_1, outputs=graph_gather_layer_1)
#         mol2_model = Model(inputs=ip_2, outputs=graph_gather_layer_2)

#         combination_layer = add([mol1_model.output, mol2_model.output])
#         combined_dense_layer = Dense(64, activation="relu")(combination_layer)
#         conv_model_1 = Model([ip_1, ip_2], combined_dense_layer)

#         action_model = Input(shape=[self._action_shape], name=f"critic_action")
#         action_model_1 = Dense(64, activation='linear')(action_model)
#         action_model_1 = Model(inputs=action_model, outputs=action_model_1)

#         intermediate = Concatenate()([conv_model_1.output, action_model_1.output])
#         intermediate = Dense(64, activation='relu')(intermediate)
#         intermediate = Dropout(0.5)(intermediate)


#         value_layer = Dense(1, activation='linear', kernel_initializer=RandomUniform())(intermediate)
#         model = Model([[ip_1, ip_2], action_model], value_layer)
#         model.compile(optimizer=Adam(lr=self._learning_rate), loss='mse')

#         return model, model.inputs[0], action_model

    
#     def action_gradients(self, states, actor):
#         with tf.GradientTape() as tape2:
#             action_tensor = actor(states)
#             q = self.critic([states, action_tensor])
#             action_loss =  -tf.reduce_mean(q)
#             action_gradient =  tape2.gradient(action_loss, actor.trainable_variables)         

#         return action_gradient, action_loss.numpy()

#     def train(self, states, actions, rewards, terminals, next_states, target_actor):
#         with tf.GradientTape() as tape:
#             next_actions = target_actor(next_states)
#             target_q = self.critic_target([next_states, next_actions])
#             critic_target = (
#                 rewards + (1 - terminals) * self.GAMMA * target_q
#             )

#             qvals = self.critic([states, actions]) 

#             critic_loss = tf.reduce_mean((qvals - critic_target)**2)
#             q_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
            
#         self.optimizer.apply_gradients(zip(q_gradient, self.critic.trainable_variables))
#         return critic_loss.numpy()
