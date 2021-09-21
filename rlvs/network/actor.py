import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..agents.utils import timeit

from torch_geometric.nn import GENConv, Sequential, global_mean_pool
import torch_geometric.nn as t_nn

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class ActorGNN(nn.Module):
    def __init__(self, input_shape, edge_shape, action_shape, learning_rate, tau=0.001, init_w=3e-3):
        super(ActorGNN, self).__init__()
        self._learning_rate = learning_rate
        self._tau = tau
        hidden_channels = 32

        self.node_encoder = nn.Linear(input_shape, hidden_channels)
        self.edge_encoder = nn.Linear(edge_shape, hidden_channels)
        
        self.complex_gcn_in = GENConv(hidden_channels, 64, num_layers=4)

        self.action_layer_in = nn.Linear(64, 16)
        self.action_layer_out = nn.Linear(16, action_shape)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.node_encoder.weight.data = fanin_init(self.node_encoder.weight.data.size())
        self.edge_encoder.weight.data = fanin_init(self.edge_encoder.weight.data.size())
        
        self.action_layer_in.weight.data = fanin_init(self.action_layer_in.weight.data.size())
        self.action_layer_out.weight.data.uniform_(-init_w, init_w)

    @timeit("actor_forward")
    def forward(self, complex_):
        complex_data, complex_edge_index, \
            complex_edge_attr, complex_batch = complex_.x, complex_.edge_index,\
                complex_.edge_attr, complex_.batch

        complex_data = self.node_encoder(complex_data)
        complex_edge_attr = self.edge_encoder(complex_edge_attr)

        complex_data = self.complex_gcn_in(complex_data, complex_edge_index, complex_edge_attr)

        molecule_data = global_mean_pool(complex_data, complex_batch)
        
        action = F.relu(self.action_layer_in(molecule_data))
        action = self.action_layer_out(action)
        action = torch.tanh(action)
        return action


class ActorDQN(ActorGNN):
    def __init__(self, input_shape, edge_shape, action_shape, learning_rate, tau=0.001, init_w=3e-3):
        super(
            ActorDQN, self
        ).__init__(
            input_shape, edge_shape, action_shape, learning_rate, tau, init_w
        )

    def forward(self, complex_):
        complex_data, complex_edge_index, \
            complex_edge_attr, complex_batch = complex_.x, complex_.edge_index,\
                complex_.edge_attr, complex_.batch

        complex_data = self.node_encoder(complex_data)
        complex_edge_attr = self.edge_encoder(complex_edge_attr)

        complex_data = self.complex_gcn_in(complex_data, complex_edge_index, complex_edge_attr)

        molecule_data = global_mean_pool(complex_data, complex_batch)
        
        action = F.relu(self.action_layer_in(molecule_data))
        action = self.action_layer_out(action)
        return action
