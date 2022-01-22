import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import GENConv, global_mean_pool


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class CriticGNN(nn.Module):
    def __init__(
            self, input_shape, edge_shape,
            action_shape, learning_rate, tau=0.001, init_w=3e-3
    ):
        super(CriticGNN, self).__init__()
        self._learning_rate = learning_rate
        self._tau = tau
        hidden_channels = 32

        self.node_encoder = nn.Linear(input_shape, hidden_channels)
        self.edge_encoder = nn.Linear(edge_shape, hidden_channels)

        self.complex_gcn_in = GENConv(hidden_channels, 64, num_layers=4)

        self.policy_layer_in = nn.Linear(64, 16)
        self.policy_layer_hidden = nn.Linear(16 + action_shape, 10)
        self.policy_layer_out = nn.Linear(10, 1)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.node_encoder.weight.data = fanin_init(self.node_encoder.weight.data.size())
        self.edge_encoder.weight.data = fanin_init(self.edge_encoder.weight.data.size())

        self.policy_layer_in.weight.data = fanin_init(self.policy_layer_in.weight.data.size())
        self.policy_layer_hidden.weight.data = fanin_init(
            self.policy_layer_hidden.weight.data.size()
        )
        self.policy_layer_out.weight.data.uniform_(-init_w, init_w)

    def forward(self, state):
        complex_, action = state
        complex_data, complex_edge_index, \
            complex_edge_attr, complex_batch = complex_.x, complex_.edge_index,\
            complex_.edge_attr, complex_.batch

        complex_data = self.node_encoder(complex_data)
        complex_edge_attr = self.edge_encoder(complex_edge_attr)

        complex_data = self.complex_gcn_in(complex_data, complex_edge_index, complex_edge_attr)

        molecule_data = global_mean_pool(complex_data, complex_batch)

        finger_print = F.relu(self.policy_layer_in(molecule_data))

        policy = self.policy_layer_hidden(torch.cat([finger_print, action], 1))
        policy = self.policy_layer_out(F.relu(policy))

        return policy
