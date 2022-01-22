import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import GENConv, global_mean_pool
from torch_geometric.nn import GENConv, DeepGCNLayer, \
    global_mean_pool as gap, global_max_pool as gmp


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
        hidden_channels = 16
        num_layers = 4

        self.node_encoder = nn.Linear(input_shape, hidden_channels)
        self.edge_encoder = nn.Linear(edge_shape, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)
            act = nn.ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.policy_layer_in = nn.Linear(32, 128)
        self.policy_layer_hidden = nn.Linear(128 + action_shape, 10)
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

        complex_data = self.layers[0].conv(complex_data, complex_edge_index, complex_edge_attr)

        for layer in self.layers[1:]:
            complex_data = layer(complex_data, complex_edge_index, complex_edge_attr)

        complex_data = self.layers[0].act(self.layers[0].norm(complex_data))
        complex_data = F.dropout(complex_data, p=0.1, training=self.training)
        molecule_data = torch.cat([
            gmp(complex_data, complex_batch),
            gap(complex_data, complex_batch)
        ], dim=1)

        finger_print = F.relu(self.policy_layer_in(molecule_data))

        policy = self.policy_layer_hidden(torch.cat([finger_print, action], 1))
        policy = self.policy_layer_out(F.relu(policy))

        return policy
