import torch
import torch.nn as nn
from torch.nn import Linear, LayerNorm, ReLU, GRU
from torch_geometric.data import Data
import torch.nn.functional as F
import torch_geometric.transforms as T
import numpy as np
from ..agents.utils import timeit

from torch_geometric.nn import GENConv, DeepGCNLayer, global_mean_pool as gap, global_max_pool as gmp
import torch_geometric.nn as t_nn

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

transform = T.Cartesian(cat=False)
class ActorGNN(nn.Module):
    def __init__(self, input_shape, edge_shape, action_shape, learning_rate, tau=0.001, init_w=3e-3):
        super(ActorGNN, self).__init__()
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
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.action_layer_in = nn.Linear(32, 128)
        self.action_layer_out = nn.Linear(128, action_shape)

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

        complex_data = F.relu(self.complex_gcn_in1(complex_data, complex_edge_index, complex_edge_attr))
        complex_data, complex_edge_index, complex_edge_attr, complex_batch,  perm, score  = self.pool1(complex_data, complex_edge_index, complex_edge_attr, complex_batch)
        complex_data = F.relu(self.complex_gcn_in2(complex_data, complex_edge_index, complex_edge_attr))

        
        # complex_data = self.complex_gcn_in(complex_data, complex_edge_index, complex_edge_attr)

        molecule_data = global_mean_pool(complex_data, complex_batch)
        # import pdb;pdb.set_trace()
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

        complex_data = self.layers[0].conv(complex_data, complex_edge_index, complex_edge_attr)

        for layer in self.layers[1:]:
            comlpex_data = layer(complex_data, complex_edge_index, complex_edge_attr)

        complex_data = self.layers[0].act(self.layers[0].norm(complex_data))
        complex_data = F.dropout(complex_data, p=0.1, training=self.training)
        molecule_data = torch.cat([gmp(complex_data, complex_batch), gap(complex_data, complex_batch)], dim=1)
        action = F.relu(self.action_layer_in(molecule_data))
        action = self.action_layer_out(action)
        return action

class ActorGCN(ActorDQN):
    def __init__(self, input_shape, edge_shape, action_shape, learning_rate, tau=0.001, init_w=3e-3):
        super(
            ActorDQN, self
        ).__init__(
            input_shape, edge_shape, action_shape, learning_rate, tau, init_w
        )
    
    def forward(self, complex_):
        action = super().forward(complex_)
        return F.softmax(action)
