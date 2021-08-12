import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..agents.utils import timeit

from .graph_cnn import GraphConv
from torch_geometric.nn import GCNConv, Sequential, global_mean_pool
import torch_geometric.nn as t_nn

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class ActorGNN(nn.Module):
    def __init__(self, input_shape, action_shape, learning_rate, tau=0.001, init_w=3e-3):
        super(ActorGNN, self).__init__()
        self._learning_rate = learning_rate
        self._tau = tau
        
        self.complex_gcn_in = GCNConv(input_shape, 16)
        self.complex_gcn_hidden_1 = GCNConv(16, 64)
        self.complex_gcn_out = GCNConv(64, 32)

        self.action_layer_in = nn.Linear(32, 64)
        self.action_layer_out = nn.Linear(64, action_shape)
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.complex_gcn_in.weight.data = fanin_init(self.complex_gcn_in.weight.data.size())
        self.complex_gcn_hidden_1.weight.data = fanin_init(self.complex_gcn_hidden_1.weight.data.size())
        self.complex_gcn_out.weight.data = fanin_init(self.complex_gcn_out.weight.data.size())

        self.action_layer_in.weight.data = fanin_init(self.action_layer_in.weight.data.size())
        self.action_layer_out.weight.data.uniform_(-init_w, init_w)

    def forward(self, complex_):
        complex_data, complex_edge_index, complex_batch = complex_.x, complex_.edge_index, complex_.batch

        complex_data = self.complex_gcn_in(complex_data, complex_edge_index)
        complex_data = F.relu(complex_data)
        complex_data = F.dropout(complex_data, training=self.training)
        complex_data = self.complex_gcn_hidden_1(complex_data, complex_edge_index)
        complex_data = F.relu(complex_data)
        complex_data = F.dropout(complex_data, training=self.training)
        complex_data = self.complex_gcn_out(complex_data, complex_edge_index)
        molecule_data = global_mean_pool(complex_data, complex_batch)
        
        action = F.relu(self.action_layer_in(molecule_data))
        action = self.action_layer_out(action)
        action = torch.tanh(action)
        return action
