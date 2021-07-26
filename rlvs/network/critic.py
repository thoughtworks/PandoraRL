import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..agents.utils import timeit

from .graph_cnn import GraphConv
from torch_geometric.nn import GCNConv, global_mean_pool

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class CriticGNN(nn.Module):
    def __init__(self, input_shape, action_shape, learning_rate, tau=0.001, init_w=3e-3):
        super(CriticGNN, self).__init__()
        self._learning_rate = learning_rate
        self._tau = tau

        self.complex_gcn_in = GCNConv(input_shape, 16)
        self.complex_gcn_hidden_1 = GCNConv(16, 64)
        self.complex_gcn_out = GCNConv(64, 32)

        self.policy_layer_in = nn.Linear(32, 64)
        self.policy_layer_hidden = nn.Linear(64 + action_shape, 10)
        self.policy_layer_out = nn.Linear(10, 1)
        
        # self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.ligand_gcn_in = fanin_init(self.ligand_gcn_in.weight.data.size())
        self.ligand_gcn_out = fanin_init(self.ligand_gcn_out.weight.data.size())

        self.protein_gcn_in = fanin_init(self.protein_gcn_in.weight.data.size())
        self.protein_gcn_out = fanin_init(self.protein_gcn_out.weight.data.size())

        self.policy_layer_in = fanin_init(self.policy_layer_in.weight.data.size())
        self.policy_layer_hidden = fanin_init(self.policy_layer_hidden.weight.data.size())
        self.policy_layer_out.weight.data.uniform_(-init_w, init_w)

    @timeit("critic forward")
    def forward(self, state):
        complex_, action = state
        complex_data, complex_edge_index, complex_batch = complex_.x, complex_.edge_index, complex_.batch

        complex_data = self.complex_gcn_in(complex_data, complex_edge_index)
        complex_data = F.relu(complex_data)
        complex_data = F.dropout(complex_data, training=self.training)
        complex_data = self.complex_gcn_hidden_1(complex_data, complex_edge_index)
        complex_data = F.relu(complex_data)
        complex_data = F.dropout(complex_data, training=self.training)
        complex_data = self.complex_gcn_out(complex_data, complex_edge_index)
        molecule_data = global_mean_pool(complex_data, complex_batch)

        finger_print = F.relu(self.policy_layer_in(molecule_data))

        policy = self.policy_layer_hidden(torch.cat([finger_print, action], 1))
        policy = self.policy_layer_out(F.relu(policy))

        return policy
