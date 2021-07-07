import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        
        self.ligand_gcn_in = GCNConv(input_shape, 16)
        self.ligand_gcn_out = GCNConv(16, 50)

        self.protein_gcn_in = GCNConv(input_shape, 16)
        self.protein_gcn_out = GCNConv(16, 50)

        self.policy_layer_in = nn.Linear(50 + 50, 60)
        self.policy_layer_hidden = nn.Linear(60 + action_shape, 10)
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
    
    def forward(self, state):
        complex_, action = state
        protein, ligand = complex_.protein, complex_.ligand

        protein_data, protein_edge_index = protein.data.x, protein.data.edge_index
        ligand_data, ligand_edge_index = ligand.data.x, ligand.data.edge_index
        
        protein_batch = torch.tensor([0] * protein_data.shape[0])
        ligand_batch = torch.tensor([0] * ligand_data.shape[0])
        
        protein_data = self.protein_gcn_in(protein_data, protein_edge_index)
        protein_data = F.relu(protein_data)
        protein_data = F.dropout(protein_data, training=self.training)
        protein_data = self.protein_gcn_out(protein_data, protein_edge_index)
        protein_data = global_mean_pool(protein_data, protein_batch)


        ligand_data = self.ligand_gcn_in(ligand_data, ligand_edge_index)
        ligand_data = F.relu(ligand_data)
        ligand_data = F.dropout(ligand_data, training=self.training)
        ligand_data = self.ligand_gcn_out(ligand_data, ligand_edge_index)
        ligand_data = global_mean_pool(ligand_data, ligand_batch)
        molecule_data = torch.cat((protein_data, ligand_data), dim=1)

        finger_print = F.relu(self.policy_layer_in(molecule_data))
        policy = self.policy_layer_hidden(torch.cat([finger_print, action], 1))
        policy = self.policy_layer_out(F.relu(policy))

        return policy
