import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        
        self.ligand_gcn_in = GCNConv(input_shape, 16)
        self.ligand_gcn_out = GCNConv(16, 50)

        self.protein_gcn_in = GCNConv(input_shape, 16)
        self.protein_gcn_out = GCNConv(16, 50)
        self.protein_flat = nn.Flatten()
        


        self.action_layer_in = nn.Linear(50 + 50, 60)
        self.action_layer_out = nn.Linear(60, action_shape)
        # self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.ligand_gcn_in = fanin_init(self.ligand_gcn_in.weight.data.size())
        self.ligand_gcn_out = fanin_init(self.ligand_gcn_out.weight.data.size())

        self.protein_gcn_in = fanin_init(self.protein_gcn_in.weight.data.size())
        self.protein_gcn_out = fanin_init(self.protein_gcn_out.weight.data.size())

        self.action_layer_in = fanin_init(self.action_layer_in.weight.data.size())
        self.action_layer_out.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, complex_):
        protein, ligand = complex_
        
        protein_data, protein_edge_index, protein_batch = protein.x, protein.edge_index, protein.batch
        ligand_data, ligand_edge_index, ligand_batch = ligand.x, ligand.edge_index, ligand.batch

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
        
        action = F.relu(self.action_layer_in(molecule_data))
        action = self.action_layer_out(action)
        action = torch.tanh(action)
        return action
