import os
import torch
import numpy as np
from torch.autograd import Variable
from torch_geometric.data import Data


#TODO: Find better place for utils.
# Source: https://raw.githubusercontent.com/ghliu/pytorch-ddpg/master/util.py

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)

def batchify(molecules):
    X = torch.vstack([mol.data.x for mol in molecules])
    data_count = [mol.data.x.shape[0] for mol in molecules]
    edge_index = torch.hstack([
        mol.data.edge_index + (0 if index == 0 else data_count[index - 1])
        for index, mol in enumerate(molecules)
    ])
    
    batch = torch.tensor(np.concatenate([[i] * l for i, l in enumerate(data_count)]))

    return Data(x=X, edge_index=edge_index, batch=batch)
    
def interacting_edges(protein, ligand, distance_threshold):
    ligand_data = ligand.data.x
    distances = np.array([
        protein.distance(feature[:3]) for feature in ligand_data
    ])

    return torch.tensor(np.argwhere(distances <= distance_threshold), dtype=torch.long).t()

def molecule_median_distance(protein, ligand, quantile=0.5):
    ligand_data = ligand.data.x
    distances = np.array([
        protein.distance(feature[:3]) for feature in ligand_data
    ])

    return np.quantile(np.median(distances, axis=1), quantile)
