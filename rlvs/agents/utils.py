import os
import torch
import numpy as np
from torch.autograd import Variable
from torch_geometric.data import Data
from time import time


# TODO: Find better place for utils.
# Source: https://raw.githubusercontent.com/ghliu/pytorch-ddpg/master/util.py

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LONG = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
INT32 = torch.cuda.IntTensor if USE_CUDA else torch.IntTensor


def use_device(var):
    return var.cuda() if USE_CUDA else var


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
        torch.from_numpy(ndarray),
        volatile=volatile,
        requires_grad=requires_grad
    ).type(dtype)


def batchify(molecules, data=True):
    data_function = {
        True: lambda mol: mol,
        False: lambda mol: mol.data
     }[data]
    
    X = torch.vstack([data_function(mol).x for mol in molecules])

    data_count = [data_function(mol).x.shape[0] for mol in molecules]
    edge_index = torch.hstack([
        data_function(mol).edge_index + (0 if index == 0 else data_count[index - 1])
        for index, mol in enumerate(molecules)
    ])

    edge_attr = torch.vstack([
        data_function(mol).edge_attr for mol in molecules
    ])

    batch = torch.tensor(np.concatenate(
        [[i] * l for i, l in enumerate(data_count)]
    ))

    return Data(x=X, edge_index=edge_index, batch=batch, edge_attr=edge_attr)

def filter_by_distance(protein, ligand, distance_threshold=4):
    ligand_coords = ligand.atoms.coords
    if distance_threshold is not None:
        distances = np.array([
            protein.distance(coord) for coord in ligand_coords
        ])

    _interacting_edges = np.argwhere(distances <= distance_threshold)
    return _interacting_edges


def interacting_edges(protein, ligand):
    n_p_atoms = len(protein.atoms)
    c_alpha = [atom.idx for atom in protein.atoms.where(lambda x: x.is_heavy_atom)]

    heavy_element = np.array([
        atom.idx for atom in ligand.atoms.where(lambda x: x.is_heavy_atom)
    ]) + n_p_atoms
    
    adg_mat = np.array(np.meshgrid(c_alpha, heavy_element)).T.reshape(-1,2)
    return torch.from_numpy(
        np.hstack([adg_mat.T, [adg_mat[:,1], adg_mat[:,0]]])
    )


def molecule_median_distance(protein, ligand, quantile=0.5):
    ligand_data = ligand.data.x
    distances = np.array([
        protein.distance(feature[:3]) for feature in ligand_data
    ])

    return np.quantile(np.median(distances, axis=1), quantile)


def timeit(function_name):
    def timer_func(func):
        # This function shows the execution time of
        # the function object passed
        def wrap_func(*args, **kwargs):
            t1 = time()
            result = func(*args, **kwargs)
            t2 = time()
            print(f'Function {function_name} executed in {(t2-t1):.4f}s')
            return result
        return wrap_func

    return timer_func
