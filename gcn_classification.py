from rlvs.network.actor import ActorDQN
from rlvs.molecule_world.env import GraphEnv
from rlvs.molecule_world.protein import Protein
from rlvs.molecule_world.ligand import Ligand
from rlvs.molecule_world.complex import Complex
from rlvs.agents.utils import batchify, to_tensor, to_numpy, USE_CUDA

from rlvs.constants import AgentConstants

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
import rlvs.molecule_world.helper_functions as hf
from torch_geometric.data import Data

import os
import logging
from shutil import copyfile

class Loader:
    def __init__(self, path, protein_indexes):
        self.path = path
        self.protein_indexes = protein_indexes
        self.protein = lambda idx: Protein(path=f'{self.path}/{idx}/SARS_protein.pdb', filetype='pdb') 
        self.ligand = lambda idx: Ligand(path=f'{self.path}/{idx}/SARS.pdbqt', filetype='pdbqt')
        self.complexes = {}

    def get(self, indexs):
        data = []
        for i in indexs:
            ligand = self.ligand(i)
            if self.protein_indexes[i] in self.complexes:
                complex_ = self.complexes[self.protein_indexes[i]]
                complex_.ligand = ligand
                # complex_.update_edges()
            else:
                complex_ = Complex(
                    self.protein(i), ligand, ligand
                )
                
                complex_.crop(10, 10, 10)
                self.complexes[self.protein_indexes[i]] = complex_
            
            data.append(complex_.data)
            
        return batchify(data)



criterion = nn.BCELoss()
def get_network(prate=0.00005):
    env = GraphEnv(single_step=np.array([1]))
    actor = ActorDQN(
        env.input_shape,
        env.edge_shape,
        env.action_space.degree_of_freedom,
        AgentConstants.ACTOR_LEARNING_RATE ,
        AgentConstants.TAU
    )

    actor = actor.cuda() if USE_CUDA else actor

    actor_optim = Adam(actor.parameters(), lr=prate)

    return actor, actor_optim


def train(actor, actor_optim, data_loader, y_actual, epochs, indexs, batch_size=32):
    actor.train()
    losses = []
    e = j = 0
    while e < epochs:
        j = 0
        e_losses = []
        for beg_i in range(0, len(indexs), batch_size):
            batch = data_loader.get(indexs[beg_i:beg_i+batch_size])
            y = to_tensor(y_actual[indexs[beg_i:beg_i+batch_size]])

            actor_optim.zero_grad()
            y_hat = actor(batch)
            print(y_hat, y)
            loss = criterion(y_hat, y)
            loss.backward()
            actor_optim.step()
            losses.append(to_numpy(loss.data))
            e_losses.append(to_numpy(loss.data))

            print(f'E: {e}, Iter: {j}, Loss: {loss}')
            logging.info(f'E: {e}, Iter: {j}, Loss: {loss}')
            j += 1
        print("episode loss: ",np.mean(e_losses))
        e += 1
        
    return losses

def test(actor, data_loader, indexs, y_actual):
    actor.eval()
    y_hat = []
    losses = []
    for i in indexs:
        print(i)
        data = data_loader.get([i])
        y_pred = actor(data)
        y_hat.append(y_pred)
        y = to_tensor(y_actual[i:i+1])
        loss = criterion(y_pred, y)
        losses.append(to_numpy(loss.data))

    import pdb;pdb.set_trace()
    return losses


def generate_data(output_path, num_of_records=10000):
    env = GraphEnv(single_step=np.array([1]))
    y_vals = {
        1: [0, 1],
        -1: [1, 0]
    }
    i = 0
    ys = []
    prot = []
    rmse = []
    while i < num_of_records:
        print(f"{i} YVAL:", y_vals[hf.RANDOM_POS_SIGN])
        ys.append(y_vals[hf.RANDOM_POS_SIGN])
        mol, _ = env.reset()
        rmse.append(mol.ligand.rmsd(mol.original_ligand))
        _dir = f'{output_path}/{i}'
        os.makedirs(_dir)
        prot.append(mol.protein.name)
        mol.save(f'{_dir}/SARS')
        copyfile(mol.protein.path, f'{_dir}/SARS_protein.{mol.protein.filetype}')
        

        i += 1
    output = {'Y_actual':ys, 'rmsd': rmse, "protein": prot }
    with open(f'{output_path}/output.npy', 'wb') as f:
        np.save(f, output)

def read_data(path, num_of_records=10000):
    out_file = f'{path}/output.npy'
    slice_idx = round(num_of_records * 0.8)
    with open(out_file, 'rb') as f:
        output = np.load(f, allow_pickle=True)
    
    y_actual = np.array(output.item().get('Y_actual'))
    rmsd = np.array(output.item().get('rmsd'))
    prot = np.array(output.item().get('protein'))
    sample = np.random.choice(num_of_records, num_of_records, replace=False)
    test = sample[slice_idx:]
    train = sample[:slice_idx]

    return test, train, y_actual, rmsd, prot



if __name__ == '__main__':
    path = '/home/justin/Documents/Projects/LifeScience/rl-virtual-screening/test_data'
    # generate_data(path)
    actor, optim = get_network()
    test_, train_, y_actual, rmsd, prot = read_data(path)
    data_loader = Loader(path, prot)
    losses = train(actor, optim, data_loader, y_actual, 10, train_, 1)
    
    torch.save(actor.state_dict(), f'{path}_actor')
    with open(f'{path}/losses.npy', 'wb') as f:
        np.save(f, losses)
    with open(f'{path}/test.npy', 'wb') as f:
        np.save(f, test_)

    with open(f'{path}/test.npy', 'rb') as f:
        test_ = np.load(f, allow_pickle=True)

    actor.load_state_dict(torch.load(f'{path}_actor'))
    test_losses = test(actor, data_loader, test_, y_actual)
    with open(f'{path}/test_losses.npy', 'wb') as f:
         np.save(f, test_losses)
