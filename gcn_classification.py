from rlvs.network.actor import ActorDQN
from rlvs.molecule_world.env import GraphEnv
from rlvs.molecule_world.protein import Protein
from rlvs.molecule_world.ligand import Ligand
from rlvs.molecule_world.complex import Complex
from rlvs.agents.utils import batchify, to_tensor, to_numpy

from rlvs.constants import AgentConstants

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
import rlvs.molecule_world.helper_functions as hf
import os
import logging

class Loader:
    def __init__(self, path):
        self.path = path
        self.protein = Protein(path=f'/home/justin/Documents/Projects/LifeScience/rl-virtual-screening/data/dqn_spike/3v3m/3v3m_protein.pdb', name='3v3m', filetype='pdb')
        self.data = {}
    
    def get(self, indexs):
        ligand = lambda idx: Ligand(path=f'{self.path}/{idx}/3v3m_ligand.pdbqt', filetype='pdbqt')
        complex = Complex(self.protein, lig:=ligand(0), lig)
        data = []
        for idx in indexs:
            # if idx not in self.data:
            complex.ligand = ligand(idx)
            data.append(complex.data)
                
        # data = [Complex(self.protein, lig:=ligand(idx), lig).data for idx in indexs]

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

    actor_optim = Adam(actor.parameters(), lr=prate)

    return actor, actor_optim


def train(actor, actor_optim, data_loader, y_actual, epochs, indexs, batch_size=32):
    actor.train()
    losses = []
    i = j = 0
    while i < epochs:
        j = 0
        e_losses = []
        for beg_i in range(0, len(indexs), batch_size):
            batch = data_loader.get(indexs[beg_i:beg_i+batch_size])
                
            y = to_tensor(y_actual[indexs[beg_i:beg_i+batch_size]])
            # actor_optim.zero_grad()
            y_hat = actor(batch)
            loss = criterion(y_hat, y)
            loss.backward()
            actor_optim.step()
            losses.append(to_numpy(loss.data))
            e_losses.append(to_numpy(loss.data))
            print(f'E: {i}, Iter: {j}, Loss: {loss}')
            logging.info(f'E: {i}, Iter: {j}, Loss: {loss}')
            j += 1
        print("Episode Loss: ",np.mean(e_losses))
        i += 1
        
    return losses

def test(actor, data_loader, indexs, y_actual):
    actor.eval()
    y_hat = actor(data.get(indexs))
    cls = np.argmax(y_hat, dim=1)
    original = y_actual[indexs]
    import pdb;pdb.set_trace()


def generate_data(output_path, num_of_records=10000):
    env = GraphEnv(single_step=np.array([1]))
    y_vals = {
        1: [0, 1],
        -1: [1, 0]
    }
    i = 0
    ys = []
    rmse = []
    while i < num_of_records:
        print("YVAL:", y_vals[hf.RANDOM_POS_SIGN])
        ys.append(y_vals[hf.RANDOM_POS_SIGN])
        mol, _ = env.reset()
        rmse.append(mol.ligand.rmsd(mol.original_ligand))
        _dir = f'{output_path}/{i}'
        os.makedirs(_dir)
        mol.save(f'{_dir}/3v3m')

        i += 1
    output = {'Y_actual':ys, 'rmsd': rmse }
    with open(f'{output_path}/output.npy', 'wb') as f:
        np.save(f, output)

def read_data(path, num_of_records=10000):
    out_file = f'{path}/output.npy'
    slice_idx = round(num_of_records * 0.8)
    with open(out_file, 'rb') as f:
        output = np.load(f, allow_pickle=True)
    
    y_actual = np.array(output.item().get('Y_actual'))
    rmsd = np.array(output.item().get('rmsd'))
    sample = np.random.choice(num_of_records, num_of_records, replace=False)
    test = sample[slice_idx:]
    train = sample[:slice_idx]

    return test, train, y_actual, rmsd



if __name__ == '__main__':
    path = '/home/justin/Documents/Projects/LifeScience/rl-virtual-screening/test_data'
    # generate_data(path)
    actor, optim = get_network()
    data_loader = Loader(path)
    test_, train_, y_actual, rmsd = read_data(path)
    train(actor, optim, data_loader, y_actual, 10, train_)
    test(actor, data_loader, y_actual, test_)
    print(path)
