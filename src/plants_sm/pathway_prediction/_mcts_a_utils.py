import pickle
from multiprocessing import Process
import multiprocessing
import torch
import numpy as np
import signal
import os
from contextlib import contextmanager
from rdkit import Chem
from rdkit.Chem import AllChem
class TimeoutException(Exception): pass

import torch.nn as nn
import torch.nn.functional as F

class GraphLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.drop(F.relu(self.linear(x)))
        return x


class GraphModel(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim):
        super(GraphModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.graphlayer = GraphLayer(input_dim, hidden_dim)

    def forward(self, x, mask):
        x = self.graphlayer(x)
        mask = mask[:, :, None].repeat(1, 1, self.hidden_dim)
        x = torch.sum(x * mask, dim=1)
        return x


class ValueEnsemble(nn.Module):
    def __init__(self, fp_dim, latent_dim, dropout_rate):
        super(ValueEnsemble, self).__init__()
        self.fp_dim = fp_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.graphNN = GraphModel(input_dim=fp_dim, feature_dim=latent_dim, hidden_dim=latent_dim)
        self.layers = nn.Linear(latent_dim, 1, bias=False)

    def forward(self, fps, mask):
        x = self.graphNN(fps, mask)
        x = self.layers(x)
        return x


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def smiles_to_fp(s, fp_dim=2048, pack=False):
    mol = Chem.MolFromSmiles(s)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool)
    arr[onbits] = 1
    if pack:
        arr = np.packbits(arr)
    return arr


def batch_smiles_to_fp(s_list, fp_dim=2048):
    fps = []
    for s in s_list:
        fps.append(smiles_to_fp(s, fp_dim))
    fps = np.array(fps)
    assert fps.shape[0] == len(s_list) and fps.shape[1] == fp_dim
    return fps


class MinMaxStats(object):
    def __init__(self, min_value_bound=None, max_value_bound=None):
        self.maximum = min_value_bound if min_value_bound else -float('inf')
        self.minimum = max_value_bound if max_value_bound else float('inf')

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value) -> float:
        if self.maximum > self.minimum:
            return (np.array(value) - self.minimum) / (self.maximum - self.minimum)
        return value


def prepare_starting_molecules_natural(path):
    fr = open(path, 'r')
    data = []
    for line in fr.readlines():
        line = line.strip().split('\t')
        data.append(line[-1])
    return data


def prepare_value(model_f, gpu=None):
    if gpu == -1:
        device = 'cpu'
    else:
        device = 'cuda:' + str(gpu)
    model = ValueEnsemble(2048, 128, 0.1).to(device)
    model.load_state_dict(torch.load(model_f, map_location=device))
    model.eval()
    return model


def value_fn(model, mols, device):
    num_mols = len(mols)
    fps = batch_smiles_to_fp(mols, fp_dim=2048).reshape(num_mols, -1)
    index = len(fps)
    if len(fps) <= 5:
        mask = np.ones(5)
        mask[index:] = 0
        fps_input = np.zeros((5, 2048))
        fps_input[:index, :] = fps
    else:
        mask = np.ones(len(fps))
        fps_input = fps
    fps = torch.FloatTensor([fps_input.astype(np.float32)]).to(device)
    mask = torch.FloatTensor([mask.astype(np.float32)]).to(device)
    v = model(fps, mask).cpu().data.numpy()
    return v[0][0]

class Node:
    def __init__(self, state, h, prior, cost=0, action_mol=None, fmove=0, reaction=None, template=None, parent=None, cpuct=1.5):
        self.state = state
        self.cost = cost
        self.h = h
        self.prior = prior
        self.visited_time = 0
        self.is_expanded = False
        self.template = template
        self.action_mol = action_mol
        self.fmove = fmove
        self.reaction = reaction
        self.parent = parent
        self.cpuct = cpuct
        self.children = []
        self.child_illegal = np.array([])
        if parent is not None:
            self.g = self.parent.g + cost
            self.parent.children.append(self)
            self.depth = self.parent.depth + 1
        else:
            self.g = 0
            self.depth = 0
        self.f = self.g + self.h
        self.f_mean_path = []

    def child_N(self):
        N = [child.visited_time for child in self.children]
        return np.array(N)

    def child_p(self):
        prior = [child.prior for child in self.children]
        return np.array(prior)

    def child_U(self):
        child_Ns = self.child_N() + 1
        prior = self.child_p()
        child_Us = self.cpuct * np.sqrt(self.visited_time) * prior / child_Ns
        return child_Us

    def child_Q(self, min_max_stats):
        child_Qs = []
        for child in self.children:
            if len(child.f_mean_path) == 0:
                child_Qs.append(0.0)
            else:
                child_Qs.append(1 - np.mean(min_max_stats.normalize(child.f_mean_path)))
        return np.array(child_Qs)

    def select_child(self, min_max_stats):
        action_score = self.child_Q(min_max_stats) + self.child_U() - self.child_illegal
        best_move = np.argmax(action_score)
        return best_move


def gather(dataset, simulations, cpuct, times):
    result = {
        'route': [],
        'template': [],
        'success': [],
        'depth': [],
        'counts': []
    }
    for i in range(28):
        file = './test/stat_norm_retro_' + dataset + '_' + str(simulations) + '_' + str(cpuct) + '_' + str(i) + '.pkl'
        with open(file, 'rb') as f:
            data = pickle.load(f)
        for key in result.keys():
            result[key] += data[key]
        os.remove(file)
    success = np.mean(result['success'])
    depth = np.mean(result['depth'])
    fr = open('result_simulation.txt', 'a')
    fr.write(str(simulations) + '\t' + str(times) + '\t' + str(simulations) + '\t' + str(cpuct) + '\t' + str(success) + '\t' + str(depth) + '\n')
    f = open('./test/stat_norm_retro_' + dataset + '_' + str(simulations) + '_' + str(cpuct) + '_' + str(times) + '.pkl', 'wb')
    pickle.dump(result, f)
    f.close()
