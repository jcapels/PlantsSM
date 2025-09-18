import pickle
import torch
import numpy as np
import signal
import os
from contextlib import contextmanager
from rdkit import Chem
from rdkit.Chem import AllChem

class TimeoutException(Exception):
    """Exception raised when a time limit is exceeded."""
    pass

class GraphLayer(nn.Module):
    """A single graph layer with linear transformation, ReLU activation, and dropout.

    Attributes
    ----------
    linear : nn.Linear
        Linear transformation layer.
    drop : nn.Dropout
        Dropout layer for regularization.
    """

    def __init__(self, in_features, out_features):
        """Initialize the graph layer.

        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        """
        super(GraphLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        """Forward pass of the graph layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after linear transformation, ReLU activation, and dropout.
        """
        x = self.drop(F.relu(self.linear(x)))
        return x

class GraphModel(nn.Module):
    """Graph neural network model for processing molecular fingerprints.

    Attributes
    ----------
    hidden_dim : int
        Dimension of the hidden layer.
    graphlayer : GraphLayer
        Single graph layer for feature transformation.
    """

    def __init__(self, input_dim, feature_dim, hidden_dim):
        """Initialize the graph model.

        Parameters
        ----------
        input_dim : int
            Dimension of the input features.
        feature_dim : int
            Dimension of the feature space.
        hidden_dim : int
            Dimension of the hidden layer.
        """
        super(GraphModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.graphlayer = GraphLayer(input_dim, hidden_dim)

    def forward(self, x, mask):
        """Forward pass of the graph model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        mask : torch.Tensor
            Mask tensor for zeroing out padded values.

        Returns
        -------
        torch.Tensor
            Summed output tensor after applying the mask.
        """
        x = self.graphlayer(x)
        mask = mask[:, :, None].repeat(1, 1, self.hidden_dim)
        x = torch.sum(x * mask, dim=1)
        return x

class ValueEnsemble(nn.Module):
    """Neural network ensemble for value prediction from molecular fingerprints.

    Attributes
    ----------
    fp_dim : int
        Dimension of the fingerprint vector.
    latent_dim : int
        Dimension of the latent space.
    dropout_rate : float
        Dropout rate for regularization.
    graphNN : GraphModel
        Graph neural network for fingerprint processing.
    layers : nn.Linear
        Linear layer for value prediction.
    """

    def __init__(self, fp_dim, latent_dim, dropout_rate):
        """Initialize the value ensemble.

        Parameters
        ----------
        fp_dim : int
            Dimension of the fingerprint vector.
        latent_dim : int
            Dimension of the latent space.
        dropout_rate : float
            Dropout rate for regularization.
        """
        super(ValueEnsemble, self).__init__()
        self.fp_dim = fp_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.graphNN = GraphModel(input_dim=fp_dim, feature_dim=latent_dim, hidden_dim=latent_dim)
        self.layers = nn.Linear(latent_dim, 1, bias=False)

    def forward(self, fps, mask):
        """Forward pass of the value ensemble.

        Parameters
        ----------
        fps : torch.Tensor
            Fingerprint tensor.
        mask : torch.Tensor
            Mask tensor for zeroing out padded values.

        Returns
        -------
        torch.Tensor
            Predicted value tensor.
        """
        x = self.graphNN(fps, mask)
        x = self.layers(x)
        return x

@contextmanager
def time_limit(seconds):
    """Context manager to enforce a time limit on a block of code.

    Parameters
    ----------
    seconds : int
        Time limit in seconds.

    Raises
    ------
    TimeoutException
        If the time limit is exceeded.
    """
    def signal_handler(signum, frame):
        raise TimeoutException
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def smiles_to_fp(s, fp_dim=2048, pack=False):
    """Convert a SMILES string to a fingerprint vector.

    Parameters
    ----------
    s : str
        SMILES string.
    fp_dim : int, optional
        Dimension of the fingerprint vector. Default is 2048.
    pack : bool, optional
        Whether to pack the bits. Default is False.

    Returns
    -------
    numpy.ndarray
        Fingerprint vector as a boolean array.
    """
    mol = Chem.MolFromSmiles(s)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool_)
    arr[onbits] = 1
    if pack:
        arr = np.packbits(arr)
    return arr

def batch_smiles_to_fp(s_list, fp_dim=2048):
    """Convert a list of SMILES strings to a batch of fingerprint vectors.

    Parameters
    ----------
    s_list : list of str
        List of SMILES strings.
    fp_dim : int, optional
        Dimension of the fingerprint vector. Default is 2048.

    Returns
    -------
    numpy.ndarray
        Batch of fingerprint vectors.
    """
    fps = []
    for s in s_list:
        fps.append(smiles_to_fp(s, fp_dim))
    fps = np.array(fps)
    assert fps.shape[0] == len(s_list) and fps.shape[1] == fp_dim
    return fps

class MinMaxStats(object):
    """Utility class for tracking and normalizing values using min-max scaling.

    Attributes
    ----------
    maximum : float
        Maximum observed value.
    minimum : float
        Minimum observed value.
    """

    def __init__(self, min_value_bound=None, max_value_bound=None):
        """Initialize the min-max stats tracker.

        Parameters
        ----------
        min_value_bound : float, optional
            Initial minimum value bound. Default is -inf.
        max_value_bound : float, optional
            Initial maximum value bound. Default is inf.
        """
        self.maximum = min_value_bound if min_value_bound else -float('inf')
        self.minimum = max_value_bound if max_value_bound else float('inf')

    def update(self, value):
        """Update the min and max values with a new value.

        Parameters
        ----------
        value : float
            New value to consider.
        """
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        """Normalize a value using min-max scaling.

        Parameters
        ----------
        value : float or numpy.ndarray
            Value to normalize.

        Returns
        -------
        float or numpy.ndarray
            Normalized value.
        """
        if self.maximum > self.minimum:
            return (np.array(value) - self.minimum) / (self.maximum - self.minimum)
        return value

def prepare_starting_molecules_natural(path):
    """Prepare a list of starting molecules from a file.

    Parameters
    ----------
    path : str
        Path to the input file.

    Returns
    -------
    list
        List of SMILES strings.
    """
    fr = open(path, 'r')
    data = []
    for line in fr.readlines():
        line = line.strip().split('\t')
        data.append(line[-1])
    return data

def prepare_value(model_f, gpu=None):
    """Load and prepare a value prediction model.

    Parameters
    ----------
    model_f : str
        Path to the model file.
    gpu : int, optional
        GPU device index. If -1, use CPU. Default is None.

    Returns
    -------
    ValueEnsemble
        Loaded and prepared value prediction model.
    """
    if gpu == -1:
        device = 'cpu'
    else:
        device = 'cuda:' + str(gpu)
    model = ValueEnsemble(2048, 128, 0.1).to(device)
    model.load_state_dict(torch.load(model_f, map_location=device))
    model.eval()
    return model

def value_fn(model, mols, device):
    """Predict the value of a list of molecules using the model.

    Parameters
    ----------
    model : ValueEnsemble
        Value prediction model.
    mols : list of str
        List of SMILES strings.
    device : str
        Device to run the model on.

    Returns
    -------
    float
        Predicted value.
    """
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
    """Node class for Monte Carlo Tree Search (MCTS).

    Attributes
    ----------
    state : any
        State represented by the node.
    cost : float
        Cost to reach this node.
    h : float
        Heuristic value of the node.
    prior : float
        Prior probability of the node.
    visited_time : int
        Number of times the node has been visited.
    is_expanded : bool
        Whether the node has been expanded.
    template : any
        Reaction template associated with the node.
    action_mol : any
        Molecule resulting from the action.
    fmove : float
        Forward move value.
    reaction : any
        Reaction associated with the node.
    parent : Node, optional
        Parent node.
    cpuct : float
        CPUCT (Constant of exploration) value.
    reaction_solution : any
        Reaction solution associated with the node.
    children : list
        List of child nodes.
    child_illegal : numpy.ndarray
        Array indicating illegal child actions.
    g : float
        Cost from the root to this node.
    depth : int
        Depth of the node in the tree.
    f : float
        Total cost (g + h) of the node.
    f_mean_path : list
        List of mean path values.
    """

    def __init__(self, state, h, prior, reaction_solution, cost=0, action_mol=None, fmove=0,
                 reaction=None, template=None, parent=None, cpuct=1.5):
        """Initialize the node.

        Parameters
        ----------
        state : any
            State represented by the node.
        h : float
            Heuristic value of the node.
        prior : float
            Prior probability of the node.
        reaction_solution : any
            Reaction solution associated with the node.
        cost : float, optional
            Cost to reach this node. Default is 0.
        action_mol : any, optional
            Molecule resulting from the action. Default is None.
        fmove : float, optional
            Forward move value. Default is 0.
        reaction : any, optional
            Reaction associated with the node. Default is None.
        template : any, optional
            Reaction template associated with the node. Default is None.
        parent : Node, optional
            Parent node. Default is None.
        cpuct : float, optional
            CPUCT value. Default is 1.5.
        """
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
        self.reaction_solution = reaction_solution
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
        """Get the visit counts of all child nodes.

        Returns
        -------
        numpy.ndarray
            Array of visit counts.
        """
        N = [child.visited_time for child in self.children]
        return np.array(N)

    def child_p(self):
        """Get the prior probabilities of all child nodes.

        Returns
        -------
        numpy.ndarray
            Array of prior probabilities.
        """
        prior = [child.prior for child in self.children]
        return np.array(prior)

    def child_U(self):
        """Calculate the UCB (Upper Confidence Bound) values for all child nodes.

        Returns
        -------
        numpy.ndarray
            Array of UCB values.
        """
        child_Ns = self.child_N() + 1
        prior = self.child_p()
        child_Us = self.cpuct * np.sqrt(self.visited_time) * prior / child_Ns
        return child_Us

    def child_Q(self, min_max_stats):
        """Calculate the Q values for all child nodes.

        Parameters
        ----------
        min_max_stats : MinMaxStats
            Min-max statistics for normalization.

        Returns
        -------
        numpy.ndarray
            Array of Q values.
        """
        child_Qs = []
        for child in self.children:
            if len(child.f_mean_path) == 0:
                child_Qs.append(0.0)
            else:
                child_Qs.append(1 - np.mean(min_max_stats.normalize(child.f_mean_path)))
        return np.array(child_Qs)

    def select_child(self, min_max_stats):
        """Select the best child node based on Q + U values.

        Parameters
        ----------
        min_max_stats : MinMaxStats
            Min-max statistics for normalization.

        Returns
        -------
        int
            Index of the best child node.
        """
        action_score = self.child_Q(min_max_stats) + self.child_U() - self.child_illegal
        best_move = np.argmax(action_score)
        return best_move

def gather(dataset, simulations, cpuct, times):
    """Gather and aggregate results from multiple simulation files.

    Parameters
    ----------
    dataset : str
        Name of the dataset.
    simulations : int
        Number of simulations.
    cpuct : float
        CPUCT value used in the simulations.
    times : int
        Number of times the simulations were run.

    Returns
    -------
    dict
        Aggregated results.
    """
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
