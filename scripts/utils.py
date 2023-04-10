import os

import numpy as np
import scipy.sparse as sp
import torch


def save_graph(graph, graph_string, seed):
    graph_directory = os.path.join('data', 'graphs', graph_string)
    os.makedirs(graph_directory, exist_ok=True)
    sp.save_npz(os.path.join(graph_directory, str(seed)), graph)


def load_graph(graph_string, seed):
    return sp.load_npz(os.path.join('data', 'graphs', graph_string, f'{seed}.npz'))


def save_clean_signal(signal, graph_string, seed):
    clean_directory = os.path.join('data', 'clean_signals', graph_string)
    os.makedirs(clean_directory, exist_ok=True)
    np.save(os.path.join(clean_directory, str(seed)), signal)


def load_clean_signal(graph_string, seed):
    return np.load(os.path.join('data', 'clean_signals', graph_string, f'{seed}.npy'))


def save_noisy_signal(signal, snr, graph_string, seed):
    noisy_directory = os.path.join('data', 'noisy_signals', graph_string, str(snr))
    os.makedirs(noisy_directory, exist_ok=True)
    np.save(os.path.join(noisy_directory, str(seed)), signal)


def load_noisy_signal(graph_string, snr, seed):
    return np.load(os.path.join('data', 'noisy_signals', graph_string, str(snr), f'{seed}.npy'))


def save_perturbed_graph(perturbed_graph: sp.spmatrix, graph_string: str, proportion_perturb: float, attack: str, seed: int):
    directory = os.path.join('data', 'perturbed_graphs', graph_string, str(proportion_perturb), attack)
    os.makedirs(directory, exist_ok=True)
    sp.save_npz(os.path.join(directory, str(seed)), perturbed_graph)


def load_perturbed_graph(graph_string, proportion_perturb, attack, seed):
    return sp.load_npz(os.path.join('data', 'perturbed_graphs', graph_string,
                                    str(proportion_perturb), attack, f'{seed}.npz'))


def load_model(graph_string, snr):
    return torch.load(os.path.join('data', 'models', graph_string, str(snr), 'model.pt'))


def generate_graph_string(graph_model, n=None, m=None, p=None, k=None, number_communities=None, cutoff=None, **kwargs):
    """A canonical naming convention."""
    if graph_model == 'BA':
        return f'{n}_BA_{m}'
    elif graph_model == 'ER':
        return f'{n}_ER_{p}'
    elif graph_model == 'WS':
        return f'{n}_WS_{k}_{p}'
    elif graph_model == 'SBM':
        return f'{n}_SBM_{number_communities}'
    elif graph_model == 'Kreg':
        return f'{n}_kreg_{k}'
    elif graph_model == 'KNN':
        return f'{n}_knn_{k}'
    elif graph_model == 'ERA':
        return f'{n}_ERA_{p}_{cutoff}'
    else:
        raise ValueError('Invalid graph type.')


def get_device(gpu):
    """Return device string."""
    if torch.cuda.is_available() and gpu is not None:
        device = torch.device(f'cuda:{gpu}')
    else:
        device = torch.device('cpu')
    return device


def parse_seeds(string):
    """Parse seeds argument."""
    seeds = string.split('-')
    if len(seeds) == 1:
        return range(int(seeds[0]))
    else:
        return range(int(seeds[0]), int(seeds[1]))