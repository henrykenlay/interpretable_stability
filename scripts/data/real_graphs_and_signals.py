"""Generates graphs from a random graph model"""
import argparse

import numpy as np
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm
import scipy.sparse as sp

from src.gsp import _normalise_signal
from src.tudataset import get_tudataset
from scripts.utils import save_graph, save_clean_signal, parse_seeds

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=str, default='100',
                    help='Number of experiments (100)')
parser.add_argument('--dataset', type=str, help='Dataset to use.')
parser.add_argument('--n_min', type=int)
parser.add_argument('--n_max', type=int)
parser.add_argument('--channel_index', type=int)
args = parser.parse_args()

# load data
adj_matrices, feature_matrices = get_tudataset(args.dataset)

# filter data
datapoints = []
for adj_matrix, features in zip(adj_matrices, feature_matrices):
    number_of_nodes = adj_matrix.shape[0]
    # filter by size
    if args.n_min <= number_of_nodes <= args.n_max:

        # filter out graphs where the features don't vary or are zero (assumed to be an error in the data)
        # also filter out graphs that are not connected
        constant = np.isclose(np.std(features[:, args.channel_index]), 0)
        contains_zero = np.isclose(features[:, args.channel_index], 0).any()
        connected = connected_components(adj_matrix, directed=False, return_labels=False) == 1
        if connected and not constant and not contains_zero:
            features = _normalise_signal(features[:, args.channel_index])
            datapoints.append((adj_matrix, features))

    # break early if we have enough data
    if len(datapoints) == args.seeds:
        break

# save data
for seed in tqdm(parse_seeds(args.seeds)):
    graph = datapoints[seed][0]
    graph = sp.csr_matrix(graph)
    signal = datapoints[seed][1]
    assert graph.shape[0] == signal.shape[0]

    save_graph(graph, args.dataset, seed)
    save_clean_signal(signal, args.dataset, seed)