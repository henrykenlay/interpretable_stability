"""Generates random graphs from a random graph model"""
import argparse

import networkx as nx
from tqdm import tqdm

from src.graphs import random_connected_graph
from scripts.utils import save_graph, generate_graph_string, parse_seeds

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=str, default='5',
                    help='Number of experiments (100) or a range (100-200).')
parser.add_argument('--graph_model', type=str, default='BA', choices=['BA', 'ER', 'WS', 'Kreg', 'KNN', 'ERA'],
                    help='Random graph model to use.')
parser.add_argument('--n', type=int, default=100,
                    help='Number of nodes or size of community (SBM).')
parser.add_argument('--m', type=int, default=2,
                    help='BA parameter.')
parser.add_argument('--p', type=float, default=0.1,
                    help='Used in ER, WS and ERA')
parser.add_argument('--k', type=int, default=2,
                    help='WS, K-regular and KNN parameter')
parser.add_argument('--q', type=float, default=0.01,
                    help='Connectivity between SBM communities')
parser.add_argument('--cutoff', type=float, default=0.9,
                    help='Level of assortativity required from ERA graph')
parser.add_argument('--max_iterations', type=int, default=10000,
                    help='Timeout for ERA graphs')
args = parser.parse_args()

graph_properties = vars(args)
graph_string = generate_graph_string(**graph_properties)

for seed in tqdm(parse_seeds(args.seeds), desc=graph_string):
    graph = random_connected_graph(**graph_properties)
    graph = nx.adj_matrix(graph)
    save_graph(graph, graph_string, seed)
