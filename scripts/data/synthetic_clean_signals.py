"""Generate clean signals"""
import argparse

import networkx as nx
from tqdm import tqdm

from src.gsp import synthetic_signal
from scripts.utils import load_graph, save_clean_signal, parse_seeds

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=str, default='5',
                    help='Number of experiments (100) or a range (100-200).')
parser.add_argument('--graph_string', type=str, default='500_BA_2',
                    help='String representing the graph')
parser.add_argument('--number_of_eigenvectors', type=float, default=0.1,
                    help='Proportion of eigenvectors or absolute number')
args = parser.parse_args()


for seed in tqdm(parse_seeds(args.seeds), desc=args.graph_string):
    A = load_graph(args.graph_string, seed)
    G = nx.from_scipy_sparse_matrix(A)
    clean_signal = synthetic_signal(G, args.number_of_eigenvectors)
    save_clean_signal(clean_signal, args.graph_string, seed)
