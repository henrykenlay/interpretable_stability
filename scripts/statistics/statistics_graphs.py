"""Computer graph statistics."""
import argparse
import os

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import sparse_norm
from scripts.utils import parse_seeds, load_graph, load_perturbed_graph
from src.bound import removal_bound, addition_bound, remain_bound
from src.bound import node_with_largest_Eu, boundvalid, theorem_bound

parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=str, default='1',
                    help='Number of experiments (100) or a range (100-200).')
parser.add_argument('--graph_string', type=str, default='100_BA_3',
                    help='String representing the graph')
parser.add_argument('--attack', type=str, default='robust',
                    help='Type of attack.')
parser.add_argument('--proportion_perturb', type=float, default=0.1,
                    help='Budget as a proportion of total edges.')
args = parser.parse_args()

graph_statistics_dir = os.path.join('data', 'graph_statistics', args.graph_string)
os.makedirs(graph_statistics_dir, exist_ok=True)

for seed in tqdm(parse_seeds(args.seeds)):
    data = []

    # unperturbed graph
    A = load_graph(args.graph_string, seed)
    G = nx.from_scipy_sparse_matrix(A)
    L = nx.normalized_laplacian_matrix(G)
    G_spectrum = np.sort(np.linalg.eigvalsh(L.todense()))

    # get modified adj
    try:
        Ap = load_perturbed_graph(args.graph_string, args.proportion_perturb, args.attack, seed)
    except:  # happens for some pgd attacks that couldnt find a solution
        print(f'Failed seed {seed} for graph {args.graph_string} attack {args.attack} and budget {args.proportion_perturb}')
        continue

    # Laplacian distances
    Gp = nx.from_scipy_sparse_matrix(Ap)
    Lp = nx.normalized_laplacian_matrix(Gp)
    data.append(['lap_1dist', sparse_norm(L-Lp, ord=1)])
    data.append(['lap_2dist', sparse_norm(L-Lp, ord=2)])
    data.append(['lap_Fdist', sparse_norm(L-Lp, ord='fro')])

    # spectrum distance
    Gp_spectrum = np.sort(np.linalg.eigvalsh(Lp.todense()))
    data.append(['spectrum_l1_dist', np.linalg.norm(G_spectrum-Gp_spectrum, ord=1)])
    data.append(['spectrum_l2_dist', np.linalg.norm(G_spectrum-Gp_spectrum, ord=2)])

    # connectivity
    assert nx.is_connected(G)
    data.append(['number_connected_components', nx.number_connected_components(Gp)])
    data.append(['isolates', len(list(nx.isolates(Gp)))])

    # proportion of nodes effected
    E = Lp - L
    nodes_effected = (~np.isclose(E.sum(0), 0)).sum()
    data.append(['prop_nodes_effected', nodes_effected/G.number_of_nodes()])

    # related to the bound
    data.append(['is_valid', boundvalid(G, Gp)])
    Eu_node = node_with_largest_Eu(E)
    Eu_removed, Eu_removed_bound = removal_bound(G, Gp, E, Eu_node)
    Eu_added, Eu_added_bound = addition_bound(G, Gp, E, Eu_node)
    Eu_remain, Eu_remain_bound = remain_bound(G, Gp, E, Eu_node)
    assert Eu_removed <= Eu_removed_bound or np.isclose(Eu_removed, Eu_removed_bound, atol=10e-6), print(Eu_removed, Eu_removed_bound)
    assert Eu_added <= Eu_added_bound or np.isclose(Eu_added, Eu_added_bound, atol=10e-6), print(Eu_added, Eu_added_bound)
    if Eu_remain_bound is not None:
        assert Eu_remain <= Eu_remain_bound or np.isclose(Eu_remain, Eu_remain_bound, atol=10e-6), print(Eu_remain, Eu_remain_bound)

    data.append(['Eu_removed', Eu_removed])
    data.append(['Eu_removed_bound', Eu_removed_bound])
    data.append(['Eu_added', Eu_added])
    data.append(['Eu_added_bound', Eu_added_bound])
    data.append(['Eu_remain', Eu_remain])
    data.append(['Eu_remain_bound', Eu_remain_bound])

    # theorem 1
    theorem = theorem_bound(G, Gp, E)
    if_dont_use_different_u = Eu_removed_bound + Eu_added_bound
    if Eu_remain_bound is not None:
        if_dont_use_different_u += Eu_remain_bound
    assert if_dont_use_different_u <= theorem or np.isclose(if_dont_use_different_u, theorem, atol=10e-6), print(if_dont_use_different_u, theorem)
    data.append(['theorem', theorem])
    
    # spread of degree
    data.append(['degree_std', np.std([G.degree(i) for i in G.nodes()])])

    # save results
    df = pd.DataFrame(data, columns=['metric', 'value'])
    contains_nans = df.isnull().any().any()
    #assert not contains_nans, print(df)
    df.to_csv(os.path.join(graph_statistics_dir, f'{args.proportion_perturb}_{args.attack}_{seed}.csv'))
