"""Assortative graph generation."""
import numpy as np
import networkx as nx
from tqdm import tqdm
from itertools import product
from ..utils import sample_edges


def assortative_graph(n, p, cutoff=0.9, max_iterations=None):
    """Generate an assortative graph.

    Return a graph with the same degree distribution of an ER(n, p) graph (Poisson) but assortativity at least `cutoff`.
    The graph will always be connected.

    Args:
        n: Number of nodes in the graph
        p: Erdos Renyi parameter of the initial graph
        cutoff: The graph will have assortativity at least this value
        max_iterations: The graph will be rewired `max_iterations` times. If it does not have assortativity of at least
            `cutoff` then the process is started again.

    Returns: A networkx graph.
    """

    while True:
        G = nx.erdos_renyi_graph(n, p)
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(n, p)
        G = XBS(G, max_iterations=max_iterations, cutoff=cutoff)
        assortivity = nx.degree_assortativity_coefficient(G)
        if assortivity >= cutoff:
            break
        else:
            print('Failed', assortivity)
    return G


def XBS(G, max_iterations=None, cutoff=None, remain_connected=True, increase=True):
    """Xulvi-Brunet & Sokolov Algorithm.

    Implementation of Xulvi-Brunet & Sokolov Algorithm similar to described
    here: http://networksciencebook.com/chapter/7#correlated-networks.
    """
    if max_iterations is None:
        max_iterations = 10**10
        tqdm_total = 0
    else:
        tqdm_total = max_iterations

    for i in tqdm(range(max_iterations), total=tqdm_total, disable=True):
        G_before = G.copy()
        links = _link_selection(G)  # select non adjacent edges
        _rewire(G, links, increase)  # rewire these edges

        # undo rewire if it disconnects the graph
        if remain_connected and not nx.is_connected(G):
            G = G_before

        # finish early if above required cutoff
        if (cutoff is not None) and (i % 1000) and (nx.degree_assortativity_coefficient(G) > cutoff):
            break

    return G


def _link_selection(G):
    """Select two edges."""
    links = sample_edges(G, 2)
    while not _links_valid(G, links):
        links = sample_edges(G, 2)
    return links


def _links_valid(G, links):
    """Edges are valid if they do not share an end point or if edges exist between them."""
    if len(set(links[0] + links[1])) != 4:
        return False
    elif True in [edge in G.edges for edge in set(product(links[0], links[1]))]:
        return False
    else:
        return True


def _rewire(G, links, increase):
    """Rewire in a way that does not decrease assortativity."""
    degrees = []
    for key, value in dict(G.degree(links[0] + links[1])).items():
        degrees.append([key, value])
    degrees = np.array(degrees)
    degrees = degrees[degrees[:, 1].argsort()]
    if increase:
        new_links = [(degrees[0][0], degrees[1][0]), (degrees[2][0], degrees[3][0])]
    else:
        new_links = [(degrees[0][0], degrees[3][0]), (degrees[1][0], degrees[2][0])]
    G.remove_edges_from(links)
    G.add_edges_from(new_links)
