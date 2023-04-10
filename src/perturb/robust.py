"""Robust perturbation strategy."""
from ..utils import sparse_norm, sample_edges, sample_nodes
import numpy as np
import networkx as nx
from itertools import product
import scipy.sparse as sp
from scipy.sparse.csgraph import laplacian


def perturb_robust(G, budget: int, samples=None):
    """Perturb to greedily minimize Laplacian distance under one-norm using sampling.

    Args:
        G: The graph to perturb
        budget: The number of edges to flip
        samples: How many edges to sample at each stage of the perturbation.

    Returns:
        The perturbed graph

    """
    Gp = sp.lil_matrix(nx.adj_matrix(G), dtype=np.float32)
    L = nx.normalized_laplacian_matrix(G)
    edges_perturbed = set()  # stops an edge being added then removed
    for _ in range(budget):
        edges = sample_edges(G, samples)
        edges_to_consider = list(edges - edges_perturbed)  # dont consider an edge if its already being perturbed.
        best_edge_idx = np.argmin([one_laplacian_distance(Gp, L, edge) for edge in edges_to_consider])
        best_edge = edges_to_consider[best_edge_idx]
        edges_perturbed.add(best_edge)
        Gp = flip_edge(Gp, best_edge)
    return nx.from_scipy_sparse_matrix(Gp)


def sample_edges(G: nx.Graph, samples: int) -> set:
    """ Returns `samples` edges split evenly between edges which appear in the graph and edges that do not.

    Args:
        G: The input graph
        samples: The total number of edges to sample. If samples is None then consider all possible edges.

    Returns:
        A set of edges (tuples)
    """
    if samples is None:
        edges = set([(u, v) for (u, v) in product(G.nodes(), G.nodes()) if u < v])
    else:
        edges_in_graph = sample_edges_in_graph(G, int(samples/2))
        edges_not_in_graph = sample_edges_not_in_graph(G, samples-len(edges_in_graph))
        edges = edges_in_graph.union(edges_not_in_graph)
    return edges


def sample_edges_in_graph(G: nx.Graph, samples: int) -> set:
    """ Return a set of `samples` edges from the graph G

    Args:
        G: The input graph
        samples: number of edges to sample

    Returns:
        A set of edges (tuple) which exist in graph G

    """
    if G.number_of_edges() < samples:
        edges = list(G.edges())
    else:
        edges = list(G.edges())
        edge_idxs = np.random.choice(range(len(edges)), samples, replace=False)
        edges = [edges[idx] for idx in edge_idxs]
    return set(edges)


def sample_edges_not_in_graph(G: nx.Graph, samples: int) -> set:
    """ Return a set of `samples` edges which do not appear in the graph G

        Args:
            G: Input graph to sample not edges from
            samples: Number of edges to sample

        Returns:
            A set of edges (tuple) which are not in the graph G

    """
    nodes = list(G.nodes())
    edges = set()
    while len(edges) < samples:
        u, v = np.random.choice(nodes, 2, replace=False)
        u, v = min(u, v), max(u, v)
        if not G.has_edge(u, v):
            edges.add((u, v))
    return edges


def one_laplacian_distance(Gp: sp.lil_matrix, L: sp.csr_matrix, edge: tuple) -> float:
    """One-norm Laplacian distance between Laplacian of Gp after `edge` is flipped and L.

    Args:
        Gp: A graph in sparse format
        L: A normalised Laplacian matrix of an unperturbed graph
        edge: An edge that will be flipped in Gp before measuring the Laplacian distance under the 1-norm.

    Returns:
        || L - laplacian(Gp - flip) ||_1.

    """
    flip_edge(Gp, edge)
    Lp = laplacian(Gp, normed=True)
    flip_edge(Gp, edge)
    return sparse_norm(L - Lp, ord=1)


def flip_edge(A: sp.lil_matrix, edge: tuple) -> sp.lil_matrix:
    """FLips the entry of binary adjacency matrix A."""
    u, v = edge
    A[u, v] = 1 - A[u, v]
    A[v, u] = 1 - A[v, u]
    return A
