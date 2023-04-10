"""Utilities."""
import numpy as np
import scipy.sparse as sp


def sample_nodes(G, num_nodes):
    """Uniformly sample (without replacement) `num_nodes` nodes from `G`."""
    nodes = G.nodes()
    return list(np.random.choice(list(nodes), num_nodes, replace=False))


def sample_edges(G, num_edges):
    """Uniformly sample (without replacement) `num_edges` edges from `G`."""
    edges = list(G.edges)
    return [edges[i] for i in np.random.choice(range(len(edges)), num_edges, replace=False)]


def sparse_norm(A, ord=2):
    """Like scipy.sparse.lingalg.norm but with the 2-norm and max norm implemented.
    If `ord=2` or `ord='max'` a grapht implementation is used, otherwise scipy.sparse.lingalg.norm is used.
    """
    if not sp.issparse(A):
        raise TypeError('input must be sparse')
    if ord == 2:
        return sparse_2norm(A)
    elif ord == 'max':
        return sparse_maxnorm(A)
    else:
        return sp.linalg.norm(A, ord=ord)


def sparse_2norm(A):
    """Returns the matrix 2-norm of a sparse matrix `A`."""
    if not sp.issparse(A):
        raise TypeError('input must be sparse')
    return sp.linalg.svds(A, k=1, which='LM', return_singular_vectors=False)[0]


def sparse_maxnorm(A):
    """Returns the max |A_ij| for a sparse matrix `A`."""
    if not sp.issparse(A):
        raise TypeError('input must be sparse')
    return max(-A.min(), A.max())
