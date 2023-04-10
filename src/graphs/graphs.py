"""Random graph models."""
import networkx as nx
import pygsp
from .assortative import assortative_graph


def random_connected_graph(graph_model, **kwargs):
    """Return a random connected networkx graph."""
    graph = random_graph(graph_model, **kwargs)
    while not nx.is_connected(graph):
        graph = random_graph(graph_model, **kwargs)
    return graph


def random_graph(graph_model, n=None, m=None, p=None, k=None, cutoff=None, max_iterations=None, **kwargs):
    """Return a random networkx graph using an existing implementation."""
    if graph_model == 'BA':
        return nx.barabasi_albert_graph(n, m)
    elif graph_model == 'ER':
        return nx.erdos_renyi_graph(n, p)
    elif graph_model == 'WS':
        return nx.watts_strogatz_graph(n, k, p)
    elif graph_model == 'Kreg':
        return nx.random_regular_graph(k, n)
    elif graph_model == 'KNN':
        return KNN(n, k)
    elif graph_model == 'ERA':
        return assortative_graph(n, p, cutoff, max_iterations)
    else:
        raise ValueError('Invalid graph type.')


def KNN(n, k):
    """K nearest neighbour graph."""
    graph = pygsp.graphs.Sensor(n, k)
    return nx.from_scipy_sparse_matrix(graph.A)
