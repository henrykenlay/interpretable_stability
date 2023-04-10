import numpy as np
import networkx as nx
import scipy.sparse as sp


def equal_community_sizes(n: int, number_communities: int) -> list[int]:
    """Calculates size of each community so each has size approximately n / number_communities. The size of each
    community won't differ by more than 1.

    Parameters
    ----------
    n: total number of nodes.
    number_communities: number of communities to divide the nodes into.

    Returns
    -------
    A list of integers representing the size of each community.
    """
    communities = [int(n / number_communities) for _ in range(number_communities)]
    for i in range(number_communities):
        if np.sum(communities) == n:
            break
        else:
            communities[i] += 1
    return communities


def sample_sbm_graph(community_sizes: list[int]) -> sp.spmatrix:
    """Return an SBM graph in networkx format.

    The internal connection probability of each community is nc * log(nc) / nc where nc is the average size of the
    communities. The external connection probability is 10 times less.

    Parameters
    ----------
    community_sizes: A list of how large each community is

    Returns
    -------
    A scipy spare array of the adjacency matrix.
    """
    n_community = np.mean(community_sizes)
    p = np.log(n_community) / n_community
    q = p / 10
    matrix = q * np.ones((len(community_sizes), len(community_sizes)))
    np.fill_diagonal(matrix, p)
    graph = nx.generators.community.stochastic_block_model(community_sizes, matrix)
    while not nx.is_connected(graph):
        graph = nx.generators.community.stochastic_block_model(community_sizes, matrix)
    return nx.adj_matrix(graph)


def sample_sbm_signal(community_sizes: list[int], mu: float, std: float) -> np.array:
    """Generates a signal for an SBM graph with two or three communities.

    If the graph has two communities the node signals have means -mu and mu. If the graph has three communities the
    node signals have mean -mu, 0 and mu. All graphs have i.i.d. noise ~N(0, std^2) added to each node signal value.

    Parameters
    ----------
    community_sizes: A list of how many nodes are in each community.
    mu: Mean separation between the node values.
    std: Standard deviation of the centred Gaussian mean added to the node values.

    Returns
    -------
    A 1-D signal for the SBM graph
    """
    if len(community_sizes) == 2:
        signal = np.concatenate([mu*np.ones(community_sizes[0]),
                                 -mu*np.ones(community_sizes[1])])
    elif len(community_sizes) == 3:
        signal = np.concatenate([mu*np.ones(community_sizes[0]),
                                 np.zeros(community_sizes[1]),
                                 -mu*np.ones(community_sizes[2])])
    else:
        raise NotImplementedError('Only designed for 2 or 3 communities')
    signal = signal + np.random.normal(scale=std, size=len(signal))
    return signal
