import networkx as nx


def contains_isolates(G):
    """Determine if a graph contains a node with degree 0."""
    if nx.number_of_isolates(G) > 0:
        return True
    else:
        return False
