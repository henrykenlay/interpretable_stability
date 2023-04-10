"""Random perturbations."""
import numpy as np
from ..utils import sample_edges, sample_nodes
from .utils import contains_isolates


def random_rewire(G, rewires, max_attempts=10, max_attempts_per_rewire=100):
    """Rewire of G without isolated nodes.

    Parameters
    ----------
    G: unperturbed graph
    rewires: number of rewiring operations
    max_attempts: attempt to perturb the graph this number of times
    max_attempts_per_rewire: at each step of the perturbation, try to rewire this many times

    Returns
    -------
    Gp: a perturbed graph of None if a perturbed graph could not be found.
    """
    for _ in range(max_attempts):
        Gp = _rewire(G, rewires, max_attempts_per_rewire)
        if Gp is not None:
            break
    return Gp


def _rewire(G, rewires, max_attempts_per_rewire):
    Gp = G.copy()
    flipped = set()  # keep track of already flipped edges
    for _ in range(rewires):
        success = False
        for _ in range(max_attempts_per_rewire):

            # randomly select 2 edges
            Gp_edges = list(Gp.edges())
            edges = np.random.choice(Gp.number_of_edges(), 2, replace=False)
            edges = Gp_edges[edges[0]], Gp_edges[edges[1]]
            u, v, uprime, vprime = *edges[0], *edges[1]

            # check if they share endpoints
            if len(set((u, v, uprime, vprime))) < 4:
                continue

            # check if edges inbetween disqualify the operation
            if Gp.has_edge(u, vprime) or Gp.has_edge(u, uprime) or Gp.has_edge(v, vprime) or Gp.has_edge(v, uprime):
                continue

            # check edges have not already been flipped
            edge_remove_1 = (min(u, v), max(u, v))
            edge_remove_2 = (min(uprime, vprime), max(uprime, vprime))
            edge_add_1 = (min(u, uprime), max(u, uprime))
            edge_add_2 = (min(v, vprime), max(v, vprime))
            to_flip = set([edge_remove_1, edge_remove_2, edge_add_1, edge_add_2])
            if len(to_flip.intersection(flipped)) > 0:
                continue

            # check perturbation doesn't isolate a node
            G_temp = Gp.copy()
            G_temp.remove_edges_from(edges)
            G_temp.add_edges_from([(u, uprime), (v, vprime)])
            if contains_isolates(G_temp):
                continue

            # update perturbed graph and go to next rewire
            Gp = G_temp.copy()
            flipped = flipped.union(to_flip)
            success = True
            break

        if not success:
            return None

    return Gp


def randomly_perturb(G, add=0, remove=0):
    """Add and remove edges uniformly at random."""
    while True:
        Gp = G.copy()
        edges_to_remove = sample_edges(Gp, remove)
        edges_to_add = []
        while len(edges_to_add) < add:
            edge = sample_nodes(Gp, 2)
            edge = (min(edge), max(edge))
            if edge not in G.edges and edge not in edges_to_add:
                edges_to_add.append(edge)
        Gp.remove_edges_from(edges_to_remove)
        Gp.add_edges_from(edges_to_add)
        if not contains_isolates(Gp):
            break
    return Gp

