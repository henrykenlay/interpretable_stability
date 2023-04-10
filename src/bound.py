"""Functions to evaluate the bound."""
import numpy as np


def node_with_largest_Eu(E):
    """Argmax_u ||Eu||_1."""
    return np.argmax(np.abs(np.array(E.todense())).sum(0))


def removal_bound(G, Gp, E, node):
    """Part of the bound related to edge removal."""
    # neighbours where there was a removal
    vs = []
    for v in G.neighbors(node):
        if (node, v) not in Gp.edges():
            vs.append(v)

    # contribution to Eu
    Eu_removed = 0
    for v in vs:
        Eu_removed += np.abs(E[node, v])

    # bound on contribution to Eu
    delta = np.min([G.degree(v) for v in G.neighbors(node)])
    Eu_bound = len(vs)/np.sqrt(delta*G.degree(node))
    return Eu_removed, Eu_bound


def addition_bound(G, Gp, E, node):
    """Part of the bound related to edge addition."""
    # new neighbours after an addition
    vs = []
    for v in Gp.neighbors(node):
        if (node, v) not in G.edges():
            vs.append(v)

    # contribution to Eu
    Eu_added = 0
    for v in vs:
        Eu_added += np.abs(E[node, v])

    # bound on contribution to Eu
    delta_prime = np.min([Gp.degree(v) for v in Gp.neighbors(node)])
    Eu_bound = len(vs)/np.sqrt(delta_prime*Gp.degree(node))
    return Eu_added, Eu_bound


def remain_bound(G, Gp, E, node):
    """Part of the bound related to edges that remain."""
    # neighbours that remain
    vs = []
    for v in G.neighbors(node):
        if (node, v) in Gp.edges():
            vs.append(v)

    # contribution to Eu
    Eu_remain = 0
    for v in vs:
        Eu_remain += np.abs(E[node, v])

    # bound on contribution to Eu
    alphau = compute_alphau(G, Gp, node)
    for v in G.neighbors(node):
        alphau = max(alphau, compute_alphau(G, Gp, v))
    delta = np.min([G.degree(v) for v in G.neighbors(node)])

    if alphau >= 1:  # lemma does not apply in this case.
        Eu_bound = None
    else:
        first_part = alphau / (1-alphau)
        second_part = len(vs) / np.sqrt(G.degree(node)*delta)
        Eu_bound = first_part * second_part
    return Eu_remain, Eu_bound


def boundvalid(G, Gp):
    """If Lemma 2 is valid for all nodes."""
    alpha = np.max([compute_alphau(G, Gp, u) for u in G.nodes()])
    if alpha >= 1:
        return False
    else:
        return True


def compute_alphau(G, Gp, u):
    """Relative change around node u."""
    delta = np.abs(G.degree(u) - Gp.degree(u))
    if delta == 0:
        return 0
    else:
        return delta/G.degree(u)

def theorem_bound(G, Gp, E):
    max_rhs = 0.0
    for node in G.nodes():
        _, remove = removal_bound(G, Gp, E, node)
        _, add = addition_bound(G, Gp, E, node)
        _, remain = remain_bound(G, Gp, E, node)
        if remain is None:
            remain = 0.0
        rhs = remove + add + remain
        max_rhs = max(max_rhs, rhs)
    return max_rhs