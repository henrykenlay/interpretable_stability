from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import scipy.sparse as sp
import numpy as np


def get_tudataset(name):
    """Returns a list of adj matrices and a list of feature matrices."""
    # file names
    resp = urlopen(f'https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip')
    zipfile = ZipFile(BytesIO(resp.read()))
    features_fname=f'{name}/{name}_node_attributes.txt'
    graphs_fname=f'{name}/{name}_A.txt'
    graph_indicator_fname=f'{name}/{name}_graph_indicator.txt'

    # map graph_id to set of nodes
    graphs = {}
    for node, line in enumerate(zipfile.open(graph_indicator_fname).readlines()):
        graph_id = int(line.decode('utf-8').strip('\n'))-1
        if graph_id in graphs:
            graphs[graph_id].add(node)
        else:
            graphs[graph_id] = {node}


    # construct adjacent matrices
    number_of_graphs = len(graphs)
    adj_matrices = [sp.lil_matrix((len(graphs[i]), len(graphs[i])), dtype=np.float32) for i in range(number_of_graphs)]
    for line in zipfile.open(graphs_fname).readlines():
        u, v = line.decode('utf-8').strip('\n').split(',')
        u, v = int(u)-1, int(v)-1

        for graph_id in graphs:
            if u in graphs[graph_id]:
                u = u - min(graphs[graph_id])
                v = v - min(graphs[graph_id])
                adj_matrices[graph_id][u, v] = 1
                adj_matrices[graph_id][v, u] = 1

    # calculate feature vector dim
    for line in zipfile.open(features_fname).readlines():
        feature_dim = len(line.decode('utf-8').strip('\n').split(','))
        break

    # construct feature matrices
    feature_matrices = [np.zeros((len(graphs[i]), feature_dim), dtype=np.float32) for i in range(number_of_graphs)]
    for node, line in enumerate(zipfile.open(features_fname).readlines()):
        features = np.array([float(x) for x in line.decode('utf-8').strip('\n').split(',')])
        for graph_id in graphs:
            if node in graphs[graph_id]:
                node = node - min(graphs[graph_id])
                feature_matrices[graph_id][node] = features

    return adj_matrices, feature_matrices
