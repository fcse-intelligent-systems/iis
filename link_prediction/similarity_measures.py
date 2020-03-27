import numpy as np
from igraph import *


def similarity(graph, i, j, method):
    """Node-based topological similarity metrics

    :param graph: graph
    :type graph: igraph.Graph
    :param i: id for node i
    :param j: id for node j
    :param method: name of the similarity method; one of:
                   ['ommon_neighbors', 'jaccard', 'adamic_adar', 'preferential_attachment']
    :return: similarity value for the node i and node j
    :rtype: float
    """
    if method == "common_neighbors":
        return len(set(graph.neighbors(i)).intersection(set(graph.neighbors(j))))
    elif method == "jaccard":
        return len(set(graph.neighbors(i)).intersection(set(graph.neighbors(j)))) / float(
            len(set(graph.neighbors(i)).union(set(graph.neighbors(j)))))
    elif method == "adamic_adar":
        return sum(
            [1.0 / math.log(graph.degree(v)) for v in set(graph.neighbors(i)).intersection(set(graph.neighbors(j)))])
    elif method == "preferential_attachment":
        return graph.degree(i) * graph.degree(j)
