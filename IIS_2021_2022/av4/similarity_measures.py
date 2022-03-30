import networkx as nx
import math


def similarity(graph, i, j, method):
    """Node-based topological similarity metrics

    :param graph: graph
    :type graph: networkx.Graph
    :param i: id for node i
    :param j: id for node j
    :param method: name of the similarity method; one of:
                   ['common_neighbors', 'jaccard', 'adamic_adar', 'preferential_attachment']
    :return: similarity value for the node i and node j
    :rtype: float
    """
    if method == 'common_neighbors':
        return len(set(graph[i]).intersection(set(graph[j])))
    elif method == 'jaccard':
        return len(set(graph[i]).intersection(set(graph[j]))) / float(len(set(graph[i]).union(set(graph[j]))))
    elif method == 'adamic_adar':
        return sum([1.0 / math.log(graph.degree([v])[v]) for v in set(graph[i]).intersection(set(graph[j]))])
    elif method == 'preferential_attachment':
        return graph.degree([i])[i] * graph.degree([j])[j]


if __name__ == '__main__':
    graph = nx.read_edgelist('../av1/Wiki-Vote.txt')

    print(similarity(graph, '30', '1412', 'common_neighbors'))

    print(similarity(graph, '30', '1412', 'jaccard'))

    print(similarity(graph, '30', '1412', 'adamic_adar'))

    print(similarity(graph, '30', '1412', 'preferential_attachment'))
