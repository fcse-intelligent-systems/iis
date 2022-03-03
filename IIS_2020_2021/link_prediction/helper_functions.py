import random
from igraph import Graph
from collections import Counter


def split_data(edge_list, percent):
    """ Splits the edge list into two parts for train/test with 1-percent/percent ratio

    :param edge_list: list of edges
    :param percent: percentage with value from 0 to 1
    :type percent: float
    :return: list of train edges, and list of test edges
    :rtype: list, list
    """
    random.seed(350)
    node_degrees = Counter([es.source for es in edge_list])

    available_edges = [edge for edge in edge_list if node_degrees[edge.source] > 3 and node_degrees[edge.target] > 3]
    other_edges = [edge for edge in edge_list if not node_degrees[edge.source] > 3 and node_degrees[edge.target] > 3]
    indexes = range(len(available_edges))
    test_indexes = set(random.sample(indexes, int(len(indexes) * percent)))  # removing percent edges from test data
    train_indexes = set(indexes).difference(test_indexes)
    test_list = [available_edges[i] for i in test_indexes]
    train_list = [available_edges[i] for i in train_indexes] + other_edges
    return train_list, test_list


def generate_negative_samples(graph, number):
    """Generates negative samples for links, i.e. links that do not exist

    :param graph: graph
    :type graph: igraph.Graph
    :param number: number of negative samples
    :type number: int
    :return: list of negative
    """
    random.seed(350)
    nodes = set(graph.vs.indices)
    result = []
    while number > 0:
        v1 = random.sample(nodes, 1)[0]
        not_neighbors = nodes.difference(graph.neighbors(v1)).difference({v1})
        v2 = random.sample(not_neighbors, 1)[0]
        if [v1, v2] in result or [v2, v1] in result:
            continue
        result.append([v1, v2])
        number -= 1

    return result


if __name__ == '__main__':
    g = Graph.Full(3)
    print(list(map(len, split_data(g.es, 0.2))))
