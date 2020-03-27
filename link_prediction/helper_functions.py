import random


def split_data(edge_list, percent):
    """ Splits the edge list into two parts for train/test with 1-percent/percent ratio

    :param edge_list: list of edges
    :param percent: percentage with value from 0 to 1
    :type percent: float
    :return: list of train edges, and list of test edges
    :rtype: list, list
    """
    random.seed(350)
    indexes = range(len(edge_list))
    test_indexes = set(random.sample(indexes, len(indexes) * percent))  # removing percent edges from test data
    train_indexes = set(indexes).difference(test_indexes)
    test_list = [edge_list[i] for i in test_indexes]
    train_list = [edge_list[i] for i in train_indexes]
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
