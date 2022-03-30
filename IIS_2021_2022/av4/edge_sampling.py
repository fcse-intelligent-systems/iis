import random
import networkx as nx


def generate_positive_samples(graph, number):
    """Generates positive samples for links, i.e. links that exist in the graph

    :param graph: graph
    :type graph: networkx.Graph
    :param number: number of positive samples
    :type number: int
    :return: list of positive edges
    """
    random.seed(350)
    nodes = set(graph.nodes())
    result = []
    while number > 0:
        v1 = random.sample(nodes, 1)[0]
        v2 = random.sample(set(graph[v1]), 1)[0]
        if graph.degree([v1])[v1] < 3 or graph.degree([v2])[v2] < 3:
            continue
        result.append([v1, v2])
        number -= 1

    return result


def generate_negative_samples(graph, number):
    """Generates negative samples for links, i.e. links that do not exist

    :param graph: graph
    :type graph: networkx.Graph
    :param number: number of negative samples
    :type number: int
    :return: list of negative edges
    """
    random.seed(350)
    nodes = set(graph.nodes())
    result = []
    while number > 0:
        v1 = random.sample(nodes, 1)[0]
        not_neighbors = nodes.difference(set(graph[v1]))
        v2 = random.sample(not_neighbors, 1)[0]
        if [v1, v2] in result or [v2, v1] in result:
            continue
        result.append([v1, v2])
        number -= 1

    return result


if __name__ == '__main__':
    graph = nx.read_edgelist('../av1/Wiki-Vote.txt')

    print(generate_positive_samples(graph, 10))

    print(generate_negative_samples(graph, 10))
