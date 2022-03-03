from collections import Counter
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

def plot_graph(g):
    pos = nx.spring_layout(g)
    nx.draw_networkx_nodes(g, pos, node_size=50)
    nx.draw_networkx_edges(g, pos)
    plt.show()

def plot_degree_distribution(node_degrees):
    node_degrees = [d[1] for d in node_degrees]
    counts = Counter(node_degrees)
    degrees = list(counts.keys())
    values = list(counts.values())
    sns.barplot(degrees, values)
    plt.show()


if __name__ == '__main__':
    graph = nx.read_edgelist('Wiki-Vote.txt')

    plot_graph(graph)

    nodes = graph.nodes()
    print(nodes)

    num_nodes = graph.number_of_nodes()
    print(f'Number of nodes: {num_nodes}')

    edges = graph.edges()
    print(edges)

    num_edges = graph.number_of_edges()
    print(f'Number of edges: {num_edges}')

    print(f'Number of connected components: {nx.number_connected_components(graph)}')
    print(f'Average clustering coefficient: {nx.average_clustering(graph)}')

    degree_30 = graph.degree(['30'])
    print(f'Degree of node with id 30: {degree_30}')

    degree = graph.degree(nodes)
    print(degree)

    plot_degree_distribution(degree)

    graph_er = nx.erdos_renyi_graph(50, 0.1)
    plot_graph(graph_er)
    plot_degree_distribution(graph_er.degree(graph_er.nodes()))

    graph_sm = nx.watts_strogatz_graph(50, 3, 0.1)
    plot_graph(graph_sm)
    plot_degree_distribution(graph_sm.degree(graph_sm.nodes()))
