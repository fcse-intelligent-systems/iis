import pandas as pd
import networkx as nx
from gem.embedding.lap import LaplacianEigenmaps
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def read_graph():
    # Load the graph from edgelist
    edgelist = pd.read_table('../cora/cora.cites',
                             header=None, names=['source', 'target'])
    edgelist['label'] = 'cites'
    graph = nx.from_pandas_edgelist(edgelist, edge_attr='label')
    nx.set_node_attributes(graph, 'paper', 'label')

    # Load the features and subject for the nodes
    feature_names = ['w_{}'.format(ii) for ii in range(1433)]
    column_names = feature_names + ['subject']
    node_data = pd.read_table('../cora/cora.content',
                              header=None, names=column_names)

    return graph, node_data, feature_names


def plot_communities(data, community_groups):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(data[:, 0], data[:, 1], c=community_groups,
               cmap='jet', alpha=0.7)
    ax.set(aspect='equal', xlabel='$X_1$', ylabel='$X_2$',
           title=f'TSNE visualization of clusters obtained by KMeans for CORA dataset')
    plt.show()


subjects_map = {
    'Probabilistic_Methods': 0,
    'Rule_Learning': 1,
    'Neural_Networks': 2,
    'Theory': 3,
    'Reinforcement_Learning': 4,
    'Genetic_Algorithms': 5,
    'Case_Based': 6
}

if __name__ == '__main__':
    graph, node_features, feature_names = read_graph()
    nodes = graph.nodes()

    #  Laplacian Eigenmaps
    le = LaplacianEigenmaps(d=50)
    embeddings_le, _ = le.learn_embedding(graph=graph, edge_f=None, is_weighted=False, no_python=True)

    embeddings_df = pd.DataFrame(embeddings_le, index=nodes)
    subject_df = pd.DataFrame(node_features['subject'])

    kmeans = KMeans(n_clusters=7, random_state=0)
    clusters = kmeans.fit(embeddings_df)

    subject_df['subject'] = subject_df['subject'].apply(lambda x: subjects_map[x])
    subject_df['predicted'] = clusters.labels_

    #   gt 0 0 1 1 0
    # pred 1 1 0 0 1

    print(adjusted_mutual_info_score(subject_df['subject'].values, subject_df['predicted'].values))

    tsne = TSNE(n_components=2)
    X_reduced = tsne.fit_transform(embeddings_df)

    plot_communities(X_reduced, subject_df['subject'].values)
    plot_communities(X_reduced, subject_df['predicted'].values)
