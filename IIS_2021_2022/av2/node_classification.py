import pandas as pd
import networkx as nx
from gem.embedding.lap import LaplacianEigenmaps
from gem.embedding.node2vec import node2vec
from graph_embeddings import save_embeddings
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


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


if __name__ == '__main__':
    graph, node_features, feature_names = read_graph()
    nodes = graph.nodes()

    #  Laplacian Eigenmaps
    le = LaplacianEigenmaps(d=50)
    embeddings_le, _ = le.learn_embedding(graph=graph, edge_f=None, is_weighted=False, no_python=True)
    save_embeddings(file_path='../data/cora_laplacian_eigenmaps.emb', embs=embeddings_le, nodes=nodes)

    #  Node2Vec
    node2vec_obj = node2vec(d=50, max_iter=1, walk_len=10, num_walks=5, ret_p=1, inout_p=1)
    embeddings_nv, _ = node2vec_obj.learn_embedding(graph=graph, edge_f=None, is_weighted=False, no_python=True)

    embeddings_df = pd.DataFrame(embeddings_le, index=nodes)
    subject_df = pd.DataFrame(node_features['subject'])

    encoder = OrdinalEncoder()
    encodings = encoder.fit_transform(subject_df)
    subject_df = pd.DataFrame(encodings, columns=['subject'], index=nodes)

    train_x, test_x, train_y, test_y = train_test_split(embeddings_df, subject_df, test_size=0.1, stratify=subject_df)

    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(train_x, train_y)

    pred_y = classifier.predict(test_x)

    accuracy = accuracy_score(test_y, pred_y)

    print(f'Accuracy: {accuracy}')
