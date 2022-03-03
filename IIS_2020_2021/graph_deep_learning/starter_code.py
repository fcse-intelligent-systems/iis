import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn import feature_extraction, model_selection


def read_graph():
    # Load the graph from edgelist
    edgelist = pd.read_table("data/cora/cora.cites",
                             header=None, names=["source", "target"])
    edgelist["label"] = "cites"
    graph = nx.from_pandas_edgelist(edgelist, edge_attr="label")
    nx.set_node_attributes(graph, "paper", "label")

    # Load the features and subject for the nodes
    feature_names = ["w_{}".format(ii) for ii in range(1433)]
    column_names = feature_names + ["subject"]
    node_data = pd.read_table("data/cora/cora.content",
                              header=None, names=column_names)

    return graph, node_data, feature_names


def split_data(node_data):
    train_data, test_data = model_selection.train_test_split(node_data, train_size=0.7, test_size=None,
                                                             stratify=node_data['subject'])
    return train_data, test_data


def encode_classes(train_data, test_data):
    target_encoding = feature_extraction.DictVectorizer(sparse=False)

    train_targets = target_encoding.fit_transform(train_data[["subject"]].to_dict('records'))
    test_targets = target_encoding.transform(test_data[["subject"]].to_dict('records'))

    return train_targets, test_targets


def calculate_metrics(test_targets, predictions):
    """Calculation of accuracy score, F1 micro and F1 macro"""
    print(f'\tAccuracy score: {accuracy_score(test_targets, predictions)}')
    print(f'\tF1-micro: {f1_score(test_targets, predictions, average="micro")}')
    print(f'\tF1-macro: {f1_score(test_targets, predictions, average="macro")}')

    def save_embeddings(file_path, embs, nodes):
        """Save node embeddings

        :param file_path: path to the output file
        :type file_path: str
        :param embs: matrix containing the embedding vectors
        :type embs: numpy.array
        :param nodes: list of node names
        :type nodes: list(int)
        :return: None
        """
        with open(file_path, 'w') as f:
            f.write(f'{embs.shape[0]} {embs.shape[1]}\n')
            for node, emb in zip(nodes, embs):
                f.write(f'{node} {" ".join(map(str, emb.tolist()))}\n')


def read_embeddings(file_path):
    """ Load node embeddings

    :param file_path: path to the embedding file
    :type file_path: str
    :return: dictionary containing the node names as keys
             and the embeddings vectors as values
    :rtype: dict(int, numpy.array)
    """
    with open(file_path, 'r') as f:
        f.readline()
        embs = {}
        line = f.readline().strip()
        while line != '':
            parts = line.split()
            embs[int(parts[0])] = np.array(list(map(float, parts[1:])))
            line = f.readline().strip()
    return embs


if __name__ == '__main__':
    g, nodes, features_names = read_graph()
    train_data, test_data = split_data(nodes)
    train_targets, test_targets = encode_classes(train_data, test_data)
    train_features, test_features = train_data[features_names], test_data[features_names]
    train_nodes = train_features.index.values.tolist()
    test_nodes = test_features.index.values.tolist()
