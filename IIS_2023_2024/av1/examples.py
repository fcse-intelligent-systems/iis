import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.nn import Node2Vec
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import Planetoid
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from node_embeddings import train

if __name__ == '__main__':
    data = Planetoid('../data', 'Cora')
    dataset = data[0]

    model = Node2Vec(dataset.edge_index,
                     embedding_dim=50,
                     walk_length=30,
                     context_size=10,
                     walks_per_node=20,
                     num_negative_samples=1,
                     p=200, q=1,
                     sparse=True)

    train(model, epochs=5)

    labels = dataset.y.detach().cpu().numpy()
    node_embeddings = model().detach().cpu().numpy()

    train_x, test_x, train_y, test_y = train_test_split(node_embeddings, labels,
                                                        test_size=0.1,
                                                        stratify=labels)

    classifier = RandomForestClassifier(n_estimators=50)
    classifier.fit(train_x, train_y)

    preds = classifier.predict(test_x)

    print(f'Accuracy: {accuracy_score(preds, test_y)}')

    tsne = TSNE(n_components=2)
    node_embeddings_2d = tsne.fit_transform(node_embeddings)

    plt.scatter(node_embeddings_2d[:, 0], node_embeddings_2d[:, 1],
                c=labels, cmap='jet', alpha=0.7)
