import matplotlib.pyplot as plt
from torch.optim import Adam
from sklearn.manifold import TSNE
from torch.nn import CrossEntropyLoss
from torch_geometric.datasets import Planetoid
from model_utils import train
from models import GCN


if __name__ == '__main__':
    data = Planetoid('../data', 'Cora')
    dataset = data[0]

    model = GCN(num_classes=7)

    optimizer = Adam(model.parameters(), lr=0.0001)
    criterion = CrossEntropyLoss()

    node_embeddings = model(dataset.x, dataset.edge_index)
    ...

    train(model, dataset, optimizer, criterion, 10)

    node_embeddings = model(dataset.x, dataset.edge_index)
    ...




