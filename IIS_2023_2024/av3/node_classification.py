from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch_geometric.nn import to_hetero
from torch_geometric.datasets import IMDB
from torch_geometric.loader import NeighborLoader
from models import GraphSAGE
from model_utils import train_classification

if __name__ == '__main__':
    data = IMDB('../data/IMDB')
    dataset = data[0]

    print(dataset)

    base_model = GraphSAGE(num_classes=3)

    model = to_hetero(base_model, dataset.metadata(), aggr='sum')

    train_input_nodes = ('movie', dataset['movie'].train_mask)
    train_loader = NeighborLoader(dataset, num_neighbors=[10, 10, 10],
                                  shuffle=True, input_nodes=train_input_nodes,
                                  batch_size=128)

    val_input_nodes = ('movie', dataset['movie'].val_mask)
    val_loader = NeighborLoader(dataset, num_neighbors=[10, 10, 10],
                                shuffle=False, input_nodes=val_input_nodes,
                                batch_size=128)

    test_input_nodes = ('movie', dataset['movie'].test_mask)
    test_loader = NeighborLoader(dataset, num_neighbors=[10, 10, 10],
                                 shuffle=False, input_nodes=test_input_nodes,
                                 batch_size=128)

    optimizer = SGD(model.parameters(), lr=0.0001)
    criterion = CrossEntropyLoss()

    train_classification(model, train_loader, val_loader, optimizer, criterion, 1)
