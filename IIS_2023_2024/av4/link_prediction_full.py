from torch.optim import SGD
from torch_geometric.datasets import IMDB
from torch_geometric.transforms import RandomLinkSplit
from gnn_link_prediction import Model, train_link_prediction

if __name__ == '__main__':
    data = IMDB('../data/IMDB')
    dataset = data[0]

    print(dataset)

    train_val_test_split = RandomLinkSplit(num_val=0.1,
                                           num_test=0.1,
                                           add_negative_train_samples=True,
                                           edge_types=('movie', 'to', 'actor'),
                                           rev_edge_types=('actor', 'to', 'movie'))

    train_data, val_data, test_data = train_val_test_split(dataset)

    model = Model(hidden_channels=128, data=dataset)

    optimizer = SGD(model.parameters(), lr=0.0001)

    train_link_prediction(model, train_data, val_data, optimizer, 1)
