from torch.optim import Adam
from torch_geometric.nn import LightGCN
from torch.utils.data import DataLoader
from torch_geometric.datasets import IMDB
from lightgcn_recommendation import train

if __name__ == '__main__':
    data = IMDB('../data/IMDB')
    dataset = data[0]

    print(dataset)

    num_actors, num_movies = dataset['actor'].num_nodes, dataset['movie'].num_nodes
    num_nodes = dataset.num_nodes
    dataset = dataset.to_homogeneous()

    data_loader = DataLoader(range(dataset.edge_index.size(1)),
                             shuffle=True,
                             batch_size=16)

    model = LightGCN(num_nodes=num_nodes, embedding_dim=128, num_layers=1)

    optimizer = Adam(model.parameters(), lr=0.0001)

    train(dataset, data_loader, model, optimizer, num_actors, num_movies, 1)
