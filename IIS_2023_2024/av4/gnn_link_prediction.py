import torch
from torch.nn.functional import mse_loss
from torch_geometric.nn import to_hetero
from torch_geometric.nn import Linear, SAGEConv


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['movie'][row], z_dict['actor'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, data):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


def train_link_prediction(model, train_data, val_data, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(train_data.x_dict, train_data.edge_index_dict,
                     train_data['movie', 'actor'].edge_label_index)
        target = train_data['movie', 'actor'].edge_label
        loss = mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        model.eval()
        pred = model(val_data.x_dict, val_data.edge_index_dict,
                     val_data['movie', 'actor'].edge_label_index)
        pred = pred.clamp(min=0, max=5)
        target = val_data['movie', 'actor'].edge_label.float()
        val_loss = mse_loss(pred, target).sqrt()

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')
