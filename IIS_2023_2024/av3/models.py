import torch
from torch.nn.functional import dropout
from torch_geometric.nn import to_hetero
from torch_geometric.nn import Linear, SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = SAGEConv((-1, -1), 64)
        self.conv2 = SAGEConv((-1, -1), 128)
        self.conv3 = SAGEConv((-1, -1), 64)

        self.linear1 = Linear(64, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).tanh()
        x = dropout(x, p=0.3)

        x = self.conv2(x, edge_index).tanh()
        x = dropout(x, p=0.3)

        x = self.conv3(x, edge_index).tanh()
        x = dropout(x, p=0.3)

        x = self.linear1(x)

        return x
