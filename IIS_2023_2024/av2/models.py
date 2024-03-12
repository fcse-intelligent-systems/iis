import torch
from torch_geometric.nn import GCNConv, Linear, SAGEConv
from torch.nn.functional import dropout


class GCN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = GCNConv(-1, 64)  # SAGEConv
        self.conv2 = GCNConv(-1, 128)  # SAGEConv

        self.linear1 = Linear(128, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).tanh()
        x = dropout(x, p=0.3)

        x = self.conv2(x, edge_index).tanh()
        x = dropout(x, p=0.3)

        x = self.linear1(x)

        return x
