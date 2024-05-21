import torch
from torch.nn import Linear
from torch.nn.functional import dropout
from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_classes):
        super(GraphSAGE, self).__init__()

        self.conv1 = SAGEConv((-1, -1), 64)
        self.conv2 = SAGEConv((-1, -1), 128)
        self.conv3 = SAGEConv((-1, -1), 64)

        self.linear1 = Linear(64, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = dropout(x, p=0.5, training=self.training)
        x = self.linear1(x)

        return x
