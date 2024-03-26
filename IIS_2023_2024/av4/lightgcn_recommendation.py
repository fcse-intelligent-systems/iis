import torch


def train(dataset, train_loader, model, optimizer, num_actors, num_movies, epochs=1):
    for epoch in range(epochs):
        total_loss, total_examples = 0, 0

        for node_ids in train_loader:
            pos_edge_label_index = dataset.edge_index[:, node_ids]
            neg_edge_label_index = torch.stack([pos_edge_label_index[0],
                                                torch.randint(num_actors, num_actors + num_movies,
                                                              (node_ids.numel(),))],
                                               dim=0)
            edge_label_index = torch.cat([pos_edge_label_index, neg_edge_label_index], dim=1)

            optimizer.zero_grad()

            pos_rank, neg_rank = model(dataset.edge_index, edge_label_index).chunk(2)

            loss = model.recommendation_loss(pos_rank, neg_rank, node_id=edge_label_index.unique())
            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pos_rank.numel()
            total_examples += pos_rank.numel()

            print(f'Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}')
