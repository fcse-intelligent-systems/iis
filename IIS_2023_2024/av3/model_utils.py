from torch.nn.functional import one_hot


def train_classification(model,
                         train_loader, val_loader,
                         optimizer, criterion,
                         epochs=5):
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            x_dict = batch.x_dict
            edge_index_dict = batch.edge_index_dict
            train_mask = batch['movie'].train_mask
            data_y = one_hot(batch['movie'].y, 3).float()

            out = model(x_dict, edge_index_dict)
            loss = criterion(out['movie'][train_mask], data_y[train_mask])
            train_loss = loss.item()

            loss.backward()
            optimizer.step()

            print(f'Epoch: {epoch:03d}, Step: {i:03d}, Loss: {train_loss:.4f}')

        for i, batch in enumerate(val_loader):
            model.eval()

            x_dict = batch.x_dict
            edge_index_dict = batch.edge_index_dict
            val_mask = batch['movie'].val_mask
            data_y = one_hot(batch['movie'].y, 3).float()

            out = model(x_dict, edge_index_dict)
            loss = criterion(out['movie'][val_mask], data_y[val_mask])
            val_loss = loss.item()

            print(f'Epoch: {epoch:03d}, Step: {i:03d}, Val Loss: {val_loss:.4f}')
