def train(model, train_loader, val_loader, optimizer, criterion, epochs=5):
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            model.train()

            out = model(batch.x, batch.edge_index, batch.batch)

            loss = criterion(out, batch.y)
            loss.backward()
            train_loss = loss.item()

            optimizer.step()
            optimizer.zero_grad()

            print(f'Epoch: {epoch:03d}, Step: {i:03d}, Loss: {train_loss:.4f}')

        for i, batch in enumerate(val_loader):
            model.eval()

            out = model(batch.x, batch.edge_index, batch.batch)

            loss = criterion(out, batch.y)
            val_loss = loss.item()

            print(f'Epoch: {epoch:03d}, Step: {i:03d}, Val Loss: {val_loss:.4f}')
