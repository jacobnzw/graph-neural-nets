import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch_geometric
    import torch
    import torch.nn.functional as F
    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import GCNConv
    return F, GCNConv, Planetoid, torch


@app.cell
def _(Planetoid):
    dataset = Planetoid(root="../datasets/cora", name="Cora")
    # NOTE: CORA contains only 1 graph, nodes divided into train, test, val sets
    data = dataset[0]

    return data, dataset


@app.cell
def _(data):
    data.x
    return


@app.cell
def _(dataset):
    print(f"# graphs: {len(dataset)}")
    print(f"{dataset.num_classes=}")
    print(f"{dataset.num_features=}")
    print(f"{dataset.num_node_features=}")
    print(f"{dataset.num_edge_features=}")
    # data.get_summary()
    # data.print_summary()
    return


@app.cell
def _(GCNConv, torch):
    from torch.nn import ReLU, Linear, Softmax, Dropout
    from torch_geometric.nn import Sequential  # extension of torch.nn.Sequential to >1 input args in forward

    class GCNClassifier(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.embedder = Sequential(
                'x, edge_index',
                (GCNConv(in_channels, hidden_channels),
                ReLU(),
                GCNConv(hidden_channels, hidden_channels),
                ReLU(),
                GCNConv(hidden_channels, out_channels))
            )
            # self.dropout = Dropout(p=0.3)

        def forward(self, x, edge_index):
            x = self.embedder(x, edge_index)
            # NOTE: Outputting the embedded node features directly performs better
            # than adding a dedicated linear classifier
            # x = self.relu(x)
            # x = self.classifier(x)
            return x
    return (GCNClassifier,)


@app.cell
def _(F, GCNClassifier, data, dataset, torch):
    hidden_dim = 16
    model = GCNClassifier(dataset.num_node_features, hidden_dim, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    device = torch.accelerator.current_accelerator()
    print_every = 10
    n_epochs = 1000

    model.to(device)
    data.to(device)

    # Training loop
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
    
        out = model(data.x, data.edge_index)
    
        # Compute loss only from the train nodes
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()

        if epoch % (print_every - 1) == 0:
            test_out, test_y = out[data.test_mask], data.y[data.test_mask]
            test_loss = F.cross_entropy(test_out, test_y)
            test_preds = test_out.argmax(dim=1)
            # accuracy = torch.sum(test_preds == test_y) / len(test_out)
            accuracy = (test_preds == test_y).float().mean()
            print(f"{epoch=} | test_loss={test_loss.item():.4f} | acc={accuracy.item():.4f}")
    
        optimizer.step()

    print(f"{out.device=}")
    return model, out


@app.cell
def _(data, dataset, model, torch):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    model.eval()
    with torch.no_grad():
        # Get embeddings from last hidden layer (before classifier)
        embeddings = model(data.x, data.edge_index).cpu().numpy()
        labels = data.y.cpu().numpy()

    # Reduce to 2D
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(8,6))
    for i in range(dataset.num_classes):
        plt.scatter(emb_2d[labels==i,0], emb_2d[labels==i,1], label=f"Class {i}", alpha=0.7)
    plt.legend()
    plt.title("Node embeddings visualized with t-SNE")
    plt.show()

    return


@app.cell
def _(data, out, torch):
    # len(out[data.test_mask])
    # len(data.y[data.test_mask])
    # out[data.test_mask].shape
    torch.argmax(out[data.test_mask], dim=1).shape
    # data.y[data.test_mask].shape

    return


if __name__ == "__main__":
    app.run()
