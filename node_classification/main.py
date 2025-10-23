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
    from torch.nn import ReLU
    return F, GCNConv, Planetoid, ReLU, torch


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
def _(GCNConv, Linear, ReLU, Sequential, torch):
    # Simple 2-layer GCN
    class GCNClassifier(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.relu = ReLU()

            self.mlp = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, out_channels))

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = self.relu(x)
            x = self.conv2(x, edge_index)
            x = self.relu(x)
            x = self.mlp(x)
            return x
    return (GCNClassifier,)


@app.cell
def _(F, GCNClassifier, data, dataset, torch):
    hidden_dim = 32
    model = GCNClassifier(dataset.num_node_features, hidden_dim, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    print_every = 10
    # Training loop
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
    
        out = model(data.x, data.edge_index)
        # Compute loss only from the train nodes
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()

        if epoch % (print_every - 1) == 0:
            # test_out = model(data.x[data.test_mask], data.edge_index[data.test_mask])
            test_out, test_y = out[data.test_mask], data.y[data.test_mask]
            test_loss = F.cross_entropy(test_out, test_y)
            test_preds = torch.argmax(test_out, dim=1)
            accuracy = torch.sum(test_preds == test_y) / len(test_out)
            print(f"{epoch=} | test_loss={test_loss.item():.4f} | acc={accuracy.item():.4f}")
    
        optimizer.step()

    return (out,)


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
