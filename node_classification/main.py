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
    return F, Planetoid, mo, torch


@app.cell
def _(Planetoid):
    dataset = Planetoid(root="../datasets", name="Cora")
    # NOTE: CORA contains only 1 graph, nodes divided into train, test, val sets
    data = dataset[0]
    return data, dataset


@app.cell
def _(data, dataset):
    from tabulate import tabulate

    dataset_table = (
        ('# graphs', len(dataset)),
        ('# classes', dataset.num_classes),
        ('# features', dataset.num_features),
        ('# node features', dataset.num_node_features),
        ('# edge features', dataset.num_edge_features),
    )
    data_table = (
        ('# nodes', data.num_nodes),
        ('# edges', data.num_edges),
        ('node attrs', data.node_attrs()),
        ('edge attrs', data.edge_attrs()),
    )

    print('\nDataset properties')
    print(tabulate(dataset_table, headers='firstrow', tablefmt='plain'))

    print('\nGraph properties')
    print(tabulate(data_table, headers='firstrow', tablefmt='plain'))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Graph data structure

    CORA contains 1 graph with 2708 nodes and 10556 edges.

    ```python
    data.x  # node features (1433,)
    data.y  # node labels (7,)
    # Binary masks selecting nodes for train/test/validation
    data.train_mask  
    data.test_mask
    data.val_mask
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Node Feature t-SNE Visualization""")
    return


@app.cell
def _(data, model, torch):
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
    return emb_2d, labels


@app.cell
def _(torch):
    from torch.nn import ReLU, Linear, Cro
    from torch_geometric.nn import Sequential  # extension of torch.nn.Sequential to >1 input args in forward
    from torch_geometric.nn import GCNConv

    class GCNClassifier(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.embedder = Sequential(
                'x, edge_index', [
                    # Need to add string specs of signatures of GCNConv.forward()
                    (GCNConv(in_channels, hidden_channels), 'x, edge_index -> x'),
                    # ReLU(),
                    # (GCNConv(hidden_channels, hidden_channels), 'x, edge_index -> x'),
                    ReLU(),
                    (GCNConv(hidden_channels, out_channels), 'x, edge_index -> x')
                ]
            )

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    device = torch.accelerator.current_accelerator()
    n_epochs = 1000
    print_every = n_epochs // 10

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
            accuracy = (test_preds == test_y).float().mean()
            print(f"{epoch=} | test_loss={test_loss.item():.4f} | acc={accuracy.item():.4f}")

        optimizer.step()

    print(f"{out.device=}")
    return (model,)


@app.cell
def _(F, data, model, torch):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # TODO: just reduce the feature dim to 2D and see the clusters if any
    model.eval()
    with torch.no_grad():
        # Embeddings are 7-dimensional vectors 
        embeddings = model(data.x, data.edge_index)
        # Convert embeddings/logits to hard predictions
        preds = F.softmax(embeddings, dim=1).argmax(dim=1).cpu().numpy()
        labels = data.y.cpu().numpy()

    # Reduce to 2D
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings.cpu().numpy())
    return emb_2d, labels, preds


@app.cell
def _(emb_2d, labels, preds):
    import altair as alt
    import pandas as pd

    df = pd.DataFrame({
        "x": emb_2d[:, 0],
        "y": emb_2d[:, 1],
        "label": labels.astype(str),
        "pred": preds.astype(str)
    })

    # Flag incorrect predictions
    df["correct"] = df["label"] == df["pred"]

    # Define base chart (all nodes)
    base = (
        alt.Chart(df)
        .mark_circle(size=60, opacity=0.8)
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color("label:N", legend=alt.Legend(title="True Label")),
            tooltip=["label", "pred"]
        )
    )

    # Overlay layer for misclassified nodes (black border)
    wrong = (
        alt.Chart(df.query("correct == False"))
        .mark_circle(size=100, fillOpacity=0, stroke="black", strokeWidth=1.5)
        .encode(x="x:Q", y="y:Q")
    )

    chart = (
        (base + wrong)
        .properties(
            title="t-SNE Visualization of Node Embeddings (Altair)",
            width=700, height=600
        )
        .interactive()
    )

    chart
    return


if __name__ == "__main__":
    app.run()
