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
def _(mo):
    mo.md(
        r"""
    ## Graph Dataset Structure

    CORA contains 1 graph with 2708 nodes and 10556 edges.

    ```python
    # First and only graph of the CORA dataset; dataset[1] => index out of bounds
    data = dataset[0]
    data.x  # node features (1433,)
    data.y  # node labels (7,)
    # Binary masks selecting nodes for train/test/validation
    data.train_mask  
    data.test_mask
    data.val_mask
    # Edges encoded by a pair of node indices it the edge connects 
    data.edge_index  # (2, num_edges)
    ```

    ### Node
    - Has a **feature vector** describing the paper’s content (bag-of-words of the abstract)
    - Has a **class label** (topic area)

    ### Node features
    - 1,433-dimensional **binary** feature vector
    - Comes from a bag-of-words (BoW) representation:
      - They took the entire corpus of paper abstracts in the dataset. Built a dictionary of 1,433 unique words (after preprocessing). Each node’s feature vector has a **1 at position i if that word appears in the paper’s abstract, else 0.**

    ### Edge
    - Is a **citation link between two papers**
    - The assumption is that **papers that cite each other are topically related** — so the citation structure encodes semantic relationships.

    ### Edge features
    There are **none!**

    ### Labels
    Each label is an integer encoding a paper topic according to the map
    ```python
    topic_map = {
        0: "Case Based",
        1: "Genetic Algorithms",
        2: "Neural Networks",
        3: "Probabilistic Methods",
        4: "Reinforcement Learning",
        5: "Rule Learning",
        6: "Theory",
    }
    ```
    """
    )
    return


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
    ## Node Classification

    ### Multi-Class Prediction

    Predict the topic (label) of each paper from its text features (node features) and citation links (encoded by edge_index).
    """
    )
    return


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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=10e-4)

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
            test_preds = F.softmax(test_out, dim=1).argmax(dim=1)
            accuracy = (test_preds == test_y).float().mean()
            print(f"{epoch=} | test_loss={test_loss.item():.4f} | acc={accuracy.item():.4f}")

        optimizer.step()

    print(f"{out.device=}")
    return (model,)


@app.cell
def _(F, data, model, torch):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import numpy as np

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

    # Map label indexes to their class descriptions
    topic_map = np.array([
        "Case Based",
        "Genetic Algorithms",
        "Neural Networks",
        "Probabilistic Methods",
        "Reinforcement Learning",
        "Rule Learning",
        "Theory"
    ])
    labels_topics = topic_map[labels]
    preds_topics = topic_map[preds]
    return emb_2d, labels, labels_topics, preds, preds_topics


@app.cell
def _(emb_2d, labels, labels_topics, preds, preds_topics):
    import altair as alt
    import pandas as pd

    df = pd.DataFrame({
        "x": emb_2d[:, 0],
        "y": emb_2d[:, 1],
        "True": labels_topics,
        "Predicted": preds_topics,
        "correct": labels == preds,
    })

    # Define base chart (all nodes)
    base = (
        alt.Chart(df)
        .mark_circle(size=60, opacity=0.8)
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color("True:N", legend=alt.Legend(title="Paper Topics")),
            tooltip=["True", "Predicted"]
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
            title="Learned Node Embeddings in 2D (t-SNE)",
            width=700, height=500
        )
        .interactive()
    )

    chart
    return


if __name__ == "__main__":
    app.run()
