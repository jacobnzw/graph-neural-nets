# Graph Neural Networks (GNNs)

Graph goes in, graph comes out, each node feature has a new embedding, which can be later used for node classification, link prediction, etc. GNNs could be seen as a feature extractor generalizated to data supported on a non-Euclidean domain.

> A GNN is an optimizable transformation on all attributes of the graph (nodes, edges, global-context) that preserves graph symmetries (permutation invariances).

> GNNs adopt a “graph-in, graph-out” architecture meaning that these model types accept a graph as input, with information loaded into its nodes, edges and global-context, and progressively transform these embeddings, without changing the connectivity of the input graph.


## Prediction Tasks on Graphs

  - **Graph-level:** predicting single property of the whole graph.
    - Graph classification (e.g. molecule classification=is the molecule toxic? how reactive is it? how does it interact with other molecules?)
  - **Node-level:** predict property of each node.
    - Node classification (e.g. in social networks: is a user male or female? what group/team does a user belong to?)
  - **Edge-level:** predict property of each edge or predict existence of edges.
    - Link prediction (e.g. in social networks: will two people become friends? in recommender systems: will a user buy a product?)
    - Image scene understanding: given a set of objects in an image, predict which objects are interacting with each other? What is their relationship? 


## Representing Graphs

### Adjacency matrix
$A_{ij} = 1$ if there is an edge between nodes $i$ and $j$, 0 otherwise. This is not a good representation for several reasons:
  - Sparse: for large graphs, most of the entries are 0.
  - Not scalable: for large graphs, the adjacency matrix is too large to fit in memory.
  - Not permutation invariant: the matrix will change if we permute the nodes.

### Adjacency list
List of edges $e_k$ between nodes $n_i$ and $n_j$ as a tuple $(i, j)$. This is a better representation for several reasons:
  - Sparse: only store the edges.
  - Scalable: can be streamed.
  - Permutation invariant: the list will not change if we permute the nodes.

### Graphs as tensors
Nodes, edges and graphs themselves can have associated features. We can represent them as tensors. Which means in the end our graph representation is a tuple of node/edge/graph feature tensors plus an adjacency list encoding the connectivity.

For example, a graph with 3 nodes and 2 edges, with node features of size 3, edge features of size 2, can be represented as:

```python
node_features = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
edge_features = torch.tensor([[1, 2], [3, 4]])
adjacency_list = [(0, 1), (1, 2)]
```
The node feature tensor has shape `(3, 3)`, the edge feature tensor has shape `(2, 2)`, and the adjacency list has length 2.

In general, node feature tensor is of shape `(n_nodes, n_node_features)`, the edge feature tensor is of shape `(n_edges, n_edge_features)`, and the adjacency list has length `n_edges`. The graph-level features (aka master node or global context vector) is a tensor of shape `(n_graph_features,)`.


## The Simplest GNN Layer
Contains independent transformation for nodes, edges and graph. This transformation is applied to each node/edge/graph independently. It could be an MLP, a CNN, a RNN, etc. The output of the transformation is the new embedding of the node/edge/graph features.

### Predictions by Pooling
>If the task is to make predictions on nodes, and the graph already contains node features/embeddings, the approach is straightforward — for each node embedding, apply a classifier.

But your graph might have only edge feaures, but you still need to make predictions on nodes. We need a way to collect information from edges and give them to nodes for prediction. We can do this by pooling. 

Pooling proceeds in two steps:
  - For each node/edge to be pooled, stack their features into a matrix.
  - Aggregate the matrix into a single vector, usually via a sum operation.

We can pool in several directions:
  - nodes to edges: for each edge, pool the features of the two nodes it connects.
  - edges to nodes: for each node, pool the features of all edges that connect to it.
  - nodes to graph: for the graph, pool the features of all nodes.
  - edges to graph: for the graph, pool the features of all edges.

> Note that **in this simplest GNN formulation, we’re not using the connectivity** of the graph at all inside the GNN layer. Each node is processed independently, as is each edge, as well as the global context. We only use connectivity when pooling information for prediction.


## Message Passing GNNs
In the simplest GNN layer, the only place connectivity is used is when pooling for prediction, which is done after the GNN layer computes the new embeddings.

To make use of the graph connectivity in the GNN layer we need to use message passing, which can occur between nodes, edges and the graph (global context vector). Broadly speaking, in message passing neighboring nodes/edges exchange messages with each other to influence each other's embeddings.

Message passing works in three steps (assuming nodes), for each node:
  1. Gather messages from neighbors: gather the embeddings from all neighboring nodes.
  2. Aggregate messages: aggregate the gathered embeddings into a single vector.
  3. Update embedding: update the node's embedding using the aggregated embedding.

A sum is the most common aggregation function, but other functions can be used, such as mean, max, etc. The update function can be any differentiable function, but it is usually an MLP.

Formally, neural message passing can be defined as:
$$
\begin{align*}
\mathbf{m}_{\mathcal{N}(v)}^{(l)} &= \mathrm{Aggregate}^{(l)}(\{\mathbf{h}^{(l)}_{u}: u \in \mathcal{N}(v)\}) \\
\mathbf{h}_v^{(l+1)} &= \mathrm{Update}^{(l)} \left( \mathbf{h}_v^{(l)}, \mathbf{m}_{\mathcal{N}(v)}^{(l)} \right)
\end{align*}
$$
where $\mathbf{h}_v^{(l)}$ is the embedding of the current node $v$ at layer $l$, $\{\mathbf{h}_{u}\}_{u \in \mathcal{N}(v)}$ is the set of embeddings from all neighbors of $v$ from which we calculate the message $\mathbf{m}_{\mathcal{N}(v)}^{(l)}$, and $\mathrm{Update}^{(l)}$ is the update function. Both $\mathrm{Aggregate}^{(l)}$ and $\mathrm{Update}^{(l)}$ are differentiable functions and thus could be modelled by neural networks, such as MLPs or whatever else.


## Batching in GNNs
> A common practice for training neural networks is to update network parameters with gradients calculated on randomized constant size (batch size) subsets of the training data (mini-batches). This practice presents a challenge for graphs due to the variability in the number of nodes and edges adjacent to each other, meaning that **we cannot have a constant batch size**. 



## Resources

[DeepFindr](https://deepfindr.github.io/videos/)
  - [Understanding GNNs](https://deepfindr.github.io/videos/gnns)
  - [Temporal GNNs](https://deepfindr.github.io/videos/stgnn)

[Distill](https://distill.pub/)
  - [Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)
  - [Understanding Convolutions on Graphs](https://distill.pub/2021/understanding-gnns/)

[PyTorch Geometric Tutorials](https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html)

[Stanford CS224W: Machine Learning with Graphs](https://web.stanford.edu/class/cs224w/)

[Graph Representation Learning [book]](https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book.pdf)