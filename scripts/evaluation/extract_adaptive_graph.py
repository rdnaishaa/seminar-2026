from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


# =========================================================
# PATH
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = ROOT / "outputs" / "results"

WEIGHT_PATH = (
    RESULT_DIR / "adaptive_stgnn.weights.h5"
)

FIXED_GRAPH_PATH = (
    RESULT_DIR / "fixed_graph_adjacency.csv"
)


# =========================================================
# CONFIG
# =========================================================

INPUT_LENGTH = 12
HORIZONS = [1, 3, 6]

EMBEDDING_DIM = 8
SPATIAL_UNITS = 16
GRU_UNITS = 32


# =========================================================
# NODE ORDER
# =========================================================

fixed_df = pd.read_csv(
    FIXED_GRAPH_PATH,
    index_col=0
)

corridors = fixed_df.index.tolist()

NUM_NODES = len(corridors)

print("\n=== ADAPTIVE GRAPH EXTRACTION ===")
print("Nodes :", NUM_NODES)

for i, corridor in enumerate(corridors):
    print(f"{i:2d} -> {corridor}")


# =========================================================
# ADAPTIVE GRAPH LAYER
# =========================================================

class AdaptiveGraphConv(tf.keras.layers.Layer):

    def __init__(
        self,
        units,
        num_nodes,
        embedding_dim=8,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.units = units
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim

        self.projection = tf.keras.layers.Dense(
            units,
            activation="relu"
        )

    def build(self, input_shape):

        self.node_embeddings = self.add_weight(
            name="node_embeddings",
            shape=(self.num_nodes, self.embedding_dim),
            initializer="glorot_uniform",
            trainable=True
        )

        super().build(input_shape)

    def compute_adjacency(self):

        scores = tf.matmul(
            self.node_embeddings,
            self.node_embeddings,
            transpose_b=True
        )

        scores = tf.nn.relu(scores)

        adjacency = (
            scores
            + tf.eye(
                self.num_nodes,
                dtype=scores.dtype
            )
        )

        degree = tf.reduce_sum(
            adjacency,
            axis=1
        )

        d_inv_sqrt = tf.math.rsqrt(
            degree + 1e-8
        )

        adjacency_norm = (
            adjacency
            * d_inv_sqrt[:, None]
            * d_inv_sqrt[None, :]
        )

        return adjacency_norm

    def call(self, inputs):

        adjacency = self.compute_adjacency()

        graph_x = tf.einsum(
            "ij,btjf->btif",
            adjacency,
            inputs
        )

        x = tf.concat(
            [inputs, graph_x],
            axis=-1
        )

        x = self.projection(x)

        return x

    def get_adjacency(self):

        return self.compute_adjacency()


# =========================================================
# REBUILD MODEL
# =========================================================

inputs = tf.keras.Input(
    shape=(
        INPUT_LENGTH,
        NUM_NODES,
        1
    )
)

adaptive_layer = AdaptiveGraphConv(
    units=SPATIAL_UNITS,
    num_nodes=NUM_NODES,
    embedding_dim=EMBEDDING_DIM,
    name="adaptive_graph_conv"
)

x = adaptive_layer(inputs)

x = tf.keras.layers.Permute(
    (2, 1, 3)
)(x)

x = tf.keras.layers.Reshape(
    (
        NUM_NODES,
        INPUT_LENGTH * SPATIAL_UNITS
    )
)(x)

x = tf.keras.layers.Reshape(
    (
        NUM_NODES,
        INPUT_LENGTH,
        SPATIAL_UNITS
    )
)(x)

x = tf.keras.layers.TimeDistributed(
    tf.keras.layers.GRU(
        GRU_UNITS
    )
)(x)

x = tf.keras.layers.Dropout(
    0.2
)(x)

x = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Dense(
        len(HORIZONS)
    )
)(x)

outputs = tf.keras.layers.Permute(
    (2, 1)
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs
)


# =========================================================
# LOAD TRAINED WEIGHTS
# =========================================================

model.load_weights(
    WEIGHT_PATH
)

print("\nWeights loaded:")
print(WEIGHT_PATH)


# =========================================================
# EXTRACT LEARNED ADJACENCY
# =========================================================

adaptive_adj = (
    adaptive_layer
    .get_adjacency()
    .numpy()
)

adaptive_df = pd.DataFrame(
    adaptive_adj,
    index=corridors,
    columns=corridors
)

adaptive_output_path = (
    RESULT_DIR /
    "adaptive_graph_adjacency.csv"
)

adaptive_df.to_csv(
    adaptive_output_path
)


# =========================================================
# CREATE DIRECTED EDGE TABLE
# Exclude self-connections
# =========================================================

edge_rows = []

for i, source in enumerate(corridors):

    for j, target in enumerate(corridors):

        if i == j:
            continue

        edge_rows.append({
            "source": source,
            "target": target,
            "weight": float(
                adaptive_adj[i, j]
            )
        })

edges_df = pd.DataFrame(
    edge_rows
).sort_values(
    "weight",
    ascending=False
)

edges_output_path = (
    RESULT_DIR /
    "adaptive_graph_edges.csv"
)

edges_df.to_csv(
    edges_output_path,
    index=False
)


# =========================================================
# COMPARE ADAPTIVE VS FIXED GRAPH
# =========================================================

comparison_rows = []

fixed_adj = fixed_df.values

for i, source in enumerate(corridors):

    for j, target in enumerate(corridors):

        if i == j:
            continue

        comparison_rows.append({
            "source": source,
            "target": target,

            "adaptive_weight":
                float(adaptive_adj[i, j]),

            "fixed_edge":
                int(fixed_adj[i, j] > 0)
        })

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_output_path = (
    RESULT_DIR /
    "adaptive_vs_fixed_graph.csv"
)

comparison_df.to_csv(
    comparison_output_path,
    index=False
)


# =========================================================
# TOP ADAPTIVE NEIGHBORS PER CORRIDOR
# =========================================================

top_neighbor_rows = []

TOP_K = 3

for i, source in enumerate(corridors):

    weights = adaptive_adj[i].copy()

    # Jangan pilih dirinya sendiri
    weights[i] = -1

    top_indices = np.argsort(
        weights
    )[::-1][:TOP_K]

    for rank, j in enumerate(
        top_indices,
        start=1
    ):

        top_neighbor_rows.append({
            "source": source,
            "rank": rank,
            "target": corridors[j],
            "adaptive_weight":
                float(adaptive_adj[i, j]),
            "is_fixed_neighbor":
                int(fixed_adj[i, j] > 0)
        })


top_neighbors_df = pd.DataFrame(
    top_neighbor_rows
)

top_neighbors_output_path = (
    RESULT_DIR /
    "adaptive_graph_top3_neighbors.csv"
)

top_neighbors_df.to_csv(
    top_neighbors_output_path,
    index=False
)


# =========================================================
# SIMPLE AUDIT
# =========================================================

row_sums = adaptive_adj.sum(
    axis=1
)

self_weights = np.diag(
    adaptive_adj
)

print("\n=== ADJACENCY AUDIT ===")

print(
    "Shape:",
    adaptive_adj.shape
)

print(
    "Row sum min/max:",
    f"{row_sums.min():.6f}",
    f"{row_sums.max():.6f}"
)

print(
    "Mean self-weight:",
    f"{self_weights.mean():.6f}"
)

print(
    "Min self-weight:",
    f"{self_weights.min():.6f}"
)

print(
    "Max self-weight:",
    f"{self_weights.max():.6f}"
)


# =========================================================
# TOP 15 LEARNED DIRECTED EDGES
# =========================================================

print(
    "\n=== TOP 15 ADAPTIVE EDGES ==="
)

print(
    edges_df.head(15)
    .to_string(
        index=False
    )
)


# =========================================================
# TOP 3 NEIGHBORS PER NODE
# =========================================================

print(
    "\n=== TOP 3 ADAPTIVE NEIGHBORS "
    "PER CORRIDOR ==="
)

print(
    top_neighbors_df.to_string(
        index=False
    )
)


# =========================================================
# SAVE
# =========================================================

print("\n=== SAVED ===")

print(
    "outputs/results/"
    "adaptive_graph_adjacency.csv"
)

print(
    "outputs/results/"
    "adaptive_graph_edges.csv"
)

print(
    "outputs/results/"
    "adaptive_vs_fixed_graph.csv"
)

print(
    "outputs/results/"
    "adaptive_graph_top3_neighbors.csv"
)