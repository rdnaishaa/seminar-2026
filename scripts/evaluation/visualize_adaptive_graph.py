from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]

ADJ_PATH = ROOT / "outputs/results/adaptive_graph_adjacency.csv"
COORD_PATH = ROOT / "corridors.csv"
OUT_PATH = ROOT / "outputs/figures/adaptive_graph_day7.png"

TOP_K = 3


# =========================================================
# LOAD DATA
# =========================================================

adj_df = pd.read_csv(ADJ_PATH, index_col=0)
coords_df = pd.read_csv(COORD_PATH)

nodes = adj_df.index.tolist()

coords_df = coords_df[
    coords_df["station_id"].isin(nodes)
].copy()

coords_df = coords_df.set_index("station_id")


# =========================================================
# BUILD GRAPH
# =========================================================

G = nx.Graph()

for node in nodes:
    G.add_node(node)

# Karena adjacency symmetric:
# ambil top-k neighbor tiap node,
# tetapi edge duplikat otomatis digabung oleh nx.Graph.
for source in nodes:

    weights = adj_df.loc[source].copy()

    # jangan ambil self-loop
    weights.loc[source] = 0.0

    top_neighbors = (
        weights
        .sort_values(ascending=False)
        .head(TOP_K)
    )

    for target, weight in top_neighbors.items():

        if weight <= 0:
            continue

        if G.has_edge(source, target):
            # pertahankan weight terbesar
            G[source][target]["weight"] = max(
                G[source][target]["weight"],
                float(weight)
            )
        else:
            G.add_edge(
                source,
                target,
                weight=float(weight)
            )


# =========================================================
# POSITIONS
# =========================================================

pos = {}

for node in nodes:
    pos[node] = (
        coords_df.loc[node, "lon"],
        coords_df.loc[node, "lat"]
    )


# =========================================================
# DRAW
# =========================================================

plt.figure(figsize=(13, 10))

weights = np.array([
    G[u][v]["weight"]
    for u, v in G.edges()
])

if len(weights) > 0:
    min_w = weights.min()
    max_w = weights.max()

    if max_w > min_w:
        widths = 1.0 + 5.0 * (
            (weights - min_w) /
            (max_w - min_w)
        )
    else:
        widths = np.full_like(weights, 2.5)

else:
    widths = []


nx.draw_networkx_edges(
    G,
    pos,
    width=widths,
    alpha=0.45
)

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=900
)

labels = {
    node: node.replace("tt_", "")
    for node in nodes
}

nx.draw_networkx_labels(
    G,
    pos,
    labels=labels,
    font_size=8
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title(
    "Adaptive Graph Learned from Traffic Data\n"
    f"Top-{TOP_K} Learned Neighbors per Corridor"
)

plt.grid(alpha=0.2)

plt.tight_layout()

OUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    OUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# SUMMARY
# =========================================================

print("\n=== ADAPTIVE GRAPH VISUALIZATION ===")
print(f"Nodes : {G.number_of_nodes()}")
print(f"Edges : {G.number_of_edges()}")
print(f"Top-k : {TOP_K}")
print("\nSaved:")
print(OUT_PATH)