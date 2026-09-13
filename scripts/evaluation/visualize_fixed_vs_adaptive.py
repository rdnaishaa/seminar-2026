from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]

COMPARE_PATH = ROOT / "outputs/results/adaptive_vs_fixed_graph.csv"
COORD_PATH = ROOT / "corridors.csv"
OUT_PATH = ROOT / "outputs/figures/fixed_vs_adaptive_graph.png"

TOP_K = 3


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(COMPARE_PATH)
coords = pd.read_csv(COORD_PATH).set_index("station_id")

nodes = sorted(
    set(df["source"]).union(set(df["target"]))
)


# =========================================================
# BUILD ADAPTIVE TOP-K EDGE SET
# =========================================================

adaptive_edges = set()

for source in nodes:

    temp = df[
        (df["source"] == source) &
        (df["adaptive_weight"] > 0)
    ].copy()

    temp = temp.sort_values(
        "adaptive_weight",
        ascending=False
    ).head(TOP_K)

    for _, row in temp.iterrows():

        edge = tuple(sorted([
            row["source"],
            row["target"]
        ]))

        adaptive_edges.add(edge)


# =========================================================
# BUILD FIXED EDGE SET
# =========================================================

fixed_edges = set()

fixed_df = df[df["fixed_edge"] == 1]

for _, row in fixed_df.iterrows():

    edge = tuple(sorted([
        row["source"],
        row["target"]
    ]))

    fixed_edges.add(edge)


# =========================================================
# CLASSIFY EDGES
# =========================================================

overlap_edges = adaptive_edges & fixed_edges
adaptive_only = adaptive_edges - fixed_edges
fixed_only = fixed_edges - adaptive_edges


print("\n=== FIXED VS ADAPTIVE GRAPH ===")
print("Adaptive edges :", len(adaptive_edges))
print("Fixed edges    :", len(fixed_edges))
print("Overlap        :", len(overlap_edges))
print("Adaptive only  :", len(adaptive_only))
print("Fixed only     :", len(fixed_only))


# =========================================================
# BUILD GRAPH
# =========================================================

G = nx.Graph()

for node in nodes:
    G.add_node(node)

for edge in fixed_only:
    G.add_edge(*edge, edge_type="fixed_only")

for edge in adaptive_only:
    G.add_edge(*edge, edge_type="adaptive_only")

for edge in overlap_edges:
    G.add_edge(*edge, edge_type="overlap")


# =========================================================
# POSITIONS
# =========================================================

pos = {
    node: (
        coords.loc[node, "lon"],
        coords.loc[node, "lat"]
    )
    for node in nodes
}


# =========================================================
# DRAW
# =========================================================

plt.figure(figsize=(14, 10))

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=850
)

nx.draw_networkx_edges(
    G,
    pos,
    edgelist=list(fixed_only),
    width=1.2,
    alpha=0.35,
    style="dashed",
    edge_color="gray"
)

nx.draw_networkx_edges(
    G,
    pos,
    edgelist=list(adaptive_only),
    width=2.0,
    alpha=0.55,
    edge_color="tab:blue"
)

nx.draw_networkx_edges(
    G,
    pos,
    edgelist=list(overlap_edges),
    width=3.0,
    alpha=0.8,
    edge_color="tab:green"
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


# =========================================================
# LEGEND
# =========================================================

from matplotlib.lines import Line2D

legend_items = [
    Line2D(
        [0], [0],
        color="gray",
        lw=1.5,
        linestyle="--",
        label="Fixed only"
    ),
    Line2D(
        [0], [0],
        color="tab:blue",
        lw=2,
        label="Adaptive only"
    ),
    Line2D(
        [0], [0],
        color="tab:green",
        lw=3,
        label="Overlap"
    ),
]

plt.legend(
    handles=legend_items,
    loc="best"
)

plt.title(
    "Comparison of Fixed Geographic Graph and Adaptive Learned Graph\n"
    f"Adaptive Top-{TOP_K} Neighbors per Corridor"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

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

print("\nSaved:")
print(OUT_PATH)