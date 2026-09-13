from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "final_64"

FIXED_DIR = (
    ROOT
    / "outputs"
    / "final_64"
    / "fixed_graph"
)

ADAPTIVE_DIR = (
    ROOT
    / "outputs"
    / "final_64"
    / "adaptive_stgnn"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "final_64"
    / "analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEEDS = [42, 123, 456, 789, 2026]

TOP_K = 4


# ============================================================
# LOAD NODE ORDER
# ============================================================

node_df = (
    pd.read_csv(
        DATA_DIR / "final_64_corridors.csv"
    )
    .sort_values("node_index")
    .reset_index(drop=True)
)

corridors = (
    node_df["corridor_file"]
    .astype(str)
    .tolist()
)

NUM_NODES = len(corridors)

assert NUM_NODES == 64


# ============================================================
# LOAD FIXED GRAPH
# ============================================================

fixed_npz = np.load(
    FIXED_DIR / "fixed_graph_64.npz"
)

print(
    "\nAvailable keys in fixed_graph_64.npz:"
)

print(
    fixed_npz.files
)


# Try common adjacency key names
if "adjacency" in fixed_npz.files:

    fixed_adj = (
        fixed_npz["adjacency"]
        .astype(np.float32)
    )

elif "adjacency_with_self" in fixed_npz.files:

    fixed_adj = (
        fixed_npz["adjacency_with_self"]
        .astype(np.float32)
    )

elif "normalized_adjacency" in fixed_npz.files:

    fixed_adj = (
        fixed_npz["normalized_adjacency"]
        .astype(np.float32)
    )

else:

    raise KeyError(
        "Tidak menemukan adjacency pada "
        "fixed_graph_64.npz. "
        f"Available keys: {fixed_npz.files}"
    )


assert fixed_adj.shape == (
    NUM_NODES,
    NUM_NODES,
)


# Remove self-loop for edge comparison
fixed_no_diag = fixed_adj.copy()

np.fill_diagonal(
    fixed_no_diag,
    0.0,
)


# Binary representation
fixed_binary = (
    fixed_no_diag > 0
).astype(np.int8)


# ============================================================
# LOAD ADAPTIVE GRAPHS
# ============================================================

adaptive_graphs = {}

for seed in SEEDS:

    path = (
        ADAPTIVE_DIR
        / f"adaptive_stgnn_seed{seed}_adjacency.npy"
    )

    adjacency = np.load(
        path
    ).astype(
        np.float32
    )

    assert adjacency.shape == (
        NUM_NODES,
        NUM_NODES,
    )

    adaptive_graphs[seed] = adjacency


adaptive_stack = np.stack(
    [
        adaptive_graphs[seed]
        for seed in SEEDS
    ],
    axis=0,
)


adaptive_mean = np.mean(
    adaptive_stack,
    axis=0,
)


adaptive_std = np.std(
    adaptive_stack,
    axis=0,
)


np.save(
    OUTPUT_DIR
    / "adaptive_mean_adjacency.npy",
    adaptive_mean,
)


np.save(
    OUTPUT_DIR
    / "adaptive_std_adjacency.npy",
    adaptive_std,
)


pd.DataFrame(
    adaptive_mean,
    index=corridors,
    columns=corridors,
).to_csv(
    OUTPUT_DIR
    / "adaptive_mean_adjacency.csv"
)


pd.DataFrame(
    adaptive_std,
    index=corridors,
    columns=corridors,
).to_csv(
    OUTPUT_DIR
    / "adaptive_std_adjacency.csv"
)


# ============================================================
# FUNCTION: TOP-K ADAPTIVE EDGES
# ============================================================

def top_k_binary(
    adjacency,
    k=4,
):

    adjacency = (
        adjacency.copy()
    )


    # Remove self-loop
    np.fill_diagonal(
        adjacency,
        -np.inf,
    )


    directed = np.zeros_like(
        adjacency,
        dtype=np.int8,
    )


    for i in range(
        adjacency.shape[0]
    ):

        idx = np.argsort(
            adjacency[i]
        )[
            -k:
        ]

        directed[
            i,
            idx
        ] = 1


    # Convert directed top-k
    # into undirected graph
    undirected = np.maximum(
        directed,
        directed.T,
    )


    np.fill_diagonal(
        undirected,
        0,
    )


    return undirected


# ============================================================
# FUNCTION: EDGE SET
# ============================================================

def edge_set(
    binary_adj,
):

    edges = set()

    n = binary_adj.shape[0]

    for i in range(n):

        for j in range(
            i + 1,
            n
        ):

            if binary_adj[
                i,
                j
            ] > 0:

                edges.add(
                    (
                        i,
                        j,
                    )
                )

    return edges


# ============================================================
# ADAPTIVE TOP-K PER SEED
# ============================================================

adaptive_binary = {}

adaptive_edges = {}


for seed in SEEDS:

    binary = top_k_binary(
        adaptive_graphs[seed],
        TOP_K,
    )

    adaptive_binary[seed] = binary

    adaptive_edges[seed] = edge_set(
        binary
    )


# Mean graph top-K
adaptive_mean_binary = top_k_binary(
    adaptive_mean,
    TOP_K,
)


adaptive_mean_edges = edge_set(
    adaptive_mean_binary
)


fixed_edges = edge_set(
    fixed_binary
)


# ============================================================
# BASIC EDGE COMPARISON
# ============================================================

overlap = (
    fixed_edges
    & adaptive_mean_edges
)

fixed_only = (
    fixed_edges
    - adaptive_mean_edges
)

adaptive_only = (
    adaptive_mean_edges
    - fixed_edges
)


union = (
    fixed_edges
    | adaptive_mean_edges
)


jaccard = (
    len(overlap)
    / len(union)
    if len(union) > 0
    else np.nan
)


print(
    "\n"
    + "=" * 60
)

print(
    "FINAL GRAPH COMPARISON"
)

print(
    "=" * 60
)

print(
    f"Fixed edges        : "
    f"{len(fixed_edges)}"
)

print(
    f"Adaptive mean edges: "
    f"{len(adaptive_mean_edges)}"
)

print(
    f"Overlap            : "
    f"{len(overlap)}"
)

print(
    f"Fixed-only         : "
    f"{len(fixed_only)}"
)

print(
    f"Adaptive-only      : "
    f"{len(adaptive_only)}"
)

print(
    f"Jaccard similarity : "
    f"{jaccard:.4f}"
)


# ============================================================
# SAVE EDGE LISTS
# ============================================================

def edges_to_df(
    edges,
    label,
):

    rows = []

    for i, j in sorted(
        edges
    ):

        rows.append(
            {
                "edge_type":
                    label,

                "node_i":
                    i,

                "corridor_i":
                    corridors[i],

                "node_j":
                    j,

                "corridor_j":
                    corridors[j],
            }
        )

    return pd.DataFrame(
        rows
    )


edge_tables = [
    edges_to_df(
        overlap,
        "overlap",
    ),

    edges_to_df(
        fixed_only,
        "fixed_only",
    ),

    edges_to_df(
        adaptive_only,
        "adaptive_only",
    ),
]


edge_comparison_df = pd.concat(
    edge_tables,
    ignore_index=True,
)


edge_comparison_df.to_csv(
    OUTPUT_DIR
    / "fixed_vs_adaptive_edges.csv",
    index=False,
)


summary_df = pd.DataFrame(
    [
        {
            "fixed_edges":
                len(
                    fixed_edges
                ),

            "adaptive_mean_edges":
                len(
                    adaptive_mean_edges
                ),

            "overlap_edges":
                len(
                    overlap
                ),

            "fixed_only_edges":
                len(
                    fixed_only
                ),

            "adaptive_only_edges":
                len(
                    adaptive_only
                ),

            "jaccard_similarity":
                jaccard,

            "top_k":
                TOP_K,
        }
    ]
)


summary_df.to_csv(
    OUTPUT_DIR
    / "graph_comparison_summary.csv",
    index=False,
)


# ============================================================
# EDGE STABILITY ACROSS SEEDS
# ============================================================

all_possible_edges = set()

for seed in SEEDS:

    all_possible_edges.update(
        adaptive_edges[seed]
    )


stability_rows = []


for edge in sorted(
    all_possible_edges
):

    count = sum(
        edge
        in adaptive_edges[seed]
        for seed in SEEDS
    )


    stability = (
        count
        / len(
            SEEDS
        )
    )


    i, j = edge


    stability_rows.append(
        {
            "node_i":
                i,

            "corridor_i":
                corridors[i],

            "node_j":
                j,

            "corridor_j":
                corridors[j],

            "seed_count":
                count,

            "stability":
                stability,

            "present_in_fixed":
                edge
                in fixed_edges,
        }
    )


stability_df = pd.DataFrame(
    stability_rows
)


stability_df = (
    stability_df
    .sort_values(
        [
            "seed_count",
            "corridor_i",
            "corridor_j",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


stability_df.to_csv(
    OUTPUT_DIR
    / "adaptive_edge_stability.csv",
    index=False,
)


# ============================================================
# PAIRWISE JACCARD BETWEEN ADAPTIVE SEEDS
# ============================================================

pairwise_rows = []


for seed_a, seed_b in combinations(
    SEEDS,
    2,
):

    edges_a = (
        adaptive_edges[
            seed_a
        ]
    )

    edges_b = (
        adaptive_edges[
            seed_b
        ]
    )


    intersection = (
        edges_a
        & edges_b
    )


    union_pair = (
        edges_a
        | edges_b
    )


    pairwise_jaccard = (
        len(
            intersection
        )
        / len(
            union_pair
        )
        if len(
            union_pair
        ) > 0
        else np.nan
    )


    pairwise_rows.append(
        {
            "seed_a":
                seed_a,

            "seed_b":
                seed_b,

            "edges_a":
                len(
                    edges_a
                ),

            "edges_b":
                len(
                    edges_b
                ),

            "overlap_edges":
                len(
                    intersection
                ),

            "jaccard_similarity":
                pairwise_jaccard,
        }
    )


pairwise_df = pd.DataFrame(
    pairwise_rows
)


pairwise_df.to_csv(
    OUTPUT_DIR
    / "adaptive_pairwise_seed_similarity.csv",
    index=False,
)


# ============================================================
# EDGE STABILITY SUMMARY
# ============================================================

stability_summary = (
    stability_df
    .groupby(
        "seed_count"
    )
    .size()
    .reset_index(
        name="edge_count"
    )
)


stability_summary[
    "stability"
] = (
    stability_summary[
        "seed_count"
    ]
    / len(
        SEEDS
    )
)


stability_summary.to_csv(
    OUTPUT_DIR
    / "adaptive_edge_stability_summary.csv",
    index=False,
)


# ============================================================
# FIGURE 1: FIXED GRAPH HEATMAP
# ============================================================

plt.figure(
    figsize=(
        10,
        8,
    )
)

plt.imshow(
    fixed_binary,
    aspect="auto",
)

plt.colorbar(
    label="Edge presence"
)

plt.title(
    "Fixed Geographic Graph (K=4)"
)

plt.xlabel(
    "Node index"
)

plt.ylabel(
    "Node index"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "fig_fixed_graph_heatmap.png",
    dpi=300,
)

plt.close()


# ============================================================
# FIGURE 2: ADAPTIVE MEAN ADJACENCY
# ============================================================

adaptive_mean_no_diag = (
    adaptive_mean.copy()
)

np.fill_diagonal(
    adaptive_mean_no_diag,
    0.0,
)


plt.figure(
    figsize=(
        10,
        8,
    )
)

plt.imshow(
    adaptive_mean_no_diag,
    aspect="auto",
)

plt.colorbar(
    label="Normalized learned weight"
)

plt.title(
    "Adaptive Graph Mean Adjacency Across 5 Seeds"
)

plt.xlabel(
    "Node index"
)

plt.ylabel(
    "Node index"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "fig_adaptive_mean_heatmap.png",
    dpi=300,
)

plt.close()


# ============================================================
# FIGURE 3: ADAPTIVE TOP-K HEATMAP
# ============================================================

plt.figure(
    figsize=(
        10,
        8,
    )
)

plt.imshow(
    adaptive_mean_binary,
    aspect="auto",
)

plt.colorbar(
    label="Top-K edge presence"
)

plt.title(
    f"Adaptive Mean Graph Top-{TOP_K}"
)

plt.xlabel(
    "Node index"
)

plt.ylabel(
    "Node index"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "fig_adaptive_topk_heatmap.png",
    dpi=300,
)

plt.close()


# ============================================================
# FIGURE 4: GRAPH DIFFERENCE MATRIX
#
#  1  = adaptive only
# -1  = fixed only
#  0  = same / neither
# ============================================================

difference = (
    adaptive_mean_binary
    - fixed_binary
)


plt.figure(
    figsize=(
        10,
        8,
    )
)

plt.imshow(
    difference,
    aspect="auto",
)

plt.colorbar(
    label=(
        "-1 Fixed-only | "
        "0 Same | "
        "1 Adaptive-only"
    )
)

plt.title(
    "Adaptive vs Fixed Edge Difference"
)

plt.xlabel(
    "Node index"
)

plt.ylabel(
    "Node index"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "fig_graph_difference.png",
    dpi=300,
)

plt.close()


# ============================================================
# FIGURE 5: EDGE CATEGORY COUNTS
# ============================================================

categories = [
    "Overlap",
    "Fixed-only",
    "Adaptive-only",
]

counts = [
    len(
        overlap
    ),
    len(
        fixed_only
    ),
    len(
        adaptive_only
    ),
]


plt.figure(
    figsize=(
        8,
        5,
    )
)

plt.bar(
    categories,
    counts,
)

plt.ylabel(
    "Number of undirected edges"
)

plt.title(
    "Fixed vs Adaptive Edge Comparison"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "fig_edge_overlap_counts.png",
    dpi=300,
)

plt.close()


# ============================================================
# FIGURE 6: EDGE STABILITY ACROSS SEEDS
# ============================================================

stability_plot = (
    stability_summary
    .sort_values(
        "seed_count"
    )
)


plt.figure(
    figsize=(
        8,
        5,
    )
)

plt.bar(
    stability_plot[
        "seed_count"
    ].astype(str),
    stability_plot[
        "edge_count"
    ],
)

plt.xlabel(
    "Number of seeds containing edge"
)

plt.ylabel(
    "Number of edges"
)

plt.title(
    "Adaptive Edge Stability Across 5 Seeds"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "fig_adaptive_edge_stability.png",
    dpi=300,
)

plt.close()


# ============================================================
# FINAL PRINT
# ============================================================

print(
    "\n=== ADAPTIVE SEED STABILITY ==="
)

print(
    pairwise_df[
        [
            "seed_a",
            "seed_b",
            "overlap_edges",
            "jaccard_similarity",
        ]
    ].to_string(
        index=False
    )
)


mean_pairwise_jaccard = (
    pairwise_df[
        "jaccard_similarity"
    ]
    .mean()
)


print(
    "\nMean pairwise adaptive "
    f"Jaccard: "
    f"{mean_pairwise_jaccard:.4f}"
)


fully_stable_edges = (
    stability_df[
        stability_df[
            "seed_count"
        ]
        == len(
            SEEDS
        )
    ]
)


print(
    "Edges present in all 5 seeds: "
    f"{len(fully_stable_edges)}"
)


print(
    "\n=== OUTPUT SAVED TO ==="
)

print(
    OUTPUT_DIR
)

print(
    "\nFigures generated:"
)

print(
    "- fig_fixed_graph_heatmap.png"
)

print(
    "- fig_adaptive_mean_heatmap.png"
)

print(
    "- fig_adaptive_topk_heatmap.png"
)

print(
    "- fig_graph_difference.png"
)

print(
    "- fig_edge_overlap_counts.png"
)

print(
    "- fig_adaptive_edge_stability.png"
)