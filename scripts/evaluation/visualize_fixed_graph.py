from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

CORRIDOR_PATH = ROOT / "corridors.csv"
EDGE_PATH = ROOT / "outputs" / "results" / "fixed_graph_edges.csv"
FIGURE_DIR = ROOT / "outputs" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

corridors = pd.read_csv(CORRIDOR_PATH)
edges = pd.read_csv(EDGE_PATH)

nodes_used = sorted(
    set(edges["source"]) |
    set(edges["target"])
)

nodes = corridors[
    corridors["station_id"].isin(nodes_used)
].copy()

nodes = nodes.set_index("station_id")

print("\n=== FIXED GRAPH VISUALIZATION ===")
print(f"Nodes : {len(nodes)}")
print(f"Edges : {len(edges)}")


# =========================================================
# PLOT
# =========================================================
fig, ax = plt.subplots(
    figsize=(11, 9)
)

# Draw edges
for _, edge in edges.iterrows():

    source = edge["source"]
    target = edge["target"]

    x1 = nodes.loc[source, "lon"]
    y1 = nodes.loc[source, "lat"]

    x2 = nodes.loc[target, "lon"]
    y2 = nodes.loc[target, "lat"]

    ax.plot(
        [x1, x2],
        [y1, y2],
        linewidth=1,
        alpha=0.45
    )


# Draw nodes
ax.scatter(
    nodes["lon"],
    nodes["lat"],
    s=90,
    zorder=3
)


# Labels
for station_id, row in nodes.iterrows():

    label = station_id.replace(
        "tt_", ""
    )

    ax.annotate(
        label,
        (row["lon"], row["lat"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9
    )


ax.set_title(
    "Fixed Spatial Graph — Top 14 TomTom Corridors\n"
    "3-Nearest Neighbors Based on Haversine Distance"
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.grid(
    alpha=0.25
)

plt.tight_layout()

OUTPUT = (
    FIGURE_DIR /
    "fixed_graph_top14.png"
)

plt.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# DISTANCE AUDIT
# =========================================================
print("\n=== EDGE DISTANCE AUDIT ===")

print(
    edges["distance_km"]
    .describe()
    .round(2)
)

print("\n5 SHORTEST EDGES:")

print(
    edges
    .sort_values("distance_km")
    .head(5)
    .to_string(index=False)
)

print("\n5 LONGEST EDGES:")

print(
    edges
    .sort_values(
        "distance_km",
        ascending=False
    )
    .head(5)
    .to_string(index=False)
)


print("\n=== SAVED ===")
print("outputs/figures/fixed_graph_top14.png")