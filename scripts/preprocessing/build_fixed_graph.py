from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CORRIDOR_PATH = ROOT / "corridors.csv"
TRAIN_PATH = ROOT / "data" / "processed" / "forecasting_train_preprocessed.csv"
RESULT_DIR = ROOT / "outputs" / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

K = 3


# =========================================================
# LOAD DATA
# =========================================================
corridors = pd.read_csv(CORRIDOR_PATH)
train = pd.read_csv(TRAIN_PATH)

top14 = sorted(train["corridor_file"].unique())

corridors = corridors[
    corridors["station_id"].isin(top14)
].copy()

corridors = (
    corridors
    .set_index("station_id")
    .loc[top14]
    .reset_index()
)

print("\n=== FIXED GRAPH BUILD ===")
print(f"Nodes : {len(corridors)}")
print(f"KNN   : k={K}")


# =========================================================
# HAVERSINE DISTANCE
# =========================================================
def haversine(lat1, lon1, lat2, lon2):

    R = 6371.0  # km

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    return R * c


# =========================================================
# DISTANCE MATRIX
# =========================================================
n = len(corridors)

distance_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):

        if i == j:
            distance_matrix[i, j] = 0
            continue

        distance_matrix[i, j] = haversine(
            corridors.loc[i, "lat"],
            corridors.loc[i, "lon"],
            corridors.loc[j, "lat"],
            corridors.loc[j, "lon"]
        )


distance_df = pd.DataFrame(
    distance_matrix,
    index=top14,
    columns=top14
)

distance_df.to_csv(
    RESULT_DIR / "fixed_graph_distance_matrix.csv"
)


# =========================================================
# BUILD KNN ADJACENCY
# =========================================================
adjacency = np.zeros((n, n))

edges = []

for i in range(n):

    distances = distance_matrix[i].copy()

    distances[i] = np.inf

    nearest = np.argsort(distances)[:K]

    for j in nearest:

        adjacency[i, j] = 1

        edges.append({
            "source": top14[i],
            "target": top14[j],
            "distance_km": distance_matrix[i, j]
        })


# =========================================================
# MAKE GRAPH UNDIRECTED
# =========================================================
adjacency = np.maximum(
    adjacency,
    adjacency.T
)


# Self-loop
np.fill_diagonal(
    adjacency,
    1
)


adjacency_df = pd.DataFrame(
    adjacency.astype(int),
    index=top14,
    columns=top14
)

adjacency_df.to_csv(
    RESULT_DIR / "fixed_graph_adjacency.csv"
)


# =========================================================
# UNIQUE EDGE LIST
# =========================================================
unique_edges = []

for i in range(n):

    for j in range(i + 1, n):

        if adjacency[i, j] == 1:

            unique_edges.append({
                "source": top14[i],
                "target": top14[j],
                "distance_km": distance_matrix[i, j]
            })


edges_df = pd.DataFrame(
    unique_edges
)

edges_df.to_csv(
    RESULT_DIR / "fixed_graph_edges.csv",
    index=False
)


# =========================================================
# SUMMARY
# =========================================================
degrees = (
    adjacency.sum(axis=1) - 1
)

print("\n=== GRAPH SUMMARY ===")

print(
    f"Undirected edges : {len(unique_edges)}"
)

print(
    f"Average degree   : {degrees.mean():.2f}"
)

print(
    f"Min degree       : {degrees.min():.0f}"
)

print(
    f"Max degree       : {degrees.max():.0f}"
)


print("\n=== NODE DEGREE ===")

for node, degree in zip(
    top14,
    degrees
):

    print(
        f"{node:<20} degree={int(degree)}"
    )


print("\n=== EDGES ===")

for edge in unique_edges:

    print(
        f"{edge['source']:<20} "
        f"<-> "
        f"{edge['target']:<20} "
        f"{edge['distance_km']:.2f} km"
    )


print("\n=== SAVED ===")
print("outputs/results/fixed_graph_distance_matrix.csv")
print("outputs/results/fixed_graph_adjacency.csv")
print("outputs/results/fixed_graph_edges.csv")