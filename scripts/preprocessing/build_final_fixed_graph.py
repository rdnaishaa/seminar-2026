from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT
    / "data"
    / "final_64"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "final_64"
    / "fixed_graph"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CORRIDORS_FILE = (
    ROOT
    / "corridors.csv"
)

NODE_FILE = (
    DATA_DIR
    / "final_64_corridors.csv"
)


# ============================================================
# CONFIG
# ============================================================

K = 4
EARTH_RADIUS_KM = 6371.0088


# ============================================================
# LOAD FINAL NODE ORDER
# ============================================================

nodes = pd.read_csv(
    NODE_FILE
)

nodes = nodes.sort_values(
    "node_index"
).reset_index(
    drop=True
)

corridor_files = (
    nodes[
        "corridor_file"
    ]
    .astype(str)
    .tolist()
)

NUM_NODES = len(
    corridor_files
)


print(
    "\n=== FINAL FIXED GRAPH ==="
)

print(
    f"Nodes : {NUM_NODES}"
)

print(
    f"K     : {K}"
)


# ============================================================
# LOAD CORRIDOR METADATA
# ============================================================

meta = pd.read_csv(
    CORRIDORS_FILE
)


# ============================================================
# DETECT COLUMN NAMES
# ============================================================

def find_column(
    df,
    candidates,
    label,
):

    lower_map = {
        c.lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        if (
            candidate.lower()
            in lower_map
        ):
            return lower_map[
                candidate.lower()
            ]

    raise ValueError(
        f"Tidak menemukan kolom {label}. "
        f"Available columns: "
        f"{list(df.columns)}"
    )


file_col = find_column(
    meta,
    [
        "station_id",
        "corridor_file",
        "file",
        "filename",
        "csv_file",
    ],
    "corridor identifier",
)

lat_col = find_column(
    meta,
    [
        "lat",
        "latitude",
    ],
    "latitude",
)

lon_col = find_column(
    meta,
    [
        "lon",
        "lng",
        "longitude",
    ],
    "longitude",
)


print(
    "\n=== METADATA COLUMNS ==="
)

print(
    f"File : {file_col}"
)

print(
    f"Lat  : {lat_col}"
)

print(
    f"Lon  : {lon_col}"
)


# ============================================================
# NORMALIZE FILENAMES
# ============================================================

meta[
    "_corridor_file"
] = (
    meta[
        file_col
    ]
    .astype(str)
    .map(
        lambda x:
        x
        if x.endswith(".csv")
        else f"{x}.csv"
    )
)


# ============================================================
# MATCH FINAL 64 WITH METADATA
# ============================================================

selected = (
    pd.DataFrame({
        "node_index":
            range(
                NUM_NODES
            ),

        "corridor_file":
            corridor_files,
    })
    .merge(
        meta,
        left_on="corridor_file",
        right_on="_corridor_file",
        how="left",
        validate="one_to_one",
    )
)


missing = selected[
    lat_col
].isna() | selected[
    lon_col
].isna()


if missing.any():

    missing_files = (
        selected.loc[
            missing,
            "corridor_file"
        ]
        .tolist()
    )

    raise ValueError(
        "Metadata koordinat tidak ditemukan "
        "untuk corridor berikut:\n"
        + "\n".join(
            missing_files
        )
    )


selected[
    lat_col
] = pd.to_numeric(
    selected[
        lat_col
    ],
    errors="coerce",
)

selected[
    lon_col
] = pd.to_numeric(
    selected[
        lon_col
    ],
    errors="coerce",
)


if (
    selected[
        [lat_col, lon_col]
    ]
    .isna()
    .any()
    .any()
):

    raise ValueError(
        "Ada latitude/longitude "
        "yang tidak numerik."
    )


# ============================================================
# HAVERSINE
# ============================================================

def haversine_matrix(
    latitudes,
    longitudes,
):

    lat = np.radians(
        np.asarray(
            latitudes,
            dtype=float,
        )
    )

    lon = np.radians(
        np.asarray(
            longitudes,
            dtype=float,
        )
    )

    lat1 = lat[
        :, None
    ]

    lat2 = lat[
        None, :
    ]

    lon1 = lon[
        :, None
    ]

    lon2 = lon[
        None, :
    ]

    dlat = (
        lat2
        - lat1
    )

    dlon = (
        lon2
        - lon1
    )

    a = (
        np.sin(
            dlat / 2
        ) ** 2

        + np.cos(
            lat1
        )

        * np.cos(
            lat2
        )

        * np.sin(
            dlon / 2
        ) ** 2
    )

    c = (
        2
        * np.arcsin(
            np.sqrt(a)
        )
    )

    return (
        EARTH_RADIUS_KM
        * c
    )


distance_matrix = (
    haversine_matrix(
        selected[
            lat_col
        ].values,

        selected[
            lon_col
        ].values,
    )
)


# ============================================================
# DIRECTED K-NN GRAPH
# ============================================================

directed_adj = np.zeros(
    (
        NUM_NODES,
        NUM_NODES,
    ),
    dtype=np.float32,
)


for i in range(
    NUM_NODES
):

    distances = (
        distance_matrix[
            i
        ]
        .copy()
    )

    # Jangan pilih diri sendiri
    distances[
        i
    ] = np.inf

    nearest = np.argsort(
        distances
    )[
        :K
    ]

    directed_adj[
        i,
        nearest
    ] = 1.0


# ============================================================
# SYMMETRIZE
#
# Jika A memilih B ATAU B memilih A,
# edge dianggap ada.
# ============================================================

adjacency = np.maximum(
    directed_adj,
    directed_adj.T,
)


# ============================================================
# ADD SELF-LOOPS
# ============================================================

adjacency_with_self = (
    adjacency.copy()
)

np.fill_diagonal(
    adjacency_with_self,
    1.0,
)


# ============================================================
# NORMALIZED ADJACENCY
#
# A_norm = D^(-1/2) A D^(-1/2)
# ============================================================

degree = adjacency_with_self.sum(axis=1)

degree_inv_sqrt = 1.0 / np.sqrt(degree)

normalized_adjacency = (
    adjacency_with_self
    * degree_inv_sqrt[:, None]
    * degree_inv_sqrt[None, :]
).astype(np.float32)


# ============================================================
# BUILD EDGE TABLE
# ============================================================

edge_rows = []

for i in range(
    NUM_NODES
):

    for j in range(
        i + 1,
        NUM_NODES
    ):

        if (
            adjacency[
                i,
                j
            ]
            > 0
        ):

            edge_rows.append({
                "source_index":
                    i,

                "target_index":
                    j,

                "source":
                    corridor_files[
                        i
                    ],

                "target":
                    corridor_files[
                        j
                    ],

                "distance_km":
                    float(
                        distance_matrix[
                            i,
                            j
                        ]
                    ),
            })


edges_df = pd.DataFrame(
    edge_rows
)


# ============================================================
# DEGREE STATISTICS
# ============================================================

degree_without_self = (
    adjacency
    .sum(
        axis=1
    )
)


print(
    "\n=== GRAPH SUMMARY ==="
)

print(
    f"Undirected edges : "
    f"{len(edges_df)}"
)

print(
    f"Self-loops       : "
    f"{NUM_NODES}"
)

print(
    f"Min degree       : "
    f"{int(degree_without_self.min())}"
)

print(
    f"Max degree       : "
    f"{int(degree_without_self.max())}"
)

print(
    f"Mean degree      : "
    f"{degree_without_self.mean():.2f}"
)


# ============================================================
# CHECK CONNECTIVITY
# ============================================================

def connected_components(
    adj
):

    visited = set()
    components = []

    for start in range(
        len(adj)
    ):

        if start in visited:
            continue

        stack = [
            start
        ]

        component = []

        while stack:

            node = (
                stack.pop()
            )

            if node in visited:
                continue

            visited.add(
                node
            )

            component.append(
                node
            )

            neighbors = np.where(
                adj[
                    node
                ]
                > 0
            )[0]

            for neighbor in neighbors:

                neighbor = int(
                    neighbor
                )

                if (
                    neighbor
                    not in visited
                ):

                    stack.append(
                        neighbor
                    )

        components.append(
            component
        )

    return components


components = (
    connected_components(
        adjacency
    )
)

print(
    f"Connected components : "
    f"{len(components)}"
)

print(
    "Component sizes      : "
    + str(
        [
            len(c)
            for c in components
        ]
    )
)


# ============================================================
# SAVE GRAPH ARRAYS
# ============================================================

graph_file = (
    OUTPUT_DIR
    / "fixed_graph_64.npz"
)

np.savez_compressed(
    graph_file,

    adjacency=
        adjacency,

    adjacency_with_self=
        adjacency_with_self,

    normalized_adjacency=
        normalized_adjacency,

    distance_matrix=
        distance_matrix.astype(
            np.float32
        ),

    directed_knn_adjacency=
        directed_adj,
)


# ============================================================
# SAVE EDGES
# ============================================================

edge_file = (
    OUTPUT_DIR
    / "fixed_graph_64_edges.csv"
)

edges_df.to_csv(
    edge_file,
    index=False,
)


# ============================================================
# SAVE NODE METADATA
# ============================================================

node_output = selected[
    [
        "node_index",
        "corridor_file",
        lat_col,
        lon_col,
    ]
].copy()

node_output = node_output.rename(
    columns={
        lat_col:
            "latitude",

        lon_col:
            "longitude",
    }
)

node_output[
    "degree"
] = (
    degree_without_self
    .astype(int)
)

node_output.to_csv(
    OUTPUT_DIR
    / "fixed_graph_64_nodes.csv",
    index=False,
)


# ============================================================
# FINAL VALIDATION
# ============================================================

assert (
    adjacency.shape
    == (
        NUM_NODES,
        NUM_NODES,
    )
)

assert np.allclose(
    adjacency,
    adjacency.T,
)

assert np.all(
    np.diag(
        adjacency
    )
    == 0
)

assert np.all(
    np.diag(
        adjacency_with_self
    )
    == 1
)

assert np.all(
    np.isfinite(
        normalized_adjacency
    )
)


print(
    "\nGraph validation : PASS"
)

print(
    "\n=== SAVED ==="
)

print(
    graph_file
)

print(
    edge_file
)

print(
    OUTPUT_DIR
    / "fixed_graph_64_nodes.csv"
)