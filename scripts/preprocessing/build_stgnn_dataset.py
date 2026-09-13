from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
RESULT_DIR = ROOT / "outputs" / "results"

INPUT_LENGTH = 12
HORIZONS = [1, 3, 6]
TARGET = "current_speed"

splits = ["train", "val", "test"]

# =========================================================
# LOAD CORRIDOR ORDER FROM ADJACENCY
# =========================================================
adjacency_df = pd.read_csv(
    RESULT_DIR / "fixed_graph_adjacency.csv",
    index_col=0
)

corridors = adjacency_df.index.tolist()

print("\n=== ST-GNN DATASET ===")
print(f"Nodes        : {len(corridors)}")
print(f"Input length : {INPUT_LENGTH}")
print(f"Horizons     : {HORIZONS}")

print("\nNode order:")
for i, corridor in enumerate(corridors):
    print(f"{i:2d}: {corridor}")


# =========================================================
# LOAD SPLIT
# =========================================================
def load_split(split):

    path = (
        DATA_DIR /
        f"forecasting_{split}_preprocessed.csv"
    )

    df = pd.read_csv(path)

    df["obs_time_utc"] = pd.to_datetime(
        df["obs_time_utc"],
        utc=True
    )

    if df["is_observed"].dtype == object:
        df["is_observed"] = (
            df["is_observed"]
            .astype(str)
            .str.lower()
            .map({
                "true": True,
                "false": False
            })
        )

    return df


# =========================================================
# BUILD MATRICES
# =========================================================
def build_matrices(df):

    df = df.dropna(
        subset=["segment_id"]
    ).copy()

    speed = df.pivot(
        index="obs_time_utc",
        columns="corridor_file",
        values=TARGET
    )

    observed = df.pivot(
        index="obs_time_utc",
        columns="corridor_file",
        values="is_observed"
    )

    speed = (
        speed
        .reindex(columns=corridors)
        .sort_index()
    )

    observed = (
        observed
        .reindex(columns=corridors)
        .sort_index()
    )

    segment = (
        df[
            ["obs_time_utc", "segment_id"]
        ]
        .drop_duplicates()
        .set_index("obs_time_utc")
        ["segment_id"]
        .reindex(speed.index)
    )

    return speed, observed, segment


data = {}

for split in splits:

    df = load_split(split)

    speed, observed, segment = (
        build_matrices(df)
    )

    data[split] = {
        "speed": speed,
        "observed": observed,
        "segment": segment
    }


# =========================================================
# FIT SCALER — TRAIN ONLY
# =========================================================
scaler = StandardScaler()

train_values = (
    data["train"]["speed"]
    .values
    .reshape(-1, 1)
)

scaler.fit(train_values)

print("\n=== SCALER ===")
print(f"Train mean : {scaler.mean_[0]:.4f}")
print(f"Train std  : {scaler.scale_[0]:.4f}")


# =========================================================
# BUILD WINDOWS
# =========================================================
def build_windows(split):

    speed_df = data[split]["speed"]
    observed_df = data[split]["observed"]
    segment_series = data[split]["segment"]

    raw = speed_df.values

    scaled = scaler.transform(
        raw.reshape(-1, 1)
    ).reshape(raw.shape)

    observed = observed_df.values.astype(bool)

    segments = segment_series.values
    timestamps = speed_df.index.to_numpy()

    X = []
    y = []
    mask = []
    target_times = []

    max_horizon = max(HORIZONS)

    for start in range(
        len(raw) - INPUT_LENGTH - max_horizon + 1
    ):

        input_end = (
            start + INPUT_LENGTH - 1
        )

        target_indices = [
            input_end + h
            for h in HORIZONS
        ]

        used_indices = (
            list(
                range(
                    start,
                    input_end + 1
                )
            )
            + target_indices
        )

        # Do not cross temporal segment
        if len(
            set(segments[used_indices])
        ) != 1:
            continue

        x = scaled[
            start:
            input_end + 1
        ]

        targets = scaled[
            target_indices
        ]

        target_mask = observed[
            target_indices
        ]

        times = timestamps[
            target_indices
        ]

        X.append(x)
        y.append(targets)
        mask.append(target_mask)
        target_times.append(times)

    X = np.array(X)[..., np.newaxis]

    y = np.array(y)

    mask = np.array(mask)

    target_times = np.array(
        target_times
    )

    return X, y, mask, target_times


# =========================================================
# BUILD + SAVE
# =========================================================
for split in splits:

    X, y, mask, times = (
        build_windows(split)
    )

    print(f"\n=== {split.upper()} ===")
    print(f"X shape    : {X.shape}")
    print(f"y shape    : {y.shape}")
    print(f"mask shape : {mask.shape}")

    print(
        f"Observed target points : "
        f"{mask.sum()}"
    )

    np.savez_compressed(
        DATA_DIR /
        f"stgnn_{split}.npz",
        X=X,
        y=y,
        mask=mask,
        target_times=times
    )


# =========================================================
# SAVE SCALER
# =========================================================
np.savez(
    DATA_DIR / "stgnn_scaler.npz",
    mean=scaler.mean_,
    scale=scaler.scale_
)


print("\n=== SAVED ===")
print("data/processed/stgnn_train.npz")
print("data/processed/stgnn_val.npz")
print("data/processed/stgnn_test.npz")
print("data/processed/stgnn_scaler.npz")