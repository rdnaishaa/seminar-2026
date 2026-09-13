from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT
    / "data"
    / "final_64"
)

INPUT_LENGTH = 12
HORIZONS = [1, 3, 6]
TARGET = "current_speed"

SPLITS = [
    "train",
    "val",
    "test",
]


# ============================================================
# LOAD FINAL NODE ORDER
# ============================================================

node_file = (
    DATA_DIR
    / "final_64_corridors.csv"
)

node_df = pd.read_csv(
    node_file
)

corridors = (
    node_df
    .sort_values("node_index")
    ["corridor_file"]
    .tolist()
)

NUM_NODES = len(corridors)


print("\n=== FINAL ST-GNN DATASET ===")
print(f"Nodes        : {NUM_NODES}")
print(f"Input length : {INPUT_LENGTH}")
print(f"Horizons     : {HORIZONS}")


# ============================================================
# LOAD SPLIT
# ============================================================

def load_split(split_name):

    path = (
        DATA_DIR
        / f"forecasting_{split_name}_preprocessed.csv"
    )

    df = pd.read_csv(path)

    df["obs_time_utc"] = pd.to_datetime(
        df["obs_time_utc"],
        utc=True
    )

    # pastikan boolean benar
    for col in [
        "is_observed",
        "was_imputed",
    ]:

        if col in df.columns:

            if df[col].dtype == object:

                df[col] = (
                    df[col]
                    .astype(str)
                    .str.lower()
                    .map({
                        "true": True,
                        "false": False,
                    })
                )

    return df


# ============================================================
# BUILD MATRICES
# ============================================================

def build_matrices(df):

    valid_df = (
        df
        .dropna(
            subset=["segment_id"]
        )
        .copy()
    )

    speed = (
        valid_df
        .pivot(
            index="obs_time_utc",
            columns="corridor_file",
            values=TARGET
        )
        .reindex(
            columns=corridors
        )
        .sort_index()
    )

    observed = (
        valid_df
        .pivot(
            index="obs_time_utc",
            columns="corridor_file",
            values="is_observed"
        )
        .reindex(
            columns=corridors
        )
        .sort_index()
    )

    imputed = (
        valid_df
        .pivot(
            index="obs_time_utc",
            columns="corridor_file",
            values="was_imputed"
        )
        .reindex(
            columns=corridors
        )
        .sort_index()
    )

    segment = (
        valid_df[
            [
                "obs_time_utc",
                "segment_id",
            ]
        ]
        .drop_duplicates()
        .set_index(
            "obs_time_utc"
        )
        ["segment_id"]
        .reindex(
            speed.index
        )
    )

    return (
        speed,
        observed,
        imputed,
        segment,
    )


# ============================================================
# LOAD ALL DATA
# ============================================================

data = {}

for split_name in SPLITS:

    df = load_split(
        split_name
    )

    (
        speed,
        observed,
        imputed,
        segment,
    ) = build_matrices(df)

    data[split_name] = {
        "speed":
            speed,

        "observed":
            observed,

        "imputed":
            imputed,

        "segment":
            segment,
    }

    print(
        f"\n=== {split_name.upper()} MATRIX ==="
    )

    print(
        f"Speed shape : "
        f"{speed.shape}"
    )

    print(
        f"Time range  : "
        f"{speed.index.min()} "
        f"-> "
        f"{speed.index.max()}"
    )


# ============================================================
# VALIDATION BEFORE SCALING
# ============================================================

for split_name in SPLITS:

    speed = (
        data[split_name]["speed"]
    )

    missing = int(
        speed.isna().sum().sum()
    )

    if missing > 0:

        raise ValueError(
            f"{split_name}: "
            f"{missing} NaN ditemukan "
            f"di valid speed matrix."
        )

    if (
        speed.shape[1]
        != NUM_NODES
    ):

        raise ValueError(
            f"{split_name}: "
            f"expected {NUM_NODES} nodes, "
            f"got {speed.shape[1]}"
        )


# ============================================================
# FIT SCALER ON TRAIN ONLY
# ============================================================

scaler = StandardScaler()

train_values = (
    data["train"]["speed"]
    .values
    .reshape(-1, 1)
)

scaler.fit(
    train_values
)

print(
    "\n=== SCALER ==="
)

print(
    f"Train mean : "
    f"{scaler.mean_[0]:.4f}"
)

print(
    f"Train std  : "
    f"{scaler.scale_[0]:.4f}"
)


# ============================================================
# BUILD WINDOWS
# ============================================================

def build_windows(split_name):

    speed_df = (
        data[split_name]["speed"]
    )

    observed_df = (
        data[split_name]["observed"]
    )

    imputed_df = (
        data[split_name]["imputed"]
    )

    segment_series = (
        data[split_name]["segment"]
    )

    raw = (
        speed_df
        .values
        .astype(np.float32)
    )

    scaled = (
        scaler
        .transform(
            raw.reshape(-1, 1)
        )
        .reshape(
            raw.shape
        )
        .astype(
            np.float32
        )
    )

    observed = (
        observed_df
        .values
        .astype(bool)
    )

    imputed = (
        imputed_df
        .values
        .astype(bool)
    )

    segments = (
        segment_series
        .values
    )

    timestamps = (
        speed_df
        .index
        .to_numpy()
    )

    X = []
    y = []

    target_mask = []
    target_times = []

    input_observed_mask = []
    input_imputed_mask = []

    max_horizon = max(
        HORIZONS
    )

    for start in range(
        len(raw)
        - INPUT_LENGTH
        - max_horizon
        + 1
    ):

        input_end = (
            start
            + INPUT_LENGTH
            - 1
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

        used_segments = (
            segments[
                used_indices
            ]
        )

        # Tidak boleh cross temporal segment
        if (
            len(
                set(
                    used_segments
                )
            )
            != 1
        ):
            continue

        x = (
            scaled[
                start:
                input_end + 1
            ]
        )

        targets = (
            scaled[
                target_indices
            ]
        )

        target_obs = (
            observed[
                target_indices
            ]
        )

        times = (
            timestamps[
                target_indices
            ]
        )

        input_obs = (
            observed[
                start:
                input_end + 1
            ]
        )

        input_imp = (
            imputed[
                start:
                input_end + 1
            ]
        )

        X.append(
            x
        )

        y.append(
            targets
        )

        target_mask.append(
            target_obs
        )

        target_times.append(
            times
        )

        input_observed_mask.append(
            input_obs
        )

        input_imputed_mask.append(
            input_imp
        )

    X = (
        np.array(
            X,
            dtype=np.float32
        )[..., np.newaxis]
    )

    y = np.array(
        y,
        dtype=np.float32
    )

    target_mask = np.array(
        target_mask,
        dtype=bool
    )

    target_times = np.array(
        target_times
    )

    input_observed_mask = np.array(
        input_observed_mask,
        dtype=bool
    )

    input_imputed_mask = np.array(
        input_imputed_mask,
        dtype=bool
    )

    return (
        X,
        y,
        target_mask,
        target_times,
        input_observed_mask,
        input_imputed_mask,
    )


# ============================================================
# BUILD + SAVE
# ============================================================

summary_rows = []

for split_name in SPLITS:

    (
        X,
        y,
        mask,
        times,
        input_obs_mask,
        input_imp_mask,
    ) = build_windows(
        split_name
    )

    print(
        f"\n=== {split_name.upper()} WINDOWS ==="
    )

    print(
        f"X shape       : "
        f"{X.shape}"
    )

    print(
        f"y shape       : "
        f"{y.shape}"
    )

    print(
        f"mask shape    : "
        f"{mask.shape}"
    )

    print(
        f"target times  : "
        f"{times.shape}"
    )

    print(
        f"Observed target points : "
        f"{int(mask.sum())}"
    )

    print(
        f"Imputed input points   : "
        f"{int(input_imp_mask.sum())}"
    )

    total_input_points = (
        input_imp_mask.size
    )

    imputed_input_pct = (
        input_imp_mask.sum()
        / total_input_points
        * 100
        if total_input_points > 0
        else 0
    )

    print(
        f"Imputed input ratio    : "
        f"{imputed_input_pct:.2f}%"
    )

    output_file = (
        DATA_DIR
        / f"stgnn_{split_name}.npz"
    )

    np.savez_compressed(
        output_file,

        X=X,
        y=y,

        mask=mask,

        target_times=times,

        input_observed_mask=
            input_obs_mask,

        input_imputed_mask=
            input_imp_mask,
    )

    summary_rows.append({
        "split":
            split_name,

        "samples":
            len(X),

        "X_shape":
            str(X.shape),

        "y_shape":
            str(y.shape),

        "observed_target_points":
            int(mask.sum()),

        "total_target_points":
            int(mask.size),

        "imputed_input_points":
            int(
                input_imp_mask.sum()
            ),

        "total_input_points":
            int(
                total_input_points
            ),

        "imputed_input_pct":
            float(
                imputed_input_pct
            ),
    })


# ============================================================
# SAVE SCALER
# ============================================================

np.savez(
    DATA_DIR
    / "stgnn_scaler.npz",

    mean=scaler.mean_,

    scale=scaler.scale_,
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_file = (
    DATA_DIR
    / "stgnn_dataset_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print(
    "\n=== FINAL ST-GNN SUMMARY ==="
)

print(
    summary_df[
        [
            "split",
            "samples",
            "observed_target_points",
            "total_target_points",
            "imputed_input_points",
            "total_input_points",
            "imputed_input_pct",
        ]
    ].to_string(
        index=False,
        formatters={
            "imputed_input_pct":
                lambda x: f"{x:.2f}%"
        }
    )
)

print(
    "\n=== EXPECTED WINDOW CHECK ==="
)

expected = {
    "train": 415,
    "val": 92,
    "test": 93,
}

all_match = True

for row in summary_rows:

    split_name = row["split"]

    actual = row["samples"]

    exp = expected[
        split_name
    ]

    match = (
        actual == exp
    )

    all_match = (
        all_match
        and match
    )

    print(
        f"{split_name.upper():5s} | "
        f"actual={actual:3d} | "
        f"expected={exp:3d} | "
        f"{'OK' if match else 'CHECK'}"
    )


print(
    "\nWindow consistency : "
    + (
        "PASS"
        if all_match
        else "CHECK REQUIRED"
    )
)

print(
    "\n=== SAVED ==="
)

print(
    DATA_DIR
    / "stgnn_train.npz"
)

print(
    DATA_DIR
    / "stgnn_val.npz"
)

print(
    DATA_DIR
    / "stgnn_test.npz"
)

print(
    DATA_DIR
    / "stgnn_scaler.npz"
)

print(
    summary_file
)