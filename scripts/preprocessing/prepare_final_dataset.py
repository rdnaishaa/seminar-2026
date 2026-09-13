from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT
    / "data_update_dosen"
    / "full_data_update"
)

AUDIT_DIR = (
    ROOT
    / "outputs"
    / "data_audit"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "final_64"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

N_NODES = 64

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

FEATURES = [
    "current_speed",
    "congestion_ratio",
]


# ============================================================
# LOAD FROZEN 64 CORRIDORS
# ============================================================

candidate_file = (
    AUDIT_DIR
    / "candidate_64_corridors.csv"
)

if not candidate_file.exists():
    raise FileNotFoundError(
        "candidate_64_corridors.csv tidak ditemukan. "
        "Jalankan audit_candidate_64.py terlebih dahulu."
    )

candidate_df = pd.read_csv(
    candidate_file
)

corridors = (
    candidate_df["corridor_file"]
    .astype(str)
    .tolist()
)

if len(corridors) != N_NODES:
    raise ValueError(
        f"Expected {N_NODES} corridors, "
        f"found {len(corridors)}."
    )


print("\n=== FINAL DATASET FREEZE ===")
print(f"Nodes : {len(corridors)}")


# ============================================================
# LOAD RAW CORRIDOR FILES
# ============================================================

corridor_data = {}

for corridor in corridors:

    path = DATA_DIR / corridor

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    df = pd.read_csv(path)

    df["obs_time_utc"] = pd.to_datetime(
        df["obs_time_utc"],
        utc=True,
        errors="coerce"
    )

    df = (
        df
        .dropna(subset=["obs_time_utc"])
        .sort_values("obs_time_utc")
        .drop_duplicates(
            subset=["obs_time_utc"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    corridor_data[corridor] = df


# ============================================================
# COMMON TEMPORAL OVERLAP
# ============================================================

overlap_start = max(
    df["obs_time_utc"].min()
    for df in corridor_data.values()
)

overlap_end = min(
    df["obs_time_utc"].max()
    for df in corridor_data.values()
)

hourly_grid = pd.date_range(
    start=overlap_start,
    end=overlap_end,
    freq="h",
    tz="UTC"
)

print(
    f"Overlap : {overlap_start} -> {overlap_end}"
)

print(
    f"Hourly grid : {len(hourly_grid)} hours"
)


# ============================================================
# BUILD LONG-FORM HOURLY GRID
# ============================================================

rows = []

for corridor in corridors:

    raw = (
        corridor_data[corridor]
        .set_index("obs_time_utc")
        .reindex(hourly_grid)
    )

    for timestamp, row in raw.iterrows():

        is_observed = pd.notna(
            row.get("current_speed")
        )

        rows.append({
            "obs_time_utc":
                timestamp,

            "corridor_file":
                corridor,

            "name":
                row.get("name"),

            "city":
                row.get("city"),

            "current_speed":
                row.get("current_speed"),

            "free_flow_speed":
                row.get("free_flow_speed"),

            "current_travel_time":
                row.get(
                    "current_travel_time"
                ),

            "free_flow_travel_time":
                row.get(
                    "free_flow_travel_time"
                ),

            "congestion_ratio":
                row.get(
                    "congestion_ratio"
                ),

            "road_closure":
                row.get(
                    "road_closure"
                ),

            "confidence":
                row.get("confidence"),

            "is_observed":
                bool(is_observed),

            "was_imputed":
                False,
        })


grid_df = pd.DataFrame(rows)

grid_df = (
    grid_df
    .sort_values(
        [
            "obs_time_utc",
            "corridor_file",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# IDENTIFY GLOBAL ISOLATED 1-HOUR GAPS
# ============================================================

timestamp_status = (
    grid_df
    .groupby(
        "obs_time_utc"
    )["is_observed"]
    .all()
    .sort_index()
)

missing = ~timestamp_status

previous_observed = (
    ~missing.shift(
        1,
        fill_value=True
    )
)

next_observed = (
    ~missing.shift(
        -1,
        fill_value=True
    )
)

isolated_missing = (
    missing
    & previous_observed
    & next_observed
)

isolated_times = set(
    isolated_missing[
        isolated_missing
    ].index
)

print(
    f"Isolated 1h gaps : "
    f"{len(isolated_times)}"
)


# ============================================================
# FORWARD FILL ONLY ISOLATED 1-HOUR GAPS
# ============================================================

for corridor in corridors:

    corridor_mask = (
        grid_df[
            "corridor_file"
        ] == corridor
    )

    corridor_df = (
        grid_df
        .loc[corridor_mask]
        .copy()
        .sort_values(
            "obs_time_utc"
        )
    )

    for feature in FEATURES:

        previous_value = (
            corridor_df[
                feature
            ]
            .shift(1)
        )

        fill_mask = (
            corridor_df[
                "obs_time_utc"
            ].isin(
                isolated_times
            )
            & corridor_df[
                feature
            ].isna()
        )

        corridor_df.loc[
            fill_mask,
            feature
        ] = previous_value[
            fill_mask
        ]

        corridor_df.loc[
            fill_mask,
            "was_imputed"
        ] = True

    grid_df.loc[
        corridor_mask,
        FEATURES
        + ["was_imputed"]
    ] = corridor_df[
        FEATURES
        + ["was_imputed"]
    ].values


# ============================================================
# DETERMINE VALID TIMESTAMPS
# ============================================================

valid_by_time = (
    grid_df
    .groupby(
        "obs_time_utc"
    )[FEATURES]
    .apply(
        lambda x:
        x.notna().all().all()
    )
)

valid_times = (
    valid_by_time[
        valid_by_time
    ].index
)


# ============================================================
# BUILD TEMPORAL SEGMENTS
# ============================================================

segment_map = {}

segment_id = 0
previous_time = None

for timestamp in valid_times:

    if (
        previous_time is None
        or (
            timestamp
            - previous_time
        )
        != pd.Timedelta(hours=1)
    ):
        segment_id += 1

    segment_map[
        timestamp
    ] = segment_id

    previous_time = timestamp


grid_df[
    "segment_id"
] = (
    grid_df[
        "obs_time_utc"
    ].map(segment_map)
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

timestamps = hourly_grid

n_hours = len(timestamps)

train_end = int(
    n_hours
    * TRAIN_RATIO
)

val_end = int(
    n_hours
    * (
        TRAIN_RATIO
        + VAL_RATIO
    )
)


train_times = (
    timestamps[
        :train_end
    ]
)

val_times = (
    timestamps[
        train_end:val_end
    ]
)

test_times = (
    timestamps[
        val_end:
    ]
)


split_times = {
    "train": train_times,
    "val": val_times,
    "test": test_times,
}


# ============================================================
# SAVE SPLITS
# ============================================================

summary_rows = []

for split_name, times in (
    split_times.items()
):

    split_df = (
        grid_df[
            grid_df[
                "obs_time_utc"
            ].isin(times)
        ]
        .copy()
        .sort_values(
            [
                "corridor_file",
                "obs_time_utc",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    output_file = (
        OUTPUT_DIR
        / f"forecasting_{split_name}_preprocessed.csv"
    )

    split_df.to_csv(
        output_file,
        index=False
    )

    split_valid_times = (
        split_df[
            "segment_id"
        ]
        .notna()
    )

    valid_timestamp_count = (
        split_df.loc[
            split_valid_times,
            "obs_time_utc"
        ]
        .nunique()
    )

    observed_timestamp_count = (
        split_df[
            split_df[
                "is_observed"
            ]
        ][
            "obs_time_utc"
        ]
        .nunique()
    )

    imputed_timestamp_count = (
        split_df[
            split_df[
                "was_imputed"
            ]
        ][
            "obs_time_utc"
        ]
        .nunique()
    )

    num_segments = (
        split_df[
            "segment_id"
        ]
        .dropna()
        .nunique()
    )

    summary_rows.append({
        "split":
            split_name,

        "start":
            times.min(),

        "end":
            times.max(),

        "grid_hours":
            len(times),

        "valid_hours":
            valid_timestamp_count,

        "observed_timestamps":
            observed_timestamp_count,

        "imputed_timestamps":
            imputed_timestamp_count,

        "segments":
            int(num_segments),
    })

    print(
        f"\n=== {split_name.upper()} ==="
    )

    print(
        f"Start          : "
        f"{times.min()}"
    )

    print(
        f"End            : "
        f"{times.max()}"
    )

    print(
        f"Grid hours     : "
        f"{len(times)}"
    )

    print(
        f"Valid hours    : "
        f"{valid_timestamp_count}"
    )

    print(
        f"Segments       : "
        f"{num_segments}"
    )

    print(
        f"Saved          : "
        f"{output_file.name}"
    )


# ============================================================
# SAVE NODE LIST
# ============================================================

node_file = (
    OUTPUT_DIR
    / "final_64_corridors.csv"
)

pd.DataFrame({
    "node_index":
        range(
            len(corridors)
        ),

    "corridor_file":
        corridors,
}).to_csv(
    node_file,
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_file = (
    OUTPUT_DIR
    / "final_dataset_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# SAVE FREEZE METADATA
# ============================================================

metadata = {
    "num_nodes":
        N_NODES,

    "input_length":
        12,

    "horizons":
        [1, 3, 6],

    "target":
        "current_speed",

    "features_imputed":
        FEATURES,

    "imputation":
        "causal forward-fill only for globally isolated 1-hour gaps",

    "long_gap_handling":
        "not imputed; used as temporal segment boundaries",

    "split_strategy":
        "chronological 70/15/15 on common hourly grid",

    "overlap_start":
        str(overlap_start),

    "overlap_end":
        str(overlap_end),

    "total_grid_hours":
        int(len(hourly_grid)),

    "valid_hours_after_preprocessing":
        int(len(valid_times)),

    "num_temporal_segments":
        int(segment_id),
}

metadata_file = (
    OUTPUT_DIR
    / "final_dataset_metadata.json"
)

with open(
    metadata_file,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


# ============================================================
# FINAL REPORT
# ============================================================

print(
    "\n=== FINAL DATASET SUMMARY ==="
)

print(
    summary_df.to_string(
        index=False
    )
)

print(
    "\n=== GLOBAL ==="
)

print(
    f"Nodes             : "
    f"{N_NODES}"
)

print(
    f"Grid hours        : "
    f"{len(hourly_grid)}"
)

print(
    f"Valid hours       : "
    f"{len(valid_times)}"
)

print(
    f"Temporal segments : "
    f"{segment_id}"
)

print(
    f"Imputed timestamps: "
    f"{len(isolated_times)}"
)

print(
    "\n=== SAVED ==="
)

print(OUTPUT_DIR)
print(node_file)
print(summary_file)
print(metadata_file)