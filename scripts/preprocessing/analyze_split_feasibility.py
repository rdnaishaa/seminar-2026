from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT
    / "data_update_dosen"
    / "full_data_update"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "data_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INPUT_LENGTH = 12
HORIZONS = [1, 3, 6]

CANDIDATE_N = [14, 24, 64]

# Chronological split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# LOAD + RANK CORRIDORS
# ============================================================

records = []
corridor_data = {}

for file in sorted(DATA_DIR.glob("*.csv")):

    df = pd.read_csv(file)

    df["obs_time_utc"] = pd.to_datetime(
        df["obs_time_utc"],
        utc=True,
        errors="coerce"
    )

    df = (
        df
        .dropna(
            subset=[
                "obs_time_utc",
                "current_speed",
            ]
        )
        .sort_values("obs_time_utc")
        .drop_duplicates(
            subset=["obs_time_utc"],
            keep="last"
        )
    )

    corridor_data[file.name] = df

    records.append({
        "file": file.name,
        "unique_ts":
            df["obs_time_utc"].nunique(),
        "start":
            df["obs_time_utc"].min(),
        "end":
            df["obs_time_utc"].max(),
    })


ranking = (
    pd.DataFrame(records)
    .sort_values(
        ["unique_ts", "file"],
        ascending=[False, True]
    )
    .reset_index(drop=True)
)


# ============================================================
# PREPARE ALIGNED NETWORK
# ============================================================

def prepare_network(selected_files):

    selected = {
        name: corridor_data[name]
        for name in selected_files
    }

    overlap_start = max(
        df["obs_time_utc"].min()
        for df in selected.values()
    )

    overlap_end = min(
        df["obs_time_utc"].max()
        for df in selected.values()
    )

    hourly_grid = pd.date_range(
        overlap_start,
        overlap_end,
        freq="h",
        tz="UTC"
    )

    speed = pd.DataFrame(
        index=hourly_grid
    )

    for name, df in selected.items():

        speed[name] = (
            df
            .set_index("obs_time_utc")
            ["current_speed"]
            .reindex(hourly_grid)
        )

    # ========================================================
    # ISOLATED 1-HOUR GAP
    # ========================================================

    complete = (
        speed.notna()
        .all(axis=1)
    )

    missing = ~complete

    previous_complete = (
        ~missing.shift(
            1,
            fill_value=True
        )
    )

    next_complete = (
        ~missing.shift(
            -1,
            fill_value=True
        )
    )

    isolated_missing = (
        missing
        & previous_complete
        & next_complete
    )

    speed_filled = speed.copy()

    for timestamp in (
        isolated_missing[
            isolated_missing
        ].index
    ):

        loc = (
            speed_filled
            .index
            .get_loc(timestamp)
        )

        if loc == 0:
            continue

        current = (
            speed_filled.iloc[loc]
        )

        previous = (
            speed_filled.iloc[loc - 1]
        )

        fill_mask = (
            current.isna()
        )

        speed_filled.loc[
            timestamp,
            fill_mask
        ] = previous[
            fill_mask
        ]

    # ========================================================
    # VALID TIMESTAMPS + SEGMENTS
    # ========================================================

    valid = (
        speed_filled.notna()
        .all(axis=1)
    )

    segment = pd.Series(
        pd.NA,
        index=hourly_grid,
        dtype="Float64"
    )

    segment_id = 0
    previous_time = None

    for timestamp in hourly_grid[valid]:

        if (
            previous_time is None
            or timestamp - previous_time
            != pd.Timedelta(hours=1)
        ):
            segment_id += 1

        segment.loc[timestamp] = (
            segment_id
        )

        previous_time = timestamp

    return (
        speed_filled,
        segment
    )


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    speed,
    segment
):

    n = len(speed)

    train_end = int(
        n * TRAIN_RATIO
    )

    val_end = int(
        n * (
            TRAIN_RATIO
            + VAL_RATIO
        )
    )

    splits = {
        "train": (
            speed.iloc[:train_end],
            segment.iloc[:train_end],
        ),

        "val": (
            speed.iloc[
                train_end:val_end
            ],
            segment.iloc[
                train_end:val_end
            ],
        ),

        "test": (
            speed.iloc[val_end:],
            segment.iloc[val_end:],
        ),
    }

    return splits


# ============================================================
# COUNT WINDOWS
# ============================================================

def count_windows(
    speed,
    segment
):

    max_horizon = max(
        HORIZONS
    )

    segments = (
        segment.to_numpy()
    )

    total = 0

    for start in range(
        len(speed)
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

        used_segments = [
            segments[i]
            for i in used_indices
        ]

        if any(
            pd.isna(x)
            for x in used_segments
        ):
            continue

        if len(
            set(used_segments)
        ) != 1:
            continue

        total += 1

    return total


# ============================================================
# ANALYZE CANDIDATES
# ============================================================

results = []

for n_nodes in CANDIDATE_N:

    selected_files = (
        ranking
        .head(n_nodes)
        ["file"]
        .tolist()
    )

    speed, segment = (
        prepare_network(
            selected_files
        )
    )

    splits = (
        chronological_split(
            speed,
            segment
        )
    )

    print(
        f"\n=== N={n_nodes} ==="
    )

    for split_name, (
        split_speed,
        split_segment
    ) in splits.items():

        windows = count_windows(
            split_speed,
            split_segment
        )

        valid_hours = int(
            split_segment
            .notna()
            .sum()
        )

        num_segments = int(
            split_segment
            .dropna()
            .nunique()
        )

        start_time = (
            split_speed.index.min()
        )

        end_time = (
            split_speed.index.max()
        )

        print(
            f"{split_name.upper():5s} | "
            f"{start_time} -> "
            f"{end_time} | "
            f"grid={len(split_speed):4d} | "
            f"valid={valid_hours:4d} | "
            f"segments={num_segments:2d} | "
            f"windows={windows:4d}"
        )

        results.append({
            "N": n_nodes,
            "split": split_name,
            "start": start_time,
            "end": end_time,
            "grid_hours":
                len(split_speed),
            "valid_hours":
                valid_hours,
            "num_segments":
                num_segments,
            "usable_windows":
                windows,
        })


# ============================================================
# SUMMARY
# ============================================================

result_df = pd.DataFrame(
    results
)

summary = (
    result_df
    .pivot(
        index="N",
        columns="split",
        values="usable_windows"
    )
    .reset_index()
)

summary = summary[
    [
        "N",
        "train",
        "val",
        "test",
    ]
]

summary["total_after_split"] = (
    summary[
        [
            "train",
            "val",
            "test",
        ]
    ].sum(axis=1)
)


print(
    "\n=== SPLIT WINDOW SUMMARY ==="
)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

detail_file = (
    OUTPUT_DIR
    / "network_split_feasibility.csv"
)

summary_file = (
    OUTPUT_DIR
    / "network_split_summary.csv"
)

result_df.to_csv(
    detail_file,
    index=False
)

summary.to_csv(
    summary_file,
    index=False
)


print(
    "\n=== SAVED ==="
)

print(detail_file)
print(summary_file)