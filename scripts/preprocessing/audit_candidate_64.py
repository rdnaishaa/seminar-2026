from pathlib import Path
import numpy as np
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

N_CANDIDATE = 64


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
        .dropna(subset=["obs_time_utc"])
        .sort_values("obs_time_utc")
        .drop_duplicates(
            subset=["obs_time_utc"],
            keep="last"
        )
        .reset_index(drop=True)
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

selected_files = (
    ranking
    .head(N_CANDIDATE)
    ["file"]
    .tolist()
)


print(
    f"\n=== AUDIT CANDIDATE N={N_CANDIDATE} ==="
)

print(
    f"Selected corridors : "
    f"{len(selected_files)}"
)


# ============================================================
# PER-CORRIDOR QUALITY AUDIT
# ============================================================

audit_rows = []

for file_name in selected_files:

    df = corridor_data[file_name].copy()

    # --------------------------------------------------------
    # BASIC
    # --------------------------------------------------------

    rows = len(df)

    start = (
        df["obs_time_utc"].min()
    )

    end = (
        df["obs_time_utc"].max()
    )

    # --------------------------------------------------------
    # CURRENT SPEED
    # --------------------------------------------------------

    speed = pd.to_numeric(
        df["current_speed"],
        errors="coerce"
    )

    speed_null = int(
        speed.isna().sum()
    )

    speed_min = (
        speed.min()
    )

    speed_max = (
        speed.max()
    )

    speed_mean = (
        speed.mean()
    )

    speed_std = (
        speed.std()
    )

    speed_unique = (
        speed.nunique(
            dropna=True
        )
    )

    speed_zero = int(
        (speed == 0).sum()
    )

    speed_negative = int(
        (speed < 0).sum()
    )

    # --------------------------------------------------------
    # FREE FLOW SPEED
    # --------------------------------------------------------

    free_flow = pd.to_numeric(
        df["free_flow_speed"],
        errors="coerce"
    )

    ff_null = int(
        free_flow.isna().sum()
    )

    ff_min = (
        free_flow.min()
    )

    ff_max = (
        free_flow.max()
    )

    ff_mean = (
        free_flow.mean()
    )

    ff_negative = int(
        (free_flow < 0).sum()
    )

    # --------------------------------------------------------
    # SPEED > FREE FLOW
    # Not automatically invalid, but worth checking.
    # --------------------------------------------------------

    valid_pair = (
        speed.notna()
        & free_flow.notna()
    )

    speed_above_ff = int(
        (
            speed[valid_pair]
            >
            free_flow[valid_pair]
        ).sum()
    )

    speed_above_ff_pct = (
        speed_above_ff
        / valid_pair.sum()
        * 100
        if valid_pair.sum() > 0
        else 0
    )

    # --------------------------------------------------------
    # CONGESTION RATIO
    # --------------------------------------------------------

    congestion = pd.to_numeric(
        df["congestion_ratio"],
        errors="coerce"
    )

    congestion_null = int(
        congestion.isna().sum()
    )

    congestion_min = (
        congestion.min()
    )

    congestion_max = (
        congestion.max()
    )

    congestion_mean = (
        congestion.mean()
    )

    congestion_negative = int(
        (congestion < 0).sum()
    )

    congestion_above_1 = int(
        (congestion > 1).sum()
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = pd.to_numeric(
        df["confidence"],
        errors="coerce"
    )

    confidence_null = int(
        confidence.isna().sum()
    )

    confidence_min = (
        confidence.min()
    )

    confidence_max = (
        confidence.max()
    )

    confidence_mean = (
        confidence.mean()
    )

    confidence_below_0 = int(
        (confidence < 0).sum()
    )

    confidence_above_1 = int(
        (confidence > 1).sum()
    )

    # --------------------------------------------------------
    # ROAD CLOSURE
    # --------------------------------------------------------

    road_closure_count = 0

    if "road_closure" in df.columns:

        road_closure = (
            df["road_closure"]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "yes"
                ]
            )
        )

        road_closure_count = int(
            road_closure.sum()
        )

    # --------------------------------------------------------
    # FLAGS
    # --------------------------------------------------------

    near_constant_speed = (
        speed_unique <= 5
        or (
            pd.notna(speed_std)
            and speed_std < 0.5
        )
    )

    extreme_speed_flag = (
        (
            pd.notna(speed_max)
            and speed_max > 150
        )
        or speed_negative > 0
    )

    confidence_flag = (
        confidence_below_0 > 0
        or confidence_above_1 > 0
    )

    congestion_flag = (
        congestion_negative > 0
        or congestion_above_1 > 0
    )

    audit_rows.append({
        "file": file_name,
        "rows": rows,
        "start": start,
        "end": end,

        "speed_null": speed_null,
        "speed_min": speed_min,
        "speed_max": speed_max,
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "speed_unique": speed_unique,
        "speed_zero": speed_zero,
        "speed_negative":
            speed_negative,

        "ff_null": ff_null,
        "ff_min": ff_min,
        "ff_max": ff_max,
        "ff_mean": ff_mean,
        "ff_negative":
            ff_negative,

        "speed_above_ff":
            speed_above_ff,

        "speed_above_ff_pct":
            speed_above_ff_pct,

        "congestion_null":
            congestion_null,

        "congestion_min":
            congestion_min,

        "congestion_max":
            congestion_max,

        "congestion_mean":
            congestion_mean,

        "congestion_negative":
            congestion_negative,

        "congestion_above_1":
            congestion_above_1,

        "confidence_null":
            confidence_null,

        "confidence_min":
            confidence_min,

        "confidence_max":
            confidence_max,

        "confidence_mean":
            confidence_mean,

        "confidence_below_0":
            confidence_below_0,

        "confidence_above_1":
            confidence_above_1,

        "road_closure_count":
            road_closure_count,

        "flag_near_constant_speed":
            near_constant_speed,

        "flag_extreme_speed":
            extreme_speed_flag,

        "flag_confidence":
            confidence_flag,

        "flag_congestion":
            congestion_flag,
    })


audit_df = pd.DataFrame(
    audit_rows
)


# ============================================================
# OVERALL FLAGS
# ============================================================

flag_columns = [
    "flag_near_constant_speed",
    "flag_extreme_speed",
    "flag_confidence",
    "flag_congestion",
]

audit_df[
    "num_flags"
] = (
    audit_df[
        flag_columns
    ]
    .sum(axis=1)
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n=== QUALITY SUMMARY ==="
)

print(
    f"Corridors with any flag : "
    f"{(audit_df['num_flags'] > 0).sum()}"
)

print(
    f"Near-constant speed      : "
    f"{audit_df['flag_near_constant_speed'].sum()}"
)

print(
    f"Extreme speed            : "
    f"{audit_df['flag_extreme_speed'].sum()}"
)

print(
    f"Confidence anomaly       : "
    f"{audit_df['flag_confidence'].sum()}"
)

print(
    f"Congestion anomaly       : "
    f"{audit_df['flag_congestion'].sum()}"
)


# ============================================================
# FLAGGED CORRIDORS
# ============================================================

print(
    "\n=== FLAGGED CORRIDORS ==="
)

flagged = (
    audit_df[
        audit_df["num_flags"] > 0
    ]
    .sort_values(
        [
            "num_flags",
            "file",
        ],
        ascending=[
            False,
            True,
        ]
    )
)

if flagged.empty:

    print(
        "Tidak ada corridor dengan flag."
    )

else:

    print(
        flagged[
            [
                "file",
                "num_flags",
                "speed_min",
                "speed_max",
                "speed_mean",
                "speed_std",
                "speed_unique",
                "confidence_min",
                "confidence_max",
                "congestion_min",
                "congestion_max",
                "flag_near_constant_speed",
                "flag_extreme_speed",
                "flag_confidence",
                "flag_congestion",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# SPEED ABOVE FREE FLOW
# ============================================================

print(
    "\n=== TOP 15 SPEED > FREE FLOW ==="
)

print(
    audit_df[
        [
            "file",
            "speed_above_ff",
            "speed_above_ff_pct",
            "speed_mean",
            "ff_mean",
        ]
    ]
    .sort_values(
        "speed_above_ff_pct",
        ascending=False
    )
    .head(15)
    .to_string(
        index=False,
        formatters={
            "speed_above_ff_pct":
                lambda x: f"{x:.1f}%"
        }
    )
)


# ============================================================
# LOWEST CONFIDENCE
# ============================================================

print(
    "\n=== 15 LOWEST MEAN CONFIDENCE ==="
)

print(
    audit_df[
        [
            "file",
            "confidence_min",
            "confidence_mean",
            "confidence_max",
        ]
    ]
    .sort_values(
        "confidence_mean"
    )
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# LOWEST SPEED VARIABILITY
# ============================================================

print(
    "\n=== 15 LOWEST SPEED STD ==="
)

print(
    audit_df[
        [
            "file",
            "speed_mean",
            "speed_std",
            "speed_unique",
        ]
    ]
    .sort_values(
        "speed_std"
    )
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    OUTPUT_DIR
    / "candidate_64_quality_audit.csv"
)

selected_file = (
    OUTPUT_DIR
    / "candidate_64_corridors.csv"
)

audit_df.to_csv(
    output_file,
    index=False
)

pd.DataFrame({
    "corridor_file":
        selected_files
}).to_csv(
    selected_file,
    index=False
)


print(
    "\n=== SAVED ==="
)

print(output_file)
print(selected_file)