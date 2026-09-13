from pathlib import Path

import pandas as pd

from src.config import (
    CORRIDORS_PATH,
    FINAL_CORRIDORS_PATH,
    UPDATED_DATA_DIR,
    COLLECTED_PATH,
    station_id_to_display_name,
)


# ============================================================
# HELPERS
# ============================================================

def normalize_corridor_file(value):
    """
    Normalize corridor identifier into filename form:
        tt_sudirman -> tt_sudirman.csv
        tt_sudirman.csv -> tt_sudirman.csv
    """

    value = str(value)

    if not value.endswith(".csv"):
        value = f"{value}.csv"

    return value


def corridor_file_to_station_id(value):
    """
    Convert:
        tt_sudirman.csv -> tt_sudirman
    """

    value = str(value)

    if value.endswith(".csv"):
        value = value[:-4]

    return value


# ============================================================
# LOAD FINAL 64 CORRIDORS
# ============================================================

def load_final64():

    # --------------------------------------------------------
    # Load master metadata
    # --------------------------------------------------------

    corridors = pd.read_csv(
        CORRIDORS_PATH
    )

    corridors["station_id"] = (
        corridors["station_id"]
        .astype(str)
    )

    # --------------------------------------------------------
    # Load frozen final 64 corridor list
    # --------------------------------------------------------

    final64 = pd.read_csv(
        FINAL_CORRIDORS_PATH
    )

    # Support several possible column names
    if "corridor_file" in final64.columns:

        corridor_files = (
            final64["corridor_file"]
            .dropna()
            .astype(str)
            .apply(normalize_corridor_file)
            .tolist()
        )

    elif "station_id" in final64.columns:

        corridor_files = (
            final64["station_id"]
            .dropna()
            .astype(str)
            .apply(normalize_corridor_file)
            .tolist()
        )

    elif "file" in final64.columns:

        corridor_files = (
            final64["file"]
            .dropna()
            .astype(str)
            .apply(normalize_corridor_file)
            .tolist()
        )

    else:

        raise ValueError(
            "Tidak menemukan kolom corridor_file, station_id, "
            "atau file pada final_64_corridors.csv"
        )

    station_ids = [
        corridor_file_to_station_id(x)
        for x in corridor_files
    ]

    # --------------------------------------------------------
    # Match with corridor metadata
    # --------------------------------------------------------

    selected = corridors[
        corridors["station_id"].isin(
            station_ids
        )
    ].copy()

    # Preserve exact final-64 node order
    selected = (
        selected
        .set_index("station_id")
        .reindex(station_ids)
        .dropna(
            how="all"
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Display names
    # --------------------------------------------------------

    selected["display_name"] = (
        selected.apply(
            lambda row: (
                f"{station_id_to_display_name(row['station_id'])}"
                f" — {row['name']}"
            ),
            axis=1,
        )
    )

    return selected


# ============================================================
# PREPARE TRAFFIC DATA
# ============================================================

def prepare_history(
    df,
    station_id=None,
):

    if df.empty:
        return df

    df = df.copy()

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    if "obs_time_utc" in df.columns:

        df["obs_time_utc"] = pd.to_datetime(
            df["obs_time_utc"],
            utc=True,
            errors="coerce",
        )

    # --------------------------------------------------------
    # station_id
    # --------------------------------------------------------

    if "station_id" not in df.columns:

        if station_id is not None:
            df["station_id"] = str(
                station_id
            )

    else:

        df["station_id"] = (
            df["station_id"]
            .astype(str)
        )

    # --------------------------------------------------------
    # Numeric traffic columns
    # --------------------------------------------------------

    numeric_columns = [
        "current_speed",
        "free_flow_speed",
        "current_travel_time",
        "free_flow_travel_time",
        "congestion_ratio",
        "confidence",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Generate congestion_ratio if absent
    # --------------------------------------------------------

    if (
        "congestion_ratio"
        not in df.columns
        and "current_speed" in df.columns
        and "free_flow_speed" in df.columns
    ):

        denominator = (
            df["free_flow_speed"]
            .replace(
                0,
                pd.NA,
            )
        )

        df["congestion_ratio"] = (
            1
            - (
                df["current_speed"]
                / denominator
            )
        )

    # --------------------------------------------------------
    # Keep ratio between 0 and 1
    # --------------------------------------------------------

    if "congestion_ratio" in df.columns:

        df["congestion_ratio"] = (
            df["congestion_ratio"]
            .clip(
                lower=0,
                upper=1,
            )
        )

    return df


# ============================================================
# LOAD UPDATED RAW HISTORY — FINAL 64
# ============================================================

def load_updated_history():

    corridors = load_final64()

    datasets = []

    for _, corridor in corridors.iterrows():

        station_id = str(
            corridor["station_id"]
        )

        filename = (
            f"{station_id}.csv"
        )

        file_path = (
            UPDATED_DATA_DIR
            / filename
        )

        if not file_path.exists():

            print(
                f"[WARNING] Missing updated file: "
                f"{file_path}"
            )

            continue

        df = pd.read_csv(
            file_path
        )

        df = prepare_history(
            df,
            station_id=station_id,
        )

        datasets.append(
            df
        )

    if not datasets:

        return pd.DataFrame()

    return pd.concat(
        datasets,
        ignore_index=True,
        sort=False,
    )


# ============================================================
# LOAD HISTORICAL + COLLECTOR
# ============================================================

def load_history():

    datasets = []

    # --------------------------------------------------------
    # Updated TomTom historical dataset
    # --------------------------------------------------------

    historical = (
        load_updated_history()
    )

    if not historical.empty:

        datasets.append(
            historical
        )

    # --------------------------------------------------------
    # Automatic live collector
    # --------------------------------------------------------

    if COLLECTED_PATH.exists():

        collected = pd.read_csv(
            COLLECTED_PATH
        )

        collected = prepare_history(
            collected
        )

        datasets.append(
            collected
        )

    # --------------------------------------------------------
    # No data
    # --------------------------------------------------------

    if not datasets:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    df = pd.concat(
        datasets,
        ignore_index=True,
        sort=False,
    )

    # --------------------------------------------------------
    # Drop invalid timestamps
    # --------------------------------------------------------

    if "obs_time_utc" in df.columns:

        df = df.dropna(
            subset=[
                "obs_time_utc"
            ]
        )

    # --------------------------------------------------------
    # Keep final 64 only
    # --------------------------------------------------------

    final64 = (
        load_final64()[
            "station_id"
        ]
        .astype(str)
        .tolist()
    )

    if "station_id" in df.columns:

        df = df[
            df["station_id"].isin(
                final64
            )
        ].copy()

    # --------------------------------------------------------
    # Remove duplicate observations
    #
    # Collector comes after historical dataset,
    # therefore latest collector row is retained.
    # --------------------------------------------------------

    if (
        "station_id" in df.columns
        and "obs_time_utc" in df.columns
    ):

        df = df.drop_duplicates(
            subset=[
                "station_id",
                "obs_time_utc",
            ],
            keep="last",
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    if (
        "station_id" in df.columns
        and "obs_time_utc" in df.columns
    ):

        df = df.sort_values(
            [
                "obs_time_utc",
                "station_id",
            ]
        )

    return df.reset_index(
        drop=True
    )