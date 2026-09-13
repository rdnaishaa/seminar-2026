import pandas as pd

from src.config import (
    CORRIDORS_PATH,
    FINAL_CORRIDORS_PATH,
    TRAIN_HISTORY_PATH,
    station_id_to_display_name,
)


# ============================================================
# FINAL 64 CORRIDORS
# ============================================================

def load_final64():

    corridors = pd.read_csv(
        CORRIDORS_PATH
    )

    corridors["station_id"] = (
        corridors["station_id"]
        .astype(str)
    )

    final64 = pd.read_csv(
        FINAL_CORRIDORS_PATH
    )

    if "corridor_file" in final64.columns:

        station_ids = (
            final64["corridor_file"]
            .dropna()
            .astype(str)
            .str.replace(".csv", "", regex=False)
            .tolist()
        )

    elif "station_id" in final64.columns:

        station_ids = (
            final64["station_id"]
            .dropna()
            .astype(str)
            .str.replace(".csv", "", regex=False)
            .tolist()
        )

    elif "file" in final64.columns:

        station_ids = (
            final64["file"]
            .dropna()
            .astype(str)
            .str.replace(".csv", "", regex=False)
            .tolist()
        )

    else:
        raise ValueError(
            "final_64_corridors.csv tidak memiliki "
            "kolom corridor_file, station_id, atau file."
        )

    selected = corridors[
        corridors["station_id"].isin(station_ids)
    ].copy()

    selected = (
        selected
        .set_index("station_id")
        .reindex(station_ids)
        .dropna(how="all")
        .reset_index()
    )

    if "name" in selected.columns:

        selected["display_name"] = selected.apply(
            lambda row:
                f"{station_id_to_display_name(row['station_id'])}"
                f" — {row['name']}",
            axis=1,
        )

    else:

        selected["display_name"] = (
            selected["station_id"]
            .apply(station_id_to_display_name)
        )

    return selected


# ============================================================
# PREPARE TRAINING HISTORY
# ============================================================

def prepare_history(df):

    if df.empty:
        return df

    df = df.copy()

    # corridor_file contoh:
    # tt_bekasi.csv
    # menjadi station_id:
    # tt_bekasi
    if "corridor_file" in df.columns:

        df["station_id"] = (
            df["corridor_file"]
            .astype(str)
            .str.replace(
                ".csv",
                "",
                regex=False,
            )
        )

    # Timestamp sudah UTC pada dataset training
    if "obs_time_utc" in df.columns:

        df["obs_time_utc"] = pd.to_datetime(
            df["obs_time_utc"],
            utc=True,
            errors="coerce",
        )

    numeric_columns = [
        "current_speed",
        "free_flow_speed",
        "current_travel_time",
        "free_flow_travel_time",
        "congestion_ratio",
        "confidence",
        "segment_id",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


# ============================================================
# LOAD HISTORY
# ============================================================

def load_history():
    """
    Memuat data historis yang benar-benar digunakan
    sebagai TRAINING SET penelitian.

    Sumber:
    data/final_64/forecasting_train_preprocessed.csv

    Tidak menggunakan:
    - validation set
    - test set
    - live collector
    """

    if not TRAIN_HISTORY_PATH.exists():

        raise FileNotFoundError(
            f"Training dataset tidak ditemukan: "
            f"{TRAIN_HISTORY_PATH}"
        )

    history = pd.read_csv(
        TRAIN_HISTORY_PATH
    )

    history = prepare_history(
        history
    )

    if history.empty:
        return pd.DataFrame()

    # Hapus timestamp invalid
    history = history.dropna(
        subset=[
            "obs_time_utc",
            "station_id",
        ]
    )

    # Hanya Final64
    final_station_ids = (
        load_final64()["station_id"]
        .astype(str)
        .tolist()
    )

    history = history[
        history["station_id"].isin(
            final_station_ids
        )
    ].copy()

    # Hindari duplicate timestamp per corridor
    history = history.drop_duplicates(
        subset=[
            "station_id",
            "obs_time_utc",
        ],
        keep="last",
    )

    history = history.sort_values(
        [
            "obs_time_utc",
            "station_id",
        ]
    )

    return history.reset_index(
        drop=True
    )