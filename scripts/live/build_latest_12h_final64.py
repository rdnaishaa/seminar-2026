from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FINAL64_PATH = ROOT / "data" / "final_64" / "final_64_corridors.csv"
UPDATED_DATA_DIR = ROOT / "data_update_dosen" / "full_data_update"

OUTPUT_DIR = ROOT / "data" / "collected"
OUTPUT_PATH = OUTPUT_DIR / "latest_12h_final64.csv"


def get_station_ids():
    final64 = pd.read_csv(FINAL64_PATH)

    if "station_id" in final64.columns:
        station_ids = final64["station_id"].astype(str).tolist()

    elif "corridor_file" in final64.columns:
        station_ids = (
            final64["corridor_file"]
            .astype(str)
            .str.replace(".csv", "", regex=False)
            .tolist()
        )

    else:
        raise ValueError(
            "final_64_corridors.csv tidak memiliki kolom station_id/corridor_file"
        )

    return station_ids


def load_corridor(station_id):
    file_path = UPDATED_DATA_DIR / f"{station_id}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {file_path}"
        )

    df = pd.read_csv(file_path)

    df["obs_time_utc"] = pd.to_datetime(
        df["obs_time_utc"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(subset=["obs_time_utc"])
    df = df.sort_values("obs_time_utc")

    return df


def align_hourly(df, station_id):
    """
    Align data menjadi grid hourly.

    Gap 1 jam diisi secara causal menggunakan nilai sebelumnya.
    Gap lebih panjang tidak diisi.
    """

    df = df.set_index("obs_time_utc")

    hourly_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="1h",
        tz="UTC",
    )

    aligned = df.reindex(hourly_index)

    observed = aligned["current_speed"].notna()

    # hanya isi gap maksimum 1 timestep
    aligned = aligned.ffill(limit=1)

    aligned["is_imputed"] = (
        ~observed
        & aligned["current_speed"].notna()
    )

    aligned["station_id"] = station_id

    aligned = aligned.reset_index()
    aligned = aligned.rename(
        columns={"index": "obs_time_utc"}
    )

    return aligned


def build_latest_12h():
    station_ids = get_station_ids()

    print("Expected corridors:", len(station_ids))

    frames = []

    for station_id in station_ids:
        df = load_corridor(station_id)
        aligned = align_hourly(df, station_id)
        frames.append(aligned)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    # Timestamp dianggap valid hanya jika seluruh 64 corridor tersedia
    valid_counts = (
        combined
        .dropna(subset=["current_speed"])
        .groupby("obs_time_utc")["station_id"]
        .nunique()
    )

    valid_timestamps = valid_counts[
        valid_counts == len(station_ids)
    ].index

    valid_timestamps = (
        pd.Series(valid_timestamps)
        .sort_values()
        .reset_index(drop=True)
    )

    if len(valid_timestamps) < 12:
        raise RuntimeError(
            f"Hanya ada {len(valid_timestamps)} timestamp valid."
        )

    # cari 12 jam BERURUTAN terakhir
    latest_window = None

    for end_idx in range(
        len(valid_timestamps) - 1,
        10,
        -1,
    ):
        candidate = valid_timestamps.iloc[
            end_idx - 11:end_idx + 1
        ]

        diffs = candidate.diff().dropna()

        if (diffs == pd.Timedelta(hours=1)).all():
            latest_window = candidate
            break

    if latest_window is None:
        raise RuntimeError(
            "Tidak ditemukan window 12 jam berturut-turut."
        )

    result = combined[
        combined["obs_time_utc"].isin(
            latest_window.tolist()
        )
    ].copy()

    result = result.dropna(
        subset=["current_speed"]
    )

    result = result.sort_values(
        ["obs_time_utc", "station_id"]
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n=== LATEST 12H FINAL64 ===")
    print("Output:", OUTPUT_PATH)
    print("Rows:", len(result))
    print(
        "Corridors:",
        result["station_id"].nunique(),
    )
    print(
        "Timestamps:",
        result["obs_time_utc"].nunique(),
    )
    print(
        "Start:",
        result["obs_time_utc"].min(),
    )
    print(
        "End:",
        result["obs_time_utc"].max(),
    )
    print(
        "Imputed rows:",
        int(result["is_imputed"].sum()),
    )


if __name__ == "__main__":
    build_latest_12h()