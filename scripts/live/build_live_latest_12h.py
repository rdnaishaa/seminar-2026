from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "data" / "collected" / "tomtom_flow_final64.csv"
OUTPUT_PATH = ROOT / "data" / "collected" / "latest_12h_final64.csv"

EXPECTED_CORRIDORS = 64
EXPECTED_HOURS = 12


def build_latest_12h():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File live belum ditemukan: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["obs_time_utc"]
    )

    required_columns = [
        "obs_time_utc",
        "station_id",
        "current_speed",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {missing}"
        )

    df["obs_time_utc"] = pd.to_datetime(
        df["obs_time_utc"],
        utc=True,
        format="mixed"
    )

    df = (
        df
        .sort_values([
            "obs_time_utc",
            "station_id"
        ])
        .drop_duplicates(
            subset=[
                "obs_time_utc",
                "station_id"
            ],
            keep="last"
        )
        .reset_index(drop=True)
    )

    timestamps = sorted(
        df["obs_time_utc"].dropna().unique()
    )

    if len(timestamps) < EXPECTED_HOURS:
        raise RuntimeError(
            f"Data live baru memiliki {len(timestamps)} jam. "
            f"Model membutuhkan {EXPECTED_HOURS} jam."
        )

    latest_12 = timestamps[-EXPECTED_HOURS:]

    latest_12 = pd.DatetimeIndex(
        latest_12
    )

    expected_range = pd.date_range(
        start=latest_12[0],
        periods=EXPECTED_HOURS,
        freq="h",
        tz="UTC"
    )

    if not latest_12.equals(expected_range):
        raise RuntimeError(
            "12 timestamp terbaru belum consecutive per jam."
        )

    latest_df = df[
        df["obs_time_utc"].isin(latest_12)
    ].copy()

    corridor_count = (
        latest_df["station_id"]
        .nunique()
    )

    if corridor_count != EXPECTED_CORRIDORS:
        raise RuntimeError(
            f"Corridor tidak lengkap. "
            f"Ditemukan {corridor_count}, "
            f"seharusnya {EXPECTED_CORRIDORS}."
        )

    counts_per_timestamp = (
        latest_df
        .groupby("obs_time_utc")["station_id"]
        .nunique()
    )

    incomplete = counts_per_timestamp[
        counts_per_timestamp != EXPECTED_CORRIDORS
    ]

    if not incomplete.empty:
        raise RuntimeError(
            "Ada timestamp yang tidak memiliki 64 corridor:\n"
            + incomplete.to_string()
        )

    if latest_df["current_speed"].isna().any():
        bad = latest_df[
            latest_df["current_speed"].isna()
        ]

        raise RuntimeError(
            "Ada current_speed kosong:\n"
            + bad[
                [
                    "obs_time_utc",
                    "station_id"
                ]
            ].to_string(index=False)
        )

    latest_df = (
        latest_df
        .sort_values([
            "obs_time_utc",
            "station_id"
        ])
        .reset_index(drop=True)
    )

    latest_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    latest_time = latest_df[
        "obs_time_utc"
    ].max()

    print("=== LIVE 12H INPUT BUILDER ===")
    print(f"Rows: {len(latest_df)}")
    print(
        f"Corridors: "
        f"{latest_df['station_id'].nunique()}"
    )
    print(
        f"Timestamps: "
        f"{latest_df['obs_time_utc'].nunique()}"
    )
    print(
        f"Start: "
        f"{latest_df['obs_time_utc'].min()}"
    )
    print(
        f"End: "
        f"{latest_time}"
    )
    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    build_latest_12h()