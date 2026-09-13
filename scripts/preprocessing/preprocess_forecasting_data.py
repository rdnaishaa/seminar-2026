from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

FILES = {
    "train": DATA_DIR / "forecasting_train_hourly_grid.csv",
    "val": DATA_DIR / "forecasting_val_hourly_grid.csv",
    "test": DATA_DIR / "forecasting_test_hourly_grid.csv",
}

FEATURES = [
    "current_speed",
    "congestion_ratio",
]

for split_name, path in FILES.items():

    df = pd.read_csv(path)

    df["obs_time_utc"] = pd.to_datetime(
        df["obs_time_utc"],
        utc=True
    )

    df = df.sort_values(
        ["corridor_file", "obs_time_utc"]
    ).reset_index(drop=True)

    # Simpan status asli sebelum imputasi
    df["was_imputed"] = False

    # -----------------------------------------------------
    # Tentukan isolated missing timestamp secara GLOBAL
    # karena missing pada top-14 bersifat sinkron
    # -----------------------------------------------------
    timestamp_status = (
        df.groupby("obs_time_utc")["is_observed"]
        .any()
        .sort_index()
    )

    missing = ~timestamp_status

    previous_observed = ~missing.shift(1, fill_value=True)
    next_observed = ~missing.shift(-1, fill_value=True)

    isolated_missing = (
        missing
        & previous_observed
        & next_observed
    )

    isolated_times = set(
        isolated_missing[isolated_missing].index
    )

    # -----------------------------------------------------
    # Forward fill HANYA isolated 1-hour gaps
    # -----------------------------------------------------
    for corridor in df["corridor_file"].unique():

        mask_corridor = df["corridor_file"] == corridor

        corridor_df = df.loc[mask_corridor].copy()

        for feature in FEATURES:

            previous_value = corridor_df[feature].shift(1)

            fill_mask = (
                corridor_df["obs_time_utc"].isin(isolated_times)
                & corridor_df[feature].isna()
            )

            corridor_df.loc[
                fill_mask, feature
            ] = previous_value[fill_mask]

            corridor_df.loc[
                fill_mask, "was_imputed"
            ] = True

        df.loc[
            mask_corridor,
            FEATURES + ["was_imputed"]
        ] = corridor_df[
            FEATURES + ["was_imputed"]
        ].values

    # -----------------------------------------------------
    # Tentukan timestamp valid setelah preprocessing
    # -----------------------------------------------------
    valid_by_time = (
        df.groupby("obs_time_utc")[FEATURES]
        .apply(lambda x: x.notna().all().all())
    )

    # Segment baru setiap kali ada break dalam valid sequence
    valid_times = valid_by_time[valid_by_time].index

    segment_map = {}
    segment_id = 0
    previous_time = None

    for timestamp in valid_times:

        if (
            previous_time is None
            or (timestamp - previous_time).total_seconds() != 3600
        ):
            segment_id += 1

        segment_map[timestamp] = segment_id
        previous_time = timestamp

    df["segment_id"] = df["obs_time_utc"].map(segment_map)

    # Long-gap rows tetap ada, segment_id = NaN
    output = DATA_DIR / f"forecasting_{split_name}_preprocessed.csv"

    df.to_csv(output, index=False)

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------
    total_imputed_rows = int(df["was_imputed"].sum())

    imputed_timestamps = df.loc[
        df["was_imputed"],
        "obs_time_utc"
    ].nunique()

    valid_timestamp_count = len(valid_times)

    number_segments = len(
        set(segment_map.values())
    )

    print(f"\n=== {split_name.upper()} ===")
    print(f"Isolated timestamps filled : {imputed_timestamps}")
    print(f"Rows imputed               : {total_imputed_rows}")
    print(f"Valid hourly timestamps     : {valid_timestamp_count}")
    print(f"Temporal segments           : {number_segments}")

    if number_segments > 0:
        segment_sizes = (
            pd.Series(segment_map)
            .value_counts()
            .sort_index()
        )

        print("Segment lengths:")
        for seg, length in segment_sizes.items():
            print(f"  Segment {seg}: {length} hours")

    print(f"Saved: {output.name}")