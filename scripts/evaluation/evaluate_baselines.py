from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

TRAIN_PATH = DATA_DIR / "forecasting_train_preprocessed.csv"
VAL_PATH = DATA_DIR / "forecasting_val_preprocessed.csv"
TEST_PATH = DATA_DIR / "forecasting_test_preprocessed.csv"

INPUT_LENGTH = 12
HORIZONS = [1, 3, 6]
TARGET = "current_speed"


# =========================================================
# METRICS
# =========================================================
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# =========================================================
# LOAD DATA
# =========================================================
def load_data(path):
    df = pd.read_csv(path)

    df["obs_time_utc"] = pd.to_datetime(
        df["obs_time_utc"],
        utc=True
    )

    # is_observed kadang terbaca string
    if df["is_observed"].dtype == object:
        df["is_observed"] = (
            df["is_observed"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False})
        )

    return df.sort_values(
        ["obs_time_utc", "corridor_file"]
    ).reset_index(drop=True)


train = load_data(TRAIN_PATH)
val = load_data(VAL_PATH)
test = load_data(TEST_PATH)


# =========================================================
# HISTORICAL AVERAGE
# FIT HANYA DARI OBSERVASI ASLI TRAIN
# =========================================================
train_observed = train[
    train["is_observed"] == True
].copy()

train_observed["hour"] = (
    train_observed["obs_time_utc"].dt.hour
)

historical_average = (
    train_observed
    .groupby(["corridor_file", "hour"])[TARGET]
    .mean()
)

# fallback jika corridor-hour tertentu tidak tersedia
corridor_average = (
    train_observed
    .groupby("corridor_file")[TARGET]
    .mean()
)


# =========================================================
# EVALUATION
# =========================================================
def evaluate_split(df, split_name):

    results = []
    detail_rows = []

    # Long gap tidak punya segment_id
    df = df.dropna(subset=["segment_id"]).copy()

    for horizon in HORIZONS:

        y_true_all = []
        naive_all = []
        ha_all = []

        sample_count = 0
        evaluated_points = 0

        for segment_id, segment in df.groupby("segment_id"):

            segment = segment.sort_values(
                ["obs_time_utc", "corridor_file"]
            )

            timestamps = sorted(
                segment["obs_time_utc"].unique()
            )

            corridor_names = sorted(
                segment["corridor_file"].unique()
            )

            n = len(timestamps)

            max_start = (
                n
                - INPUT_LENGTH
                - horizon
                + 1
            )

            if max_start <= 0:
                continue

            for start_idx in range(max_start):

                input_end_idx = (
                    start_idx + INPUT_LENGTH - 1
                )

                target_idx = (
                    input_end_idx + horizon
                )

                input_end_time = timestamps[
                    input_end_idx
                ]

                target_time = timestamps[
                    target_idx
                ]

                valid_sample = False

                for corridor in corridor_names:

                    last_input = segment[
                        (segment["obs_time_utc"] == input_end_time)
                        & (segment["corridor_file"] == corridor)
                    ]

                    target_row = segment[
                        (segment["obs_time_utc"] == target_time)
                        & (segment["corridor_file"] == corridor)
                    ]

                    if (
                        len(last_input) == 0
                        or len(target_row) == 0
                    ):
                        continue

                    # -------------------------------------------------
                    # PENTING:
                    # evaluasi hanya jika target adalah observasi asli
                    # -------------------------------------------------
                    if not bool(
                        target_row.iloc[0]["is_observed"]
                    ):
                        continue

                    true_value = float(
                        target_row.iloc[0][TARGET]
                    )

                    # Naive / persistence
                    naive_pred = float(
                        last_input.iloc[0][TARGET]
                    )

                    # Historical Average
                    target_hour = pd.Timestamp(
                        target_time
                    ).hour

                    key = (
                        corridor,
                        target_hour
                    )

                    if key in historical_average.index:
                        ha_pred = float(
                            historical_average.loc[key]
                        )
                    else:
                        ha_pred = float(
                            corridor_average.loc[corridor]
                        )

                    y_true_all.append(true_value)
                    naive_all.append(naive_pred)
                    ha_all.append(ha_pred)

                    evaluated_points += 1
                    valid_sample = True

                    detail_rows.append({
                        "split": split_name,
                        "horizon": horizon,
                        "target_time": target_time,
                        "corridor": corridor,
                        "y_true": true_value,
                        "naive_pred": naive_pred,
                        "historical_average_pred": ha_pred,
                    })

                if valid_sample:
                    sample_count += 1

        y_true_all = np.array(y_true_all)
        naive_all = np.array(naive_all)
        ha_all = np.array(ha_all)

        if len(y_true_all) == 0:
            continue

        results.append({
            "split": split_name,
            "horizon": horizon,
            "model": "Naive Persistence",
            "samples": sample_count,
            "evaluated_points": evaluated_points,
            "MAE": mae(
                y_true_all,
                naive_all
            ),
            "RMSE": rmse(
                y_true_all,
                naive_all
            ),
        })

        results.append({
            "split": split_name,
            "horizon": horizon,
            "model": "Historical Average",
            "samples": sample_count,
            "evaluated_points": evaluated_points,
            "MAE": mae(
                y_true_all,
                ha_all
            ),
            "RMSE": rmse(
                y_true_all,
                ha_all
            ),
        })

    return (
        pd.DataFrame(results),
        pd.DataFrame(detail_rows)
    )


# =========================================================
# VALIDATION + TEST
# =========================================================
val_results, val_detail = evaluate_split(
    val,
    "VALIDATION"
)

test_results, test_detail = evaluate_split(
    test,
    "TEST"
)

results = pd.concat(
    [val_results, test_results],
    ignore_index=True
)

details = pd.concat(
    [val_detail, test_detail],
    ignore_index=True
)

# =========================================================
# SAVE
# =========================================================
results.to_csv(
    DATA_DIR / "baseline_results.csv",
    index=False
)

details.to_csv(
    DATA_DIR / "baseline_predictions.csv",
    index=False
)

# =========================================================
# PRINT RESULTS
# =========================================================
print("\n=== BASELINE FORECASTING RESULTS ===")

for split in ["VALIDATION", "TEST"]:

    print(f"\n{'=' * 50}")
    print(split)
    print("=" * 50)

    subset = results[
        results["split"] == split
    ]

    for horizon in HORIZONS:

        print(
            f"\n--- Horizon t+{horizon}h ---"
        )

        horizon_df = subset[
            subset["horizon"] == horizon
        ]

        for row in horizon_df.itertuples():

            print(
                f"{row.model:<20} | "
                f"MAE = {row.MAE:.3f} km/h | "
                f"RMSE = {row.RMSE:.3f} km/h | "
                f"Points = {row.evaluated_points}"
            )


print("\n=== SAVED ===")
print("data/processed/baseline_results.csv")
print("data/processed/baseline_predictions.csv")