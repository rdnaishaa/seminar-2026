from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

FILES = {
    "TRAIN": DATA_DIR / "forecasting_train_preprocessed.csv",
    "VALIDATION": DATA_DIR / "forecasting_val_preprocessed.csv",
    "TEST": DATA_DIR / "forecasting_test_preprocessed.csv",
}

INPUT_LENGTHS = [3, 6, 12]
HORIZONS = [1, 3, 6]

results = []

for split_name, path in FILES.items():

    df = pd.read_csv(path)

    df["obs_time_utc"] = pd.to_datetime(
        df["obs_time_utc"],
        utc=True
    )

    # Buang timestamp yang berada di long gap
    df = df.dropna(subset=["segment_id"])

    # Kita hanya butuh satu baris per timestamp
    time_df = (
        df[
            [
                "obs_time_utc",
                "segment_id"
            ]
        ]
        .drop_duplicates()
        .sort_values("obs_time_utc")
    )

    print(f"\n=== {split_name} ===")

    for input_len in INPUT_LENGTHS:

        for horizon in HORIZONS:

            sample_count = 0

            for segment_id, segment in time_df.groupby("segment_id"):

                segment = segment.sort_values(
                    "obs_time_utc"
                )

                n = len(segment)

                # Input:
                # t-input_len+1 ... t
                #
                # Target:
                # t+horizon
                #
                # Jumlah window:
                # n - input_len - horizon + 1

                windows = (
                    n
                    - input_len
                    - horizon
                    + 1
                )

                if windows > 0:
                    sample_count += windows

            results.append({
                "split": split_name,
                "input_length": input_len,
                "horizon": horizon,
                "samples": sample_count,
            })

            print(
                f"Input {input_len:>2}h -> "
                f"t+{horizon}h : "
                f"{sample_count} samples"
            )


results_df = pd.DataFrame(results)

output = (
    DATA_DIR
    / "forecasting_window_summary.csv"
)

results_df.to_csv(
    output,
    index=False
)

print("\n=== SAVED ===")
print(output)