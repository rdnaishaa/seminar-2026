from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
RESULT_DIR = ROOT / "outputs" / "results"

BASELINE_PATH = DATA_DIR / "baseline_predictions.csv"
GRU_PATH = RESULT_DIR / "gru_predictions.csv"

baseline = pd.read_csv(BASELINE_PATH)
gru = pd.read_csv(GRU_PATH)

baseline["target_time"] = pd.to_datetime(
    baseline["target_time"],
    utc=True
)

gru["target_time"] = pd.to_datetime(
    gru["target_time"],
    utc=True
)

# Ambil TEST saja
baseline = baseline[
    baseline["split"] == "TEST"
].copy()

# Gabungkan hanya titik yang benar-benar sama
merged = pd.merge(
    gru,
    baseline,
    on=[
        "horizon",
        "target_time",
        "corridor"
    ],
    how="inner",
    suffixes=("_gru", "_baseline")
)

print("\n=== MATCHED EVALUATION ===")
print(f"Matched rows: {len(merged)}")

results = []

for horizon in [1, 3, 6]:

    h = merged[
        merged["horizon"] == horizon
    ].copy()

    # Gunakan ground truth dari GRU
    y_true = h["y_true_gru"].to_numpy()

    predictions = {
        "Naive Persistence":
            h["naive_pred"].to_numpy(),

        "Historical Average":
            h["historical_average_pred"].to_numpy(),

        "GRU":
            h["gru_pred"].to_numpy()
    }

    print(f"\n--- Horizon t+{horizon}h ---")
    print(f"Evaluated points: {len(h)}")

    for model_name, y_pred in predictions.items():

        mae = np.mean(
            np.abs(y_true - y_pred)
        )

        rmse = np.sqrt(
            np.mean(
                (y_true - y_pred) ** 2
            )
        )

        results.append({
            "horizon": horizon,
            "model": model_name,
            "evaluated_points": len(h),
            "MAE": mae,
            "RMSE": rmse
        })

        print(
            f"{model_name:<20} | "
            f"MAE = {mae:.3f} km/h | "
            f"RMSE = {rmse:.3f} km/h"
        )

results_df = pd.DataFrame(results)

output = (
    RESULT_DIR
    / "baseline_gru_comparison.csv"
)

results_df.to_csv(
    output,
    index=False
)

print("\n=== SAVED ===")
print("outputs/results/baseline_gru_comparison.csv")