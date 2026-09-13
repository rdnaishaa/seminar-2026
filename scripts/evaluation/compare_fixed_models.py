from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "outputs" / "results"

# =========================================================
# LOAD
# =========================================================
baseline = pd.read_csv(
    ROOT / "data" / "processed" / "baseline_predictions.csv"
)

gru = pd.read_csv(
    RESULT_DIR / "gru_predictions.csv"
)

fixed = pd.read_csv(
    RESULT_DIR / "fixed_stgnn_predictions.csv"
)

# =========================================================
# NORMALIZE COLUMN NAMES
# =========================================================
for df in [baseline, gru, fixed]:

    if "target_time" in df.columns:
        df["target_time"] = pd.to_datetime(
            df["target_time"],
            utc=True
        )

# Only test split if split column exists
if "split" in baseline.columns:
    baseline = baseline[
        baseline["split"].str.lower() == "test"
    ].copy()

if "split" in gru.columns:
    gru = gru[
        gru["split"].str.lower() == "test"
    ].copy()

# =========================================================
# SHOW COLUMNS
# =========================================================
print("\n=== AVAILABLE COLUMNS ===")
print("Baseline:", baseline.columns.tolist())
print("GRU     :", gru.columns.tolist())
print("Fixed   :", fixed.columns.tolist())


# =========================================================
# DETECT PREDICTION COLUMNS
# =========================================================
def find_column(columns, candidates):

    for candidate in candidates:
        if candidate in columns:
            return candidate

    return None


naive_col = find_column(
    baseline.columns,
    [
        "naive_pred",
        "naive_prediction",
        "persistence_pred"
    ]
)

ha_col = find_column(
    baseline.columns,
    [
        "ha_pred",
        "historical_average_pred",
        "historical_avg_pred"
    ]
)

gru_col = find_column(
    gru.columns,
    [
        "gru_pred",
        "prediction",
        "y_pred"
    ]
)

fixed_col = find_column(
    fixed.columns,
    [
        "fixed_stgnn_pred",
        "prediction",
        "y_pred"
    ]
)


if naive_col is None:
    raise ValueError(
        "Naive prediction column tidak ditemukan."
    )

if ha_col is None:
    raise ValueError(
        "Historical Average prediction column tidak ditemukan."
    )

if gru_col is None:
    raise ValueError(
        "GRU prediction column tidak ditemukan."
    )

if fixed_col is None:
    raise ValueError(
        "Fixed ST-GNN prediction column tidak ditemukan."
    )


# =========================================================
# MERGE EXACT SAME TARGETS
# =========================================================
keys = [
    "horizon",
    "target_time",
    "corridor"
]

base_keep = baseline[
    keys + [
        "y_true",
        naive_col,
        ha_col
    ]
].copy()

base_keep = base_keep.rename(
    columns={
        naive_col: "Naive",
        ha_col: "Historical Average"
    }
)

gru_keep = gru[
    keys + [
        gru_col
    ]
].copy()

gru_keep = gru_keep.rename(
    columns={
        gru_col: "GRU"
    }
)

fixed_keep = fixed[
    keys + [
        fixed_col
    ]
].copy()

fixed_keep = fixed_keep.rename(
    columns={
        fixed_col: "Fixed ST-GNN"
    }
)


merged = (
    base_keep
    .merge(
        gru_keep,
        on=keys,
        how="inner"
    )
    .merge(
        fixed_keep,
        on=keys,
        how="inner"
    )
)


print("\n=== MATCHED DATA ===")
print(f"Matched rows: {len(merged)}")


# =========================================================
# METRICS
# =========================================================
models = [
    "Naive",
    "Historical Average",
    "GRU",
    "Fixed ST-GNN"
]

results = []


for horizon in sorted(
    merged["horizon"].unique()
):

    subset = merged[
        merged["horizon"] == horizon
    ]

    print(
        f"\n=== t+{horizon}h ==="
    )

    print(
        f"Matched points: {len(subset)}"
    )

    true = subset["y_true"].values

    for model_name in models:

        pred = subset[
            model_name
        ].values

        mae = mean_absolute_error(
            true,
            pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                true,
                pred
            )
        )

        print(
            f"{model_name:20s} "
            f"MAE={mae:.3f} "
            f"RMSE={rmse:.3f}"
        )

        results.append({
            "horizon": horizon,
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "points": len(subset)
        })


# =========================================================
# FIXED ST-GNN PER CORRIDOR
# =========================================================
print(
    "\n=== FIXED ST-GNN ERROR PER CORRIDOR ==="
)

corridor_results = []

for corridor in sorted(
    merged["corridor"].unique()
):

    subset = merged[
        merged["corridor"] == corridor
    ]

    true = subset["y_true"].values
    pred = subset["Fixed ST-GNN"].values

    mae = mean_absolute_error(
        true,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            true,
            pred
        )
    )

    corridor_results.append({
        "corridor": corridor,
        "MAE": mae,
        "RMSE": rmse,
        "points": len(subset)
    })


corridor_df = pd.DataFrame(
    corridor_results
).sort_values("MAE")


print(
    corridor_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# =========================================================
# TRAINING HISTORY AUDIT
# =========================================================
history_path = (
    RESULT_DIR /
    "fixed_stgnn_training_history.csv"
)

history = pd.read_csv(
    history_path
)

best_epoch = (
    history["val_loss"].idxmin()
    + 1
)

best_val_loss = (
    history["val_loss"].min()
)

print("\n=== TRAINING AUDIT ===")
print(
    f"Epochs run    : {len(history)}"
)
print(
    f"Best epoch    : {best_epoch}"
)
print(
    f"Best val_loss : {best_val_loss:.4f}"
)


# =========================================================
# SAVE
# =========================================================
pd.DataFrame(
    results
).to_csv(
    RESULT_DIR /
    "fixed_model_comparison.csv",
    index=False
)

corridor_df.to_csv(
    RESULT_DIR /
    "fixed_stgnn_per_corridor.csv",
    index=False
)

merged.to_csv(
    RESULT_DIR /
    "fixed_model_matched_predictions.csv",
    index=False
)


print("\n=== SAVED ===")
print(
    "outputs/results/fixed_model_comparison.csv"
)
print(
    "outputs/results/fixed_stgnn_per_corridor.csv"
)
print(
    "outputs/results/fixed_model_matched_predictions.csv"
)