from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# PATH
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = ROOT / "outputs" / "results"
PROCESSED_DIR = ROOT / "data" / "processed"


# =========================================================
# LOAD PREDICTIONS
# =========================================================

baseline = pd.read_csv(
    PROCESSED_DIR / "baseline_predictions.csv"
)

gru = pd.read_csv(
    RESULT_DIR / "gru_predictions.csv"
)

nograph = pd.read_csv(
    RESULT_DIR / "nograph_stgnn_predictions.csv"
)

fixed = pd.read_csv(
    RESULT_DIR / "fixed_stgnn_predictions.csv"
)

adaptive = pd.read_csv(
    RESULT_DIR / "adaptive_stgnn_predictions.csv"
)


# Test only
baseline = baseline[
    baseline["split"].str.lower() == "test"
].copy()


# =========================================================
# MERGE ON EXACT SAME TARGET
# =========================================================

keys = [
    "horizon",
    "target_time",
    "corridor"
]

matched = baseline.merge(
    gru[
        keys +
        ["gru_pred"]
    ],
    on=keys,
    how="inner"
)

matched = matched.merge(
    nograph[
        keys +
        ["nograph_stgnn_pred"]
    ],
    on=keys,
    how="inner"
)

matched = matched.merge(
    fixed[
        keys +
        ["fixed_stgnn_pred"]
    ],
    on=keys,
    how="inner"
)

matched = matched.merge(
    adaptive[
        keys +
        ["adaptive_stgnn_pred"]
    ],
    on=keys,
    how="inner"
)


print("\n=== MATCHED DATA ===")
print("Rows :", len(matched))


# =========================================================
# METRIC
# =========================================================

def calculate_metrics(
    y_true,
    y_pred
):

    error = (
        y_true - y_pred
    )

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    return mae, rmse


# =========================================================
# EVALUATE
# =========================================================

models = {
    "Naive Persistence":
        "naive_pred",

    "Historical Average":
        "historical_average_pred",

    "GRU":
        "gru_pred",

    "No-Graph ST-GNN":
        "nograph_stgnn_pred",

    "Fixed ST-GNN":
        "fixed_stgnn_pred",

    "Adaptive ST-GNN":
        "adaptive_stgnn_pred",
}


results = []


print(
    "\n=== FINAL MATCHED MODEL COMPARISON ==="
)

for horizon in sorted(
    matched["horizon"].unique()
):

    subset = matched[
        matched["horizon"] == horizon
    ].copy()

    print(
        f"\n--- Horizon t+{horizon}h ---"
    )

    print(
        "Matched points:",
        len(subset)
    )

    for model_name, pred_col in models.items():

        mae, rmse = calculate_metrics(
            subset["y_true"].values,
            subset[pred_col].values
        )

        results.append({
            "horizon": horizon,
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "points": len(subset)
        })

        print(
            f"{model_name:<22} "
            f"MAE={mae:.3f} "
            f"RMSE={rmse:.3f}"
        )


results_df = pd.DataFrame(
    results
)


# =========================================================
# GRAPH-ONLY COMPARISON
# =========================================================

print(
    "\n=== GRAPH STRATEGY COMPARISON ==="
)

graph_models = [
    "No-Graph ST-GNN",
    "Fixed ST-GNN",
    "Adaptive ST-GNN"
]

graph_results = results_df[
    results_df["model"].isin(
        graph_models
    )
].copy()

for horizon in sorted(
    graph_results["horizon"].unique()
):

    subset = graph_results[
        graph_results["horizon"] == horizon
    ].sort_values("MAE")

    print(
        f"\n--- t+{horizon}h ---"
    )

    print(
        subset[
            [
                "model",
                "MAE",
                "RMSE"
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )


# =========================================================
# ADAPTIVE IMPROVEMENT VS FIXED
# =========================================================

improvement_rows = []

for horizon in sorted(
    results_df["horizon"].unique()
):

    fixed_row = results_df[
        (
            results_df["horizon"] == horizon
        )
        &
        (
            results_df["model"]
            == "Fixed ST-GNN"
        )
    ].iloc[0]

    adaptive_row = results_df[
        (
            results_df["horizon"] == horizon
        )
        &
        (
            results_df["model"]
            == "Adaptive ST-GNN"
        )
    ].iloc[0]

    mae_improvement = (
        (
            fixed_row["MAE"]
            - adaptive_row["MAE"]
        )
        /
        fixed_row["MAE"]
        * 100
    )

    rmse_improvement = (
        (
            fixed_row["RMSE"]
            - adaptive_row["RMSE"]
        )
        /
        fixed_row["RMSE"]
        * 100
    )

    improvement_rows.append({
        "horizon": horizon,
        "MAE_improvement_percent":
            mae_improvement,
        "RMSE_improvement_percent":
            rmse_improvement
    })


improvement_df = pd.DataFrame(
    improvement_rows
)


print(
    "\n=== ADAPTIVE IMPROVEMENT VS FIXED ==="
)

print(
    improvement_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# =========================================================
# SAVE
# =========================================================

results_df.to_csv(
    RESULT_DIR /
    "all_models_matched_comparison.csv",
    index=False
)

matched.to_csv(
    RESULT_DIR /
    "all_models_matched_predictions.csv",
    index=False
)

improvement_df.to_csv(
    RESULT_DIR /
    "adaptive_vs_fixed_improvement.csv",
    index=False
)


print("\n=== SAVED ===")

print(
    "outputs/results/"
    "all_models_matched_comparison.csv"
)

print(
    "outputs/results/"
    "all_models_matched_predictions.csv"
)

print(
    "outputs/results/"
    "adaptive_vs_fixed_improvement.csv"
)