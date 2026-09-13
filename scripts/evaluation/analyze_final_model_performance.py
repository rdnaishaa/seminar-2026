from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_ROOT = ROOT / "outputs" / "final_64"

BASELINE_DIR = OUTPUT_ROOT / "baselines"
GRU_DIR = OUTPUT_ROOT / "gru"
FIXED_DIR = OUTPUT_ROOT / "fixed_stgnn"
ADAPTIVE_DIR = OUTPUT_ROOT / "adaptive_stgnn"

ANALYSIS_DIR = OUTPUT_ROOT / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 6]


# ============================================================
# LOAD RESULTS
# ============================================================

baseline = pd.read_csv(
    BASELINE_DIR / "baseline_results.csv"
)

gru = pd.read_csv(
    GRU_DIR / "gru_multiseed_results.csv"
)

fixed = pd.read_csv(
    FIXED_DIR / "fixed_stgnn_multiseed_results.csv"
)

adaptive = pd.read_csv(
    ADAPTIVE_DIR / "adaptive_stgnn_multiseed_results.csv"
)

# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

baseline = baseline.rename(
    columns={
        "mae": "MAE",
        "rmse": "RMSE",
        "observed_points": "evaluated_points",
    }
)

gru = gru.rename(
    columns={
        "mae": "MAE",
        "rmse": "RMSE",
        "observed_points": "evaluated_points",
    }
)

print("\n=== FILES LOADED ===")
print(f"Baseline rows : {len(baseline)}")
print(f"GRU rows      : {len(gru)}")
print(f"Fixed rows    : {len(fixed)}")
print(f"Adaptive rows : {len(adaptive)}")


# ============================================================
# INSPECT COLUMN NAMES
# ============================================================

print("\nBaseline columns:")
print(baseline.columns.tolist())

print("\nGRU columns:")
print(gru.columns.tolist())

print("\nFixed columns:")
print(fixed.columns.tolist())

print("\nAdaptive columns:")
print(adaptive.columns.tolist())


# ============================================================
# STANDARDIZE BASELINE
# ============================================================

baseline = baseline.copy()

if "model" not in baseline.columns:
    raise KeyError(
        "baseline_results.csv harus punya kolom 'model'"
    )

if "horizon" not in baseline.columns:
    raise KeyError(
        "baseline_results.csv harus punya kolom 'horizon'"
    )

baseline_summary = baseline[
    [
        "model",
        "horizon",
        "MAE",
        "RMSE",
        "evaluated_points",
    ]
].copy()

baseline_summary = baseline_summary.rename(
    columns={
        "MAE": "MAE_mean",
        "RMSE": "RMSE_mean",
    }
)

baseline_summary["MAE_std"] = 0.0
baseline_summary["RMSE_std"] = 0.0
baseline_summary["num_seeds"] = 1


# ============================================================
# FUNCTION: MULTI-SEED SUMMARY
# ============================================================

def summarize_multiseed(
    df,
    model_name,
):

    rows = []

    for horizon in HORIZONS:

        h = df[
            df["horizon"] == horizon
        ]

        rows.append(
            {
                "model": model_name,
                "horizon": horizon,
                "MAE_mean": h["MAE"].mean(),
                "MAE_std": h["MAE"].std(ddof=0),
                "RMSE_mean": h["RMSE"].mean(),
                "RMSE_std": h["RMSE"].std(ddof=0),
                "num_seeds": len(h),
                "evaluated_points": int(
                    h["evaluated_points"].iloc[0]
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SUMMARIZE MODELS
# ============================================================

gru_summary = summarize_multiseed(
    gru,
    "GRU",
)

fixed_summary = summarize_multiseed(
    fixed,
    "Fixed ST-GNN",
)

adaptive_summary = summarize_multiseed(
    adaptive,
    "Adaptive ST-GNN",
)


# ============================================================
# COMBINE
# ============================================================

final_summary = pd.concat(
    [
        baseline_summary,
        gru_summary,
        fixed_summary,
        adaptive_summary,
    ],
    ignore_index=True,
)


# normalize names if needed
final_summary["model"] = (
    final_summary["model"]
    .replace(
        {
            "Naive Persistence": "Naive",
            "Historical Average": "Historical Average",
        }
    )
)


model_order = [
    "Naive",
    "Historical Average",
    "GRU",
    "Fixed ST-GNN",
    "Adaptive ST-GNN",
]


final_summary["model"] = pd.Categorical(
    final_summary["model"],
    categories=model_order,
    ordered=True,
)


final_summary = (
    final_summary
    .sort_values(
        [
            "horizon",
            "model",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# SAVE FINAL TABLE
# ============================================================

final_summary.to_csv(
    ANALYSIS_DIR
    / "final_model_performance_summary.csv",
    index=False,
)


# ============================================================
# PRINT FINAL TABLE
# ============================================================

print("\n" + "=" * 80)
print("FINAL MODEL PERFORMANCE")
print("=" * 80)


for horizon in HORIZONS:

    h = final_summary[
        final_summary["horizon"]
        == horizon
    ]

    print(f"\n--- t+{horizon}h ---")

    print(
        h[
            [
                "model",
                "MAE_mean",
                "MAE_std",
                "RMSE_mean",
                "RMSE_std",
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# BEST MODEL PER HORIZON
# ============================================================

best_rows = []


for horizon in HORIZONS:

    h = final_summary[
        final_summary["horizon"]
        == horizon
    ]


    best_mae_row = (
        h.loc[
            h["MAE_mean"].idxmin()
        ]
    )


    best_rmse_row = (
        h.loc[
            h["RMSE_mean"].idxmin()
        ]
    )


    best_rows.append(
        {
            "horizon": horizon,
            "best_mae_model": best_mae_row["model"],
            "best_mae": best_mae_row["MAE_mean"],
            "best_rmse_model": best_rmse_row["model"],
            "best_rmse": best_rmse_row["RMSE_mean"],
        }
    )


best_df = pd.DataFrame(
    best_rows
)


best_df.to_csv(
    ANALYSIS_DIR
    / "best_model_per_horizon.csv",
    index=False,
)


print("\n=== BEST MODEL PER HORIZON ===")
print(
    best_df.to_string(
        index=False
    )
)


# ============================================================
# ADAPTIVE VS FIXED IMPROVEMENT
# ============================================================

fixed_cmp = (
    final_summary[
        final_summary["model"]
        == "Fixed ST-GNN"
    ]
    .copy()
)


adaptive_cmp = (
    final_summary[
        final_summary["model"]
        == "Adaptive ST-GNN"
    ]
    .copy()
)


comparison = fixed_cmp.merge(
    adaptive_cmp,
    on="horizon",
    suffixes=(
        "_fixed",
        "_adaptive",
    ),
)


comparison[
    "mae_improvement"
] = (
    comparison["MAE_mean_fixed"]
    - comparison["MAE_mean_adaptive"]
)


comparison[
    "mae_improvement_pct"
] = (
    comparison["mae_improvement"]
    / comparison["MAE_mean_fixed"]
    * 100
)


comparison[
    "rmse_improvement"
] = (
    comparison["RMSE_mean_fixed"]
    - comparison["RMSE_mean_adaptive"]
)


comparison[
    "rmse_improvement_pct"
] = (
    comparison["rmse_improvement"]
    / comparison["RMSE_mean_fixed"]
    * 100
)


comparison[
    [
        "horizon",
        "MAE_mean_fixed",
        "MAE_mean_adaptive",
        "mae_improvement",
        "mae_improvement_pct",
        "RMSE_mean_fixed",
        "RMSE_mean_adaptive",
        "rmse_improvement",
        "rmse_improvement_pct",
    ]
].to_csv(
    ANALYSIS_DIR
    / "adaptive_vs_fixed_performance.csv",
    index=False,
)


print("\n=== ADAPTIVE VS FIXED ===")

print(
    comparison[
        [
            "horizon",
            "mae_improvement",
            "mae_improvement_pct",
            "rmse_improvement",
            "rmse_improvement_pct",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# FIGURE 1: MAE ALL MODELS
# ============================================================

plot_df = final_summary.copy()


x = np.arange(
    len(HORIZONS)
)

width = 0.15


plt.figure(
    figsize=(11, 6)
)


for idx, model in enumerate(
    model_order
):

    model_df = (
        plot_df[
            plot_df["model"]
            == model
        ]
        .sort_values(
            "horizon"
        )
    )


    plt.bar(
        x
        + (
            idx
            - (
                len(model_order)
                - 1
            )
            / 2
        )
        * width,
        model_df["MAE_mean"],
        width,
        yerr=model_df["MAE_std"],
        capsize=3,
        label=model,
    )


plt.xticks(
    x,
    [
        f"t+{h}"
        for h in HORIZONS
    ],
)

plt.ylabel(
    "MAE"
)

plt.xlabel(
    "Forecast horizon"
)

plt.title(
    "Final Model Comparison — MAE"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    ANALYSIS_DIR
    / "fig_final_model_mae.png",
    dpi=300,
)

plt.close()


# ============================================================
# FIGURE 2: RMSE ALL MODELS
# ============================================================

plt.figure(
    figsize=(11, 6)
)


for idx, model in enumerate(
    model_order
):

    model_df = (
        plot_df[
            plot_df["model"]
            == model
        ]
        .sort_values(
            "horizon"
        )
    )


    plt.bar(
        x
        + (
            idx
            - (
                len(model_order)
                - 1
            )
            / 2
        )
        * width,
        model_df["RMSE_mean"],
        width,
        yerr=model_df["RMSE_std"],
        capsize=3,
        label=model,
    )


plt.xticks(
    x,
    [
        f"t+{h}"
        for h in HORIZONS
    ],
)

plt.ylabel(
    "RMSE"
)

plt.xlabel(
    "Forecast horizon"
)

plt.title(
    "Final Model Comparison — RMSE"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    ANALYSIS_DIR
    / "fig_final_model_rmse.png",
    dpi=300,
)

plt.close()


# ============================================================
# FIGURE 3: ADAPTIVE VS FIXED MAE
# ============================================================

fixed_plot = (
    fixed_cmp
    .sort_values(
        "horizon"
    )
)

adaptive_plot = (
    adaptive_cmp
    .sort_values(
        "horizon"
    )
)


plt.figure(
    figsize=(8, 5)
)


plt.errorbar(
    HORIZONS,
    fixed_plot["MAE_mean"],
    yerr=fixed_plot["MAE_std"],
    marker="o",
    capsize=4,
    label="Fixed ST-GNN",
)


plt.errorbar(
    HORIZONS,
    adaptive_plot["MAE_mean"],
    yerr=adaptive_plot["MAE_std"],
    marker="o",
    capsize=4,
    label="Adaptive ST-GNN",
)


plt.xticks(
    HORIZONS,
    [
        f"t+{h}"
        for h in HORIZONS
    ],
)

plt.ylabel(
    "MAE"
)

plt.xlabel(
    "Forecast horizon"
)

plt.title(
    "Fixed vs Adaptive ST-GNN — MAE"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    ANALYSIS_DIR
    / "fig_fixed_vs_adaptive_mae.png",
    dpi=300,
)

plt.close()


# ============================================================
# FIGURE 4: ADAPTIVE IMPROVEMENT %
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.bar(
    [
        f"t+{h}"
        for h in comparison["horizon"]
    ],
    comparison["mae_improvement_pct"],
)


plt.ylabel(
    "MAE improvement (%)"
)

plt.xlabel(
    "Forecast horizon"
)

plt.title(
    "Adaptive Improvement over Fixed ST-GNN"
)

plt.tight_layout()

plt.savefig(
    ANALYSIS_DIR
    / "fig_adaptive_improvement_pct.png",
    dpi=300,
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print("\n=== SAVED ===")

print(
    ANALYSIS_DIR
    / "final_model_performance_summary.csv"
)

print(
    ANALYSIS_DIR
    / "best_model_per_horizon.csv"
)

print(
    ANALYSIS_DIR
    / "adaptive_vs_fixed_performance.csv"
)

print("\nFigures:")

print(
    "- fig_final_model_mae.png"
)

print(
    "- fig_final_model_rmse.png"
)

print(
    "- fig_fixed_vs_adaptive_mae.png"
)

print(
    "- fig_adaptive_improvement_pct.png"
)