from pathlib import Path
import pandas as pd
import numpy as np

RESULTS_DIR = Path("outputs/results")
SEEDS = [42, 123, 456, 789, 2026]

rows = []

for seed in SEEDS:
    fixed_path = RESULTS_DIR / f"fixed_stgnn_seed{seed}_predictions.csv"
    adaptive_path = RESULTS_DIR / f"adaptive_stgnn_seed{seed}_predictions.csv"

    fixed = pd.read_csv(fixed_path)
    adaptive = pd.read_csv(adaptive_path)

    # Calculate absolute error for each prediction
    fixed["abs_error"] = np.abs(
        fixed["y_true"] - fixed["fixed_stgnn_pred"]
    )

    adaptive["abs_error"] = np.abs(
        adaptive["y_true"] - adaptive["adaptive_stgnn_pred"]
    )

    # MAE for every corridor × horizon
    fixed_mae = (
        fixed.groupby(["corridor", "horizon"])["abs_error"]
        .mean()
        .reset_index(name="fixed_mae")
    )

    adaptive_mae = (
        adaptive.groupby(["corridor", "horizon"])["abs_error"]
        .mean()
        .reset_index(name="adaptive_mae")
    )

    comparison = fixed_mae.merge(
        adaptive_mae,
        on=["corridor", "horizon"],
        how="inner"
    )

    comparison["seed"] = seed

    comparison["improvement_pct"] = (
        (comparison["fixed_mae"] - comparison["adaptive_mae"])
        / comparison["fixed_mae"]
        * 100
    )

    rows.append(comparison)


# ============================================================
# ALL SEEDS
# ============================================================

all_results = pd.concat(rows, ignore_index=True)

all_results = all_results[
    [
        "seed",
        "corridor",
        "horizon",
        "fixed_mae",
        "adaptive_mae",
        "improvement_pct",
    ]
]

all_results.to_csv(
    RESULTS_DIR / "corridor_multiseed_results.csv",
    index=False
)


# ============================================================
# SUMMARY ACROSS SEEDS
# ============================================================

summary = (
    all_results
    .groupby(["corridor", "horizon"])
    .agg(
        fixed_mae_mean=("fixed_mae", "mean"),
        fixed_mae_std=("fixed_mae", "std"),
        adaptive_mae_mean=("adaptive_mae", "mean"),
        adaptive_mae_std=("adaptive_mae", "std"),
        improvement_mean_pct=("improvement_pct", "mean"),
    )
    .reset_index()
)

summary["winner"] = np.where(
    summary["adaptive_mae_mean"] < summary["fixed_mae_mean"],
    "Adaptive",
    "Fixed"
)

summary.to_csv(
    RESULTS_DIR / "corridor_multiseed_summary.csv",
    index=False
)


# ============================================================
# WIN CONSISTENCY
# ============================================================

all_results["adaptive_win"] = (
    all_results["adaptive_mae"] < all_results["fixed_mae"]
)

wins = (
    all_results
    .groupby(["corridor", "horizon"])["adaptive_win"]
    .sum()
    .reset_index(name="adaptive_wins")
)

wins["total_seeds"] = len(SEEDS)

wins.to_csv(
    RESULTS_DIR / "corridor_win_consistency.csv",
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n=== CORRIDOR-LEVEL MULTI-SEED SUMMARY ===")

for _, row in summary.iterrows():
    print(
        f"{row['corridor']:15s} | "
        f"t+{int(row['horizon'])}h | "
        f"Fixed = {row['fixed_mae_mean']:.3f} ± {row['fixed_mae_std']:.3f} | "
        f"Adaptive = {row['adaptive_mae_mean']:.3f} ± {row['adaptive_mae_std']:.3f} | "
        f"Improvement = {row['improvement_mean_pct']:.2f}% | "
        f"{row['winner']} wins"
    )


print("\n=== ADAPTIVE WIN CONSISTENCY PER CORRIDOR ===")

for _, row in wins.iterrows():
    print(
        f"{row['corridor']:15s} | "
        f"t+{int(row['horizon'])}h | "
        f"Adaptive wins {int(row['adaptive_wins'])}/{int(row['total_seeds'])} seeds"
    )


print("\n=== SAVED ===")
print("outputs/results/corridor_multiseed_results.csv")
print("outputs/results/corridor_multiseed_summary.csv")
print("outputs/results/corridor_win_consistency.csv")