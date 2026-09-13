from pathlib import Path
import pandas as pd
import numpy as np
import re

RESULT_DIR = Path("outputs/results")

models = ["fixed", "adaptive"]

all_rows = []

for model in models:
    files = sorted(
        RESULT_DIR.glob(f"{model}_stgnn_seed*_results.csv")
    )

    for file in files:
        seed_match = re.search(r"seed(\d+)", file.name)
        seed = int(seed_match.group(1))

        df = pd.read_csv(file)

        for _, row in df.iterrows():
            all_rows.append({
                "model": model,
                "seed": seed,
                "horizon": row["horizon"],
                "MAE": row["MAE"],
                "RMSE": row["RMSE"],
            })

results = pd.DataFrame(all_rows)

print("\n=== RAW MULTI-SEED RESULTS ===")
print(results.sort_values(["model", "seed", "horizon"]).to_string(index=False))


summary = (
    results
    .groupby(["model", "horizon"])
    .agg(
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
    )
    .reset_index()
)

print("\n=== MEAN ± STD ===")

for _, row in summary.iterrows():
    print(
        f"{row['model'].capitalize()} "
        f"{row['horizon']} | "
        f"MAE = {row['MAE_mean']:.3f} ± {row['MAE_std']:.3f} | "
        f"RMSE = {row['RMSE_mean']:.3f} ± {row['RMSE_std']:.3f}"
    )


print("\n=== ADAPTIVE VS FIXED PER SEED ===")

fixed = results[results["model"] == "fixed"]
adaptive = results[results["model"] == "adaptive"]

merged = fixed.merge(
    adaptive,
    on=["seed", "horizon"],
    suffixes=("_fixed", "_adaptive")
)

merged["MAE_improvement_pct"] = (
    (merged["MAE_fixed"] - merged["MAE_adaptive"])
    / merged["MAE_fixed"]
    * 100
)

merged["adaptive_wins"] = (
    merged["MAE_adaptive"] < merged["MAE_fixed"]
)

for _, row in merged.sort_values(["seed", "horizon"]).iterrows():
    status = "Adaptive wins" if row["adaptive_wins"] else "Fixed wins"

    print(
        f"Seed {row['seed']} | "
        f"{row['horizon']} | "
        f"Fixed MAE = {row['MAE_fixed']:.3f} | "
        f"Adaptive MAE = {row['MAE_adaptive']:.3f} | "
        f"Improvement = {row['MAE_improvement_pct']:.2f}% | "
        f"{status}"
    )


print("\n=== WIN CONSISTENCY ===")

win_summary = (
    merged
    .groupby("horizon")["adaptive_wins"]
    .agg(["sum", "count"])
    .reset_index()
)

for _, row in win_summary.iterrows():
    print(
        f"{row['horizon']} | "
        f"Adaptive wins {int(row['sum'])}/{int(row['count'])} seeds"
    )


results.to_csv(
    RESULT_DIR / "multiseed_all_results.csv",
    index=False
)

summary.to_csv(
    RESULT_DIR / "multiseed_summary.csv",
    index=False
)

merged.to_csv(
    RESULT_DIR / "multiseed_comparison.csv",
    index=False
)

print("\n=== SAVED ===")
print("outputs/results/multiseed_all_results.csv")
print("outputs/results/multiseed_summary.csv")
print("outputs/results/multiseed_comparison.csv")