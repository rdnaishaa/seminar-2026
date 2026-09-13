from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("outputs/results")
FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS_DIR / "corridor_multiseed_summary.csv")

for horizon in [1, 3, 6]:
    d = df[df["horizon"] == horizon].copy()

    # Sort by Adaptive improvement
    d = d.sort_values("improvement_mean_pct")

    labels = (
        d["corridor"]
        .str.replace("tt_", "", regex=False)
        .tolist()
    )

    x = np.arange(len(d))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.bar(
        x - width / 2,
        d["fixed_mae_mean"],
        width,
        yerr=d["fixed_mae_std"],
        capsize=3,
        label="Fixed Graph"
    )

    ax.bar(
        x + width / 2,
        d["adaptive_mae_mean"],
        width,
        yerr=d["adaptive_mae_std"],
        capsize=3,
        label="Adaptive Graph"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_ylabel("MAE (km/h)")
    ax.set_xlabel("Corridor")
    ax.set_title(
        f"Corridor-Level Fixed vs Adaptive ST-GNN — t+{horizon}h"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    output = FIG_DIR / f"corridor_error_t{horizon}.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output}")

print("\nDone.")