from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FIXED_DIR = (
    ROOT
    / "outputs"
    / "final_64"
    / "fixed_stgnn"
)

ADAPTIVE_DIR = (
    ROOT
    / "outputs"
    / "final_64"
    / "adaptive_stgnn"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "final_64"
    / "analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HORIZONS = [1, 3, 6]


# ============================================================
# LOAD RESULTS
# ============================================================

fixed = pd.read_csv(
    FIXED_DIR
    / "fixed_stgnn_corridor_results.csv"
)

adaptive = pd.read_csv(
    ADAPTIVE_DIR
    / "adaptive_stgnn_corridor_results.csv"
)


print("\n=== INPUT ===")
print(f"Fixed rows    : {len(fixed)}")
print(f"Adaptive rows : {len(adaptive)}")


# ============================================================
# MULTI-SEED MEAN PER CORRIDOR
# ============================================================

fixed_mean = (
    fixed
    .groupby(
        [
            "horizon",
            "node_index",
            "corridor",
        ],
        as_index=False,
    )
    .agg(
        fixed_mae=("MAE", "mean"),
        fixed_mae_std=("MAE", "std"),
        fixed_rmse=("RMSE", "mean"),
        fixed_rmse_std=("RMSE", "std"),
        fixed_points=("evaluated_points", "sum"),
    )
)


adaptive_mean = (
    adaptive
    .groupby(
        [
            "horizon",
            "node_index",
            "corridor",
        ],
        as_index=False,
    )
    .agg(
        adaptive_mae=("MAE", "mean"),
        adaptive_mae_std=("MAE", "std"),
        adaptive_rmse=("RMSE", "mean"),
        adaptive_rmse_std=("RMSE", "std"),
        adaptive_points=("evaluated_points", "sum"),
    )
)


# ============================================================
# MERGE
# ============================================================

comparison = fixed_mean.merge(
    adaptive_mean,
    on=[
        "horizon",
        "node_index",
        "corridor",
    ],
    how="inner",
)


# ============================================================
# IMPROVEMENT
#
# positive = Adaptive better
# ============================================================

comparison[
    "mae_improvement"
] = (
    comparison["fixed_mae"]
    - comparison["adaptive_mae"]
)


comparison[
    "mae_improvement_pct"
] = (
    comparison["mae_improvement"]
    / comparison["fixed_mae"]
    * 100
)


comparison[
    "rmse_improvement"
] = (
    comparison["fixed_rmse"]
    - comparison["adaptive_rmse"]
)


comparison[
    "rmse_improvement_pct"
] = (
    comparison["rmse_improvement"]
    / comparison["fixed_rmse"]
    * 100
)


comparison[
    "winner_mae"
] = np.where(
    comparison["adaptive_mae"]
    < comparison["fixed_mae"],
    "Adaptive",
    np.where(
        comparison["adaptive_mae"]
        > comparison["fixed_mae"],
        "Fixed",
        "Tie",
    ),
)


comparison[
    "winner_rmse"
] = np.where(
    comparison["adaptive_rmse"]
    < comparison["fixed_rmse"],
    "Adaptive",
    np.where(
        comparison["adaptive_rmse"]
        > comparison["fixed_rmse"],
        "Fixed",
        "Tie",
    ),
)


# ============================================================
# SAVE FULL COMPARISON
# ============================================================

comparison.to_csv(
    OUTPUT_DIR
    / "fixed_vs_adaptive_corridor_comparison.csv",
    index=False,
)


# ============================================================
# SUMMARY PER HORIZON
# ============================================================

summary_rows = []


for horizon in HORIZONS:

    h = comparison[
        comparison["horizon"]
        == horizon
    ]


    adaptive_wins = (
        h["winner_mae"]
        == "Adaptive"
    ).sum()


    fixed_wins = (
        h["winner_mae"]
        == "Fixed"
    ).sum()


    ties = (
        h["winner_mae"]
        == "Tie"
    ).sum()


    summary_rows.append(
        {
            "horizon": horizon,
            "num_corridors": len(h),
            "adaptive_wins_mae": int(adaptive_wins),
            "fixed_wins_mae": int(fixed_wins),
            "ties_mae": int(ties),
            "mean_mae_improvement": h[
                "mae_improvement"
            ].mean(),
            "median_mae_improvement": h[
                "mae_improvement"
            ].median(),
            "mean_mae_improvement_pct": h[
                "mae_improvement_pct"
            ].mean(),
            "mean_rmse_improvement": h[
                "rmse_improvement"
            ].mean(),
            "mean_rmse_improvement_pct": h[
                "rmse_improvement_pct"
            ].mean(),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    OUTPUT_DIR
    / "corridor_improvement_summary.csv",
    index=False,
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CORRIDOR-LEVEL FIXED VS ADAPTIVE")
print("=" * 70)


for _, row in summary_df.iterrows():

    print(
        f"\nt+{int(row['horizon'])}h"
    )

    print(
        f"Adaptive wins : "
        f"{int(row['adaptive_wins_mae'])}/"
        f"{int(row['num_corridors'])}"
    )

    print(
        f"Fixed wins    : "
        f"{int(row['fixed_wins_mae'])}/"
        f"{int(row['num_corridors'])}"
    )

    print(
        f"Mean MAE improvement: "
        f"{row['mean_mae_improvement']:.4f}"
    )

    print(
        f"Mean MAE improvement %: "
        f"{row['mean_mae_improvement_pct']:.2f}%"
    )


# ============================================================
# TOP / WORST CORRIDORS PER HORIZON
# ============================================================

top_rows = []


for horizon in HORIZONS:

    h = (
        comparison[
            comparison["horizon"]
            == horizon
        ]
        .sort_values(
            "mae_improvement",
            ascending=False,
        )
    )


    print(
        "\n"
        + "-" * 70
    )

    print(
        f"t+{horizon}h — TOP 5 Adaptive Improvement"
    )


    print(
        h[
            [
                "corridor",
                "fixed_mae",
                "adaptive_mae",
                "mae_improvement",
                "mae_improvement_pct",
            ]
        ]
        .head(5)
        .to_string(
            index=False
        )
    )


    print(
        f"\nt+{horizon}h — TOP 5 Fixed Advantage"
    )


    print(
        h[
            [
                "corridor",
                "fixed_mae",
                "adaptive_mae",
                "mae_improvement",
                "mae_improvement_pct",
            ]
        ]
        .tail(5)
        .sort_values(
            "mae_improvement"
        )
        .to_string(
            index=False
        )
    )


    top_adaptive = h.head(5).copy()
    top_adaptive["category"] = "Adaptive strongest"

    top_fixed = (
        h.tail(5)
        .sort_values(
            "mae_improvement"
        )
        .copy()
    )

    top_fixed["category"] = "Fixed strongest"

    top_rows.append(
        top_adaptive
    )

    top_rows.append(
        top_fixed
    )


top_corridors_df = pd.concat(
    top_rows,
    ignore_index=True,
)


top_corridors_df.to_csv(
    OUTPUT_DIR
    / "top_corridor_differences.csv",
    index=False,
)


# ============================================================
# CROSS-HORIZON CONSISTENCY
# ============================================================

pivot = comparison.pivot_table(
    index=[
        "node_index",
        "corridor",
    ],
    columns="horizon",
    values="mae_improvement",
)


pivot.columns = [
    f"improvement_t{int(h)}"
    for h in pivot.columns
]


pivot = pivot.reset_index()


improvement_cols = [
    c
    for c in pivot.columns
    if c.startswith(
        "improvement_t"
    )
]


pivot[
    "adaptive_win_count"
] = (
    pivot[
        improvement_cols
    ]
    > 0
).sum(
    axis=1
)


pivot[
    "fixed_win_count"
] = (
    pivot[
        improvement_cols
    ]
    < 0
).sum(
    axis=1
)


pivot[
    "mean_improvement"
] = (
    pivot[
        improvement_cols
    ]
    .mean(
        axis=1
    )
)


pivot[
    "consistency"
] = np.select(
    [
        pivot[
            "adaptive_win_count"
        ]
        == len(
            HORIZONS
        ),

        pivot[
            "fixed_win_count"
        ]
        == len(
            HORIZONS
        ),
    ],
    [
        "Adaptive wins all horizons",
        "Fixed wins all horizons",
    ],
    default="Mixed",
)


pivot = pivot.sort_values(
    "mean_improvement",
    ascending=False,
)


pivot.to_csv(
    OUTPUT_DIR
    / "corridor_cross_horizon_consistency.csv",
    index=False,
)


print(
    "\n"
    + "=" * 70
)

print(
    "CROSS-HORIZON CONSISTENCY"
)

print(
    "=" * 70
)


print(
    pivot[
        "consistency"
    ]
    .value_counts()
)


# ============================================================
# FIGURE 1
# MAE IMPROVEMENT PER CORRIDOR, EACH HORIZON
# ============================================================

for horizon in HORIZONS:

    h = (
        comparison[
            comparison["horizon"]
            == horizon
        ]
        .sort_values(
            "mae_improvement",
            ascending=True,
        )
    )


    plt.figure(
        figsize=(
            10,
            14,
        )
    )


    plt.barh(
        h["corridor"],
        h["mae_improvement"],
    )


    plt.axvline(
        0,
        linewidth=1,
    )


    plt.xlabel(
        "MAE improvement (Fixed - Adaptive)"
    )


    plt.ylabel(
        "Corridor"
    )


    plt.title(
        f"Adaptive vs Fixed MAE Improvement — t+{horizon}h"
    )


    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIR
        / f"fig_corridor_mae_improvement_t{horizon}.png",
        dpi=300,
    )


    plt.close()


# ============================================================
# FIGURE 2
# WIN COUNTS PER HORIZON
# ============================================================

win_plot = summary_df[
    [
        "horizon",
        "adaptive_wins_mae",
        "fixed_wins_mae",
    ]
].copy()


x = np.arange(
    len(
        HORIZONS
    )
)

width = 0.35


plt.figure(
    figsize=(
        8,
        5,
    )
)


plt.bar(
    x - width / 2,
    win_plot[
        "adaptive_wins_mae"
    ],
    width,
    label="Adaptive",
)


plt.bar(
    x + width / 2,
    win_plot[
        "fixed_wins_mae"
    ],
    width,
    label="Fixed",
)


plt.xticks(
    x,
    [
        f"t+{h}"
        for h in HORIZONS
    ],
)


plt.ylabel(
    "Number of corridors"
)


plt.title(
    "Corridor-Level MAE Wins"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "fig_corridor_win_counts.png",
    dpi=300,
)


plt.close()


# ============================================================
# FIGURE 3
# MEAN IMPROVEMENT PER CORRIDOR ACROSS HORIZONS
# ============================================================

mean_rank = (
    pivot
    .sort_values(
        "mean_improvement",
        ascending=True,
    )
)


plt.figure(
    figsize=(
        10,
        14,
    )
)


plt.barh(
    mean_rank[
        "corridor"
    ],
    mean_rank[
        "mean_improvement"
    ],
)


plt.axvline(
    0,
    linewidth=1,
)


plt.xlabel(
    "Mean MAE improvement across horizons"
)


plt.ylabel(
    "Corridor"
)


plt.title(
    "Mean Adaptive Improvement Across t+1, t+3, t+6"
)


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "fig_corridor_mean_improvement.png",
    dpi=300,
)


plt.close()


# ============================================================
# FINAL
# ============================================================

print(
    "\n=== SAVED ==="
)

print(
    OUTPUT_DIR
    / "fixed_vs_adaptive_corridor_comparison.csv"
)

print(
    OUTPUT_DIR
    / "corridor_improvement_summary.csv"
)

print(
    OUTPUT_DIR
    / "corridor_cross_horizon_consistency.csv"
)

print(
    OUTPUT_DIR
    / "top_corridor_differences.csv"
)

print(
    "\nFigures:"
)

print(
    "- fig_corridor_mae_improvement_t1.png"
)

print(
    "- fig_corridor_mae_improvement_t3.png"
)

print(
    "- fig_corridor_mae_improvement_t6.png"
)

print(
    "- fig_corridor_win_counts.png"
)

print(
    "- fig_corridor_mean_improvement.png"
)