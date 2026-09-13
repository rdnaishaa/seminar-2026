from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT
    / "data"
    / "final_64"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "final_64"
    / "baselines"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HORIZONS = [1, 3, 6]


# ============================================================
# LOAD DATA
# ============================================================

train = np.load(
    DATA_DIR / "stgnn_train.npz",
    allow_pickle=True
)

val = np.load(
    DATA_DIR / "stgnn_val.npz",
    allow_pickle=True
)

test = np.load(
    DATA_DIR / "stgnn_test.npz",
    allow_pickle=True
)

scaler_data = np.load(
    DATA_DIR / "stgnn_scaler.npz"
)

scaler_mean = float(
    scaler_data["mean"][0]
)

scaler_scale = float(
    scaler_data["scale"][0]
)

node_df = pd.read_csv(
    DATA_DIR / "final_64_corridors.csv"
)

corridors = (
    node_df
    .sort_values("node_index")
    ["corridor_file"]
    .tolist()
)

NUM_NODES = len(corridors)


print("\n=== FINAL BASELINES ===")
print(f"Nodes    : {NUM_NODES}")
print(f"Horizons : {HORIZONS}")


# ============================================================
# HELPER
# ============================================================

def inverse_scale(x):
    return (
        x * scaler_scale
        + scaler_mean
    )


def masked_metrics(
    y_true,
    y_pred,
    mask
):

    valid = mask.astype(bool)

    true_valid = (
        y_true[valid]
    )

    pred_valid = (
        y_pred[valid]
    )

    mae = (
        mean_absolute_error(
            true_valid,
            pred_valid
        )
    )

    rmse = np.sqrt(
        mean_squared_error(
            true_valid,
            pred_valid
        )
    )

    return (
        mae,
        rmse,
        len(true_valid)
    )


# ============================================================
# EXTRACT TEST
# ============================================================

X_test = test["X"]

y_test_scaled = test["y"]

mask_test = test["mask"].astype(bool)

target_times = test["target_times"]


# X shape:
# samples x 12 x nodes x 1

# y shape:
# samples x 3 x nodes


y_test = inverse_scale(
    y_test_scaled
)


# ============================================================
# BASELINE 1 — NAIVE PERSISTENCE
#
# Prediksi semua horizon = speed terakhir
# dari input 12 jam.
# ============================================================

last_input_scaled = (
    X_test[
        :,
        -1,
        :,
        0
    ]
)

last_input = inverse_scale(
    last_input_scaled
)

naive_pred = np.stack(
    [
        last_input,
        last_input,
        last_input,
    ],
    axis=1
)


# ============================================================
# BASELINE 2 — HISTORICAL AVERAGE
#
# Mean per node dihitung dari TRAIN ONLY.
# Tidak melihat validation/test.
# ============================================================

X_train = train["X"]

input_obs_train = (
    train[
        "input_observed_mask"
    ].astype(bool)
)

# unscale train input
X_train_raw = inverse_scale(
    X_train[
        ...,
        0
    ]
)

historical_mean = np.zeros(
    NUM_NODES,
    dtype=float
)

for node_idx in range(
    NUM_NODES
):

    values = (
        X_train_raw[
            :,
            :,
            node_idx
        ]
    )

    observed_mask = (
        input_obs_train[
            :,
            :,
            node_idx
        ]
    )

    observed_values = (
        values[
            observed_mask
        ]
    )

    if len(
        observed_values
    ) == 0:

        raise ValueError(
            f"Tidak ada observed train data "
            f"untuk node "
            f"{corridors[node_idx]}"
        )

    historical_mean[
        node_idx
    ] = (
        observed_values.mean()
    )


historical_pred = np.tile(
    historical_mean[
        np.newaxis,
        np.newaxis,
        :
    ],
    (
        len(X_test),
        len(HORIZONS),
        1
    )
)


# ============================================================
# EVALUATION
# ============================================================

models = {
    "Naive Persistence":
        naive_pred,

    "Historical Average":
        historical_pred,
}

result_rows = []
corridor_rows = []


for model_name, predictions in models.items():

    print(
        f"\n=== {model_name.upper()} ==="
    )

    for h_idx, horizon in enumerate(
        HORIZONS
    ):

        y_true_h = (
            y_test[
                :,
                h_idx,
                :
            ]
        )

        y_pred_h = (
            predictions[
                :,
                h_idx,
                :
            ]
        )

        mask_h = (
            mask_test[
                :,
                h_idx,
                :
            ]
        )

        mae, rmse, n_points = (
            masked_metrics(
                y_true_h,
                y_pred_h,
                mask_h
            )
        )

        print(
            f"t+{horizon}h | "
            f"MAE={mae:.4f} | "
            f"RMSE={rmse:.4f} | "
            f"N={n_points}"
        )

        result_rows.append({
            "model":
                model_name,

            "horizon":
                horizon,

            "mae":
                mae,

            "rmse":
                rmse,

            "observed_points":
                n_points,
        })


        # ====================================================
        # PER-CORRIDOR METRICS
        # ====================================================

        for node_idx, corridor in enumerate(
            corridors
        ):

            node_mask = (
                mask_h[
                    :,
                    node_idx
                ]
            )

            if (
                node_mask.sum()
                == 0
            ):
                continue

            true_node = (
                y_true_h[
                    node_mask,
                    node_idx
                ]
            )

            pred_node = (
                y_pred_h[
                    node_mask,
                    node_idx
                ]
            )

            node_mae = (
                mean_absolute_error(
                    true_node,
                    pred_node
                )
            )

            node_rmse = np.sqrt(
                mean_squared_error(
                    true_node,
                    pred_node
                )
            )

            corridor_rows.append({
                "model":
                    model_name,

                "corridor":
                    corridor,

                "horizon":
                    horizon,

                "mae":
                    node_mae,

                "rmse":
                    node_rmse,

                "observed_points":
                    int(
                        node_mask.sum()
                    ),
            })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    result_rows
)

corridor_df = pd.DataFrame(
    corridor_rows
)


result_file = (
    OUTPUT_DIR
    / "baseline_results.csv"
)

corridor_file = (
    OUTPUT_DIR
    / "baseline_corridor_results.csv"
)

mean_file = (
    OUTPUT_DIR
    / "historical_mean_per_corridor.csv"
)


results_df.to_csv(
    result_file,
    index=False
)

corridor_df.to_csv(
    corridor_file,
    index=False
)

pd.DataFrame({
    "corridor":
        corridors,

    "historical_mean_speed":
        historical_mean,
}).to_csv(
    mean_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n=== BASELINE SUMMARY ==="
)

print(
    results_df.to_string(
        index=False,
        formatters={
            "mae":
                lambda x: f"{x:.4f}",
            "rmse":
                lambda x: f"{x:.4f}",
        }
    )
)


print(
    "\n=== SAVED ==="
)

print(result_file)
print(corridor_file)
print(mean_file)
