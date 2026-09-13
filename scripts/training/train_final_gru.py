from pathlib import Path
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# PATHS
# ============================================================

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
    / "gru"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIG
# ============================================================

SEEDS = [
    42,
    123,
    456,
    789,
    2026,
]

HORIZONS = [
    1,
    3,
    6,
]

GRU_UNITS = 32
DROPOUT = 0.2

BATCH_SIZE = 32
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3

PATIENCE = 10


# ============================================================
# DETERMINISM
# ============================================================

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


def set_seed(seed):

    os.environ[
        "PYTHONHASHSEED"
    ] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    tf.keras.utils.set_random_seed(
        seed
    )


# ============================================================
# LOAD DATA
# ============================================================

train = np.load(
    DATA_DIR
    / "stgnn_train.npz",
    allow_pickle=True,
)

val = np.load(
    DATA_DIR
    / "stgnn_val.npz",
    allow_pickle=True,
)

test = np.load(
    DATA_DIR
    / "stgnn_test.npz",
    allow_pickle=True,
)

scaler_data = np.load(
    DATA_DIR
    / "stgnn_scaler.npz"
)


X_train = train["X"]
y_train = train["y"]
mask_train = (
    train["mask"]
    .astype(bool)
)

X_val = val["X"]
y_val = val["y"]
mask_val = (
    val["mask"]
    .astype(bool)
)

X_test = test["X"]
y_test = test["y"]
mask_test = (
    test["mask"]
    .astype(bool)
)

target_times_test = (
    test["target_times"]
)


SCALER_MEAN = float(
    scaler_data["mean"][0]
)

SCALER_SCALE = float(
    scaler_data["scale"][0]
)


NUM_NODES = (
    X_train.shape[2]
)

INPUT_LENGTH = (
    X_train.shape[1]
)


print(
    "\n=== FINAL GRU BASELINE ==="
)

print(
    f"Train : {X_train.shape}"
)

print(
    f"Val   : {X_val.shape}"
)

print(
    f"Test  : {X_test.shape}"
)

print(
    f"Nodes : {NUM_NODES}"
)

print(
    f"Seeds : {SEEDS}"
)


# ============================================================
# TARGET WITH NaN MASK
#
# NaN hanya dipakai untuk memberi tahu loss bahwa target
# tersebut bukan observasi asli dan tidak boleh ikut loss.
# ============================================================

y_train_masked = np.where(
    mask_train,
    y_train,
    np.nan,
).astype(
    np.float32
)

y_val_masked = np.where(
    mask_val,
    y_val,
    np.nan,
).astype(
    np.float32
)


# ============================================================
# MASKED MSE
# ============================================================

def masked_mse(
    y_true,
    y_pred,
):

    valid = tf.math.is_finite(
        y_true
    )

    safe_true = tf.where(
        valid,
        y_true,
        y_pred,
    )

    squared_error = tf.square(
        safe_true
        - y_pred
    )

    valid_float = tf.cast(
        valid,
        tf.float32,
    )

    numerator = tf.reduce_sum(
        squared_error
        * valid_float
    )

    denominator = tf.reduce_sum(
        valid_float
    )

    return (
        numerator
        / tf.maximum(
            denominator,
            1.0,
        )
    )


# ============================================================
# MODEL
#
# Tidak ada graph operation.
#
# Input:
# (batch, 12, 64, 1)
#
# Setelah permute:
# (batch, 64, 12, 1)
#
# GRU yang sama digunakan untuk setiap corridor.
# ============================================================

def build_gru_model():

    inputs = tf.keras.Input(
        shape=(
            INPUT_LENGTH,
            NUM_NODES,
            1,
        ),
        name="traffic_input",
    )

    # B, T, N, F
    # ->
    # B, N, T, F
    x = tf.keras.layers.Permute(
        (2, 1, 3),
        name="node_first",
    )(inputs)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.GRU(
            GRU_UNITS,
            activation="tanh",
        ),
        name="shared_gru",
    )(x)

    x = tf.keras.layers.Dropout(
        DROPOUT,
        name="dropout",
    )(x)

    # B, N, 3
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            len(HORIZONS)
        ),
        name="horizon_output",
    )(x)

    # B, 3, N
    outputs = tf.keras.layers.Permute(
        (2, 1),
        name="forecast",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="GRU_Baseline",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss=masked_mse,
    )

    return model


# ============================================================
# HELPERS
# ============================================================

def inverse_scale(x):

    return (
        x
        * SCALER_SCALE
        + SCALER_MEAN
    )


def calculate_metrics(
    y_true,
    y_pred,
    mask,
):

    true_valid = (
        y_true[
            mask
        ]
    )

    pred_valid = (
        y_pred[
            mask
        ]
    )

    mae = mean_absolute_error(
        true_valid,
        pred_valid,
    )

    rmse = np.sqrt(
        mean_squared_error(
            true_valid,
            pred_valid,
        )
    )

    return (
        float(mae),
        float(rmse),
        int(mask.sum()),
    )


# ============================================================
# TRAIN MULTI-SEED
# ============================================================

all_results = []

print(
    "\n=== TRAINING START ==="
)


for seed in SEEDS:

    print(
        "\n"
        + "=" * 60
    )

    print(
        f"SEED {seed}"
    )

    print(
        "=" * 60
    )

    tf.keras.backend.clear_session()

    set_seed(seed)

    model = build_gru_model()

    weight_file = (
        OUTPUT_DIR
        / f"gru_seed{seed}.weights.h5"
    )

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=weight_file,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
    ]


    history = model.fit(
        X_train,
        y_train_masked,

        validation_data=(
            X_val,
            y_val_masked,
        ),

        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,

        shuffle=True,

        callbacks=callbacks,

        verbose=1,
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_df = pd.DataFrame(
        history.history
    )

    history_df[
        "epoch"
    ] = (
        np.arange(
            1,
            len(history_df) + 1
        )
    )

    history_file = (
        OUTPUT_DIR
        / (
            f"gru_seed{seed}"
            f"_training_history.csv"
        )
    )

    history_df.to_csv(
        history_file,
        index=False,
    )


    # ========================================================
    # TEST PREDICTION
    # ========================================================

    pred_scaled = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    y_true_raw = inverse_scale(
        y_test
    )

    pred_raw = inverse_scale(
        pred_scaled
    )


    # ========================================================
    # METRICS PER HORIZON
    # ========================================================

    seed_rows = []

    print(
        f"\n=== TEST RESULTS SEED {seed} ==="
    )

    for h_idx, horizon in enumerate(
        HORIZONS
    ):

        true_h = (
            y_true_raw[
                :,
                h_idx,
                :
            ]
        )

        pred_h = (
            pred_raw[
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

        (
            mae,
            rmse,
            n_points,
        ) = calculate_metrics(
            true_h,
            pred_h,
            mask_h,
        )

        print(
            f"t+{horizon}h | "
            f"MAE={mae:.4f} | "
            f"RMSE={rmse:.4f} | "
            f"N={n_points}"
        )

        row = {
            "seed":
                seed,

            "horizon":
                horizon,

            "mae":
                mae,

            "rmse":
                rmse,

            "observed_points":
                n_points,

            "epochs":
                len(history_df),

            "best_val_loss":
                float(
                    history_df[
                        "val_loss"
                    ].min()
                ),
        }

        seed_rows.append(
            row
        )

        all_results.append(
            row
        )


    # ========================================================
    # SAVE SEED RESULTS
    # ========================================================

    pd.DataFrame(
        seed_rows
    ).to_csv(
        OUTPUT_DIR
        / f"gru_seed{seed}_results.csv",
        index=False,
    )


    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    np.savez_compressed(
        OUTPUT_DIR
        / f"gru_seed{seed}_predictions.npz",

        predictions=pred_raw,

        targets=y_true_raw,

        mask=mask_test,

        target_times=
            target_times_test,
    )


# ============================================================
# ALL-SEED RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_file = (
    OUTPUT_DIR
    / "gru_multiseed_results.csv"
)

results_df.to_csv(
    results_file,
    index=False,
)


# ============================================================
# SUMMARY MEAN ± STD
# ============================================================

summary = (
    results_df
    .groupby(
        "horizon"
    )
    .agg(
        mae_mean=(
            "mae",
            "mean",
        ),

        mae_std=(
            "mae",
            "std",
        ),

        rmse_mean=(
            "rmse",
            "mean",
        ),

        rmse_std=(
            "rmse",
            "std",
        ),

        epochs_mean=(
            "epochs",
            "mean",
        ),
    )
    .reset_index()
)


summary_file = (
    OUTPUT_DIR
    / "gru_multiseed_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False,
)


# ============================================================
# FINAL REPORT
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "FINAL GRU MULTI-SEED SUMMARY"
)

print(
    "=" * 60
)


for _, row in summary.iterrows():

    print(
        f"t+{int(row['horizon'])}h | "
        f"MAE "
        f"{row['mae_mean']:.4f} "
        f"± "
        f"{row['mae_std']:.4f} | "
        f"RMSE "
        f"{row['rmse_mean']:.4f} "
        f"± "
        f"{row['rmse_std']:.4f}"
    )


print(
    "\n=== SAVED ==="
)

print(
    results_file
)

print(
    summary_file
)