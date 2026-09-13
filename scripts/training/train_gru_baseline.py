from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# CONFIG
# =========================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULT_DIR = ROOT / "outputs" / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "forecasting_train_preprocessed.csv"
VAL_PATH = DATA_DIR / "forecasting_val_preprocessed.csv"
TEST_PATH = DATA_DIR / "forecasting_test_preprocessed.csv"

INPUT_LENGTH = 12
HORIZONS = [1, 3, 6]
TARGET = "current_speed"

EPOCHS = 50
BATCH_SIZE = 32
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================================================
# LOAD DATA
# =========================================================
def load_split(path):

    df = pd.read_csv(path)

    df["obs_time_utc"] = pd.to_datetime(
        df["obs_time_utc"],
        utc=True
    )

    if df["is_observed"].dtype == object:
        df["is_observed"] = (
            df["is_observed"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False})
        )

    return df


train_df = load_split(TRAIN_PATH)
val_df = load_split(VAL_PATH)
test_df = load_split(TEST_PATH)

corridors = sorted(
    train_df["corridor_file"].unique()
)

NUM_NODES = len(corridors)

print("\n=== DATASET ===")
print(f"Corridors     : {NUM_NODES}")
print(f"Input length  : {INPUT_LENGTH} hours")
print(f"Horizons      : {HORIZONS}")


# =========================================================
# BUILD TEMPORAL MATRIX
# =========================================================
def build_matrix(df):

    df = df.dropna(
        subset=["segment_id"]
    ).copy()

    speed_matrix = df.pivot(
        index="obs_time_utc",
        columns="corridor_file",
        values=TARGET
    )

    observed_matrix = df.pivot(
        index="obs_time_utc",
        columns="corridor_file",
        values="is_observed"
    )

    segment_series = (
        df[
            ["obs_time_utc", "segment_id"]
        ]
        .drop_duplicates()
        .set_index("obs_time_utc")
        ["segment_id"]
        .sort_index()
    )

    speed_matrix = (
        speed_matrix
        .reindex(columns=corridors)
        .sort_index()
    )

    observed_matrix = (
        observed_matrix
        .reindex(columns=corridors)
        .sort_index()
    )

    return (
        speed_matrix,
        observed_matrix,
        segment_series
    )


train_speed, train_obs, train_segments = build_matrix(train_df)
val_speed, val_obs, val_segments = build_matrix(val_df)
test_speed, test_obs, test_segments = build_matrix(test_df)


# =========================================================
# SCALER
# FIT HANYA PADA TRAIN
# =========================================================
scaler = StandardScaler()

train_values = train_speed.values

scaler.fit(
    train_values.reshape(-1, 1)
)


def transform_matrix(matrix):

    values = matrix.values

    scaled = scaler.transform(
        values.reshape(-1, 1)
    ).reshape(values.shape)

    return scaled


train_scaled = transform_matrix(train_speed)
val_scaled = transform_matrix(val_speed)
test_scaled = transform_matrix(test_speed)


# =========================================================
# BUILD MULTI-HORIZON WINDOWS
# =========================================================
def create_windows(
    scaled_values,
    raw_values,
    observed_values,
    timestamps,
    segments
):

    X = []
    y = []
    y_mask = []
    target_times = []

    max_horizon = max(HORIZONS)

    n = len(timestamps)

    for start_idx in range(
        n - INPUT_LENGTH - max_horizon + 1
    ):

        input_end_idx = (
            start_idx + INPUT_LENGTH - 1
        )

        target_indices = [
            input_end_idx + h
            for h in HORIZONS
        ]

        all_indices = (
            list(
                range(
                    start_idx,
                    input_end_idx + 1
                )
            )
            + target_indices
        )

        window_segments = segments[
            all_indices
        ]

        # Jangan menyeberangi segment
        if len(set(window_segments)) != 1:
            continue

        input_window = scaled_values[
            start_idx:
            input_end_idx + 1
        ]

        targets = []
        masks = []
        times = []

        for target_idx in target_indices:

            targets.append(
                scaled_values[
                    target_idx
                ]
            )

            masks.append(
                observed_values[
                    target_idx
                ].astype(float)
            )

            times.append(
                timestamps[
                    target_idx
                ]
            )

        X.append(input_window)
        y.append(targets)
        y_mask.append(masks)
        target_times.append(times)

    return (
        np.array(X),
        np.array(y),
        np.array(y_mask),
        np.array(target_times)
    )


def prepare_windows(
    speed_matrix,
    obs_matrix,
    segment_series,
    scaled_matrix
):

    timestamps = speed_matrix.index.to_numpy()

    segments = (
        segment_series
        .reindex(speed_matrix.index)
        .to_numpy()
    )

    return create_windows(
        scaled_values=scaled_matrix,
        raw_values=speed_matrix.values,
        observed_values=obs_matrix.values,
        timestamps=timestamps,
        segments=segments
    )


X_train, y_train, mask_train, time_train = prepare_windows(
    train_speed,
    train_obs,
    train_segments,
    train_scaled
)

X_val, y_val, mask_val, time_val = prepare_windows(
    val_speed,
    val_obs,
    val_segments,
    val_scaled
)

X_test, y_test, mask_test, time_test = prepare_windows(
    test_speed,
    test_obs,
    test_segments,
    test_scaled
)


print("\n=== WINDOWS ===")
print(f"Train : {X_train.shape}")
print(f"Val   : {X_val.shape}")
print(f"Test  : {X_test.shape}")


# =========================================================
# RESHAPE OUTPUT
# =========================================================
# y awal:
# (samples, horizons, nodes)
#
# output model:
# (samples, horizons * nodes)

y_train_flat = y_train.reshape(
    y_train.shape[0],
    -1
)

y_val_flat = y_val.reshape(
    y_val.shape[0],
    -1
)


# =========================================================
# MODEL
# =========================================================
model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(INPUT_LENGTH, NUM_NODES)
    ),

    tf.keras.layers.GRU(
        64,
        return_sequences=False
    ),

    tf.keras.layers.Dropout(
        0.2
    ),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        len(HORIZONS) * NUM_NODES
    )
])

# =========================================================
# MASKED LOSS
# =========================================================

def masked_mse(y_true_with_mask, y_pred):

    num_outputs = len(HORIZONS) * NUM_NODES

    y_true = y_true_with_mask[
        :, :num_outputs
    ]

    mask = y_true_with_mask[
        :, num_outputs:
    ]

    squared_error = tf.square(
        y_true - y_pred
    )

    masked_error = (
        squared_error * mask
    )

    return (
        tf.reduce_sum(masked_error)
        /
        (tf.reduce_sum(mask) + 1e-8)
    )


mask_train_flat = mask_train.reshape(
    -1,
    len(HORIZONS) * NUM_NODES
).astype(np.float32)

mask_val_flat = mask_val.reshape(
    -1,
    len(HORIZONS) * NUM_NODES
).astype(np.float32)


train_target_masked = np.concatenate(
    [
        y_train_flat.astype(np.float32),
        mask_train_flat
    ],
    axis=1
)

val_target_masked = np.concatenate(
    [
        y_val_flat.astype(np.float32),
        mask_val_flat
    ],
    axis=1
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss=masked_mse
)

model.summary()


# =========================================================
# TRAIN
# =========================================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )
]

history = model.fit(
    X_train,
    train_target_masked,
    validation_data=(
        X_val,
        val_target_masked
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)


# =========================================================
# PREDICT TEST
# =========================================================
pred_scaled = model.predict(
    X_test,
    verbose=0
)

pred_scaled = pred_scaled.reshape(
    -1,
    len(HORIZONS),
    NUM_NODES
)


# =========================================================
# INVERSE TRANSFORM
# =========================================================
def inverse_transform(values):

    shape = values.shape

    flat = values.reshape(-1, 1)

    inverse = scaler.inverse_transform(
        flat
    )

    return inverse.reshape(shape)


pred = inverse_transform(
    pred_scaled
)

y_true = inverse_transform(
    y_test
)


# =========================================================
# EVALUATION
# hanya target observasi asli
# =========================================================
results = []
prediction_rows = []

for h_idx, horizon in enumerate(HORIZONS):

    horizon_mask = (
        mask_test[:, h_idx, :]
        == 1
    )

    true_values = (
        y_true[:, h_idx, :]
        [horizon_mask]
    )

    pred_values = (
        pred[:, h_idx, :]
        [horizon_mask]
    )

    mae = mean_absolute_error(
        true_values,
        pred_values
    )

    rmse = np.sqrt(
        mean_squared_error(
            true_values,
            pred_values
        )
    )

    results.append({
        "model": "GRU",
        "horizon": horizon,
        "MAE": mae,
        "RMSE": rmse,
        "evaluated_points": len(true_values)
    })

    print(
        f"\nGRU t+{horizon}h | "
        f"MAE = {mae:.3f} km/h | "
        f"RMSE = {rmse:.3f} km/h | "
        f"Points = {len(true_values)}"
    )

    # detail prediction
    for sample_idx in range(
        y_true.shape[0]
    ):

        for node_idx, corridor in enumerate(
            corridors
        ):

            if not horizon_mask[
                sample_idx,
                node_idx
            ]:
                continue

            prediction_rows.append({
                "horizon": horizon,
                "target_time": time_test[
                    sample_idx,
                    h_idx
                ],
                "corridor": corridor,
                "y_true": y_true[
                    sample_idx,
                    h_idx,
                    node_idx
                ],
                "gru_pred": pred[
                    sample_idx,
                    h_idx,
                    node_idx
                ]
            })


# =========================================================
# SAVE RESULTS
# =========================================================
results_df = pd.DataFrame(results)

predictions_df = pd.DataFrame(
    prediction_rows
)

results_df.to_csv(
    RESULT_DIR / "gru_results.csv",
    index=False
)

predictions_df.to_csv(
    RESULT_DIR / "gru_predictions.csv",
    index=False
)

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    RESULT_DIR / "gru_training_history.csv",
    index=False
)

model.save(
    RESULT_DIR / "gru_baseline.keras"
)


print("\n=== SAVED ===")
print("outputs/results/gru_results.csv")
print("outputs/results/gru_predictions.csv")
print("outputs/results/gru_training_history.csv")
print("outputs/results/gru_baseline.keras")