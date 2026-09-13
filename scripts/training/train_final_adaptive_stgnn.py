from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "final_64"
OUTPUT_DIR = ROOT / "outputs" / "final_64" / "adaptive_stgnn"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 6]
SEEDS = [42, 123, 456, 789, 2026]

GRAPH_UNITS = 16
EMBEDDING_DIM = 8
GRU_UNITS = 32
DROPOUT = 0.2

BATCH_SIZE = 32
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 10
LR_PATIENCE = 5

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

train = np.load(DATA_DIR / "stgnn_train.npz", allow_pickle=True)
val = np.load(DATA_DIR / "stgnn_val.npz", allow_pickle=True)
test = np.load(DATA_DIR / "stgnn_test.npz", allow_pickle=True)

X_train = train["X"].astype(np.float32)
y_train = train["y"].astype(np.float32)
mask_train = train["mask"].astype(bool)

X_val = val["X"].astype(np.float32)
y_val = val["y"].astype(np.float32)
mask_val = val["mask"].astype(bool)

X_test = test["X"].astype(np.float32)
y_test = test["y"].astype(np.float32)
mask_test = test["mask"].astype(bool)
test_times = test["target_times"]

NUM_NODES = X_train.shape[2]
INPUT_LENGTH = X_train.shape[1]

scaler = np.load(DATA_DIR / "stgnn_scaler.npz")
mean = float(np.asarray(scaler["mean"]).reshape(-1)[0])

if "std" in scaler.files:
    scale = float(np.asarray(scaler["std"]).reshape(-1)[0])
elif "scale" in scaler.files:
    scale = float(np.asarray(scaler["scale"]).reshape(-1)[0])
else:
    raise KeyError(
        "Scaler tidak memiliki key 'std' atau 'scale'. "
        f"Available keys: {scaler.files}"
    )

node_df = (
    pd.read_csv(DATA_DIR / "final_64_corridors.csv")
    .sort_values("node_index")
    .reset_index(drop=True)
)

corridors = node_df["corridor_file"].astype(str).tolist()

assert NUM_NODES == 64
assert len(corridors) == NUM_NODES

print("\n" + "=" * 60)
print("FINAL ADAPTIVE ST-GNN")
print("=" * 60)
print(f"Train : {X_train.shape}")
print(f"Val   : {X_val.shape}")
print(f"Test  : {X_test.shape}")
print(f"Nodes : {NUM_NODES}")
print(f"Input : {INPUT_LENGTH} hours")
print(f"Horizons : {HORIZONS}")
print(f"Seeds : {SEEDS}")
print(f"Scaler mean : {mean:.4f}")
print(f"Scaler std  : {scale:.4f}")
print("\n=== ADAPTIVE GRAPH ===")
print("Graph structure : learned end-to-end")
print(f"Embedding dim   : {EMBEDDING_DIM}")

def combine_y_mask(y, mask):
    return np.stack([y, mask.astype(np.float32)], axis=-1)

def masked_mse(y_true_with_mask, y_pred):
    y_true = y_true_with_mask[..., 0]
    mask = y_true_with_mask[..., 1]
    squared_error = tf.square(y_true - y_pred)
    masked_error = squared_error * mask
    return tf.reduce_sum(masked_error) / (tf.reduce_sum(mask) + 1e-8)

train_target = combine_y_mask(y_train, mask_train)
val_target = combine_y_mask(y_val, mask_val)

class AdaptiveGraphConv(tf.keras.layers.Layer):

    def __init__(self, units, num_nodes, embedding_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.projection = tf.keras.layers.Dense(units, activation="relu")

    def build(self, input_shape):
        self.node_embeddings = self.add_weight(
            name="node_embeddings",
            shape=(self.num_nodes, self.embedding_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def compute_raw_adjacency(self):
        scores = tf.matmul(
            self.node_embeddings,
            self.node_embeddings,
            transpose_b=True,
        )
        scores = tf.nn.relu(scores)
        return scores + tf.eye(self.num_nodes, dtype=scores.dtype)

    def compute_adjacency(self):
        adjacency = self.compute_raw_adjacency()
        degree = tf.reduce_sum(adjacency, axis=1)
        d_inv_sqrt = tf.math.rsqrt(degree + 1e-8)
        return (
            adjacency
            * d_inv_sqrt[:, None]
            * d_inv_sqrt[None, :]
        )

    def call(self, inputs):
        adjacency = self.compute_adjacency()
        graph_x = tf.einsum(
            "ij,btjf->btif",
            adjacency,
            inputs,
        )
        x = tf.concat([inputs, graph_x], axis=-1)
        return self.projection(x)

    def get_adjacency(self):
        return self.compute_adjacency()

    def get_raw_adjacency(self):
        return self.compute_raw_adjacency()

    def get_embeddings(self):
        return self.node_embeddings

def build_model():
    inputs = tf.keras.Input(
        shape=(INPUT_LENGTH, NUM_NODES, 1),
        name="traffic_input",
    )

    x = AdaptiveGraphConv(
        units=GRAPH_UNITS,
        num_nodes=NUM_NODES,
        embedding_dim=EMBEDDING_DIM,
        name="adaptive_graph_conv",
    )(inputs)

    x = tf.keras.layers.Permute((2, 1, 3), name="node_first")(x)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.GRU(GRU_UNITS),
        name="node_gru",
    )(x)

    x = tf.keras.layers.Dropout(DROPOUT, name="dropout")(x)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(len(HORIZONS)),
        name="horizon_output",
    )(x)

    outputs = tf.keras.layers.Permute((2, 1), name="forecast_output")(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Adaptive_STGNN",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss=masked_mse,
    )
    return model

all_results = []
all_corridor_results = []
all_prediction_rows = []

for seed in SEEDS:
    print("\n" + "=" * 60)
    print(f"SEED {seed}")
    print("=" * 60)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    model = build_model()

    if seed == SEEDS[0]:
        print("\n=== MODEL SUMMARY ===")
        model.summary()

    weight_file = OUTPUT_DIR / f"adaptive_stgnn_seed{seed}.weights.h5"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=LR_PATIENCE,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(weight_file),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train,
        train_target,
        validation_data=(X_val, val_target),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    model.load_weights(weight_file)

    adaptive_layer = model.get_layer("adaptive_graph_conv")
    learned_adjacency = adaptive_layer.get_adjacency().numpy()
    learned_raw_adjacency = adaptive_layer.get_raw_adjacency().numpy()
    learned_embeddings = adaptive_layer.get_embeddings().numpy()

    np.save(
        OUTPUT_DIR / f"adaptive_stgnn_seed{seed}_adjacency.npy",
        learned_adjacency,
    )
    pd.DataFrame(
        learned_adjacency,
        index=corridors,
        columns=corridors,
    ).to_csv(
        OUTPUT_DIR / f"adaptive_stgnn_seed{seed}_adjacency.csv"
    )

    np.save(
        OUTPUT_DIR / f"adaptive_stgnn_seed{seed}_raw_adjacency.npy",
        learned_raw_adjacency,
    )
    pd.DataFrame(
        learned_raw_adjacency,
        index=corridors,
        columns=corridors,
    ).to_csv(
        OUTPUT_DIR / f"adaptive_stgnn_seed{seed}_raw_adjacency.csv"
    )

    embedding_columns = [
        f"embedding_{i}"
        for i in range(EMBEDDING_DIM)
    ]
    embeddings_df = pd.DataFrame(
        learned_embeddings,
        columns=embedding_columns,
    )
    embeddings_df.insert(0, "corridor", corridors)
    embeddings_df.to_csv(
        OUTPUT_DIR / f"adaptive_stgnn_seed{seed}_embeddings.csv",
        index=False,
    )

    history_df = pd.DataFrame(history.history)
    history_df.insert(
        0,
        "epoch",
        np.arange(1, len(history_df) + 1),
    )
    history_df.to_csv(
        OUTPUT_DIR / f"adaptive_stgnn_seed{seed}_training_history.csv",
        index=False,
    )

    print(f"\nTraining epochs seed {seed}: {len(history_df)}")

    pred_scaled = model.predict(X_test, verbose=0)
    y_true = y_test * scale + mean
    y_pred = pred_scaled * scale + mean

    seed_results = []
    print(f"\n=== TEST RESULTS SEED {seed} ===")

    for h_idx, horizon in enumerate(HORIZONS):
        mask_h = mask_test[:, h_idx, :]
        true_h = y_true[:, h_idx, :]
        pred_h = y_pred[:, h_idx, :]

        true_valid = true_h[mask_h]
        pred_valid = pred_h[mask_h]

        mae = mean_absolute_error(true_valid, pred_valid)
        rmse = np.sqrt(mean_squared_error(true_valid, pred_valid))
        n_points = len(true_valid)

        print(
            f"t+{horizon}h | "
            f"MAE={mae:.4f} | "
            f"RMSE={rmse:.4f} | "
            f"N={n_points}"
        )

        row = {
            "model": "Adaptive ST-GNN",
            "seed": seed,
            "horizon": horizon,
            "MAE": mae,
            "RMSE": rmse,
            "evaluated_points": n_points,
        }
        all_results.append(row)
        seed_results.append(row)

        for node_idx, corridor in enumerate(corridors):
            corridor_mask = mask_test[:, h_idx, node_idx]
            corridor_true = y_true[:, h_idx, node_idx][corridor_mask]
            corridor_pred = y_pred[:, h_idx, node_idx][corridor_mask]

            if len(corridor_true) == 0:
                continue

            corridor_mae = mean_absolute_error(
                corridor_true,
                corridor_pred,
            )
            corridor_rmse = np.sqrt(
                mean_squared_error(
                    corridor_true,
                    corridor_pred,
                )
            )

            all_corridor_results.append({
                "model": "Adaptive ST-GNN",
                "seed": seed,
                "horizon": horizon,
                "node_index": node_idx,
                "corridor": corridor,
                "MAE": corridor_mae,
                "RMSE": corridor_rmse,
                "evaluated_points": len(corridor_true),
            })

        for sample_idx in range(len(y_true)):
            for node_idx, corridor in enumerate(corridors):
                if not mask_test[
                    sample_idx,
                    h_idx,
                    node_idx
                ]:
                    continue

                all_prediction_rows.append({
                    "model": "Adaptive ST-GNN",
                    "seed": seed,
                    "horizon": horizon,
                    "sample_index": sample_idx,
                    "target_time": test_times[
                        sample_idx,
                        h_idx
                    ],
                    "node_index": node_idx,
                    "corridor": corridor,
                    "y_true": float(
                        y_true[
                            sample_idx,
                            h_idx,
                            node_idx
                        ]
                    ),
                    "y_pred": float(
                        y_pred[
                            sample_idx,
                            h_idx,
                            node_idx
                        ]
                    ),
                })

    pd.DataFrame(seed_results).to_csv(
        OUTPUT_DIR / f"adaptive_stgnn_seed{seed}_results.csv",
        index=False,
    )

results_df = pd.DataFrame(all_results)
corridor_df = pd.DataFrame(all_corridor_results)
prediction_df = pd.DataFrame(all_prediction_rows)

results_df.to_csv(
    OUTPUT_DIR / "adaptive_stgnn_multiseed_results.csv",
    index=False,
)
corridor_df.to_csv(
    OUTPUT_DIR / "adaptive_stgnn_corridor_results.csv",
    index=False,
)
prediction_df.to_csv(
    OUTPUT_DIR / "adaptive_stgnn_predictions.csv",
    index=False,
)

summary_rows = []

print("\n" + "=" * 60)
print("FINAL ADAPTIVE ST-GNN MULTI-SEED SUMMARY")
print("=" * 60)

for horizon in HORIZONS:
    h_df = results_df[results_df["horizon"] == horizon]

    mae_mean = h_df["MAE"].mean()
    mae_std = h_df["MAE"].std(ddof=0)
    rmse_mean = h_df["RMSE"].mean()
    rmse_std = h_df["RMSE"].std(ddof=0)
    n_points = int(h_df["evaluated_points"].iloc[0])

    summary_rows.append({
        "model": "Adaptive ST-GNN",
        "horizon": horizon,
        "MAE_mean": mae_mean,
        "MAE_std": mae_std,
        "RMSE_mean": rmse_mean,
        "RMSE_std": rmse_std,
        "num_seeds": len(h_df),
        "evaluated_points": n_points,
    })

    print(
        f"t+{horizon}h | "
        f"MAE {mae_mean:.4f} ± {mae_std:.4f} | "
        f"RMSE {rmse_mean:.4f} ± {rmse_std:.4f}"
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    OUTPUT_DIR / "adaptive_stgnn_multiseed_summary.csv",
    index=False,
)

config_df = pd.DataFrame([{
    "num_nodes": NUM_NODES,
    "input_length": INPUT_LENGTH,
    "horizons": str(HORIZONS),
    "graph_type": "adaptive",
    "graph_units": GRAPH_UNITS,
    "embedding_dim": EMBEDDING_DIM,
    "gru_units": GRU_UNITS,
    "dropout": DROPOUT,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "seeds": str(SEEDS),
}])

config_df.to_csv(
    OUTPUT_DIR / "adaptive_stgnn_config.csv",
    index=False,
)

print("\n=== SAVED ===")
print(OUTPUT_DIR / "adaptive_stgnn_multiseed_results.csv")
print(OUTPUT_DIR / "adaptive_stgnn_multiseed_summary.csv")
print(OUTPUT_DIR / "adaptive_stgnn_corridor_results.csv")
print(OUTPUT_DIR / "adaptive_stgnn_predictions.csv")
print("\nLearned adjacency + embeddings disimpan untuk setiap seed.")
