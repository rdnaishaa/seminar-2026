from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# CONFIG
# =========================================================
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
RESULT_DIR = ROOT / "outputs" / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 6]

SEED = 42
EPOCHS = 100
BATCH_SIZE = 32

np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================================================
# LOAD DATA
# =========================================================
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


print("\n=== ADAPTIVE GRAPH ST-GNN ===")
print(f"Train X : {X_train.shape}")
print(f"Val X   : {X_val.shape}")
print(f"Test X  : {X_test.shape}")


# =========================================================
# LOAD Adaptive ADJACENCY
# =========================================================
adj_df = pd.read_csv(
    RESULT_DIR / "fixed_graph_adjacency.csv",
    index_col=0
)

corridors = adj_df.index.tolist()

A = adj_df.values.astype(np.float32)


# =========================================================
# NORMALIZE ADJACENCY
#
# A_hat = D^(-1/2) A D^(-1/2)
# =========================================================
degree = np.sum(A, axis=1)

D_inv_sqrt = np.diag(
    1.0 / np.sqrt(degree)
)

A_norm = (
    D_inv_sqrt
    @ A
    @ D_inv_sqrt
).astype(np.float32)

A_tf = tf.constant(
    A_norm,
    dtype=tf.float32
)

print("\nAdjacency:")
print(f"Shape : {A.shape}")
print(f"Edges including self-loop : {int(A.sum())}")


# =========================================================
# MASKED LOSS
#
# Hanya target OBSERVASI ASLI yang dihitung.
# Target hasil imputasi tidak ikut menjadi label training.
# =========================================================
def masked_mse(y_true_with_mask, y_pred):

    y_true = y_true_with_mask[..., 0]
    mask = y_true_with_mask[..., 1]

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


def combine_y_mask(y, mask):

    return np.stack(
        [
            y,
            mask.astype(np.float32)
        ],
        axis=-1
    )


train_target = combine_y_mask(
    y_train,
    mask_train
)

val_target = combine_y_mask(
    y_val,
    mask_val
)


# =========================================================
# ADAPTIVE GRAPH CONVOLUTION
# Same symmetric normalization as Fixed Graph:
# A_norm = D^(-1/2) A D^(-1/2)
# =========================================================
class AdaptiveGraphConv(tf.keras.layers.Layer):

    def __init__(
        self,
        units,
        num_nodes,
        embedding_dim=8,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.units = units
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim

        self.projection = tf.keras.layers.Dense(
            units,
            activation="relu"
        )

    def build(self, input_shape):

        self.node_embeddings = self.add_weight(
            name="node_embeddings",
            shape=(self.num_nodes, self.embedding_dim),
            initializer="glorot_uniform",
            trainable=True
        )

        super().build(input_shape)

    def compute_adjacency(self):

        # Learned symmetric similarity
        scores = tf.matmul(
            self.node_embeddings,
            self.node_embeddings,
            transpose_b=True
        )

        scores = tf.nn.relu(scores)

        # Explicit self-loop
        adjacency = (
            scores
            + tf.eye(
                self.num_nodes,
                dtype=scores.dtype
            )
        )

        # Same symmetric normalization as training
        degree = tf.reduce_sum(
            adjacency,
            axis=1
        )

        d_inv_sqrt = tf.math.rsqrt(
            degree + 1e-8
        )

        adjacency_norm = (
            adjacency
            * d_inv_sqrt[:, None]
            * d_inv_sqrt[None, :]
        )

        return adjacency_norm

    def call(self, inputs):

        adjacency = self.compute_adjacency()

        graph_x = tf.einsum(
            "ij,btjf->btif",
            adjacency,
            inputs
        )

        # Same raw + graph path as training
        x = tf.concat(
            [inputs, graph_x],
            axis=-1
        )

        x = self.projection(x)

        return x

    def get_adjacency(self):

        return self.compute_adjacency()
        
# =========================================================
# MODEL
# =========================================================
inputs = tf.keras.Input(
    shape=(
        INPUT_LENGTH,
        NUM_NODES,
        1
    )
)

# Spatial modeling
x = AdaptiveGraphConv(
    units=16,
    num_nodes=NUM_NODES,
    embedding_dim=8,
    name="adaptive_graph_conv"
)(inputs)


# =========================================================
# TEMPORAL MODELING PER NODE
#
# (batch, time, node, feature)
# ->
# (batch, node, time, feature)
# =========================================================
x = tf.keras.layers.Permute(
    (2, 1, 3)
)(x)


# Merge batch & node temporarily
x = tf.keras.layers.Reshape(
    (
        NUM_NODES,
        INPUT_LENGTH * 16
    )
)(x)

# Each node gets temporal representation
x = tf.keras.layers.Reshape(
    (
        NUM_NODES,
        INPUT_LENGTH,
        16
    )
)(x)


# Apply GRU independently to each node
x = tf.keras.layers.TimeDistributed(
    tf.keras.layers.GRU(
        32
    )
)(x)


x = tf.keras.layers.Dropout(
    0.2
)(x)


# Multi-horizon output per node
x = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Dense(
        len(HORIZONS)
    )
)(x)


# batch × node × horizon
# ->
# batch × horizon × node
outputs = tf.keras.layers.Permute(
    (2, 1)
)(x)


model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs
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
        patience=10,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-5
    )
]


history = model.fit(
    X_train,
    train_target,
    validation_data=(
        X_val,
        val_target
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)


# =========================================================
# TEST PREDICTION
# =========================================================
pred_scaled = model.predict(
    X_test,
    verbose=0
)


# =========================================================
# INVERSE SCALE
# =========================================================
scaler_data = np.load(
    DATA_DIR / "stgnn_scaler.npz"
)

mean = float(
    scaler_data["mean"][0]
)

scale = float(
    scaler_data["scale"][0]
)


y_true = (
    y_test * scale
    + mean
)

y_pred = (
    pred_scaled * scale
    + mean
)


# =========================================================
# EVALUATION
# =========================================================
results = []
prediction_rows = []


for h_idx, horizon in enumerate(HORIZONS):

    mask = mask_test[
        :, h_idx, :
    ]

    true = y_true[
        :, h_idx, :
    ][mask]

    pred = y_pred[
        :, h_idx, :
    ][mask]

    mae = mean_absolute_error(
        true,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            true,
            pred
        )
    )

    print(
        f"\nAdaptive ST-GNN t+{horizon}h | "
        f"MAE = {mae:.3f} km/h | "
        f"RMSE = {rmse:.3f} km/h | "
        f"Points = {len(true)}"
    )

    results.append({
        "model": "Adaptive ST-GNN",
        "horizon": horizon,
        "MAE": mae,
        "RMSE": rmse,
        "evaluated_points": len(true)
    })


    for sample_idx in range(
        len(y_true)
    ):

        for node_idx, corridor in enumerate(
            corridors
        ):

            if not mask_test[
                sample_idx,
                h_idx,
                node_idx
            ]:
                continue

            prediction_rows.append({
                "horizon": horizon,
                "target_time": test_times[
                    sample_idx,
                    h_idx
                ],
                "corridor": corridor,
                "y_true": y_true[
                    sample_idx,
                    h_idx,
                    node_idx
                ],
                "adaptive_stgnn_pred": y_pred[
                    sample_idx,
                    h_idx,
                    node_idx
                ]
            })


# =========================================================
# SAVE
# =========================================================
pd.DataFrame(
    results
).to_csv(
    RESULT_DIR /
    "adaptive_stgnn_results.csv",
    index=False
)


pd.DataFrame(
    prediction_rows
).to_csv(
    RESULT_DIR /
    "adaptive_stgnn_predictions.csv",
    index=False
)


pd.DataFrame(
    history.history
).to_csv(
    RESULT_DIR /
    "adaptive_stgnn_training_history.csv",
    index=False
)


model.save_weights(
    RESULT_DIR /
    "adaptive_stgnn.weights.h5"
)


print("\n=== SAVED ===")
print("outputs/results/adaptive_stgnn_results.csv")
print("outputs/results/adaptive_stgnn_predictions.csv")
print("outputs/results/adaptive_stgnn_training_history.csv")
print("outputs/results/adaptive_stgnn.weights.h5")
