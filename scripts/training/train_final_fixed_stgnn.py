from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "final_64"

GRAPH_DIR = (
    ROOT
    / "outputs"
    / "final_64"
    / "fixed_graph"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "final_64"
    / "fixed_stgnn"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


HORIZONS = [1, 3, 6]

SEEDS = [
    42,
    123,
    456,
    789,
    2026,
]

GRAPH_UNITS = 16
GRU_UNITS = 32
DROPOUT = 0.2

BATCH_SIZE = 32
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3

EARLY_STOP_PATIENCE = 10
LR_PATIENCE = 5


# ============================================================
# DETERMINISM
# ============================================================

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


# ============================================================
# LOAD DATA
# ============================================================

train = np.load(
    DATA_DIR / "stgnn_train.npz",
    allow_pickle=True,
)

val = np.load(
    DATA_DIR / "stgnn_val.npz",
    allow_pickle=True,
)

test = np.load(
    DATA_DIR / "stgnn_test.npz",
    allow_pickle=True,
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


# ============================================================
# LOAD SCALER
# ============================================================

scaler = np.load(
    DATA_DIR / "stgnn_scaler.npz"
)


mean = float(
    np.asarray(
        scaler["mean"]
    ).reshape(-1)[0]
)


if "std" in scaler.files:

    scale = float(
        np.asarray(
            scaler["std"]
        ).reshape(-1)[0]
    )

elif "scale" in scaler.files:

    scale = float(
        np.asarray(
            scaler["scale"]
        ).reshape(-1)[0]
    )

else:

    raise KeyError(
        "Scaler tidak memiliki key 'std' atau 'scale'. "
        f"Available keys: {scaler.files}"
    )


# ============================================================
# LOAD FIXED GRAPH
# ============================================================

graph_data = np.load(
    GRAPH_DIR / "fixed_graph_64.npz"
)

A = graph_data[
    "adjacency_with_self"
].astype(np.float32)

A_norm = graph_data[
    "normalized_adjacency"
].astype(np.float32)


node_df = (
    pd.read_csv(
        GRAPH_DIR
        / "fixed_graph_64_nodes.csv"
    )
    .sort_values(
        "node_index"
    )
    .reset_index(
        drop=True
    )
)

corridors = (
    node_df[
        "corridor_file"
    ]
    .astype(str)
    .tolist()
)


# ============================================================
# VALIDATION
# ============================================================

assert NUM_NODES == 64

assert A.shape == (
    NUM_NODES,
    NUM_NODES,
)

assert A_norm.shape == (
    NUM_NODES,
    NUM_NODES,
)

assert len(corridors) == NUM_NODES

assert np.isfinite(
    A_norm
).all()


print(
    "\n"
    + "=" * 60
)

print(
    "FINAL FIXED ST-GNN"
)

print(
    "=" * 60
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
    f"Input : {INPUT_LENGTH} hours"
)

print(
    f"Horizons : {HORIZONS}"
)

print(
    f"Seeds : {SEEDS}"
)

print(
    f"Scaler mean : {mean:.4f}"
)

print(
    f"Scaler std  : {scale:.4f}"
)

print(
    "\n=== FIXED GRAPH ==="
)

print(
    f"Adjacency shape : {A.shape}"
)

print(
    f"Undirected edges : "
    f"{int((A.sum() - NUM_NODES) / 2)}"
)

print(
    f"Self-loops : {NUM_NODES}"
)


# ============================================================
# MASKED TARGET
# ============================================================

def combine_y_mask(
    y,
    mask,
):

    return np.stack(
        [
            y,
            mask.astype(
                np.float32
            ),
        ],
        axis=-1,
    )


train_target = combine_y_mask(
    y_train,
    mask_train,
)

val_target = combine_y_mask(
    y_val,
    mask_val,
)


# ============================================================
# MASKED LOSS
# ============================================================

def masked_mse(
    y_true_with_mask,
    y_pred,
):

    y_true = (
        y_true_with_mask[
            ...,
            0
        ]
    )

    mask = (
        y_true_with_mask[
            ...,
            1
        ]
    )

    squared_error = tf.square(
        y_true
        - y_pred
    )

    masked_error = (
        squared_error
        * mask
    )

    return (
        tf.reduce_sum(
            masked_error
        )
        /
        (
            tf.reduce_sum(
                mask
            )
            + 1e-8
        )
    )


# ============================================================
# FIXED GRAPH CONVOLUTION
# ============================================================

class FixedGraphConv(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        units,
        adjacency,
        **kwargs,
    ):

        super().__init__(
            **kwargs
        )

        self.units = units

        self.adjacency = (
            tf.constant(
                adjacency,
                dtype=tf.float32,
            )
        )

        self.projection = (
            tf.keras.layers.Dense(
                units,
                activation="relu",
            )
        )


    def call(
        self,
        inputs,
    ):

        # inputs:
        # batch × time × node × feature

        graph_x = tf.einsum(
            "ij,btjf->btif",
            self.adjacency,
            inputs,
        )

        # Preserve raw node information
        # + graph aggregated information
        x = tf.concat(
            [
                inputs,
                graph_x,
            ],
            axis=-1,
        )

        return self.projection(
            x
        )


# ============================================================
# MODEL BUILDER
# ============================================================

def build_model():

    inputs = tf.keras.Input(
        shape=(
            INPUT_LENGTH,
            NUM_NODES,
            1,
        ),
        name="traffic_input",
    )


    # --------------------------------------------------------
    # SPATIAL
    # --------------------------------------------------------

    x = FixedGraphConv(
        units=GRAPH_UNITS,
        adjacency=A_norm,
        name="fixed_graph_conv",
    )(
        inputs
    )


    # --------------------------------------------------------
    # TEMPORAL
    #
    # batch × time × node × feature
    # ->
    # batch × node × time × feature
    # --------------------------------------------------------

    x = tf.keras.layers.Permute(
        (
            2,
            1,
            3,
        ),
        name="node_first",
    )(
        x
    )


    # GRU yang sama digunakan ke setiap node
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.GRU(
            GRU_UNITS
        ),
        name="node_gru",
    )(
        x
    )


    x = tf.keras.layers.Dropout(
        DROPOUT,
        name="dropout",
    )(
        x
    )


    # --------------------------------------------------------
    # MULTI-HORIZON OUTPUT
    # --------------------------------------------------------

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            len(
                HORIZONS
            )
        ),
        name="horizon_output",
    )(
        x
    )


    # batch × node × horizon
    # ->
    # batch × horizon × node
    outputs = tf.keras.layers.Permute(
        (
            2,
            1,
        ),
        name="forecast_output",
    )(
        x
    )


    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Fixed_STGNN",
    )


    model.compile(
        optimizer=(
            tf.keras.optimizers.Adam(
                learning_rate=
                    LEARNING_RATE
            )
        ),
        loss=masked_mse,
    )


    return model


# ============================================================
# RESULT CONTAINERS
# ============================================================

all_results = []

all_corridor_results = []

all_prediction_rows = []


# ============================================================
# MULTI-SEED TRAINING
# ============================================================

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


    # --------------------------------------------------------
    # RESET TF STATE
    # --------------------------------------------------------

    tf.keras.backend.clear_session()

    tf.keras.utils.set_random_seed(
        seed
    )


    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    model = build_model()


    if seed == SEEDS[0]:

        print(
            "\n=== MODEL SUMMARY ==="
        )

        model.summary()


    # --------------------------------------------------------
    # OUTPUT FILES
    # --------------------------------------------------------

    weight_file = (
        OUTPUT_DIR
        / f"fixed_stgnn_seed{seed}.weights.h5"
    )


    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=
                EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=
                LR_PATIENCE,
            min_lr=1e-5,
            verbose=1,
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=
                str(
                    weight_file
                ),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
    ]


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    history = model.fit(
        X_train,
        train_target,

        validation_data=(
            X_val,
            val_target,
        ),

        epochs=
            MAX_EPOCHS,

        batch_size=
            BATCH_SIZE,

        callbacks=
            callbacks,

        verbose=1,

        shuffle=True,
    )


    # --------------------------------------------------------
    # LOAD BEST CHECKPOINT
    # --------------------------------------------------------

    model.load_weights(
        weight_file
    )


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history.history
    )

    history_df.insert(
        0,
        "epoch",
        np.arange(
            1,
            len(
                history_df
            )
            + 1
        ),
    )

    history_df.to_csv(
        OUTPUT_DIR
        / (
            f"fixed_stgnn_seed"
            f"{seed}_training_history.csv"
        ),
        index=False,
    )


    print(
        f"\nTraining epochs seed {seed}: "
        f"{len(history_df)}"
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    pred_scaled = model.predict(
        X_test,
        verbose=0,
    )


    y_true = (
        y_test
        * scale
        + mean
    )

    y_pred = (
        pred_scaled
        * scale
        + mean
    )


    # --------------------------------------------------------
    # EVALUATE EACH HORIZON
    # --------------------------------------------------------

    seed_results = []


    print(
        f"\n=== TEST RESULTS SEED {seed} ==="
    )


    for h_idx, horizon in enumerate(
        HORIZONS
    ):

        mask_h = (
            mask_test[
                :,
                h_idx,
                :
            ]
        )


        true_h = (
            y_true[
                :,
                h_idx,
                :
            ]
        )


        pred_h = (
            y_pred[
                :,
                h_idx,
                :
            ]
        )


        true_valid = (
            true_h[
                mask_h
            ]
        )


        pred_valid = (
            pred_h[
                mask_h
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


        n_points = len(
            true_valid
        )


        print(
            f"t+{horizon}h | "
            f"MAE={mae:.4f} | "
            f"RMSE={rmse:.4f} | "
            f"N={n_points}"
        )


        row = {
            "model":
                "Fixed ST-GNN",

            "seed":
                seed,

            "horizon":
                horizon,

            "MAE":
                mae,

            "RMSE":
                rmse,

            "evaluated_points":
                n_points,
        }


        all_results.append(
            row
        )

        seed_results.append(
            row
        )


        # ----------------------------------------------------
        # CORRIDOR LEVEL
        # ----------------------------------------------------

        for node_idx, corridor in enumerate(
            corridors
        ):

            corridor_mask = (
                mask_test[
                    :,
                    h_idx,
                    node_idx
                ]
            )


            corridor_true = (
                y_true[
                    :,
                    h_idx,
                    node_idx
                ][
                    corridor_mask
                ]
            )


            corridor_pred = (
                y_pred[
                    :,
                    h_idx,
                    node_idx
                ][
                    corridor_mask
                ]
            )


            if len(
                corridor_true
            ) == 0:

                continue


            corridor_mae = (
                mean_absolute_error(
                    corridor_true,
                    corridor_pred,
                )
            )


            corridor_rmse = np.sqrt(
                mean_squared_error(
                    corridor_true,
                    corridor_pred,
                )
            )


            all_corridor_results.append({
                "model":
                    "Fixed ST-GNN",

                "seed":
                    seed,

                "horizon":
                    horizon,

                "node_index":
                    node_idx,

                "corridor":
                    corridor,

                "MAE":
                    corridor_mae,

                "RMSE":
                    corridor_rmse,

                "evaluated_points":
                    len(
                        corridor_true
                    ),
            })


        # ----------------------------------------------------
        # SAVE POINT-LEVEL PREDICTIONS
        # ----------------------------------------------------

        for sample_idx in range(
            len(
                y_true
            )
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


                all_prediction_rows.append({

                    "model":
                        "Fixed ST-GNN",

                    "seed":
                        seed,

                    "horizon":
                        horizon,

                    "sample_index":
                        sample_idx,

                    "target_time":
                        test_times[
                            sample_idx,
                            h_idx
                        ],

                    "node_index":
                        node_idx,

                    "corridor":
                        corridor,

                    "y_true":
                        float(
                            y_true[
                                sample_idx,
                                h_idx,
                                node_idx
                            ]
                        ),

                    "y_pred":
                        float(
                            y_pred[
                                sample_idx,
                                h_idx,
                                node_idx
                            ]
                        ),
                })


    # --------------------------------------------------------
    # SAVE PER-SEED RESULT
    # --------------------------------------------------------

    pd.DataFrame(
        seed_results
    ).to_csv(
        OUTPUT_DIR
        / (
            f"fixed_stgnn_seed"
            f"{seed}_results.csv"
        ),
        index=False,
    )


# ============================================================
# SAVE ALL RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    OUTPUT_DIR
    / "fixed_stgnn_multiseed_results.csv",
    index=False,
)


corridor_df = pd.DataFrame(
    all_corridor_results
)

corridor_df.to_csv(
    OUTPUT_DIR
    / "fixed_stgnn_corridor_results.csv",
    index=False,
)


prediction_df = pd.DataFrame(
    all_prediction_rows
)

prediction_df.to_csv(
    OUTPUT_DIR
    / "fixed_stgnn_predictions.csv",
    index=False,
)


# ============================================================
# MULTI-SEED SUMMARY
# ============================================================

summary_rows = []


print(
    "\n"
    + "=" * 60
)

print(
    "FINAL FIXED ST-GNN MULTI-SEED SUMMARY"
)

print(
    "=" * 60
)


for horizon in HORIZONS:

    h_df = results_df[
        results_df[
            "horizon"
        ]
        == horizon
    ]


    mae_mean = (
        h_df[
            "MAE"
        ].mean()
    )

    mae_std = (
        h_df[
            "MAE"
        ].std(
            ddof=0
        )
    )


    rmse_mean = (
        h_df[
            "RMSE"
        ].mean()
    )

    rmse_std = (
        h_df[
            "RMSE"
        ].std(
            ddof=0
        )
    )


    n_points = int(
        h_df[
            "evaluated_points"
        ].iloc[0]
    )


    summary_rows.append({

        "model":
            "Fixed ST-GNN",

        "horizon":
            horizon,

        "MAE_mean":
            mae_mean,

        "MAE_std":
            mae_std,

        "RMSE_mean":
            rmse_mean,

        "RMSE_std":
            rmse_std,

        "num_seeds":
            len(
                h_df
            ),

        "evaluated_points":
            n_points,
    })


    print(
        f"t+{horizon}h | "
        f"MAE {mae_mean:.4f} "
        f"± {mae_std:.4f} | "
        f"RMSE {rmse_mean:.4f} "
        f"± {rmse_std:.4f}"
    )


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUTPUT_DIR
    / "fixed_stgnn_multiseed_summary.csv",
    index=False,
)


# ============================================================
# SAVE CONFIG
# ============================================================

config_df = pd.DataFrame(
    [
        {
            "num_nodes":
                NUM_NODES,

            "input_length":
                INPUT_LENGTH,

            "horizons":
                str(
                    HORIZONS
                ),

            "graph_k":
                4,

            "graph_units":
                GRAPH_UNITS,

            "gru_units":
                GRU_UNITS,

            "dropout":
                DROPOUT,

            "batch_size":
                BATCH_SIZE,

            "max_epochs":
                MAX_EPOCHS,

            "learning_rate":
                LEARNING_RATE,

            "seeds":
                str(
                    SEEDS
                ),
        }
    ]
)

config_df.to_csv(
    OUTPUT_DIR
    / "fixed_stgnn_config.csv",
    index=False,
)


print(
    "\n=== SAVED ==="
)

print(
    OUTPUT_DIR
    / "fixed_stgnn_multiseed_results.csv"
)

print(
    OUTPUT_DIR
    / "fixed_stgnn_multiseed_summary.csv"
)

print(
    OUTPUT_DIR
    / "fixed_stgnn_corridor_results.csv"
)

print(
    OUTPUT_DIR
    / "fixed_stgnn_predictions.csv"
)