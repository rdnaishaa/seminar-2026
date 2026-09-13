from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import ROOT
from src.agent.data_agent import load_live_data, check_live_data_status


DATA_DIR = ROOT / "data" / "final_64"
OUTPUT_DIR = ROOT / "outputs" / "final_64" / "adaptive_stgnn"

WEIGHTS_PATH = OUTPUT_DIR / "adaptive_stgnn_seed42.weights.h5"

HORIZONS = [1, 3, 6]

NUM_NODES = 64
INPUT_LENGTH = 12

GRAPH_UNITS = 16
EMBEDDING_DIM = 8
GRU_UNITS = 32
DROPOUT = 0.2


class AdaptiveGraphConv(tf.keras.layers.Layer):

    def __init__(
        self,
        units,
        num_nodes,
        embedding_dim=8,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.units = units
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim

        self.projection = tf.keras.layers.Dense(
            units,
            activation="relu",
        )

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

        return scores + tf.eye(
            self.num_nodes,
            dtype=scores.dtype,
        )

    def compute_adjacency(self):

        adjacency = self.compute_raw_adjacency()

        degree = tf.reduce_sum(
            adjacency,
            axis=1,
        )

        d_inv_sqrt = tf.math.rsqrt(
            degree + 1e-8
        )

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

        x = tf.concat(
            [inputs, graph_x],
            axis=-1,
        )

        return self.projection(x)


def build_model():

    inputs = tf.keras.Input(
        shape=(
            INPUT_LENGTH,
            NUM_NODES,
            1,
        ),
        name="traffic_input",
    )

    x = AdaptiveGraphConv(
        units=GRAPH_UNITS,
        num_nodes=NUM_NODES,
        embedding_dim=EMBEDDING_DIM,
        name="adaptive_graph_conv",
    )(inputs)

    x = tf.keras.layers.Permute(
        (2, 1, 3),
        name="node_first",
    )(x)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.GRU(
            GRU_UNITS
        ),
        name="node_gru",
    )(x)

    x = tf.keras.layers.Dropout(
        DROPOUT,
        name="dropout",
    )(x)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            len(HORIZONS)
        ),
        name="horizon_output",
    )(x)

    outputs = tf.keras.layers.Permute(
        (2, 1),
        name="forecast_output",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Adaptive_STGNN",
    )

    return model


def load_scaler():

    scaler_path = DATA_DIR / "stgnn_scaler.npz"

    scaler = np.load(
        scaler_path
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
            "Scaler tidak memiliki key std/scale."
        )

    return mean, scale


def load_node_order():

    node_df = pd.read_csv(
        DATA_DIR / "final_64_corridors.csv"
    )

    node_df = (
        node_df
        .sort_values("node_index")
        .reset_index(drop=True)
    )

    corridor_files = (
        node_df["corridor_file"]
        .astype(str)
        .tolist()
    )

    station_ids = [
        name.replace(".csv", "")
        for name in corridor_files
    ]

    if len(station_ids) != NUM_NODES:
        raise ValueError(
            f"Node count salah: "
            f"{len(station_ids)} != {NUM_NODES}"
        )

    return station_ids


def prepare_model_input():

    status = check_live_data_status()

    if status["status"] != "READY":
        raise RuntimeError(
            f"Live data belum siap: "
            f"{status['message']}"
        )

    df = load_live_data()

    station_ids = load_node_order()

    timestamps = (
        df["obs_time_utc"]
        .drop_duplicates()
        .sort_values()
        .tail(INPUT_LENGTH)
        .tolist()
    )

    if len(timestamps) != INPUT_LENGTH:
        raise RuntimeError(
            f"Input hanya memiliki "
            f"{len(timestamps)} timestep."
        )

    matrix = (
        df[
            df["obs_time_utc"]
            .isin(timestamps)
        ]
        .pivot(
            index="obs_time_utc",
            columns="station_id",
            values="current_speed",
        )
        .reindex(
            index=timestamps,
            columns=station_ids,
        )
    )

    if matrix.isna().any().any():
        missing_count = int(
            matrix.isna().sum().sum()
        )

        raise RuntimeError(
            f"Model input masih memiliki "
            f"{missing_count} missing value."
        )

    mean, scale = load_scaler()

    values = matrix.to_numpy(
        dtype=np.float32
    )

    values_scaled = (
        values - mean
    ) / scale

    X = values_scaled[
        None,
        :,
        :,
        None,
    ]

    return X, station_ids, timestamps, mean, scale


def run_forecast():

    X, station_ids, timestamps, mean, scale = (
        prepare_model_input()
    )

    model = build_model()

    model.load_weights(
        WEIGHTS_PATH
    )

    prediction_scaled = model.predict(
        X,
        verbose=0,
    )

    prediction = (
        prediction_scaled * scale
        + mean
    )

    prediction = prediction[0]

    latest_timestamp = timestamps[-1]

    rows = []

    for horizon_idx, horizon in enumerate(
        HORIZONS
    ):

        target_time = (
            latest_timestamp
            + pd.Timedelta(
                hours=horizon
            )
        )

        for node_idx, station_id in enumerate(
            station_ids
        ):

            rows.append(
                {
                    "station_id": station_id,
                    "horizon": horizon,
                    "target_time_utc": target_time,
                    "predicted_speed": float(
                        prediction[
                            horizon_idx,
                            node_idx,
                        ]
                    ),
                }
            )

    result = pd.DataFrame(rows)

    return result