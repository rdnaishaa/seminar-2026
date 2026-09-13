import pandas as pd

from src.config import ROOT


LIVE_DATA_PATH = ROOT / "data" / "collected" / "latest_12h_final64.csv"


def load_live_data():
    """
    Load data terbaru untuk kebutuhan forecasting.
    """

    if not LIVE_DATA_PATH.exists():
        return None

    df = pd.read_csv(LIVE_DATA_PATH)

    if "obs_time_utc" not in df.columns:
        return None

    df["obs_time_utc"] = pd.to_datetime(
        df["obs_time_utc"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(subset=["obs_time_utc"])

    return df


def check_live_data_status():
    """
    Memeriksa kesiapan data sebelum forecasting.
    """

    df = load_live_data()

    if df is None or df.empty:
        return {
            "status": "NOT_READY",
            "message": "File live data belum tersedia.",
            "latest_timestamp": None,
            "corridor_count": None,
            "timestamp_count": None,
            "observed_ratio": None,
            "imputed_ratio": None,
        }

    required_columns = [
        "obs_time_utc",
        "station_id",
        "current_speed",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        return {
            "status": "NOT_READY",
            "message": f"Kolom belum lengkap: {missing_columns}",
            "latest_timestamp": None,
            "corridor_count": None,
            "timestamp_count": None,
            "observed_ratio": None,
            "imputed_ratio": None,
        }

    latest_timestamp = df["obs_time_utc"].max()
    corridor_count = df["station_id"].nunique()
    timestamp_count = df["obs_time_utc"].nunique()

    expected_corridors = 64
    expected_hours = 12

    corridor_ready = corridor_count == expected_corridors
    history_ready = timestamp_count >= expected_hours

    # Hitung proporsi observed dan imputed
    observed_ratio = None
    imputed_ratio = None

    if "is_imputed" in df.columns:
        imputed_ratio = float(df["is_imputed"].mean())
        observed_ratio = 1.0 - imputed_ratio

    if corridor_ready and history_ready:
        status = "READY"
        message = "Data siap digunakan untuk forecasting."

    else:
        status = "NOT_READY"
        problems = []

        if not corridor_ready:
            problems.append(
                f"corridor hanya {corridor_count}/{expected_corridors}"
            )

        if not history_ready:
            problems.append(
                f"history hanya {timestamp_count}/{expected_hours} jam"
            )

        message = "; ".join(problems)

    return {
        "status": status,
        "message": message,
        "latest_timestamp": latest_timestamp,
        "corridor_count": corridor_count,
        "timestamp_count": timestamp_count,
        "observed_ratio": observed_ratio,
        "imputed_ratio": imputed_ratio,
    }