import pandas as pd

from zoneinfo import ZoneInfo


WIB = ZoneInfo("Asia/Jakarta")


# ============================================================
# LATEST STATUS
# ============================================================

def get_latest_status(history):

    if history.empty:
        return pd.DataFrame()

    df = history.copy()

    if "obs_time_utc" not in df.columns:
        return pd.DataFrame()

    latest_time = (
        df["obs_time_utc"]
        .max()
    )

    latest = df[
        df["obs_time_utc"]
        == latest_time
    ].copy()

    return latest


# ============================================================
# CORRIDOR HISTORY
# ============================================================

def get_corridor_history(
    history,
    station_id,
):

    if history.empty:
        return pd.DataFrame()

    if "station_id" not in history.columns:
        return pd.DataFrame()

    df = history[
        history["station_id"]
        == station_id
    ].copy()

    if "obs_time_utc" in df.columns:

        df = df.sort_values(
            "obs_time_utc"
        )

    return df.reset_index(
        drop=True
    )


# ============================================================
# HOURLY PROFILE
# ============================================================

def hourly_profile(
    history,
    station_id,
):

    df = get_corridor_history(
        history,
        station_id,
    )

    if df.empty:
        return pd.DataFrame()

    required_columns = [
        "obs_time_utc",
        "current_speed",
        "free_flow_speed",
        "congestion_ratio",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Convert UTC timestamp to WIB
    # --------------------------------------------------------

    df["obs_time_wib"] = (
        df["obs_time_utc"]
        .dt
        .tz_convert(
            WIB
        )
    )

    df["hour_wib"] = (
        df["obs_time_wib"]
        .dt
        .hour
    )

    # --------------------------------------------------------
    # Aggregate per WIB hour
    # --------------------------------------------------------

    profile = (
        df
        .groupby(
            "hour_wib"
        )
        .agg(
            avg_speed=(
                "current_speed",
                "mean",
            ),
            avg_free_flow_speed=(
                "free_flow_speed",
                "mean",
            ),
            avg_congestion_ratio=(
                "congestion_ratio",
                "mean",
            ),
            observations=(
                "current_speed",
                "count",
            ),
        )
        .reset_index()
        .sort_values(
            "hour_wib"
        )
    )

    return profile.reset_index(
        drop=True
    )


# ============================================================
# CORRIDOR SUMMARY
# ============================================================

def get_corridor_summary(
    history,
    station_id,
):

    df = get_corridor_history(
        history,
        station_id,
    )

    if df.empty:
        return None

    required_columns = [
        "current_speed",
        "free_flow_speed",
        "congestion_ratio",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        return None

    return {
        "avg_speed": float(
            df["current_speed"].mean()
        ),
        "avg_free_flow_speed": float(
            df["free_flow_speed"].mean()
        ),
        "avg_congestion_ratio": float(
            df["congestion_ratio"].mean()
        ),
        "observations": int(
            df["current_speed"].count()
        ),
    }


# ============================================================
# BEST / WORST HOUR
# ============================================================

def get_best_worst_hour(
    history,
    station_id,
):

    profile = hourly_profile(
        history,
        station_id,
    )

    if profile.empty:
        return None

    if (
        profile["hour_wib"]
        .nunique()
        < 3
    ):
        return None

    valid_profile = (
        profile
        .dropna(
            subset=[
                "avg_speed"
            ]
        )
    )

    if valid_profile.empty:
        return None

    best = valid_profile.loc[
        valid_profile[
            "avg_speed"
        ].idxmax()
    ]

    worst = valid_profile.loc[
        valid_profile[
            "avg_speed"
        ].idxmin()
    ]

    return {
        "best_hour": int(
            best["hour_wib"]
        ),
        "best_speed": float(
            best["avg_speed"]
        ),
        "best_congestion_ratio": float(
            best["avg_congestion_ratio"]
        ),

        "worst_hour": int(
            worst["hour_wib"]
        ),
        "worst_speed": float(
            worst["avg_speed"]
        ),
        "worst_congestion_ratio": float(
            worst["avg_congestion_ratio"]
        ),
    }