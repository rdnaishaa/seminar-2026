import pandas as pd
from zoneinfo import ZoneInfo


WIB = ZoneInfo("Asia/Jakarta")


# ============================================================
# ROUTE CORRIDOR HISTORY
# ============================================================

def get_route_history(
    history,
    origin_id,
    destination_id,
):
    """
    Ambil data historis untuk corridor asal dan tujuan.
    """

    if history.empty:
        return pd.DataFrame()

    required_columns = [
        "station_id",
        "obs_time_utc",
        "current_speed",
        "free_flow_speed",
        "congestion_ratio",
    ]

    if any(
        column not in history.columns
        for column in required_columns
    ):
        return pd.DataFrame()

    df = history[
        history["station_id"].isin(
            [
                origin_id,
                destination_id,
            ]
        )
    ].copy()

    if df.empty:
        return pd.DataFrame()

    df["obs_time_wib"] = (
        df["obs_time_utc"]
        .dt
        .tz_convert(WIB)
    )

    df["hour_wib"] = (
        df["obs_time_wib"]
        .dt
        .hour
    )

    return (
        df.sort_values(
            [
                "obs_time_utc",
                "station_id",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# HISTORICAL PROFILE FOR ROUTE ENDPOINTS
# ============================================================

def get_route_hourly_profile(
    history,
    origin_id,
    destination_id,
):
    """
    Buat profil historis berdasarkan jam WIB
    untuk lokasi asal dan tujuan.
    """

    df = get_route_history(
        history,
        origin_id,
        destination_id,
    )

    if df.empty:
        return pd.DataFrame()

    profile = (
        df
        .groupby(
            [
                "station_id",
                "hour_wib",
            ]
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
    )

    return profile


# ============================================================
# SELECTED DEPARTURE HOUR SUMMARY
# ============================================================

def get_route_departure_summary(
    history,
    origin_id,
    destination_id,
    departure_hour,
):
    """
    Ambil kondisi historis rata-rata asal dan tujuan
    pada jam keberangkatan yang dipilih.
    """

    profile = get_route_hourly_profile(
        history,
        origin_id,
        destination_id,
    )

    if profile.empty:
        return pd.DataFrame()

    selected = profile[
        profile["hour_wib"]
        == int(departure_hour)
    ].copy()

    return selected.reset_index(
        drop=True
    )