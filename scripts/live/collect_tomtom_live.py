import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]

ENV_PATH = ROOT / ".env"
FINAL_CORRIDORS_PATH = ROOT / "data" / "final_64" / "final_64_corridors.csv"
CORRIDORS_PATH = ROOT / "corridors.csv"

OUTPUT_DIR = ROOT / "data" / "collected"
OUTPUT_PATH = OUTPUT_DIR / "tomtom_flow_final64.csv"

load_dotenv(ENV_PATH)

TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")

if not TOMTOM_API_KEY:
    raise ValueError(
        "TOMTOM_API_KEY tidak ditemukan di file .env"
    )


def load_final_64():
    final_df = pd.read_csv(FINAL_CORRIDORS_PATH)
    metadata_df = pd.read_csv(CORRIDORS_PATH)

    if "station_id" not in metadata_df.columns:
        raise ValueError(
            "Kolom station_id tidak ditemukan di corridors.csv"
        )

    if "corridor_file" not in final_df.columns:
        raise ValueError(
            "Kolom corridor_file tidak ditemukan di final_64_corridors.csv"
        )

    final_df["station_id"] = (
        final_df["corridor_file"]
        .astype(str)
        .str.replace(".csv", "", regex=False)
    )

    merged = final_df.merge(
        metadata_df,
        on="station_id",
        how="left"
    )

    required = [
        "station_id",
        "name",
        "city",
        "lat",
        "lon",
    ]

    missing = [
        col for col in required
        if col not in merged.columns
    ]

    if missing:
        raise ValueError(
            f"Kolom metadata tidak lengkap: {missing}"
        )

    if merged[required].isna().any().any():
        bad = merged[
            merged[required].isna().any(axis=1)
        ]

        raise ValueError(
            "Ada corridor final64 yang metadata-nya tidak ditemukan:\n"
            + bad[["station_id"]].to_string(index=False)
        )

    return merged.reset_index(drop=True)


def fetch_flow(lat, lon):
    url = (
        "https://api.tomtom.com/traffic/services/4/"
        "flowSegmentData/absolute/10/json"
    )

    params = {
        "point": f"{lat},{lon}",
        "unit": "KMPH",
        "key": TOMTOM_API_KEY,
    }

    response = requests.get(
        url,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if "flowSegmentData" not in data:
        raise ValueError(
            f"Response TomTom tidak memiliki flowSegmentData: {data}"
        )

    return data["flowSegmentData"]


def collect_once():
    corridors = load_final_64()

    collected_at = pd.Timestamp.now(tz="UTC").floor("h")
    rows = []

    print("=== TOMTOM LIVE COLLECTOR ===")
    print(f"Target corridors: {len(corridors)}")
    print(f"Collection time UTC: {collected_at}")
    print()

    for idx, row in corridors.iterrows():
        station_id = row["station_id"]

        try:
            flow = fetch_flow(
                lat=row["lat"],
                lon=row["lon"],
            )

            current_speed = flow.get("currentSpeed")
            free_flow_speed = flow.get("freeFlowSpeed")
            current_travel_time = flow.get("currentTravelTime")
            free_flow_travel_time = flow.get("freeFlowTravelTime")
            confidence = flow.get("confidence")
            road_closure = flow.get("roadClosure")

            congestion_ratio = None

            if (
                current_speed is not None
                and free_flow_speed not in (None, 0)
            ):
                congestion_ratio = (
                    1 - (current_speed / free_flow_speed)
                )

            rows.append({
                "obs_time_utc": collected_at,
                "station_id": station_id,
                "name": row["name"],
                "city": row["city"],
                "lat": row["lat"],
                "lon": row["lon"],
                "current_speed": current_speed,
                "free_flow_speed": free_flow_speed,
                "current_travel_time": current_travel_time,
                "free_flow_travel_time": free_flow_travel_time,
                "congestion_ratio": congestion_ratio,
                "road_closure": road_closure,
                "confidence": confidence,
            })

            print(
                f"[{idx + 1:02d}/{len(corridors)}] "
                f"{station_id:<25} "
                f"speed={current_speed}"
            )

        except Exception as error:
            print(
                f"[ERROR] {station_id}: {error}"
            )

        # sedikit jeda supaya request tidak terlalu agresif
        time.sleep(0.10)

    new_df = pd.DataFrame(rows)

    if new_df.empty:
        raise RuntimeError(
            "Tidak ada data yang berhasil dikumpulkan."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if OUTPUT_PATH.exists():
        old_df = pd.read_csv(
            OUTPUT_PATH,
            parse_dates=["obs_time_utc"]
        )

        combined = pd.concat(
            [old_df, new_df],
            ignore_index=True
        )

        combined = (
            combined
            .sort_values([
                "obs_time_utc",
                "station_id"
            ])
            .drop_duplicates(
                subset=[
                    "obs_time_utc",
                    "station_id"
                ],
                keep="last"
            )
            .reset_index(drop=True)
        )

    else:
        combined = new_df

    combined.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("=== COLLECTION SUMMARY ===")
    print(f"Successful corridors: {len(new_df)}")
    print(f"Expected corridors: {len(corridors)}")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Total stored rows: {len(combined)}")


if __name__ == "__main__":
    collect_once()