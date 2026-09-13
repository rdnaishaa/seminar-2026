from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_live_latest_12h import build_latest_12h
from src.agent.data_agent import load_live_data, check_live_data_status
from src.agent.forecast_agent import run_forecast


def run_live_pipeline():
    print("========================================")
    print("LIVE TRAFFIC FORECAST PIPELINE")
    print("========================================")

    print("\n[1] Building latest 12h input...")

    try:
        build_latest_12h()

    except RuntimeError as error:
        print(f"\nPIPELINE BELUM SIAP:")
        print(error)
        return

    print("\n[2] Checking Data Agent...")

    live_df = load_live_data()
    status = check_live_data_status(live_df)

    print(f"Status: {status['status']}")
    print(f"Message: {status['message']}")

    if status["status"] != "READY":
        print("\nPipeline dihentikan karena data belum siap.")
        return

    if status.get("latest_timestamp") is not None:
        print(
            f"Latest timestamp: "
            f"{status['latest_timestamp']}"
        )

    if status.get("corridor_count") is not None:
        print(
            f"Corridors: "
            f"{status['corridor_count']}"
        )

    if status.get("history_hours") is not None:
        print(
            f"History hours: "
            f"{status['history_hours']}"
        )

    print("\n[3] Running Adaptive ST-GNN forecast...")

    forecast_df = run_forecast()

    print("\nForecast berhasil.")
    print(f"Rows: {len(forecast_df)}")
    print(
        f"Corridors: "
        f"{forecast_df['station_id'].nunique()}"
    )
    print(
        f"Horizons: "
        f"{sorted(forecast_df['horizon'].unique().tolist())}"
    )

    output_path = (
        ROOT
        / "data"
        / "collected"
        / "latest_live_forecast.csv"
    )

    forecast_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nForecast saved to:\n{output_path}"
    )

    print("\n========================================")
    print("PIPELINE SELESAI")
    print("========================================")


if __name__ == "__main__":
    run_live_pipeline()