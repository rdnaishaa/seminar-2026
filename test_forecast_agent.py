from src.agent.forecast_agent import run_forecast


forecast = run_forecast()

print("\n=== TRAFFIC FORECAST AGENT ===")

print(
    forecast.head(15).to_string(
        index=False
    )
)

print("\nRows:", len(forecast))

print(
    "Horizons:",
    sorted(
        forecast["horizon"]
        .unique()
        .tolist()
    )
)

print(
    "Corridors:",
    forecast[
        "station_id"
    ].nunique()
)