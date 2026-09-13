from src.agent.data_agent import check_live_data_status


status = check_live_data_status()

print("\n=== TRAFFIC DATA AGENT ===")
print("Status:", status.get("status"))
print("Message:", status.get("message"))
print("Latest timestamp:", status.get("latest_timestamp"))
print("Corridors:", status.get("corridor_count"))
print("History hours:", status.get("timestamp_count"))

observed_ratio = status.get("observed_ratio")
imputed_ratio = status.get("imputed_ratio")

if observed_ratio is not None:
    print(f"Observed data: {observed_ratio * 100:.1f}%")
    print(f"Imputed data: {imputed_ratio * 100:.1f}%")