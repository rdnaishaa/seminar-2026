from src.agent.forecast_agent import run_forecast
from src.agent.llama_agent import run_ollama_agent


forecast_df = run_forecast()

answer = run_ollama_agent(
    forecast_df=forecast_df,
    station_id="tt_sudirman",
    user_question="Bagaimana prediksi lalu lintas Sudirman?"
)

print("\n=== OLLAMA TRAFFIC AGENT ===")
print(answer)