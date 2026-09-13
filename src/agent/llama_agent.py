import requests
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"


def get_corridor_forecast_summary(
    forecast_df: pd.DataFrame,
    station_id: str
) -> dict:
    """
    Mengambil hasil forecast untuk satu corridor
    dan mengubah waktu target dari UTC ke WIB.
    """

    corridor_df = forecast_df[
        forecast_df["station_id"] == station_id
    ].copy()

    if corridor_df.empty:
        raise ValueError(f"Corridor {station_id} tidak ditemukan.")

    corridor_df = corridor_df.sort_values("horizon")

    forecasts = []

    for _, row in corridor_df.iterrows():

        target_utc = pd.to_datetime(
            row["target_time_utc"],
            utc=True
        )

        target_wib = target_utc.tz_convert(
            ZoneInfo("Asia/Jakarta")
        )

        forecasts.append({
            "horizon": int(row["horizon"]),
            "target_time_utc": str(target_utc),
            "target_time_wib": target_wib.strftime(
                "%d %B %Y, %H:%M WIB"
            ),
            "predicted_speed": round(
                float(row["predicted_speed"]),
                2
            ),
        })

    return {
        "station_id": station_id,
        "forecasts": forecasts,
    }


def build_ollama_prompt(
    forecast_df: pd.DataFrame,
    station_id: str,
    user_question: Optional[str] = None
) -> str:
    """
    Membuat prompt untuk Ollama berdasarkan
    hasil prediksi Adaptive ST-GNN.
    """

    data = get_corridor_forecast_summary(
        forecast_df=forecast_df,
        station_id=station_id
    )

    forecast_lines = []

    for item in data["forecasts"]:

        forecast_lines.append(
            f"- {item['horizon']} jam ke depan "
            f"({item['target_time_wib']}): "
            f"{item['predicted_speed']} km/jam"
        )

    forecast_text = "\n".join(forecast_lines)

    question = (
        user_question
        if user_question
        else "Jelaskan prediksi kondisi lalu lintas corridor ini."
    )

    prompt = f"""
Anda adalah Traffic Forecast Interpretation Agent.

Tugas Anda hanya menjelaskan hasil prediksi lalu lintas
yang telah dihasilkan oleh model Adaptive ST-GNN.

ATURAN PENTING:

1. Jangan membuat angka prediksi baru.
2. Jangan mengubah angka prediksi yang diberikan.
3. Jangan mengubah waktu target yang diberikan.
4. Jangan menghitung waktu sendiri.
5. Gunakan waktu WIB yang telah diberikan.
6. Gunakan istilah:
   - "1 jam ke depan"
   - "3 jam ke depan"
   - "6 jam ke depan"
7. Jangan menggunakan istilah "jam 1", "jam 3", atau "jam 6".
8. Angka kecepatan merupakan hasil prediksi model,
   bukan kondisi aktual.
9. Jangan mengklaim data sebagai real-time jika sumber
   data merupakan data historis.
10. Jangan mengatakan bahwa adaptive graph menemukan
    jalur jalan sebenarnya.
11. Jelaskan arah perubahan kecepatan antar-horizon.
12. Gunakan bahasa Indonesia yang singkat,
    natural, dan mudah dipahami.
13. Jangan menambahkan informasi yang tidak tersedia
    pada data berikut.

Corridor:
{data['station_id']}

Hasil prediksi Adaptive ST-GNN:

{forecast_text}

Pertanyaan pengguna:
{question}

Gunakan format jawaban berikut:

"Prediksi untuk Corridor {data['station_id']}:
Pada 1 jam ke depan, pukul [WAKTU_WIB], kecepatan diprediksi [ANGKA] km/jam.
Pada 3 jam ke depan, pukul [WAKTU_WIB], kecepatan diprediksi [ANGKA] km/jam.
Pada 6 jam ke depan, pukul [WAKTU_WIB], kecepatan diprediksi [ANGKA] km/jam.
Secara umum, [jelaskan tren naik/turun/stabil secara singkat]."

WAJIB gunakan waktu dan angka persis dari data yang diberikan.
Jangan menghilangkan waktu WIB.
""".strip()

    return prompt


def run_ollama_agent(
    forecast_df: pd.DataFrame,
    station_id: str,
    user_question: Optional[str] = None
) -> str:
    """
    Mengirim hasil forecast ke Ollama lokal
    untuk menghasilkan penjelasan natural language.
    """

    prompt = build_ollama_prompt(
        forecast_df=forecast_df,
        station_id=station_id,
        user_question=user_question
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Ollama tidak dapat dihubungi. "
            "Pastikan 'ollama serve' sedang berjalan."
        )

    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Ollama membutuhkan waktu terlalu lama untuk merespons."
        )

    except requests.exceptions.RequestException as error:
        raise RuntimeError(
            f"Terjadi error saat menghubungi Ollama: {error}"
        )

    result = response.json()

    if "response" not in result:
        raise RuntimeError(
            "Response dari Ollama tidak memiliki field 'response'."
        )

    return result["response"].strip()