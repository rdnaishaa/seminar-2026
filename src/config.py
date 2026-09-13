from pathlib import Path
import os

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]

CORRIDORS_PATH = ROOT / "corridors.csv"

FINAL_CORRIDORS_PATH = (
    ROOT
    / "data"
    / "final_64"
    / "final_64_corridors.csv"
)

TRAIN_HISTORY_PATH = (
    ROOT
    / "data"
    / "final_64"
    / "forecasting_train_preprocessed.csv"
)

ENV_PATH = ROOT / ".env"

load_dotenv(
    dotenv_path=ENV_PATH,
    override=False,
)

TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")


def station_id_to_display_name(station_id):
    value = str(station_id)

    if value.startswith("tt_"):
        value = value[3:]

    value = value.replace("_", " ")

    return value.title()
