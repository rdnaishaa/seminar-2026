from pathlib import Path
import os

from dotenv import load_dotenv


# ============================================================
# ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# DATA PATHS
# ============================================================

CORRIDORS_PATH = ROOT / "corridors.csv"

FINAL_CORRIDORS_PATH = (
    ROOT
    / "data"
    / "final_64"
    / "final_64_corridors.csv"
)

UPDATED_DATA_DIR = (
    ROOT
    / "data_update_dosen"
    / "full_data_update"
)

COLLECTED_PATH = (
    ROOT
    / "data"
    / "collected"
    / "tomtom_flow_final64.csv"
)


# ============================================================
# ENVIRONMENT
# ============================================================

ENV_PATH = ROOT / ".env"


load_dotenv(
    dotenv_path=ENV_PATH,
    override=False,
)


TOMTOM_API_KEY = os.getenv(
    "TOMTOM_API_KEY"
)


# ============================================================
# DISPLAY NAME HELPER
# ============================================================

def station_id_to_display_name(station_id: str) -> str:
    """
    Convert station IDs such as:
        tt_kebonjeruk
        tt_kelapagading
        tt_tbsimatupang

    into readable fallback labels.

    More specific names can still come from corridors.csv.
    """

    name = str(station_id)

    if name.startswith("tt_"):
        name = name[3:]

    special_names = {
        "bsd": "BSD",
        "gatsu": "Gatot Subroto",
        "tbsimatupang": "TB Simatupang",
        "mtharyono": "MT Haryono",
        "sparman": "S. Parman",
        "merdekabar": "Merdeka Barat",
    }

    if name in special_names:
        return special_names[name]

    return (
        name
        .replace("_", " ")
        .title()
    )