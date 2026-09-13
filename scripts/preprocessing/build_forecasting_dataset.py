from pathlib import Path
import pandas as pd

# =========================================================
# PATH
# =========================================================
ROOT = Path(__file__).resolve().parents[1]
FULL_DATA_DIR = ROOT / "full_data"
OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOAD ALL CORRIDOR DATA
# =========================================================
records = []

for file in sorted(FULL_DATA_DIR.glob("*.csv")):
    df = pd.read_csv(file)

    if "obs_time_utc" not in df.columns:
        continue

    df["obs_time_utc"] = pd.to_datetime(
        df["obs_time_utc"],
        utc=True,
        errors="coerce"
    )

    df = df.dropna(subset=["obs_time_utc"])

    records.append({
        "file": file,
        "rows": len(df),
        "start": df["obs_time_utc"].min(),
        "end": df["obs_time_utc"].max()
    })

coverage = pd.DataFrame(records)

# =========================================================
# SELECT TOP 14
# =========================================================
top14 = (
    coverage
    .sort_values(
        ["rows", "start"],
        ascending=[False, True]
    )
    .head(14)
)

print("\n=== TOP 14 CORRIDORS ===")

for i, row in enumerate(top14.itertuples(), start=1):
    print(
        f"{i:02d}. {row.file.stem:<35} "
        f"{row.rows} observations"
    )

# =========================================================
# LOAD TOP 14
# =========================================================
all_data = []

for row in top14.itertuples():

    df = pd.read_csv(row.file)

    df["obs_time_utc"] = pd.to_datetime(
        df["obs_time_utc"],
        utc=True
    )

    df["corridor_file"] = row.file.stem

    all_data.append(df)

combined = pd.concat(
    all_data,
    ignore_index=True
)

combined = combined.sort_values(
    ["obs_time_utc", "corridor_file"]
)

# =========================================================
# GET COMMON TIMESTAMPS
# =========================================================
counts = (
    combined
    .groupby("obs_time_utc")["corridor_file"]
    .nunique()
)

common_timestamps = counts[counts == 14].index

combined = combined[
    combined["obs_time_utc"].isin(common_timestamps)
].copy()

timestamps = sorted(common_timestamps)

print("\n=== TEMPORAL COVERAGE ===")
print(f"Corridors         : 14")
print(f"Common timestamps : {len(timestamps)}")
print(f"Start             : {timestamps[0]}")
print(f"End               : {timestamps[-1]}")

# =========================================================
# CHRONOLOGICAL SPLIT
# 70% TRAIN
# 15% VALIDATION
# 15% TEST
# =========================================================
n = len(timestamps)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_times = timestamps[:train_end]
val_times = timestamps[train_end:val_end]
test_times = timestamps[val_end:]

train = combined[
    combined["obs_time_utc"].isin(train_times)
].copy()

val = combined[
    combined["obs_time_utc"].isin(val_times)
].copy()

test = combined[
    combined["obs_time_utc"].isin(test_times)
].copy()

# =========================================================
# SAVE
# =========================================================
combined.to_csv(
    OUTPUT_DIR / "forecasting_top14_raw.csv",
    index=False
)

train.to_csv(
    OUTPUT_DIR / "forecasting_train_raw.csv",
    index=False
)

val.to_csv(
    OUTPUT_DIR / "forecasting_val_raw.csv",
    index=False
)

test.to_csv(
    OUTPUT_DIR / "forecasting_test_raw.csv",
    index=False
)

# =========================================================
# SUMMARY
# =========================================================
print("\n=== CHRONOLOGICAL SPLIT ===")

print(
    f"TRAIN : {len(train_times)} timestamps "
    f"({len(train_times)/n*100:.2f}%)"
)

print(
    f"VAL   : {len(val_times)} timestamps "
    f"({len(val_times)/n*100:.2f}%)"
)

print(
    f"TEST  : {len(test_times)} timestamps "
    f"({len(test_times)/n*100:.2f}%)"
)

print("\nTrain:")
print(f"  {train_times[0]} -> {train_times[-1]}")

print("\nValidation:")
print(f"  {val_times[0]} -> {val_times[-1]}")

print("\nTest:")
print(f"  {test_times[0]} -> {test_times[-1]}")

print("\n=== SAVED ===")
print("data/processed/forecasting_top14_raw.csv")
print("data/processed/forecasting_train_raw.csv")
print("data/processed/forecasting_val_raw.csv")
print("data/processed/forecasting_test_raw.csv")