from pathlib import Path
import pandas as pd

DATA_DIR = Path("data_update_dosen/full_data_update")

HISTORY = 12
MAX_HORIZON = 6

records = []

# ============================================================
# LOAD + RANK 96 CORRIDORS
# ============================================================

for f in DATA_DIR.glob("*.csv"):

    df = pd.read_csv(f)

    ts = pd.to_datetime(
        df["obs_time_utc"],
        utc=True,
        errors="coerce"
    ).dropna()

    records.append({
        "file": f.name,
        "unique_ts": ts.nunique(),
        "start": ts.min(),
        "end": ts.max(),
        "timestamps": set(ts),
    })

audit = (
    pd.DataFrame(records)
    .sort_values(
        ["unique_ts", "file"],
        ascending=[False, True]
    )
    .reset_index(drop=True)
)


# ============================================================
# COUNT CONTINUOUS SEQUENCES
# ============================================================

def get_sequences(timestamps):

    if not timestamps:
        return []

    ts = sorted(timestamps)

    sequences = []
    current = [ts[0]]

    for prev, curr in zip(ts[:-1], ts[1:]):

        if curr - prev == pd.Timedelta(hours=1):
            current.append(curr)

        else:
            sequences.append(current)
            current = [curr]

    sequences.append(current)

    return sequences


# ============================================================
# TEST N = 2 ... 96
# ============================================================

results = []

for n in range(2, 97):

    selected = audit.head(n)

    timestamp_sets = selected["timestamps"].tolist()

    common = set.intersection(*timestamp_sets)

    sequences = get_sequences(common)

    sequence_lengths = [
        len(seq)
        for seq in sequences
    ]

    # Need:
    # 12 input hours + future up to t+6
    # => sequence length must support 18 consecutive timestamps.
    usable_windows = sum(
        max(
            0,
            length - HISTORY - MAX_HORIZON + 1
        )
        for length in sequence_lengths
    )

    results.append({
        "N": n,
        "min_obs": selected["unique_ts"].min(),
        "common_ts": len(common),
        "num_sequences": len(sequences),
        "longest_sequence": (
            max(sequence_lengths)
            if sequence_lengths
            else 0
        ),
        "usable_windows": usable_windows,
    })


result = pd.DataFrame(results)

OUTPUT_DIR = Path("outputs/data_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

result.to_csv(
    OUTPUT_DIR / "network_size_analysis.csv",
    index=False,
)

# ============================================================
# IMPORTANT NETWORK SIZES
# ============================================================

important = [
    14,
    20,
    24,
    30,
    40,
    50,
    64,
    80,
    96,
]

print("\n=== IMPORTANT CANDIDATES ===")

print(
    result[
        result["N"].isin(important)
    ].to_string(index=False)
)


# ============================================================
# BEST N BY USABLE WINDOWS
# ============================================================

print("\n=== TOP 20 NETWORK SIZES BY USABLE WINDOWS ===")

print(
    result
    .sort_values(
        ["usable_windows", "N"],
        ascending=[False, False]
    )
    .head(20)
    .to_string(index=False)
)


# ============================================================
# WHERE COVERAGE CHANGES
# ============================================================

print("\n=== CHANGE POINTS ===")

previous = None

for _, row in result.iterrows():

    state = (
        row["common_ts"],
        row["usable_windows"],
    )

    if state != previous:

        print(
            f"N={int(row['N']):2d} | "
            f"common={int(row['common_ts']):3d} | "
            f"sequences={int(row['num_sequences']):3d} | "
            f"longest={int(row['longest_sequence']):3d}h | "
            f"windows={int(row['usable_windows']):3d}"
        )

        previous = state


# ============================================================
# CORRIDORS AT KEY CANDIDATES
# ============================================================

for n in [14, 24, 64]:

    print(f"\n=== N={n} CORRIDORS ===")

    for file in audit.head(n)["file"]:
        print(file)