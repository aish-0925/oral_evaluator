import pandas as pd

# Path to your logged evaluation results
CSV_PATH = "outputs/evaluation_results.csv"

df = pd.read_csv(CSV_PATH)

before = len(df)

# Drop exact duplicates OR duplicates by audio_path + timestamp
df = df.drop_duplicates(subset=["audio_path", "timestamp"], keep="last")

df.to_csv(CSV_PATH, index=False)

after = len(df)

print(f"Deduplication complete: {before} → {after}")
