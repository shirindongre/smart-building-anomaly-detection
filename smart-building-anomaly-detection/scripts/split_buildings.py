import pandas as pd
import os

INPUT_PATH = "the-building-data-genome-project/data/processed/temp_open_utc_complete.csv"
OUTPUT_PATH = "data/raw/"

os.makedirs(OUTPUT_PATH, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

timestamp = df["timestamp"]

for building in df.columns[1:]:

    building_df = pd.DataFrame({
        "timestamp": timestamp,
        "energy": df[building]
    })

    building_df.to_csv(f"{OUTPUT_PATH}{building}.csv", index=False)

    print("Saved:", building)