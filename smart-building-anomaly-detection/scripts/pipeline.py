import os
import pandas as pd
import logging

from scripts.preprocessing import preprocess_building
from scripts.feature_engineering import engineer_features


# -----------------------------
# Paths
# -----------------------------
RAW_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"
META_PATH = "the-building-data-genome-project/data/raw/meta_open.csv"


# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# -----------------------------
# Load metadata
# -----------------------------
meta = pd.read_csv(META_PATH)


# -----------------------------
# Get building list
# -----------------------------
buildings = [
    f.replace(".csv", "")
    for f in os.listdir(RAW_PATH)
    if f.endswith(".csv")
]


# -----------------------------
# Process buildings
# -----------------------------
for i, building in enumerate(buildings):

    try:

        logging.info(f"Processing {i+1}/{len(buildings)}: {building}")
        print(f"Processing {i+1}/{len(buildings)}: {building}")

        file_path = RAW_PATH + f"{building}.csv"

        # Preprocess raw data
        df = preprocess_building(file_path)

        # Feature engineering
        df = engineer_features(df, building, meta)

        # Save processed dataset
        output_path = PROCESSED_PATH + f"{building}_features.csv"
        df.to_csv(output_path)

        logging.info(f"Saved {building} with shape {df.shape}")

    except Exception as e:

        logging.error(f"Failed {building}: {e}")
        print("Failed:", building, e)


print("Pipeline finished.")