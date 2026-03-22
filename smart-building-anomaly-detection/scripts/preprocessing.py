# scripts/preprocessing.py

import pandas as pd

def preprocess_building(df_wide: pd.DataFrame, building: str) -> pd.DataFrame:
    """
    Extract one building column from the wide DataFrame,
    clean and resample to hourly.
    """
    if building not in df_wide.columns:
        print(f"  [SKIP] {building} not found in dataset")
        return None

    # Pull just this building's column and rename to 'energy'
    df = df_wide[[building]].copy()
    df.columns = ["energy"]

    # Resample to hourly
    df_hr = df.resample("1h").sum(min_count=1)

    # Drop rows where energy is NaN
    df_clean = df_hr.loc[df_hr["energy"].notna()].copy()

    print(f"  {building}: {len(df_clean)} valid rows")
    return df_clean