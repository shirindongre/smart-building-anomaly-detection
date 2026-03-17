import pandas as pd

def preprocess_building(file_path):

    df = pd.read_csv(file_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp")

    df = df.set_index("timestamp")

    df_hr = df.resample("1h").sum(min_count=1)

    df_valid = df_hr.loc[df_hr["energy"].notna()]

    return df_valid