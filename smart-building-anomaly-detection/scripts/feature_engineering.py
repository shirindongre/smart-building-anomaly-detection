import numpy as np


def engineer_features(df, building, meta):

    # get building timezone
    tz = meta.loc[meta["uid"] == building, "timezone"].values[0]

    # convert UTC → building timezone
    df.index = df.index.tz_convert(tz)

    # remove timezone info
    df.index = df.index.tz_localize(None)

    # -------------------
    # time features
    # -------------------
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    # cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # weekend indicator
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # -------------------
    # lag features
    # -------------------
    df["lag_1"] = df["energy"].shift(1)
    df["lag_24"] = df["energy"].shift(24)
    df["lag_168"] = df["energy"].shift(168)

    # -------------------
    # rolling statistics
    # -------------------
    df["rolling_mean_24"] = df["energy"].rolling(24).mean()
    df["rolling_std_24"] = df["energy"].rolling(24).std()

    df["rolling_mean_168"] = df["energy"].rolling(168).mean()

    # -------------------
    # change features
    # -------------------
    df["diff_1"] = df["energy"].diff()
    df["pct_change"] = df["energy"].pct_change()

    # drop NaN rows created by lag/rolling
    df = df.dropna()

    # ensure timestamps sorted
    df = df.sort_index()

    return df