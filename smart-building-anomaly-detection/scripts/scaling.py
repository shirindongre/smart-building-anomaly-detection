from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def scale_features(df: pd.DataFrame,
                   feature_cols: list,
                   test_split: float = 0.2):
    """
    Scale features after feature engineering has been applied.
    Fitted only on training portion to prevent data leakage.

    Parameters
    ----------
    df           : DataFrame output from feature engineering
    feature_cols : List of feature column names to scale
    test_split   : Fraction of data held for testing (default 0.2)

    Returns
    -------
    X_train_sc : Scaled training array  (n_train, n_features)
    X_test_sc  : Scaled test array      (n_test,  n_features)
    scaler     : Fitted MinMaxScaler
    split_idx  : Integer index of train/test boundary
    """

    # Extract only the feature columns as a numpy array
    data = df[feature_cols].values

    # Calculate where training ends and test begins
    split_idx = int(len(data) * (1 - test_split))

    X_train_raw = data[:split_idx]
    X_test_raw  = data[split_idx:]

    # Fit scaler ONLY on training data — never on test
    scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train_raw)
    X_test_sc  = scaler.transform(X_test_raw)

    print(f"  Building split — Train: {X_train_sc.shape[0]} rows | Test: {X_test_sc.shape[0]} rows")

    return X_train_sc, X_test_sc, scaler, split_idx