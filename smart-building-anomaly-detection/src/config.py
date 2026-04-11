CONFIG = {
    # Wide training source (training only). Inference uses data/processed/{building}.csv per building.
    "data_url": "https://raw.githubusercontent.com/buds-lab/the-building-data-genome-project/refs/heads/master/data/processed/temp_open_utc_complete.csv",
    "meta_url": "https://raw.githubusercontent.com/buds-lab/the-building-data-genome-project/refs/heads/master/data/raw/meta_open.csv",
    "timestamp_col": "timestamp",
    "energy_col": "energy",
    "window_size": 24,  # Sliding window length (timesteps) for LSTM sequences
    "test_split": 0.2,
    "lstm_units": [32, 16],
    "latent_dim": 8,
    "decoder_units": [16, 32],
    "dropout_rate": 0.1,
    "learning_rate": 1e-3,
    "epochs": 25,
    "batch_size": 32,
    "patience": 8,
    "threshold_sigma": 3.0,
    # Sensor LSTM post-processing (Model_Testing): rolling error stats, threshold, drift
    "error_rolling_window_hours": 168,  # 7 days of hourly samples — median / MAD of reconstruction_error
    "mad_epsilon": 1e-6,  # Stabilizes normalized_error = (err - median) / (MAD + eps)
    "threshold_mad_k": 6.0,  # threshold_raw = rolling_median + k * rolling_mad
    "threshold_ema_span": 24,  # Smooth threshold_raw with EWM before drift-freeze
    "drift_z_threshold": 5.0,  # Drift candidate when normalized_error exceeds this
    "drift_sustain_points": 24,  # Drift if condition holds for this many consecutive windows
    "output_dir": "output_plots",
    "model_path": "lstm_ae_model.h5",
    "seed": 42,
    "model_dir": "models",
}

FEATURE_COLS = [
    "energy",
    "hour",
    "day_of_week",
    "month",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_std_24",
    "rolling_mean_168",
    "diff_1",
    "pct_change",
]

BUILDINGS = [
    "Office_Annika",
    "Office_Cristina",
    "PrimClass_Jolie",
    "Office_Jesus",
    "PrimClass_Jaylin",
]
