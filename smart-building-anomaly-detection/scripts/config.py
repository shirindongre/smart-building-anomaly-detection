CONFIG = {
    "data_url"       : "https://raw.githubusercontent.com/buds-lab/the-building-data-genome-project/refs/heads/master/data/processed/temp_open_utc_complete.csv",     # URL to the input dataset
    "meta_url"       : "https://raw.githubusercontent.com/buds-lab/the-building-data-genome-project/refs/heads/master/data/raw/meta_open.csv",      # URL to the metadata dataset
    "timestamp_col"   : "timestamp",           # Name of the timestamp column
    "energy_col"      : "energy",              # Name of the energy column
    "window_size"     : 24,                    # Sliding window length (timesteps)
    "test_split"      : 0.2,                   # Fraction held out for evaluation
    "lstm_units"      : [32, 16],              # Encoder LSTM units (stack)
    "latent_dim"      : 8,                    # Latent vector dimension
    "decoder_units"   : [16, 32],              # Decoder LSTM units (stack)
    "dropout_rate"    : 0.1,                   # Dropout for regularisation
    "learning_rate"   : 1e-3,                  # Adam learning rate
    "epochs"          : 25,                    # Max training epochs
    "batch_size"      : 32,                    # Mini-batch size
    "patience"        : 8,                     # EarlyStopping patience
    "threshold_sigma" : 3.0,                   # Anomaly threshold = mean + k*std
    "output_dir"      : "output_plots",        # Directory for saved figures
    "model_path"      : "lstm_ae_model.h5",    # Path to save the trained model
    "seed"            : 42,                    # Reproducibility seed
    "Window_SIZE"     : 24,                    # Sliding window length (timesteps)
    "n_features"     : 24,                    # Number of features per timestep (after engineering)
    "model_dir"      : "models",              # Directory to save trained models
    "output_dir"     : "output_plots",        # Directory to save output plots
}

FEATURE_COLS = [
    "energy",
    "hour", "day_of_week", "month",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "is_weekend",
    "lag_1", "lag_24", "lag_168",
    "rolling_mean_24", "rolling_std_24", "rolling_mean_168",
    "diff_1", "pct_change",
]

BUILDINGS = [
    "Office_Cristina",
    "PrimClass_Jolie",
    "Office_Jesus",
    "PrimClass_Jaylin",
]