import os
from pathlib import Path
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = info, 2 = warnings, 3 = errors only
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
_SCRIPT_DIR = Path(__file__).resolve().parent
for p in (_PROJECT_ROOT, _SCRIPT_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
import matplotlib
matplotlib.use("Agg")              # Use non-interactive backend for file saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib

import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scripts.preprocessing import preprocess_building
from scripts.feature_engineering import engineer_features
from scripts.scaling import scale_features
from scripts.sequence_creations import create_sequences

# Global Configurations
from src.config import CONFIG
# Reproducibility
np.random.seed(CONFIG["seed"])
tf.random.set_seed(CONFIG["seed"])

_output_dir = _PROJECT_ROOT / CONFIG["output_dir"]
_output_dir.mkdir(parents=True, exist_ok=True)

#-----------------------------------------------------------
# Data Loading and Preprocessing
#-----------------------------------------------------------


# Dataset importing
df_wide = pd.read_csv(CONFIG["data_url"])
df_wide["timestamp"] = pd.to_datetime(df_wide["timestamp"], utc=True)
df_wide = df_wide.set_index("timestamp")
print(df_wide.head())
print(f"Dataset shape: {df_wide.shape}")
print(f"Total buildings: {len(df_wide.columns)}")

META_PATH = CONFIG["meta_url"]
meta = pd.read_csv(META_PATH)

from src.config import FEATURE_COLS
from src.config import BUILDINGS

selected_buildings = BUILDINGS  # Select the first five buildings for analysis

building_data = {}

for building in selected_buildings:
    df_clean = preprocess_building(df_wide, building)
    if df_clean is None or len(df_clean) <= 500:
        print(f"  [SKIP] {building}: insufficient rows or missing from wide dataset")
        continue

    building_data[building] = df_clean

    df_features = engineer_features(df_clean, building, meta)
    print(f"  Features shape: {df_features.shape}")

    # Per-building CSV for testing (load → scale → sequence → predict; no wide table at inference)
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = _PROCESSED_DIR / f"{building}.csv"
    df_features.reset_index().to_csv(out_csv, index=False)
    print(f"  Saved {out_csv}")

    X_train_sc, X_test_sc, scaler, split_idx = scale_features(
        df=df_features,
        feature_cols=FEATURE_COLS,
        test_split=CONFIG["test_split"],
    )
    _scalers_dir = _PROJECT_ROOT / "scalers"
    _scalers_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, _scalers_dir / f"{building}_scaler.pkl")

    # Step D — sliding windows
    X_train_seq = create_sequences(X_train_sc, CONFIG["window_size"])
    X_test_seq = create_sequences(X_test_sc, CONFIG["window_size"])

    print(f"  Train windows  : {X_train_seq.shape}")   # (n_windows, 30, n_features)
    print(f"  Test windows   : {X_test_seq.shape}")

    # Store everything needed for model training
    building_data[building] = {
        "df"          : df_features,
        "X_train"     : X_train_seq,
        "X_test"      : X_test_seq,
        "scaler"      : scaler,
        "split_idx"   : split_idx,
    }


print(f"\nReady: {len(building_data)} buildings")

if not building_data:
    raise SystemExit("No buildings processed — check BUILDINGS and wide dataset columns.")

sample = building_data[next(iter(building_data))]
print("\nTrain shape :", sample["X_train"].shape)
print("Test shape  :", sample["X_test"].shape)
print("Features per step:", sample["X_train"].shape[2])


#-----------------------------------------------------------
# Model Implemenntation
#-----------------------------------------------------------

from scripts.Model_Architechture import build_lstm_autoencoder

trained_models = {}

for building, data in building_data.items():

    print(f"\n{'='*40}")
    print(f"Training model: {building}")

    X_train    = data["X_train"]
  

    # build a fresh model for each building
    model = build_lstm_autoencoder(
        window_size   = X_train.shape[1],  # number of timesteps per window
        n_features    = X_train.shape[2],  # number of features per timestep
        lstm_units    = CONFIG["lstm_units"],
        latent_dim    = CONFIG["latent_dim"],
        decoder_units = CONFIG["decoder_units"],
        dropout_rate  = CONFIG["dropout_rate"],
        learning_rate = CONFIG["learning_rate"],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=CONFIG["patience"],
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(_PROJECT_ROOT / CONFIG["model_dir"] / f"{building}_best.h5"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0
        ),
    ]

    (_PROJECT_ROOT / CONFIG["model_dir"]).mkdir(parents=True, exist_ok=True)

    # autoencoder training — input == target
    history = model.fit(
        X_train, X_train,
        epochs          = CONFIG["epochs"],
        batch_size      = CONFIG["batch_size"],
        validation_split= 0.1,
        callbacks       = callbacks,
        shuffle         = True,
        verbose         = 1,
    )

    trained_models[building] = {
        "model"  : model,
        "history": history,
    }

    # plot training loss
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"],     label="Train loss")
    plt.plot(history.history["val_loss"], label="Val loss", linestyle="--")
    plt.title(f"Training loss — {building}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    _output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(_output_dir / f"{building}_loss.png")
    plt.close()
    print(f"  Loss plot saved for {building}")

print(f"\nAll done. {len(trained_models)} models trained.")


