import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = info, 2 = warnings, 3 = errors only
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")              # Use non-interactive backend for file saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocessing import preprocess_building
from feature_engineering import engineer_features
from scaling import scale_features
from sequence_creations import create_sequences

# Global Configurations
from config import CONFIG
# Reproducibility
np.random.seed(CONFIG["seed"])
tf.random.set_seed(CONFIG["seed"])

os.makedirs(CONFIG["output_dir"], exist_ok=True)

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

from config import FEATURE_COLS

selected_buildings = list(df_wide.columns[0:4])  # Select the first five buildings for analysis

building_data = {}

for building in selected_buildings:
    # Data preprocessing
    df_clean = preprocess_building(df_wide, building)
    if df_clean is not None and len(df_clean) > 500:
        building_data[building] = df_clean

    # Feature engineering
    df_features = engineer_features(df_clean, building, meta)
    print(f"  Features shape: {df_features.shape}")
    
    # Scaling 
    X_train_sc, X_test_sc, scaler, split_idx = scale_features(
        df           = df_features,
        feature_cols = FEATURE_COLS,
        test_split   = 0.2
    )

    # Step D — sliding windows
    X_train_seq = create_sequences(X_train_sc, CONFIG["Window_SIZE"])
    X_test_seq  = create_sequences(X_test_sc,  CONFIG["Window_SIZE"])

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

sample = building_data[selected_buildings[0]]
print("\nTrain shape :", sample["X_train"].shape)
print("Test shape  :", sample["X_test"].shape)
print("Features per step:", sample["X_train"].shape[2])


#-----------------------------------------------------------
# Model Implemenntation
#-----------------------------------------------------------

from Model_Architechture import build_lstm_autoencoder

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
            filepath=f"models/{building}_best.h5",
            monitor="val_loss",
            save_best_only=True,
            verbose=0
        ),
    ]

    os.makedirs("models", exist_ok=True)

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
    os.makedirs("output_plots", exist_ok=True)
    plt.savefig(f"output_plots/{building}_loss.png")
    plt.close()
    print(f"  Loss plot saved for {building}")

print(f"\nAll done. {len(trained_models)} models trained.")


