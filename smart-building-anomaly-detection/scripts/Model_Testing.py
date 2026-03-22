import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from config import CONFIG, FEATURE_COLS, BUILDINGS
os.makedirs(CONFIG["output_dir"], exist_ok=True)
from preprocessing import preprocess_building
from feature_engineering import engineer_features
from scaling import scale_features
from sequence_creations import create_sequences

def compute_reconstruction_errors(model, sequences: np.ndarray) -> np.ndarray:
    """
    Run sequences through loaded model and compute per-window MSE.
    MSE is averaged across all timesteps and all features → one scalar per window.
    """
    reconstructed = model.predict(sequences, verbose=0)
    errors = np.mean(np.power(sequences - reconstructed, 2), axis=(1, 2))
    return errors


def compute_threshold(train_errors: np.ndarray,
                      sigma: float = 3.0) -> float:
    """
    Threshold = mean(train_errors) + sigma * std(train_errors)
    Derived from training errors only — no labels needed.
    """
    return float(np.mean(train_errors) + sigma * np.std(train_errors))


def detect_anomalies(train_errors, test_errors, threshold):
    """
    Apply threshold to flag anomalies.
    Returns flags for test set only.
    """
    all_errors    = np.concatenate([train_errors, test_errors])
    anomaly_flags = (all_errors > threshold).astype(int)
    return all_errors, anomaly_flags


def build_results_df(df_features, all_errors, anomaly_flags,
                     window_size, split_idx):
    """
    Align errors and flags back to the original time index.
    Each error is assigned to the last timestep of its window.
    """
    time_index = df_features.index[window_size - 1 :
                                   window_size - 1 + len(all_errors)]

    results_df = pd.DataFrame({
        "energy"               : df_features["energy"].values[
                                     window_size - 1 :
                                     window_size - 1 + len(all_errors)],
        "reconstruction_error" : all_errors,
        "anomaly"              : anomaly_flags,
        "split"                : ["train" if i < (split_idx - window_size + 1)
                                  else "test"
                                  for i in range(len(all_errors))],
    }, index=time_index)

    return results_df


def save_plots(results_df, train_errors, threshold,
               building, output_dir):
    """
    Three plots per building:
      1. Reconstruction error over time with threshold
      2. Energy signal with anomaly points highlighted
      3. Error distribution (train vs all)
    """
    all_errors    = results_df["reconstruction_error"].values
    anomaly_mask  = results_df["anomaly"] == 1
    normal_mask   = results_df["anomaly"] == 0

    # ── Plot 1: reconstruction error timeline ──
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(results_df.index, results_df["reconstruction_error"],
            color="#6B7280", lw=0.8, label="Reconstruction error", alpha=0.8)
    ax.axhline(threshold, color="#DC2626", lw=2, linestyle="--",
               label=f"Threshold (μ+{CONFIG['threshold_sigma']}σ = {threshold:.4f})")
    ax.fill_between(results_df.index,
                    results_df["reconstruction_error"], threshold,
                    where=results_df["reconstruction_error"] > threshold,
                    color="#FCA5A5", alpha=0.5, label="Anomaly region")
    ax.set_title(f"Reconstruction Error — {building}", fontweight="bold")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{building}_01_error_timeline.png", dpi=150)
    plt.close()

    # ── Plot 2: energy with anomalies highlighted ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(results_df.index[normal_mask],
                 results_df["energy"][normal_mask],
                 color="#6B7280", lw=0.8, label="Normal", alpha=0.9)
    axes[0].scatter(results_df.index[anomaly_mask],
                    results_df["energy"][anomaly_mask],
                    color="#DC2626", s=18, zorder=5, label="Anomaly")
    axes[0].set_title(f"Energy Consumption — Anomalies Highlighted — {building}",
                      fontweight="bold")
    axes[0].set_ylabel("Energy (scaled)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(results_df.index, results_df["reconstruction_error"],
                 color="#6B7280", lw=0.8, alpha=0.8)
    axes[1].axhline(threshold, color="#DC2626", lw=1.5, linestyle="--",
                    label=f"Threshold = {threshold:.4f}")
    axes[1].fill_between(results_df.index,
                         results_df["reconstruction_error"], threshold,
                         where=results_df["reconstruction_error"] > threshold,
                         color="#FCA5A5", alpha=0.6)
    axes[1].set_title("Reconstruction Error (MSE)")
    axes[1].set_xlabel("Timestamp")
    axes[1].set_ylabel("MSE")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{building}_02_anomaly_on_energy.png", dpi=150)
    plt.close()

    # ── Plot 3: error distribution ──
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(train_errors, bins=60, color="#3B82F6",
            alpha=0.7, label="Train errors", density=True)
    ax.hist(all_errors,   bins=60, color="#F97316",
            alpha=0.5, label="All errors",   density=True)
    ax.axvline(threshold, color="#DC2626", lw=2, linestyle="--",
               label=f"Threshold = {threshold:.4f}")
    ax.set_title(f"Error Distribution — {building}", fontweight="bold")
    ax.set_xlabel("MSE")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{building}_03_error_distribution.png", dpi=150)
    plt.close()

    print(f"  Plots saved → {output_dir}/{building}_01/02/03_*.png")


def export_report(results_df, threshold, building, output_dir):
    """
    Save results CSV and print summary statistics.
    """
    csv_path = f"{output_dir}/{building}_anomaly_report.csv"
    results_df.to_csv(csv_path)

    total      = len(results_df)
    n_anomaly  = results_df["anomaly"].sum()
    test_df    = results_df[results_df["split"] == "test"]
    n_test_anom = test_df["anomaly"].sum()

    print(f"  Threshold         : {threshold:.6f}")
    print(f"  Total windows     : {total}")
    print(f"  Anomalies (all)   : {n_anomaly}  ({100*n_anomaly/total:.2f} %)")
    print(f"  Anomalies (test)  : {n_test_anom}  ({100*n_test_anom/len(test_df):.2f} %)")
    print(f"  Report saved      : {csv_path}")


def main():
    for building in BUILDINGS:

        print(f"\n{'='*50}")
        print(f"Testing: {building}")

        model_path = f"{CONFIG['model_dir']}/{building}_best.h5"
        if not os.path.exists(model_path):
            print(f"  [SKIP] No model found at {model_path}")
            continue
        
        # ── load saved model ──
        model = load_model(model_path, compile=False)
        model.compile(optimizer="adam", loss="mse")
        print(f"  Model loaded : {model_path}")

        

    # Dataset importing
    df_wide = pd.read_csv(CONFIG["data_url"])
    df_wide["timestamp"] = pd.to_datetime(df_wide["timestamp"], utc=True)
    df_wide = df_wide.set_index("timestamp")
    print(df_wide.head())
    print(f"Dataset shape: {df_wide.shape}")
    print(f"Total buildings: {len(df_wide.columns)}")

    META_PATH = CONFIG["meta_url"]
    meta = pd.read_csv(META_PATH)

    building_data = {}

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

    y_pred = model.predict(building_data[building]["X_test"], verbose=0)
    mse = np.mean(np.power(building_data[building]["X_test"] - y_pred, 2))

    X_all_seq = np.vstack([building_data[building]["X_train"], building_data[building]["X_test"]])
    train_errors = compute_reconstruction_errors(model, building_data[building]["X_train"])
    all_errors   = compute_reconstruction_errors(model, X_all_seq)

    threshold = compute_threshold(train_errors, CONFIG["threshold_sigma"])

    test_errors   = compute_reconstruction_errors(model, building_data[building]["X_test"])
    all_errors_c, anomaly_flags = detect_anomalies(
        train_errors, test_errors, threshold
    )

    print(f"\n  Results for {building}:")

    # ── build results DataFrame ──
    results_df = build_results_df(
        df_features  = building_data[building]["df"],
        all_errors   = all_errors,
        anomaly_flags= anomaly_flags,
        window_size  = CONFIG["Window_SIZE"],
        split_idx    = building_data[building]["split_idx"]
    )

    # ── plots + report ──
    save_plots(results_df, train_errors, threshold, building, CONFIG["output_dir"])
    export_report(results_df, threshold, building, CONFIG["output_dir"])

    print(f"\n{'='*50}")
    print(f"Testing complete. Outputs saved to: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()