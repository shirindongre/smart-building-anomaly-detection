"""
Per-building inference: load one processed CSV (energy + engineered features),
apply the saved scaler, build sequences, and run the trained LSTM autoencoder.
No wide multi-building download and no preprocess_building / engineer_features here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import joblib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    joblib = None
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_PLOTS_BASE_DIR = "output_plots/sensor_lstm"


def _sensor_lstm_building_plots_dir(building: str) -> str:
    """Per-building directory for sensor LSTM figures under the project root."""
    return os.path.join(str(_PROJECT_ROOT), _PLOTS_BASE_DIR, building)


_EXPECTED_VENV_PY = _PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
try:
    _RUNNING_PY = Path(sys.executable).resolve()
except Exception:
    _RUNNING_PY = Path(sys.executable)
if _EXPECTED_VENV_PY.exists() and _RUNNING_PY != _EXPECTED_VENV_PY.resolve():
    print(
        "[WARN] You are not running the project virtualenv interpreter.\n"
        f"       Running:  {_RUNNING_PY}\n"
        f"       Expected: {_EXPECTED_VENV_PY}\n"
        "       Fix: run with the venv python, e.g.\n"
        f"         & \"{_EXPECTED_VENV_PY}\" \"{Path(__file__).resolve()}\""
    )
for p in (_PROJECT_ROOT, _SCRIPT_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from src.config import BUILDINGS, CONFIG, FEATURE_COLS

from sequence_creations import create_sequences

os.makedirs(_PROJECT_ROOT / CONFIG["output_dir"], exist_ok=True)


def load_scaler(path: Path):
    """
    Load a scaler saved as a .pkl.

    Uses joblib when available; falls back to pickle to avoid env mismatch issues.
    """
    if joblib is not None:
        return joblib.load(path)

    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def _resolve_building_features_path(building: str) -> Path:
    """Prefer ``{building}.csv``, then ``{building}_features.csv`` under data/processed/."""
    processed = _PROJECT_ROOT / "data" / "processed"
    for name in (f"{building}.csv", f"{building}_features.csv"):
        candidate = processed / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No processed file for {building!r} in {processed}. "
        f"Expected {building}.csv or {building}_features.csv (run training export or pipeline)."
    )


def load_building_features(building: str) -> pd.DataFrame:
    """Load single-building table: timestamp + energy + feature columns."""
    path = _resolve_building_features_path(building)
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path}: missing 'timestamp' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing FEATURE_COLS {missing}")
    return df


def compute_reconstruction_errors(model, sequences: np.ndarray) -> np.ndarray:
    reconstructed = model.predict(sequences, verbose=0)
    errors = np.mean(np.power(sequences - reconstructed, 2), axis=(1, 2))
    return errors


def _rolling_mad(values: np.ndarray) -> float:
    """MAD within one window: median(|x - median(x)|). Used with rolling(..., apply, raw=True)."""
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def add_adaptive_thresholding(
    results_df: pd.DataFrame,
    *,
    threshold_method: str = "rolling_percentile",
    rolling_window_days: int = 7,
    robust_z_k: float = 3.5,
    anomaly_rate_drift_threshold: float = 0.30,
) -> pd.DataFrame:
    """
    Post-process reconstruction errors: robust normalization, drift vs anomaly separation,
    smoothed MAD-based threshold with freeze-during-drift.

    ``threshold_method``, ``rolling_window_days``, ``robust_z_k``, and
    ``anomaly_rate_drift_threshold`` are kept for call-site compatibility; tunables are read
    from ``CONFIG`` (see ``error_rolling_window_hours``, ``threshold_mad_k``, etc.).

    No future leakage: rolling statistics on errors use only past data (closed='left').
    """
    if results_df.index.name is None:
        results_df = results_df.copy()
        results_df.index.name = "timestamp"

    if not isinstance(results_df.index, pd.DatetimeIndex):
        raise ValueError("results_df must be indexed by timestamp (DatetimeIndex)")

    if "reconstruction_error" not in results_df.columns:
        raise ValueError("results_df missing required column 'reconstruction_error'")

    df = results_df.sort_index().copy()
    err = df["reconstruction_error"]

    # --- 1) Rolling robust location / scale on reconstruction_error (calendar / time window) ---
    hours = int(CONFIG.get("error_rolling_window_hours", 168))
    roll_spec = f"{hours}h"
    eps = float(CONFIG.get("mad_epsilon", 1e-6))
    past = err.rolling(roll_spec, min_periods=1, closed="left")

    rolling_median = past.median()
    rolling_mad = past.apply(_rolling_mad, raw=True)
    # Legacy diagnostic column (unchanged name) for CSV consumers that expect rolling_p99.
    rolling_p99 = past.quantile(0.99)

    # Early rows: no past window yet — fall back so downstream math stays finite.
    rolling_median = rolling_median.fillna(err)
    rolling_mad = rolling_mad.fillna(0.0)
    rolling_p99 = rolling_p99.fillna(err)

    df["rolling_median"] = rolling_median.astype(float)
    df["rolling_mad"] = rolling_mad.astype(float)
    df["rolling_p99"] = rolling_p99.astype(float)

    # --- 2) Stabilized robust z-score (same rolling median / MAD + epsilon) ---
    df["normalized_error"] = ((err - df["rolling_median"]) / (df["rolling_mad"] + eps)).astype(float)

    # --- 3) Drift from normalized score only (independent of threshold; no leakage into drift rule) ---
    drift_z = float(CONFIG.get("drift_z_threshold", 5.0))
    sustain = int(CONFIG.get("drift_sustain_points", 24))
    high_norm = (df["normalized_error"] > drift_z).astype(np.int8)
    # Sustained elevation: last ``sustain`` points must all exceed drift_z.
    sustained_ct = high_norm.rolling(sustain, min_periods=sustain).sum()
    df["drift_flag"] = (sustained_ct >= sustain).astype(int)

    # --- 4) Threshold: median + k * MAD, then EMA smooth ---
    k = float(CONFIG.get("threshold_mad_k", 6.0))
    ema_span = int(CONFIG.get("threshold_ema_span", 24))
    threshold_raw = df["rolling_median"] + k * df["rolling_mad"]
    threshold_smooth = threshold_raw.ewm(span=ema_span, adjust=False).mean()

    # --- 5) Freeze threshold while drift_flag is active (carry last pre-drift smoothed value) ---
    thr_carry = threshold_smooth.mask(df["drift_flag"].astype(bool)).ffill().bfill()
    df["threshold"] = threshold_smooth.where(df["drift_flag"] == 0, thr_carry).astype(float)

    # --- 6) Anomalies only when error exceeds threshold and we are not in a drift regime ---
    df["anomaly_flag"] = ((err > df["threshold"]) & (df["drift_flag"] == 0)).astype(int)

    return df


def build_results_df(df_features, all_errors, window_size, split_idx):
    time_index = df_features.index[window_size - 1 : window_size - 1 + len(all_errors)]

    results_df = pd.DataFrame(
        {
            "energy": df_features["energy"].values[
                window_size - 1 : window_size - 1 + len(all_errors)
            ],
            "reconstruction_error": all_errors,
            "split": [
                "train" if i < (split_idx - window_size + 1) else "test"
                for i in range(len(all_errors))
            ],
        },
        index=time_index,
    )

    return results_df


def _iter_true_segments(mask: pd.Series):
    idx = mask.index
    m = mask.to_numpy(dtype=bool, copy=False)
    n = len(m)
    i = 0
    while i < n:
        if not m[i]:
            i += 1
            continue
        start = idx[i]
        j = i + 1
        while j < n and m[j]:
            j += 1
        end = idx[j - 1]
        yield start, end
        i = j


def _shade_mask_regions(ax, mask: pd.Series, *, color: str, alpha: float, label: str):
    first = True
    for start, end in _iter_true_segments(mask):
        ax.axvspan(start, end, color=color, alpha=alpha, lw=0, label=(label if first else None))
        first = False


def save_plots(
    results_df,
    train_errors,
    building,
    plots_output_dir,
):
    """Sensor LSTM diagnostic figures using ``threshold``, ``drift_flag``, ``normalized_error``."""
    err = results_df["reconstruction_error"]
    thr = results_df["threshold"]
    anomaly_mask = results_df["anomaly_flag"] == 1
    normal_mask = results_df["anomaly_flag"] == 0
    drift_mask = results_df["drift_flag"] == 1
    drift_z = float(CONFIG.get("drift_z_threshold", 5.0))
    plots_dir = os.path.normpath(plots_output_dir)
    os.makedirs(plots_dir, exist_ok=True)

    # --- 01: error vs smoothed/frozen threshold, drift bands, anomaly markers ---
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(results_df.index, err, color="#6B7280", lw=0.8, label="Reconstruction error", alpha=0.85)
    _shade_mask_regions(ax, drift_mask, color="#93C5FD", alpha=0.25, label="Drift region")
    ax.plot(
        results_df.index,
        thr,
        color="#DC2626",
        lw=1.6,
        linestyle="-",
        label="Threshold (median + k*MAD, EMA, drift-frozen)",
    )
    over_thr_clear = (err > thr) & (results_df["drift_flag"] == 0)
    ax.fill_between(
        results_df.index,
        err,
        thr,
        where=over_thr_clear,
        color="#FCA5A5",
        alpha=0.45,
        label="Above threshold (non-drift)",
    )
    ax.scatter(
        results_df.index[anomaly_mask],
        err[anomaly_mask],
        color="#B91C1C",
        s=22,
        zorder=5,
        label="Anomaly",
    )
    ax.set_title(f"Reconstruction Error — {building}", fontweight="bold")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "01_error_timeline.png"), dpi=150)
    plt.close()

    # --- 02: energy + error with same threshold context ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(
        results_df.index[normal_mask],
        results_df["energy"][normal_mask],
        color="#6B7280",
        lw=0.8,
        label="Normal",
        alpha=0.9,
    )
    axes[0].scatter(
        results_df.index[anomaly_mask],
        results_df["energy"][anomaly_mask],
        color="#DC2626",
        s=18,
        zorder=5,
        label="Anomaly",
    )
    _shade_mask_regions(axes[0], drift_mask, color="#93C5FD", alpha=0.20, label="Drift region")
    axes[0].set_title(
        f"Energy Consumption — Anomalies Highlighted — {building}", fontweight="bold"
    )
    axes[0].set_ylabel("Energy (scaled)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(results_df.index, err, color="#6B7280", lw=0.8, alpha=0.85, label="Reconstruction error")
    _shade_mask_regions(axes[1], drift_mask, color="#93C5FD", alpha=0.20, label="Drift region")
    axes[1].plot(
        results_df.index,
        thr,
        color="#DC2626",
        lw=1.4,
        label="Threshold",
    )
    axes[1].fill_between(
        results_df.index,
        err,
        thr,
        where=over_thr_clear,
        color="#FCA5A5",
        alpha=0.5,
    )
    axes[1].scatter(
        results_df.index[anomaly_mask],
        err[anomaly_mask],
        color="#B91C1C",
        s=18,
        zorder=5,
        label="Anomaly",
    )
    axes[1].set_title("Reconstruction Error (MSE)")
    axes[1].set_xlabel("Timestamp")
    axes[1].set_ylabel("MSE")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "02_anomaly_on_energy.png"), dpi=150)
    plt.close()

    # --- 03: normalized robust z (stabilized MAD) vs drift gate level ---
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(
        results_df.index,
        results_df["normalized_error"],
        color="#6B7280",
        lw=0.8,
        alpha=0.85,
        label="Normalized error (robust z)",
    )
    _shade_mask_regions(ax, drift_mask, color="#93C5FD", alpha=0.25, label="Drift region")
    ax.axhline(drift_z, color="#DC2626", lw=1.8, linestyle="--", label=f"Drift gate (z = {drift_z:g})")
    ax.fill_between(
        results_df.index,
        results_df["normalized_error"],
        drift_z,
        where=results_df["normalized_error"] > drift_z,
        color="#FDE68A",
        alpha=0.35,
        label="Above drift gate",
    )
    ax.set_title(f"Normalized Error — {building}", fontweight="bold")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("z-score (median / (MAD + ε))")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "03_normalized_error.png"), dpi=150)
    plt.close()

    # --- 04: error distribution vs median applied threshold ---
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(train_errors, bins=60, color="#3B82F6", alpha=0.7, label="Train errors", density=True)
    ax.hist(err.values, bins=60, color="#F97316", alpha=0.5, label="All errors", density=True)
    ax.axvline(float(thr.median()), color="#DC2626", lw=2, linestyle="--", label="Median threshold")
    ax.set_title(f"Error Distribution — {building}", fontweight="bold")
    ax.set_xlabel("MSE")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "04_error_distribution.png"), dpi=150)
    plt.close()

    print(f"  Plots saved -> {plots_dir} (01-04_*.png)")


def print_diagnostics(building: str, df: pd.DataFrame):
    err = df["reconstruction_error"]
    ar = float(df["anomaly_flag"].mean())
    dr = float(df["drift_flag"].mean()) if "drift_flag" in df.columns else float("nan")

    print(f"\n=== Diagnostics: {building} ===")
    print(f"Anomaly rate: {ar:.4f}")
    print(f"Drift rate: {dr:.4f}")
    print(f"Mean error: {float(err.mean()):.6g}")
    print(f"Std error: {float(err.std()):.6g}")

    if ar < 0.01:
        print("Note: anomaly rate < 1% (threshold may be strict).")
    if ar > 0.10:
        print("Note: anomaly rate > 10% (threshold may be loose).")


def export_report(results_df, building, output_dir, *, threshold_method: str, robust_z_k: float):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{building}_anomaly_report.csv"
    results_df.to_csv(csv_path)

    total = len(results_df)
    n_anomaly = results_df["anomaly_flag"].sum()
    test_df = results_df[results_df["split"] == "test"]
    n_test_anom = test_df["anomaly_flag"].sum()

    print(
        "  Threshold policy : median + k·MAD (k="
        f"{float(CONFIG.get('threshold_mad_k', 6)):g}), EMA span="
        f"{int(CONFIG.get('threshold_ema_span', 24))}, drift-frozen; drift_gate_z="
        f"{float(CONFIG.get('drift_z_threshold', 5)):g}"
    )
    print(f"  Total windows     : {total}")
    print(f"  Anomalies (all)   : {n_anomaly}  ({100 * n_anomaly / total:.2f} %)")
    print(
        f"  Anomalies (test)  : {n_test_anom}  ({100 * n_test_anom / len(test_df):.2f} %)"
    )
    print(f"  Report saved      : {csv_path}")


def export_sensor_outputs(results_df, building):
    out_dir = _PROJECT_ROOT / "data" / "processed" / "sensor_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = results_df.reset_index()
    if "timestamp" not in df.columns:
        df = df.rename(columns={df.columns[0]: "timestamp"})
    df["anomaly_score"] = df["reconstruction_error"]
    df["building_id"] = building
    df = df[
        [
            "timestamp",
            "building_id",
            "reconstruction_error",
            "anomaly_score",
            "anomaly_flag",
            "normalized_error",
            "drift_flag",
        ]
    ]
    save_path = out_dir / f"{building}_anomalies.csv"
    df.to_csv(save_path, index=False)
    print(f"  Sensor output saved -> {save_path}")


def main():
    output_dir = _PROJECT_ROOT / CONFIG["output_dir"]
    model_dir = _PROJECT_ROOT / CONFIG["model_dir"]
    scalers_dir = _PROJECT_ROOT / "scalers"

    threshold_method = CONFIG.get("threshold_method", "rolling_percentile")
    rolling_window_days = int(CONFIG.get("rolling_window_days", 7))
    robust_z_k = float(CONFIG.get("robust_z_k", 3.5))

    for building in BUILDINGS:
        print(f"\n{'=' * 50}")
        print(f"Testing: {building}")

        model_path = model_dir / f"{building}_best.h5"
        if not model_path.is_file():
            print(f"  [SKIP] No model found at {model_path}")
            continue

        try:
            df_features = load_building_features(building)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue
        except ValueError as e:
            print(f"  [SKIP] {e}")
            continue

        print(f"  Loaded features: {df_features.shape} (per-building CSV)")

        scaler_path = scalers_dir / f"{building}_scaler.pkl"
        if not scaler_path.is_file():
            print(f"  [SKIP] No scaler found at {scaler_path}")
            continue
        scaler = load_scaler(scaler_path)

        test_split = float(CONFIG.get("test_split", 0.2))
        split_idx = int(len(df_features) * (1.0 - test_split))
        window = int(CONFIG["window_size"])

        train_df = df_features.iloc[:split_idx]
        test_df = df_features.iloc[split_idx:]

        X_train_sc = scaler.transform(train_df[FEATURE_COLS].to_numpy())
        X_test_sc = scaler.transform(test_df[FEATURE_COLS].to_numpy())

        X_train_seq = create_sequences(X_train_sc, window)
        X_test_seq = create_sequences(X_test_sc, window)

        print(f"  Train windows  : {X_train_seq.shape}")
        print(f"  Test windows   : {X_test_seq.shape}")

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            print("  [SKIP] Not enough rows for at least one train/test window.")
            continue

        model = load_model(model_path, compile=False)
        model.compile(optimizer="adam", loss="mse")
        print(f"  Model loaded : {model_path}")

        X_all_seq = np.vstack((X_train_seq, X_test_seq))
        train_errors = compute_reconstruction_errors(model, X_train_seq)
        all_errors = compute_reconstruction_errors(model, X_all_seq)

        print(f"\n  Results for {building}:")

        results_df = build_results_df(
            df_features=df_features,
            all_errors=all_errors,
            window_size=window,
            split_idx=split_idx,
        )

        results_df = add_adaptive_thresholding(
            results_df,
            threshold_method=threshold_method,
            rolling_window_days=rolling_window_days,
            robust_z_k=robust_z_k,
        )

        print_diagnostics(building, results_df.reset_index())
        save_plots(
            results_df,
            train_errors,
            building,
            _sensor_lstm_building_plots_dir(building),
        )
        export_report(
            results_df,
            building,
            output_dir,
            threshold_method=threshold_method,
            robust_z_k=robust_z_k,
        )
        export_sensor_outputs(results_df, building)

    print(f"\n{'=' * 50}")
    plots_root = os.path.join(str(_PROJECT_ROOT), _PLOTS_BASE_DIR)
    print(
        f"Testing complete. Sensor LSTM plots under {plots_root}/<building>/; "
        f"reports under {output_dir}; sensor_outputs under data/processed/sensor_outputs/"
    )


if __name__ == "__main__":
    main()
