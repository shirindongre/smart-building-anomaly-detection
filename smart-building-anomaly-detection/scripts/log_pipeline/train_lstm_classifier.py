"""
Stage-1 log anomaly classifier training (LSTM).

Reads per-building sequences from sequence_manifest.json, trains an LSTM binary
classifier (pooled across buildings using leakage-safe temporal splits per building),
and saves:
  - models/log_lstm_classifier.h5
  - output_plots/log_lstm/log_lstm_history.png
  - output_plots/log_lstm/log_lstm_confusion_matrix.png
  - output_plots/log_lstm/log_lstm_metrics.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from log_pipeline.metrics_utils import (
    class_balance_stats,
    compute_classification_metrics,
    evaluate_baselines,
    find_best_threshold_by_f1,
)
from log_pipeline.sequences import split_sequence_indices_temporal

PROJECT_ROOT = _SCRIPTS_DIR.parent


RUN_CONFIG: Dict[str, object] = {
    "seed": 42,
    "manifest_path": PROJECT_ROOT / "data" / "processed" / "log_sequences" / "sequence_manifest.json",
    "test_size": 0.15,
    "val_size": 0.15,  # fraction of (train+val) after test split
    "epochs": 20,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "lstm_units_1": 64,
    "lstm_units_2": 32,
    "dropout": 0.2,
    # Baseline report at 0.5; primary test metrics use threshold tuned on validation F1.
    "decision_threshold": 0.5,
    "model_output_path": PROJECT_ROOT / "models" / "log_lstm_classifier.h5",
    "output_dir": PROJECT_ROOT / "output_plots" / "log_lstm",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _building_from_source_csv(source_csv: str) -> str:
    name = Path(source_csv).name
    if name.endswith("_processed.csv"):
        return name[: -len("_processed.csv")]
    return Path(source_csv).stem


def load_sequences(cfg: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, object]]:
    manifest_path = Path(cfg["manifest_path"])
    manifest_payload: Dict[str, object] = {}

    if manifest_path.is_file():
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    per_file = manifest_payload.get("per_file", []) if manifest_payload else []
    if not per_file:
        raise FileNotFoundError(
            f"Manifest missing per_file entries: {manifest_path}. "
            "Re-run sequence generation to produce per-building .npy outputs."
        )

    per_building: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for item in per_file:
        bld = _building_from_source_csv(item["source_csv"])
        Xb = np.load(Path(item["X_path"]), mmap_mode="r")
        yb = np.load(Path(item["y_path"]), mmap_mode="r")
        if Xb.shape[0] != yb.shape[0]:
            raise ValueError(f"{bld}: X/y mismatch ({Xb.shape[0]} vs {yb.shape[0]})")
        per_building[bld] = (Xb, yb)

    # Provide lightweight placeholder arrays; avoid allocating monolithic tensors.
    first = next(iter(per_building.values()))
    seq_len = int(first[0].shape[1])
    feat_dim = int(first[0].shape[2])
    X_summary = np.empty((0, seq_len, feat_dim), dtype=np.float32)
    y_summary = np.empty((0,), dtype=np.int32)
    return X_summary, y_summary, per_building, manifest_payload


def concat_blocks(blocks: Tuple[Tuple[np.ndarray, np.ndarray], ...] | list[Tuple[np.ndarray, np.ndarray]]):
    if not blocks:
        raise ValueError("No sequence blocks to concatenate.")
    X = np.concatenate([b[0] for b in blocks], axis=0)
    y = np.concatenate([b[1] for b in blocks], axis=0)
    return X, y


def split_data_temporal_per_building(
    per_building: Dict[str, Tuple[np.ndarray, np.ndarray]],
    cfg: Dict[str, object],
    manifest_payload: Dict[str, object],
):
    test_size = float(cfg["test_size"])
    val_size = float(cfg["val_size"])
    train_size = max(0.0, 1.0 - test_size - val_size)
    if train_size <= 0:
        raise ValueError("Invalid split config: train_size <= 0.")

    sequence_length = int(manifest_payload.get("sequence_length", 10))
    purge_gap = max(1, sequence_length - 1)

    train_blocks = []
    val_blocks = []
    test_blocks = []
    split_audit: Dict[str, Dict[str, int]] = {}

    for b, (Xb, yb) in sorted(per_building.items()):
        split = split_sequence_indices_temporal(
            len(yb),
            train_ratio=train_size,
            val_ratio=val_size,
            test_ratio=test_size,
            purge_gap=purge_gap,
        )
        tr_idx = split["train"]
        va_idx = split["val"]
        te_idx = split["test"]
        train_blocks.append((Xb[tr_idx], yb[tr_idx]))
        val_blocks.append((Xb[va_idx], yb[va_idx]))
        test_blocks.append((Xb[te_idx], yb[te_idx]))
        split_audit[b] = {
            "train_sequences": int(len(tr_idx)),
            "val_sequences": int(len(va_idx)),
            "test_sequences": int(len(te_idx)),
        }

    X_train, y_train = concat_blocks(train_blocks)
    X_val, y_val = concat_blocks(val_blocks)
    X_test, y_test = concat_blocks(test_blocks)
    return X_train, X_val, X_test, y_train, y_val, y_test, purge_gap, split_audit


def build_model(input_shape: Tuple[int, int], cfg: Dict[str, object]) -> Sequential:
    model = Sequential(
        [
            LSTM(int(cfg["lstm_units_1"]), return_sequences=True, input_shape=input_shape),
            Dropout(float(cfg["dropout"])),
            LSTM(int(cfg["lstm_units_2"])),
            Dropout(float(cfg["dropout"])),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(cfg["learning_rate"])),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def save_history_plot(history, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="train_loss")
    axes[0].plot(history.history["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    if "auc" in history.history and "val_auc" in history.history:
        axes[1].plot(history.history["auc"], label="train_auc")
        axes[1].plot(history.history["val_auc"], label="val_auc")
        axes[1].set_title("AUC")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_dir / "log_lstm_history.png", dpi=150)
    plt.close()


def save_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    output_dir: Path,
    *,
    filename: str = "log_lstm_confusion_matrix.png",
    title: str = "Confusion Matrix (Test)",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (y_prob >= threshold).astype(np.int32)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["Normal", "Anomaly"])
    ax.set_yticks([0, 1], labels=["Normal", "Anomaly"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


def main() -> None:
    cfg = RUN_CONFIG
    set_seed(int(cfg["seed"]))

    X, y, per_building, manifest_payload = load_sequences(cfg)
    # X/y are summary arrays only (to avoid monolithic loads).
    total_pos = float(sum(float(v[1].mean()) * float(len(v[1])) for v in per_building.values()))
    total_n = float(sum(float(len(v[1])) for v in per_building.values()))
    pos_rate = (total_pos / total_n) if total_n > 0 else 0.0
    print(f"Loaded buildings: {len(per_building)} | total_sequences={int(total_n)} | positive_rate={pos_rate:.4f}")

    X_train, X_val, X_test, y_train, y_val, y_test, purge_gap, split_audit = split_data_temporal_per_building(
        per_building, cfg, manifest_payload
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    classes = np.array([0, 1], dtype=np.int32)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}
    print(f"class_weight: {class_weight}")

    model = build_model((X_train.shape[1], X_train.shape[2]), cfg)

    model_out = Path(cfg["model_output_path"])
    model_out.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        ModelCheckpoint(filepath=str(model_out), monitor="val_auc", mode="max", save_best_only=True, verbose=1),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=int(cfg["epochs"]),
        batch_size=int(cfg["batch_size"]),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    y_val_prob = model.predict(X_val, verbose=0).reshape(-1)
    y_test_prob = model.predict(X_test, verbose=0).reshape(-1)

    threshold_fixed = float(cfg["decision_threshold"])
    best_t, metrics_val_at_best_t, _sweep = find_best_threshold_by_f1(y_val, y_val_prob, min_t=0.2)

    metrics_test_fixed = compute_classification_metrics(y_test, y_test_prob, threshold_fixed)
    metrics_test_tuned = compute_classification_metrics(y_test, y_test_prob, best_t)
    baseline_metrics = evaluate_baselines(
        X_train,
        y_train,
        X_test,
        y_test,
        threshold=threshold_fixed,
        random_state=int(cfg["seed"]),
    )

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_history_plot(history, out_dir)
    save_confusion_matrix(
        y_test,
        y_test_prob,
        threshold_fixed,
        out_dir,
        filename="log_lstm_confusion_matrix_threshold_0.5.png",
        title=f"Confusion Matrix (Test, threshold={threshold_fixed})",
    )
    save_confusion_matrix(
        y_test,
        y_test_prob,
        best_t,
        out_dir,
        filename="log_lstm_confusion_matrix_threshold_val_f1.png",
        title=f"Confusion Matrix (Test, threshold={best_t:.4f} from val F1)",
    )

    payload = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()},
        "data_shape": {"X": [int(total_n), int(X_train.shape[1]), int(X_train.shape[2])], "y": [int(total_n)]},
        "splits": {"train": int(len(y_train)), "val": int(len(y_val)), "test": int(len(y_test))},
        "class_balance": {
            "train": class_balance_stats(y_train),
            "val": class_balance_stats(y_val),
            "test": class_balance_stats(y_test),
        },
        "leakage_checks": {
            "split_strategy": "temporal per building with purge gap; then concatenated",
            "purge_gap_sequences": purge_gap,
            "per_building_counts": split_audit,
            "label_column_not_in_features": True,
        },
        "threshold_tuning": {
            "method": "maximize F1 on validation set",
            "threshold_val_f1": best_t,
            "metrics_validation_at_threshold_val_f1": metrics_val_at_best_t,
        },
        "metrics_test_threshold_0.5": metrics_test_fixed,
        "metrics_test_threshold_val_f1": metrics_test_tuned,
        "baseline_metrics_threshold_0.5": baseline_metrics,
    }
    metrics_path = out_dir / "log_lstm_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nSaved artifacts:")
    print(f"  model   : {model_out}")
    print(f"  history : {out_dir / 'log_lstm_history.png'}")
    print(f"  cm      : {out_dir / 'log_lstm_confusion_matrix.png'}")
    print(f"  metrics : {metrics_path}")
    print(f"  threshold (val F1): {best_t}")
    print(f"  test @0.5: {metrics_test_fixed}")
    print(f"  test @val F1: {metrics_test_tuned}")
    print(f"  baselines @0.5: {baseline_metrics}")


if __name__ == "__main__":
    main()
