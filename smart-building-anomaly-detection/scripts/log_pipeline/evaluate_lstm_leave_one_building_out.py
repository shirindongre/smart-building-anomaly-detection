"""
Leave-one-building-out (LOBO) evaluation for log LSTM classifier.

For each building in sequence_manifest.json:
  - train on all other buildings
  - validate on split of training buildings
  - test on the held-out building

Saves:
  - output_plots/log_lstm_lobo/log_lstm_lobo_metrics.json
  - output_plots/log_lstm_lobo/log_lstm_lobo_summary.csv
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

PROJECT_ROOT = _SCRIPTS_DIR.parent

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import LSTM, Dense, Dropout, Input
from keras.models import Sequential
from sklearn.utils.class_weight import compute_class_weight

from log_pipeline.metrics_utils import (
    class_balance_stats,
    compute_classification_metrics,
    evaluate_baselines,
    find_best_threshold_by_f1,
)
from log_pipeline.sequences import split_sequence_indices_temporal


RUN_CONFIG: Dict[str, object] = {
    "seed": 42,
    "manifest_path": PROJECT_ROOT / "data" / "processed" / "log_sequences" / "sequence_manifest.json",
    "val_size": 0.15,
    "epochs": 5,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "lstm_units_1": 32,
    "lstm_units_2": 16,
    "dropout": 0.2,
    "decision_threshold": 0.5,
    "output_dir": PROJECT_ROOT / "output_plots" / "log_lstm_lobo",
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


def load_per_building_sequences(manifest_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    per_file = payload.get("per_file", [])
    if not per_file:
        raise ValueError("Manifest has no per_file entries.")
    data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for item in per_file:
        bld = _building_from_source_csv(item["source_csv"])
        X = np.load(Path(item["X_path"]), mmap_mode="r")
        y = np.load(Path(item["y_path"]), mmap_mode="r")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"{bld}: X/y mismatch ({X.shape[0]} vs {y.shape[0]})")
        data[bld] = (X, y)
    return data


def concat_blocks(blocks: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.concatenate([b[0] for b in blocks], axis=0)
    y = np.concatenate([b[1] for b in blocks], axis=0)
    return X, y


def build_model(input_shape: Tuple[int, int], cfg: Dict[str, object]) -> Sequential:
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(int(cfg["lstm_units_1"]), return_sequences=True),
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


def main() -> None:
    cfg = RUN_CONFIG
    set_seed(int(cfg["seed"]))
    manifest_path = Path(cfg["manifest_path"])
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    all_data = load_per_building_sequences(manifest_path)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    sequence_length = int(manifest_payload.get("sequence_length", 10))
    purge_gap = max(1, sequence_length - 1)
    buildings = sorted(all_data.keys())
    if len(buildings) < 2:
        raise ValueError("Need at least 2 buildings for LOBO evaluation.")

    threshold_fixed = float(cfg["decision_threshold"])
    rows_fixed: List[Dict[str, object]] = []
    rows_tuned: List[Dict[str, object]] = []
    baseline_rows: List[Dict[str, object]] = []

    for holdout in buildings:
        train_buildings = [b for b in buildings if b != holdout]
        train_blocks: List[Tuple[np.ndarray, np.ndarray]] = []
        val_blocks: List[Tuple[np.ndarray, np.ndarray]] = []
        split_audit: Dict[str, Dict[str, int]] = {}
        val_ratio = float(cfg["val_size"])

        for b in train_buildings:
            Xb, yb = all_data[b]
            split = split_sequence_indices_temporal(
                len(yb),
                train_ratio=max(0.0, 1.0 - val_ratio),
                val_ratio=val_ratio,
                test_ratio=0.0,
                purge_gap=purge_gap,
            )
            tr_idx = split["train"]
            va_idx = split["val"]
            train_blocks.append((Xb[tr_idx], yb[tr_idx]))
            val_blocks.append((Xb[va_idx], yb[va_idx]))
            split_audit[b] = {"train_sequences": int(len(tr_idx)), "val_sequences": int(len(va_idx))}

        X_train, y_train = concat_blocks(train_blocks)
        X_val, y_val = concat_blocks(val_blocks)
        X_test, y_test = all_data[holdout]

        classes = np.array([0, 1], dtype=np.int32)
        w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weight = {0: float(w[0]), 1: float(w[1])}

        model = build_model((X_train.shape[1], X_train.shape[2]), cfg)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=0),
        ]
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=int(cfg["epochs"]),
            batch_size=int(cfg["batch_size"]),
            class_weight=class_weight,
            callbacks=callbacks,
            shuffle=True,
            verbose=1,
        )

        y_val_prob = model.predict(X_val, verbose=0).reshape(-1)
        y_test_prob = model.predict(X_test, verbose=0).reshape(-1)

        best_t, _val_m, _ = find_best_threshold_by_f1(y_val, y_val_prob, min_t=0.2)

        m_fixed = compute_classification_metrics(y_test, y_test_prob, threshold_fixed)
        m_tuned = compute_classification_metrics(y_test, y_test_prob, best_t)
        baselines = evaluate_baselines(
            X_train,
            y_train,
            X_test,
            y_test,
            threshold=threshold_fixed,
            random_state=int(cfg["seed"]),
        )

        row_base = {
            "holdout_building": holdout,
            "num_sequences": int(len(y_test)),
            "positive_rate": float(y_test.mean()),
            "threshold_val_f1": best_t,
            "purge_gap_sequences": purge_gap,
        }
        rows_fixed.append({**row_base, **{f"metric_{k}": v for k, v in m_fixed.items()}})
        rows_tuned.append({**row_base, **{f"metric_{k}": v for k, v in m_tuned.items()}})
        baseline_rows.append(
            {
                "holdout_building": holdout,
                "positive_rate": float(y_test.mean()),
                "class_balance_test": class_balance_stats(y_test),
                "baseline_logistic_regression": baselines["logistic_regression"],
                "baseline_random_forest": baselines["random_forest"],
            }
        )
        print(f"Holdout {holdout} @0.5: {m_fixed} | @val_F1({best_t:.4f}): {m_tuned}")
        print(f"  split audit (train buildings): {split_audit}")

    df_fixed = pd.DataFrame(rows_fixed)
    df_tuned = pd.DataFrame(rows_tuned)

    def _summarize(df: pd.DataFrame) -> Dict[str, object]:
        cols = [c for c in df.columns if c.startswith("metric_")]
        out: Dict[str, object] = {}
        for c in cols:
            key = c.replace("metric_", "")
            out[key] = {
                "mean": float(df[c].mean()),
                "std": float(df[c].std(ddof=0)),
            }
        return out

    output = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()},
        "buildings": buildings,
        "leakage_checks": {
            "split_strategy": "LOBO + temporal train/val split per non-holdout building with purge gap",
            "purge_gap_sequences": purge_gap,
            "holdout_building_excluded_from_train_val": True,
        },
        "threshold_tuning_note": "Per fold: threshold maximizes F1 on validation (train-building split only).",
        "per_building_threshold_0.5": rows_fixed,
        "per_building_threshold_val_f1": rows_tuned,
        "per_building_baselines_threshold_0.5": baseline_rows,
        "summary_threshold_0.5": _summarize(df_fixed),
        "summary_threshold_val_f1": _summarize(df_tuned),
    }

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "log_lstm_lobo_metrics.json"
    csv_path_fixed = out_dir / "log_lstm_lobo_summary_threshold_0.5.csv"
    csv_path_tuned = out_dir / "log_lstm_lobo_summary_threshold_val_f1.csv"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    df_fixed.to_csv(csv_path_fixed, index=False)
    df_tuned.to_csv(csv_path_tuned, index=False)

    print("\nSaved LOBO outputs:")
    print(f"  {json_path}")
    print(f"  {csv_path_fixed}")
    print(f"  {csv_path_tuned}")
    print(f"Summary @0.5: {output['summary_threshold_0.5']}")
    print(f"Summary @val F1: {output['summary_threshold_val_f1']}")


if __name__ == "__main__":
    main()
