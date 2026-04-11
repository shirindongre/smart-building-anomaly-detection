"""
Stage-2 style evaluation: train on some buildings, test on unseen buildings.

Uses per-building sequence arrays from sequence_manifest.json, then:
  - train/val split inside train buildings only
  - evaluate on holdout buildings only
  - save model + metrics + plots
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM, Dense, Dropout, Input
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
    # If empty, script auto-selects the lexicographically-last building as holdout.
    "holdout_buildings": [],
    "val_size": 0.15,
    "epochs": 20,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "lstm_units_1": 64,
    "lstm_units_2": 32,
    "dropout": 0.2,
    "decision_threshold": 0.5,
    "model_output_path": PROJECT_ROOT / "models" / "log_lstm_holdout_classifier.h5",
    "output_dir": PROJECT_ROOT / "output_plots" / "log_lstm_holdout",
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
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    per_file = payload.get("per_file", [])
    if not per_file:
        raise ValueError("Manifest has no per_file entries. Re-run notebook 04.")

    data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for item in per_file:
        bld = _building_from_source_csv(item["source_csv"])
        x_path = Path(item["X_path"])
        y_path = Path(item["y_path"])
        if not x_path.is_file() or not y_path.is_file():
            raise FileNotFoundError(f"Missing X/y for {bld}: {x_path} | {y_path}")
        X = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"{bld}: X/y mismatch ({X.shape[0]} vs {y.shape[0]})")
        data[bld] = (X, y)
    return data


def _span_from_indices(idx: np.ndarray) -> Optional[Tuple[int, int]]:
    if len(idx) == 0:
        return None
    return int(idx[0]), int(idx[-1]) + 1


class SequenceSpanGenerator(tf.keras.utils.Sequence):
    """Memory-safe batch loader over per-building memmap arrays."""

    def __init__(
        self,
        data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        spans: Dict[str, Tuple[int, int]],
        *,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data = data
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self._rng = random.Random(seed)
        self._batches: List[Tuple[str, int, int]] = []
        for b in sorted(spans):
            start, stop = spans[b]
            for s in range(start, stop, self.batch_size):
                e = min(s + self.batch_size, stop)
                self._batches.append((b, s, e))
        if not self._batches:
            raise ValueError("No batches available for generator.")
        self.on_epoch_end()

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, index: int):
        b, s, e = self._batches[index]
        Xb, yb = self.data[b]
        X = np.asarray(Xb[s:e], dtype=np.float32)
        y = np.asarray(yb[s:e], dtype=np.int32)
        return X, y

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self._rng.shuffle(self._batches)


def _count_labels_in_spans(
    data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    spans: Dict[str, Tuple[int, int]],
) -> Tuple[int, int]:
    total = 0
    positives = 0
    for b, (start, stop) in spans.items():
        y = data[b][1][start:stop]
        total += int(stop - start)
        positives += int(np.count_nonzero(y))
    return total, positives


def _collect_labels_in_spans(
    data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    spans: Dict[str, Tuple[int, int]],
) -> np.ndarray:
    total = sum(int(stop - start) for start, stop in spans.values())
    out = np.empty((total,), dtype=np.int32)
    cursor = 0
    for b in sorted(spans):
        start, stop = spans[b]
        block = np.asarray(data[b][1][start:stop], dtype=np.int32)
        n = block.shape[0]
        out[cursor : cursor + n] = block
        cursor += n
    return out


def _collect_probs_for_spans(
    model: Sequential,
    data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    spans: Dict[str, Tuple[int, int]],
    *,
    batch_size: int,
) -> np.ndarray:
    gen = SequenceSpanGenerator(data, spans, batch_size=batch_size, shuffle=False, seed=0)
    return model.predict(gen, verbose=0).reshape(-1)


def _sample_arrays_from_spans(
    data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    spans: Dict[str, Tuple[int, int]],
    *,
    max_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect at most max_samples rows for baseline models."""
    rng = random.Random(seed)
    total = sum(int(stop - start) for start, stop in spans.values())
    take = min(int(max_samples), int(total))
    if take <= 0:
        raise ValueError("No samples available for baseline computation.")
    items = sorted(spans.items())
    feat_dim = int(data[items[0][0]][0].shape[2])
    seq_len = int(data[items[0][0]][0].shape[1])
    X_out = np.empty((take, seq_len, feat_dim), dtype=np.float32)
    y_out = np.empty((take,), dtype=np.int32)
    cursor = 0
    remaining = take
    for i, (b, (start, stop)) in enumerate(items):
        available = int(stop - start)
        if available <= 0:
            continue
        if i == len(items) - 1:
            n_take = remaining
        else:
            n_take = int(round(take * (available / total)))
            n_take = max(0, min(n_take, available, remaining))
        if n_take == 0:
            continue
        if n_take == available:
            idx_local = list(range(start, stop))
        else:
            idx_local = sorted(rng.sample(range(start, stop), n_take))
        Xb, yb = data[b]
        block_X = np.asarray(Xb[idx_local], dtype=np.float32)
        block_y = np.asarray(yb[idx_local], dtype=np.int32)
        n = block_X.shape[0]
        X_out[cursor : cursor + n] = block_X
        y_out[cursor : cursor + n] = block_y
        cursor += n
        remaining -= n
        if remaining <= 0:
            break
    return X_out[:cursor], y_out[:cursor]


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


def save_history_plot(history, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
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
    plt.savefig(out_dir / "log_lstm_holdout_history.png", dpi=150)
    plt.close()


def save_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    out_dir: Path,
    *,
    filename: str = "log_lstm_holdout_confusion_matrix.png",
    title: str = "Confusion Matrix (Holdout Buildings)",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
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
    plt.savefig(out_dir / filename, dpi=150)
    plt.close()


def main() -> None:
    cfg = RUN_CONFIG
    set_seed(int(cfg["seed"]))

    all_data = load_per_building_sequences(Path(cfg["manifest_path"]))
    all_buildings = sorted(all_data.keys())
    if len(all_buildings) < 2:
        raise ValueError("Need at least 2 buildings for holdout evaluation.")

    holdout = list(cfg["holdout_buildings"]) if cfg["holdout_buildings"] else [all_buildings[-1]]
    holdout = [b for b in holdout if b in all_data]
    if not holdout:
        raise ValueError("Configured holdout_buildings are not present in manifest data.")

    train_buildings = [b for b in all_buildings if b not in holdout]
    if not train_buildings:
        raise ValueError("No train buildings left after holdout selection.")

    sequence_length = int(json.loads(Path(cfg["manifest_path"]).read_text(encoding="utf-8")).get("sequence_length", 10))
    purge_gap = max(1, sequence_length - 1)

    train_spans: Dict[str, Tuple[int, int]] = {}
    val_spans: Dict[str, Tuple[int, int]] = {}
    holdout_spans: Dict[str, Tuple[int, int]] = {}
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
        tr_span = _span_from_indices(tr_idx)
        va_span = _span_from_indices(va_idx)
        if tr_span is None or va_span is None:
            raise ValueError(f"{b}: train/val span unexpectedly empty after temporal split.")
        train_spans[b] = tr_span
        val_spans[b] = va_span
        split_audit[b] = {"train_sequences": int(len(tr_idx)), "val_sequences": int(len(va_idx))}

    for b in holdout:
        yb = all_data[b][1]
        holdout_spans[b] = (0, int(yb.shape[0]))

    train_n, train_pos = _count_labels_in_spans(all_data, train_spans)
    val_n, val_pos = _count_labels_in_spans(all_data, val_spans)
    test_n, test_pos = _count_labels_in_spans(all_data, holdout_spans)

    print(f"Train buildings : {train_buildings}")
    print(f"Holdout buildings: {holdout}")
    print(
        "Train/Val/Test sequences: "
        f"{train_n} / {val_n} / {test_n} | "
        f"positive rates: {train_pos / max(1, train_n):.4f} / "
        f"{val_pos / max(1, val_n):.4f} / {test_pos / max(1, test_n):.4f}"
    )

    batch_size = int(cfg["batch_size"])
    train_gen = SequenceSpanGenerator(
        all_data, train_spans, batch_size=batch_size, shuffle=True, seed=int(cfg["seed"])
    )
    val_gen = SequenceSpanGenerator(
        all_data, val_spans, batch_size=batch_size, shuffle=False, seed=int(cfg["seed"])
    )

    neg = max(1, train_n - train_pos)
    pos = max(1, train_pos)
    class_weight = {0: float(train_n / (2.0 * neg)), 1: float(train_n / (2.0 * pos))}
    print(f"class_weight: {class_weight}")

    first_train_building = sorted(train_spans)[0]
    seq_len = int(all_data[first_train_building][0].shape[1])
    feat_dim = int(all_data[first_train_building][0].shape[2])
    model = build_model((seq_len, feat_dim), cfg)
    model_out = Path(cfg["model_output_path"])
    model_out.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        ModelCheckpoint(filepath=str(model_out), monitor="val_auc", mode="max", save_best_only=True, verbose=1),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=int(cfg["epochs"]),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    threshold_fixed = float(cfg["decision_threshold"])
    y_val = _collect_labels_in_spans(all_data, val_spans)
    y_test = _collect_labels_in_spans(all_data, holdout_spans)
    y_val_prob = _collect_probs_for_spans(model, all_data, val_spans, batch_size=batch_size)
    y_test_prob = _collect_probs_for_spans(model, all_data, holdout_spans, batch_size=batch_size)
    best_t, metrics_val_at_best, _ = find_best_threshold_by_f1(y_val, y_val_prob, min_t=0.2)

    overall_fixed = compute_classification_metrics(y_test, y_test_prob, threshold_fixed)
    overall_tuned = compute_classification_metrics(y_test, y_test_prob, best_t)
    # Use bounded sampled arrays for baseline models to keep memory bounded.
    X_train_base, y_train_base = _sample_arrays_from_spans(
        all_data, train_spans, max_samples=200_000, seed=int(cfg["seed"])
    )
    X_test_base, y_test_base = _sample_arrays_from_spans(
        all_data, holdout_spans, max_samples=50_000, seed=int(cfg["seed"]) + 1
    )
    baseline_metrics = evaluate_baselines(
        X_train_base,
        y_train_base,
        X_test_base,
        y_test_base,
        threshold=threshold_fixed,
        random_state=int(cfg["seed"]),
    )

    per_building_fixed: Dict[str, Dict[str, float]] = {}
    per_building_tuned: Dict[str, Dict[str, float]] = {}
    for b in holdout:
        Xb, yb = all_data[b]
        yb_prob = model.predict(np.asarray(Xb, dtype=np.float32), verbose=0).reshape(-1)
        per_building_fixed[b] = compute_classification_metrics(yb, yb_prob, threshold_fixed)
        per_building_tuned[b] = compute_classification_metrics(yb, yb_prob, best_t)
        per_building_fixed[b]["num_sequences"] = int(len(yb))
        per_building_fixed[b]["positive_rate"] = float(yb.mean())
        per_building_tuned[b]["num_sequences"] = int(len(yb))
        per_building_tuned[b]["positive_rate"] = float(yb.mean())

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_history_plot(history, out_dir)
    save_confusion_matrix(
        y_test,
        y_test_prob,
        threshold_fixed,
        out_dir,
        filename="log_lstm_holdout_confusion_matrix_threshold_0.5.png",
        title=f"Holdout CM (threshold={threshold_fixed})",
    )
    save_confusion_matrix(
        y_test,
        y_test_prob,
        best_t,
        out_dir,
        filename="log_lstm_holdout_confusion_matrix_threshold_val_f1.png",
        title=f"Holdout CM (threshold={best_t:.4f} from val F1)",
    )

    payload = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()},
        "train_buildings": train_buildings,
        "holdout_buildings": holdout,
        "shapes": {
            "train": [int(train_n), int(seq_len), int(feat_dim)],
            "val": [int(val_n), int(seq_len), int(feat_dim)],
            "holdout_test": [int(test_n), int(seq_len), int(feat_dim)],
        },
        "rates": {
            "train_positive_rate": float(train_pos / max(1, train_n)),
            "val_positive_rate": float(val_pos / max(1, val_n)),
            "holdout_positive_rate": float(test_pos / max(1, test_n)),
        },
        "class_balance": {
            "train": class_balance_stats(_collect_labels_in_spans(all_data, train_spans)),
            "val": class_balance_stats(y_val),
            "holdout_test": class_balance_stats(y_test),
        },
        "leakage_checks": {
            "split_strategy": "temporal per building with purge gap",
            "purge_gap_sequences": purge_gap,
            "per_train_building_counts": split_audit,
            "holdout_buildings_excluded_from_train_val": True,
        },
        "threshold_tuning": {
            "method": "maximize F1 on validation (train-building split)",
            "threshold_val_f1": best_t,
            "metrics_validation_at_threshold": metrics_val_at_best,
        },
        "metrics_holdout_overall_threshold_0.5": overall_fixed,
        "metrics_holdout_overall_threshold_val_f1": overall_tuned,
        "baseline_metrics_holdout_threshold_0.5": baseline_metrics,
        "baseline_sampling": {
            "train_max_samples": 200000,
            "test_max_samples": 50000,
            "train_used_samples": int(len(y_train_base)),
            "test_used_samples": int(len(y_test_base)),
        },
        "metrics_holdout_per_building_threshold_0.5": per_building_fixed,
        "metrics_holdout_per_building_threshold_val_f1": per_building_tuned,
    }
    metrics_path = out_dir / "log_lstm_holdout_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nSaved artifacts:")
    print(f"  model   : {model_out}")
    print(f"  history : {out_dir / 'log_lstm_holdout_history.png'}")
    print(f"  cm      : {out_dir / 'log_lstm_holdout_confusion_matrix_threshold_val_f1.png'}")
    print(f"  metrics : {metrics_path}")
    print(f"  holdout @0.5: {overall_fixed}")
    print(f"  holdout @val F1 ({best_t:.4f}): {overall_tuned}")
    print(f"  baselines @0.5: {baseline_metrics}")


if __name__ == "__main__":
    main()
