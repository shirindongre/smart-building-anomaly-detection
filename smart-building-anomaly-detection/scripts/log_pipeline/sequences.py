"""Build fixed-length, time-ordered log sequences for LSTM input."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .preprocessing import (
    LSTM_FEATURE_COLUMNS,
    building_id_from_processed_path,
    load_sensor_building_ids,
)

IDENTITY_LEAKAGE_FEATURES = {"building_id", "building", "site_id", "device_id"}


def list_processed_log_files(logs_processed_dir: str | Path) -> List[Path]:
    root = Path(logs_processed_dir)
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("*_processed.csv") if p.is_file())


def load_processed_log_csv(path: str | Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
    except pd.errors.ParserError:
        # Fallback parser is slower but more robust for very wide/noisy CSV chunks.
        df = pd.read_csv(path, engine="python", low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df


def build_sequences(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    *,
    sequence_length: int = 10,
    stride: int = 1,
    label_col: str = "anomaly_link",
    label_mode: str = "any",
) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
    """
    Return ``X`` with shape ``(num_sequences, sequence_length, num_features)`` (float32),
    labels ``y`` (optional), and per-sequence metadata.

    * ``label_mode='any'``: label 1 if any event in the window has ``anomaly_link >= 1``.
    * ``label_mode='last'``: label from the last row of the window only.
    * ``label_mode='none'``: do not return labels (``y`` is None).
    """
    cols = list(feature_cols or LSTM_FEATURE_COLUMNS)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe missing feature columns: {missing}")

    n = len(df)
    feat_dim = len(cols)
    if n < sequence_length:
        empty_X = np.zeros((0, sequence_length, feat_dim), dtype=np.float32)
        empty_meta = pd.DataFrame(
            columns=["sequence_idx", "start_row_idx", "end_row_idx", "start_timestamp", "end_timestamp"]
        )
        if label_mode == "none" or label_col not in df.columns:
            return empty_X, None, empty_meta
        return empty_X, np.array([], dtype=np.int8), empty_meta

    data = df[cols].to_numpy(dtype=np.float32, copy=False)
    has_y = label_mode != "none" and label_col in df.columns
    labels = df[label_col].to_numpy(dtype=np.float64, copy=False) if has_y else None

    num_sequences = ((n - sequence_length) // stride) + 1
    X = np.empty((num_sequences, sequence_length, feat_dim), dtype=np.float32)
    y = np.empty((num_sequences,), dtype=np.int8) if labels is not None else None
    meta_rows: List[Dict[str, Any]] = []
    last_start = n - sequence_length
    for sequence_idx, start in enumerate(range(0, last_start + 1, stride)):
        end = start + sequence_length
        X[sequence_idx] = data[start:end]
        meta_rows.append(
            {
                "sequence_idx": sequence_idx,
                "start_row_idx": int(start),
                "end_row_idx": int(end - 1),
                "start_timestamp": pd.Timestamp(df["timestamp"].iloc[start]).isoformat(),
                "end_timestamp": pd.Timestamp(df["timestamp"].iloc[end - 1]).isoformat(),
            }
        )
        if labels is not None:
            window = labels[start:end]
            if label_mode == "any":
                y[sequence_idx] = 1 if np.nanmax(window) >= 1 else 0
            elif label_mode == "last":
                y[sequence_idx] = 1 if labels[end - 1] >= 1 else 0
            else:
                raise ValueError(f"Unknown label_mode: {label_mode!r}")

    meta = pd.DataFrame(meta_rows)
    return X, y, meta


def assert_feature_columns_clean(feature_cols: Sequence[str], *, label_col: str = "anomaly_link") -> None:
    if label_col in set(feature_cols):
        raise ValueError(
            f"Leakage detected: label column '{label_col}' is present in feature columns {list(feature_cols)}."
        )
    leakage_cols = sorted(c for c in feature_cols if c in IDENTITY_LEAKAGE_FEATURES)
    if leakage_cols:
        raise ValueError(
            f"Potential building-identity leakage: remove {leakage_cols} from sequence features."
        )


def split_sequence_indices_temporal(
    num_sequences: int,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    purge_gap: int,
) -> Dict[str, np.ndarray]:
    """
    Build non-overlapping temporal index splits for sliding-window sequences.
    A purge gap between boundaries prevents overlapping windows across splits.
    """
    if num_sequences <= 0:
        empty = np.array([], dtype=np.int64)
        return {"train": empty, "val": empty, "test": empty}

    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    n_train = int(num_sequences * train_ratio)
    n_val = int(num_sequences * val_ratio)
    n_test = num_sequences - n_train - n_val
    if n_train <= 0:
        raise ValueError(
            f"Split too small for temporal separation: num_sequences={num_sequences}, "
            f"(train,val,test)=({n_train},{n_val},{n_test})."
        )
    if val_ratio > 0 and n_val <= 0:
        raise ValueError(
            f"Validation split empty with num_sequences={num_sequences}, val_ratio={val_ratio}."
        )
    if test_ratio > 0 and n_test <= 0:
        raise ValueError(
            f"Test split empty with num_sequences={num_sequences}, test_ratio={test_ratio}."
        )

    train_end = n_train
    train_idx = np.arange(0, train_end, dtype=np.int64)
    val_idx = np.array([], dtype=np.int64)
    test_idx = np.array([], dtype=np.int64)

    cursor = train_end + purge_gap
    if val_ratio > 0:
        val_end = min(cursor + n_val, num_sequences)
        val_idx = np.arange(cursor, val_end, dtype=np.int64)
        cursor = val_end + purge_gap
        if len(val_idx) == 0:
            raise ValueError(
                f"Temporal validation split became empty after purge gap. "
                f"num_sequences={num_sequences}, purge_gap={purge_gap}"
            )

    if test_ratio > 0:
        if cursor >= num_sequences:
            raise ValueError(
                f"Not enough sequences for purge_gap={purge_gap}. "
                f"num_sequences={num_sequences}, cursor={cursor}"
            )
        test_idx = np.arange(cursor, num_sequences, dtype=np.int64)
        if len(test_idx) == 0:
            raise ValueError(
                f"Temporal test split became empty after purge gap. "
                f"num_sequences={num_sequences}, purge_gap={purge_gap}"
            )

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def assert_no_overlap_in_row_ranges(meta_df: pd.DataFrame, left_idx: np.ndarray, right_idx: np.ndarray) -> None:
    """
    Assert no raw-row overlap between two sequence index sets.
    Uses [start_row_idx, end_row_idx] interval intersection.
    """
    if len(left_idx) == 0 or len(right_idx) == 0:
        return
    left = meta_df.iloc[left_idx]
    right = meta_df.iloc[right_idx]

    # Fast boundary checks for time-ordered windows.
    left_last_end = int(left["end_row_idx"].max())
    right_first_start = int(right["start_row_idx"].min())
    if left_last_end >= right_first_start:
        raise AssertionError(
            f"Leakage detected: overlapping row ranges between splits "
            f"(left_last_end={left_last_end}, right_first_start={right_first_start})."
        )


def run_sequence_pipeline(
    logs_processed_dir: str | Path,
    output_dir: str | Path,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    sequence_length: int = 10,
    stride: int = 1,
    label_mode: str = "any",
    combine_all: bool = False,
    manifest_name: str = "sequence_manifest.json",
    overwrite: bool = True,
    skip_existing: bool = False,
    process_all_buildings: bool = False,
    building_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Read ``*_processed.csv`` files and save **per-building** ``*_X.npy`` / ``*_y.npy`` / ``*_meta.csv``.

    By default only ``src.config.BUILDINGS`` are included (sensor study cohort). Set
    ``process_all_buildings=True`` to sequence every ``*_processed.csv`` under the input
    directory. This pipeline stays memory-safe: it never concatenates all buildings.
    """
    proc_dir = Path(logs_processed_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(feature_cols or LSTM_FEATURE_COLUMNS)
    assert_feature_columns_clean(cols, label_col="anomaly_link")
    if combine_all:
        raise ValueError(
            "combine_all=True is not supported in the memory-safe pipeline. "
            "Train/eval scripts should load per-building arrays from the manifest."
        )

    all_paths = list_processed_log_files(proc_dir)
    if not all_paths:
        return {
            "files": 0,
            "per_file": [],
            "warnings": [f"No *_processed.csv files in {proc_dir.resolve()}"],
        }

    if process_all_buildings:
        paths = all_paths
        cohort: Optional[List[str]] = None
        skipped_outside_cohort = 0
    else:
        allow = set(building_ids if building_ids is not None else load_sensor_building_ids())
        paths = [p for p in all_paths if building_id_from_processed_path(p) in allow]
        cohort = sorted(allow)
        skipped_outside_cohort = len(all_paths) - len(paths)

    if not paths:
        return {
            "files": 0,
            "per_file": [],
            "warnings": [
                f"No processed logs for selected cohort in {proc_dir.resolve()}. "
                f"Processed file count (all): {len(all_paths)}; skipped outside cohort: {skipped_outside_cohort}. "
                f"Cohort: {cohort}"
            ],
            "building_cohort": cohort,
            "files_skipped_outside_cohort": skipped_outside_cohort,
        }

    per_file: List[Dict[str, Any]] = []

    for i, p in enumerate(paths, start=1):
        stem = p.stem
        x_path = out_dir / f"{stem}_X.npy"
        y_path = out_dir / f"{stem}_y.npy"
        meta_path = out_dir / f"{stem}_meta.csv"

        if skip_existing and x_path.is_file() and y_path.is_file() and meta_path.is_file():
            try:
                X_existing = np.load(x_path, mmap_mode="r")
                y_existing = np.load(y_path, mmap_mode="r")
                meta_existing = pd.read_csv(meta_path)
                if len(meta_existing) == int(X_existing.shape[0]) == int(y_existing.shape[0]):
                    print(f"[{i}/{len(paths)}] Skipping existing {stem}: {X_existing.shape}")
                    per_file.append(
                        {
                            "building": stem.replace("_processed", ""),
                            "source_csv": str(p),
                            "X_path": str(x_path),
                            "y_path": str(y_path),
                            "meta_path": str(meta_path),
                            "num_sequences": int(X_existing.shape[0]),
                            "shape_X": [int(v) for v in X_existing.shape],
                        }
                    )
                    continue
            except Exception:
                # If any validation fails, fall through and regenerate this building.
                pass

        if overwrite:
            for fp in (x_path, y_path, meta_path):
                if fp.exists():
                    fp.unlink(missing_ok=True)

        print(f"[{i}/{len(paths)}] Processing {stem} ...")
        df = load_processed_log_csv(p)
        X, y, meta = build_sequences(
            df,
            cols,
            sequence_length=sequence_length,
            stride=stride,
            label_mode=label_mode,
        )
        np.save(x_path, X)
        if y is None:
            raise ValueError(f"{stem}: label_mode={label_mode!r} produced y=None unexpectedly.")
        np.save(y_path, y)
        meta.to_csv(meta_path, index=False)

        entry: Dict[str, Any] = {
            "building": stem.replace("_processed", ""),
            "source_csv": str(p),
            "X_path": str(x_path),
            "y_path": str(y_path),
            "meta_path": str(meta_path),
            "num_sequences": int(X.shape[0]),
            "shape_X": list(X.shape),
        }
        per_file.append(entry)
        print(f"  saved X={X.shape} y={y.shape} meta_rows={len(meta)}")

    manifest: Dict[str, Any] = {
        "feature_columns": cols,
        "sequence_length": sequence_length,
        "stride": stride,
        "label_mode": label_mode,
        "per_file": per_file,
    }
    if not process_all_buildings:
        manifest["building_cohort"] = cohort
        manifest["files_skipped_outside_cohort"] = skipped_outside_cohort

    mf_path = out_dir / manifest_name
    mf_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(mf_path)

    return manifest
