"""
Validate per-building sequence artifacts against sequence_manifest.json.

Checks (random sample of buildings):
  1) X.npy, y.npy, meta.csv exist
  2) lengths match (X rows == y len == meta rows)
  3) shapes match manifest
  4) sampled windows align with meta timestamps and labels against the source processed CSV
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_processed_csv(path: Path) -> pd.DataFrame:
    # Keep consistent with sequences.py loader, but local to this validator script.
    try:
        df = pd.read_csv(path, low_memory=False)
    except pd.errors.ParserError:
        df = pd.read_csv(path, engine="python", low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df


def _compute_window_label(labels: np.ndarray, start: int, end: int, *, mode: str) -> int:
    w = labels[start:end]
    if mode == "any":
        return 1 if np.nanmax(w) >= 1 else 0
    if mode == "last":
        return 1 if labels[end - 1] >= 1 else 0
    raise ValueError(f"Unknown label_mode: {mode!r}")


def validate_manifest(
    manifest_path: Path,
    *,
    sample_buildings: int = 5,
    sample_indices_per_building: int = 5,
    seed: int = 42,
) -> Tuple[bool, List[str]]:
    payload = _load_manifest(manifest_path)
    per_file = payload.get("per_file", [])
    if not per_file:
        return False, ["Manifest has no per_file entries."]

    label_mode = str(payload.get("label_mode", "any"))
    seq_len = int(payload.get("sequence_length", 10))

    rnd = random.Random(seed)
    chosen = per_file[:] if len(per_file) <= sample_buildings else rnd.sample(per_file, sample_buildings)

    errors: List[str] = []
    for item in chosen:
        building = item.get("building") or Path(str(item["source_csv"])).stem.replace("_processed", "")
        src_csv = Path(item["source_csv"])
        x_path = Path(item["X_path"])
        y_path = Path(item["y_path"])
        meta_path = Path(item["meta_path"])
        expected_shape = tuple(item.get("shape_X", ()))
        local_errors: List[str] = []

        # (1) existence
        for fp in (src_csv, x_path, y_path, meta_path):
            if not fp.is_file():
                local_errors.append(f"missing file {fp}")

        if local_errors:
            errors.extend(f"{building}: {msg}" for msg in local_errors)
            print(f"[FAIL] {building}: {local_errors[0]}")
            continue

        # (2)/(3) load + length/shape checks
        try:
            X = np.load(x_path, mmap_mode="r")
            y = np.load(y_path, mmap_mode="r")
            meta = pd.read_csv(meta_path)
        except Exception as e:
            local_errors.append(f"failed to load artifacts: {e}")
            errors.extend(f"{building}: {msg}" for msg in local_errors)
            print(f"[FAIL] {building}: failed to load artifacts")
            continue

        if expected_shape and tuple(expected_shape) != tuple(X.shape):
            local_errors.append(f"X shape mismatch manifest={expected_shape} actual={tuple(X.shape)}")
        if int(X.shape[0]) != int(y.shape[0]):
            local_errors.append(f"X/y length mismatch {int(X.shape[0])} vs {int(y.shape[0])}")
        if int(X.shape[0]) != int(len(meta)):
            local_errors.append(f"X/meta length mismatch {int(X.shape[0])} vs {int(len(meta))}")

        if int(X.shape[1]) != seq_len:
            local_errors.append(f"sequence_length mismatch manifest={seq_len} actual={int(X.shape[1])}")

        # (4) alignment checks against source CSV
        try:
            df = _load_processed_csv(src_csv)
            if "anomaly_link" not in df.columns:
                local_errors.append("source CSV missing anomaly_link column")
                # can't do deeper alignment checks without labels, but continue to report other issues
            src_labels = pd.to_numeric(df["anomaly_link"], errors="coerce").fillna(0).to_numpy(dtype=float)

            if len(X) > 0 and "anomaly_link" in df.columns:
                idxs = list(range(len(X)))
                if len(idxs) > sample_indices_per_building:
                    idxs = rnd.sample(idxs, sample_indices_per_building)

                for seq_idx in idxs:
                    row = meta.iloc[int(seq_idx)]
                    start = int(row["start_row_idx"])
                    end = int(row["end_row_idx"]) + 1
                    if end - start != seq_len:
                        local_errors.append(
                            f"meta window length wrong at seq_idx={seq_idx} "
                            f"(start={start}, end={end}, expected_len={seq_len})"
                        )
                        continue

                    ts_start_meta = pd.to_datetime(row["start_timestamp"], errors="coerce")
                    ts_end_meta = pd.to_datetime(row["end_timestamp"], errors="coerce")
                    ts_start_src = pd.to_datetime(df["timestamp"].iloc[start], errors="coerce")
                    ts_end_src = pd.to_datetime(df["timestamp"].iloc[end - 1], errors="coerce")
                    if (
                        pd.isna(ts_start_meta)
                        or pd.isna(ts_end_meta)
                        or pd.isna(ts_start_src)
                        or pd.isna(ts_end_src)
                    ):
                        local_errors.append(f"timestamp parse failure at seq_idx={seq_idx}")
                    else:
                        if ts_start_meta != ts_start_src or ts_end_meta != ts_end_src:
                            local_errors.append(
                                f"timestamp mismatch at seq_idx={seq_idx} "
                                f"(meta {ts_start_meta}->{ts_end_meta} vs src {ts_start_src}->{ts_end_src})"
                            )

                    y_expected = _compute_window_label(src_labels, start, end, mode=label_mode)
                    y_actual = int(y[int(seq_idx)])
                    if y_expected != y_actual:
                        local_errors.append(
                            f"label mismatch at seq_idx={seq_idx} expected={y_expected} actual={y_actual} "
                            f"(label_mode={label_mode})"
                        )
        except Exception as e:
            local_errors.append(f"alignment validation failed: {e}")

        if local_errors:
            errors.extend(f"{building}: {msg}" for msg in local_errors)
            print(f"[FAIL] {building}: {local_errors[0]}")
        else:
            print(
                f"[PASS] {building}: "
                f"X.shape={tuple(X.shape)}, len(y)={len(y)}, len(meta)={len(meta)}, seq_len={seq_len}"
            )

    ok = len(errors) == 0
    return ok, errors


def main() -> None:
    # Allow running directly: python scripts/log_pipeline/validate_sequences.py [manifest_path]
    manifest_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/log_sequences/sequence_manifest.json")
    ok, errors = validate_manifest(manifest_path)
    if ok:
        print(f"OK: validation passed for sampled buildings in {manifest_path}")
        return
    print(f"FAILED: validation found {len(errors)} issue(s):")
    for e in errors:
        print(f"  - {e}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()

