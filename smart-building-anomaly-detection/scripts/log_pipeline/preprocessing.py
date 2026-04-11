"""Load raw building logs, encode categoricals, and write processed CSV + metadata."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Numeric columns fed to the LSTM (after preprocessing).
LSTM_FEATURE_COLUMNS: List[str] = [
    "event_type",
    "event_code",
    "severity",
    "subsystem",
    "value_zscore",
    "value_dev_roll_mean",
    "value_roll_std",
    "value_diff_1",
    "value_pct_change",
    "time_delta_seconds",
]

CATEGORICAL_COLS: List[str] = [
    "event_type",
    "event_code",
    "device_id",
    "subsystem",
]

SEVERITY_MAP: Dict[str, int] = {"INFO": 0, "WARN": 1, "CRITICAL": 2}

MISSING_TOKEN = "__MISSING__"


def _project_root() -> Path:
    """Repository root (directory that contains ``src/`` and ``scripts/``)."""
    return Path(__file__).resolve().parent.parent.parent


def load_sensor_building_ids() -> List[str]:
    """IDs matching ``src.config.BUILDINGS`` (sensor LSTM evaluation cohort)."""
    root = _project_root()
    import sys

    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    from src.config import BUILDINGS

    return list(BUILDINGS)


def building_id_from_raw_log_path(path: Path) -> str:
    """``Office_Annika_logs.csv`` -> ``Office_Annika``."""
    name = path.name
    if name.endswith("_logs.csv"):
        return name[: -len("_logs.csv")]
    return path.stem


def building_id_from_processed_path(path: Path) -> str:
    """``Office_Annika_processed.csv`` -> ``Office_Annika``."""
    stem = path.stem
    if stem.endswith("_processed"):
        return stem[: -len("_processed")]
    return stem


def list_raw_log_files(logs_raw_dir: str | Path) -> List[Path]:
    """Return sorted ``*.csv`` paths under ``logs_raw``."""
    root = Path(logs_raw_dir)
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("*.csv") if p.is_file())


def processed_csv_name(raw_path: Path) -> str:
    """``BuildingName_logs.csv`` -> ``BuildingName_processed.csv`` (matches simulation naming)."""
    name = raw_path.name
    if "_logs.csv" in name:
        return name.replace("_logs.csv", "_processed.csv")
    return f"{raw_path.stem}_processed.csv"


def load_raw_log_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path}: missing required column 'timestamp'")
    return df


def _fill_missing_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in CATEGORICAL_COLS:
        if c not in out.columns:
            out[c] = MISSING_TOKEN
        else:
            s = out[c].astype(object)
            out[c] = s.where(s.notna(), MISSING_TOKEN).astype(str)
            out[c] = out[c].replace({"nan": MISSING_TOKEN})
    return out


def collect_vocabularies(dfs: Iterable[pd.DataFrame]) -> Dict[str, List[str]]:
    """Sorted union of category strings per column across all frames (global encoding)."""
    seen: Dict[str, set] = {c: set() for c in CATEGORICAL_COLS}
    for df in dfs:
        d = _fill_missing_categoricals(df)
        for c in CATEGORICAL_COLS:
            seen[c].update(d[c].astype(str).unique())
    return {c: sorted(seen[c]) for c in CATEGORICAL_COLS}


def vocabularies_to_mappings(vocabularies: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    return {col: {v: i for i, v in enumerate(vals)} for col, vals in vocabularies.items()}


def preprocess_dataframe(
    df: pd.DataFrame,
    category_maps: Dict[str, Dict[str, int]],
    *,
    encode_message: bool = False,
    message_map: Optional[Dict[str, int]] = None,
) -> Tuple[pd.DataFrame, Optional[Dict[str, int]]]:
    """
    Parse timestamps, sort, encode categoricals / severity / value, keep ``anomaly_link``.
    Unknown category strings map to ``len(map)`` (one extra index per column).
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])

    out = _fill_missing_categoricals(out)
    for c in CATEGORICAL_COLS:
        m = category_maps[c]
        unk = len(m)
        out[c] = out[c].astype(str).map(lambda x, mm=m, u=unk: mm.get(x, u)).astype(int)

    if "severity" not in out.columns:
        out["severity"] = 0
    else:
        sev = out["severity"].astype(str).str.upper()
        out["severity"] = pd.to_numeric(sev.map(SEVERITY_MAP), errors="coerce").fillna(0).astype(int)

    if "value" not in out.columns:
        out["value"] = 0.0
    else:
        out["value"] = pd.to_numeric(out["value"], errors="coerce").fillna(0.0)

    # Per-building/series normalization and temporal behavior features.
    # Each processed file is a single building stream. We use causal (past-only)
    # statistics so features for timestamp t do not depend on future samples.
    val = out["value"].astype(float)
    eps = 1e-8

    # Expanding stats from previous timesteps only.
    prior_mean = val.expanding(min_periods=2).mean().shift(1)
    prior_std = val.expanding(min_periods=2).std(ddof=0).shift(1)
    safe_prior_std = prior_std.where(prior_std.abs() > eps, np.nan)
    out["value_zscore"] = ((val - prior_mean) / safe_prior_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Rolling behavior features over recent history (excluding current timestep).
    roll_win = 12
    prior_val = val.shift(1)
    roll_mean = prior_val.rolling(window=roll_win, min_periods=1).mean()
    roll_std = prior_val.rolling(window=roll_win, min_periods=1).std(ddof=0).fillna(0.0)
    out["value_dev_roll_mean"] = (val - roll_mean).fillna(0.0)
    out["value_roll_std"] = roll_std
    out["value_diff_1"] = val.diff().fillna(0.0)
    out["value_pct_change"] = (
        val.pct_change().replace([pd.NA, np.inf, -np.inf], np.nan).fillna(0.0)
    )
    out["time_delta_seconds"] = (
        out["timestamp"].diff().dt.total_seconds().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    if "anomaly_link" not in out.columns:
        out["anomaly_link"] = 0
    else:
        out["anomaly_link"] = (
            pd.to_numeric(out["anomaly_link"], errors="coerce").fillna(0).astype(int).clip(0, 1)
        )

    msg_map_out: Optional[Dict[str, int]] = message_map
    if encode_message:
        if "message" not in out.columns:
            out["message"] = ""
        msgs = out["message"].fillna("").astype(str)
        if message_map is None:
            unique = sorted(msgs.unique())
            msg_map_out = {s: i for i, s in enumerate(unique)}
        assert msg_map_out is not None
        unk = len(msg_map_out)
        out["message_id"] = msgs.map(lambda x, mm=msg_map_out, u=unk: mm.get(x, u)).astype(int)

    out = out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return out, msg_map_out


def save_encoding_metadata(path: str | Path, category_maps: Dict[str, Dict[str, int]], **extra: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"category_maps": category_maps, **extra}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_encoding_metadata(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_preprocessing_pipeline(
    logs_raw_dir: str | Path,
    logs_processed_dir: str | Path,
    *,
    encode_message: bool = False,
    metadata_filename: str = "encoding_metadata.json",
    process_all_buildings: bool = False,
    building_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Discover raw CSV files under ``logs_raw``, fit **global** category maps, write one
    processed CSV per file and ``encoding_metadata.json``.

    By default only buildings in ``src.config.BUILDINGS`` are processed (aligned with the
    sensor LSTM study). Set ``process_all_buildings=True`` to use every ``*.csv`` in
    ``logs_raw``. Optionally pass ``building_ids`` to override the allow-list explicitly.
    """
    raw_dir = Path(logs_raw_dir)
    out_dir = Path(logs_processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_paths = list_raw_log_files(raw_dir)
    if not all_paths:
        return {
            "files_processed": 0,
            "processed_paths": [],
            "metadata_path": None,
            "warnings": [f"No CSV files found in {raw_dir.resolve()}"],
        }

    if process_all_buildings:
        paths = all_paths
        cohort: Optional[List[str]] = None
        skipped_outside_cohort = 0
    else:
        allow = set(building_ids if building_ids is not None else load_sensor_building_ids())
        paths = [p for p in all_paths if building_id_from_raw_log_path(p) in allow]
        cohort = sorted(allow)
        skipped_outside_cohort = len(all_paths) - len(paths)

    if not paths:
        return {
            "files_processed": 0,
            "processed_paths": [],
            "metadata_path": None,
            "warnings": [
                f"No raw logs for selected cohort in {raw_dir.resolve()}. "
                f"CSV count (all): {len(all_paths)}; skipped as outside cohort: {skipped_outside_cohort}. "
                f"Cohort: {cohort}"
            ],
            "building_cohort": cohort,
            "files_skipped_outside_cohort": skipped_outside_cohort,
        }

    dfs = [load_raw_log_csv(p) for p in paths]
    vocabs = collect_vocabularies(dfs)
    category_maps = vocabularies_to_mappings(vocabs)

    message_map: Optional[Dict[str, int]] = None
    if encode_message:
        parts = []
        for df in dfs:
            if "message" in df.columns:
                parts.append(df["message"].fillna("").astype(str))
        if parts:
            unique = sorted(pd.concat(parts, ignore_index=True).unique())
            message_map = {s: i for i, s in enumerate(unique)}

    processed_paths: List[str] = []
    for path, df in zip(paths, dfs):
        proc, _ = preprocess_dataframe(
            df,
            category_maps,
            encode_message=encode_message,
            message_map=message_map,
        )
        out_path = out_dir / processed_csv_name(path)
        proc.to_csv(out_path, index=False)
        processed_paths.append(str(out_path))

    meta_path = out_dir / metadata_filename
    extra: Dict[str, Any] = {}
    if encode_message and message_map is not None:
        extra["message_map"] = message_map
    save_encoding_metadata(meta_path, category_maps, **extra)

    out: Dict[str, Any] = {
        "files_processed": len(processed_paths),
        "processed_paths": processed_paths,
        "metadata_path": str(meta_path),
        "category_vocab_sizes": {k: len(v) for k, v in vocabs.items()},
        "lstm_feature_columns_default": list(LSTM_FEATURE_COLUMNS),
    }
    if not process_all_buildings:
        out["building_cohort"] = cohort
        out["files_skipped_outside_cohort"] = skipped_outside_cohort
    return out
