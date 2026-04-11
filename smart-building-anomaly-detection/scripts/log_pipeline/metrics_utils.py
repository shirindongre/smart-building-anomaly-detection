"""Shared metrics, threshold tuning, and baseline model evaluation."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, y_prob),
    }


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def find_best_threshold_by_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    min_t: float = 0.20,
    max_t: float = 0.95,
    num: int = 91,
) -> Tuple[float, Dict[str, float], List[Dict[str, float]]]:
    """
    Scan thresholds on validation labels; pick threshold that maximizes F1.
    Returns (best_threshold, metrics_at_that_threshold_on_same_set, sweep_rows).
    """
    thresholds = np.linspace(min_t, max_t, num=num)
    best_t = 0.5
    best_f1 = -1.0
    sweep: List[Dict[str, float]] = []

    for t in thresholds:
        m = compute_classification_metrics(y_true, y_prob, float(t))
        row = {"threshold": float(t), **m}
        sweep.append(row)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = float(t)

    best_metrics = compute_classification_metrics(y_true, y_prob, best_t)
    return best_t, best_metrics, sweep


def class_balance_stats(y: np.ndarray) -> Dict[str, float]:
    n = int(len(y))
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    return {
        "n": n,
        "n_pos": pos,
        "n_neg": neg,
        "positive_rate": float(pos / n) if n else 0.0,
        "negative_rate": float(neg / n) if n else 0.0,
    }


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError(f"Expected 3D array for sequences, got shape {X.shape}")
    return X.reshape((X.shape[0], X.shape[1] * X.shape[2]))


def evaluate_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    threshold: float = 0.5,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    Xtr = flatten_sequences(X_train)
    Xte = flatten_sequences(X_test)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    lr.fit(Xtr, y_train)
    lr_prob = lr.predict_proba(Xte)[:, 1]

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(Xtr, y_train)
    rf_prob = rf.predict_proba(Xte)[:, 1]

    return {
        "logistic_regression": compute_classification_metrics(y_test, lr_prob, threshold),
        "random_forest": compute_classification_metrics(y_test, rf_prob, threshold),
    }
