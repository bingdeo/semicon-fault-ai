"""Shared binary classification metrics and threshold selection."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_max_f1_threshold(y_true: Any, y_score: Any) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5, 0.0
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def expected_calibration_error(
    y_true: Any, y_score: Any, n_bins: int = 10
) -> float:
    truth = np.asarray(y_true, dtype=float)
    scores = np.asarray(y_score, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.digitize(scores, edges[1:-1], right=True), n_bins - 1)
    error = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        error += float(mask.mean()) * abs(float(truth[mask].mean() - scores[mask].mean()))
    return error


def recall_at_fixed_precision(
    y_true: Any, y_score: Any, minimum_precision: float
) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    eligible = recall[precision >= minimum_precision]
    return float(eligible.max()) if len(eligible) else 0.0


def binary_metrics(
    y_true: Any,
    y_score: Any,
    threshold: float,
    fixed_precision: float = 0.50,
) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(truth, scores)),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "brier_score": float(brier_score_loss(truth, scores)),
        "ece_10_bins": expected_calibration_error(truth, scores, n_bins=10),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "recall_at_precision_50": recall_at_fixed_precision(
            truth, scores, minimum_precision=fixed_precision
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
