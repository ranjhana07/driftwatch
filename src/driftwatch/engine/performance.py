from __future__ import annotations

import numpy as np
from scipy import stats


def auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(bool)
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = stats.rankdata(np.asarray(score, dtype=float))
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true).astype(bool)
    pred = np.asarray(y_pred).astype(bool)
    tp = float(np.sum(truth & pred))
    fp = float(np.sum(~truth & pred))
    fn = float(np.sum(truth & ~pred))
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def expected_calibration_error(y_true: np.ndarray, score: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(score, dtype=float), 0.0, 1.0)
    index = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += mask.sum() * abs(y[mask].mean() - p[mask].mean())
    return float(total / y.size)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def classification_metrics(
    labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray | None, positive_label: object
) -> dict[str, float]:
    truth = np.array([str(v) == str(positive_label) for v in labels])
    predicted = np.array([str(v) == str(positive_label) for v in predictions])
    metrics = {"f1": f1(truth, predicted), "accuracy": float(np.mean(truth == predicted))}
    if scores is not None and not np.isnan(scores).any():
        value = auc(truth, scores)
        if not np.isnan(value):
            metrics["auc"] = value
        metrics["ece"] = expected_calibration_error(truth, scores)
    return metrics


def regression_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {"rmse": rmse(labels, predictions)}
