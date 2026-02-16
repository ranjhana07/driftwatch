from __future__ import annotations

import numpy as np
import pytest
from sklearn import metrics as sk
from sklearn.calibration import calibration_curve

from driftwatch.engine.performance import (
    auc,
    classification_metrics,
    expected_calibration_error,
    f1,
    regression_metrics,
    rmse,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(21)
    y = rng.integers(0, 2, 500)
    score = np.clip(y * 0.3 + rng.normal(0.35, 0.25, 500), 0, 1)
    return y, score


def test_auc_matches_sklearn(data):
    y, score = data
    assert auc(y, score) == pytest.approx(sk.roc_auc_score(y, score))


def test_auc_with_tied_scores():
    y = np.array([0, 1, 0, 1, 1, 0])
    score = np.array([0.2, 0.2, 0.5, 0.5, 0.9, 0.1])
    assert auc(y, score) == pytest.approx(sk.roc_auc_score(y, score))


def test_auc_single_class_is_nan():
    assert np.isnan(auc(np.ones(5), np.linspace(0, 1, 5)))


def test_f1_matches_sklearn(data):
    y, score = data
    pred = score > 0.5
    assert f1(y, pred) == pytest.approx(sk.f1_score(y, pred))


def test_f1_no_positives():
    assert f1(np.zeros(4), np.zeros(4)) == 0.0


def test_ece_perfectly_calibrated_bins():
    score = np.array([0.1] * 10 + [0.9] * 10)
    y = np.array([1] + [0] * 9 + [1] * 9 + [0])
    assert expected_calibration_error(y, score) == pytest.approx(0.0)


def test_ece_overconfident_model():
    score = np.full(100, 0.95)
    y = np.array([1] * 50 + [0] * 50)
    assert expected_calibration_error(y, score) == pytest.approx(0.45)


def test_ece_matches_reference_binning(data):
    y, score = data
    prob_true, prob_pred = calibration_curve(y, score, n_bins=10, strategy="uniform")
    edges = np.minimum((score * 10).astype(int), 9)
    counts = np.bincount(edges, minlength=10)
    counts = counts[counts > 0]
    expected = float(np.sum(counts * np.abs(prob_true - prob_pred)) / len(y))
    assert expected_calibration_error(y, score) == pytest.approx(expected)


def test_rmse_matches_sklearn():
    rng = np.random.default_rng(2)
    a = rng.normal(size=100)
    b = a + rng.normal(0, 0.3, 100)
    assert rmse(a, b) == pytest.approx(sk.root_mean_squared_error(a, b))


def test_classification_metrics_with_string_labels(data):
    y, score = data
    labels = np.where(y == 1, ">50K", "<=50K")
    predictions = np.where(score > 0.5, ">50K", "<=50K")
    result = classification_metrics(labels, predictions, score, ">50K")
    assert result["auc"] == pytest.approx(sk.roc_auc_score(y, score))
    assert set(result) == {"f1", "accuracy", "auc", "ece"}


def test_classification_metrics_without_scores(data):
    y, score = data
    result = classification_metrics(y, (score > 0.5).astype(int), None, 1)
    assert set(result) == {"f1", "accuracy"}


def test_regression_metrics():
    assert regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 4.0])) == {"rmse": pytest.approx(np.sqrt(2))}
