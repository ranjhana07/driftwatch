from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from driftwatch.engine.embedding import (
    domain_classifier_drift,
    median_bandwidth,
    mmd2_unbiased,
    mmd_test,
    rbf_kernel,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(11)


def naive_mmd2(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    def k(a, b):
        return float(np.exp(-np.sum((a - b) ** 2) / (2 * bandwidth**2)))

    n, m = len(x), len(y)
    xx = sum(k(x[i], x[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    yy = sum(k(y[i], y[j]) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
    xy = sum(k(a, b) for a in x for b in y) / (n * m)
    return xx + yy - 2 * xy


def test_rbf_kernel_properties(rng):
    x = rng.normal(size=(6, 3))
    kernel = rbf_kernel(x, x, 1.3)
    assert np.allclose(np.diag(kernel), 1.0)
    assert np.allclose(kernel, kernel.T)
    assert kernel.min() > 0


def test_mmd2_matches_naive_loops(rng):
    x = rng.normal(size=(15, 4))
    y = rng.normal(0.5, 1, size=(12, 4))
    assert mmd2_unbiased(x, y, 1.7) == pytest.approx(naive_mmd2(x, y, 1.7))


def test_median_bandwidth_positive_and_scale_equivariant(rng):
    x = rng.normal(size=(50, 3))
    y = rng.normal(size=(50, 3))
    base = median_bandwidth(x, y)
    assert base > 0
    assert median_bandwidth(3 * x, 3 * y) == pytest.approx(3 * base)


def test_mmd_detects_mean_shift(rng):
    reference = rng.normal(size=(400, 5))
    current = rng.normal(0.6, 1, size=(300, 5))
    result = mmd_test(reference, current, seed=1)
    assert result.p_value < 0.02
    assert result.statistic > 0


def test_mmd_is_quiet_without_drift(rng):
    reference = rng.normal(size=(400, 5))
    current = rng.normal(size=(300, 5))
    assert mmd_test(reference, current, seed=2).p_value > 0.05


def test_mmd_p_values_roughly_uniform_under_null():
    p_values = []
    for seed in range(60):
        local = np.random.default_rng(seed)
        p_values.append(mmd_test(local.normal(size=(80, 3)), local.normal(size=(80, 3)), seed=seed).p_value)
    assert np.mean(np.array(p_values) < 0.1) < 0.25


def test_mmd_is_deterministic_for_a_seed(rng):
    x = rng.normal(size=(120, 3))
    y = rng.normal(0.2, 1, size=(100, 3))
    assert mmd_test(x, y, seed=5) == mmd_test(x, y, seed=5)


def test_domain_classifier_detects_shift(rng):
    reference = rng.normal(size=(600, 6))
    current = rng.normal(0.0, 1, size=(500, 6)) + np.array([1.0, 0, 0, 0, 0, 0])
    result = domain_classifier_drift(reference, current, seed=3)
    assert result.auc > 0.65
    assert result.p_value < 0.001


def test_domain_classifier_near_chance_without_drift(rng):
    reference = rng.normal(size=(600, 6))
    current = rng.normal(size=(500, 6))
    result = domain_classifier_drift(reference, current, seed=4)
    assert 0.44 < result.auc < 0.56


def test_domain_auc_matches_sklearn_on_same_scores(rng):
    from scipy import stats

    scores = rng.uniform(size=40)
    target = np.array([0] * 20 + [1] * 20)
    ranks = stats.rankdata(scores)
    ours = (ranks[target == 1].sum() - 20 * 21 / 2) / 400
    assert ours == pytest.approx(roc_auc_score(target, scores))
