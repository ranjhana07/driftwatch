from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class MMDResult:
    statistic: float
    p_value: float
    bandwidth: float


@dataclass(frozen=True)
class DomainResult:
    auc: float
    p_value: float


def rbf_kernel(x: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
    return np.exp(-cdist(x, y, "sqeuclidean") / (2.0 * bandwidth**2))


def median_bandwidth(x: np.ndarray, y: np.ndarray, max_points: int = 400, seed: int = 0) -> float:
    pooled = np.vstack([x, y])
    if pooled.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        pooled = pooled[rng.choice(pooled.shape[0], max_points, replace=False)]
    distances = cdist(pooled, pooled)
    upper = distances[np.triu_indices_from(distances, k=1)]
    median = float(np.median(upper)) if upper.size else 1.0
    return median if median > 0 else 1.0


def mmd2_unbiased(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    n, m = x.shape[0], y.shape[0]
    kxx = rbf_kernel(x, x, bandwidth)
    kyy = rbf_kernel(y, y, bandwidth)
    kxy = rbf_kernel(x, y, bandwidth)
    term_x = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
    term_y = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
    return float(term_x + term_y - 2.0 * kxy.mean())


def mmd_test(
    reference: np.ndarray,
    current: np.ndarray,
    n_permutations: int = 100,
    max_samples: int = 400,
    seed: int = 0,
) -> MMDResult:
    rng = np.random.default_rng(seed)
    x = _subsample(np.asarray(reference, dtype=float), max_samples, rng)
    y = _subsample(np.asarray(current, dtype=float), max_samples, rng)
    bandwidth = median_bandwidth(x, y, seed=seed)
    pooled = np.vstack([x, y])
    n, m = x.shape[0], y.shape[0]
    kernel = rbf_kernel(pooled, pooled, bandwidth)
    np.fill_diagonal(kernel, 0.0)

    def statistic(in_x: np.ndarray) -> np.ndarray:
        in_y = 1.0 - in_x
        xx = np.einsum("pi,ij,pj->p", in_x, kernel, in_x)
        yy = np.einsum("pi,ij,pj->p", in_y, kernel, in_y)
        xy = np.einsum("pi,ij,pj->p", in_x, kernel, in_y)
        return xx / (n * (n - 1)) + yy / (m * (m - 1)) - 2.0 * xy / (n * m)

    observed_mask = np.zeros((1, n + m))
    observed_mask[0, :n] = 1.0
    observed = float(statistic(observed_mask)[0])
    permuted_masks = np.zeros((n_permutations, n + m))
    for row in permuted_masks:
        row[rng.permutation(n + m)[:n]] = 1.0
    permuted = statistic(permuted_masks)
    p_value = float((1 + np.sum(permuted >= observed)) / (1 + n_permutations))
    return MMDResult(observed, p_value, bandwidth)


def domain_classifier_drift(
    reference: np.ndarray,
    current: np.ndarray,
    max_samples: int = 1000,
    folds: int = 5,
    seed: int = 0,
) -> DomainResult:
    rng = np.random.default_rng(seed)
    x = _subsample(np.asarray(reference, dtype=float), max_samples, rng)
    y = _subsample(np.asarray(current, dtype=float), max_samples, rng)
    size = min(x.shape[0], y.shape[0])
    x, y = x[:size], y[:size]
    features = StandardScaler().fit_transform(np.vstack([x, y]))
    target = np.concatenate([np.zeros(size), np.ones(size)])
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_predict(
        LogisticRegression(max_iter=200), features, target, cv=splitter, method="predict_proba"
    )[:, 1]
    ranks = stats.rankdata(scores)
    auc = float((ranks[target == 1].sum() - size * (size + 1) / 2) / (size * size))
    p_value = float(stats.mannwhitneyu(scores[target == 1], scores[target == 0], alternative="greater").pvalue)
    return DomainResult(auc, p_value)


def _subsample(values: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if values.shape[0] <= limit:
        return values
    return values[rng.choice(values.shape[0], limit, replace=False)]
