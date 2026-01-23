from __future__ import annotations

import numpy as np
from scipy import stats


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, epsilon: float = 1e-4) -> float:
    reference = _finite(reference)
    current = _finite(current)
    edges = np.quantile(reference, np.linspace(0.0, 1.0, bins + 1))
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    expected = np.clip(ref_counts / ref_counts.sum(), epsilon, None)
    actual = np.clip(cur_counts / max(cur_counts.sum(), 1), epsilon, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    ref = np.sort(_finite(reference))
    cur = np.sort(_finite(current))
    grid = np.concatenate([ref, cur])
    cdf_ref = np.searchsorted(ref, grid, side="right") / ref.size
    cdf_cur = np.searchsorted(cur, grid, side="right") / cur.size
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    ref = _finite(reference)
    cur = _finite(current)
    statistic = ks_statistic(ref, cur)
    effective_n = ref.size * cur.size / (ref.size + cur.size)
    p_value = float(stats.kstwo.sf(statistic, np.round(effective_n)))
    return statistic, min(max(p_value, 0.0), 1.0)


def wasserstein(reference: np.ndarray, current: np.ndarray) -> float:
    ref = np.sort(_finite(reference))
    cur = np.sort(_finite(current))
    grid = np.sort(np.concatenate([ref, cur]))
    widths = np.diff(grid)
    cdf_ref = np.searchsorted(ref, grid[:-1], side="right") / ref.size
    cdf_cur = np.searchsorted(cur, grid[:-1], side="right") / cur.size
    return float(np.sum(np.abs(cdf_ref - cdf_cur) * widths))


def normalized_wasserstein(reference: np.ndarray, current: np.ndarray) -> float:
    ref = _finite(reference)
    scale = float(np.std(ref))
    if scale == 0.0:
        scale = 1.0
    return wasserstein(ref, current) / scale
