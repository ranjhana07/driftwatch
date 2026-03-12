from __future__ import annotations

import numpy as np
from scipy import stats


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def quantile_edges(reference: np.ndarray, bins: int = 10) -> np.ndarray:
    interior = np.quantile(_finite(reference), np.linspace(0.0, 1.0, bins + 1)[1:-1])
    return np.unique(interior)


def _bin_shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.searchsorted(edges, values, side="left"), minlength=edges.size + 1)
    return counts / max(counts.sum(), 1)


def psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-4,
    edges: np.ndarray | None = None,
) -> float:
    reference = _finite(reference)
    current = _finite(current)
    if edges is None:
        edges = quantile_edges(reference, bins)
    expected = np.clip(_bin_shares(reference, edges), epsilon, None)
    actual = np.clip(_bin_shares(current, edges), epsilon, None)
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
