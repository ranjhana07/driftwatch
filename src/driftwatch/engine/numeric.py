from __future__ import annotations

import numpy as np


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
