from __future__ import annotations

import numpy as np


def bonferroni(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    return np.minimum(p * p.size, 1.0)


def holm(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    scaled = (p.size - np.arange(p.size)) * p[order]
    adjusted = np.minimum(np.maximum.accumulate(scaled), 1.0)
    result = np.empty_like(p)
    result[order] = adjusted
    return result


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return p.copy()
    order = np.argsort(p)[::-1]
    ranks = np.arange(p.size, 0, -1)
    scaled = p[order] * p.size / ranks
    adjusted = np.minimum(np.minimum.accumulate(scaled), 1.0)
    result = np.empty_like(p)
    result[order] = adjusted
    return result


METHODS = {"bonferroni": bonferroni, "holm": holm, "bh": benjamini_hochberg}


def adjust(p_values: np.ndarray, method: str = "bh") -> np.ndarray:
    return METHODS[method](p_values)
