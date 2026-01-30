from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
from scipy import stats

MISSING = "__missing__"


def normalize(values: Iterable[object]) -> list[str]:
    out = []
    for value in values:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            out.append(MISSING)
        else:
            out.append(str(value))
    return out


def aligned_counts(reference: Iterable[object], current: Iterable[object]) -> tuple[list[str], np.ndarray, np.ndarray]:
    ref = Counter(normalize(reference))
    cur = Counter(normalize(current))
    categories = sorted(set(ref) | set(cur))
    ref_counts = np.array([ref.get(c, 0) for c in categories], dtype=float)
    cur_counts = np.array([cur.get(c, 0) for c in categories], dtype=float)
    return categories, ref_counts, cur_counts


def chi_square(ref_counts: np.ndarray, cur_counts: np.ndarray) -> tuple[float, float, int]:
    table = np.vstack([ref_counts, cur_counts]).astype(float)
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2 or table[0].sum() == 0 or table[1].sum() == 0:
        return 0.0, 1.0, 0
    total = table.sum()
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / total
    statistic = float(np.sum((table - expected) ** 2 / expected))
    dof = table.shape[1] - 1
    return statistic, float(stats.chi2.sf(statistic, dof)), dof


def cramers_v(statistic: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return float(np.sqrt(statistic / total))


def js_divergence(ref_counts: np.ndarray, cur_counts: np.ndarray, base: float = 2.0) -> float:
    p = np.asarray(ref_counts, dtype=float)
    q = np.asarray(cur_counts, dtype=float)
    if p.sum() == 0 or q.sum() == 0:
        return 0.0
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return float((0.5 * kl(p, m) + 0.5 * kl(q, m)) / np.log(base))
