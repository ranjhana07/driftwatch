from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from driftwatch.engine.categorical import MISSING, normalize

TOP_CATEGORIES = 12
OTHER = "other"


def compare_numeric(reference: pd.Series, current: pd.Series, bins: int = 20) -> dict[str, Any]:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
    if ref.size == 0:
        return {"kind": "numeric", "edges": [], "reference": [], "current": []}
    pooled = np.concatenate([ref, cur]) if cur.size else ref
    low, high = np.quantile(pooled, [0.005, 0.995])
    if high <= low:
        high = low + 1.0
    edges = np.linspace(low, high, bins + 1)
    return {
        "kind": "numeric",
        "edges": edges.tolist(),
        "reference": _density(ref, edges),
        "current": _density(cur, edges),
        "n_reference": int(ref.size),
        "n_current": int(cur.size),
    }


def _density(values: np.ndarray, edges: np.ndarray) -> list[float]:
    if values.size == 0:
        return [0.0] * (len(edges) - 1)
    clipped = np.clip(values, edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    return (counts / values.size).tolist()


def compare_categorical(reference: pd.Series, current: pd.Series) -> dict[str, Any]:
    ref = pd.Series(normalize(reference), dtype="object")
    cur = pd.Series(normalize(current), dtype="object")
    ref_share = ref.value_counts(normalize=True)
    cur_share = cur.value_counts(normalize=True) if len(cur) else pd.Series(dtype=float)
    ranked = list(ref_share.index[:TOP_CATEGORIES])
    for category in cur_share.index:
        if category not in ranked and len(ranked) < TOP_CATEGORIES + 4 and cur_share[category] >= 0.02:
            ranked.append(category)
    ref_values = [float(ref_share.get(c, 0.0)) for c in ranked]
    cur_values = [float(cur_share.get(c, 0.0)) for c in ranked]
    ref_values.append(max(0.0, 1.0 - sum(ref_values)))
    cur_values.append(max(0.0, 1.0 - sum(cur_values)) if len(cur) else 0.0)
    labels = [("(missing)" if c == MISSING else c) for c in ranked] + [OTHER]
    return {
        "kind": "categorical",
        "categories": labels,
        "reference": ref_values,
        "current": cur_values,
        "n_reference": int(len(ref)),
        "n_current": int(len(cur)),
    }
