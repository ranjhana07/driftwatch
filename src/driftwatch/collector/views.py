from __future__ import annotations

from typing import Any, Sequence

from driftwatch.engine.severity import level_for
from driftwatch.types import MetricResult


def heatmap(results: Sequence[MetricResult]) -> dict[str, Any]:
    drift = [r for r in results if r.kind == "drift"]
    windows = sorted({r.window_end for r in drift})
    features = sorted({r.feature for r in drift}, key=lambda f: (f.startswith("__"), f))
    window_index = {w: i for i, w in enumerate(windows)}
    feature_index = {f: i for i, f in enumerate(features)}
    cells: dict[tuple[int, int], MetricResult] = {}
    for r in drift:
        key = (window_index[r.window_end], feature_index[r.feature])
        if key not in cells or r.severity > cells[key].severity:
            cells[key] = r
    return {
        "windows": [w.isoformat() for w in windows],
        "features": features,
        "cells": [
            [x, y, round(r.severity, 4), int(r.flagged), r.metric] for (x, y), r in sorted(cells.items())
        ],
    }


def model_status(latest: Sequence[MetricResult]) -> dict[str, Any]:
    if not latest:
        return {"level": "none", "flagged": 0, "window_end": None, "worst": None}
    flagged = [r for r in latest if r.flagged]
    worst = max(latest, key=lambda r: r.severity)
    return {
        "level": level_for(worst.severity) if flagged else "none",
        "flagged": len(flagged),
        "window_end": max(r.window_end for r in latest).isoformat(),
        "worst": {"kind": worst.kind, "feature": worst.feature, "metric": worst.metric, "severity": worst.severity},
    }


def performance_series(results: Sequence[MetricResult]) -> dict[str, list[list[Any]]]:
    series: dict[str, list[list[Any]]] = {}
    for r in results:
        if r.kind == "performance":
            series.setdefault(r.metric, []).append([r.window_end.isoformat(), r.value, int(r.flagged)])
    return series

