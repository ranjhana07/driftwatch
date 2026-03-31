from __future__ import annotations

from datetime import datetime
from typing import Any

from driftwatch.engine.detectors import (
    Detection,
    categorical_detections,
    embedding_detections,
    finalize,
    numeric_detections,
)
from driftwatch.engine.profile import ReferenceProfile, performance_with_errors
from driftwatch.engine.quality import quality_findings
from driftwatch.engine.severity import level_for, severity_from_ratio
from driftwatch.engine.window import Window
from driftwatch.schemas import MonitorConfig
from driftwatch.types import MetricResult

NOISE_MULTIPLE = 2.0


def evaluate_window(
    model: str,
    profile: ReferenceProfile,
    reference: Window,
    current: Window,
    labeled: Window | None,
    config: MonitorConfig,
    window_start: datetime,
    window_end: datetime,
    task_type: str = "classification",
    positive_label: Any = 1,
    seed: int = 0,
) -> list[MetricResult]:
    if current.size < config.min_samples:
        return []
    detections: list[Detection] = []
    for name, feature in profile.monitored.items():
        if name not in current.frame.columns or name not in reference.frame.columns:
            continue
        if feature.kind == "numeric":
            detections += numeric_detections(name, reference.frame[name], current.frame[name], config)
        else:
            detections += categorical_detections(name, reference.frame[name], current.frame[name], config)
    if reference.embeddings is not None and current.embeddings is not None:
        if reference.embeddings.shape[1] == current.embeddings.shape[1]:
            detections += embedding_detections(reference.embeddings, current.embeddings, config, seed=seed)
    results = finalize(detections, config, model, window_start, window_end)
    results += _quality_results(model, profile, current, config, window_start, window_end)
    if labeled is not None:
        results += _performance_results(
            model, profile, labeled, config, task_type, positive_label, window_start, window_end
        )
    return results


def _quality_results(
    model: str,
    profile: ReferenceProfile,
    current: Window,
    config: MonitorConfig,
    window_start: datetime,
    window_end: datetime,
) -> list[MetricResult]:
    results = []
    for finding in quality_findings(profile, current, config):
        if finding.threshold > 0:
            ratio = finding.value / finding.threshold
        else:
            ratio = 4.0 if finding.value > 0 else 0.0
        severity = severity_from_ratio(ratio, finding.flagged)
        results.append(
            MetricResult(
                model=model,
                window_start=window_start,
                window_end=window_end,
                kind="quality",
                feature=finding.feature,
                metric=finding.metric,
                value=finding.value,
                severity=severity,
                level=level_for(severity),
                flagged=finding.flagged,
                n_cur=finding.n,
            )
        )
    return results


def _performance_results(
    model: str,
    profile: ReferenceProfile,
    labeled: Window,
    config: MonitorConfig,
    task_type: str,
    positive_label: Any,
    window_start: datetime,
    window_end: datetime,
) -> list[MetricResult]:
    if labeled.size < config.min_labeled or not profile.baseline:
        return []
    current = performance_with_errors(labeled, task_type, positive_label)
    results = []
    for metric, (value, error) in current.items():
        baseline = profile.baseline.get(metric)
        if baseline is None:
            continue
        ratio, trackable = _performance_effect(metric, baseline, value, config)
        change = abs(value - baseline)
        flagged = trackable and ratio >= 1.0 and change > NOISE_MULTIPLE * error
        severity = severity_from_ratio(ratio, flagged)
        results.append(
            MetricResult(
                model=model,
                window_start=window_start,
                window_end=window_end,
                kind="performance",
                metric=metric,
                value=float(value),
                severity=severity,
                level=level_for(severity),
                flagged=flagged,
                n_ref=profile.n_rows,
                n_cur=labeled.size,
            )
        )
    return results


def _performance_effect(metric: str, baseline: float, value: float, config: MonitorConfig) -> tuple[float, bool]:
    if metric in ("auc", "f1"):
        return (baseline - value) / config.performance_drop, True
    if metric == "ece":
        return (value - baseline) / config.calibration_increase, True
    if metric == "rmse":
        return (value / baseline - 1.0) / (config.rmse_ratio - 1.0) if baseline > 0 else 0.0, True
    return 0.0, False
