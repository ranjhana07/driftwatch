from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from driftwatch.engine.categorical import aligned_counts, chi_square, cramers_v, js_divergence
from driftwatch.engine.embedding import domain_classifier_drift, mmd_test
from driftwatch.engine.multitest import benjamini_hochberg
from driftwatch.engine.numeric import ks_test, normalized_wasserstein, psi
from driftwatch.engine.severity import level_for, severity_from_ratio
from driftwatch.schemas import MonitorConfig
from driftwatch.types import MetricResult

P_VALUE_DETECTORS = {"ks", "chi2", "mmd", "domain_classifier"}


@dataclass
class Detection:
    detector: str
    feature: str
    value: float
    p_value: float | None
    effect: float
    n_ref: int
    n_cur: int


def numeric_detections(name: str, reference: pd.Series, current: pd.Series, config: MonitorConfig) -> list[Detection]:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
    if ref.size < 2 or cur.size < 2:
        return []
    psi_value = psi(ref, cur)
    ks_stat, ks_p = ks_test(ref, cur)
    distance = normalized_wasserstein(ref, cur)
    return [
        Detection("psi", name, psi_value, None, psi_value / config.psi_threshold, ref.size, cur.size),
        Detection("ks", name, ks_stat, ks_p, ks_stat / config.ks_effect, ref.size, cur.size),
        Detection("wasserstein", name, distance, None, distance / config.wasserstein_threshold, ref.size, cur.size),
    ]


def categorical_detections(name: str, reference: pd.Series, current: pd.Series, config: MonitorConfig) -> list[Detection]:
    if len(reference) < 2 or len(current) < 2:
        return []
    _, ref_counts, cur_counts = aligned_counts(reference, current)
    statistic, p_value, _ = chi_square(ref_counts, cur_counts)
    effect_size = cramers_v(statistic, ref_counts.sum() + cur_counts.sum())
    divergence = js_divergence(ref_counts, cur_counts)
    n_ref, n_cur = int(ref_counts.sum()), int(cur_counts.sum())
    return [
        Detection("chi2", name, effect_size, p_value, effect_size / config.chi2_effect, n_ref, n_cur),
        Detection("js", name, divergence, None, divergence / config.js_threshold, n_ref, n_cur),
    ]


def embedding_detections(reference: np.ndarray, current: np.ndarray, config: MonitorConfig, seed: int = 0) -> list[Detection]:
    mmd = mmd_test(reference, current, seed=seed)
    domain = domain_classifier_drift(reference, current, seed=seed)
    auc_effect = (domain.auc - 0.5) / (config.domain_auc_threshold - 0.5)
    n_ref, n_cur = len(reference), len(current)
    return [
        Detection("mmd", "__embedding", mmd.statistic, mmd.p_value, mmd.statistic / config.mmd_effect, n_ref, n_cur),
        Detection("domain_classifier", "__embedding", domain.auc, domain.p_value, auc_effect, n_ref, n_cur),
    ]


def finalize(
    detections: list[Detection],
    config: MonitorConfig,
    model: str,
    window_start,
    window_end,
) -> list[MetricResult]:
    by_detector: dict[str, list[Detection]] = {}
    for detection in detections:
        by_detector.setdefault(detection.detector, []).append(detection)
    results = []
    for detector, group in by_detector.items():
        adjusted: list[float | None] = [None] * len(group)
        if detector in P_VALUE_DETECTORS:
            adjusted = list(benjamini_hochberg(np.array([d.p_value for d in group])))
        for detection, p_adjusted in zip(group, adjusted):
            flagged = _is_flagged(detector, detection.effect, p_adjusted, config)
            severity = severity_from_ratio(detection.effect, flagged)
            results.append(
                MetricResult(
                    model=model,
                    window_start=window_start,
                    window_end=window_end,
                    kind="drift",
                    feature=detection.feature,
                    metric=detector,
                    value=float(detection.value),
                    p_value=detection.p_value,
                    p_adjusted=None if p_adjusted is None else float(p_adjusted),
                    severity=severity,
                    level=level_for(severity),
                    flagged=flagged,
                    n_ref=detection.n_ref,
                    n_cur=detection.n_cur,
                )
            )
    return results


def _is_flagged(detector: str, effect: float, p_adjusted: float | None, config: MonitorConfig) -> bool:
    if detector not in P_VALUE_DETECTORS:
        return effect >= 1.0
    significant = p_adjusted is not None and p_adjusted < config.alpha
    if detector == "domain_classifier":
        return significant and effect >= 1.0
    return significant and effect >= config.effect_floor
