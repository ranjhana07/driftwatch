from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from driftwatch.engine.categorical import MISSING, normalize
from driftwatch.engine.profile import ReferenceProfile
from driftwatch.engine.window import Window
from driftwatch.schemas import MonitorConfig

RANGE_MARGIN = 0.05


@dataclass(frozen=True)
class Finding:
    feature: str
    metric: str
    value: float
    threshold: float
    flagged: bool
    n: int


def null_rate_findings(profile: ReferenceProfile, window: Window, config: MonitorConfig) -> list[Finding]:
    findings = []
    for name, reference in profile.monitored.items():
        if name not in window.frame.columns:
            continue
        current = float(window.frame[name].isna().mean())
        increase = current - reference.null_rate
        findings.append(
            Finding(name, "null_rate", current, reference.null_rate + config.null_rate_increase, increase > config.null_rate_increase, window.size)
        )
    return findings


def range_findings(profile: ReferenceProfile, window: Window, config: MonitorConfig) -> list[Finding]:
    findings = []
    for name, reference in profile.monitored.items():
        if reference.kind != "numeric" or reference.minimum is None or name not in window.frame.columns:
            continue
        values = pd.to_numeric(window.frame[name], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        span = (reference.maximum - reference.minimum) or 1.0
        low = reference.minimum - RANGE_MARGIN * span
        high = reference.maximum + RANGE_MARGIN * span
        rate = float(np.mean((values < low) | (values > high)))
        findings.append(Finding(name, "out_of_range_rate", rate, config.out_of_range_rate, rate > config.out_of_range_rate, int(values.size)))
    return findings


def schema_findings(profile: ReferenceProfile, window: Window, config: MonitorConfig) -> list[Finding]:
    rows = window.raw_features
    if not rows:
        return []
    n = len(rows)
    findings = []
    for name, reference in profile.features.items():
        missing = sum(1 for row in rows if name not in row) / n
        findings.append(Finding(name, "missing_key_rate", missing, config.schema_violation_rate, missing > config.schema_violation_rate, n))
        if reference.kind == "numeric":
            bad = 0
            present = 0
            for row in rows:
                value = row.get(name)
                if value is None:
                    continue
                present += 1
                if isinstance(value, str) or isinstance(value, bool):
                    bad += 1
            rate = bad / present if present else 0.0
            findings.append(Finding(name, "type_mismatch_rate", rate, config.schema_violation_rate, rate > config.schema_violation_rate, present))
    unexpected = {key for row in rows for key in row if key not in profile.features}
    findings.append(Finding("", "unexpected_features", float(len(unexpected)), 0.0, len(unexpected) > 0, n))
    return findings


def cardinality_findings(profile: ReferenceProfile, window: Window, config: MonitorConfig) -> list[Finding]:
    findings = []
    for name, reference in profile.monitored.items():
        if reference.kind != "categorical" or name not in window.frame.columns:
            continue
        values = [v for v in normalize(window.frame[name]) if v != MISSING]
        if not values:
            continue
        if reference.cardinality <= len(reference.categories):
            unseen = sum(1 for v in values if v not in reference.categories) / len(values)
            findings.append(Finding(name, "unseen_category_rate", unseen, config.unseen_category_rate, unseen > config.unseen_category_rate, len(values)))
        if reference.cardinality >= 2:
            ratio = len(set(values)) / reference.cardinality
            findings.append(Finding(name, "cardinality_ratio", ratio, config.cardinality_ratio, ratio > config.cardinality_ratio, len(values)))
    return findings


def quality_findings(profile: ReferenceProfile, window: Window, config: MonitorConfig) -> list[Finding]:
    if window.size == 0:
        return []
    return [
        *null_rate_findings(profile, window, config),
        *range_findings(profile, window, config),
        *schema_findings(profile, window, config),
        *cardinality_findings(profile, window, config),
    ]
