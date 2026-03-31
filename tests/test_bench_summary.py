from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driftwatch.bench.detection import DETECTORS, detector_flags, summarize
from driftwatch.bench.overhead import make_payloads, percentiles
from driftwatch.types import MetricResult
from datetime import datetime, timezone

T = datetime(2026, 1, 1, tzinfo=timezone.utc)


def result(kind, metric, flagged, feature="x"):
    return MetricResult(window_start=T, window_end=T, kind=kind, feature=feature, metric=metric, value=1.0, flagged=flagged)


def test_detector_flags_map_results_to_detectors():
    flags = detector_flags(
        [result("drift", "psi", True), result("drift", "ks", False), result("quality", "null_rate", True), result("performance", "auc", True)]
    )
    assert flags["psi"] and flags["null_rate"] and flags["performance"] and flags["any"]
    assert not flags["ks"] and not flags["mmd"]
    assert set(flags) == set(DETECTORS)


def test_other_quality_flags_only_count_for_any():
    flags = detector_flags([result("quality", "missing_key_rate", True)])
    assert flags["any"] and not flags["null_rate"]


def windows_frame():
    rows = []
    for seed in range(2):
        for hours in (-6, -2, 2, 6, 10):
            phase = "pre" if hours <= 0 else "post"
            rows.append(
                {
                    "scenario": "covariate_shift",
                    "seed": seed,
                    "hours_after_onset": float(hours),
                    "phase": phase,
                    **{d: False for d in DETECTORS},
                }
            )
    frame = pd.DataFrame(rows)
    frame.loc[(frame.hours_after_onset >= 6), "psi"] = True
    frame.loc[(frame.hours_after_onset == -2) & (frame.seed == 0), "psi"] = True
    return frame


def test_summary_rates_and_delay():
    summary = summarize(windows_frame(), onset_hours=0)
    row = summary[(summary.detector == "psi")].iloc[0]
    assert row.negative_windows == 4
    assert row.false_positive_rate == pytest.approx(0.25)
    assert row.positive_windows == 6
    assert row.true_positive_rate == pytest.approx(4 / 6)
    assert row.runs_detected == 2
    assert row.median_delay_hours == pytest.approx(6.0)
    silent = summary[(summary.detector == "ks")].iloc[0]
    assert silent.true_positive_rate == 0 and np.isnan(silent.median_delay_hours)


def test_control_scenario_has_no_delay():
    frame = windows_frame().assign(scenario="none", phase="pre")
    row = summarize(frame, onset_hours=0).query("detector == 'psi'").iloc[0]
    assert row.runs_detected == 0
    assert np.isnan(row.true_positive_rate)


def test_percentiles_and_payloads():
    stats = percentiles(list(range(1, 101)))
    assert stats["p50"] == 51 and stats["max"] == 100
    payloads = make_payloads(5)
    assert len(payloads) == 5 and len(payloads[0]["embedding"]) == 8
    assert len(payloads[0]["features"]) == 10
