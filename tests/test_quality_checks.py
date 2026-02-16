from __future__ import annotations

import pytest
from synthetic import make_rows

from driftwatch.engine.profile import build_profile
from driftwatch.engine.quality import quality_findings
from driftwatch.engine.window import Window
from driftwatch.schemas import MonitorConfig


@pytest.fixture(scope="module")
def profile():
    return build_profile(Window.from_rows(make_rows(3000, seed=1)))


def by_key(findings):
    return {(f.feature, f.metric): f for f in findings}


def test_clean_window_raises_no_flags(profile):
    window = Window.from_rows(make_rows(600, seed=2))
    flagged = [f for f in quality_findings(profile, window, MonitorConfig()) if f.flagged]
    assert flagged == []


def test_null_burst_is_flagged(profile):
    window = Window.from_rows(make_rows(600, seed=3, null_rate=0.4))
    findings = by_key(quality_findings(profile, window, MonitorConfig()))
    finding = findings[("income", "null_rate")]
    assert finding.flagged
    assert finding.value == pytest.approx(0.4, abs=0.07)
    assert not findings[("age", "null_rate")].flagged


def test_null_rate_relative_to_reference_baseline():
    rows = make_rows(2000, seed=4, null_rate=0.3)
    profile = build_profile(Window.from_rows(rows))
    window = Window.from_rows(make_rows(500, seed=5, null_rate=0.3))
    findings = by_key(quality_findings(profile, window, MonitorConfig()))
    assert not findings[("income", "null_rate")].flagged


def test_out_of_range_values_are_flagged(profile):
    rows = make_rows(400, seed=6)
    for row in rows[:100]:
        row["features"]["age"] = 500.0
    findings = by_key(quality_findings(profile, Window.from_rows(rows), MonitorConfig()))
    assert findings[("age", "out_of_range_rate")].flagged
    assert findings[("age", "out_of_range_rate")].value == pytest.approx(0.25)


def test_missing_key_and_type_mismatch(profile):
    rows = make_rows(200, seed=7)
    for row in rows[:20]:
        del row["features"]["tenure"]
    for row in rows[20:40]:
        row["features"]["age"] = "unknown"
    findings = by_key(quality_findings(profile, Window.from_rows(rows), MonitorConfig()))
    assert findings[("tenure", "missing_key_rate")].value == pytest.approx(0.1)
    assert findings[("tenure", "missing_key_rate")].flagged
    assert findings[("age", "type_mismatch_rate")].flagged


def test_unexpected_features_reported(profile):
    rows = make_rows(50, seed=8)
    rows[0]["features"]["debug_flag"] = 1
    rows[1]["features"]["other"] = 2
    finding = by_key(quality_findings(profile, Window.from_rows(rows), MonitorConfig()))[("", "unexpected_features")]
    assert finding.value == 2
    assert finding.flagged


def test_unseen_category_flagged(profile):
    rows = make_rows(300, seed=9)
    for row in rows[:30]:
        row["features"]["plan"] = "enterprise"
    findings = by_key(quality_findings(profile, Window.from_rows(rows), MonitorConfig()))
    assert findings[("plan", "unseen_category_rate")].flagged
    assert findings[("plan", "unseen_category_rate")].value == pytest.approx(0.1)


def test_cardinality_explosion_flagged(profile):
    rows = make_rows(400, seed=10)
    for i, row in enumerate(rows):
        row["features"]["region"] = f"r{i}"
    findings = by_key(quality_findings(profile, Window.from_rows(rows), MonitorConfig()))
    assert findings[("region", "cardinality_ratio")].flagged


def test_empty_window_has_no_findings(profile):
    assert quality_findings(profile, Window.from_rows([]), MonitorConfig()) == []
