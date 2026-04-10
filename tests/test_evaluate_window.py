from __future__ import annotations

from datetime import timedelta

import pytest
from synthetic import T0, make_rows

from driftwatch.engine.evaluate import evaluate_window
from driftwatch.engine.profile import build_profile
from driftwatch.engine.window import Window
from driftwatch.schemas import MonitorConfig

CONFIG = MonitorConfig(min_samples=100, min_labeled=100)


@pytest.fixture(scope="module")
def reference():
    return Window.from_rows(make_rows(3000, seed=100))


@pytest.fixture(scope="module")
def profile(reference):
    return build_profile(reference)


def run(profile, reference, rows, config=CONFIG):
    current = Window.from_rows(rows)
    return evaluate_window(
        "churn", profile, reference, current, current.with_labels_only(), config, T0, T0 + timedelta(hours=1)
    )


def flagged(results, kind=None):
    return {(r.kind, r.feature, r.metric) for r in results if r.flagged and (kind is None or r.kind == kind)}


def test_no_drift_has_no_flags(profile, reference):
    results = run(profile, reference, make_rows(600, seed=1))
    assert flagged(results) == set()
    assert {r.kind for r in results} == {"drift", "quality", "performance"}


def test_covariate_shift_flags_numeric_detectors(profile, reference):
    results = run(profile, reference, make_rows(600, seed=2, age_shift=8))
    hits = flagged(results, "drift")
    assert {("drift", "age", "psi"), ("drift", "age", "ks"), ("drift", "age", "wasserstein")} <= hits
    assert ("drift", "income", "ks") not in hits


def test_category_shift_flags_chi2_and_js(profile, reference):
    results = run(profile, reference, make_rows(800, seed=3, plan_probs=(0.2, 0.3, 0.5)))
    hits = flagged(results, "drift")
    assert ("drift", "plan", "chi2") in hits
    assert ("drift", "plan", "js") in hits


def test_embedding_shift_flags_mmd_and_domain_classifier(profile, reference):
    results = run(profile, reference, make_rows(500, seed=4, embedding_shift=0.8))
    hits = flagged(results, "drift")
    assert ("drift", "__embedding", "mmd") in hits
    assert ("drift", "__embedding", "domain_classifier") in hits


def test_null_burst_flags_quality_check(profile, reference):
    results = run(profile, reference, make_rows(600, seed=5, null_rate=0.5))
    assert ("quality", "income", "null_rate") in flagged(results, "quality")


def test_concept_drift_only_shows_up_in_performance(profile, reference):
    results = run(profile, reference, make_rows(800, seed=6, flip_rate=0.4))
    assert flagged(results, "drift") == set()
    assert ("performance", "", "auc") in flagged(results, "performance")


def test_small_windows_are_skipped(profile, reference):
    assert run(profile, reference, make_rows(50, seed=7)) == []


def test_labels_are_optional(profile, reference):
    current = Window.from_rows(make_rows(300, seed=8, labeled=False))
    results = evaluate_window("churn", profile, reference, current, None, CONFIG, T0, T0 + timedelta(hours=1))
    assert {r.kind for r in results} == {"drift", "quality"}


def test_p_values_are_adjusted_across_features(profile, reference):
    results = run(profile, reference, make_rows(600, seed=9))
    ks = [r for r in results if r.metric == "ks"]
    assert len(ks) == 4
    assert all(r.p_adjusted >= r.p_value for r in ks)
    assert all(r.level in {"none", "low"} for r in ks)


def test_severity_grows_with_shift(profile, reference):
    mild = run(profile, reference, make_rows(600, seed=10, age_shift=5))
    strong = run(profile, reference, make_rows(600, seed=10, age_shift=15))
    pick = lambda results: next(r for r in results if r.feature == "age" and r.metric == "psi")
    assert pick(strong).severity > pick(mild).severity
    assert pick(strong).level in {"high", "critical"}


def test_small_performance_wobble_is_not_flagged(profile, reference):
    results = run(profile, reference, make_rows(300, seed=21, flip_rate=0.03))
    assert ("performance", "", "auc") not in flagged(results, "performance")


def test_performance_results_report_labeled_count(profile, reference):
    results = run(profile, reference, make_rows(400, seed=22))
    auc = next(r for r in results if r.kind == "performance" and r.metric == "auc")
    assert auc.n_cur == 400
    assert auc.n_ref == 3000
