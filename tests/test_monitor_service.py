from __future__ import annotations

from datetime import timedelta

import pytest
from synthetic import T0, make_rows, to_events, to_labels

from driftwatch.monitor import Monitor, MonitorError, reference_rows
from driftwatch.schemas import MonitorConfig
from driftwatch.types import ModelRecord


@pytest.fixture
def monitor(store):
    store.upsert_model(ModelRecord(name="churn", config=MonitorConfig(window_seconds=3600, min_samples=100, min_labeled=50)))
    service = Monitor(store)
    reference = make_rows(2500, seed=500, start=T0 - timedelta(days=2))
    service.set_reference("churn", reference_rows(reference))
    return service


def ingest(store, rows, label_delay=60):
    store.add_events(to_events(rows))
    store.add_labels(to_labels(rows, delay_seconds=label_delay))


def test_run_requires_known_model_and_reference(store):
    service = Monitor(store)
    with pytest.raises(MonitorError):
        service.run("nope")
    store.upsert_model(ModelRecord(name="churn"))
    with pytest.raises(MonitorError):
        service.run("churn")


def test_set_reference_stores_profile(monitor, store):
    payload = store.load_reference("churn")
    assert payload["profile"]["n_rows"] == 2500
    assert len(payload["rows"]) == 2500
    assert monitor.reference("churn").profile.features["age"].kind == "numeric"


def test_run_persists_results_for_the_window(monitor, store):
    ingest(store, make_rows(400, seed=1, spacing_seconds=8))
    results = monitor.run("churn", window_end=T0 + timedelta(hours=1))
    assert results
    stored = store.read_metrics("churn")
    assert len(stored) == len(results)
    assert {r.window_end for r in stored} == {T0 + timedelta(hours=1)}
    assert not any(r.flagged for r in stored if r.kind == "drift")


def test_run_flags_injected_shift(monitor, store):
    ingest(store, make_rows(450, seed=2, spacing_seconds=8, age_shift=10))
    results = monitor.run("churn", window_end=T0 + timedelta(hours=1))
    assert any(r.flagged and r.feature == "age" and r.metric == "psi" for r in results)


def test_labels_arriving_late_are_ignored_by_as_of(monitor, store):
    ingest(store, make_rows(450, seed=3, spacing_seconds=8), label_delay=7200)
    results = monitor.run("churn", window_end=T0 + timedelta(hours=1), persist=False)
    assert not any(r.kind == "performance" for r in results)
    later = monitor.run("churn", window_end=T0 + timedelta(hours=1), persist=False)
    assert later == results
    assert store.read_metrics("churn") == []


def test_empty_window_returns_nothing(monitor):
    assert monitor.run("churn", window_end=T0 + timedelta(days=30)) == []


def test_backfill_walks_windows(monitor, store):
    ingest(store, make_rows(900, seed=4, spacing_seconds=8))
    windows = monitor.backfill("churn", T0, T0 + timedelta(hours=2), step_seconds=1800)
    assert windows >= 2
    ends = sorted({r.window_end for r in store.read_metrics("churn")})
    assert ends[0] == T0 + timedelta(hours=1)
    assert all((b - a) == timedelta(minutes=30) for a, b in zip(ends, ends[1:]))


def test_reference_is_cached_until_replaced(monitor):
    first = monitor.reference("churn")
    assert monitor.reference("churn") is first
    monitor.set_reference("churn", reference_rows(make_rows(300, seed=9)))
    assert monitor.reference("churn") is not first
