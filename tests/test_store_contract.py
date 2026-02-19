from __future__ import annotations

from datetime import timedelta

import pytest
from synthetic import T0, make_rows, to_events, to_labels

from driftwatch.schemas import ModelRegistration, MonitorConfig
from driftwatch.types import Alert, MetricResult, ModelRecord


def seed(store, n=40, **kwargs):
    rows = make_rows(n, **kwargs)
    store.add_events(to_events(rows))
    return rows


def test_model_registry(store):
    assert store.get_model("churn") is None
    record = ModelRecord(name="churn", config=MonitorConfig(window_seconds=900))
    saved = store.upsert_model(record)
    assert saved.created_at is not None
    fetched = store.get_model("churn")
    assert fetched.config.window_seconds == 900
    assert fetched.task_type == "classification"
    store.upsert_model(ModelRecord(name="churn", task_type="regression"))
    assert store.get_model("churn").task_type == "regression"
    assert store.get_model("churn").created_at == saved.created_at
    assert [m.name for m in store.list_models()] == ["churn"]


def test_registration_schema_validates_names():
    with pytest.raises(ValueError):
        ModelRegistration(name="bad name!")


def test_add_and_read_events(store):
    rows = seed(store, 40, spacing_seconds=60)
    assert store.event_count("churn") == 40
    window = store.read_events("churn", T0, T0 + timedelta(minutes=10))
    assert len(window) == 10
    assert window[0]["event_id"] == rows[0]["event_id"]
    assert window[0]["features"]["plan"] == rows[0]["features"]["plan"]
    assert window[0]["timestamp"].tzinfo is not None
    assert window[0]["embedding"] == pytest.approx(rows[0]["embedding"])


def test_duplicate_events_are_ignored(store):
    rows = make_rows(10)
    assert store.add_events(to_events(rows)) == 10
    store.add_events(to_events(rows))
    assert store.event_count("churn") == 10


def test_events_are_scoped_by_model_and_version(store):
    store.add_events(to_events(make_rows(5, model="a")))
    store.add_events(to_events(make_rows(7, model="b", seed=1)))
    end = T0 + timedelta(days=1)
    assert len(store.read_events("a", T0, end)) == 5
    assert len(store.read_events("b", T0, end)) == 7
    assert store.read_events("b", T0, end, version="2") == []


def test_labels_join_and_as_of_filter(store):
    rows = seed(store, 20, spacing_seconds=60)
    store.add_labels(to_labels(rows, delay_seconds=600))
    end = T0 + timedelta(hours=1)
    assert len(store.read_labeled_events("churn", T0, end)) == 20
    early = store.read_labeled_events("churn", T0, end, as_of=T0 + timedelta(minutes=5, seconds=30))
    assert len(early) == 0
    some = store.read_labeled_events("churn", T0, end, as_of=T0 + timedelta(minutes=15, seconds=30))
    assert 0 < len(some) < 20
    assert all(r["labeled_at"] <= T0 + timedelta(minutes=15, seconds=30) for r in some)
    assert some[0]["label"] in (0, 1)


def test_labels_for_unknown_events_do_not_appear(store):
    rows = make_rows(5)
    store.add_labels(to_labels(rows))
    assert store.read_labeled_events("churn", T0, T0 + timedelta(days=1)) == []


def test_time_bounds(store):
    assert store.time_bounds("churn") is None
    seed(store, 10, spacing_seconds=30)
    low, high = store.time_bounds("churn")
    assert low == T0
    assert high == T0 + timedelta(seconds=270)


def test_reference_round_trip(store):
    assert store.load_reference("churn") is None
    store.save_reference("churn", {"rows": [{"a": 1}], "profile": {"n_rows": 1}})
    assert store.load_reference("churn")["profile"]["n_rows"] == 1
    store.save_reference("churn", {"rows": [], "profile": {"n_rows": 0}})
    assert store.load_reference("churn")["profile"]["n_rows"] == 0


def result(end_minutes, feature="age", metric="psi", kind="drift", value=0.3, flagged=True, model="churn"):
    return MetricResult(
        model=model,
        window_start=T0,
        window_end=T0 + timedelta(minutes=end_minutes),
        kind=kind,
        feature=feature,
        metric=metric,
        value=value,
        severity=0.6,
        level="medium",
        flagged=flagged,
    )


def test_metrics_filters_and_latest(store):
    store.add_metrics(
        [
            result(60),
            result(60, feature="plan", metric="js"),
            result(120, value=0.5),
            result(120, kind="quality", metric="null_rate", feature="income", value=0.1),
        ]
    )
    assert len(store.read_metrics("churn")) == 4
    assert len(store.read_metrics("churn", kind="quality")) == 1
    assert len(store.read_metrics("churn", feature="age", metric="psi")) == 2
    assert len(store.read_metrics("churn", start=T0 + timedelta(minutes=90))) == 2
    assert len(store.read_metrics("churn", end=T0 + timedelta(minutes=60))) == 2
    latest = store.latest_metrics("churn")
    assert {r.window_end for r in latest} == {T0 + timedelta(minutes=120)}
    assert store.latest_metrics("other") == []
    first = store.read_metrics("churn", feature="age", metric="psi")[0]
    assert first.flagged is True
    assert first.level == "medium"


def test_add_metrics_empty_is_noop(store):
    store.add_metrics([])
    assert store.read_metrics("churn") == []


def alert(minutes, rule="drift-high", model="churn"):
    return Alert(
        model=model,
        rule=rule,
        level="high",
        message="psi on age",
        window_end=T0 + timedelta(minutes=minutes),
        created_at=T0 + timedelta(minutes=minutes, seconds=5),
    )


def test_alert_lifecycle(store):
    first = store.add_alert(alert(10))
    store.add_alert(alert(70))
    store.add_alert(alert(80, rule="quality"))
    assert first.id is not None
    assert [a.rule for a in store.list_alerts("churn")] == ["quality", "drift-high", "drift-high"]
    assert len(store.list_alerts("churn", limit=2)) == 2
    assert len(store.list_alerts("churn", since=T0 + timedelta(minutes=60))) == 2
    assert store.list_alerts("other") == []
    assert store.last_alert("churn", "drift-high").window_end == T0 + timedelta(minutes=70)
    assert store.last_alert("churn", "missing") is None
    store.mark_alert_delivered(first.id)
    delivered = {a.id: a.delivered for a in store.list_alerts("churn")}
    assert delivered[first.id] is True
    assert sum(delivered.values()) == 1
