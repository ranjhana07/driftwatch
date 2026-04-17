from __future__ import annotations

from datetime import timedelta

import pytest
from apscheduler.schedulers.background import BackgroundScheduler
from synthetic import T0, make_rows, to_events, to_labels

from driftwatch.alerting import AlertManager, default_rules
from driftwatch.monitor import Monitor, reference_rows
from driftwatch.scheduler import JOB_PREFIX, MonitorScheduler, SchedulerMetrics
from driftwatch.schemas import MonitorConfig
from driftwatch.types import ModelRecord


class Recorder:
    name = "rec"

    def __init__(self):
        self.sent = []

    def send(self, alert):
        self.sent.append(alert)


@pytest.fixture
def setup(store):
    store.upsert_model(
        ModelRecord(name="churn", config=MonitorConfig(min_samples=100, min_labeled=50, interval_seconds=120))
    )
    monitor = Monitor(store)
    monitor.set_reference("churn", reference_rows(make_rows(2000, seed=700, start=T0 - timedelta(days=1))))
    recorder = Recorder()
    now = T0 + timedelta(hours=1)
    alerts = AlertManager(store, default_rules(), {"rec": recorder}, clock=lambda: now)
    scheduler = MonitorScheduler(
        store, monitor, alerts, scheduler=BackgroundScheduler(timezone="UTC"), metrics=SchedulerMetrics(), clock=lambda: now
    )
    return scheduler, recorder, store


def test_run_model_persists_and_raises_alerts(setup):
    scheduler, recorder, store = setup
    rows = make_rows(450, seed=1, spacing_seconds=8, age_shift=12)
    store.add_events(to_events(rows))
    store.add_labels(to_labels(rows))
    results = scheduler.run_model("churn")
    assert results
    assert store.read_metrics("churn")
    assert recorder.sent
    assert recorder.sent[0].rule == "drift-high"
    assert any(a.delivered for a in store.list_alerts("churn"))


def test_clean_traffic_raises_no_alerts(setup):
    scheduler, recorder, store = setup
    store.add_events(to_events(make_rows(450, seed=2, spacing_seconds=8)))
    scheduler.run_model("churn")
    assert recorder.sent == []


def test_missing_reference_is_skipped_not_fatal(store):
    store.upsert_model(ModelRecord(name="fresh"))
    alerts = AlertManager(store, default_rules(), {})
    scheduler = MonitorScheduler(store, Monitor(store), alerts, scheduler=BackgroundScheduler(), metrics=SchedulerMetrics())
    assert scheduler.run_model("fresh") == []
    assert scheduler.run_cycle() == {"fresh": 0}


def test_metrics_count_runs(setup):
    scheduler, _, store = setup
    store.add_events(to_events(make_rows(450, seed=3, spacing_seconds=8, age_shift=12)))
    scheduler.run_model("churn")
    sample = scheduler.metrics.runs.labels("churn", "ok")._value.get()
    assert sample == 1


def test_sync_jobs_follows_registered_models(setup):
    scheduler, _, store = setup
    scheduler.sync_jobs()
    jobs = {j.id: j for j in scheduler.scheduler.get_jobs()}
    assert set(jobs) == {f"{JOB_PREFIX}churn"}
    assert jobs[f"{JOB_PREFIX}churn"].trigger.interval == timedelta(seconds=120)
    store.upsert_model(ModelRecord(name="fraud", config=MonitorConfig(interval_seconds=30)))
    store.upsert_model(ModelRecord(name="churn", config=MonitorConfig(interval_seconds=240)))
    scheduler.sync_jobs()
    jobs = {j.id: j for j in scheduler.scheduler.get_jobs()}
    assert set(jobs) == {f"{JOB_PREFIX}churn", f"{JOB_PREFIX}fraud"}
    assert jobs[f"{JOB_PREFIX}churn"].trigger.interval == timedelta(seconds=240)
    assert jobs[f"{JOB_PREFIX}fraud"].trigger.interval == timedelta(seconds=30)


def test_compact_all_merges_local_parquet_parts(tmp_path):
    from driftwatch.store.duck import DuckStore

    duck = DuckStore(tmp_path)
    duck.upsert_model(ModelRecord(name="churn"))
    rows = make_rows(60)
    for i in range(0, 60, 20):
        duck.add_events(to_events(rows[i : i + 20]))
    runner = MonitorScheduler(
        duck, Monitor(duck), AlertManager(duck, [], {}), scheduler=BackgroundScheduler(), metrics=SchedulerMetrics()
    )
    assert runner.compact_all() == 3
    assert duck.event_count("churn") == 60
    duck.close()


def test_compact_all_is_a_noop_for_sql_stores(setup):
    scheduler, _, _ = setup
    assert scheduler.compact_all() == 0
