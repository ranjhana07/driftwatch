from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler

from driftwatch.config import Settings
from driftwatch.runtime import build_scheduler
from driftwatch.store.sql import SqlStore
from driftwatch.types import ModelRecord


def test_build_scheduler_uses_default_rules_and_log_notifier():
    store = SqlStore("sqlite://", create_schema=True)
    runner = build_scheduler(Settings(database_url=None), store, scheduler=BackgroundScheduler())
    assert {r.name for r in runner.alerts.rules} == {"drift-high", "quality-medium", "performance-medium"}
    assert set(runner.alerts.notifiers) == {"log"}


def test_build_scheduler_loads_rules_file_and_sync_interval(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text("rules:\n  - name: only\n    min_level: critical\n")
    settings = Settings(database_url=None, alert_rules_path=path, sync_interval_seconds=5)
    store = SqlStore("sqlite://", create_schema=True)
    store.upsert_model(ModelRecord(name="m"))
    runner = build_scheduler(settings, store, scheduler=BackgroundScheduler())
    assert [r.name for r in runner.alerts.rules] == ["only"]
    assert runner.sync_interval_seconds == 5
    runner.sync_jobs()
    assert [j.id for j in runner.scheduler.get_jobs()] == ["monitor:m"]
