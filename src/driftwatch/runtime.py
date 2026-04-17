from __future__ import annotations

from apscheduler.schedulers.base import BaseScheduler

from driftwatch.alerting import AlertManager, build_notifiers, default_rules, load_rules
from driftwatch.config import Settings
from driftwatch.monitor import Monitor
from driftwatch.scheduler import MonitorScheduler, SchedulerMetrics
from driftwatch.store.base import Store


def build_scheduler(
    settings: Settings,
    store: Store,
    scheduler: BaseScheduler | None = None,
    metrics: SchedulerMetrics | None = None,
) -> MonitorScheduler:
    rules = load_rules(settings.alert_rules_path) if settings.alert_rules_path else default_rules()
    alerts = AlertManager(store, rules, build_notifiers(settings))
    return MonitorScheduler(
        store,
        Monitor(store),
        alerts,
        scheduler=scheduler,
        metrics=metrics,
        sync_interval_seconds=settings.sync_interval_seconds,
        compact_interval_seconds=settings.compact_interval_seconds,
    )
