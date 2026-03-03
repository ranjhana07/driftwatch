from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Callable

from apscheduler.schedulers.base import BaseScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from prometheus_client import CollectorRegistry, Counter, Histogram

from driftwatch.alerting.manager import AlertManager
from driftwatch.monitor import Monitor, MonitorError
from driftwatch.store.base import Store
from driftwatch.timeutil import utcnow
from driftwatch.types import MetricResult

logger = logging.getLogger("driftwatch.scheduler")

SYNC_JOB_ID = "sync-models"
JOB_PREFIX = "monitor:"


class SchedulerMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.runs = Counter("driftwatch_monitor_runs_total", "Monitor runs", ["model", "status"], registry=self.registry)
        self.duration = Histogram(
            "driftwatch_monitor_run_seconds",
            "Time spent evaluating one window",
            ["model"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
            registry=self.registry,
        )
        self.alerts = Counter("driftwatch_alerts_raised_total", "Alerts raised", ["model", "level"], registry=self.registry)


class MonitorScheduler:
    def __init__(
        self,
        store: Store,
        monitor: Monitor,
        alerts: AlertManager,
        scheduler: BaseScheduler | None = None,
        metrics: SchedulerMetrics | None = None,
        sync_interval_seconds: int = 60,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self.store = store
        self.monitor = monitor
        self.alerts = alerts
        self.scheduler = scheduler or BlockingScheduler(timezone="UTC")
        self.metrics = metrics or SchedulerMetrics()
        self.sync_interval_seconds = sync_interval_seconds
        self.clock = clock

    def run_model(self, name: str, window_end: datetime | None = None) -> list[MetricResult]:
        started = time.perf_counter()
        try:
            results = self.monitor.run(name, window_end=window_end or self.clock())
        except MonitorError as exc:
            logger.info("skipping %s: %s", name, exc)
            self.metrics.runs.labels(name, "skipped").inc()
            return []
        except Exception:
            logger.exception("monitor run failed for %s", name)
            self.metrics.runs.labels(name, "error").inc()
            return []
        self.metrics.duration.labels(name).observe(time.perf_counter() - started)
        self.metrics.runs.labels(name, "ok" if results else "empty").inc()
        for alert in self.alerts.process(name, results):
            self.metrics.alerts.labels(name, alert.level).inc()
        return results

    def run_cycle(self) -> dict[str, int]:
        return {m.name: len(self.run_model(m.name)) for m in self.store.list_models()}

    def sync_jobs(self) -> None:
        wanted = {}
        for record in self.store.list_models():
            wanted[f"{JOB_PREFIX}{record.name}"] = (record.name, record.config.interval_seconds)
        for job in self.scheduler.get_jobs():
            if job.id.startswith(JOB_PREFIX) and job.id not in wanted:
                job.remove()
        existing = {job.id: job for job in self.scheduler.get_jobs()}
        for job_id, (name, interval) in wanted.items():
            current = existing.get(job_id)
            if current is not None and current.trigger.interval.total_seconds() == interval:
                continue
            self.scheduler.add_job(
                self.run_model,
                "interval",
                seconds=interval,
                args=[name],
                id=job_id,
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )

    def start(self) -> None:
        self.sync_jobs()
        self.scheduler.add_job(
            self.sync_jobs,
            "interval",
            seconds=self.sync_interval_seconds,
            id=SYNC_JOB_ID,
            replace_existing=True,
            max_instances=1,
        )
        self.scheduler.start()
