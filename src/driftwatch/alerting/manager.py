from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Callable, Sequence

from driftwatch.alerting.notifiers import Notifier
from driftwatch.alerting.rules import AlertRule
from driftwatch.engine.severity import LEVELS, level_rank
from driftwatch.store.base import Store
from driftwatch.timeutil import utcnow
from driftwatch.types import Alert, MetricResult

logger = logging.getLogger("driftwatch.alerts")

LOOKBACK = timedelta(days=1)
MESSAGE_ITEMS = 5


class AlertManager:
    def __init__(
        self,
        store: Store,
        rules: Sequence[AlertRule],
        notifiers: dict[str, Notifier],
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self.store = store
        self.rules = list(rules)
        self.notifiers = notifiers
        self.clock = clock

    def process(self, model: str, results: Sequence[MetricResult]) -> list[Alert]:
        if not results:
            return []
        now = self.clock()
        window_end = max(r.window_end for r in results)
        raised = []
        for rule in self.rules:
            if not rule.applies_to(model):
                continue
            hits = rule.select(results)
            if not hits:
                continue
            if not self._sustained(model, rule, window_end):
                continue
            if self._cooling_down(model, rule, now):
                continue
            alert = self.store.add_alert(self._build(model, rule, hits, window_end, now))
            self._deliver(alert, rule)
            raised.append(alert)
        return raised

    def _sustained(self, model: str, rule: AlertRule, window_end: datetime) -> bool:
        needed = rule.consecutive - 1
        if needed == 0:
            return True
        earlier = self.store.read_metrics(model, start=window_end - LOOKBACK, end=window_end - timedelta(microseconds=1))
        by_window: dict[datetime, list[MetricResult]] = {}
        for result in earlier:
            by_window.setdefault(result.window_end, []).append(result)
        recent = sorted(by_window, reverse=True)[:needed]
        if len(recent) < needed:
            return False
        return all(rule.select(by_window[w]) for w in recent)

    def _cooling_down(self, model: str, rule: AlertRule, now: datetime) -> bool:
        last = self.store.last_alert(model, rule.name)
        if last is None:
            return False
        return now - last.created_at < timedelta(seconds=rule.cooldown_seconds)

    @staticmethod
    def _build(model: str, rule: AlertRule, hits: Sequence[MetricResult], window_end: datetime, now: datetime) -> Alert:
        ranked = sorted(hits, key=lambda r: r.severity, reverse=True)
        parts = []
        for r in ranked[:MESSAGE_ITEMS]:
            where = f" on {r.feature}" if r.feature else ""
            parts.append(f"{r.metric}{where} = {r.value:.3g} ({r.level})")
        extra = len(ranked) - MESSAGE_ITEMS
        message = "; ".join(parts) + (f"; and {extra} more" if extra > 0 else "")
        level = LEVELS[max(level_rank(r.level) for r in hits)]
        return Alert(model=model, rule=rule.name, level=level, message=message, window_end=window_end, created_at=now)

    def _deliver(self, alert: Alert, rule: AlertRule) -> None:
        targets = rule.notifiers if rule.notifiers is not None else list(self.notifiers)
        delivered = False
        failed = False
        for name in targets:
            notifier = self.notifiers.get(name)
            if notifier is None:
                continue
            try:
                notifier.send(alert)
                delivered = True
            except Exception:
                failed = True
                logger.exception("notifier %s failed for alert %s", name, alert.rule)
        if delivered and not failed and alert.id is not None:
            self.store.mark_alert_delivered(alert.id)
