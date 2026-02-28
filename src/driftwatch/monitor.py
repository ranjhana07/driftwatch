from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Sequence

from driftwatch.engine.evaluate import evaluate_window
from driftwatch.engine.profile import ReferenceProfile, build_profile
from driftwatch.engine.window import Window
from driftwatch.schemas import ReferenceRow
from driftwatch.store.base import Store
from driftwatch.timeutil import as_utc, utcnow
from driftwatch.types import MetricResult


class MonitorError(Exception):
    pass


@dataclass
class ReferenceBundle:
    version: str
    profile: ReferenceProfile
    window: Window


class Monitor:
    def __init__(self, store: Store) -> None:
        self.store = store
        self._references: dict[str, ReferenceBundle] = {}

    def set_reference(self, model: str, rows: Sequence[ReferenceRow]) -> ReferenceProfile:
        record = self.store.get_model(model)
        if record is None:
            raise MonitorError(f"unknown model {model}")
        dumped = [r.model_dump() for r in rows]
        window = Window.from_rows(dumped)
        profile = build_profile(window, record.task_type, record.positive_label)
        self.store.save_reference(model, {"profile": profile.model_dump(mode="json"), "rows": dumped})
        self._references.pop(model, None)
        return profile

    def reference(self, model: str) -> ReferenceBundle | None:
        payload = self.store.load_reference(model)
        if payload is None:
            return None
        version = str(payload["profile"]["created_at"])
        cached = self._references.get(model)
        if cached is not None and cached.version == version:
            return cached
        bundle = ReferenceBundle(
            version=version,
            profile=ReferenceProfile.model_validate(payload["profile"]),
            window=Window.from_rows(payload["rows"]),
        )
        self._references[model] = bundle
        return bundle

    def run(
        self,
        model: str,
        window_end: datetime | None = None,
        window_seconds: int | None = None,
        persist: bool = True,
    ) -> list[MetricResult]:
        record = self.store.get_model(model)
        if record is None:
            raise MonitorError(f"unknown model {model}")
        bundle = self.reference(model)
        if bundle is None:
            raise MonitorError(f"model {model} has no reference data")
        config = record.config
        end = as_utc(window_end) if window_end else utcnow()
        start = end - timedelta(seconds=window_seconds or config.window_seconds)
        current = Window.from_rows(self.store.read_events(model, start, end))
        labeled = Window.from_rows(self.store.read_labeled_events(model, start, end, as_of=end))
        results = evaluate_window(
            model,
            bundle.profile,
            bundle.window,
            current,
            labeled,
            config,
            start,
            end,
            task_type=record.task_type,
            positive_label=record.positive_label,
        )
        if persist and results:
            self.store.add_metrics(results)
        return results

    def backfill(self, model: str, start: datetime, end: datetime, step_seconds: int) -> int:
        record = self.store.get_model(model)
        if record is None:
            raise MonitorError(f"unknown model {model}")
        step = timedelta(seconds=step_seconds)
        cursor = as_utc(start) + timedelta(seconds=record.config.window_seconds)
        windows = 0
        while cursor <= as_utc(end):
            if self.run(model, window_end=cursor):
                windows += 1
            cursor += step
        return windows


def reference_rows(raw: Sequence[dict[str, Any]]) -> list[ReferenceRow]:
    return [ReferenceRow.model_validate(r) for r in raw]
