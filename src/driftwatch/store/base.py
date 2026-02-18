from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Sequence

from driftwatch.schemas import LabelEvent, PredictionEvent
from driftwatch.types import Alert, MetricResult, ModelRecord


class Store(ABC):
    @abstractmethod
    def upsert_model(self, record: ModelRecord) -> ModelRecord: ...

    @abstractmethod
    def get_model(self, name: str) -> ModelRecord | None: ...

    @abstractmethod
    def list_models(self) -> list[ModelRecord]: ...

    @abstractmethod
    def add_events(self, events: Sequence[PredictionEvent]) -> int: ...

    @abstractmethod
    def add_labels(self, labels: Sequence[LabelEvent]) -> int: ...

    @abstractmethod
    def read_events(
        self, model: str, start: datetime, end: datetime, version: str | None = None
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    def read_labeled_events(
        self, model: str, start: datetime, end: datetime, as_of: datetime | None = None
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    def event_count(self, model: str) -> int: ...

    @abstractmethod
    def time_bounds(self, model: str) -> tuple[datetime, datetime] | None: ...

    @abstractmethod
    def save_reference(self, model: str, payload: dict[str, Any]) -> None: ...

    @abstractmethod
    def load_reference(self, model: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def add_metrics(self, results: Sequence[MetricResult]) -> None: ...

    @abstractmethod
    def read_metrics(
        self,
        model: str,
        start: datetime | None = None,
        end: datetime | None = None,
        kind: str | None = None,
        feature: str | None = None,
        metric: str | None = None,
    ) -> list[MetricResult]: ...

    @abstractmethod
    def latest_metrics(self, model: str) -> list[MetricResult]: ...

    @abstractmethod
    def add_alert(self, alert: Alert) -> Alert: ...

    @abstractmethod
    def mark_alert_delivered(self, alert_id: int) -> None: ...

    @abstractmethod
    def list_alerts(self, model: str | None = None, since: datetime | None = None, limit: int = 100) -> list[Alert]: ...

    @abstractmethod
    def last_alert(self, model: str, rule: str) -> Alert | None: ...

    def close(self) -> None:
        return None
