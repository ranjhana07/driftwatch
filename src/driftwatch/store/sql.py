from __future__ import annotations

from datetime import datetime
from typing import Any, Iterator, Sequence

from sqlalchemy import create_engine, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from driftwatch.schemas import LabelEvent, MonitorConfig, PredictionEvent
from driftwatch.store.base import Store
from driftwatch.store.tables import (
    AlertRow,
    Base,
    EventRow,
    LabelRow,
    MetricRow,
    ModelRow,
    ReferenceRow,
)
from driftwatch.timeutil import from_naive_utc, to_naive_utc, utcnow
from driftwatch.types import Alert, MetricResult, ModelRecord

INSERT_CHUNK = 500


def _chunks(rows: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


class SqlStore(Store):
    def __init__(self, url: str, create_schema: bool = False) -> None:
        options: dict[str, Any] = {}
        if url.startswith("sqlite"):
            options["connect_args"] = {"check_same_thread": False}
            if url in ("sqlite://", "sqlite:///:memory:"):
                options["poolclass"] = StaticPool
        else:
            options["pool_pre_ping"] = True
        self.engine = create_engine(url, **options)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)
        if create_schema:
            Base.metadata.create_all(self.engine)

    def close(self) -> None:
        self.engine.dispose()

    def _insert_ignore(self, session: Session, table: Any, rows: list[dict[str, Any]]) -> int:
        insert = pg_insert if self.engine.dialect.name == "postgresql" else sqlite_insert
        inserted = 0
        for chunk in _chunks(rows, INSERT_CHUNK):
            result = session.execute(insert(table).values(chunk).on_conflict_do_nothing())
            inserted += result.rowcount
        return inserted

    def upsert_model(self, record: ModelRecord) -> ModelRecord:
        with self._sessions.begin() as session:
            existing = session.get(ModelRow, record.name)
            created = existing.created_at if existing else to_naive_utc(utcnow())
            session.merge(
                ModelRow(
                    name=record.name,
                    task_type=record.task_type,
                    positive_label=record.positive_label,
                    config=record.config.model_dump(),
                    created_at=created,
                )
            )
        return record.model_copy(update={"created_at": from_naive_utc(created)})

    def get_model(self, name: str) -> ModelRecord | None:
        with self._sessions() as session:
            row = session.get(ModelRow, name)
            return _model_record(row) if row else None

    def list_models(self) -> list[ModelRecord]:
        with self._sessions() as session:
            rows = session.scalars(select(ModelRow).order_by(ModelRow.name)).all()
            return [_model_record(r) for r in rows]

    def add_events(self, events: Sequence[PredictionEvent]) -> int:
        rows = [
            {
                "event_id": e.event_id,
                "model": e.model,
                "model_version": e.model_version,
                "ts": to_naive_utc(e.timestamp),
                "features": e.features,
                "prediction": e.prediction,
                "score": e.score,
                "embedding": e.embedding,
                "latency_ms": e.latency_ms,
            }
            for e in events
        ]
        with self._sessions.begin() as session:
            return self._insert_ignore(session, EventRow.__table__, rows)

    def add_labels(self, labels: Sequence[LabelEvent]) -> int:
        rows = [
            {
                "event_id": l.event_id,
                "model": l.model,
                "label": l.label,
                "labeled_at": to_naive_utc(l.timestamp),
            }
            for l in labels
        ]
        with self._sessions.begin() as session:
            return self._insert_ignore(session, LabelRow.__table__, rows)

    def read_events(
        self, model: str, start: datetime, end: datetime, version: str | None = None
    ) -> list[dict[str, Any]]:
        query = select(EventRow).where(
            EventRow.model == model, EventRow.ts >= to_naive_utc(start), EventRow.ts < to_naive_utc(end)
        )
        if version is not None:
            query = query.where(EventRow.model_version == version)
        with self._sessions() as session:
            return [_event_dict(r) for r in session.scalars(query.order_by(EventRow.ts))]

    def read_labeled_events(
        self, model: str, start: datetime, end: datetime, as_of: datetime | None = None
    ) -> list[dict[str, Any]]:
        query = (
            select(EventRow, LabelRow.label, LabelRow.labeled_at)
            .join(LabelRow, LabelRow.event_id == EventRow.event_id)
            .where(EventRow.model == model, EventRow.ts >= to_naive_utc(start), EventRow.ts < to_naive_utc(end))
        )
        if as_of is not None:
            query = query.where(LabelRow.labeled_at <= to_naive_utc(as_of))
        with self._sessions() as session:
            out = []
            for event, label, labeled_at in session.execute(query.order_by(EventRow.ts)):
                row = _event_dict(event)
                row["label"] = label
                row["labeled_at"] = from_naive_utc(labeled_at)
                out.append(row)
            return out

    def event_count(self, model: str) -> int:
        with self._sessions() as session:
            return int(session.scalar(select(func.count()).select_from(EventRow).where(EventRow.model == model)) or 0)

    def time_bounds(self, model: str) -> tuple[datetime, datetime] | None:
        with self._sessions() as session:
            low, high = session.execute(
                select(func.min(EventRow.ts), func.max(EventRow.ts)).where(EventRow.model == model)
            ).one()
        if low is None:
            return None
        return from_naive_utc(low), from_naive_utc(high)

    def save_reference(self, model: str, payload: dict[str, Any]) -> None:
        with self._sessions.begin() as session:
            session.merge(ReferenceRow(model=model, payload=payload, created_at=to_naive_utc(utcnow())))

    def load_reference(self, model: str) -> dict[str, Any] | None:
        with self._sessions() as session:
            row = session.get(ReferenceRow, model)
            return row.payload if row else None

    def add_metrics(self, results: Sequence[MetricResult]) -> None:
        rows = [
            {
                "model": r.model,
                "window_start": to_naive_utc(r.window_start),
                "window_end": to_naive_utc(r.window_end),
                "kind": r.kind,
                "feature": r.feature,
                "metric": r.metric,
                "value": r.value,
                "p_value": r.p_value,
                "p_adjusted": r.p_adjusted,
                "severity": r.severity,
                "level": r.level,
                "flagged": r.flagged,
                "n_ref": r.n_ref,
                "n_cur": r.n_cur,
            }
            for r in results
        ]
        if not rows:
            return
        with self._sessions.begin() as session:
            for chunk in _chunks(rows, INSERT_CHUNK):
                session.execute(MetricRow.__table__.insert(), chunk)

    def read_metrics(
        self,
        model: str,
        start: datetime | None = None,
        end: datetime | None = None,
        kind: str | None = None,
        feature: str | None = None,
        metric: str | None = None,
    ) -> list[MetricResult]:
        query = select(MetricRow).where(MetricRow.model == model)
        if start is not None:
            query = query.where(MetricRow.window_end >= to_naive_utc(start))
        if end is not None:
            query = query.where(MetricRow.window_end <= to_naive_utc(end))
        if kind is not None:
            query = query.where(MetricRow.kind == kind)
        if feature is not None:
            query = query.where(MetricRow.feature == feature)
        if metric is not None:
            query = query.where(MetricRow.metric == metric)
        with self._sessions() as session:
            rows = session.scalars(query.order_by(MetricRow.window_end, MetricRow.feature, MetricRow.metric))
            return [_metric_result(r) for r in rows]

    def latest_metrics(self, model: str) -> list[MetricResult]:
        latest = select(func.max(MetricRow.window_end)).where(MetricRow.model == model).scalar_subquery()
        with self._sessions() as session:
            rows = session.scalars(select(MetricRow).where(MetricRow.model == model, MetricRow.window_end == latest))
            return [_metric_result(r) for r in rows]

    def add_alert(self, alert: Alert) -> Alert:
        with self._sessions.begin() as session:
            row = AlertRow(
                model=alert.model,
                rule=alert.rule,
                level=alert.level,
                message=alert.message,
                window_end=to_naive_utc(alert.window_end),
                created_at=to_naive_utc(alert.created_at),
                delivered=alert.delivered,
            )
            session.add(row)
            session.flush()
            return alert.model_copy(update={"id": row.id})

    def mark_alert_delivered(self, alert_id: int) -> None:
        with self._sessions.begin() as session:
            session.execute(update(AlertRow).where(AlertRow.id == alert_id).values(delivered=True))

    def list_alerts(self, model: str | None = None, since: datetime | None = None, limit: int = 100) -> list[Alert]:
        query = select(AlertRow)
        if model is not None:
            query = query.where(AlertRow.model == model)
        if since is not None:
            query = query.where(AlertRow.created_at >= to_naive_utc(since))
        with self._sessions() as session:
            rows = session.scalars(query.order_by(AlertRow.created_at.desc(), AlertRow.id.desc()).limit(limit))
            return [_alert(r) for r in rows]

    def last_alert(self, model: str, rule: str) -> Alert | None:
        query = (
            select(AlertRow)
            .where(AlertRow.model == model, AlertRow.rule == rule)
            .order_by(AlertRow.created_at.desc(), AlertRow.id.desc())
            .limit(1)
        )
        with self._sessions() as session:
            row = session.scalars(query).first()
            return _alert(row) if row else None


def _model_record(row: ModelRow) -> ModelRecord:
    return ModelRecord(
        name=row.name,
        task_type=row.task_type,
        positive_label=row.positive_label,
        config=MonitorConfig.model_validate(row.config),
        created_at=from_naive_utc(row.created_at),
    )


def _event_dict(row: EventRow) -> dict[str, Any]:
    return {
        "event_id": row.event_id,
        "timestamp": from_naive_utc(row.ts),
        "model_version": row.model_version,
        "features": row.features,
        "prediction": row.prediction,
        "score": row.score,
        "embedding": row.embedding,
        "latency_ms": row.latency_ms,
    }


def _metric_result(row: MetricRow) -> MetricResult:
    return MetricResult(
        model=row.model,
        window_start=from_naive_utc(row.window_start),
        window_end=from_naive_utc(row.window_end),
        kind=row.kind,
        feature=row.feature,
        metric=row.metric,
        value=row.value,
        p_value=row.p_value,
        p_adjusted=row.p_adjusted,
        severity=row.severity,
        level=row.level,
        flagged=row.flagged,
        n_ref=row.n_ref,
        n_cur=row.n_cur,
    )


def _alert(row: AlertRow) -> Alert:
    return Alert(
        id=row.id,
        model=row.model,
        rule=row.rule,
        level=row.level,
        message=row.message,
        window_end=from_naive_utc(row.window_end),
        created_at=from_naive_utc(row.created_at),
        delivered=row.delivered,
    )
