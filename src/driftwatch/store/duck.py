from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from driftwatch.schemas import LabelEvent, MonitorConfig, PredictionEvent
from driftwatch.store.base import Store
from driftwatch.timeutil import from_naive_utc, to_naive_utc, utcnow
from driftwatch.types import Alert, MetricResult, ModelRecord

EVENT_SCHEMA = pa.schema(
    [
        ("event_id", pa.string()),
        ("model_version", pa.string()),
        ("ts", pa.timestamp("us")),
        ("features", pa.string()),
        ("prediction", pa.string()),
        ("score", pa.float64()),
        ("embedding", pa.list_(pa.float64())),
        ("latency_ms", pa.float64()),
    ]
)

LABEL_SCHEMA = pa.schema(
    [
        ("event_id", pa.string()),
        ("label", pa.string()),
        ("labeled_at", pa.timestamp("us")),
    ]
)

EVENT_COLUMNS = "event_id, ts, model_version, features, prediction, score, embedding, latency_ms"

DDL = [
    "create table if not exists models (name varchar primary key, task_type varchar, positive_label varchar, config varchar, created_at timestamp)",
    "create table if not exists reference_sets (model varchar primary key, payload varchar, created_at timestamp)",
    "create sequence if not exists metric_seq",
    "create table if not exists metric_results (id bigint default nextval('metric_seq'), model varchar, window_start timestamp, window_end timestamp, kind varchar, feature varchar, metric varchar, value double, p_value double, p_adjusted double, severity double, level varchar, flagged boolean, n_ref integer, n_cur integer)",
    "create sequence if not exists alert_seq",
    "create table if not exists alerts (id bigint default nextval('alert_seq'), model varchar, rule varchar, level varchar, message varchar, window_end timestamp, created_at timestamp, delivered boolean)",
]


class DuckStore(Store):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = duckdb.connect(str(self.root / "meta.duckdb"))
        for statement in DDL:
            self._conn.execute(statement)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _run(self, sql: str, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        with self._lock:
            return self._conn.execute(sql, list(params)).fetchall()

    def _files(self, kind: str, model: str) -> list[str]:
        directory = self.root / kind / model
        return sorted(p.as_posix() for p in directory.glob("*.parquet"))

    def _write_part(self, kind: str, model: str, table: pa.Table) -> None:
        directory = self.root / kind / model
        directory.mkdir(parents=True, exist_ok=True)
        name = f"part-{time.time_ns()}-{uuid.uuid4().hex[:8]}.parquet"
        staging = directory / (name + ".tmp")
        pq.write_table(table, staging)
        staging.replace(directory / name)

    def upsert_model(self, record: ModelRecord) -> ModelRecord:
        existing = self.get_model(record.name)
        created = existing.created_at if existing else from_naive_utc(to_naive_utc(utcnow()))
        self._run("delete from models where name = ?", [record.name])
        self._run(
            "insert into models values (?, ?, ?, ?, ?)",
            [
                record.name,
                record.task_type,
                json.dumps(record.positive_label),
                record.config.model_dump_json(),
                to_naive_utc(created),
            ],
        )
        return record.model_copy(update={"created_at": created})

    def get_model(self, name: str) -> ModelRecord | None:
        rows = self._run("select name, task_type, positive_label, config, created_at from models where name = ?", [name])
        return _model_record(rows[0]) if rows else None

    def list_models(self) -> list[ModelRecord]:
        rows = self._run("select name, task_type, positive_label, config, created_at from models order by name")
        return [_model_record(r) for r in rows]

    def add_events(self, events: Sequence[PredictionEvent]) -> int:
        if not events:
            return 0
        by_model: dict[str, list[PredictionEvent]] = {}
        for event in events:
            by_model.setdefault(event.model, []).append(event)
        for model, group in by_model.items():
            table = pa.table(
                {
                    "event_id": [e.event_id for e in group],
                    "model_version": [e.model_version for e in group],
                    "ts": [to_naive_utc(e.timestamp) for e in group],
                    "features": [json.dumps(e.features) for e in group],
                    "prediction": [json.dumps(e.prediction) for e in group],
                    "score": [e.score for e in group],
                    "embedding": [e.embedding for e in group],
                    "latency_ms": [e.latency_ms for e in group],
                },
                schema=EVENT_SCHEMA,
            )
            self._write_part("events", model, table)
        return len(events)

    def add_labels(self, labels: Sequence[LabelEvent]) -> int:
        if not labels:
            return 0
        by_model: dict[str, list[LabelEvent]] = {}
        for label in labels:
            by_model.setdefault(label.model, []).append(label)
        for model, group in by_model.items():
            table = pa.table(
                {
                    "event_id": [l.event_id for l in group],
                    "label": [json.dumps(l.label) for l in group],
                    "labeled_at": [to_naive_utc(l.timestamp) for l in group],
                },
                schema=LABEL_SCHEMA,
            )
            self._write_part("labels", model, table)
        return len(labels)

    def read_events(
        self, model: str, start: datetime, end: datetime, version: str | None = None
    ) -> list[dict[str, Any]]:
        files = self._files("events", model)
        if not files:
            return []
        sql = f"select {EVENT_COLUMNS} from read_parquet(?) where ts >= ? and ts < ?"
        params: list[Any] = [files, to_naive_utc(start), to_naive_utc(end)]
        if version is not None:
            sql += " and model_version = ?"
            params.append(version)
        sql += " qualify row_number() over (partition by event_id order by ts) = 1 order by ts"
        return [_event_dict(r) for r in self._run(sql, params)]

    def read_labeled_events(
        self, model: str, start: datetime, end: datetime, as_of: datetime | None = None
    ) -> list[dict[str, Any]]:
        event_files = self._files("events", model)
        label_files = self._files("labels", model)
        if not event_files or not label_files:
            return []
        sql = (
            f"with e as (select {EVENT_COLUMNS} from read_parquet(?) where ts >= ? and ts < ? "
            "qualify row_number() over (partition by event_id order by ts) = 1), "
            "l as (select event_id, label, labeled_at from read_parquet(?) "
            "qualify row_number() over (partition by event_id order by labeled_at) = 1) "
            "select e.event_id, e.ts, e.model_version, e.features, e.prediction, e.score, e.embedding, "
            "e.latency_ms, l.label, l.labeled_at from e join l on l.event_id = e.event_id"
        )
        params: list[Any] = [event_files, to_naive_utc(start), to_naive_utc(end), label_files]
        if as_of is not None:
            sql += " where l.labeled_at <= ?"
            params.append(to_naive_utc(as_of))
        sql += " order by e.ts"
        out = []
        for row in self._run(sql, params):
            event = _event_dict(row[:8])
            event["label"] = json.loads(row[8])
            event["labeled_at"] = from_naive_utc(row[9])
            out.append(event)
        return out

    def event_count(self, model: str) -> int:
        files = self._files("events", model)
        if not files:
            return 0
        return int(self._run("select count(distinct event_id) from read_parquet(?)", [files])[0][0])

    def time_bounds(self, model: str) -> tuple[datetime, datetime] | None:
        files = self._files("events", model)
        if not files:
            return None
        low, high = self._run("select min(ts), max(ts) from read_parquet(?)", [files])[0]
        if low is None:
            return None
        return from_naive_utc(low), from_naive_utc(high)

    def save_reference(self, model: str, payload: dict[str, Any]) -> None:
        self._run("delete from reference_sets where model = ?", [model])
        self._run(
            "insert into reference_sets values (?, ?, ?)",
            [model, json.dumps(payload), to_naive_utc(utcnow())],
        )

    def load_reference(self, model: str) -> dict[str, Any] | None:
        rows = self._run("select payload from reference_sets where model = ?", [model])
        return json.loads(rows[0][0]) if rows else None

    def add_metrics(self, results: Sequence[MetricResult]) -> None:
        if not results:
            return
        rows = [
            [
                r.model,
                to_naive_utc(r.window_start),
                to_naive_utc(r.window_end),
                r.kind,
                r.feature,
                r.metric,
                r.value,
                r.p_value,
                r.p_adjusted,
                r.severity,
                r.level,
                r.flagged,
                r.n_ref,
                r.n_cur,
            ]
            for r in results
        ]
        with self._lock:
            self._conn.executemany(
                "insert into metric_results (model, window_start, window_end, kind, feature, metric, value, "
                "p_value, p_adjusted, severity, level, flagged, n_ref, n_cur) "
                "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def read_metrics(
        self,
        model: str,
        start: datetime | None = None,
        end: datetime | None = None,
        kind: str | None = None,
        feature: str | None = None,
        metric: str | None = None,
    ) -> list[MetricResult]:
        clauses = ["model = ?"]
        params: list[Any] = [model]
        if start is not None:
            clauses.append("window_end >= ?")
            params.append(to_naive_utc(start))
        if end is not None:
            clauses.append("window_end <= ?")
            params.append(to_naive_utc(end))
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if feature is not None:
            clauses.append("feature = ?")
            params.append(feature)
        if metric is not None:
            clauses.append("metric = ?")
            params.append(metric)
        sql = f"select {_METRIC_COLUMNS} from metric_results where {' and '.join(clauses)} order by window_end, feature, metric"
        return [_metric_result(r) for r in self._run(sql, params)]

    def latest_metrics(self, model: str) -> list[MetricResult]:
        sql = (
            f"select {_METRIC_COLUMNS} from metric_results where model = ? "
            "and window_end = (select max(window_end) from metric_results where model = ?)"
        )
        return [_metric_result(r) for r in self._run(sql, [model, model])]

    def add_alert(self, alert: Alert) -> Alert:
        rows = self._run(
            "insert into alerts (model, rule, level, message, window_end, created_at, delivered) "
            "values (?, ?, ?, ?, ?, ?, ?) returning id",
            [
                alert.model,
                alert.rule,
                alert.level,
                alert.message,
                to_naive_utc(alert.window_end),
                to_naive_utc(alert.created_at),
                alert.delivered,
            ],
        )
        return alert.model_copy(update={"id": int(rows[0][0])})

    def mark_alert_delivered(self, alert_id: int) -> None:
        self._run("update alerts set delivered = true where id = ?", [alert_id])

    def list_alerts(self, model: str | None = None, since: datetime | None = None, limit: int = 100) -> list[Alert]:
        clauses = ["1 = 1"]
        params: list[Any] = []
        if model is not None:
            clauses.append("model = ?")
            params.append(model)
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(to_naive_utc(since))
        sql = (
            f"select {_ALERT_COLUMNS} from alerts where {' and '.join(clauses)} "
            "order by created_at desc, id desc limit ?"
        )
        return [_alert(r) for r in self._run(sql, [*params, limit])]

    def last_alert(self, model: str, rule: str) -> Alert | None:
        rows = self._run(
            f"select {_ALERT_COLUMNS} from alerts where model = ? and rule = ? order by created_at desc, id desc limit 1",
            [model, rule],
        )
        return _alert(rows[0]) if rows else None


_METRIC_COLUMNS = (
    "model, window_start, window_end, kind, feature, metric, value, p_value, p_adjusted, severity, level, flagged, n_ref, n_cur"
)
_ALERT_COLUMNS = "id, model, rule, level, message, window_end, created_at, delivered"


def _model_record(row: tuple[Any, ...]) -> ModelRecord:
    return ModelRecord(
        name=row[0],
        task_type=row[1],
        positive_label=json.loads(row[2]),
        config=MonitorConfig.model_validate_json(row[3]),
        created_at=from_naive_utc(row[4]),
    )


def _event_dict(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "event_id": row[0],
        "timestamp": from_naive_utc(row[1]),
        "model_version": row[2],
        "features": json.loads(row[3]),
        "prediction": json.loads(row[4]),
        "score": row[5],
        "embedding": list(row[6]) if row[6] is not None else None,
        "latency_ms": row[7],
    }


def _metric_result(row: tuple[Any, ...]) -> MetricResult:
    return MetricResult(
        model=row[0],
        window_start=from_naive_utc(row[1]),
        window_end=from_naive_utc(row[2]),
        kind=row[3],
        feature=row[4],
        metric=row[5],
        value=row[6],
        p_value=row[7],
        p_adjusted=row[8],
        severity=row[9],
        level=row[10],
        flagged=row[11],
        n_ref=row[12],
        n_cur=row[13],
    )


def _alert(row: tuple[Any, ...]) -> Alert:
    return Alert(
        id=int(row[0]),
        model=row[1],
        rule=row[2],
        level=row[3],
        message=row[4],
        window_end=from_naive_utc(row[5]),
        created_at=from_naive_utc(row[6]),
        delivered=row[7],
    )
