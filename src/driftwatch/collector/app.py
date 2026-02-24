from __future__ import annotations

import secrets
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from driftwatch import __version__
from driftwatch.collector import views
from driftwatch.config import Settings
from driftwatch.engine.distribution import compare_categorical, compare_numeric
from driftwatch.engine.window import Window
from driftwatch.exporter import build_registry
from driftwatch.monitor import Monitor, MonitorError
from driftwatch.schemas import EventBatch, IngestResult, LabelBatch, ModelRegistration, ReferenceUpload
from driftwatch.store.base import Store
from driftwatch.timeutil import as_utc, utcnow
from driftwatch.types import ModelRecord


def create_app(store: Store, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    monitor = Monitor(store)
    registry, ingest_metrics = build_registry(store)
    known_models: set[str] = set()

    def guard(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key and not secrets.compare_digest(x_api_key or "", settings.api_key):
            raise HTTPException(status_code=401, detail="invalid api key")

    def ensure_models(names: set[str]) -> None:
        for name in names - known_models:
            if store.get_model(name) is None:
                store.upsert_model(ModelRecord(name=name))
            known_models.add(name)

    def require_model(name: str) -> ModelRecord:
        record = store.get_model(name)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown model {name}")
        return record

    app = FastAPI(title="driftwatch collector", version=__version__)
    app.state.store = store
    app.state.monitor = monitor
    protected = [Depends(guard)]

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/events", status_code=202, response_model=IngestResult, dependencies=protected)
    def ingest_events(batch: EventBatch) -> IngestResult:
        started = time.perf_counter()
        ensure_models({e.model for e in batch.events})
        accepted = store.add_events(batch.events)
        ingest_metrics.batch_seconds.labels("events").observe(time.perf_counter() - started)
        for model in {e.model for e in batch.events}:
            ingest_metrics.events.labels(model).inc(sum(1 for e in batch.events if e.model == model))
        return IngestResult(accepted=accepted)

    @app.post("/v1/labels", status_code=202, response_model=IngestResult, dependencies=protected)
    def ingest_labels(batch: LabelBatch) -> IngestResult:
        started = time.perf_counter()
        accepted = store.add_labels(batch.labels)
        ingest_metrics.batch_seconds.labels("labels").observe(time.perf_counter() - started)
        for model in {l.model for l in batch.labels}:
            ingest_metrics.labels.labels(model).inc(sum(1 for l in batch.labels if l.model == model))
        return IngestResult(accepted=accepted)

    @app.post("/v1/models", status_code=201, response_model=ModelRecord, dependencies=protected)
    def register_model(registration: ModelRegistration) -> ModelRecord:
        record = store.upsert_model(ModelRecord(**registration.model_dump()))
        known_models.add(record.name)
        return record

    @app.get("/v1/models", dependencies=protected)
    def list_models() -> list[dict[str, Any]]:
        out = []
        for record in store.list_models():
            bounds = store.time_bounds(record.name)
            out.append(
                {
                    "name": record.name,
                    "task_type": record.task_type,
                    "events": store.event_count(record.name),
                    "first_event": bounds[0].isoformat() if bounds else None,
                    "last_event": bounds[1].isoformat() if bounds else None,
                    "has_reference": store.load_reference(record.name) is not None,
                    "window_seconds": record.config.window_seconds,
                    "status": views.model_status(store.latest_metrics(record.name)),
                }
            )
        return out

    @app.get("/v1/models/{name}", response_model=ModelRecord, dependencies=protected)
    def get_model(name: str) -> ModelRecord:
        return require_model(name)

    @app.put("/v1/models/{name}/reference", dependencies=protected)
    def upload_reference(name: str, upload: ReferenceUpload) -> dict[str, Any]:
        require_model(name)
        profile = monitor.set_reference(name, upload.rows)
        return {
            "n_rows": profile.n_rows,
            "features": sorted(profile.features),
            "baseline": profile.baseline,
            "embedding_dim": profile.embedding_dim,
        }

    @app.post("/v1/models/{name}/monitor", dependencies=protected)
    def run_monitor(name: str, window_end: datetime | None = None) -> dict[str, Any]:
        require_model(name)
        try:
            results = monitor.run(name, window_end=window_end)
        except MonitorError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"results": len(results), "flagged": sum(1 for r in results if r.flagged)}

    @app.post("/v1/models/{name}/backfill", dependencies=protected)
    def backfill(name: str, start: datetime, end: datetime, step_seconds: int = Query(gt=0)) -> dict[str, int]:
        require_model(name)
        try:
            return {"windows": monitor.backfill(name, start, end, step_seconds)}
        except MonitorError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/models/{name}/metrics", dependencies=protected)
    def read_metrics(
        name: str,
        start: datetime | None = None,
        end: datetime | None = None,
        kind: str | None = None,
        feature: str | None = None,
        metric: str | None = None,
    ) -> list[dict[str, Any]]:
        require_model(name)
        results = store.read_metrics(name, start, end, kind, feature, metric)
        return [r.model_dump(mode="json") for r in results]

    @app.get("/v1/models/{name}/latest", dependencies=protected)
    def latest(name: str) -> list[dict[str, Any]]:
        require_model(name)
        return [r.model_dump(mode="json") for r in store.latest_metrics(name)]

    @app.get("/v1/models/{name}/heatmap", dependencies=protected)
    def heatmap(name: str, start: datetime | None = None, end: datetime | None = None) -> dict[str, Any]:
        require_model(name)
        return views.heatmap(store.read_metrics(name, start, end, kind="drift"))

    @app.get("/v1/models/{name}/performance", dependencies=protected)
    def performance(name: str, start: datetime | None = None, end: datetime | None = None) -> dict[str, Any]:
        require_model(name)
        return views.performance_series(store.read_metrics(name, start, end, kind="performance"))

    @app.get("/v1/models/{name}/distribution", dependencies=protected)
    def distribution(name: str, feature: str, window_end: datetime | None = None) -> dict[str, Any]:
        record = require_model(name)
        bundle = monitor.reference(name)
        if bundle is None:
            raise HTTPException(status_code=409, detail="model has no reference data")
        if feature not in bundle.window.frame.columns:
            raise HTTPException(status_code=404, detail=f"unknown feature {feature}")
        end = as_utc(window_end) if window_end else _default_window_end(store, name)
        start = end - timedelta(seconds=record.config.window_seconds)
        current = Window.from_rows(store.read_events(name, start, end))
        reference_column = bundle.window.frame[feature]
        current_column = current.frame[feature] if feature in current.frame.columns else pd.Series(dtype=float)
        kind = bundle.profile.monitored[feature].kind
        compare = compare_numeric if kind == "numeric" else compare_categorical
        body = compare(reference_column, current_column)
        body.update({"feature": feature, "window_start": start.isoformat(), "window_end": end.isoformat()})
        return body

    @app.get("/v1/alerts", dependencies=protected)
    def list_alerts(model: str | None = None, limit: int = Query(default=100, le=1000)) -> list[dict[str, Any]]:
        return [a.model_dump(mode="json") for a in store.list_alerts(model, limit=limit)]

    if settings.ui_dir and Path(settings.ui_dir).is_dir():
        app.mount("/", StaticFiles(directory=str(settings.ui_dir), html=True), name="ui")
    return app


def _default_window_end(store: Store, model: str) -> datetime:
    latest = store.latest_metrics(model)
    if latest:
        return max(r.window_end for r in latest)
    bounds = store.time_bounds(model)
    return bounds[1] + timedelta(seconds=1) if bounds else utcnow()
