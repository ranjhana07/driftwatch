from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from driftwatch import __version__
from driftwatch.config import Settings, make_store
from driftwatch.schemas import MonitorConfig
from driftwatch.timeutil import as_utc, utcnow
from driftwatch.types import ModelRecord

app = typer.Typer(help="driftwatch: model monitoring for production ML", no_args_is_help=True)
db_app = typer.Typer(help="database migrations", no_args_is_help=True)
models_app = typer.Typer(help="registered models", no_args_is_help=True)
monitor_app = typer.Typer(help="run monitors by hand", no_args_is_help=True)
alerts_app = typer.Typer(help="alert history", no_args_is_help=True)
demo_app = typer.Typer(help="demo traffic with injected drift", no_args_is_help=True)
app.add_typer(db_app, name="db")
app.add_typer(models_app, name="models")
app.add_typer(monitor_app, name="monitor")
app.add_typer(alerts_app, name="alerts")
app.add_typer(demo_app, name="demo")


def _settings() -> Settings:
    return Settings()


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="debug logging")) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@app.command()
def version() -> None:
    typer.echo(__version__)


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, help="bind address"),
    port: Optional[int] = typer.Option(None, help="bind port"),
    ui_dir: Optional[Path] = typer.Option(None, help="serve the static ui from this directory"),
) -> None:
    import uvicorn

    from driftwatch.collector import create_app

    settings = _settings()
    if ui_dir is not None:
        settings.ui_dir = ui_dir
    store = make_store(settings)
    uvicorn.run(create_app(store, settings), host=host or settings.host, port=port or settings.port, log_level="info")


@app.command()
def scheduler() -> None:
    from prometheus_client import start_http_server

    from driftwatch.alerting import AlertManager, build_notifiers, default_rules, load_rules
    from driftwatch.monitor import Monitor
    from driftwatch.scheduler import MonitorScheduler, SchedulerMetrics

    settings = _settings()
    store = make_store(settings)
    rules = load_rules(settings.alert_rules_path) if settings.alert_rules_path else default_rules()
    alerts = AlertManager(store, rules, build_notifiers(settings))
    metrics = SchedulerMetrics()
    start_http_server(settings.scheduler_metrics_port, registry=metrics.registry)
    runner = MonitorScheduler(
        store, Monitor(store), alerts, metrics=metrics, sync_interval_seconds=settings.sync_interval_seconds
    )
    typer.echo(f"scheduler running, metrics on :{settings.scheduler_metrics_port}")
    runner.start()


@db_app.command("upgrade")
def db_upgrade(revision: str = "head") -> None:
    from driftwatch.migrate import upgrade

    settings = _settings()
    if not settings.database_url:
        typer.echo("DRIFTWATCH_DATABASE_URL is not set; local mode needs no migrations", err=True)
        raise typer.Exit(1)
    upgrade(settings.database_url, revision)
    typer.echo(f"database at {revision}")


@models_app.command("list")
def models_list() -> None:
    store = make_store(_settings())
    for record in store.list_models():
        bounds = store.time_bounds(record.name)
        span = f"{bounds[0].isoformat()} .. {bounds[1].isoformat()}" if bounds else "no events"
        typer.echo(f"{record.name}\t{record.task_type}\t{store.event_count(record.name)} events\t{span}")


@models_app.command("register")
def models_register(
    name: str,
    task_type: str = typer.Option("classification", help="classification or regression"),
    window_seconds: int = typer.Option(3600, help="monitoring window"),
    interval_seconds: int = typer.Option(300, help="how often the scheduler evaluates"),
) -> None:
    store = make_store(_settings())
    config = MonitorConfig(window_seconds=window_seconds, interval_seconds=interval_seconds)
    store.upsert_model(ModelRecord(name=name, task_type=task_type, config=config))
    typer.echo(f"registered {name}")


def _parse_time(value: str) -> datetime:
    return as_utc(datetime.fromisoformat(value))


@monitor_app.command("run")
def monitor_run(
    model: str,
    now: Optional[str] = typer.Option(None, help="window end, ISO 8601; defaults to the current time"),
) -> None:
    from driftwatch.monitor import Monitor, MonitorError

    store = make_store(_settings())
    try:
        results = Monitor(store).run(model, window_end=_parse_time(now) if now else utcnow())
    except MonitorError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
    flagged = [r for r in results if r.flagged]
    typer.echo(f"{len(results)} results, {len(flagged)} flagged")
    for r in sorted(flagged, key=lambda r: -r.severity):
        typer.echo(f"  {r.level:<8} {r.kind}/{r.metric} {r.feature or '-'} = {r.value:.4g}")


@monitor_app.command("backfill")
def monitor_backfill(
    model: str,
    start: Optional[str] = typer.Option(None, help="ISO 8601, defaults to the first event"),
    end: Optional[str] = typer.Option(None, help="ISO 8601, defaults to the last event"),
    step_seconds: Optional[int] = typer.Option(None, help="defaults to half the window"),
) -> None:
    from driftwatch.monitor import Monitor, MonitorError

    store = make_store(_settings())
    record = store.get_model(model)
    bounds = store.time_bounds(model)
    if record is None or bounds is None:
        typer.echo("unknown model or no events", err=True)
        raise typer.Exit(1)
    step = step_seconds or max(record.config.window_seconds // 2, 1)
    try:
        windows = Monitor(store).backfill(
            model, _parse_time(start) if start else bounds[0], _parse_time(end) if end else bounds[1], step
        )
    except MonitorError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
    typer.echo(f"evaluated {windows} windows")


@alerts_app.command("list")
def alerts_list(model: Optional[str] = None, limit: int = 20, as_json: bool = typer.Option(False, "--json")) -> None:
    store = make_store(_settings())
    alerts = store.list_alerts(model, limit=limit)
    if as_json:
        typer.echo(json.dumps([a.model_dump(mode="json") for a in alerts], indent=2))
        return
    for alert in alerts:
        typer.echo(f"{alert.created_at.isoformat()}  {alert.level:<8} {alert.model}/{alert.rule}  {alert.message}")


@demo_app.command("run")
def demo_run(
    endpoint: str = typer.Option("http://127.0.0.1:8000", help="collector url"),
    scenario: str = typer.Option("all", help="none, covariate_shift, prior_shift, concept_drift, null_burst or all"),
    events: int = typer.Option(9000, help="events per scenario"),
    onset: int = typer.Option(4500, help="event index where drift starts"),
    ramp: int = typer.Option(0, help="events over which drift ramps in, 0 for abrupt"),
    seed: int = 0,
    api_key: Optional[str] = typer.Option(None, envvar="DRIFTWATCH_API_KEY"),
    cache_dir: Path = typer.Option(Path(".driftwatch/demo"), help="dataset and model cache"),
) -> None:
    from driftwatch.demo.producer import DemoConfig, run_demo
    from driftwatch.demo.scenarios import SCENARIOS

    chosen = list(SCENARIOS) if scenario == "all" else [scenario]
    unknown = [s for s in chosen if s not in SCENARIOS]
    if unknown:
        typer.echo(f"unknown scenario {unknown[0]}", err=True)
        raise typer.Exit(2)
    config = DemoConfig(
        endpoint=endpoint,
        api_key=api_key,
        scenarios=chosen,
        n_events=events,
        onset=onset,
        ramp=ramp,
        seed=seed,
        cache_dir=cache_dir,
    )
    run_demo(config, log=typer.echo)
